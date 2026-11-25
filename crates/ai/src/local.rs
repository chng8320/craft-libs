//! Local GGUF inference on candle (pure Rust — keeps the no-C++ rule and
//! the door open for wasm later). Runs on the CPU, or on a GPU when the
//! build has the `cuda` (NVIDIA) or `metal` (Apple) feature and the
//! caller asks for one ([`DeviceChoice`]).
//!
//! Any GGUF of a supported architecture loads ([`LocalGguf::load_spec`]):
//! the architecture comes from the file when not given, and the prompt
//! format ([`Template`]) from the file's own chat template when not given,
//! else the architecture's usual one.

use std::fs::File;
use std::path::Path;

use candle_core::quantized::gguf_file;
use candle_core::{Device, IndexOp, Tensor};

use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::{quantized_gemma3, quantized_llama, quantized_phi3, quantized_qwen2, quantized_qwen3};
use tokenizers::Tokenizer;

use crate::device::DeviceChoice;
use crate::error::AiError;
use crate::provider::{GenOpts, Provider, ProviderInfo};
use crate::registry::{Arch, ModelEntry};
use crate::template::Template;

enum Weights {
    Qwen2(quantized_qwen2::ModelWeights),
    Qwen3(quantized_qwen3::ModelWeights),
    Llama(quantized_llama::ModelWeights),
    Gemma3(quantized_gemma3::ModelWeights),
    Phi3(quantized_phi3::ModelWeights),
}

impl Weights {
    fn forward(&mut self, ids: &Tensor, pos: usize) -> Result<Tensor, candle_core::Error> {
        match self {
            Self::Qwen2(w) => w.forward(ids, pos),
            Self::Qwen3(w) => w.forward(ids, pos),
            Self::Llama(w) => w.forward(ids, pos),
            Self::Gemma3(w) => w.forward(ids, pos),
            Self::Phi3(w) => w.forward(ids, pos),
        }
    }

    /// Forget the previous prompt. The others start over by themselves at
    /// position 0; Qwen3's cache only grows, so it is cleared here.
    fn reset(&mut self) {
        if let Self::Qwen3(w) = self {
            w.clear_kv_cache();
        }
    }
}

pub use crate::device::{gpus, open as open_device};

/// What to load: a GGUF and its tokenizer, and whatever the caller knows
/// about it. `None` fields are read from the file.
pub struct LoadSpec<'a> {
    pub id: String,
    pub name: String,
    pub gguf: &'a Path,
    pub tokenizer: &'a Path,
    pub arch: Option<Arch>,
    pub template: Option<Template>,
    pub device: DeviceChoice,
}

/// A downloaded GGUF loaded and ready. Construction is the expensive
/// part; `complete` is then just a forward loop.
pub struct LocalGguf {
    weights: Weights,
    tokenizer: Tokenizer,
    device: Device,
    stop: Vec<u32>,
    template: Template,
    id: String,
    name: String,
    on: String,
}

const SEED: u64 = 0xC0FFEE;

impl LocalGguf {
    /// Load the GGUF + tokenizer.json of `entry` from the model cache, on
    /// the CPU. All failures are `Inference` errors — a corrupt file, an
    /// OOM, a wrong-arch GGUF, all land in the status line, never in a panic.
    pub fn load(entry: &'static ModelEntry, gguf_path: &Path, tokenizer_path: &Path) -> Result<Self, AiError> {
        Self::load_on(entry, gguf_path, tokenizer_path, DeviceChoice::Cpu)
    }

    /// [`Self::load`] on the processor `device` names.
    pub fn load_on(entry: &'static ModelEntry, gguf_path: &Path, tokenizer_path: &Path, device: DeviceChoice) -> Result<Self, AiError> {
        Self::load_spec(&LoadSpec {
            id: entry.id.into(),
            name: entry.name.into(),
            gguf: gguf_path,
            tokenizer: tokenizer_path,
            arch: Some(entry.arch),
            template: Some(entry.arch.default_template()),
            device,
        })
    }

    /// Load a GGUF that is not in the registry, reading its architecture
    /// and prompt format from the file.
    pub fn load_file(gguf_path: &Path, tokenizer_path: &Path, device: DeviceChoice) -> Result<Self, AiError> {
        let stem = gguf_path.file_stem().map_or_else(|| "model".to_string(), |s| s.to_string_lossy().into_owned());
        Self::load_spec(&LoadSpec {
            id: gguf_path.display().to_string(),
            name: stem,
            gguf: gguf_path,
            tokenizer: tokenizer_path,
            arch: None,
            template: None,
            device,
        })
    }

    /// Load any GGUF: what `spec` leaves `None` is read from the file.
    pub fn load_spec(spec: &LoadSpec<'_>) -> Result<Self, AiError> {
        let (device, on) = open_device(spec.device)?;
        let gguf_path = spec.gguf;
        let mut file = File::open(gguf_path).map_err(AiError::Io)?;
        // Malformed GGUF metadata (e.g. `general.alignment = 0`, or a
        // zero `attention.head_count` reaching a division in the weight
        // constructors) makes candle panic — hostile input must come
        // back as an error, not a crash. The guard covers parsing *and*
        // weight construction.
        let (weights, arch, own_template) =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<(Weights, Arch, Option<Template>), AiError> {
                let content = gguf_file::Content::read(&mut file).map_err(|e| AiError::Inference(format!("bad GGUF: {e}")))?;
                let meta = |k: &str| content.metadata.get(k).and_then(|v| v.to_string().ok()).cloned();
                let own_template = meta("tokenizer.chat_template").as_deref().and_then(Template::detect);
                let arch = match spec.arch {
                    Some(a) => a,
                    None => {
                        let named = meta("general.architecture").unwrap_or_default();
                        Arch::from_gguf_name(&named)
                            .ok_or_else(|| AiError::Inference(format!("cannot run a {named:?} model here; supported: {}", Arch::NAMES)))?
                    }
                };
                let cannot = |e: candle_core::Error| AiError::Inference(format!("cannot load {}: {e}", gguf_path.display()));
                let w = match arch {
                    Arch::Qwen2 => Weights::Qwen2(quantized_qwen2::ModelWeights::from_gguf(content, &mut file, &device).map_err(cannot)?),
                    Arch::Qwen3 => Weights::Qwen3(quantized_qwen3::ModelWeights::from_gguf(content, &mut file, &device).map_err(cannot)?),
                    Arch::Llama => Weights::Llama(quantized_llama::ModelWeights::from_gguf(content, &mut file, &device).map_err(cannot)?),
                    Arch::Gemma3 => Weights::Gemma3(quantized_gemma3::ModelWeights::from_gguf(content, &mut file, &device).map_err(cannot)?),
                    Arch::Phi3 => Weights::Phi3(quantized_phi3::ModelWeights::from_gguf(false, content, &mut file, &device).map_err(cannot)?),
                };
                Ok((w, arch, own_template))
            }))
            .map_err(|_| AiError::Inference("GGUF loader panicked on malformed metadata".to_owned()))??;
        let tokenizer = Tokenizer::from_file(spec.tokenizer).map_err(|e| AiError::Inference(format!("bad tokenizer.json: {e}")))?;
        let template = spec.template.clone().or(own_template).unwrap_or_else(|| arch.default_template());
        let stop: Vec<u32> = template.stop().iter().filter_map(|t| tokenizer.token_to_id(t)).collect();
        if stop.is_empty() {
            return Err(AiError::Inference(format!(
                "the tokenizer knows none of the {template} template's end tokens ({})",
                template.stop().join(", ")
            )));
        }
        Ok(Self { weights, tokenizer, device, stop, template, id: spec.id.clone(), name: spec.name.clone(), on })
    }

    /// What the model runs on, e.g. `"NVIDIA GPU 0 (CUDA)"` or `"CPU"`.
    pub fn device_label(&self) -> &str {
        &self.on
    }

    /// The prompt format in use.
    pub fn template(&self) -> &Template {
        &self.template
    }

    fn chat(&self, prompt: &str) -> String {
        self.template.render("You are a precise translation engine. Follow the instructions exactly.", prompt)
    }

    fn generate(&mut self, prompt: &str, opts: &GenOpts) -> Result<String, AiError> {
        let text = self.chat(prompt);
        let encoding = self.tokenizer.encode(text, true).map_err(|e| AiError::Inference(format!("encode failed: {e}")))?;
        let mut tokens: Vec<u32> = encoding.get_ids().to_vec();
        self.weights.reset();

        let temp = if opts.temperature <= 0.02 { None } else { Some(f64::from(opts.temperature)) };
        let mut sampler = LogitsProcessor::new(SEED, temp, None);

        let mut pos = 0usize;
        let mut generated: Vec<u32> = Vec::with_capacity(opts.max_tokens.min(4096) as usize);
        let mut hit_eos = false;
        for _ in 0..opts.max_tokens {
            let input_len = tokens.len() - pos;
            let input =
                Tensor::new(&tokens[pos..], &self.device).and_then(|t| t.reshape((1, t.dims1()?))).map_err(|e| AiError::Inference(e.to_string()))?;
            let logits = self.weights.forward(&input, pos).map_err(|e| AiError::Inference(e.to_string()))?;
            pos += input_len;
            // `forward` already returns the last sequence position's
            // logits [batch, vocab] — index the batch row only.
            let logits = logits.i(0).map_err(|e| AiError::Inference(e.to_string()))?;
            let next = sampler.sample(&logits).map_err(|e| AiError::Inference(e.to_string()))?;
            if self.stop.contains(&next) {
                hit_eos = true;
                break;
            }
            tokens.push(next);
            generated.push(next);
        }
        // No EOS inside the budget → the reply was cut mid-translation;
        // partial output would silently truncate the document.
        if !hit_eos {
            return Err(AiError::Inference(format!("generation hit max_tokens ({}) without EOS", opts.max_tokens)));
        }
        // Decode the generated ids through the tokenizer: byte-level BPE
        // vocab strings can't be concatenated (spaces/newlines/non-ASCII
        // bytes stay encoded, and UTF-8 can split across tokens).
        self.tokenizer.decode(&generated, true).map_err(|e| AiError::Inference(format!("decode failed: {e}")))
    }
}

impl Provider for LocalGguf {
    fn name(&self) -> String {
        format!("local: {} on {}", self.name, self.on)
    }

    fn complete(&mut self, prompt: &str, opts: &GenOpts) -> Result<String, AiError> {
        self.generate(prompt, opts).map(|s| s.trim().to_string())
    }

    fn test(&mut self) -> Result<ProviderInfo, AiError> {
        Ok(ProviderInfo { label: format!("loaded: {} on {}", self.name, self.on), models: vec![self.id.clone()] })
    }

    fn is_local(&self) -> bool {
        true
    }
}
