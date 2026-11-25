//! Local GGUF inference on candle (pure Rust — keeps the no-C++ rule and
//! the door open for wasm later). Runs on the CPU, or on a GPU when the
//! build has the `cuda` (NVIDIA) or `metal` (Apple) feature and the
//! caller asks for one ([`DeviceChoice`]).
//!
//! Every supported architecture speaks ChatML — one prompt template covers
//! Qwen2.5, Qwen3 and SmolLM2. A GGUF that is not in the registry can be
//! loaded too ([`LocalGguf::load_file`]); its architecture is read from
//! the file, and it must use the ChatML template.

use std::fs::File;
use std::path::Path;

use candle_core::quantized::gguf_file;
use candle_core::{Device, IndexOp, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::{quantized_llama, quantized_qwen2, quantized_qwen3};
use tokenizers::Tokenizer;

use crate::device::DeviceChoice;
use crate::error::AiError;
use crate::provider::{GenOpts, Provider, ProviderInfo};
use crate::registry::{Arch, ModelEntry};

enum Weights {
    Qwen2(quantized_qwen2::ModelWeights),
    Qwen3(quantized_qwen3::ModelWeights),
    Llama(quantized_llama::ModelWeights),
}

impl Weights {
    fn forward(&mut self, ids: &Tensor, pos: usize) -> Result<Tensor, candle_core::Error> {
        match self {
            Self::Qwen2(w) => w.forward(ids, pos),
            Self::Qwen3(w) => w.forward(ids, pos),
            Self::Llama(w) => w.forward(ids, pos),
        }
    }

    /// Forget the previous prompt. Qwen2 and Llama start over by themselves
    /// at position 0; Qwen3's cache only grows, so it is cleared here.
    fn reset(&mut self) {
        if let Self::Qwen3(w) = self {
            w.clear_kv_cache();
        }
    }
}

/// Open the processor `choice` names. The label says what was opened, for
/// the status line and command-line results; `Auto` that finds no GPU
/// says so in the label rather than failing.
pub fn open_device(choice: DeviceChoice) -> Result<(Device, String), AiError> {
    match choice {
        DeviceChoice::Cpu => Ok((Device::Cpu, "CPU".into())),
        DeviceChoice::Cuda(n) => cuda(n),
        DeviceChoice::Metal(n) => metal(n),
        DeviceChoice::Gpu => any_gpu(),
        DeviceChoice::Auto => Ok(any_gpu().unwrap_or_else(|e| (Device::Cpu, format!("CPU ({e})")))),
    }
}

fn cuda(n: usize) -> Result<(Device, String), AiError> {
    if !cfg!(feature = "cuda") {
        return Err(AiError::Inference("this build has no NVIDIA GPU support (it was built without the `cuda` feature)".into()));
    }
    Device::new_cuda(n).map(|d| (d, format!("NVIDIA GPU {n} (CUDA)"))).map_err(|e| AiError::Inference(format!("cannot open NVIDIA GPU {n}: {e}")))
}

fn metal(n: usize) -> Result<(Device, String), AiError> {
    if !cfg!(feature = "metal") {
        return Err(AiError::Inference("this build has no Apple GPU support (it was built without the `metal` feature)".into()));
    }
    Device::new_metal(n).map(|d| (d, format!("Apple GPU {n} (Metal)"))).map_err(|e| AiError::Inference(format!("cannot open Apple GPU {n}: {e}")))
}

fn any_gpu() -> Result<(Device, String), AiError> {
    if !cfg!(feature = "cuda") && !cfg!(feature = "metal") {
        return Err(AiError::Inference("no GPU support in this build".into()));
    }
    let mut why = Vec::new();
    let backends: [fn(usize) -> Result<(Device, String), AiError>; 2] = [cuda, metal];
    for open in backends {
        match open(0) {
            Ok(d) => return Ok(d),
            Err(e) => why.push(e.to_string()),
        }
    }
    Err(AiError::Inference(format!("no usable GPU: {}", why.join("; "))))
}

/// The GPUs this build can open on this machine, as `(device, label)`
/// pairs like `("cuda:0", "NVIDIA GPU 0 (CUDA)")`. Opening each one costs
/// a moment, so this is for listings, not for every run.
pub fn gpus() -> Vec<(String, String)> {
    let mut found = Vec::new();
    if cfg!(feature = "cuda") {
        for n in 0..16 {
            match cuda(n) {
                Ok((_, label)) => found.push((format!("cuda:{n}"), label)),
                Err(_) => break,
            }
        }
    }
    if cfg!(feature = "metal") {
        for n in 0..4 {
            match metal(n) {
                Ok((_, label)) => found.push((format!("metal:{n}"), label)),
                Err(_) => break,
            }
        }
    }
    found
}

/// A downloaded GGUF loaded and ready. Construction is the expensive
/// part; `complete` is then just a forward loop.
pub struct LocalGguf {
    weights: Weights,
    tokenizer: Tokenizer,
    device: Device,
    eos: u32,
    arch: Arch,
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
        Self::open(Some(entry.arch), entry.id.to_string(), entry.name.to_string(), gguf_path, tokenizer_path, device)
    }

    /// Load a GGUF that is not in the registry. Its architecture comes from
    /// the file's `general.architecture`; only those this crate can run
    /// ([`Arch::from_gguf_name`]) load, and the model must use ChatML.
    pub fn load_file(gguf_path: &Path, tokenizer_path: &Path, device: DeviceChoice) -> Result<Self, AiError> {
        let stem = gguf_path.file_stem().map_or_else(|| "model".to_string(), |s| s.to_string_lossy().into_owned());
        Self::open(None, gguf_path.display().to_string(), stem, gguf_path, tokenizer_path, device)
    }

    fn open(arch: Option<Arch>, id: String, name: String, gguf_path: &Path, tokenizer_path: &Path, choice: DeviceChoice) -> Result<Self, AiError> {
        let (device, on) = open_device(choice)?;
        let mut file = File::open(gguf_path).map_err(AiError::Io)?;
        // Malformed GGUF metadata (e.g. `general.alignment = 0`, or a
        // zero `attention.head_count` reaching a division in the weight
        // constructors) makes candle panic — hostile input must come
        // back as an error, not a crash. The guard covers parsing *and*
        // weight construction.
        let (weights, arch) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<(Weights, Arch), AiError> {
            let content = gguf_file::Content::read(&mut file).map_err(|e| AiError::Inference(format!("bad GGUF: {e}")))?;
            let arch = match arch {
                Some(a) => a,
                None => {
                    let named = content.metadata.get("general.architecture").and_then(|v| v.to_string().ok()).cloned().unwrap_or_default();
                    Arch::from_gguf_name(&named)
                        .ok_or_else(|| AiError::Inference(format!("cannot run a {named:?} model; supported: qwen2, qwen3, llama")))?
                }
            };
            let cannot = |e: candle_core::Error| AiError::Inference(format!("cannot load {}: {e}", gguf_path.display()));
            let w = match arch {
                Arch::Qwen2 => Weights::Qwen2(quantized_qwen2::ModelWeights::from_gguf(content, &mut file, &device).map_err(cannot)?),
                Arch::Qwen3 => Weights::Qwen3(quantized_qwen3::ModelWeights::from_gguf(content, &mut file, &device).map_err(cannot)?),
                Arch::Llama => Weights::Llama(quantized_llama::ModelWeights::from_gguf(content, &mut file, &device).map_err(cannot)?),
            };
            Ok((w, arch))
        }))
        .map_err(|_| AiError::Inference("GGUF loader panicked on malformed metadata".to_owned()))??;
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| AiError::Inference(format!("bad tokenizer.json: {e}")))?;
        let eos = ["<|im_end|>", "<|endoftext|>"]
            .iter()
            .find_map(|t| tokenizer.token_to_id(t))
            .ok_or_else(|| AiError::Inference("tokenizer has no ChatML end token (<|im_end|>)".into()))?;
        Ok(Self { weights, tokenizer, device, eos, arch, id, name, on })
    }

    /// What the model runs on, e.g. `"NVIDIA GPU 0 (CUDA)"` or `"CPU"`.
    pub fn device_label(&self) -> &str {
        &self.on
    }

    /// ChatML wrap. Qwen3 thinks aloud unless its answer opens with an
    /// empty think block, so that block is given.
    fn chat(&self, prompt: &str) -> String {
        let think = if self.arch == Arch::Qwen3 { "<think>\n\n</think>\n\n" } else { "" };
        format!(
            "<|im_start|>system\nYou are a precise translation engine. Follow the instructions exactly.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{think}",
        )
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
            if next == self.eos {
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
