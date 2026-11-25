//! A command line over craft-ai, for servers and scripts: list and download the curated local
//! models, complete a prompt, and translate text or a list of segments, with a local model or any
//! OpenAI-compatible endpoint. Every answer is one JSON document on stdout; errors go to stderr
//! with exit status 1.
//!
//! ```sh
//! cargo run --release -p craft-ai --features local --example craft-ai-cli -- models
//! craft-ai-cli download qwen2.5-1.5b-q4km
//! craft-ai-cli translate --model qwen2.5-1.5b-q4km --to Vietnamese --text "Hello, world"
//! craft-ai-cli translate --endpoint http://localhost:11434/v1 --remote-model qwen2.5 --to fr --segments-file in.json
//! craft-ai-cli complete --model qwen2.5-1.5b-q4km --prompt-file prompt.txt --max-tokens 400
//! craft-ai-cli translate --model qwen3-8b-q4km --device cuda --to Vietnamese --file in.txt
//! craft-ai-cli translate --model /models/my-model.gguf --device cpu --to fr --text "Hello"
//! craft-ai-cli devices
//! craft-ai-cli transcribe --file talk.mp4 --model whisper-large-v3-turbo --device gpu --language vi
//! ```
//!
//! `--device` picks the processor: `auto` (the default: a GPU when this build supports one and the
//! machine has it, else the CPU), `cpu`, `gpu` (any GPU, an error without one), `cuda[:N]` or
//! `metal[:N]`. GPUs need a build with the `cuda` or `metal` feature. The CPU uses every core;
//! set `RAYON_NUM_THREADS` to use fewer.
//!
//! Models live in `--dir`, else `CRAFT_AI_MODELS`, else the suite's shared cache. A token for an
//! endpoint is read from the environment variable `--token-env` names (default
//! `CRAFT_AI_TOKEN`), never from the command line, so it doesn't show in process lists.
//!
//! The model catalogue is the built-in models plus `<model folder>/catalog.json`, the files in
//! `CRAFT_AI_CATALOG` and `--catalog FILE` (see `craft_ai::catalog`): a model there is one id
//! whether it is downloaded, a file on disk or an OpenAI-compatible endpoint. Without `--model`,
//! `CRAFT_AI_DEFAULT_MODEL` (else qwen3-4b-q4km); without `--device`, the model's catalogue
//! device, else `CRAFT_AI_DEVICE`, else auto.

use std::path::PathBuf;
use std::process::ExitCode;

use craft_ai::{GenOpts, PdfGrammar, Provider, Segment, fits, segment, suggest, translate_segments};
use serde_json::{Value, json};

const HELP: &str = "\
craft-ai-cli models   [--dir DIR] [--catalog FILE]
craft-ai-cli commands   (every command with its inputs, as JSON: the same listing every engine gives)
craft-ai-cli devices
craft-ai-cli transcribe --file AUDIO [--model whisper-small] [--device D] [--language vi|auto] [--speakers N]
  any audio or video FFmpeg reads (raw mono 16 kHz f32 when it ends in .f32); --model is a built-in Whisper
  model, or a catalogue speech model (a Whisper folder or an endpoint); default CRAFT_AI_SPEECH_MODEL, else
  whisper-small
craft-ai-cli download <model id> [--dir DIR] [--catalog FILE]
craft-ai-cli live --in URL|DEVICE|FILE [--in-format F] [--model SPEECH] [--device D] [--language vi|auto]
  [--translate vi,en] [--translate-model CHAT] [--translate-device D] --captions SPEC… [--pause 0.6]
  [--max-segment 8] [--seconds N]
  live captions as they are spoken: each pause ends a caption, which is translated into each language.
  --captions (repeatable): vtt:PATH, srt:PATH, jsonl:PATH (jsonl:- standard output) or an http(s) webhook;
  {lang} in it makes one output per language (the spoken one and each translation)
craft-ai-cli summarize <model> (--text T | --file F) [--length short|medium|bullets] [--language L]
craft-ai-cli extract   <model> (--text T | --file F) --fields a,b,c [--instructions TEXT]
craft-ai-cli chat      <model> --messages FILE.json   (a JSON list of {role: system, user or assistant; content})
craft-ai-cli ask       <model> (--text T | --file F) --question Q
craft-ai-cli detect-language (--text T | --file F) [<model>]   (without a model: by script and common words)
craft-ai-cli translate-subtitles <model> --file IN --to L --out OUT   (SRT, WebVTT, TTML, SCC…; times kept)
craft-ai-cli translate-xliff     <model> --file IN [--to L] [--overwrite] --out OUT   (inline codes kept)
craft-ai-cli embed     <model> (--text T | --texts-file FILE.json)   (an endpoint model: OpenAI-compatible /embeddings)
craft-ai-cli ocr       --file IMAGE [--languages vie+eng]   (Tesseract on the CPU; words with boxes)
craft-ai-cli complete  <model> (--prompt TEXT | --prompt-file FILE) [--max-tokens N] [--temperature T]
craft-ai-cli translate <model> --to LANGUAGE (--text TEXT | --file FILE | --segments-file FILE.json)
  <model>: [--model ID] [--dir DIR] [--catalog FILE] [--device D] [--template T [--stop A,B]]
           a model from `models`: built in, or from a catalogue (downloaded, a file, or an endpoint)
       or: --model FILE.gguf [--tokenizer FILE]   any GGUF of a qwen2, qwen3, llama, gemma3 or phi3
           architecture (the tokenizer defaults to tokenizer.json beside it)
  --template: chatml, qwen3, llama3, gemma, phi3, mistral, or a text with {system} and {prompt}
           (then --stop names the end tokens); default: the catalogue's, else the file's own
       or: --endpoint URL --remote-model NAME [--token-env VAR]   any OpenAI-compatible API
  --segments-file: a JSON array of strings; the answer keeps their order and count, and
  protected runs (URLs, e-mail addresses, {placeholders}, <markup>, numbers) come back unchanged.
  Each text goes to the model whole, so sentences around a placeholder are translated as one; a
  translation that loses or adds a protected run is redone split at them. --split always splits.
  --device: auto (default), cpu, gpu, cuda, cuda:N, metal, metal:N.";

struct Args(Vec<String>);

impl Args {
    /// Every value of a flag given more than once.
    fn flags(&self, name: &str) -> Vec<&str> {
        self.0.windows(2).filter(|w| w[0] == name).map(|w| w[1].as_str()).collect()
    }
    fn flag(&self, name: &str) -> Option<&str> {
        self.0.iter().position(|a| a == name).and_then(|i| self.0.get(i + 1)).map(String::as_str)
    }
    fn dir(&self) -> Result<PathBuf, String> {
        if let Some(d) = self.flag("--dir") {
            return Ok(PathBuf::from(d));
        }
        if let Some(d) = std::env::var_os("CRAFT_AI_MODELS").filter(|d| !d.is_empty()) {
            return Ok(PathBuf::from(d));
        }
        craft_ai::probe::models_dir().ok_or_else(|| "no model folder: pass --dir or set CRAFT_AI_MODELS".to_string())
    }
    fn num<T: std::str::FromStr>(&self, name: &str) -> Result<Option<T>, String> {
        self.flag(name).map(|v| v.parse().map_err(|_| format!("{name}: not a number: {v}"))).transpose()
    }
}

fn main() -> ExitCode {
    let args = Args(std::env::args().skip(1).collect());
    let result = match args.0.first().map(String::as_str) {
        Some("models") => models(&args),
        Some("devices") => devices(),
        Some("commands") => Ok(commands()),
        Some("download") => download(&args),
        Some("complete") => complete(&args),
        Some("translate") => translate(&args),
        Some("transcribe") => transcribe(&args),
        Some("live") => live(&args),
        Some("summarize") => summarize(&args),
        Some("extract") => extract(&args),
        Some("chat") => chat(&args),
        Some("ask") => ask(&args),
        Some("detect-language") => detect_language(&args),
        Some("translate-subtitles") => translate_subtitles(&args),
        Some("translate-xliff") => translate_xliff(&args),
        Some("embed") => embed(&args),
        Some("ocr") => ocr(&args),
        Some("help" | "--help" | "-h") | None => {
            println!("{HELP}");
            return ExitCode::SUCCESS;
        }
        Some(other) => Err(format!("unknown command {other:?}\n{HELP}")),
    };
    match result {
        Ok(v) => {
            println!("{v}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn catalog(a: &Args, dir: &std::path::Path) -> Result<craft_ai::catalog::Catalog, String> {
    let mut files = craft_ai::catalog::default_files(dir);
    if let Some(f) = a.flag("--catalog") {
        files.push(PathBuf::from(f));
    }
    craft_ai::catalog::Catalog::load(&files).map_err(|e| e.to_string())
}

fn models(a: &Args) -> Result<Value, String> {
    use craft_ai::catalog::Source;
    let dir = a.dir()?;
    let caps = craft_ai::probe::probe(&dir);
    let suggested = suggest(&caps).map(|m| m.id);
    let cat = catalog(a, &dir)?;
    let list: Vec<Value> = cat
        .models
        .iter()
        .map(|m| {
            let mut v = json!({
                "id": m.id, "name": m.name, "kind": m.kind(), "from": m.origin,
                "arch": m.arch.map(|x| format!("{x:?}").to_lowercase()), "template": m.template.as_ref().map(ToString::to_string),
                "device": m.device.map(|d| d.to_string()),
            });
            match &m.source {
                Source::Download(e) => {
                    v["size_mb"] = json!(e.size_mb);
                    v["min_ram_gb"] = json!(e.min_ram_gb);
                    v["rec_ram_gb"] = json!(e.rec_ram_gb);
                    v["license"] = json!(e.license);
                    v["languages"] = json!(e.langs);
                    v["quality"] = json!(format!("{:?}", e.quality));
                    v["downloaded"] = json!(craft_ai::fetch::is_downloaded(e, &dir));
                    v["fits"] = json!(fits(&caps, e));
                    v["suggested"] = json!(suggested == Some(e.id));
                }
                Source::File { gguf, tokenizer } => {
                    v["path"] = json!(gguf);
                    v["tokenizer"] = json!(tokenizer);
                    v["present"] = json!(gguf.is_file() && tokenizer.is_file());
                }
                Source::Endpoint { endpoint, model, token_env } => {
                    v["endpoint"] = json!(endpoint);
                    v["remote_model"] = json!(model);
                    v["token_env"] = json!(token_env);
                }
            }
            v
        })
        .collect();
    Ok(json!({
        "dir": dir,
        "machine": { "ram_gb": caps.ram_gb, "threads": caps.threads, "avx2": caps.avx2, "disk_free_gb": caps.disk_free_gb },
        "backends": craft_ai::compiled_backends(),
        "default_model": default_model(),
        "models": list,
        "speech": speech_models(&dir)?,
    }))
}

fn default_model() -> String {
    craft_ai::open::default_model()
}

/// The back ends this build has and the GPUs it can open here.
fn devices() -> Result<Value, String> {
    let gpus: Vec<Value> = found_gpus().into_iter().map(|(device, label)| json!({ "device": device, "label": label })).collect();
    let auto = if gpus.is_empty() { "cpu".to_string() } else { gpus[0]["device"].as_str().unwrap_or("cpu").to_string() };
    Ok(json!({ "backends": craft_ai::compiled_backends(), "gpus": gpus, "auto": auto }))
}

#[cfg(feature = "local")]
fn found_gpus() -> Vec<(String, String)> {
    craft_ai::local::gpus()
}

#[cfg(not(feature = "local"))]
fn found_gpus() -> Vec<(String, String)> {
    Vec::new()
}

fn download(a: &Args) -> Result<Value, String> {
    let id = a.0.get(1).filter(|s| !s.starts_with("--")).ok_or("download: which model? (see `models`)")?;
    let dir = a.dir()?;
    if let Some(m) = craft_ai::speech::models::find(id) {
        std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        craft_ai::speech::models::download(&dir, m, &mut |_, _, _| true).map_err(|e| e.to_string())?;
        return Ok(json!({ "model": m.id, "task": "speech-to-text", "dir": craft_ai::speech::models::model_dir(&dir, m), "verified": true }));
    }
    let cat = catalog(a, &dir)?;
    let m = cat.find(id).ok_or_else(|| format!("unknown model {id:?} (see `models`)"))?;
    let craft_ai::catalog::Source::Download(entry) = m.source else {
        return Err(format!("{id} is a {} model: nothing to download", m.kind()));
    };
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let verified = std::cell::Cell::new(false);
    let path = craft_ai::fetch::download(
        entry,
        &dir,
        |e| {
            if let craft_ai::fetch::Event::Done { verified: v } = e {
                verified.set(*v);
            }
        },
        None,
    )
    .map_err(|e| e.to_string())?;
    let tokenizer = craft_ai::fetch::download_tokenizer(entry, &dir).map_err(|e| e.to_string())?;
    Ok(json!({ "model": entry.id, "path": path, "tokenizer": tokenizer, "verified": verified.get() }))
}

/// The provider the arguments name: a remote endpoint, or a model from the catalogue (which may
/// itself be an endpoint), or a GGUF file.
fn provider(a: &Args) -> Result<Box<dyn Provider>, String> {
    provider_with(a, a.flag("--model"), a.flag("--device"))
}

/// [`provider`] with the model and device taken from other flags (live translation has its own).
fn provider_with(a: &Args, model: Option<&str>, device_flag: Option<&str>) -> Result<Box<dyn Provider>, String> {
    if let Some(endpoint) = a.flag("--endpoint") {
        let model = a.flag("--remote-model").ok_or("--endpoint needs --remote-model")?;
        let var = a.flag("--token-env").unwrap_or("CRAFT_AI_TOKEN");
        return Ok(remote(endpoint, model, var));
    }
    let template = match a.flag("--template") {
        Some(t) => {
            let stop: Vec<String> = a.flag("--stop").map(|s| s.split(',').map(str::to_string).collect()).unwrap_or_default();
            Some(craft_ai::Template::from_parts(t, &stop)?)
        }
        None => None,
    };
    let choice = craft_ai::open::ChatChoice {
        model: model.map(str::to_string),
        device: device_flag.map(str::parse::<craft_ai::DeviceChoice>).transpose()?,
        template,
        catalog: a.flag("--catalog").map(PathBuf::from),
        dir: Some(a.dir()?),
        tokenizer: a.flag("--tokenizer").map(PathBuf::from),
    };
    craft_ai::open::open_chat(&choice).map_err(|e| e.to_string())
}

fn remote(endpoint: &str, model: &str, token_env: &str) -> Box<dyn Provider> {
    let token = std::env::var(token_env).ok().filter(|t| !t.is_empty());
    Box::new(craft_ai::OpenAiCompat::new(endpoint, token, model))
}

fn opts(a: &Args) -> Result<GenOpts, String> {
    let mut o = GenOpts::default();
    if let Some(n) = a.num("--max-tokens")? {
        o.max_tokens = n;
    }
    if let Some(t) = a.num("--temperature")? {
        o.temperature = t;
    }
    Ok(o)
}

fn read(path: &str) -> Result<String, String> {
    std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))
}

fn complete(a: &Args) -> Result<Value, String> {
    let prompt = match (a.flag("--prompt"), a.flag("--prompt-file")) {
        (Some(p), _) => p.to_string(),
        (None, Some(f)) => read(f)?,
        (None, None) => return Err("complete: give --prompt or --prompt-file".into()),
    };
    let o = opts(a)?;
    let mut p = provider(a)?;
    let text = p.complete(&prompt, &o).map_err(|e| e.to_string())?;
    Ok(json!({ "provider": p.name(), "local": p.is_local(), "text": text }))
}

fn translate(a: &Args) -> Result<Value, String> {
    let to = a.flag("--to").ok_or("translate: --to LANGUAGE")?;
    let (texts, list) = match (a.flag("--text"), a.flag("--file"), a.flag("--segments-file")) {
        (Some(t), _, _) => (vec![t.to_string()], false),
        (None, Some(f), _) => (vec![read(f)?], false),
        (None, None, Some(f)) => {
            let v: Vec<String> = serde_json::from_str(&read(f)?).map_err(|e| format!("{f}: a JSON array of strings: {e}"))?;
            (v, true)
        }
        _ => return Err("translate: give --text, --file or --segments-file".into()),
    };
    let mut p = provider(a)?;
    let joined = if a.0.iter().any(|x| x == "--split") { split_translate(p.as_mut(), &texts, to)? } else { whole_translate(p.as_mut(), &texts, to)? };
    let mut v = json!({ "provider": p.name(), "local": p.is_local(), "to": to });
    if list {
        v["segments"] = json!(joined);
    } else {
        v["text"] = json!(joined.into_iter().next().unwrap_or_default());
    }
    Ok(v)
}

/// Every input's segments in one list, so the model sees whole batches; each input's range puts
/// its translation back together. Protected runs never reach the model.
fn split_translate(p: &mut dyn Provider, texts: &[String], to: &str) -> Result<Vec<String>, String> {
    let mut all: Vec<Segment> = Vec::new();
    let mut ranges = Vec::with_capacity(texts.len());
    for t in texts {
        let start = all.len();
        all.extend(segment(t, &PdfGrammar));
        ranges.push(start..all.len());
    }
    let out = translate_segments(p, &all, to).map_err(|e| e.to_string())?;
    Ok(ranges.into_iter().map(|r| out[r].concat()).collect())
}

/// Each input whole, so a sentence with a placeholder in it is translated as one sentence; the
/// model is told to keep protected runs, and an input whose translation doesn't keep exactly the
/// same ones (`protected_runs`) is redone split, where they can't change.
fn whole_translate(p: &mut dyn Provider, texts: &[String], to: &str) -> Result<Vec<String>, String> {
    let whole: Vec<Segment> = texts
        .iter()
        .map(|t| if protected_runs(t).len() == t.split_whitespace().count() { Segment::Protected(t.clone()) } else { Segment::Text(t.clone()) })
        .collect();
    let mut out = translate_segments(p, &whole, to).map_err(|e| e.to_string())?;
    let redo: Vec<usize> = (0..texts.len()).filter(|&i| protected_runs(&texts[i]) != protected_runs(&out[i])).collect();
    if !redo.is_empty() {
        let again = split_translate(p, &redo.iter().map(|&i| texts[i].clone()).collect::<Vec<_>>(), to)?;
        for (i, t) in redo.into_iter().zip(again) {
            out[i] = t;
        }
    }
    Ok(out)
}

/// The protected runs of `text`, sorted: whitespace-separated tokens the grammar protects (URLs,
/// e-mail addresses, numbers…) and every `{…}` placeholder, also one inside a word.
fn protected_runs(text: &str) -> Vec<String> {
    let mut runs: Vec<String> = text
        .split_whitespace()
        .map(|t| t.trim_matches(|c: char| matches!(c, ',' | ';' | ':' | '!' | '?' | '(' | ')' | '"' | '\'')))
        .filter(|t| !t.is_empty() && t.chars().any(char::is_alphanumeric) && craft_ai::is_protected_token(t) && !t.contains('{'))
        .map(str::to_string)
        .collect();
    let mut rest = text;
    while let Some(open) = rest.find('{') {
        match rest[open..].find('}') {
            Some(close) => {
                runs.push(rest[open..open + close + 1].to_string());
                rest = &rest[open + close + 1..];
            }
            None => break,
        }
    }
    runs.sort();
    runs
}

/// The speech models: built in (Whisper, downloaded into the model folder) and the catalogue's.
fn speech_models(dir: &std::path::Path) -> Result<Vec<Value>, String> {
    use craft_ai::speech::{catalog, models};
    let mut v: Vec<Value> = models::catalogue()
        .iter()
        .map(|m| {
            json!({ "id": m.id, "name": m.name, "kind": "download", "task": "speech-to-text", "size": m.size(),
                "multilingual": m.multilingual, "license": m.license, "downloaded": models::installed(dir, m) })
        })
        .collect();
    for m in catalog::load(&catalog::default_files(dir)).map_err(|e| e.to_string())? {
        v.retain(|x| x["id"] != m.id.as_str());
        v.push(match &m.source {
            catalog::Source::Folder(f) => json!({ "id": m.id, "name": m.name, "kind": "folder", "task": "speech-to-text", "dir": f,
                "downloaded": f.join("model.safetensors").is_file(), "device": m.device.map(|d| d.to_string()), "from": m.origin }),
            catalog::Source::Endpoint { endpoint, model, token_env } => json!({ "id": m.id, "name": m.name, "kind": "endpoint",
                "task": "speech-to-text", "endpoint": endpoint, "remote_model": model, "token_env": token_env, "from": m.origin }),
        });
    }
    Ok(v)
}

/// Mono 16 kHz samples of `file`: read directly when it is raw f32 (`.f32`), else decoded by FFmpeg
/// (`CRAFT_LIVE_FFMPEG`, else `ffmpeg`).
fn audio(file: &str) -> Result<Vec<f32>, String> {
    let bytes = if file.ends_with(".f32") {
        std::fs::read(file).map_err(|e| format!("{file}: {e}"))?
    } else {
        let ffmpeg = std::env::var("CRAFT_LIVE_FFMPEG").unwrap_or_else(|_| "ffmpeg".into());
        let out = std::process::Command::new(&ffmpeg)
            .args(["-hide_banner", "-loglevel", "error", "-i", file, "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "-"])
            .output()
            .map_err(|e| format!("FFmpeg is needed to read {file}: {ffmpeg}: {e}"))?;
        if !out.status.success() {
            return Err(format!("FFmpeg could not read {file}: {}", String::from_utf8_lossy(&out.stderr).trim()));
        }
        out.stdout
    };
    Ok(bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect())
}

fn transcribe(a: &Args) -> Result<Value, String> {
    let file = a.flag("--file").ok_or("transcribe: --file AUDIO")?;
    let id = a
        .flag("--model")
        .map(str::to_string)
        .or_else(|| std::env::var("CRAFT_AI_SPEECH_MODEL").ok().filter(|m| !m.is_empty()))
        .unwrap_or_else(|| "whisper-small".into());
    let device = a.flag("--device").map(str::parse::<craft_ai::DeviceChoice>).transpose()?;
    let dir = a.dir()?;
    if !craft_ai::speech::known(&dir, &id) {
        return Err(format!("unknown speech model {id:?} (see `models`)"));
    }
    let t = craft_ai::speech::load(&dir, &id, device).map_err(|e| e.to_string())?;
    let samples = audio(file)?;
    let mut o = craft_ai::speech::Options { language: a.flag("--language").map(str::to_string).filter(|l| l != "auto"), ..Default::default() };
    match a.num::<usize>("--speakers")? {
        Some(n) => o.max_speakers = n.max(1),
        None => o.diarize = false,
    }
    let tr = t.transcribe(&samples, &o, &mut |_, _| true).map_err(|e| e.to_string())?;
    let words: Vec<Value> = tr
        .words
        .iter()
        .map(|w| json!({ "text": w.text, "start": w.start.seconds(), "end": w.end.seconds(), "speaker": w.speaker, "confidence": w.confidence }))
        .collect();
    Ok(json!({
        "model": t.id(), "device": t.device(), "language": tr.language, "seconds": samples.len() as f64 / 16_000.0,
        "text": tr.text(), "speakers": tr.speakers.iter().map(|s| s.name.clone()).collect::<Vec<_>>(), "words": words,
    }))
}

fn live(a: &Args) -> Result<Value, String> {
    use craft_ai::speech::live::{Langs, LiveOptions, Output, run};
    let input = a.flag("--in").ok_or("live: --in URL, a device or a file")?;
    let dir = a.dir()?;
    let id = a
        .flag("--model")
        .map(str::to_string)
        .or_else(|| std::env::var("CRAFT_AI_SPEECH_MODEL").ok().filter(|m| !m.is_empty()))
        .unwrap_or_else(|| "whisper-small".into());
    if !craft_ai::speech::known(&dir, &id) {
        return Err(format!("unknown speech model {id:?} (see `models`)"));
    }
    let device = a.flag("--device").map(str::parse::<craft_ai::DeviceChoice>).transpose()?;
    let speech = craft_ai::speech::load(&dir, &id, device).map_err(|e| e.to_string())?;
    let language = a.flag("--language").map(str::to_string).filter(|l| !l.is_empty() && l != "auto");
    let langs: Vec<String> =
        a.flag("--translate").map(|l| l.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect()).unwrap_or_default();
    let mut translate: Vec<(String, Box<dyn Provider>)> = Vec::new();
    for l in &langs {
        translate.push((l.clone(), provider_with(a, a.flag("--translate-model"), a.flag("--translate-device"))?));
    }
    let specs = a.flags("--captions");
    if specs.is_empty() {
        return Err("live: give at least one --captions output (jsonl:- prints them)".into());
    }
    let spoken = language.clone().unwrap_or_else(|| "spoken".into());
    let mut outputs = Vec::new();
    for spec in specs {
        if spec.contains("{lang}") {
            // the spoken language's output: that language, or (detected) any but the translations'
            let own = language.clone().map_or_else(|| Langs::Except(langs.clone()), Langs::Only);
            for (name, takes) in std::iter::once((spoken.clone(), own)).chain(langs.iter().map(|l| (l.clone(), Langs::Only(l.clone())))) {
                let sink = craft_live::CaptionSink::open(&spec.replace("{lang}", &name)).map_err(|e| format!("{spec}: {e}"))?;
                outputs.push(Output { langs: takes, sink });
            }
        } else {
            outputs.push(Output { langs: Langs::All, sink: craft_live::CaptionSink::open(spec).map_err(|e| format!("{spec}: {e}"))? });
        }
    }
    let o = LiveOptions {
        input: input.to_string(),
        input_format: a.flag("--in-format").map(str::to_string),
        seconds: a.num("--seconds")?.unwrap_or(0),
        language,
        pause: a.num("--pause")?.unwrap_or(0.6),
        max_segment: a.num("--max-segment")?.unwrap_or(8.0),
    };
    let stats = run(&o, speech.as_ref(), &mut translate, &mut outputs, &mut std::io::stderr()).map_err(|e| e.to_string())?;
    Ok(json!({ "model": speech.id(), "device": speech.device(), "stats": stats }))
}

/// Every command with its inputs, in the form the engines' `commands` use.
fn commands() -> Value {
    let p = |name: &str, kind: &str, about: &str| json!({ "name": name, "type": kind, "description": about });
    let model = p("--model", "string", "a model of the catalogue (`models`), or a .gguf file");
    let device = p("--device", "string", "auto, cpu, gpu, cuda, cuda:N, metal, metal:N");
    let list = vec![
        json!({ "id": "models", "label": "List models", "task": "all", "inputs": [p("--dir", "string", "the model folder"), p("--catalog", "string", "another catalogue file")] }),
        json!({ "id": "devices", "label": "List devices", "task": "all", "inputs": [] }),
        json!({ "id": "download", "label": "Download a model", "task": "all", "inputs": [p("<id>", "string", "a downloadable model")] }),
        json!({ "id": "complete", "label": "Complete a prompt", "task": "chat", "inputs": [model.clone(), device.clone(), p("--prompt", "string", "the prompt"), p("--prompt-file", "string", "or a file holding it"), p("--max-tokens", "integer", ""), p("--temperature", "number", ""), p("--endpoint", "string", "an OpenAI-compatible API"), p("--remote-model", "string", "its model name")] }),
        json!({ "id": "translate", "label": "Translate", "task": "chat", "inputs": [model.clone(), device.clone(), p("--to", "string", "the language"), p("--text", "string", ""), p("--file", "string", ""), p("--segments-file", "string", "a JSON array of strings"), p("--split", "boolean", "")] }),
        json!({ "id": "summarize", "label": "Summarize", "task": "chat", "inputs": [model.clone(), device.clone(), p("--text", "string", ""), p("--file", "string", ""), p("--length", "string", "short, medium or bullets"), p("--language", "string", "")] }),
        json!({ "id": "extract", "label": "Extract fields", "task": "chat", "inputs": [model.clone(), device.clone(), p("--text", "string", ""), p("--file", "string", ""), p("--fields", "string", "comma-separated"), p("--instructions", "string", "")] }),
        json!({ "id": "chat", "label": "Chat", "task": "chat", "inputs": [model.clone(), device.clone(), p("--messages", "string", "a JSON file of {role, content}"), p("--max-tokens", "integer", ""), p("--temperature", "number", "")] }),
        json!({ "id": "ask", "label": "Ask a document", "task": "chat", "inputs": [model.clone(), device.clone(), p("--text", "string", ""), p("--file", "string", ""), p("--question", "string", "")] }),
        json!({ "id": "detect-language", "label": "Detect the language", "task": "chat", "inputs": [p("--text", "string", ""), p("--file", "string", ""), p("--model", "string", "optional: ask a model")] }),
        json!({ "id": "translate-subtitles", "label": "Translate subtitles", "task": "chat", "inputs": [model.clone(), device.clone(), p("--file", "string", "SRT, WebVTT, TTML, SCC…"), p("--to", "string", ""), p("--out", "string", "")] }),
        json!({ "id": "translate-xliff", "label": "Translate XLIFF", "task": "chat", "inputs": [model.clone(), device.clone(), p("--file", "string", ""), p("--to", "string", ""), p("--overwrite", "boolean", ""), p("--out", "string", "")] }),
        json!({ "id": "embed", "label": "Embeddings", "task": "embeddings", "inputs": [p("--model", "string", "an endpoint model"), p("--text", "string", ""), p("--texts-file", "string", "a JSON array of strings")] }),
        json!({ "id": "ocr", "label": "Read text from an image", "task": "ocr", "inputs": [p("--file", "string", "an image"), p("--languages", "string", "Tesseract codes, e.g. vie+eng")] }),
        json!({ "id": "transcribe", "label": "Transcribe speech", "task": "speech-to-text", "inputs": [model.clone(), device.clone(), p("--file", "string", "audio or video"), p("--language", "string", "a code, or auto"), p("--speakers", "integer", "label up to this many speakers")] }),
        json!({ "id": "live", "label": "Live captions and translation", "task": "speech-to-text", "inputs": [p("--in", "string", "a URL, a device or a file"), p("--in-format", "string", ""), model, device, p("--language", "string", ""), p("--translate", "string", "languages, comma-separated"), p("--translate-model", "string", ""), p("--translate-device", "string", ""), p("--captions", "string", "vtt:PATH, srt:PATH, jsonl:PATH or a webhook; repeatable; {lang}"), p("--pause", "number", ""), p("--max-segment", "number", ""), p("--seconds", "integer", "")] }),
    ];
    json!({ "program": "craft-ai-cli", "commands": list })
}

/// The text a task works on: --text, or --file read as UTF-8.
fn input_text(a: &Args) -> Result<String, String> {
    match (a.flag("--text"), a.flag("--file")) {
        (Some(t), _) => Ok(t.to_string()),
        (None, Some(f)) => read(f),
        _ => Err("give --text or --file".into()),
    }
}

fn summarize(a: &Args) -> Result<Value, String> {
    let text = input_text(a)?;
    let length: craft_ai::tasks::Length = a.flag("--length").unwrap_or("medium").parse()?;
    let mut p = provider(a)?;
    let (summary, cut) = craft_ai::tasks::summarize(p.as_mut(), &text, length, a.flag("--language").unwrap_or("")).map_err(|e| e.to_string())?;
    Ok(json!({ "summary": summary, "truncated": cut, "provider": p.name(), "local": p.is_local() }))
}

fn extract(a: &Args) -> Result<Value, String> {
    let text = input_text(a)?;
    let fields: Vec<String> =
        a.flag("--fields").ok_or("extract: --fields a,b,c")?.split(',').map(|f| f.trim().to_string()).filter(|f| !f.is_empty()).collect();
    let mut p = provider(a)?;
    let v = craft_ai::tasks::extract(p.as_mut(), &text, &fields, a.flag("--instructions").unwrap_or("")).map_err(|e| e.to_string())?;
    Ok(json!({ "values": v, "provider": p.name(), "local": p.is_local() }))
}

fn chat(a: &Args) -> Result<Value, String> {
    let f = a.flag("--messages").ok_or("chat: --messages FILE.json")?;
    let messages: Vec<craft_ai::tasks::Message> = serde_json::from_str(&read(f)?).map_err(|e| format!("{f}: [{{\"role\", \"content\"}}, …]: {e}"))?;
    let o = opts(a)?;
    let mut p = provider(a)?;
    let answer = craft_ai::tasks::chat(p.as_mut(), &messages, &o).map_err(|e| e.to_string())?;
    Ok(json!({ "answer": answer, "provider": p.name(), "local": p.is_local() }))
}

fn ask(a: &Args) -> Result<Value, String> {
    let text = input_text(a)?;
    let q = a.flag("--question").ok_or("ask: --question Q")?;
    let mut p = provider(a)?;
    let (answer, used) = craft_ai::tasks::ask(p.as_mut(), &text, q).map_err(|e| e.to_string())?;
    Ok(json!({ "answer": answer, "extracts": used, "provider": p.name(), "local": p.is_local() }))
}

fn detect_language(a: &Args) -> Result<Value, String> {
    let text = input_text(a)?;
    if a.flag("--model").is_some() || a.flag("--endpoint").is_some() {
        let mut p = provider(a)?;
        let code = craft_ai::tasks::detect_language_with(p.as_mut(), &text).map_err(|e| e.to_string())?;
        return Ok(json!({ "language": code, "by": p.name() }));
    }
    Ok(match craft_ai::tasks::detect_language(&text) {
        Some((code, c)) => json!({ "language": code, "confidence": c, "by": "script and common words" }),
        None => json!({ "language": null, "by": "script and common words", "note": "too little text to tell; give --model to ask a model" }),
    })
}

fn translate_subtitles(a: &Args) -> Result<Value, String> {
    let f = a.flag("--file").ok_or("translate-subtitles: --file IN")?;
    let to = a.flag("--to").ok_or("translate-subtitles: --to LANGUAGE")?;
    let out = a.flag("--out").ok_or("translate-subtitles: --out OUT")?;
    let bytes = std::fs::read(f).map_err(|e| format!("{f}: {e}"))?;
    let ext = std::path::Path::new(f).extension().and_then(|e| e.to_str()).map(str::to_lowercase);
    let mut p = provider(a)?;
    let (data, n) = craft_ai::tasks::translate_captions(p.as_mut(), &bytes, ext.as_deref(), to).map_err(|e| e.to_string())?;
    std::fs::write(out, data).map_err(|e| format!("{out}: {e}"))?;
    Ok(json!({ "cues": n, "to": to, "out": out, "provider": p.name(), "local": p.is_local() }))
}

fn translate_xliff(a: &Args) -> Result<Value, String> {
    let f = a.flag("--file").ok_or("translate-xliff: --file IN")?;
    let out = a.flag("--out").ok_or("translate-xliff: --out OUT")?;
    let bytes = std::fs::read(f).map_err(|e| format!("{f}: {e}"))?;
    let mut p = provider(a)?;
    let (data, st) = craft_ai::tasks::translate_xliff(p.as_mut(), &bytes, a.flag("--to").unwrap_or(""), a.0.iter().any(|x| x == "--overwrite"))
        .map_err(|e| e.to_string())?;
    std::fs::write(out, data).map_err(|e| format!("{out}: {e}"))?;
    Ok(json!({ "stats": st, "out": out, "provider": p.name(), "local": p.is_local() }))
}

fn embed(a: &Args) -> Result<Value, String> {
    let texts: Vec<String> = match (a.flag("--text"), a.flag("--texts-file")) {
        (Some(t), _) => vec![t.to_string()],
        (None, Some(f)) => serde_json::from_str(&read(f)?).map_err(|e| format!("{f}: a JSON array of strings: {e}"))?,
        _ => return Err("embed: --text or --texts-file".into()),
    };
    let (endpoint, model, token_env) = match a.flag("--endpoint") {
        Some(e) => (
            e.to_string(),
            a.flag("--remote-model").ok_or("--endpoint needs --remote-model")?.to_string(),
            a.flag("--token-env").unwrap_or("CRAFT_AI_TOKEN").to_string(),
        ),
        None => {
            let id = a.flag("--model").ok_or("embed: --model (an endpoint model of the catalogue) or --endpoint")?;
            let dir = a.dir()?;
            let cat = catalog(a, &dir)?;
            match cat.find(id).map(|m| &m.source) {
                Some(craft_ai::catalog::Source::Endpoint { endpoint, model, token_env }) => (endpoint.clone(), model.clone(), token_env.clone()),
                Some(_) => return Err(format!("{id} runs here, and local embedding models aren't supported yet: use an endpoint model")),
                None => return Err(format!("unknown model {id:?} (see `models`)")),
            }
        }
    };
    let token = std::env::var(&token_env).ok().filter(|t| !t.is_empty());
    let p = craft_ai::OpenAiCompat::new(endpoint, token, model);
    let v = p.embed(&texts).map_err(|e| e.to_string())?;
    Ok(json!({ "model": p.model, "dimensions": v.first().map_or(0, Vec::len), "embeddings": v }))
}

fn ocr(a: &Args) -> Result<Value, String> {
    let f = a.flag("--file").ok_or("ocr: --file IMAGE")?;
    if a.flag("--model").is_some() {
        return Err("ocr runs on Tesseract for now; OCR and vision models through --model are not supported yet".into());
    }
    let r = craft_ai::ocr::ocr(std::path::Path::new(f), a.flag("--languages").unwrap_or("eng")).map_err(|e| e.to_string())?;
    serde_json::to_value(r).map_err(|e| e.to_string())
}
