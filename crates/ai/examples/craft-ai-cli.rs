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
craft-ai-cli devices
craft-ai-cli transcribe --file AUDIO [--model whisper-small] [--device D] [--language vi|auto] [--speakers N]
  any audio or video FFmpeg reads (raw mono 16 kHz f32 when it ends in .f32); --model is a built-in Whisper
  model, or a catalogue speech model (a Whisper folder or an endpoint); default CRAFT_AI_SPEECH_MODEL, else
  whisper-small
craft-ai-cli download <model id> [--dir DIR] [--catalog FILE]
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
        Some("download") => download(&args),
        Some("complete") => complete(&args),
        Some("translate") => translate(&args),
        Some("transcribe") => transcribe(&args),
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
    std::env::var("CRAFT_AI_DEFAULT_MODEL").ok().filter(|m| !m.is_empty()).unwrap_or_else(|| "qwen3-4b-q4km".into())
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
    use craft_ai::catalog::Source;
    if let Some(endpoint) = a.flag("--endpoint") {
        let model = a.flag("--remote-model").ok_or("--endpoint needs --remote-model")?;
        let var = a.flag("--token-env").unwrap_or("CRAFT_AI_TOKEN");
        return Ok(remote(endpoint, model, var));
    }
    let id = a.flag("--model").map_or_else(default_model, str::to_string);
    let template = match a.flag("--template") {
        Some(t) => {
            let stop: Vec<String> = a.flag("--stop").map(|s| s.split(',').map(str::to_string).collect()).unwrap_or_default();
            Some(craft_ai::Template::from_parts(t, &stop)?)
        }
        None => None,
    };
    let device = |own: Option<craft_ai::DeviceChoice>| -> Result<craft_ai::DeviceChoice, String> {
        if let Some(d) = a.flag("--device") {
            return d.parse();
        }
        if let Some(d) = own {
            return Ok(d);
        }
        std::env::var("CRAFT_AI_DEVICE").ok().filter(|d| !d.is_empty()).map_or(Ok(craft_ai::DeviceChoice::Auto), |d| d.parse())
    };
    let dir = a.dir()?;
    let cat = catalog(a, &dir)?;
    if let Some(m) = cat.find(&id) {
        let template = template.or_else(|| m.template.clone());
        return match &m.source {
            Source::Endpoint { endpoint, model, token_env } => Ok(remote(endpoint, model, token_env)),
            Source::Download(e) => {
                if !craft_ai::fetch::is_downloaded(e, &dir) {
                    return Err(format!("{id} is not downloaded in {} (craft-ai-cli download {id})", dir.display()));
                }
                let (gguf, tok) = (craft_ai::fetch::model_path(e, &dir), craft_ai::fetch::tokenizer_path(e, &dir));
                local(&m.id, &m.name, &gguf, &tok, m.arch, template, device(m.device)?)
            }
            Source::File { gguf, tokenizer } => local(&m.id, &m.name, gguf, tokenizer, m.arch, template, device(m.device)?),
        };
    }
    let path = std::path::Path::new(&id);
    if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("gguf")) {
        let tokenizer = a.flag("--tokenizer").map_or_else(|| path.with_file_name("tokenizer.json"), PathBuf::from);
        let name = path.file_stem().map_or_else(|| id.clone(), |s| s.to_string_lossy().into_owned());
        return local(&id, &name, path, &tokenizer, None, template, device(None)?);
    }
    Err(format!("unknown model {id:?} (see `models`, add it to a catalogue, or give a .gguf file)"))
}

fn remote(endpoint: &str, model: &str, token_env: &str) -> Box<dyn Provider> {
    let token = std::env::var(token_env).ok().filter(|t| !t.is_empty());
    Box::new(craft_ai::OpenAiCompat::new(endpoint, token, model))
}

#[cfg(feature = "local")]
fn local(
    id: &str,
    name: &str,
    gguf: &std::path::Path,
    tokenizer: &std::path::Path,
    arch: Option<craft_ai::Arch>,
    template: Option<craft_ai::Template>,
    device: craft_ai::DeviceChoice,
) -> Result<Box<dyn Provider>, String> {
    let spec = craft_ai::local::LoadSpec { id: id.into(), name: name.into(), gguf, tokenizer, arch, template, device };
    Ok(Box::new(craft_ai::local::LocalGguf::load_spec(&spec).map_err(|e| e.to_string())?))
}

#[cfg(not(feature = "local"))]
fn local(
    _: &str,
    _: &str,
    _: &std::path::Path,
    _: &std::path::Path,
    _: Option<craft_ai::Arch>,
    _: Option<craft_ai::Template>,
    _: craft_ai::DeviceChoice,
) -> Result<Box<dyn Provider>, String> {
    Err("this build has no local models (built without the `local` feature); use an endpoint".into())
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
