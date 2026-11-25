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
//! ```
//!
//! Models live in `--dir`, else `CRAFT_AI_MODELS`, else the suite's shared cache. A token for an
//! endpoint is read from the environment variable `--token-env` names (default
//! `CRAFT_AI_TOKEN`), never from the command line, so it doesn't show in process lists.

use std::path::PathBuf;
use std::process::ExitCode;

use craft_ai::{GenOpts, MODELS, PdfGrammar, Provider, Segment, by_id, fits, segment, suggest, translate_segments};
use serde_json::{Value, json};

const HELP: &str = "\
craft-ai-cli models   [--dir DIR]
craft-ai-cli download <model id> [--dir DIR]
craft-ai-cli complete  <model> (--prompt TEXT | --prompt-file FILE) [--max-tokens N] [--temperature T]
craft-ai-cli translate <model> --to LANGUAGE (--text TEXT | --file FILE | --segments-file FILE.json)
  <model>: --model ID [--dir DIR]   a local model from `models` (downloaded first)
       or: --endpoint URL --remote-model NAME [--token-env VAR]   any OpenAI-compatible API
  --segments-file: a JSON array of strings; the answer keeps their order and count, and
  protected runs (URLs, e-mail addresses, {placeholders}, <markup>, numbers) come back unchanged.
  Each text goes to the model whole, so sentences around a placeholder are translated as one; a
  translation that loses or adds a protected run is redone split at them. --split always splits.";

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
        Some("download") => download(&args),
        Some("complete") => complete(&args),
        Some("translate") => translate(&args),
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

fn models(a: &Args) -> Result<Value, String> {
    let dir = a.dir()?;
    let caps = craft_ai::probe::probe(&dir);
    let suggested = suggest(&caps).map(|m| m.id);
    let list: Vec<Value> = MODELS
        .iter()
        .map(|m| {
            json!({
                "id": m.id, "name": m.name, "size_mb": m.size_mb, "min_ram_gb": m.min_ram_gb, "rec_ram_gb": m.rec_ram_gb,
                "license": m.license, "languages": m.langs, "quality": format!("{:?}", m.quality),
                "downloaded": craft_ai::fetch::is_downloaded(m, &dir), "fits": fits(&caps, m), "suggested": suggested == Some(m.id),
            })
        })
        .collect();
    Ok(
        json!({ "dir": dir, "machine": { "ram_gb": caps.ram_gb, "threads": caps.threads, "avx2": caps.avx2, "disk_free_gb": caps.disk_free_gb }, "models": list }),
    )
}

fn download(a: &Args) -> Result<Value, String> {
    let id = a.0.get(1).filter(|s| !s.starts_with("--")).ok_or("download: which model? (see `models`)")?;
    let entry = by_id(id).ok_or_else(|| format!("unknown model {id:?} (see `models`)"))?;
    let dir = a.dir()?;
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

/// The provider the arguments name: a remote endpoint, or a downloaded local model.
fn provider(a: &Args) -> Result<Box<dyn Provider>, String> {
    if let Some(endpoint) = a.flag("--endpoint") {
        let model = a.flag("--remote-model").ok_or("--endpoint needs --remote-model")?;
        let var = a.flag("--token-env").unwrap_or("CRAFT_AI_TOKEN");
        let token = std::env::var(var).ok().filter(|t| !t.is_empty());
        return Ok(Box::new(craft_ai::OpenAiCompat::new(endpoint, token, model)));
    }
    let id = a.flag("--model").ok_or("say which model: --model ID, or --endpoint URL --remote-model NAME")?;
    local(id, &a.dir()?)
}

#[cfg(feature = "local")]
fn local(id: &str, dir: &std::path::Path) -> Result<Box<dyn Provider>, String> {
    let entry = by_id(id).ok_or_else(|| format!("unknown model {id:?} (see `models`)"))?;
    if !craft_ai::fetch::is_downloaded(entry, dir) {
        return Err(format!("{id} is not downloaded in {} (craft-ai-cli download {id})", dir.display()));
    }
    let model = craft_ai::local::LocalGguf::load(entry, &craft_ai::fetch::model_path(entry, dir), &craft_ai::fetch::tokenizer_path(entry, dir))
        .map_err(|e| e.to_string())?;
    Ok(Box::new(model))
}

#[cfg(not(feature = "local"))]
fn local(_: &str, _: &std::path::Path) -> Result<Box<dyn Provider>, String> {
    Err("this build has no local models (built without the `local` feature); use --endpoint".into())
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
