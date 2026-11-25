//! The model catalogue: the built-in models ([`crate::MODELS`]) plus any the operator lists in
//! catalogue files, so a new model is a few lines of JSON, not a new build. A model is one id
//! whichever way it runs.
//!
//! One file serves every engine: each entry has a `task` (`chat`, the default, is this crate's;
//! `speech-to-text` is FilmCraft's, and so on), and an engine reads only its own tasks. The
//! format is described in the Dataverse SDK's `sdk/models.md`.
//!
//! ```json
//! { "models": [
//!   { "id": "gemma3-12b", "url": "https://…/gemma-3-12b-it-Q4_K_M.gguf", "tokenizer_url": "https://…/tokenizer.json",
//!     "arch": "gemma3", "size_mb": 7000, "device": "gpu" },
//!   { "id": "my-finetune", "path": "models/my-finetune.gguf", "tokenizer": "models/tokenizer.json",
//!     "template": "<s>[SYS]{system}[/SYS]{prompt}[ANS]", "stop": ["</s>"] },
//!   { "id": "big", "endpoint": "http://gpu-box:8000/v1", "remote_model": "Qwen/Qwen3-32B", "token_env": "BIG_TOKEN" }
//! ] }
//! ```
//!
//! - **Download** (`url`): fetched into the model folder like a built-in model (`download`).
//! - **File** (`path`): a GGUF already on disk; relative paths are relative to the catalogue file.
//! - **Endpoint** (`endpoint`): any OpenAI-compatible server (vLLM, Ollama, llama.cpp, TGI, a
//!   cloud API) — whatever model it serves, on whatever hardware it has. The key is read from the
//!   environment variable `token_env` names (default `CRAFT_AI_TOKEN`), never from the file.
//!
//! Local models (download and file) run on candle; `arch` is read from the GGUF when not given,
//! and `template` (a name, or a text with `{system}` and `{prompt}` plus `stop`) from the GGUF's
//! own chat template when not given. `device` is the model's default processor.
//!
//! Files are read in order and a later entry replaces an earlier one with the same id, so a
//! catalogue can also change a built-in model's default device.

use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::device::DeviceChoice;
use crate::error::AiError;
use crate::registry::{Arch, MODELS, ModelEntry, Quality};
use crate::template::Template;

/// How a catalogue model runs.
#[derive(Clone, Debug)]
pub enum Source {
    /// Downloaded into the model folder (built in, or a catalogue `url`).
    Download(&'static ModelEntry),
    /// A GGUF and tokenizer already on disk.
    File { gguf: PathBuf, tokenizer: PathBuf },
    /// An OpenAI-compatible endpoint.
    Endpoint { endpoint: String, model: String, token_env: String },
}

/// One model the catalogue knows.
#[derive(Clone, Debug)]
pub struct CatalogModel {
    pub id: String,
    pub name: String,
    pub source: Source,
    /// `None`: read from the GGUF.
    pub arch: Option<Arch>,
    /// `None`: the GGUF's own chat template, else the architecture's usual one.
    pub template: Option<Template>,
    /// The model's default processor, when the catalogue sets one.
    pub device: Option<DeviceChoice>,
    /// Where the entry came from: `"built in"` or the catalogue file.
    pub origin: String,
}

impl CatalogModel {
    /// `"download"`, `"file"` or `"endpoint"`.
    pub fn kind(&self) -> &'static str {
        match self.source {
            Source::Download(_) => "download",
            Source::File { .. } => "file",
            Source::Endpoint { .. } => "endpoint",
        }
    }
}

#[derive(Deserialize)]
struct FileCatalog {
    models: Vec<serde_json::Value>,
}

/// The task this crate runs.
pub const TASK: &str = "chat";

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields, default)]
struct Entry {
    id: String,
    name: Option<String>,
    // download
    url: Option<String>,
    file: Option<String>,
    tokenizer_url: Option<String>,
    sha256: Option<String>,
    size_mb: Option<u64>,
    min_ram_gb: Option<f32>,
    rec_ram_gb: Option<f32>,
    license: Option<String>,
    languages: Option<String>,
    quality: Option<String>,
    // file
    path: Option<String>,
    tokenizer: Option<String>,
    // endpoint
    endpoint: Option<String>,
    remote_model: Option<String>,
    token_env: Option<String>,
    // every local model
    arch: Option<String>,
    template: Option<String>,
    stop: Vec<String>,
    device: Option<String>,
}

/// Every model: built-in ones first, then the catalogue files' entries.
#[derive(Clone, Debug, Default)]
pub struct Catalog {
    pub models: Vec<CatalogModel>,
}

/// Catalogue strings live for the whole run (the download code takes `&'static` entries); a
/// catalogue is read once per process, so this is bounded.
fn leak(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

impl Catalog {
    /// The built-in models only.
    pub fn builtin() -> Self {
        Self {
            models: MODELS
                .iter()
                .map(|m| CatalogModel {
                    id: m.id.into(),
                    name: m.name.into(),
                    source: Source::Download(m),
                    arch: Some(m.arch),
                    template: Some(m.arch.default_template()),
                    device: None,
                    origin: "built in".into(),
                })
                .collect(),
        }
    }

    /// The built-in models plus the entries of `files`, in order.
    pub fn load(files: &[PathBuf]) -> Result<Self, AiError> {
        let mut c = Self::builtin();
        for f in files {
            c.add_file(f)?;
        }
        Ok(c)
    }

    /// Add one catalogue file's entries (a later id replaces an earlier one).
    pub fn add_file(&mut self, file: &Path) -> Result<(), AiError> {
        let text = std::fs::read_to_string(file).map_err(AiError::Io)?;
        let parsed: FileCatalog = serde_json::from_str(&text).map_err(|e| AiError::Inference(format!("{}: {e}", file.display())))?;
        let base = file.parent().unwrap_or_else(|| Path::new("."));
        for mut v in parsed.models {
            // Another engine's model (speech, voices…): not ours to check.
            if v.get("task").and_then(|t| t.as_str()).is_some_and(|t| t != TASK) {
                continue;
            }
            if let Some(o) = v.as_object_mut() {
                o.remove("task");
            }
            let e: Entry = serde_json::from_value(v).map_err(|e| AiError::Inference(format!("{}: {e}", file.display())))?;
            let m = entry(e, base, file).map_err(|e| AiError::Inference(format!("{}: {e}", file.display())))?;
            self.models.retain(|x| x.id != m.id);
            self.models.push(m);
        }
        Ok(())
    }

    pub fn find(&self, id: &str) -> Option<&CatalogModel> {
        self.models.iter().find(|m| m.id == id)
    }
}

fn entry(e: Entry, base: &Path, file: &Path) -> Result<CatalogModel, String> {
    if e.id.trim().is_empty() {
        return Err("a model without an id".into());
    }
    let id = e.id.clone();
    let at = |what: &str| format!("model {id:?}: {what}");
    let kinds = [e.url.is_some(), e.path.is_some(), e.endpoint.is_some()].iter().filter(|k| **k).count();
    if kinds != 1 {
        return Err(at("give exactly one of url (download), path (a file on disk) or endpoint"));
    }
    let arch = match &e.arch {
        Some(a) => Some(Arch::from_gguf_name(&a.to_ascii_lowercase()).ok_or_else(|| at(&format!("arch {a:?}: one of {}", Arch::NAMES)))?),
        None => None,
    };
    let template = match &e.template {
        Some(t) => Some(Template::from_parts(t, &e.stop).map_err(|m| at(&m))?),
        None if !e.stop.is_empty() => return Err(at("stop needs a template")),
        None => None,
    };
    let device = e.device.as_deref().map(str::parse::<DeviceChoice>).transpose().map_err(|m| at(&m))?;
    let name = e.name.clone().unwrap_or_else(|| id.clone());
    let resolve = |p: &str| if Path::new(p).is_absolute() { PathBuf::from(p) } else { base.join(p) };
    let source = if let Some(url) = e.url {
        let file_name =
            e.file.or_else(|| url.rsplit('/').next().map(str::to_string)).filter(|f| !f.is_empty() && !f.contains(['/', '\\']) && f != "..");
        let file_name = file_name.ok_or_else(|| at("no file name for the download (give `file`)"))?;
        let tokenizer_url = e.tokenizer_url.ok_or_else(|| at("a download needs tokenizer_url (the model's tokenizer.json)"))?;
        let quality = match e.quality.as_deref().map(str::to_ascii_lowercase).as_deref() {
            None | Some("small") => Quality::Small,
            Some("tiny") => Quality::Tiny,
            Some("medium") => Quality::Medium,
            Some("large") => Quality::Large,
            Some(q) => return Err(at(&format!("quality {q:?}: tiny, small, medium or large"))),
        };
        Source::Download(Box::leak(Box::new(ModelEntry {
            id: leak(id.clone()),
            name: leak(name.clone()),
            url: leak(url),
            file: leak(file_name),
            tokenizer_url: leak(tokenizer_url),
            sha256: e.sha256.map(leak),
            size_mb: e.size_mb.unwrap_or(0),
            min_ram_gb: e.min_ram_gb.unwrap_or(0.0),
            rec_ram_gb: e.rec_ram_gb.unwrap_or(0.0),
            license: leak(e.license.unwrap_or_else(|| "see the model's page".into())),
            quality,
            // Only the download code reads this entry; loading goes by the catalogue's `arch`
            // (None: read from the GGUF).
            arch: arch.unwrap_or(Arch::Llama),
            langs: leak(e.languages.unwrap_or_default()),
        })))
    } else if let Some(p) = e.path {
        let gguf = resolve(&p);
        let tokenizer = e.tokenizer.as_deref().map_or_else(|| gguf.with_file_name("tokenizer.json"), resolve);
        Source::File { gguf, tokenizer }
    } else {
        let endpoint = e.endpoint.unwrap_or_default();
        let model = e.remote_model.ok_or_else(|| at("an endpoint needs remote_model (the name the server knows the model by)"))?;
        Source::Endpoint { endpoint, model, token_env: e.token_env.unwrap_or_else(|| "CRAFT_AI_TOKEN".into()) }
    };
    Ok(CatalogModel { id, name, source, arch, template, device, origin: file.display().to_string() })
}

/// The catalogue files to read: `<models dir>/catalog.json` when it exists, then each path in
/// `CRAFT_AI_CATALOG` (a path list, `:`-separated, `;` on Windows).
pub fn default_files(models_dir: &Path) -> Vec<PathBuf> {
    let mut v = Vec::new();
    let own = models_dir.join("catalog.json");
    if own.is_file() {
        v.push(own);
    }
    if let Some(list) = std::env::var_os("CRAFT_AI_CATALOG") {
        v.extend(std::env::split_paths(&list).filter(|p| !p.as_os_str().is_empty()));
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_catalogue_adds_every_kind_and_overrides_a_builtin_by_id() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("catalog.json");
        std::fs::write(
            &f,
            r#"{"models": [
              {"id": "g", "url": "https://x/m/gemma.gguf", "tokenizer_url": "https://x/m/tokenizer.json", "arch": "gemma3", "device": "gpu"},
              {"id": "f", "path": "sub/f.gguf", "template": "{system}|{prompt}", "stop": ["</s>"]},
              {"id": "e", "endpoint": "http://box:8000/v1", "remote_model": "Big"},
              {"id": "qwen3-4b-q4km", "endpoint": "http://box:8000/v1", "remote_model": "Qwen3-4B"},
              {"id": "w", "task": "speech-to-text", "dir": "whisper-x", "multilingual": true}
            ]}"#,
        )
        .unwrap();
        let c = Catalog::load(&[f]).unwrap();
        let g = c.find("g").unwrap();
        assert!(matches!(g.source, Source::Download(e) if e.file == "gemma.gguf"));
        assert_eq!((g.arch, g.device), (Some(Arch::Gemma3), Some(DeviceChoice::Gpu)));
        assert!(
            matches!(&c.find("f").unwrap().source, Source::File { gguf, tokenizer } if gguf.starts_with(dir.path()) && tokenizer.ends_with("sub/tokenizer.json"))
        );
        assert!(matches!(&c.find("e").unwrap().source, Source::Endpoint { token_env, .. } if token_env == "CRAFT_AI_TOKEN"));
        assert_eq!(c.find("qwen3-4b-q4km").unwrap().kind(), "endpoint", "a catalogue entry replaces the built-in one");
        assert_eq!(c.models.iter().filter(|m| m.id == "qwen3-4b-q4km").count(), 1);
        assert!(c.find("w").is_none(), "another task's entry is left to its engine");
    }

    #[test]
    fn bad_entries_are_refused_with_the_model_named() {
        let dir = tempfile::tempdir().unwrap();
        for bad in [
            r#"{"models": [{"id": "x"}]}"#,
            r#"{"models": [{"id": "x", "path": "a.gguf", "endpoint": "http://e"}]}"#,
            r#"{"models": [{"id": "x", "path": "a.gguf", "arch": "bert"}]}"#,
            r#"{"models": [{"id": "x", "endpoint": "http://e"}]}"#,
            r#"{"models": [{"id": "x", "path": "a.gguf", "colour": "red"}]}"#,
        ] {
            let f = dir.path().join("c.json");
            std::fs::write(&f, bad).unwrap();
            assert!(Catalog::load(&[f]).is_err(), "{bad}");
        }
    }
}
