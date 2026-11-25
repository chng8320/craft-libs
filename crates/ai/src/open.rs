//! Opening a chat model from the two choices every caller has: a model id and a device. The same
//! rules for every app, the command line and the Dataverse framework:
//!
//! - the model: the id given, else `CRAFT_AI_DEFAULT_MODEL`, else qwen3-4b-q4km; looked up in the
//!   catalogue (built in, `<model folder>/catalog.json`, `CRAFT_AI_CATALOG`), else a `.gguf` path;
//! - the device: the one given, else the catalogue entry's, else `CRAFT_AI_DEVICE`, else auto;
//! - the model folder: `CRAFT_AI_MODELS`, else the suite's shared cache.
//!
//! Endpoint models need nothing local; downloaded and file models need the `local` feature, and
//! a build without it says so.

use std::path::{Path, PathBuf};

use crate::catalog::{Catalog, Source};
use crate::device::DeviceChoice;
use crate::error::AiError;
use crate::provider::Provider;
use crate::template::Template;

/// What to open.
#[derive(Clone, Debug, Default)]
pub struct ChatChoice {
    pub model: Option<String>,
    pub device: Option<DeviceChoice>,
    /// A prompt format to use instead of the catalogue's or the file's.
    pub template: Option<Template>,
    /// Another catalogue file to read.
    pub catalog: Option<PathBuf>,
    /// The model folder, instead of `CRAFT_AI_MODELS`.
    pub dir: Option<PathBuf>,
    /// A tokenizer for a `.gguf` path (default: tokenizer.json beside it).
    pub tokenizer: Option<PathBuf>,
}

/// The model id used when none is given.
pub fn default_model() -> String {
    std::env::var("CRAFT_AI_DEFAULT_MODEL").ok().filter(|m| !m.is_empty()).unwrap_or_else(|| "qwen3-4b-q4km".into())
}

/// The model folder: `CRAFT_AI_MODELS`, else the shared cache.
pub fn models_dir() -> Result<PathBuf, AiError> {
    if let Some(d) = std::env::var_os("CRAFT_AI_MODELS").filter(|d| !d.is_empty()) {
        return Ok(PathBuf::from(d));
    }
    crate::probe::models_dir().ok_or_else(|| AiError::Inference("no model folder: set CRAFT_AI_MODELS".into()))
}

/// The catalogue for `dir`, with `extra` read last.
pub fn catalog(dir: &Path, extra: Option<&Path>) -> Result<Catalog, AiError> {
    let mut files = crate::catalog::default_files(dir);
    if let Some(f) = extra {
        files.push(f.to_path_buf());
    }
    Catalog::load(&files)
}

fn env_device() -> Result<DeviceChoice, AiError> {
    std::env::var("CRAFT_AI_DEVICE").ok().filter(|d| !d.is_empty()).map_or(Ok(DeviceChoice::Auto), |d| d.parse().map_err(AiError::Inference))
}

/// Open the chat model `c` names.
pub fn open_chat(c: &ChatChoice) -> Result<Box<dyn Provider>, AiError> {
    let id = c.model.clone().filter(|m| !m.is_empty()).unwrap_or_else(default_model);
    let dir = match &c.dir {
        Some(d) => d.clone(),
        None => models_dir()?,
    };
    let device = |own: Option<DeviceChoice>| -> Result<DeviceChoice, AiError> { c.device.or(own).map_or_else(env_device, Ok) };
    let cat = catalog(&dir, c.catalog.as_deref())?;
    if let Some(m) = cat.find(&id) {
        let template = c.template.clone().or_else(|| m.template.clone());
        return match &m.source {
            Source::Endpoint { endpoint, model, token_env } => {
                let token = std::env::var(token_env).ok().filter(|t| !t.is_empty());
                Ok(Box::new(crate::OpenAiCompat::new(endpoint.clone(), token, model.clone())))
            }
            Source::Download(e) => {
                if !crate::fetch::is_downloaded(e, &dir) {
                    return Err(AiError::Inference(format!("{id} is not downloaded in {} (craft-ai-cli download {id})", dir.display())));
                }
                local(&m.id, &m.name, &crate::fetch::model_path(e, &dir), &crate::fetch::tokenizer_path(e, &dir), m.arch, template, device(m.device)?)
            }
            Source::File { gguf, tokenizer } => local(&m.id, &m.name, gguf, tokenizer, m.arch, template, device(m.device)?),
        };
    }
    let path = Path::new(&id);
    if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("gguf")) {
        let tokenizer = c.tokenizer.clone().unwrap_or_else(|| path.with_file_name("tokenizer.json"));
        let name = path.file_stem().map_or_else(|| id.clone(), |s| s.to_string_lossy().into_owned());
        return local(&id, &name, path, &tokenizer, None, c.template.clone(), device(None)?);
    }
    Err(AiError::Inference(format!("unknown model {id:?} (craft-ai-cli models lists them; add it to a catalogue, or give a .gguf file)")))
}

#[cfg(feature = "local")]
fn local(
    id: &str,
    name: &str,
    gguf: &Path,
    tokenizer: &Path,
    arch: Option<crate::Arch>,
    template: Option<Template>,
    device: DeviceChoice,
) -> Result<Box<dyn Provider>, AiError> {
    let spec = crate::local::LoadSpec { id: id.into(), name: name.into(), gguf, tokenizer, arch, template, device };
    Ok(Box::new(crate::local::LocalGguf::load_spec(&spec)?))
}

#[cfg(not(feature = "local"))]
fn local(_: &str, _: &str, _: &Path, _: &Path, _: Option<crate::Arch>, _: Option<Template>, _: DeviceChoice) -> Result<Box<dyn Provider>, AiError> {
    Err(AiError::Inference("this build runs no local models (built without the `local` feature); use an endpoint model".into()))
}
