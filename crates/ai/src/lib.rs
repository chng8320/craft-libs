//! craft-ai — AI provider abstraction for the Craft suite.
//!
//! See `README.md` for the design: BYO API provider (OpenAI-compatible) +
//! one-click local models with capability detection.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

mod device;
mod error;
mod provider;
mod registry;
pub mod tasks;
mod template;
mod translate;

#[cfg(not(target_arch = "wasm32"))]
pub mod catalog;
#[cfg(not(target_arch = "wasm32"))]
pub mod ocr;
#[cfg(not(target_arch = "wasm32"))]
pub mod open;
#[cfg(not(target_arch = "wasm32"))]
pub mod speech;

#[cfg(not(target_arch = "wasm32"))]
pub mod fetch;
#[cfg(not(target_arch = "wasm32"))]
#[cfg(feature = "local")]
pub mod local;
// `MachineCaps` is portable data the registry needs on every target; only
// the OS probe itself is native-only.
pub mod probe;

pub use device::{DeviceChoice, compiled_backends};
#[cfg(any(feature = "local", feature = "speech"))]
pub use device::{gpus, open as open_device};
pub use error::AiError;
#[cfg(not(target_arch = "wasm32"))]
pub use provider::OpenAiCompat;
pub use provider::{GenOpts, Provider, ProviderInfo};
pub use registry::{Arch, MODELS, ModelEntry, Quality, by_id, fits, suggest};
pub use template::Template;
pub use translate::{PdfGrammar, Segment, TranslateGrammar, is_protected_token, segment, translate_segments};
