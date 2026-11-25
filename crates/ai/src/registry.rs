//! The curated model registry — a hand-maintained table, compiled in.
//! Runtime-fetching this list would make our supply chain un-auditable.
//!
//! Curation rule: **permissive licenses only** (the suite is MIT OR
//! Apache-2.0 and commercial users run it). CC-BY-NC models like NLLB-200
//! are excluded by policy; their absence is deliberate.
//!
//! `sha256` may be `None`: Hugging Face serves the LFS digest in the
//! `X-Linked-ETag` header of the resolve redirect, which [`crate::fetch`]
//! uses as the expected checksum. We only hard-code a hash when we've
//! verified it ourselves — never a guessed one.

use crate::probe::MachineCaps;

/// Model quality tier — higher is better output, bigger download.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Quality {
    Tiny,
    Small,
    Medium,
    Large,
}

/// Which candle architecture loads the GGUF.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arch {
    Qwen2,
    Qwen3,
    Llama,
}

impl Arch {
    /// The architecture a GGUF's `general.architecture` names, when this
    /// crate can run it.
    pub fn from_gguf_name(name: &str) -> Option<Self> {
        match name {
            "qwen2" => Some(Self::Qwen2),
            "qwen3" => Some(Self::Qwen3),
            "llama" => Some(Self::Llama),
            _ => None,
        }
    }
}

/// One downloadable model.
#[derive(Clone, Copy, Debug)]
pub struct ModelEntry {
    /// Stable identifier stored in settings (`"qwen2.5-1.5b-q4km"`).
    pub id: &'static str,
    /// Display name for the picker.
    pub name: &'static str,
    /// Canonical download URL (Hugging Face `resolve` endpoint).
    pub url: &'static str,
    /// Filename inside the model cache.
    pub file: &'static str,
    /// Companion tokenizer (needed by local inference; the GGUF ships
    /// vocab only, not the chat-time tokenizer.json).
    pub tokenizer_url: &'static str,
    /// Verified SHA-256 of the GGUF, when we have one.
    pub sha256: Option<&'static str>,
    /// Download size, MiB — shown to the user before they commit.
    pub size_mb: u64,
    /// Below this RAM the model is refused outright.
    pub min_ram_gb: f32,
    /// Recommended RAM for comfortable use.
    pub rec_ram_gb: f32,
    /// SPDX license id, displayed in the UI.
    pub license: &'static str,
    pub quality: Quality,
    pub arch: Arch,
    /// Human-readable language coverage for the picker.
    pub langs: &'static str,
}

/// The registry, smallest first.
pub static MODELS: &[ModelEntry] = &[
    ModelEntry {
        id: "qwen2.5-0.5b-q4km",
        name: "Qwen2.5 0.5B Instruct (Q4_K_M)",
        url: "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        file: "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        tokenizer_url: "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer.json",
        sha256: None,
        size_mb: 389,
        min_ram_gb: 2.0,
        rec_ram_gb: 4.0,
        license: "Apache-2.0",
        quality: Quality::Tiny,
        arch: Arch::Qwen2,
        langs: "29+ languages incl. EN/ZH/FR/DE/ES/JA/KO/RU",
    },
    ModelEntry {
        id: "smollm2-1.7b-q4km",
        name: "SmolLM2 1.7B Instruct (Q4_K_M)",
        url: "https://huggingface.co/bartowski/SmolLM2-1.7B-Instruct-GGUF/resolve/main/SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
        file: "SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
        tokenizer_url: "https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/resolve/main/tokenizer.json",
        sha256: None,
        size_mb: 1033,
        min_ram_gb: 4.0,
        rec_ram_gb: 8.0,
        license: "Apache-2.0",
        quality: Quality::Small,
        arch: Arch::Llama,
        langs: "primarily EN/FR/DE/ES/IT",
    },
    ModelEntry {
        id: "qwen2.5-1.5b-q4km",
        name: "Qwen2.5 1.5B Instruct (Q4_K_M)",
        url: "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        file: "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        tokenizer_url: "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/resolve/main/tokenizer.json",
        sha256: None,
        size_mb: 986,
        min_ram_gb: 4.0,
        rec_ram_gb: 8.0,
        license: "Apache-2.0",
        quality: Quality::Small,
        arch: Arch::Qwen2,
        langs: "29+ languages incl. EN/ZH/FR/DE/ES/JA/KO/RU",
    },
    ModelEntry {
        id: "qwen3-1.7b-q8",
        name: "Qwen3 1.7B (Q8_0)",
        url: "https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q8_0.gguf",
        file: "Qwen3-1.7B-Q8_0.gguf",
        tokenizer_url: "https://huggingface.co/Qwen/Qwen3-1.7B/resolve/main/tokenizer.json",
        sha256: None,
        size_mb: 1749,
        min_ram_gb: 4.0,
        rec_ram_gb: 8.0,
        license: "Apache-2.0",
        quality: Quality::Small,
        arch: Arch::Qwen3,
        langs: "119 languages incl. EN/ZH/VI/TH/ID/JA/KO/FR/DE/ES",
    },
    ModelEntry {
        id: "qwen3-4b-q4km",
        name: "Qwen3 4B (Q4_K_M)",
        url: "https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf",
        file: "Qwen3-4B-Q4_K_M.gguf",
        tokenizer_url: "https://huggingface.co/Qwen/Qwen3-4B/resolve/main/tokenizer.json",
        sha256: None,
        size_mb: 2382,
        min_ram_gb: 6.0,
        rec_ram_gb: 12.0,
        license: "Apache-2.0",
        quality: Quality::Medium,
        arch: Arch::Qwen3,
        langs: "119 languages incl. EN/ZH/VI/TH/ID/JA/KO/FR/DE/ES",
    },
    ModelEntry {
        id: "qwen3-8b-q4km",
        name: "Qwen3 8B (Q4_K_M)",
        url: "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf",
        file: "Qwen3-8B-Q4_K_M.gguf",
        tokenizer_url: "https://huggingface.co/Qwen/Qwen3-8B/resolve/main/tokenizer.json",
        sha256: None,
        size_mb: 4795,
        min_ram_gb: 10.0,
        rec_ram_gb: 16.0,
        license: "Apache-2.0",
        quality: Quality::Large,
        arch: Arch::Qwen3,
        langs: "119 languages incl. EN/ZH/VI/TH/ID/JA/KO/FR/DE/ES",
    },
    // NOTE: Qwen2.5-3B was removed — its `qwen-research` license is
    // non-commercial and breaks the permissive-only rule above.
];

/// Look up a model by stable id.
pub fn by_id(id: &str) -> Option<&'static ModelEntry> {
    MODELS.iter().find(|m| m.id == id)
}

/// Whether `entry` can run at all on this machine.
pub fn fits(caps: &MachineCaps, entry: &ModelEntry) -> bool {
    let ram_ok = caps.ram_gb >= entry.min_ram_gb;
    let disk_ok = caps.disk_free_gb >= entry.size_mb as f32 / 1024.0 * 1.3; // file + headroom
    ram_ok && disk_ok
}

/// The default suggestion for `caps`: best quality that fits, preferring
/// 8B on roomy machines, 4B on good ones, 1.5B mid-range, 0.5B on small ones. Returns
/// `None` when nothing fits — the UI then says so honestly instead of
/// offering a download that can't run.
pub fn suggest(caps: &MachineCaps) -> Option<&'static ModelEntry> {
    let capable: Vec<&ModelEntry> = MODELS.iter().filter(|m| fits(caps, m)).collect();
    let best_allowed = if caps.vram_gb.is_some_and(|v| v >= 10.0) || caps.ram_gb >= 32.0 {
        Quality::Large
    } else if caps.vram_gb.is_some_and(|v| v >= 6.0) || caps.ram_gb >= 16.0 {
        Quality::Medium
    } else if caps.ram_gb >= 8.0 {
        Quality::Small
    } else {
        Quality::Tiny
    };
    capable.into_iter().filter(|m| m.quality <= best_allowed).max_by_key(|m| m.quality)
}
