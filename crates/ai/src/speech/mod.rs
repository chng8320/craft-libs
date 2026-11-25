//! Speech to text, for every app: FilmCraft's captions and transcripts, SoundCraft, meeting
//! minutes from the document tools, live captions. Moved here from FilmCraft (`crates/speech`)
//! so that every engine uses one implementation; the behaviour, limits and tests came with it.
//!
//! - [`Transcriber`]: the trait every speech recogniser implements. It takes mono 16 kHz samples
//!   and returns a [`Transcript`] whose words carry times in 16 kHz samples ([`Samples`]), which
//!   each app maps to its own time base exactly.
//! - [`models`]: the catalogue of downloadable Whisper models (source URL pinned to a revision,
//!   SHA-256, size, licence), and the verified downloader. Weights are **never** bundled or
//!   committed.
//! - [`catalog`]: further speech models from the model catalogue files (`task:
//!   "speech-to-text"`): Whisper folders and OpenAI-compatible endpoints.
//! - [`whisper`] (feature `speech`): Whisper inference in pure Rust on candle, on the CPU or a
//!   GPU ([`crate::DeviceChoice`]), with timestamp decoding, language detection and word-level
//!   timestamps from cross-attention alignment (DTW).
//! - [`remote`]: any OpenAI-compatible `/audio/transcriptions` server.
//! - [`diarize`]: speaker labelling by clustering per-chunk MFCC statistics (classical, no model).
//! - [`mel`]: the log-mel front end shared by Whisper and diarization.
//! - [`vad`]: energy-based tightening of word bounds (keeps pauses out of words).

pub mod catalog;
pub mod diarize;
pub mod live;
pub mod mel;
pub mod models;
pub mod remote;
pub mod transcript;
pub mod vad;
#[cfg(feature = "speech")]
pub mod whisper;

pub use transcript::{Samples, Speaker, Transcript, Word};

use crate::DeviceChoice;

/// The sample rate every transcriber takes.
pub const SAMPLE_RATE: u32 = 16_000;

/// The time of a 16 kHz sample index.
pub fn at_sample(n: i64) -> Samples {
    Samples(n)
}

/// The time of `s` seconds (rounded to the nearest 16 kHz sample).
pub fn at_seconds(s: f64) -> Samples {
    at_sample((s * f64::from(SAMPLE_RATE)).round() as i64)
}

#[derive(Debug, Clone, PartialEq)]
pub enum SpeechError {
    Cancelled,
    NotInstalled(String),
    UnknownModel(String),
    Unavailable(String),
    Model(String),
    Download(String),
    Io(String),
}

impl std::fmt::Display for SpeechError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cancelled => f.write_str("stopped"),
            Self::NotInstalled(m) => write!(f, "the speech model `{m}` is not downloaded"),
            Self::UnknownModel(m) => write!(f, "unknown speech model `{m}`"),
            Self::Unavailable(m) => write!(f, "speech-to-text is not available in this build ({m})"),
            Self::Model(m) => write!(f, "model: {m}"),
            Self::Download(m) => write!(f, "download: {m}"),
            Self::Io(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for SpeechError {}

impl From<std::io::Error> for SpeechError {
    fn from(e: std::io::Error) -> Self {
        SpeechError::Io(e.to_string())
    }
}

/// Transcription options.
#[derive(Clone, Debug, PartialEq)]
pub struct Options {
    /// ISO 639-1 language code; `None` = detect from the first 30 s.
    pub language: Option<String>,
    /// Label speakers ([`diarize`]); otherwise every word gets speaker 0.
    pub diarize: bool,
    /// Upper bound for the number of speakers found.
    pub max_speakers: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self { language: None, diarize: true, max_speakers: 6 }
    }
}

/// Progress callback: `(fraction 0..1, status)`; return `false` to cancel.
pub type ProgressFn<'a> = &'a mut dyn FnMut(f32, &str) -> bool;

/// A speech recogniser.
pub trait Transcriber: Send + Sync {
    /// Stable id of the model (`"whisper-base"`), stored in the transcript's `source`.
    fn id(&self) -> String;
    /// Transcribe mono 16 kHz `audio`. Word times are relative to the first sample.
    fn transcribe(&self, audio: &[f32], opts: &Options, progress: ProgressFn) -> Result<Transcript, SpeechError>;
    /// What the model runs on (`"CPU"`, `"NVIDIA GPU 0 (CUDA)"`), for results and the status line.
    fn device(&self) -> String {
        "CPU".into()
    }
}

/// A transcriber that returns a fixed transcript (tests, demos and agents that bring their own
/// transcript); words beyond the audio's length are dropped.
pub struct FixedTranscriber {
    pub transcript: Transcript,
    pub id: String,
}

impl Transcriber for FixedTranscriber {
    fn id(&self) -> String {
        self.id.clone()
    }
    fn transcribe(&self, audio: &[f32], _: &Options, progress: ProgressFn) -> Result<Transcript, SpeechError> {
        if !progress(0.5, "Transcribing") {
            return Err(SpeechError::Cancelled);
        }
        let end = at_sample(audio.len() as i64);
        let mut t = self.transcript.clone();
        t.words.retain(|w| w.start < end);
        t.source = self.id.clone();
        progress(1.0, "Done");
        Ok(t)
    }
}

/// The processor `CRAFT_AI_DEVICE` names, else auto.
pub fn device_from_env() -> DeviceChoice {
    std::env::var("CRAFT_AI_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or_default()
}

/// Whether `id` is a speech model here: built in, or in a catalogue file.
pub fn known(models_dir: &std::path::Path, id: &str) -> bool {
    models::find(id).is_some() || matches!(catalog::find(models_dir, id), Ok(Some(_)))
}

/// Load speech model `id`: a catalogue file's entry (a Whisper folder or an endpoint) when one
/// lists it, else the built-in model from `models_dir`. The processor: `device`, else the
/// catalogue entry's, else `CRAFT_AI_DEVICE`, else auto.
pub fn load(models_dir: &std::path::Path, id: &str, device: Option<DeviceChoice>) -> Result<std::sync::Arc<dyn Transcriber>, SpeechError> {
    let Some(m) = catalog::find(models_dir, id)? else {
        return load_builtin(models_dir, id, device.unwrap_or_else(device_from_env));
    };
    let device = device.or(m.device).unwrap_or_else(device_from_env);
    match m.source {
        catalog::Source::Folder(dir) => load_folder(&dir, &m.id, device),
        catalog::Source::Endpoint { endpoint, model, token_env } => {
            let token = std::env::var(&token_env).ok().filter(|t| !t.is_empty());
            Ok(std::sync::Arc::new(remote::Remote { id: m.id, endpoint, model, token }))
        }
    }
}

/// A built-in model, downloaded into `models_dir`, on the processor `device` names.
pub fn load_builtin(models_dir: &std::path::Path, id: &str, device: DeviceChoice) -> Result<std::sync::Arc<dyn Transcriber>, SpeechError> {
    let m = models::find(id).ok_or_else(|| SpeechError::UnknownModel(id.into()))?;
    if !models::installed(models_dir, m) {
        return Err(SpeechError::NotInstalled(id.into()));
    }
    load_folder(&models::model_dir(models_dir, m), m.id, device)
}

fn load_folder(dir: &std::path::Path, id: &str, device: DeviceChoice) -> Result<std::sync::Arc<dyn Transcriber>, SpeechError> {
    #[cfg(feature = "speech")]
    {
        Ok(std::sync::Arc::new(whisper::Whisper::load_on(dir, id, device)?))
    }
    #[cfg(not(feature = "speech"))]
    {
        let _ = (dir, id, device);
        Err(SpeechError::Unavailable("built without the `speech` feature".into()))
    }
}

/// Whether this build can run speech models.
pub const fn available() -> bool {
    cfg!(feature = "speech")
}

/// Mix a multi-channel buffer down to mono.
pub fn downmix(channels: &[Vec<f32>]) -> Vec<f32> {
    let n = channels.iter().map(Vec::len).min().unwrap_or(0);
    if channels.len() == 1 {
        return channels[0][..n].to_vec();
    }
    let k = 1.0 / channels.len().max(1) as f32;
    (0..n).map(|i| channels.iter().map(|c| c[i]).sum::<f32>() * k).collect()
}

/// Word error rate between a reference and a hypothesis (words compared case-insensitively
/// without punctuation, hyphenated words split): (substitutions + deletions + insertions) / reference words.
pub fn word_error_rate(reference: &str, hypothesis: &str) -> f64 {
    let norm = |s: &str| -> Vec<String> {
        s.split(|c: char| c.is_whitespace() || c == '-').map(transcript::normalize_word).filter(|w| !w.is_empty()).collect()
    };
    let (r, h) = (norm(reference), norm(hypothesis));
    if r.is_empty() {
        return if h.is_empty() { 0.0 } else { 1.0 };
    }
    let mut prev: Vec<usize> = (0..=h.len()).collect();
    for i in 1..=r.len() {
        let mut cur = vec![i; h.len() + 1];
        for j in 1..=h.len() {
            let sub = prev[j - 1] + usize::from(r[i - 1] != h[j - 1]);
            cur[j] = sub.min(prev[j] + 1).min(cur[j - 1] + 1);
        }
        prev = cur;
    }
    prev[h.len()] as f64 / r.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn second_times_round_to_the_nearest_sample() {
        assert_eq!(at_seconds(1.5), Samples(24_000));
        assert_eq!(at_seconds(1.5).seconds(), 1.5);
    }

    #[test]
    fn wer_counts_edits() {
        assert_eq!(word_error_rate("the cat sat", "The cat, sat."), 0.0);
        assert!((word_error_rate("the cat sat on the mat", "the cat sat on mat") - 1.0 / 6.0).abs() < 1e-9);
        assert!((word_error_rate("a b", "a x b y") - 1.0).abs() < 1e-9);
    }

    #[test]
    fn fixed_transcriber_trims_and_cancels() {
        let mut t = Transcript::default();
        t.words.push(Word::new("hi", at_seconds(0.1), at_seconds(0.3)));
        t.words.push(Word::new("late", at_seconds(5.0), at_seconds(5.5)));
        let f = FixedTranscriber { transcript: t, id: "fixed".into() };
        let out = f.transcribe(&vec![0.0; 16_000], &Options::default(), &mut |_, _| true).unwrap();
        assert_eq!(out.words.len(), 1);
        assert_eq!(out.source, "fixed");
        assert_eq!(f.transcribe(&[], &Options::default(), &mut |_, _| false), Err(SpeechError::Cancelled));
    }

    #[test]
    fn downmix_averages() {
        assert_eq!(downmix(&[vec![1.0, 0.0], vec![0.0, 0.0, 9.0]]), vec![0.5, 0.0]);
    }
}
