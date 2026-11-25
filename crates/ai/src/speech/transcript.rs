//! What a transcriber returns: timed words and speaker labels, in times every app can map to its
//! own (16 kHz samples from the start of the audio, exact).

use serde::{Deserialize, Serialize};

/// A time: 16 kHz samples from the first sample of the audio.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Samples(pub i64);

impl std::ops::Sub for Samples {
    type Output = Samples;
    fn sub(self, o: Samples) -> Samples {
        Samples(self.0.saturating_sub(o.0))
    }
}

impl std::ops::Add for Samples {
    type Output = Samples;
    fn add(self, o: Samples) -> Samples {
        Samples(self.0.saturating_add(o.0))
    }
}

impl Samples {
    /// Seconds from the start.
    pub fn seconds(self) -> f64 {
        self.0 as f64 / f64::from(super::SAMPLE_RATE)
    }
}

/// One recognised word.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Word {
    /// The word as written, with attached punctuation (`"Hello,"`). No surrounding spaces.
    pub text: String,
    /// Its first sample.
    pub start: Samples,
    /// Just after its last sample (exclusive).
    pub end: Samples,
    /// Index into [`Transcript::speakers`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker: Option<u32>,
    /// Recogniser confidence 0..1.
    pub confidence: f32,
}

impl Word {
    pub fn new(text: impl Into<String>, start: Samples, end: Samples) -> Self {
        Self { text: text.into(), start, end, speaker: None, confidence: 1.0 }
    }
}

/// A speaker label.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Speaker {
    pub name: String,
}

/// Distinct speakers one transcript can label; [`Transcript::normalize`] drops larger indices.
pub const MAX_SPEAKERS: u32 = 1024;

/// The words of one audio clip.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Transcript {
    /// ISO 639-1 code of the spoken language (`"en"`).
    pub language: String,
    /// What produced it: a model id (`"whisper-base"`).
    pub source: String,
    pub speakers: Vec<Speaker>,
    /// Words in time order, non-overlapping.
    pub words: Vec<Word>,
}

impl Transcript {
    /// Trim words, drop empty ones, sort by time, end each word no later than the next starts,
    /// and give every speaker index used a label.
    pub fn normalize(&mut self) {
        self.words.retain(|w| !w.text.trim().is_empty());
        for w in &mut self.words {
            w.text = w.text.trim().to_string();
            if w.end < w.start {
                w.end = w.start;
            }
            w.confidence = w.confidence.clamp(0.0, 1.0);
        }
        self.words.sort_by_key(|w| (w.start, w.end));
        for i in 1..self.words.len() {
            let s = self.words[i].start;
            if self.words[i - 1].end > s {
                self.words[i - 1].end = s.max(self.words[i - 1].start);
            }
        }
        // The largest index sizes the label table, so a single hostile word must not make it
        // allocate billions of labels: indices past the ceiling lose their speaker.
        for w in &mut self.words {
            if w.speaker.is_some_and(|s| s >= MAX_SPEAKERS) {
                w.speaker = None;
            }
        }
        if let Some(m) = self.words.iter().filter_map(|w| w.speaker).max() {
            while self.speakers.len() <= m as usize {
                let n = self.speakers.len() + 1;
                self.speakers.push(Speaker { name: format!("Speaker {n}") });
            }
        }
    }

    /// Validate the invariants [`Transcript::normalize`] establishes.
    pub fn check(&self) -> Result<(), String> {
        for (i, w) in self.words.iter().enumerate() {
            if w.end < w.start {
                return Err(format!("word {i} ({:?}) ends before it starts", w.text));
            }
            if i > 0 && self.words[i - 1].end > w.start {
                return Err(format!("word {i} ({:?}) overlaps the word before it", w.text));
            }
            if let Some(s) = w.speaker
                && s as usize >= self.speakers.len()
            {
                return Err(format!("word {i} has speaker {s}, but there are {} speakers", self.speakers.len()));
            }
        }
        Ok(())
    }

    /// The words joined with spaces.
    pub fn text(&self) -> String {
        self.words.iter().map(|w| w.text.as_str()).collect::<Vec<_>>().join(" ")
    }
}

/// A word reduced for comparison: no surrounding punctuation, lower case.
pub fn normalize_word(s: &str) -> String {
    s.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase()
}
