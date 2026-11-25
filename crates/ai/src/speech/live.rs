//! Live speech to text: a stream in (RTMP, SRT, RTSP, HLS, a device or a file, through
//! craft-live), captions out as they are spoken, and each caption translated into the languages
//! asked for. Any speech model and any chat model of the catalogue, each on its own device.
//!
//! The audio is cut into utterances at pauses ([`Segmenter`]): an utterance ends after
//! `pause` seconds of quiet, or at `max_segment` seconds, and is then transcribed whole, so a
//! caption is a sentence or a phrase, not a fixed slice. Reading runs on its own thread, so a
//! slow model never makes the input fall behind; utterances wait in order. Every caption goes
//! to every output that takes its language ([`Langs`]).

use std::io::{Read, Write};
use std::sync::mpsc;

use super::{Options, SAMPLE_RATE, Transcriber};
use crate::provider::Provider;
use crate::translate::{Segment, translate_segments};

/// 30 ms frames at 16 kHz.
const FRAME: usize = 480;

/// Cuts a running stream into utterances at pauses.
pub struct Segmenter {
    pause_frames: usize,
    max_samples: usize,
    pre_roll: usize,
    min_speech_frames: usize,
    floor: Option<f32>,
    /// Samples since the stream began, up to the end of `buf`.
    at: i64,
    buf: Vec<f32>,
    /// Where `buf` starts in the stream.
    start: i64,
    in_utterance: bool,
    quiet: usize,
    speech: usize,
    pending: Vec<f32>,
}

impl Segmenter {
    /// Utterances end after `pause` seconds of quiet or at `max_segment` seconds.
    pub fn new(pause: f32, max_segment: f32) -> Self {
        let frames = |s: f32| ((s.max(0.03) * SAMPLE_RATE as f32) as usize / FRAME).max(1);
        Self {
            pause_frames: frames(pause),
            max_samples: (max_segment.max(1.0) * SAMPLE_RATE as f32) as usize,
            pre_roll: frames(0.2) * FRAME,
            min_speech_frames: frames(0.25),
            floor: None,
            at: 0,
            buf: Vec::new(),
            start: 0,
            in_utterance: false,
            quiet: 0,
            speech: 0,
            pending: Vec::new(),
        }
    }

    fn is_speech(&mut self, frame: &[f32]) -> bool {
        let rms = (frame.iter().map(|x| x * x).sum::<f32>() / frame.len().max(1) as f32).sqrt();
        // The noise floor follows the quiet parts down quickly and rises only slowly, so a
        // stream that starts mid-sentence or talks for minutes still has speech above it.
        let floor = self.floor.unwrap_or(0.001);
        let floor = if rms < floor { floor * 0.9 + rms * 0.1 } else { (floor * 1.001).min(0.05) };
        self.floor = Some(floor);
        rms > (floor * 3.0).max(0.004)
    }

    /// Feed samples; returns the utterances that ended, as (start sample in the stream, samples).
    pub fn push(&mut self, samples: &[f32]) -> Vec<(i64, Vec<f32>)> {
        self.pending.extend_from_slice(samples);
        let mut out = Vec::new();
        let whole = self.pending.len() / FRAME * FRAME;
        let frames: Vec<f32> = self.pending.drain(..whole).collect();
        for frame in frames.chunks(FRAME) {
            let speech = self.is_speech(frame);
            self.buf.extend_from_slice(frame);
            self.at += frame.len() as i64;
            if !self.in_utterance {
                if speech {
                    self.in_utterance = true;
                    self.quiet = 0;
                    self.speech = 1;
                } else if self.buf.len() > self.pre_roll {
                    // keep a little audio before speech starts, so first sounds aren't cut
                    let drop = self.buf.len() - self.pre_roll;
                    self.buf.drain(..drop);
                    self.start += drop as i64;
                }
                continue;
            }
            if speech {
                self.speech += 1;
                self.quiet = 0;
            } else {
                self.quiet += 1;
            }
            if (self.quiet >= self.pause_frames || self.buf.len() >= self.max_samples)
                && let Some(u) = self.cut()
            {
                out.push(u);
            }
        }
        out
    }

    fn cut(&mut self) -> Option<(i64, Vec<f32>)> {
        let enough = self.speech >= self.min_speech_frames;
        let samples = std::mem::take(&mut self.buf);
        let start = self.start;
        self.start = self.at;
        self.in_utterance = false;
        self.quiet = 0;
        self.speech = 0;
        enough.then_some((start, samples))
    }

    /// The utterance still open when the stream ends.
    pub fn finish(&mut self) -> Option<(i64, Vec<f32>)> {
        if self.in_utterance { self.cut() } else { None }
    }
}

/// Which captions an output takes.
#[derive(Clone, Debug, PartialEq)]
pub enum Langs {
    /// Every caption, spoken and translated.
    All,
    /// Captions in this language.
    Only(String),
    /// Captions in none of these: the spoken language when it is detected, not given.
    Except(Vec<String>),
}

impl Langs {
    fn takes(&self, lang: &str) -> bool {
        match self {
            Self::All => true,
            Self::Only(l) => l == lang,
            Self::Except(v) => !v.iter().any(|l| l == lang),
        }
    }
}

/// Where captions go, and which of them.
pub struct Output {
    pub langs: Langs,
    pub sink: craft_live::CaptionSink,
}

/// A live captions run.
pub struct LiveOptions {
    pub input: String,
    pub input_format: Option<String>,
    /// Stop after this many seconds (0: when the input ends).
    pub seconds: u32,
    /// The spoken language's code, or `None` to detect it.
    pub language: Option<String>,
    /// Quiet that ends an utterance, seconds.
    pub pause: f32,
    /// Longest utterance, seconds.
    pub max_segment: f32,
}

/// Totals of a run.
#[derive(Debug, Default, Clone, serde::Serialize)]
pub struct LiveStats {
    pub seconds: f64,
    pub captions: u64,
    pub translations: u64,
    pub errors: u64,
}

/// Run until the input ends: each utterance is transcribed, sent to the outputs, translated by
/// each `(language, model)` of `translate`, and the translations sent too. Errors on one
/// utterance are reported on `log` and the run goes on.
pub fn run(
    o: &LiveOptions,
    speech: &dyn Transcriber,
    translate: &mut [(String, Box<dyn Provider>)],
    outputs: &mut [Output],
    log: &mut dyn Write,
) -> Result<LiveStats, String> {
    let mut lo = craft_live::LiveOptions::new(o.input.clone(), None, craft_live::Raw::Audio { rate: SAMPLE_RATE, channels: 1 });
    lo.input_format = o.input_format.clone();
    lo.seconds = o.seconds;
    let mut live = craft_live::Live::start(&lo).map_err(|e| e.to_string())?;
    let (tx, rx) = mpsc::channel::<Vec<f32>>();
    // The reader owns the FFmpeg side: it reads until the input ends, then waits for FFmpeg.
    let reader = std::thread::spawn(move || {
        let mut raw = vec![0u8; FRAME * 4 * 10];
        loop {
            match live.frames.read(&mut raw) {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    let whole = n / 4 * 4;
                    let s: Vec<f32> = raw[..whole].as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
                    if tx.send(s).is_err() {
                        live.stop();
                        break;
                    }
                }
            }
        }
        live.finish().map_err(|e| e.to_string())
    });
    let _ = writeln!(
        log,
        "{}",
        serde_json::json!({"live": "started", "input": o.input, "model": speech.id(), "device": speech.device(),
            "translate": translate.iter().map(|(l, p)| serde_json::json!({"lang": l, "model": p.name()})).collect::<Vec<_>>()})
    );
    let mut seg = Segmenter::new(o.pause, o.max_segment);
    let mut r = Run {
        speech,
        opts: Options { language: o.language.clone(), diarize: false, ..Default::default() },
        translate,
        outputs,
        stats: LiveStats::default(),
        log,
    };
    let mut total = 0usize;
    for chunk in rx {
        total += chunk.len();
        for (start, samples) in seg.push(&chunk) {
            r.utterance(start, &samples);
        }
    }
    if let Some((start, samples)) = seg.finish() {
        r.utterance(start, &samples);
    }
    let finished = reader.join().map_err(|_| "the input reader stopped unexpectedly".to_string()).and_then(|r| r);
    r.stats.seconds = total as f64 / f64::from(SAMPLE_RATE);
    let _ = writeln!(r.log, "{}", serde_json::json!({"live": "ended", "stats": r.stats}));
    finished.map(|()| r.stats)
}

/// One run's models, outputs and totals.
struct Run<'a> {
    speech: &'a dyn Transcriber,
    opts: Options,
    translate: &'a mut [(String, Box<dyn Provider>)],
    outputs: &'a mut [Output],
    stats: LiveStats,
    log: &'a mut dyn Write,
}

impl Run<'_> {
    fn note(&mut self, v: serde_json::Value) {
        self.stats.errors += 1;
        let _ = writeln!(self.log, "{v}");
    }

    fn send(&mut self, c: &craft_live::Caption) {
        let mut failed = Vec::new();
        for out in self.outputs.iter_mut().filter(|o| o.langs.takes(&c.lang)) {
            if let Err(e) = out.sink.send(c) {
                failed.push(e.to_string());
            }
        }
        for e in failed {
            self.note(serde_json::json!({"error": format!("caption output: {e}")}));
        }
    }

    /// Transcribe one utterance, send it, translate it and send the translations.
    fn utterance(&mut self, start: i64, samples: &[f32]) {
        let t = match self.speech.transcribe(samples, &self.opts, &mut |_, _| true) {
            Ok(t) => t,
            Err(e) => {
                self.note(serde_json::json!({"error": e.to_string(), "at": start as f64 / f64::from(SAMPLE_RATE)}));
                return;
            }
        };
        let text = t.text();
        if text.trim().is_empty() {
            return;
        }
        let at = |s: i64| (start + s) as f64 / f64::from(SAMPLE_RATE);
        let a = t.words.first().map_or(at(0), |w| at(w.start.0));
        let b = t.words.last().map_or(at(samples.len() as i64), |w| at(w.end.0));
        self.send(&craft_live::Caption { start: a, end: b, text: text.clone(), lang: t.language.clone(), is_final: true });
        self.stats.captions += 1;
        let mut done = Vec::new();
        let mut failed = Vec::new();
        for (lang, p) in self.translate.iter_mut() {
            match translate_segments(p.as_mut(), &[Segment::Text(text.clone())], lang) {
                Ok(v) => done.push(craft_live::Caption { start: a, end: b, text: v.concat(), lang: lang.clone(), is_final: true }),
                Err(e) => failed.push(format!("translate to {lang}: {e}")),
            }
        }
        for c in done {
            self.send(&c);
            self.stats.translations += 1;
        }
        for e in failed {
            self.note(serde_json::json!({"error": e}));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(seconds: f32) -> Vec<f32> {
        (0..(seconds * SAMPLE_RATE as f32) as usize).map(|i| 0.3 * (i as f32 * 0.07).sin()).collect()
    }

    fn quiet(seconds: f32) -> Vec<f32> {
        (0..(seconds * SAMPLE_RATE as f32) as usize).map(|i| if i % 2 == 0 { 0.0005 } else { -0.0005 }).collect()
    }

    #[test]
    fn utterances_end_at_pauses_and_keep_their_place_in_the_stream() {
        let mut s = Segmenter::new(0.5, 10.0);
        let mut got = Vec::new();
        for part in [quiet(1.0), tone(1.0), quiet(1.0), tone(0.8), quiet(1.0)] {
            // fed in uneven pieces, as a stream arrives
            for piece in part.chunks(1234) {
                got.extend(s.push(piece));
            }
        }
        got.extend(s.finish());
        assert_eq!(got.len(), 2, "two utterances");
        let secs = |x: i64| x as f64 / f64::from(SAMPLE_RATE);
        assert!(secs(got[0].0) > 0.7 && secs(got[0].0) < 1.0, "the first starts just before the speech: {}", secs(got[0].0));
        assert!(secs(got[1].0) > 2.7 && secs(got[1].0) < 3.0, "{}", secs(got[1].0));
        assert!(got[0].1.len() as f64 / f64::from(SAMPLE_RATE) < 2.0, "it ends after the pause, not at the next speech");
    }

    #[test]
    fn a_long_talk_without_pauses_is_cut_at_the_longest_utterance() {
        let mut s = Segmenter::new(0.5, 3.0);
        let mut got = s.push(&tone(10.0));
        got.extend(s.finish());
        assert!(got.len() >= 3, "{}", got.len());
        assert!(got.iter().all(|(_, v)| v.len() <= 3 * SAMPLE_RATE as usize + FRAME));
    }

    #[test]
    fn clicks_shorter_than_speech_make_no_caption() {
        let mut s = Segmenter::new(0.5, 10.0);
        let mut audio = quiet(1.0);
        audio.extend(tone(0.06));
        audio.extend(quiet(1.0));
        let mut got = s.push(&audio);
        got.extend(s.finish());
        assert!(got.is_empty());
    }
}
