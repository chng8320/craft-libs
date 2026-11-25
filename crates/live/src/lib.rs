//! Live media for the engines' live modes.
//!
//! An engine's stream mode works on raw frames or samples on a reader and a writer. This crate
//! connects those to the outside: FFmpeg (a program the caller names, found on `PATH` by default)
//! reads a live input (RTMP, SRT, RTSP, HLS, a capture device or a file) into raw RGBA frames or
//! interleaved `f32` samples, and encodes what the engine writes back to a live output, the other
//! track carried along from the input. Control messages (one JSON object per line) arrive while it
//! runs from a FIFO, a file or a TCP port. Captions go out as WebVTT, SRT, JSON lines or a webhook.
//!
//! Nothing here decodes or encodes a codec itself: that is FFmpeg's job, run as a separate process.

use std::io::{self, BufRead, BufReader, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::mpsc::{self, Receiver};

use serde::Serialize;

/// What flows between FFmpeg and the engine.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Raw {
    /// RGBA, 8 bits a channel, `width × height`, `fps` frames a second.
    Video { width: u32, height: u32, fps: f64 },
    /// Interleaved 32-bit float samples.
    Audio { rate: u32, channels: u16 },
}

impl Raw {
    /// FFmpeg's description of the raw stream: format and size or rate.
    fn ffmpeg_raw(&self) -> Vec<String> {
        match *self {
            Raw::Video { width, height, fps } => {
                vec![
                    "-f".into(),
                    "rawvideo".into(),
                    "-pix_fmt".into(),
                    "rgba".into(),
                    "-s".into(),
                    format!("{width}x{height}"),
                    "-r".into(),
                    fps_text(fps),
                ]
            }
            Raw::Audio { rate, channels } => vec!["-f".into(), "f32le".into(), "-ar".into(), rate.to_string(), "-ac".into(), channels.to_string()],
        }
    }

    /// Bytes in one frame (video) or one second (audio).
    pub fn unit_bytes(&self) -> usize {
        match *self {
            Raw::Video { width, height, .. } => width as usize * height as usize * 4,
            Raw::Audio { rate, channels } => rate as usize * channels as usize * 4,
        }
    }
}

fn fps_text(fps: f64) -> String {
    if (fps - fps.round()).abs() < 1e-9 { format!("{}", fps.round() as i64) } else { format!("{fps}") }
}

/// The container FFmpeg needs to write to `url`: by scheme, else by extension (`None`: FFmpeg
/// decides from the file name).
pub fn muxer_for(url: &str) -> Option<&'static str> {
    let l = url.to_ascii_lowercase();
    if l.starts_with("rtmp://") || l.starts_with("rtmps://") {
        Some("flv")
    } else if l.starts_with("srt://") || l.starts_with("udp://") || l.starts_with("rtp://") {
        Some("mpegts")
    } else if l.starts_with("rtsp://") {
        Some("rtsp")
    } else if l.ends_with(".m3u8") {
        Some("hls")
    } else {
        None
    }
}

/// A live input that is really a file is read at its own speed (`-re`), as a stream would arrive.
fn is_file(url: &str) -> bool {
    !url.contains("://") && !url.starts_with("/dev/")
}

/// What a live run reads and writes.
#[derive(Clone, Debug)]
pub struct LiveOptions {
    /// The input: a URL, a device (`/dev/video0`, or FFmpeg's `-f` form below) or a file.
    pub input: String,
    /// FFmpeg's input format for a device (`v4l2`, `alsa`, `dshow`, `avfoundation`…), when needed.
    pub input_format: Option<String>,
    /// The output: a URL or a file. `None`: nothing is encoded (a run that only listens, such as
    /// live captions).
    pub output: Option<String>,
    pub raw: Raw,
    /// Carry the input's other track (audio for a video run, video for an audio run) to the output.
    pub carry_other: bool,
    pub video_bitrate_kbps: u32,
    pub audio_bitrate_kbps: u32,
    /// Stop after this many seconds (0: when the input ends).
    pub seconds: u32,
    /// The FFmpeg program.
    pub ffmpeg: PathBuf,
}

impl LiveOptions {
    pub fn new(input: impl Into<String>, output: Option<String>, raw: Raw) -> Self {
        Self {
            input: input.into(),
            input_format: None,
            output,
            raw,
            carry_other: true,
            video_bitrate_kbps: 3000,
            audio_bitrate_kbps: 160,
            seconds: 0,
            ffmpeg: std::env::var_os("CRAFT_LIVE_FFMPEG").map_or_else(|| PathBuf::from("ffmpeg"), PathBuf::from),
        }
    }

    fn input_args(&self) -> Vec<String> {
        let mut a = Vec::new();
        if is_file(&self.input) && self.input_format.is_none() {
            a.push("-re".into());
        }
        if let Some(f) = &self.input_format {
            a.extend(["-f".into(), f.clone()]);
        }
        a.extend(["-i".into(), self.input.clone()]);
        a
    }

    /// FFmpeg's arguments to read the input into the raw stream on its standard output.
    pub fn decoder_args(&self) -> Vec<String> {
        let mut a: Vec<String> = vec!["-hide_banner".into(), "-loglevel".into(), "error".into(), "-nostdin".into()];
        a.extend(self.input_args());
        if self.seconds > 0 {
            a.extend(["-t".into(), self.seconds.to_string()]);
        }
        match self.raw {
            Raw::Video { width, height, fps } => {
                // keep the picture's shape inside the frame (letterboxed), at a steady rate
                let vf = format!(
                    "scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,fps={}",
                    fps_text(fps)
                );
                a.extend(["-an".into(), "-vf".into(), vf]);
            }
            Raw::Audio { .. } => a.push("-vn".into()),
        }
        a.extend(self.raw.ffmpeg_raw());
        a.push("-".into());
        a
    }

    /// FFmpeg's arguments to encode the raw stream on its standard input to the output, or `None`
    /// when there is no output.
    pub fn encoder_args(&self) -> Option<Vec<String>> {
        let out = self.output.as_ref()?;
        let mut a: Vec<String> = vec!["-hide_banner".into(), "-loglevel".into(), "error".into(), "-y".into()];
        a.extend(self.raw.ffmpeg_raw());
        a.extend(["-i".into(), "-".into()]);
        let carry = self.carry_other;
        if carry {
            a.extend(self.input_args());
        }
        match self.raw {
            Raw::Video { fps, .. } => {
                a.extend(["-map".into(), "0:v".into()]);
                if carry {
                    a.extend(["-map".into(), "1:a?".into(), "-c:a".into(), "aac".into(), "-b:a".into(), format!("{}k", self.audio_bitrate_kbps)]);
                }
                let gop = (fps * 2.0).round().max(1.0) as u32;
                a.extend(
                    ["-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency", "-pix_fmt", "yuv420p"].iter().map(|s| s.to_string()).chain([
                        "-b:v".into(),
                        format!("{}k", self.video_bitrate_kbps),
                        "-g".into(),
                        gop.to_string(),
                    ]),
                );
            }
            Raw::Audio { .. } => {
                a.extend(["-map".into(), "0:a".into()]);
                if carry {
                    a.extend(["-map".into(), "1:v?".into(), "-c:v".into(), "copy".into()]);
                }
                a.extend(["-c:a".into(), "aac".into(), "-b:a".into(), format!("{}k", self.audio_bitrate_kbps)]);
            }
        }
        if carry {
            a.push("-shortest".into());
        }
        match muxer_for(out) {
            Some("hls") => {
                a.extend(["-f", "hls", "-hls_time", "2", "-hls_list_size", "6", "-hls_flags", "delete_segments"].iter().map(|s| s.to_string()))
            }
            Some(m) => a.extend(["-f".into(), m.into()]),
            None => {}
        }
        a.push(out.clone());
        Some(a)
    }
}

/// A running live run: the raw stream to read and (with an output) to write, and the FFmpeg
/// processes behind them.
pub struct Live {
    pub frames: ChildStdout,
    pub out: Option<ChildStdin>,
    decoder: Child,
    encoder: Option<Child>,
}

#[derive(Debug)]
pub enum LiveError {
    /// FFmpeg could not be started (not installed, or not at the path given).
    NoFfmpeg(String),
    Io(io::Error),
    /// FFmpeg stopped with an error; its last words.
    Ffmpeg(String),
}

impl std::fmt::Display for LiveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoFfmpeg(m) => write!(f, "FFmpeg is needed for live input and output: {m}"),
            Self::Io(e) => write!(f, "{e}"),
            Self::Ffmpeg(m) => write!(f, "FFmpeg: {m}"),
        }
    }
}

impl std::error::Error for LiveError {}

impl From<io::Error> for LiveError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl Live {
    /// Start FFmpeg on both sides.
    pub fn start(o: &LiveOptions) -> Result<Live, LiveError> {
        let spawn = |args: Vec<String>, stdin: Stdio, stdout: Stdio| {
            Command::new(&o.ffmpeg)
                .args(&args)
                .stdin(stdin)
                .stdout(stdout)
                .stderr(Stdio::piped())
                .spawn()
                .map_err(|e| LiveError::NoFfmpeg(format!("{}: {e}", o.ffmpeg.display())))
        };
        let mut decoder = spawn(o.decoder_args(), Stdio::null(), Stdio::piped())?;
        let frames = decoder.stdout.take().ok_or_else(|| LiveError::Ffmpeg("no output from the decoder".into()))?;
        let (encoder, out) = match o.encoder_args() {
            Some(args) => {
                let mut e = spawn(args, Stdio::piped(), Stdio::null())?;
                let i = e.stdin.take();
                (Some(e), i)
            }
            None => (None, None),
        };
        Ok(Live { frames, out, decoder, encoder })
    }

    /// Close the output and wait for both FFmpeg processes; an error when one failed (an input
    /// that ended is not a failure).
    pub fn finish(mut self) -> Result<(), LiveError> {
        drop(self.out.take());
        let mut why = Vec::new();
        for (name, child) in std::iter::once(("input", &mut self.decoder)).chain(self.encoder.as_mut().map(|e| ("output", e))) {
            let status = child.wait()?;
            if !status.success() {
                let mut err = String::new();
                if let Some(mut e) = child.stderr.take() {
                    let _ = io::Read::read_to_string(&mut e, &mut err);
                }
                why.push(format!("{name}: {}", err.trim().lines().last().unwrap_or("stopped with an error")));
            }
        }
        if why.is_empty() { Ok(()) } else { Err(LiveError::Ffmpeg(why.join("; "))) }
    }

    /// Stop now (the input is told to end; the output finishes what it has).
    pub fn stop(&mut self) {
        let _ = self.decoder.kill();
    }
}

/// Control messages, one line each, from `spec`: `tcp:PORT` (or `tcp:HOST:PORT`) listens and takes
/// lines from every connection; anything else is a file or FIFO read line by line (a FIFO waits for
/// a writer; a plain file is read once). Blank lines are dropped.
pub fn control_lines(spec: &str) -> io::Result<Receiver<String>> {
    let (tx, rx) = mpsc::channel::<String>();
    if let Some(addr) = spec.strip_prefix("tcp:") {
        let addr = if addr.contains(':') { addr.to_string() } else { format!("127.0.0.1:{addr}") };
        let listener = TcpListener::bind(&addr)?;
        std::thread::spawn(move || {
            for conn in listener.incoming().map_while(Result::ok) {
                let tx = tx.clone();
                std::thread::spawn(move || {
                    for line in BufReader::new(conn).lines().map_while(Result::ok) {
                        if !line.trim().is_empty() && tx.send(line).is_err() {
                            break;
                        }
                    }
                });
            }
        });
    } else {
        let path = spec.to_string();
        std::thread::spawn(move || {
            let Ok(f) = std::fs::File::open(&path) else { return };
            for line in BufReader::new(f).lines().map_while(Result::ok) {
                if !line.trim().is_empty() && tx.send(line).is_err() {
                    break;
                }
            }
        });
    }
    Ok(rx)
}

/// One caption: its time in seconds from the start, its text and language, and whether it is
/// final (a caption still being heard may be sent again, changed, before it is final).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Caption {
    pub start: f64,
    pub end: f64,
    pub text: String,
    pub lang: String,
    #[serde(rename = "final")]
    pub is_final: bool,
}

/// Where captions go: `vtt:PATH`, `srt:PATH`, `jsonl:PATH` (`jsonl:-` standard output), or an
/// `http(s)://` webhook (each caption POSTed as JSON). Only final captions go to files.
pub enum CaptionSink {
    Vtt {
        file: std::fs::File,
    },
    Srt {
        file: std::fs::File,
        n: usize,
    },
    Jsonl {
        out: Box<dyn Write + Send>,
    },
    #[cfg(feature = "webhook")]
    Webhook {
        url: String,
        agent: ureq::Agent,
    },
}

impl CaptionSink {
    pub fn open(spec: &str) -> io::Result<CaptionSink> {
        let create = |p: &str| std::fs::File::create(p);
        if spec.starts_with("http://") || spec.starts_with("https://") {
            #[cfg(not(feature = "webhook"))]
            return Err(io::Error::new(io::ErrorKind::Unsupported, "caption webhooks need craft-live's `webhook` feature"));
            #[cfg(feature = "webhook")]
            let agent: ureq::Agent = ureq::Agent::config_builder().timeout_global(Some(std::time::Duration::from_secs(10))).build().into();
            #[cfg(feature = "webhook")]
            return Ok(CaptionSink::Webhook { url: spec.to_string(), agent });
        }
        let (kind, path) = spec.split_once(':').unwrap_or(("jsonl", spec));
        match kind {
            "vtt" => {
                let mut file = create(path)?;
                file.write_all(b"WEBVTT\n\n")?;
                Ok(CaptionSink::Vtt { file })
            }
            "srt" => Ok(CaptionSink::Srt { file: create(path)?, n: 0 }),
            "jsonl" if path == "-" => Ok(CaptionSink::Jsonl { out: Box::new(io::stdout()) }),
            "jsonl" => Ok(CaptionSink::Jsonl { out: Box::new(create(path)?) }),
            other => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("caption output {other:?}: vtt:PATH, srt:PATH, jsonl:PATH or an http(s) URL"),
            )),
        }
    }

    pub fn send(&mut self, c: &Caption) -> io::Result<()> {
        match self {
            CaptionSink::Vtt { file } if c.is_final => {
                write!(file, "{} --> {}\n{}\n\n", stamp(c.start, '.'), stamp(c.end, '.'), c.text)?;
                file.flush()
            }
            CaptionSink::Srt { file, n } if c.is_final => {
                *n += 1;
                write!(file, "{n}\n{} --> {}\n{}\n\n", stamp(c.start, ','), stamp(c.end, ','), c.text)?;
                file.flush()
            }
            CaptionSink::Vtt { .. } | CaptionSink::Srt { .. } => Ok(()),
            CaptionSink::Jsonl { out } => {
                writeln!(out, "{}", serde_json::to_string(c).map_err(io::Error::other)?)?;
                out.flush()
            }
            #[cfg(feature = "webhook")]
            CaptionSink::Webhook { url, agent } => {
                let body = serde_json::to_vec(c).map_err(io::Error::other)?;
                match agent.post(url.as_str()).header("Content-Type", "application/json").send(&body[..]) {
                    Ok(_) => Ok(()),
                    // a webhook that is down loses this caption, not the run
                    Err(e) => {
                        log::warn!("caption webhook {url}: {e}");
                        Ok(())
                    }
                }
            }
        }
    }
}

/// `HH:MM:SS.mmm` (WebVTT) or `HH:MM:SS,mmm` (SRT).
fn stamp(t: f64, sep: char) -> String {
    let ms = (t.max(0.0) * 1000.0).round() as u64;
    format!("{:02}:{:02}:{:02}{sep}{:03}", ms / 3_600_000, ms / 60_000 % 60, ms / 1000 % 60, ms % 1000)
}

#[cfg(test)]
mod tests;
