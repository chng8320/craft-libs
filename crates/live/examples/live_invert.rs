//! A live run end to end: FFmpeg reads INPUT, each frame is inverted here, FFmpeg writes OUTPUT.
//!
//!     cargo run -p craft-live --example live_invert -- INPUT OUTPUT [--format lavfi] [--seconds N]
//!
//! e.g. `-- "testsrc=size=320x180:rate=15" out.mp4 --format lavfi --seconds 2`.

use std::io::{Read, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let (input, output) = match (args.first(), args.get(1)) {
        (Some(i), Some(o)) => (i.clone(), o.clone()),
        _ => return Err("usage: live_invert INPUT OUTPUT [--format F] [--seconds N]".into()),
    };
    let mut o = craft_live::LiveOptions::new(input, Some(output), craft_live::Raw::Video { width: 320, height: 180, fps: 15.0 });
    o.input_format = flag("--format");
    o.seconds = flag("--seconds").and_then(|s| s.parse().ok()).unwrap_or(0);
    o.carry_other = false;
    let mut live = craft_live::Live::start(&o)?;
    let mut frame = vec![0u8; o.raw.unit_bytes()];
    let mut n = 0u64;
    let mut out = live.out.take().ok_or("no output")?;
    while live.frames.read_exact(&mut frame).is_ok() {
        for px in frame.as_chunks_mut::<4>().0 {
            for c in &mut px[..3] {
                *c = 255 - *c;
            }
        }
        out.write_all(&frame)?;
        n += 1;
    }
    drop(out);
    live.finish()?;
    println!("{n} frames");
    Ok(())
}
