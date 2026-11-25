use super::*;
use std::time::Duration;

fn video() -> Raw {
    Raw::Video { width: 64, height: 36, fps: 25.0 }
}

#[test]
fn each_output_gets_the_container_its_protocol_needs() {
    assert_eq!(muxer_for("rtmps://live.example/app/key"), Some("flv"));
    assert_eq!(muxer_for("srt://host:9000"), Some("mpegts"));
    assert_eq!(muxer_for("/srv/hls/live.m3u8"), Some("hls"));
    assert_eq!(muxer_for("out.mp4"), None);
}

#[test]
fn a_file_input_is_read_at_stream_speed_but_a_stream_or_device_is_not() {
    let file = LiveOptions::new("talk.mp4", None, video());
    assert!(file.decoder_args().iter().any(|a| a == "-re"));
    let url = LiveOptions::new("rtmp://in/app/key", None, video());
    assert!(!url.decoder_args().iter().any(|a| a == "-re"));
    let mut dev = LiveOptions::new("/dev/video0", None, video());
    dev.input_format = Some("v4l2".into());
    let a = dev.decoder_args();
    assert!(!a.iter().any(|x| x == "-re") && a.windows(2).any(|w| w[0] == "-f" && w[1] == "v4l2"));
}

#[test]
fn the_raw_stream_has_the_same_shape_on_both_sides() {
    let mut o = LiveOptions::new("rtmp://in/a", Some("rtmp://out/b".into()), video());
    o.seconds = 30;
    let (d, e) = (o.decoder_args(), o.encoder_args().unwrap_or_default());
    for a in [&d, &e] {
        assert!(a.windows(2).any(|w| w[0] == "-s" && w[1] == "64x36"), "{a:?}");
        assert!(a.windows(2).any(|w| w[0] == "-pix_fmt" && w[1] == "rgba"), "{a:?}");
    }
    assert!(d.windows(2).any(|w| w[0] == "-t" && w[1] == "30"));
    assert!(e.windows(2).any(|w| w[0] == "-map" && w[1] == "1:a?"), "the input's audio is carried: {e:?}");
    assert_eq!(e.last().map(String::as_str), Some("rtmp://out/b"));
    assert!(LiveOptions::new("in.wav", None, Raw::Audio { rate: 16_000, channels: 1 }).encoder_args().is_none());
}

#[test]
fn caption_times_are_written_the_way_each_format_wants() {
    assert_eq!(stamp(3723.4567, '.'), "01:02:03.457");
    assert_eq!(stamp(0.5, ','), "00:00:00,500");
}

#[test]
fn files_get_only_final_captions_and_json_lines_get_every_one() {
    let dir = std::env::temp_dir().join(format!("craft-live-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("{e}"));
    let (vtt, srt, jl) = (dir.join("c.vtt"), dir.join("c.srt"), dir.join("c.jsonl"));
    let mut sinks: Vec<CaptionSink> = [format!("vtt:{}", vtt.display()), format!("srt:{}", srt.display()), format!("jsonl:{}", jl.display())]
        .iter()
        .map(|s| CaptionSink::open(s).unwrap_or_else(|e| panic!("{e}")))
        .collect();
    let draft = Caption { start: 1.0, end: 2.0, text: "hello wor".into(), lang: "en".into(), is_final: false };
    let done = Caption { text: "hello world".into(), is_final: true, ..draft.clone() };
    for s in &mut sinks {
        s.send(&draft).unwrap_or_else(|e| panic!("{e}"));
        s.send(&done).unwrap_or_else(|e| panic!("{e}"));
    }
    drop(sinks);
    let read = |p: &std::path::Path| std::fs::read_to_string(p).unwrap_or_default();
    let v = read(&vtt);
    assert!(v.starts_with("WEBVTT") && v.contains("00:00:01.000 --> 00:00:02.000\nhello world") && !v.contains("hello wor\n"), "{v}");
    assert!(read(&srt).starts_with("1\n00:00:01,000 --> 00:00:02,000\nhello world"));
    assert_eq!(read(&jl).lines().count(), 2);
    assert!(CaptionSink::open("ftp:/x").is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn control_lines_arrive_from_a_file_and_from_a_tcp_port() {
    let dir = std::env::temp_dir().join(format!("craft-live-ctl-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("{e}"));
    let f = dir.join("ctl.txt");
    std::fs::write(&f, "{\"reset\": true}\n\n{\"bypass\": {\"effect\": 0, \"on\": true}}\n").unwrap_or_else(|e| panic!("{e}"));
    let rx = control_lines(&f.to_string_lossy()).unwrap_or_else(|e| panic!("{e}"));
    let got: Vec<String> = (0..2).filter_map(|_| rx.recv_timeout(Duration::from_secs(5)).ok()).collect();
    assert_eq!(got.len(), 2, "blank lines are dropped: {got:?}");

    let probe = std::net::TcpListener::bind("127.0.0.1:0").unwrap_or_else(|e| panic!("{e}"));
    let port = probe.local_addr().map(|a| a.port()).unwrap_or_else(|e| panic!("{e}"));
    drop(probe);
    let rx = control_lines(&format!("tcp:{port}")).unwrap_or_else(|e| panic!("{e}"));
    let mut c = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap_or_else(|e| panic!("{e}"));
    c.write_all(b"{\"reset\": true}\n").unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(rx.recv_timeout(Duration::from_secs(5)).ok().as_deref(), Some("{\"reset\": true}"));
    let _ = std::fs::remove_dir_all(&dir);
}
