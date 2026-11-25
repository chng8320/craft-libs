//! AI tasks on text and documents, each on any chat model ([`Provider`]): summaries, field
//! extraction, chat, questions about a document, language detection, and translating subtitle
//! and XLIFF files with their timing, markup and inline codes kept. The prompts live here, so
//! every caller (the apps, `craft-ai-cli`, the Dataverse scripts) gets the same results.

use crate::error::AiError;
use crate::provider::{GenOpts, Provider};
use crate::translate::{Segment, translate_segments};

/// The longest text a single prompt carries, in characters; longer input is cut and says so.
pub const MAX_TEXT: usize = 24_000;

fn clip(text: &str, max: usize) -> (&str, bool) {
    match text.char_indices().nth(max) {
        Some((i, _)) => (&text[..i], true),
        None => (text, false),
    }
}

/// How long a summary is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Length {
    Short,
    Medium,
    Bullets,
}

impl std::str::FromStr for Length {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "short" => Ok(Self::Short),
            "medium" | "" => Ok(Self::Medium),
            "bullets" => Ok(Self::Bullets),
            o => Err(format!("length {o:?}: short, medium or bullets")),
        }
    }
}

/// A summary of `text` in `language` (empty: the text's own). Returns it and whether the text
/// was cut to [`MAX_TEXT`].
pub fn summarize(p: &mut dyn Provider, text: &str, length: Length, language: &str) -> Result<(String, bool), AiError> {
    let (t, cut) = clip(text, MAX_TEXT);
    let how = match length {
        Length::Short => "in 2 or 3 sentences",
        Length::Medium => "in one paragraph of about 120 words",
        Length::Bullets => "as 3 to 7 short bullet points",
    };
    let lang = if language.trim().is_empty() { "the language of the text" } else { language };
    let prompt =
        format!("Summarize the text below {how}, in {lang}. Keep names, numbers and dates exact. Write only the summary.\n\n<text>\n{t}\n</text>");
    let s = p.complete(&prompt, &GenOpts { max_tokens: 800, temperature: 0.1 })?;
    Ok((s.trim().to_string(), cut))
}

/// The first JSON object in a model's answer.
fn json_object(answer: &str) -> Option<serde_json::Map<String, serde_json::Value>> {
    let start = answer.find('{')?;
    let end = answer.rfind('}')?;
    (end > start).then(|| serde_json::from_str::<serde_json::Value>(&answer[start..=end]).ok()).flatten().and_then(|v| v.as_object().cloned())
}

/// The values of `fields` as `text` states them (`null` where it doesn't), as one JSON object
/// with exactly those keys. An answer that is not JSON is asked for again once, more strictly.
pub fn extract(
    p: &mut dyn Provider,
    text: &str,
    fields: &[String],
    instructions: &str,
) -> Result<serde_json::Map<String, serde_json::Value>, AiError> {
    if fields.is_empty() {
        return Err(AiError::Inference("name the fields to extract".into()));
    }
    let (t, _) = clip(text, MAX_TEXT);
    let names = fields.iter().map(|f| serde_json::Value::String(f.clone()).to_string()).collect::<Vec<_>>().join(", ");
    let ask = |strict: bool| {
        format!(
            "From the text below, extract these fields: {names}. {instructions}\nAnswer with one JSON object whose keys are exactly those field names; use null for a field the text does not state. Copy values as written; do not guess.{}\n\n<text>\n{t}\n</text>",
            if strict { " Write nothing but the JSON object." } else { "" }
        )
    };
    let o = GenOpts { max_tokens: 1000, temperature: 0.0 };
    let got = match json_object(&p.complete(&ask(false), &o)?) {
        Some(v) => v,
        None => {
            json_object(&p.complete(&ask(true), &o)?).ok_or_else(|| AiError::BadResponse("the model did not answer with a JSON object".into()))?
        }
    };
    Ok(fields.iter().map(|f| (f.clone(), got.get(f).cloned().unwrap_or(serde_json::Value::Null))).collect())
}

/// One message of a conversation.
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct Message {
    /// `system`, `user` or `assistant`.
    pub role: String,
    pub content: String,
}

/// The next answer in a conversation (a system message, then turns, ending with the user's).
pub fn chat(p: &mut dyn Provider, messages: &[Message], opts: &GenOpts) -> Result<String, AiError> {
    if messages.last().is_none_or(|m| m.role != "user") {
        return Err(AiError::Inference("a conversation ends with the user's message".into()));
    }
    let mut prompt = String::new();
    for m in messages {
        let who = match m.role.as_str() {
            "system" => "Instructions",
            "assistant" => "Assistant",
            _ => "User",
        };
        prompt.push_str(&format!("{who}: {}\n\n", m.content.trim()));
    }
    prompt.push_str("Answer as the assistant, to the last user message.");
    Ok(p.complete(&prompt, opts)?.trim().to_string())
}

/// Pieces of `text` of about `size` characters, cut at paragraph or sentence ends.
pub fn chunks(text: &str, size: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for para in text.split("\n\n").map(str::trim).filter(|p| !p.is_empty()) {
        if !cur.is_empty() && cur.chars().count() + para.chars().count() > size {
            out.push(std::mem::take(&mut cur));
        }
        if para.chars().count() > size {
            for sentence in para.split_inclusive(['.', '!', '?', '。']) {
                if !cur.is_empty() && cur.chars().count() + sentence.chars().count() > size {
                    out.push(std::mem::take(&mut cur));
                }
                cur.push_str(sentence);
            }
        } else {
            if !cur.is_empty() {
                cur.push_str("\n\n");
            }
            cur.push_str(para);
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur);
    }
    out
}

fn words(s: &str) -> Vec<String> {
    s.split(|c: char| !c.is_alphanumeric()).filter(|w| w.chars().count() > 2).map(str::to_lowercase).collect()
}

/// An answer to `question` from `document` alone: the pieces that share the most words with the
/// question (up to [`MAX_TEXT`] characters) go to the model, which answers from them and names the
/// pieces it used. Returns the answer and the pieces given (their numbers, from 1).
pub fn ask(p: &mut dyn Provider, document: &str, question: &str) -> Result<(String, Vec<usize>), AiError> {
    let pieces = chunks(document, 1500);
    if pieces.is_empty() {
        return Err(AiError::Inference("the document has no text".into()));
    }
    let q: std::collections::HashSet<String> = words(question).into_iter().collect();
    let mut scored: Vec<(usize, usize)> = pieces.iter().enumerate().map(|(i, c)| (words(c).iter().filter(|w| q.contains(*w)).count(), i)).collect();
    scored.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    let mut chosen = Vec::new();
    let mut size = 0;
    for (_, i) in scored {
        if size + pieces[i].chars().count() > MAX_TEXT && !chosen.is_empty() {
            break;
        }
        size += pieces[i].chars().count();
        chosen.push(i);
    }
    chosen.sort_unstable();
    let ctx: String = chosen.iter().map(|&i| format!("[{}]\n{}\n\n", i + 1, pieces[i])).collect();
    let prompt = format!(
        "Answer the question from the numbered extracts of a document below, and from nothing else. Name the extracts you used, like [3]. If they do not answer it, say so.\n\nQuestion: {question}\n\n{ctx}"
    );
    let a = p.complete(&prompt, &GenOpts { max_tokens: 800, temperature: 0.1 })?;
    Ok((a.trim().to_string(), chosen.into_iter().map(|i| i + 1).collect()))
}

/// The language of `text` without a model: its script, then (for Latin script) the letters and
/// common words of the languages it is most often. `(ISO 639-1 code, confidence 0..1)`; `None`
/// when the text has too few letters to tell.
pub fn detect_language(text: &str) -> Option<(&'static str, f32)> {
    let letters: Vec<char> = text.chars().filter(|c| c.is_alphabetic()).collect();
    if letters.len() < 8 {
        return None;
    }
    let n = letters.len() as f32;
    let share = |f: &dyn Fn(char) -> bool| letters.iter().filter(|c| f(**c)).count() as f32 / n;
    let in_range = |lo: u32, hi: u32| move |c: char| (lo..=hi).contains(&(c as u32));
    let kana = share(&in_range(0x3040, 0x30FF));
    let scripts: [(&'static str, f32); 9] = [
        ("ja", kana + if kana > 0.05 { share(&in_range(0x4E00, 0x9FFF)) } else { 0.0 }),
        ("zh", if kana > 0.05 { 0.0 } else { share(&in_range(0x4E00, 0x9FFF)) }),
        ("ko", share(&in_range(0xAC00, 0xD7AF))),
        ("th", share(&in_range(0x0E00, 0x0E7F))),
        ("ar", share(&in_range(0x0600, 0x06FF))),
        ("he", share(&in_range(0x0590, 0x05FF))),
        ("ru", share(&in_range(0x0400, 0x04FF))),
        ("el", share(&in_range(0x0370, 0x03FF))),
        ("hi", share(&in_range(0x0900, 0x097F))),
    ];
    if let Some((code, s)) = scripts.iter().copied().max_by(|a, b| a.1.total_cmp(&b.1)).filter(|(_, s)| *s > 0.3) {
        return Some((code, s.min(1.0)));
    }
    // Vietnamese has letters no other Latin-script language uses
    let vi = share(&|c: char| "ơưăđạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ".contains(c.to_lowercase().next().unwrap_or(c)));
    if vi > 0.02 {
        return Some(("vi", (0.6 + vi * 4.0).min(1.0)));
    }
    const COMMON: &[(&str, &[&str])] = &[
        ("en", &["the", "and", "of", "to", "is", "in", "that", "it", "for", "with"]),
        ("fr", &["le", "la", "les", "et", "de", "des", "est", "une", "que", "pour"]),
        ("de", &["der", "die", "und", "das", "ist", "nicht", "mit", "ein", "zu", "den"]),
        ("es", &["el", "la", "los", "y", "de", "que", "es", "en", "una", "para"]),
        ("it", &["il", "la", "e", "di", "che", "non", "un", "per", "sono", "gli"]),
        ("pt", &["o", "a", "os", "e", "de", "que", "não", "um", "uma", "para"]),
        ("nl", &["de", "het", "een", "en", "van", "is", "niet", "dat", "op", "te"]),
        ("id", &["yang", "dan", "di", "ini", "itu", "dengan", "untuk", "tidak", "dari", "ada"]),
    ];
    let toks: Vec<String> = text.split(|c: char| !c.is_alphabetic()).filter(|w| !w.is_empty()).map(str::to_lowercase).collect();
    if toks.is_empty() {
        return None;
    }
    let mut best = ("", 0usize);
    let mut second = 0usize;
    for (code, list) in COMMON {
        let hits = toks.iter().filter(|t| list.contains(&t.as_str())).count();
        if hits > best.1 {
            second = best.1;
            best = (code, hits);
        } else if hits > second {
            second = hits;
        }
    }
    if best.1 == 0 {
        return None;
    }
    let margin = (best.1 - second) as f32 / best.1 as f32;
    Some((best.0, (0.4 + 0.6 * margin).min(1.0)))
}

/// The language of `text`, asked of a model: an ISO 639-1 code.
pub fn detect_language_with(p: &mut dyn Provider, text: &str) -> Result<String, AiError> {
    let (t, _) = clip(text, 2000);
    let a = p.complete(
        &format!("Which language is the text below written in? Answer with its two-letter ISO 639-1 code only.\n\n<text>\n{t}\n</text>"),
        &GenOpts { max_tokens: 10, temperature: 0.0 },
    )?;
    let code: String = a.trim().chars().filter(char::is_ascii_alphabetic).take(3).collect::<String>().to_lowercase();
    if code.len() < 2 { Err(AiError::BadResponse(format!("not a language code: {a:?}"))) } else { Ok(code) }
}

/// A subtitle file (SRT, WebVTT, TTML, SCC and the other formats craft-captions reads)
/// translated cue by cue, in the same format, with its times, numbering, styling and inline tags
/// kept. Returns the file and the number of cues translated.
pub fn translate_captions(p: &mut dyn Provider, bytes: &[u8], extension: Option<&str>, to: &str) -> Result<(Vec<u8>, usize), AiError> {
    let format = craft_captions::detect(bytes, extension)
        .ok_or_else(|| AiError::Inference("not a subtitle file this reads (SRT, WebVTT, TTML, SCC…)".into()))?;
    let mut doc = craft_captions::parse(bytes, format).map_err(|e| AiError::Inference(e.to_string()))?;
    let segs: Vec<Segment> =
        doc.cues.iter().map(|c| if c.text.trim().is_empty() { Segment::Protected(c.text.clone()) } else { Segment::Text(c.text.clone()) }).collect();
    let n = segs.iter().filter(|s| matches!(s, Segment::Text(_))).count();
    if n == 0 {
        return Err(AiError::Inference("no subtitles found".into()));
    }
    let out = translate_segments(p, &segs, to)?;
    for (c, t) in doc.cues.iter_mut().zip(out) {
        c.text = t.trim().to_string();
    }
    Ok((craft_captions::write(&doc, format, craft_captions::WriteOptions::default()), n))
}

/// What [`translate_xliff`] did.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct XliffStats {
    pub translated: usize,
    /// Units left as they were (already translated, `translate="no"`, empty).
    pub kept: usize,
    /// Inline codes the model dropped, put back at the end of their unit so nothing is lost.
    pub codes_moved_to_end: usize,
    pub language: String,
}

fn flatten(el: &craft_xml::El) -> (String, Vec<craft_xml::El>) {
    let mut text = String::new();
    let mut codes = Vec::new();
    for n in &el.children {
        match n {
            craft_xml::Node::Text(t) | craft_xml::Node::CData(t) => text.push_str(t),
            craft_xml::Node::El(e) => {
                codes.push(e.clone());
                text.push_str(&format!("{{{}}}", codes.len()));
            }
            _ => {}
        }
    }
    (text, codes)
}

/// `text` with its `{n}` placeholders turned back into `codes`; returns the children and how many
/// codes had to go at the end.
fn rebuild(text: &str, codes: &[craft_xml::El]) -> (Vec<craft_xml::Node>, usize) {
    let mut out = Vec::new();
    let mut used = vec![false; codes.len()];
    let mut rest = text;
    let mut plain = String::new();
    while let Some(open) = rest.find('{') {
        let close = rest[open..].find('}').map(|c| open + c);
        let n = close.and_then(|c| rest[open + 1..c].parse::<usize>().ok());
        match (n, close) {
            (Some(n), Some(c)) if (1..=codes.len()).contains(&n) && !used[n - 1] => {
                plain.push_str(&rest[..open]);
                if !plain.is_empty() {
                    out.push(craft_xml::Node::Text(std::mem::take(&mut plain)));
                }
                out.push(craft_xml::Node::El(codes[n - 1].clone()));
                used[n - 1] = true;
                rest = &rest[c + 1..];
            }
            _ => {
                plain.push_str(&rest[..=open]);
                rest = &rest[open + 1..];
            }
        }
    }
    plain.push_str(rest);
    if !plain.is_empty() {
        out.push(craft_xml::Node::Text(plain));
    }
    let mut moved = 0;
    for (i, u) in used.iter().enumerate() {
        if !u {
            out.push(craft_xml::Node::El(codes[i].clone()));
            moved += 1;
        }
    }
    (out, moved)
}

/// An XLIFF 1.2 file with every translation unit's target filled: inline codes (bold, links,
/// fields, tags) become `{n}` placeholders the model leaves alone and are put back exactly.
/// Units already translated are kept unless `overwrite`; a target that is a copy of its source
/// (as Okapi writes them) counts as untranslated. `to` empty: the file's `target-language`.
pub fn translate_xliff(p: &mut dyn Provider, bytes: &[u8], to: &str, overwrite: bool) -> Result<(Vec<u8>, XliffStats), AiError> {
    let mut root = craft_xml::parse(bytes).map_err(|e| AiError::Inference(format!("not XLIFF: {e}")))?;
    if root.local() != "xliff" {
        return Err(AiError::Inference(format!("not XLIFF: the document is <{}>", root.name)));
    }
    let lang = if to.trim().is_empty() {
        first_attr(&root, "file", "target-language")
            .ok_or_else(|| AiError::Inference("say which language (or give the file a target-language)".into()))?
    } else {
        to.trim().to_string()
    };
    // collect the units to translate, by their path in the tree
    let mut todo: Vec<(Vec<usize>, String, Vec<craft_xml::El>)> = Vec::new();
    let mut kept = 0;
    walk(&root, &mut |path, u| {
        if u.attr("translate") == Some("no") {
            kept += 1;
            return;
        }
        let Some(src) = child(u, "source") else { return };
        let (text, codes) = flatten(src);
        if text.trim().is_empty() {
            kept += 1;
            return;
        }
        if let Some(t) = child(u, "target") {
            let (tt, _) = flatten(t);
            if !overwrite && !tt.trim().is_empty() && tt.trim() != text.trim() {
                kept += 1;
                return;
            }
        }
        todo.push((path.to_vec(), text, codes));
    });
    let mut stats = XliffStats { kept, language: lang.clone(), ..Default::default() };
    if !todo.is_empty() {
        let segs: Vec<Segment> = todo.iter().map(|(_, t, _)| Segment::Text(t.clone())).collect();
        let out = translate_segments(p, &segs, &lang)?;
        for ((path, _, codes), t) in todo.iter().zip(out) {
            let Some(u) = at_path(&mut root, path) else { continue };
            let (children, moved) = rebuild(&t, codes);
            stats.codes_moved_to_end += moved;
            let prefix = u.prefix().to_string();
            let name = if prefix.is_empty() { "target".to_string() } else { format!("{prefix}:target") };
            match u.children.iter_mut().find_map(|n| if let craft_xml::Node::El(e) = n { (e.local() == "target").then_some(e) } else { None }) {
                Some(tgt) => {
                    tgt.children = children;
                    set_attr(tgt, "state", "needs-review-translation");
                }
                None => {
                    let at = u
                        .children
                        .iter()
                        .position(|n| matches!(n, craft_xml::Node::El(e) if e.local() == "source"))
                        .map_or(u.children.len(), |i| i + 1);
                    let tgt = craft_xml::El { name, attrs: vec![("state".into(), "needs-review-translation".into())], children };
                    u.children.insert(at, craft_xml::Node::El(tgt));
                }
            }
            stats.translated += 1;
        }
    }
    let mut xml = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    root.write_xml(&mut xml);
    Ok((xml.into_bytes(), stats))
}

fn child<'a>(e: &'a craft_xml::El, local: &str) -> Option<&'a craft_xml::El> {
    e.children.iter().find_map(|n| if let craft_xml::Node::El(c) = n { (c.local() == local).then_some(c) } else { None })
}

fn set_attr(e: &mut craft_xml::El, k: &str, v: &str) {
    match e.attrs.iter_mut().find(|(n, _)| n == k) {
        Some(a) => a.1 = v.to_string(),
        None => e.attrs.push((k.to_string(), v.to_string())),
    }
}

fn first_attr(e: &craft_xml::El, local: &str, attr: &str) -> Option<String> {
    let mut stack = vec![e];
    while let Some(x) = stack.pop() {
        if x.local() == local
            && let Some(v) = x.attr(attr)
        {
            return Some(v.to_string());
        }
        stack.extend(x.children.iter().rev().filter_map(|n| if let craft_xml::Node::El(c) = n { Some(c) } else { None }));
    }
    None
}

/// Every `trans-unit`, with its path (child indexes from the root), in document order.
fn walk(e: &craft_xml::El, f: &mut dyn FnMut(&[usize], &craft_xml::El)) {
    let mut stack: Vec<(&craft_xml::El, Vec<usize>)> = vec![(e, Vec::new())];
    while let Some((x, p)) = stack.pop() {
        if x.local() == "trans-unit" {
            f(&p, x);
            continue;
        }
        for (i, n) in x.children.iter().enumerate().rev() {
            if let craft_xml::Node::El(c) = n {
                let mut q = p.clone();
                q.push(i);
                stack.push((c, q));
            }
        }
    }
}

fn at_path<'a>(root: &'a mut craft_xml::El, path: &[usize]) -> Option<&'a mut craft_xml::El> {
    let mut cur = root;
    for &i in path {
        cur = match cur.children.get_mut(i)? {
            craft_xml::Node::El(e) => e,
            _ => return None,
        };
    }
    Some(cur)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::ProviderInfo;

    /// Answers by rule: `[vi] ` before each translated segment line, or a fixed reply.
    struct Echo(Option<String>);
    impl Provider for Echo {
        fn name(&self) -> String {
            "echo".into()
        }
        fn complete(&mut self, prompt: &str, _: &GenOpts) -> Result<String, AiError> {
            if let Some(r) = &self.0 {
                return Ok(r.clone());
            }
            // translate_segments sends numbered items, or one text after a blank line
            let body = prompt.split_once("\n\n").map_or("", |(_, b)| b);
            if prompt.starts_with("Translate each numbered item") {
                return Ok(body.lines().filter_map(|l| l.split_once(". ")).map(|(n, t)| format!("{n}. [vi] {t}\n")).collect());
            }
            Ok(format!("[vi] {}", body.trim()))
        }
        fn test(&mut self) -> Result<ProviderInfo, AiError> {
            Ok(ProviderInfo { label: "echo".into(), models: vec![] })
        }
        fn is_local(&self) -> bool {
            true
        }
    }

    #[test]
    fn extraction_keeps_exactly_the_fields_asked_for() {
        let mut p = Echo(Some("Sure! {\"total\": \"1.200.000 đ\", \"extra\": 1}".into()));
        let v = extract(&mut p, "Tổng cộng: 1.200.000 đ", &["total".into(), "date".into()], "").unwrap();
        assert_eq!(v.keys().cloned().collect::<std::collections::BTreeSet<_>>(), ["date", "total"].map(String::from).into());
        assert_eq!(v["date"], serde_json::Value::Null);
        assert!(extract(&mut Echo(Some("no json".into())), "x", &["a".into()], "").is_err());
    }

    #[test]
    fn languages_are_told_from_script_letters_and_common_words() {
        assert_eq!(detect_language("Xin chào, hôm nay trời đẹp quá và tôi rất vui").map(|x| x.0), Some("vi"));
        assert_eq!(detect_language("The quick brown fox jumps over the lazy dog and the cat").map(|x| x.0), Some("en"));
        assert_eq!(detect_language("これは日本語の文章です。ひらがなとカタカナ").map(|x| x.0), Some("ja"));
        assert_eq!(detect_language("这是一个中文句子，用来测试语言识别").map(|x| x.0), Some("zh"));
        assert_eq!(detect_language("Привет, как у тебя дела сегодня").map(|x| x.0), Some("ru"));
        assert_eq!(detect_language("12345 !!"), None);
    }

    #[test]
    fn questions_get_the_pieces_that_share_their_words() {
        let doc = "Cats sleep a lot during the day.\n\nThe invoice total is 500 dollars, due in March.\n\nDogs bark at night.";
        let (_, used) = ask(&mut Echo(Some("It is 500 dollars [2].".into())), doc, "What is the invoice total?").unwrap();
        assert!(used.contains(&1) || used.len() == 1, "small documents go whole: {used:?}");
        let pieces = chunks(&"word. ".repeat(2000), 1500);
        assert!(pieces.len() > 5 && pieces.iter().all(|p| p.chars().count() <= 1600));
    }

    #[test]
    fn subtitles_keep_their_times_and_numbering() {
        let srt = b"1\n00:00:01,000 --> 00:00:02,500\nHello <i>there</i>\n\n2\n00:00:03,000 --> 00:00:04,000\nBye\n";
        let (out, n) = translate_captions(&mut Echo(None), srt, Some("srt"), "vi").unwrap();
        let s = String::from_utf8(out).unwrap();
        assert_eq!(n, 2);
        assert!(s.contains("00:00:01,000 --> 00:00:02,500") && s.contains("00:00:03,000 --> 00:00:04,000"), "{s}");
        assert!(s.contains("[vi] Bye"), "{s}");
    }

    #[test]
    fn xliff_inline_codes_come_back_exactly_and_done_units_stay() {
        let x = br#"<?xml version="1.0"?><xliff version="1.2" xmlns="urn:oasis:names:tc:xliff:document:1.2"><file source-language="en" target-language="vi" datatype="x"><body>
<trans-unit id="1"><source>Click <g id="1">here</g> now</source></trans-unit>
<trans-unit id="2"><source>Done</source><target>Xong</target></trans-unit>
<trans-unit id="3" translate="no"><source>Code</source></trans-unit>
<trans-unit id="4"><source>Copy</source><target>Copy</target></trans-unit>
</body></file></xliff>"#;
        let (out, st) = translate_xliff(&mut Echo(None), x, "", false).unwrap();
        let s = String::from_utf8(out).unwrap();
        assert_eq!((st.translated, st.kept, st.language.as_str()), (2, 2, "vi"));
        assert!(s.contains(r#"<g id="1">here</g>"#), "the code is back: {s}");
        assert!(s.contains("<target>Xong</target>"), "a translated unit stays: {s}");
        assert!(s.contains("needs-review-translation"));
        assert!(craft_xml::parse(s.as_bytes()).is_ok(), "well-formed");
    }

    #[test]
    fn a_dropped_code_goes_back_at_the_end() {
        let codes = vec![craft_xml::El { name: "g".into(), attrs: vec![], children: vec!["x".into()] }];
        let (nodes, moved) = rebuild("no placeholder here", &codes);
        assert_eq!(moved, 1);
        assert!(matches!(nodes.last(), Some(craft_xml::Node::El(_))));
    }
}
