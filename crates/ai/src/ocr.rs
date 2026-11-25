//! Text from images: OCR as an AI task with the same shape as the others. The default engine is
//! Tesseract (the program, on the CPU; it must be installed, as in the Dataverse document image).
//! Each word comes with its box and confidence, so a caller can make a searchable PDF or check
//! weak spots. Further engines (OCR models, vision models) go behind the same [`ocr`] call.

use std::process::Command;

use crate::error::AiError;

/// One recognised word: its text, confidence 0..100, and box in pixels.
#[derive(Clone, Debug, serde::Serialize)]
pub struct OcrWord {
    pub text: String,
    pub confidence: f32,
    pub left: u32,
    pub top: u32,
    pub width: u32,
    pub height: u32,
    /// Which line of the page (block, paragraph and line numbers joined), for re-flowing.
    pub line: String,
}

/// What [`ocr`] read from one image.
#[derive(Clone, Debug, serde::Serialize)]
pub struct OcrResult {
    pub engine: String,
    pub text: String,
    pub words: Vec<OcrWord>,
    pub mean_confidence: f32,
}

/// Read the text of the image at `path` in `languages` (Tesseract codes joined by `+`, e.g.
/// `vie+eng`). The program is `TESSERACT` when set, else `tesseract`.
pub fn ocr(path: &std::path::Path, languages: &str) -> Result<OcrResult, AiError> {
    let exe = std::env::var("TESSERACT").unwrap_or_else(|_| "tesseract".into());
    let langs = if languages.trim().is_empty() { "eng" } else { languages.trim() };
    let out = Command::new(&exe)
        .arg(path)
        .args(["stdout", "-l", langs, "tsv"])
        .output()
        .map_err(|e| AiError::Inference(format!("OCR needs Tesseract ({exe}): {e}")))?;
    if !out.status.success() {
        return Err(AiError::Inference(format!("Tesseract: {}", String::from_utf8_lossy(&out.stderr).trim().lines().last().unwrap_or("failed"))));
    }
    let words = parse_tsv(&String::from_utf8_lossy(&out.stdout));
    let mut text = String::new();
    let mut last = String::new();
    for w in &words {
        if !text.is_empty() {
            text.push(if w.line == last { ' ' } else { '\n' });
        }
        text.push_str(&w.text);
        last.clone_from(&w.line);
    }
    let mean = if words.is_empty() { 0.0 } else { words.iter().map(|w| w.confidence).sum::<f32>() / words.len() as f32 };
    Ok(OcrResult { engine: format!("tesseract ({langs})"), text, words, mean_confidence: mean })
}

/// Words from Tesseract's TSV (level 5 rows).
pub fn parse_tsv(tsv: &str) -> Vec<OcrWord> {
    let mut out = Vec::new();
    for line in tsv.lines().skip(1) {
        let c: Vec<&str> = line.split('\t').collect();
        if c.len() < 12 || c[0] != "5" || c[11].trim().is_empty() {
            continue;
        }
        let n = |i: usize| c[i].trim().parse::<u32>().unwrap_or(0);
        out.push(OcrWord {
            text: c[11].trim().to_string(),
            confidence: c[10].trim().parse::<f32>().unwrap_or(0.0).max(0.0),
            left: n(6),
            top: n(7),
            width: n(8),
            height: n(9),
            line: format!("{}.{}.{}.{}", c[1], c[2], c[3], c[4]),
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tesseract_rows_become_words_on_their_lines() {
        let tsv = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n\
                   4\t1\t1\t1\t1\t0\t10\t10\t200\t20\t-1\t\n\
                   5\t1\t1\t1\t1\t1\t10\t10\t50\t20\t96.5\tXin\n\
                   5\t1\t1\t1\t1\t2\t70\t10\t60\t20\t91\tchào\n\
                   5\t1\t1\t1\t2\t1\t10\t40\t50\t20\t88\tbạn\n";
        let w = parse_tsv(tsv);
        assert_eq!(w.iter().map(|x| x.text.as_str()).collect::<Vec<_>>(), ["Xin", "chào", "bạn"]);
        assert_eq!(w[0].line, w[1].line);
        assert_ne!(w[1].line, w[2].line);
        assert_eq!((w[1].left, w[1].width), (70, 60));
    }
}
