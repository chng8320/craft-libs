//! How a prompt is wrapped for a chat model, and which tokens end its answer. Every model family
//! has its own wrapping; picking the wrong one makes a model ramble or never stop. The named
//! templates cover the common families, and [`Template::Custom`] takes any other one, so a new
//! model needs a catalogue entry, not a new build.

use std::fmt;
use std::str::FromStr;

/// A chat prompt format.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Template {
    /// `<|im_start|>` … `<|im_end|>`: Qwen2.5, SmolLM2 and many others.
    ChatMl,
    /// ChatML with an empty think block, so Qwen3 answers without thinking aloud first.
    Qwen3,
    /// Llama 3's header format.
    Llama3,
    /// `<start_of_turn>` … `<end_of_turn>`: Gemma 2 and 3.
    Gemma,
    /// `<|user|>` … `<|end|>`: Phi-3 and Phi-4 mini.
    Phi3,
    /// `[INST]` … `[/INST]`: Mistral and Mixtral instruct.
    Mistral,
    /// Any other format: `text` with `{system}` and `{prompt}` in it, and the tokens that end
    /// an answer.
    Custom { text: String, stop: Vec<String> },
}

impl Template {
    /// The prompt as the model expects it.
    pub fn render(&self, system: &str, prompt: &str) -> String {
        match self {
            Self::ChatMl => format!("<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"),
            Self::Qwen3 => format!("{}<think>\n\n</think>\n\n", Self::ChatMl.render(system, prompt)),
            Self::Llama3 => format!(
                "<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            ),
            Self::Gemma => format!("<start_of_turn>user\n{system}\n\n{prompt}<end_of_turn>\n<start_of_turn>model\n"),
            Self::Phi3 => format!("<|system|>\n{system}<|end|>\n<|user|>\n{prompt}<|end|>\n<|assistant|>\n"),
            Self::Mistral => format!("[INST] {system}\n\n{prompt} [/INST]"),
            Self::Custom { text, .. } => text.replace("{system}", system).replace("{prompt}", prompt),
        }
    }

    /// Tokens that end an answer; the first one the tokenizer knows is enough, all are checked.
    pub fn stop(&self) -> Vec<String> {
        let s = |v: &[&str]| v.iter().map(|t| (*t).to_string()).collect();
        match self {
            Self::ChatMl | Self::Qwen3 => s(&["<|im_end|>", "<|endoftext|>"]),
            Self::Llama3 => s(&["<|eot_id|>", "<|end_of_text|>"]),
            Self::Gemma => s(&["<end_of_turn>", "<eos>"]),
            Self::Phi3 => s(&["<|end|>", "<|endoftext|>"]),
            Self::Mistral => s(&["</s>"]),
            Self::Custom { stop, .. } => stop.clone(),
        }
    }

    /// The format a GGUF's own `tokenizer.chat_template` (a Jinja text) is written for, when it
    /// is one of the named ones.
    pub fn detect(chat_template: &str) -> Option<Self> {
        let t = chat_template;
        if t.contains("<|im_start|>") {
            Some(if t.contains("<think>") { Self::Qwen3 } else { Self::ChatMl })
        } else if t.contains("<|start_header_id|>") {
            Some(Self::Llama3)
        } else if t.contains("<start_of_turn>") {
            Some(Self::Gemma)
        } else if t.contains("<|user|>") && t.contains("<|end|>") {
            Some(Self::Phi3)
        } else if t.contains("[INST]") {
            Some(Self::Mistral)
        } else {
            None
        }
    }

    /// A catalogue's `template` (a name, or a text with `{prompt}`) and `stop` list.
    pub fn from_parts(template: &str, stop: &[String]) -> Result<Self, String> {
        if template.contains("{prompt}") {
            if stop.is_empty() {
                return Err("a custom template needs `stop`: the tokens that end an answer".into());
            }
            return Ok(Self::Custom { text: template.to_string(), stop: stop.to_vec() });
        }
        let named: Self = template.parse()?;
        Ok(if stop.is_empty() { named } else { Self::Custom { text: named.render("{system}", "{prompt}"), stop: stop.to_vec() } })
    }
}

impl FromStr for Template {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "chatml" => Ok(Self::ChatMl),
            "qwen3" => Ok(Self::Qwen3),
            "llama3" => Ok(Self::Llama3),
            "gemma" | "gemma2" | "gemma3" => Ok(Self::Gemma),
            "phi3" | "phi4" => Ok(Self::Phi3),
            "mistral" => Ok(Self::Mistral),
            other => {
                Err(format!("unknown template {other:?}: chatml, qwen3, llama3, gemma, phi3, mistral, or a text with {{system}} and {{prompt}}"))
            }
        }
    }
}

impl fmt::Display for Template {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::ChatMl => "chatml",
            Self::Qwen3 => "qwen3",
            Self::Llama3 => "llama3",
            Self::Gemma => "gemma",
            Self::Phi3 => "phi3",
            Self::Mistral => "mistral",
            Self::Custom { .. } => "custom",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_custom_template_fills_both_places_and_needs_stop_tokens() {
        let t = Template::from_parts("<s>{system}|{prompt}|", &["</s>".into()]).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(t.render("S", "P"), "<s>S|P|");
        assert_eq!(t.stop(), vec!["</s>".to_string()]);
        assert!(Template::from_parts("{prompt}", &[]).is_err());
        assert!(Template::from_parts("nonsense", &[]).is_err());
    }

    #[test]
    fn a_models_own_chat_template_picks_the_family() {
        assert_eq!(Template::detect("{% for m in messages %}<|im_start|>{{ m.role }}"), Some(Template::ChatMl));
        assert_eq!(Template::detect("<|start_header_id|>"), Some(Template::Llama3));
        assert_eq!(Template::detect("<start_of_turn>user"), Some(Template::Gemma));
        assert_eq!(Template::detect("plain"), None);
    }
}
