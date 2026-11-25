# This fork: craft-libs for the Dataverse

This is the Dataverse's fork of [craft-libs](https://github.com/storytold/craft-libs), the ArtCraft team's
shared Rust libraries (MIT OR Apache-2.0). The Dataverse improves the libraries its engines share here, and
builds command lines over the libraries no engine exposes yet.

## Branches

| Branch | What it is |
|---|---|
| `main` | Exactly ArtCraft's `main`. Never commit here; it is only updated from them. |
| `dataverse` | `main` plus our own commits on top. This is what the Dataverse builds. |

Each change is one commit on `dataverse`, so it replays on every new craft-libs and can be offered back as a
pull request. The repository's rules (`AGENTS.md`) apply to our commits too.

## Keeping up with craft-libs

`.github/workflows/dataverse.yml` runs every Monday (and on demand from the Actions tab): it updates `main`
from ArtCraft, replays our commits on `dataverse` (opening an issue when one no longer applies), builds
`craft-ai-cli` with local models and runs craft-ai's tests, and attaches the program as `dataverse-ai-cli`
(Linux x86-64).

## Our changes

| Commit | What | Offered upstream? |
|---|---|---|
| craft-live | `crates/live`: live input and output through FFmpeg (RTMP, SRT, RTSP, HLS, devices, files) as raw frames or samples, control lines from a FIFO, file or TCP port, captions to WebVTT, SRT, JSON lines or a webhook; used by the engines' `live` commands | not yet |
| craft-ai-cli | `crates/ai/examples/craft-ai-cli.rs`: models, download, complete and translate (text or segments) from the command line, local or any OpenAI-compatible endpoint, JSON out | not yet |

## Models and devices (craft-ai)

`craft-ai-cli` takes `--model` (a catalogue id such as `qwen3-4b-q4km`, or any ChatML `.gguf` file of a
qwen2, qwen3 or llama architecture) and `--device` (`auto`, `cpu`, `gpu`, `cuda[:N]`, `metal[:N]`).
`craft-ai-cli devices` lists the back ends the build has and the GPUs it can open. CI builds two
command lines: `dataverse-ai-cli-linux-x86_64` (CPU) and `dataverse-ai-cli-linux-x86_64-cuda`
(NVIDIA GPUs, CPU too).

## The model catalogue (any model, one interface)

`craft_ai::catalog` adds models from JSON files (`<model folder>/catalog.json`, `CRAFT_AI_CATALOG`,
`--catalog`) to the built-in list. Each entry is one id whichever way it runs: `url` (downloaded),
`path` (a GGUF on disk) or `endpoint` (any OpenAI-compatible server), with optional `arch`,
`template` (`chatml`, `qwen3`, `llama3`, `gemma`, `phi3`, `mistral`, or a text with `{system}` and
`{prompt}` plus `stop`) and `device`. Entries carry a `task`; craft-ai reads `chat` (the default) and
leaves other tasks (`speech-to-text`, …) to their engines, so one file serves all of them. Local
architectures: qwen2, qwen3, llama (and Llama-built models such as Mistral and SmolLM2), gemma3, phi3.

## Speech to text (craft_ai::speech)

Moved here from FilmCraft (`crates/speech`) so every engine uses one implementation: Whisper on candle
(feature `speech`; CPU, or GPU with `cuda`/`metal`), the pinned and checksummed Whisper catalogue and its
downloader, speaker labelling, word-edge tightening, catalogue speech models (`task: "speech-to-text"`: Whisper
folders and OpenAI-compatible endpoints). Times are 16 kHz samples (`Samples`), which each app maps to its own
time base. `craft-ai-cli transcribe --file AUDIO [--model] [--device] [--language] [--speakers N]`; `download`
and `models` cover speech models too.

## Live captions and translation (craft_ai::speech::live)

`craft-ai-cli live --in rtmp://… --model whisper-large-v3-turbo --device gpu --language vi --translate en,fr
--translate-model qwen3-8b-q4km --captions "vtt:/srv/live-{lang}.vtt" --captions https://app/captions`: craft-live
reads the stream, pauses cut it into utterances, each is transcribed with any speech model and translated with any
chat model (each on its own device), and the captions go to WebVTT, SRT, JSON lines or a webhook, one output per
language with `{lang}`. Times are the stream's.

## More AI commands (craft_ai::tasks, craft_ai::ocr)

`summarize`, `extract` (fields to JSON, asked again when the answer isn't JSON), `chat`, `ask` (questions about a
document: the pieces that share the question's words go to the model, which names the ones it used),
`detect-language` (by script and common words, or `--model`), `translate-subtitles` (every format craft-captions
reads, times and tags kept), `translate-xliff` (craft-xml; inline codes back exactly), `embed` (endpoint models)
and `ocr` (Tesseract, words with boxes). With `models`, `devices`, `download`, `complete`, `translate`,
`transcribe` and `live`: 16. `craft-ai-cli commands` lists them with their inputs.
