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
