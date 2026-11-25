//! Which processor runs a local model: the caller's choice, parsed from settings or a command
//! line. Opening the device is in [`crate::local`]; this part is plain data so every target
//! (wasm too) can store and show the choice.

use std::fmt;
use std::str::FromStr;

/// Where local inference runs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DeviceChoice {
    /// The first GPU that this build supports and this machine has, else the CPU.
    #[default]
    Auto,
    /// The CPU.
    Cpu,
    /// Any GPU this build supports; an error when there is none, never a silent CPU run.
    Gpu,
    /// NVIDIA GPU number `n` (CUDA). Needs a build with the `cuda` feature.
    Cuda(usize),
    /// Apple GPU number `n` (Metal). Needs a build with the `metal` feature.
    Metal(usize),
}

impl FromStr for DeviceChoice {
    type Err = String;

    /// `auto`, `cpu`, `gpu`, `cuda`, `cuda:N`, `metal`, `metal:N` (any case).
    fn from_str(s: &str) -> Result<Self, String> {
        let s = s.trim().to_ascii_lowercase();
        let (kind, n) = match s.split_once(':') {
            Some((k, n)) => (k.to_string(), Some(n.parse::<usize>().map_err(|_| format!("device {s:?}: {n:?} is not a GPU number"))?)),
            None => (s.clone(), None),
        };
        match (kind.as_str(), n) {
            ("auto", None) => Ok(Self::Auto),
            ("cpu", None) => Ok(Self::Cpu),
            ("gpu", None) => Ok(Self::Gpu),
            ("cuda" | "nvidia", n) => Ok(Self::Cuda(n.unwrap_or(0))),
            ("metal" | "apple", n) => Ok(Self::Metal(n.unwrap_or(0))),
            _ => Err(format!("unknown device {s:?}: use auto, cpu, gpu, cuda, cuda:N, metal or metal:N")),
        }
    }
}

impl fmt::Display for DeviceChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => f.write_str("auto"),
            Self::Cpu => f.write_str("cpu"),
            Self::Gpu => f.write_str("gpu"),
            Self::Cuda(n) => write!(f, "cuda:{n}"),
            Self::Metal(n) => write!(f, "metal:{n}"),
        }
    }
}

/// The GPU back ends this build was compiled with (the CPU is always there).
pub fn compiled_backends() -> Vec<&'static str> {
    let mut v = vec!["cpu"];
    if cfg!(feature = "cuda") {
        v.push("cuda");
    }
    if cfg!(feature = "metal") {
        v.push("metal");
    }
    v
}

// Opening a processor needs candle, which the local models (`local`) and speech (`speech`) bring.
#[cfg(any(feature = "local", feature = "speech"))]
mod open {
    use super::DeviceChoice;
    use crate::error::AiError;
    use candle_core::Device;

    type Opener = fn(usize) -> Result<(Device, String), AiError>;

    /// Open the processor `choice` names. The label says what was opened, for
    /// the status line and command-line results; `Auto` that finds no GPU
    /// says so in the label rather than failing.
    pub fn open(choice: DeviceChoice) -> Result<(Device, String), AiError> {
        match choice {
            DeviceChoice::Cpu => Ok((Device::Cpu, "CPU".into())),
            DeviceChoice::Cuda(n) => cuda(n),
            DeviceChoice::Metal(n) => metal(n),
            DeviceChoice::Gpu => any_gpu(),
            DeviceChoice::Auto => Ok(any_gpu().unwrap_or_else(|e| (Device::Cpu, format!("CPU ({e})")))),
        }
    }

    fn cuda(n: usize) -> Result<(Device, String), AiError> {
        if !cfg!(feature = "cuda") {
            return Err(AiError::Inference("this build has no NVIDIA GPU support (it was built without the `cuda` feature)".into()));
        }
        Device::new_cuda(n).map(|d| (d, format!("NVIDIA GPU {n} (CUDA)"))).map_err(|e| AiError::Inference(format!("cannot open NVIDIA GPU {n}: {e}")))
    }

    fn metal(n: usize) -> Result<(Device, String), AiError> {
        if !cfg!(feature = "metal") {
            return Err(AiError::Inference("this build has no Apple GPU support (it was built without the `metal` feature)".into()));
        }
        Device::new_metal(n).map(|d| (d, format!("Apple GPU {n} (Metal)"))).map_err(|e| AiError::Inference(format!("cannot open Apple GPU {n}: {e}")))
    }

    fn any_gpu() -> Result<(Device, String), AiError> {
        if !cfg!(feature = "cuda") && !cfg!(feature = "metal") {
            return Err(AiError::Inference("no GPU support in this build".into()));
        }
        let mut why = Vec::new();
        let backends: [Opener; 2] = [cuda, metal];
        for open in backends {
            match open(0) {
                Ok(d) => return Ok(d),
                Err(e) => why.push(e.to_string()),
            }
        }
        Err(AiError::Inference(format!("no usable GPU: {}", why.join("; "))))
    }

    /// The GPUs this build can open on this machine, as `(device, label)`
    /// pairs like `("cuda:0", "NVIDIA GPU 0 (CUDA)")`. Opening each one costs
    /// a moment, so this is for listings, not for every run.
    pub fn gpus() -> Vec<(String, String)> {
        let mut found = Vec::new();
        if cfg!(feature = "cuda") {
            for n in 0..16 {
                match cuda(n) {
                    Ok((_, label)) => found.push((format!("cuda:{n}"), label)),
                    Err(_) => break,
                }
            }
        }
        if cfg!(feature = "metal") {
            for n in 0..4 {
                match metal(n) {
                    Ok((_, label)) => found.push((format!("metal:{n}"), label)),
                    Err(_) => break,
                }
            }
        }
        found
    }
}
#[cfg(any(feature = "local", feature = "speech"))]
pub use open::{gpus, open};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_names_round_trip_and_bad_ones_are_refused() {
        for s in ["auto", "cpu", "gpu", "cuda:1", "metal:0"] {
            assert_eq!(s.parse::<DeviceChoice>().map(|d| d.to_string()).as_deref(), Ok(s));
        }
        assert_eq!("CUDA".parse::<DeviceChoice>(), Ok(DeviceChoice::Cuda(0)));
        assert!("cuda:x".parse::<DeviceChoice>().is_err());
        assert!("tpu".parse::<DeviceChoice>().is_err());
        assert!("cpu:1".parse::<DeviceChoice>().is_err());
    }
}
