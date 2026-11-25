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
