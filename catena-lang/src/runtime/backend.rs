use std::{fmt, str::FromStr};

use serde::{Deserialize, Serialize};

use crate::codegen::GpuDialect;

/// Provider-local runtime selection, independent of program semantics.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    /// Try CUDA, then HIP, selecting the first runtime with a usable device.
    #[default]
    Auto,
    Hip,
    Cuda,
}

impl Backend {
    pub(crate) fn dialect(self) -> Option<GpuDialect> {
        match self {
            Self::Auto => None,
            Self::Hip => Some(GpuDialect::Hip),
            Self::Cuda => Some(GpuDialect::Cuda),
        }
    }
}

impl From<GpuDialect> for Backend {
    fn from(dialect: GpuDialect) -> Self {
        match dialect {
            GpuDialect::Hip => Self::Hip,
            GpuDialect::Cuda => Self::Cuda,
        }
    }
}

impl FromStr for Backend {
    type Err = &'static str;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "auto" => Ok(Self::Auto),
            "hip" => Ok(Self::Hip),
            "cuda" => Ok(Self::Cuda),
            _ => Err("GPU backend must be auto, hip, or cuda"),
        }
    }
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Auto => "auto",
            Self::Hip => "hip",
            Self::Cuda => "cuda",
        })
    }
}
