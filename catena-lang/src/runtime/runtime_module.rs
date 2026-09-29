use serde::{Deserialize, Serialize};

use super::{GpuDialect, ValueKind};

/// Rendered GPU source and the public ABI required to execute it.
///
/// A code generator renders this module for one GPU dialect; [`super::Runtime`]
/// compiles and loads it without depending on the generator's intermediate
/// representation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuntimeModule {
    pub dialect: GpuDialect,
    pub source: String,
    pub functions: Vec<GeneratedFunction>,
}

impl RuntimeModule {
    pub fn new(
        dialect: GpuDialect,
        source: impl Into<String>,
        functions: impl IntoIterator<Item = GeneratedFunction>,
    ) -> Self {
        Self {
            dialect,
            source: source.into(),
            functions: functions.into_iter().collect(),
        }
    }
}

/// C ABI metadata for one public entry point in a [`RuntimeModule`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GeneratedFunction {
    pub source_name: String,
    pub symbol: String,
    pub inputs: Vec<ValueKind>,
    pub outputs: Vec<ValueKind>,
}
