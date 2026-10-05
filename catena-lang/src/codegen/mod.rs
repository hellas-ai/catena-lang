//! Code generator selection and dispatch.
//!
//! Each backend owns its lowering, intermediate representation, and rendering.
//! Generated output carries its backend so reports and runtime modules cannot
//! accidentally render it using another backend.

pub mod default;
mod dialect;
pub mod experimental;

use hexpr::Operation;
use thiserror::Error;

use crate::{
    pass::record_boundary_sizes::OperationWithBoundarySizes, report::TheoryTermMap,
    runtime::RuntimeModule,
};

pub use crate::runtime::GpuDialect;

type CodegenTermMap = TheoryTermMap<OperationWithBoundarySizes<Operation>>;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CodegenKind {
    #[default]
    Default,
    Experimental,
}

impl CodegenKind {
    pub(crate) fn ensure_available(self) -> Result<(), CodegenError> {
        match self {
            Self::Default => Ok(()),
            Self::Experimental => Err(experimental::CodegenError::NotImplemented.into()),
        }
    }
}

/// Backend-specific output. Experimental will add its own variant when its
/// lowering is implemented; it must not reuse default output implicitly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GeneratedModules {
    Default(default::GpuModuleMap),
}

impl GeneratedModules {
    pub fn kind(&self) -> CodegenKind {
        match self {
            Self::Default(_) => CodegenKind::Default,
        }
    }
}

#[derive(Debug, Error)]
pub enum CodegenError {
    #[error(transparent)]
    Default(#[from] default::CodegenError),
    #[error(transparent)]
    Experimental(#[from] experimental::CodegenError),
}

pub(crate) fn generate_modules(
    kind: CodegenKind,
    terms: &CodegenTermMap,
) -> Result<GeneratedModules, CodegenError> {
    match kind {
        CodegenKind::Default => Ok(GeneratedModules::Default(default::generate_modules(terms)?)),
        CodegenKind::Experimental => Err(experimental::CodegenError::NotImplemented.into()),
    }
}

/// Compile lowered terms and render them for the selected GPU dialect.
pub fn codegen(
    kind: CodegenKind,
    terms: &CodegenTermMap,
    dialect: GpuDialect,
) -> Result<RuntimeModule, CodegenError> {
    runtime_module(&generate_modules(kind, terms)?, dialect)
}

/// Render using the backend which produced these modules.
pub fn runtime_module(
    modules: &GeneratedModules,
    dialect: GpuDialect,
) -> Result<RuntimeModule, CodegenError> {
    match modules {
        GeneratedModules::Default(modules) => default::runtime_module(modules, dialect)
            .map_err(default::CodegenError::from)
            .map_err(CodegenError::from),
    }
}
