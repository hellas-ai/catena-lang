//! Adapter from catena-gpu's compiler to catena-lang's shared runtime.

use catena_lang::runtime::GeneratedFunction;
use metacat::theory::RawTheorySet;
use thiserror::Error;

use crate::codegen::{
    GpuModuleMap,
    gpu::{GpuRenderError, render_modules},
    lower_types::CType,
    runtime_type,
};

pub use catena_lang::runtime::{
    Artifact, ArtifactError, ExecError, InitError, MemError, MemOwn, MemRef, Runtime, Value,
    ValueKind,
};

#[derive(Debug, Error)]
pub enum LoadError {
    #[error(transparent)]
    Parse(#[from] metacat::theory::ast::ParseRawError),
    #[error(transparent)]
    Compile(#[from] crate::compile::CompileFailure),
    #[error("compile report did not contain GPU modules")]
    MissingModules,
    #[error(transparent)]
    Render(#[from] GpuRenderError),
    #[error(transparent)]
    Runtime(#[from] InitError),
}

/// Compile Catena GPU sources and load them into catena-lang's runtime.
pub fn load_sources<'a>(
    runtime: &mut Runtime,
    sources: impl IntoIterator<Item = &'a str>,
) -> Result<Artifact, LoadError> {
    let raw = RawTheorySet::from_texts(sources)?;
    let report = crate::compile::compile(raw)?;
    let modules = report
        .gpu_modules
        .as_ref()
        .ok_or(LoadError::MissingModules)?;
    let rendered = render_modules(modules, runtime.dialect())?;
    Ok(runtime.load_generated_source(&rendered, signatures(modules))?)
}

fn signatures(modules: &GpuModuleMap) -> Vec<GeneratedFunction> {
    modules
        .values()
        .filter_map(|module| {
            let source_name = module.source_name.as_ref()?;
            let inputs = module
                .entry
                .sources
                .iter()
                .map(|var| value_kind(runtime_type(var)?))
                .collect::<Option<Vec<_>>>()?;
            let outputs = module
                .entry
                .targets
                .iter()
                .map(|var| value_kind(runtime_type(var)?))
                .collect::<Option<Vec<_>>>()?;
            Some(GeneratedFunction {
                source_name: source_name.to_string(),
                symbol: module.name.clone(),
                inputs,
                outputs,
            })
        })
        .collect()
}

fn value_kind(ty: &CType) -> Option<ValueKind> {
    match ty {
        CType::Bool => Some(ValueKind::Bool),
        CType::U32 => Some(ValueKind::U32),
        CType::U64 => Some(ValueKind::U64),
        CType::F32 => Some(ValueKind::F32),
        CType::MemOwn => Some(ValueKind::MemOwn),
        CType::Grid
        | CType::Ptr(_)
        | CType::Generic(_)
        | CType::Ix
        | CType::Thread
        | CType::Block
        | CType::Scheduling => None,
    }
}
