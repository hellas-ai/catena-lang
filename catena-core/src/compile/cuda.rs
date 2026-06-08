use std::collections::HashMap;

use metacat::theory::{RawTheorySet, TheorySet, ast::ParseRawError};
use thiserror::Error;

mod abi;
mod boundary;
mod domain;
mod launch;
mod parameters;
mod render;
mod resources;
mod shape;
mod util;
mod views;

pub use abi::CudaAbiError;
use domain::CudaTarget;

use crate::{
    check::check as typecheck_elaborated,
    compile::{
        CompileConfig, CompileGraphError, Program, compile_graph,
        normalize::{NormalizeGraphError, normalize_graph},
        program::{ProgramCompileError, compile_program_from_graph},
        proof::ProofEvidence,
        structured::{StructuredCompileError, compile_structured_program},
    },
    elaborate::elaborate,
    structured::ir::StructuredProgram,
};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CudaOptions {
    pub static_values: HashMap<String, u64>,
}

#[derive(Debug, Error)]
pub enum CudaCompileError {
    #[error("failed to parse source: {0}")]
    Parse(#[from] ParseRawError),
    #[error("failed to elaborate source: {0}")]
    Elaborate(#[from] crate::elaborate::ElaborateError),
    #[error("failed to elaborate or typecheck source: {0}")]
    Check(#[from] crate::check::CheckError),
    #[error("failed to build compile graph: {0}")]
    CompileGraph(#[from] CompileGraphError),
    #[error(transparent)]
    Normalize(#[from] NormalizeGraphError),
    #[error(transparent)]
    Program(#[from] ProgramCompileError),
    #[error(transparent)]
    Structured(#[from] StructuredCompileError),
    #[error(transparent)]
    Abi(#[from] CudaAbiError),
}

pub fn compile_cuda_source(
    source: &str,
    theory: &str,
    entry: &str,
) -> Result<String, CudaCompileError> {
    let raw = RawTheorySet::from_text(source)?;
    let elaborated = elaborate(raw)?;
    let theory_set = typecheck_elaborated(&elaborated)?;
    compile_cuda_theory_set(&theory_set, theory, entry)
}

pub fn compile_cuda_theory_set(
    theory_set: &TheorySet,
    theory: &str,
    entry: &str,
) -> Result<String, CudaCompileError> {
    let compile_graph = compile_graph(theory_set, &CompileConfig::data_control(), theory, entry)?;
    let graph = normalize_graph(&compile_graph)?;
    let program = compile_program_from_graph(&graph)?;
    let structured = compile_structured_program(&program)?;
    Ok(render_cuda_source(
        theory_set,
        &program,
        &structured,
        &CudaOptions::default(),
        None,
    )?)
}

pub fn render_cuda_source(
    theory_set: &TheorySet,
    program: &Program,
    structured: &StructuredProgram,
    options: &CudaOptions,
    proof_evidence: Option<&ProofEvidence>,
) -> Result<String, CudaAbiError> {
    let target = CudaTarget::new(
        theory_set,
        program.entry_definition(),
        structured,
        options,
        proof_evidence,
    )?;
    Ok(target.render_cuda_with_launch(structured))
}
