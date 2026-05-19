use metacat::theory::{RawTheorySet, TheorySet, ast::ParseRawError};
use thiserror::Error;

mod abi;
mod domain;
mod render;

pub use abi::CudaAbiError;
use domain::CudaTarget;

use crate::{
    check::check as typecheck_elaborated,
    compile::{
        CompileConfig, CompileGraphError, GraphCompileOptions, Program, compile_graph,
        normalize::{NormalizeGraphError, normalize_graph},
        program::{ProgramCompileError, compile_program_from_graph},
        structured::{StructuredCompileError, compile_structured_program},
    },
    elaborate::elaborate,
    structured::ir::StructuredProgram,
};

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
    compile_cuda_theory_set_with_options(theory_set, theory, entry, GraphCompileOptions::default())
}

pub fn compile_cuda_theory_set_with_options(
    theory_set: &TheorySet,
    theory: &str,
    entry: &str,
    graph_options: GraphCompileOptions,
) -> Result<String, CudaCompileError> {
    let compile_graph = compile_graph(
        theory_set,
        &CompileConfig::data_control(),
        theory,
        entry,
        graph_options,
    )?;
    let graph = normalize_graph(&compile_graph)?;
    let program = compile_program_from_graph(&graph)?;
    let structured = compile_structured_program(&program)?;
    Ok(render_cuda_source(theory_set, &program, &structured)?)
}

pub fn render_cuda_source(
    theory_set: &TheorySet,
    program: &Program,
    structured: &StructuredProgram,
) -> Result<String, CudaAbiError> {
    let target = CudaTarget::new(theory_set, program.entry_definition())?;
    Ok(target.render_cuda_with_launch(structured))
}
