use metacat::theory::{RawTheorySet, TheorySet};
use thiserror::Error;

use crate::{
    check::CheckError,
    codegen::CodegenError,
    elaborate::ElaborateError,
    pass::forget_closures::ForgetClosuresError,
    report::CompileReport,
};

#[derive(Debug, Error)]
pub enum CompileError {
    #[error(transparent)]
    Elaborate(#[from] ElaborateError),
    #[error(transparent)]
    Load(#[from] metacat::theory::LoadError),
    #[error(transparent)]
    Check(#[from] CheckError),
    #[error(transparent)]
    ForgetClosures(#[from] ForgetClosuresError),
    #[error(transparent)]
    Codegen(#[from] CodegenError),
}

// TODO: Write a function `compile` which:
//
// - Elaborates input to include function names (finitary CMC)
// - Typechecks
// - Generates a `StructuredProgram` for each definition
// - Renders as a single CUDA file
// - Produces a CompileReport which contains all intermediate data, including graphs rendered with
//   open-hypergraphs-dot for each definition + the result of each pass.
//
// NOTE: *definitions* will never be inlined.
//
// At each stage, write debug output to an (optionally supplied) directory.
// Choose meaningful names for each file; render SVGs of terms where possible.
// Provide a top-level HTML file
//
// This should

/// Compile all definitions from the input raw theories and collect intermediate data.
pub fn compile(raw_theories: RawTheorySet) -> Result<CompileReport, CompileError> {
    let elaborated = crate::elaborate::elaborate(raw_theories.clone())?;
    let theory_set = TheorySet::from_raw(elaborated.clone())?;
    let definition_types = crate::check::check(&theory_set)?;
    let forgotten_closures = crate::pass::forget_closures::run(&theory_set, &definition_types)?;
    let structured_programs = crate::codegen::codegen(&forgotten_closures)?;
    Ok(CompileReport {
        raw_theories,
        elaborated,
        theory_set,
        definition_types,
        forgotten_closures,
        structured_programs,
    })
}
