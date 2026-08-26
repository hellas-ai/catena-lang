use metacat::theory::{RawTheorySet, TheorySet};
use thiserror::Error;

use crate::{
    check::CheckError, codegen::CodegenError, elaborate::ElaborateError, report::CompileReport,
};

#[derive(Debug, Error)]
#[error("{cause}")]
pub struct CompileFailure {
    pub report: CompileReport,
    #[source]
    pub cause: CompileError,
}

#[derive(Debug, Error)]
pub enum CompileError {
    #[error(transparent)]
    Elaborate(#[from] ElaborateError),
    #[error(transparent)]
    Load(#[from] metacat::theory::LoadError),
    #[error(transparent)]
    Check(#[from] CheckError),
    #[error(transparent)]
    Codegen(#[from] CodegenError),
}

/// Compile without partial application or any closure-related pass.
pub fn compile(raw_theories: RawTheorySet) -> Result<CompileReport, CompileFailure> {
    let mut report = CompileReport::new(raw_theories);
    if let Err(cause) = compile_into(&mut report) {
        return Err(CompileFailure { report, cause });
    }
    Ok(report)
}

fn compile_into(report: &mut CompileReport) -> Result<(), CompileError> {
    let elaborated = crate::elaborate::elaborate(report.raw_theories.clone())?;
    report.elaborated = Some(elaborated.clone());

    let theory_set = TheorySet::from_raw(elaborated)?;
    report.theory_set = Some(theory_set.clone());

    let types = crate::check::check(&theory_set)?;
    let typed = crate::check::typed_definitions(&theory_set, &types)?;
    report.definition_types = Some(types);

    report.gpu_modules = Some(crate::codegen::codegen(&typed)?);
    Ok(())
}
