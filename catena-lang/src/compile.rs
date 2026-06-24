use hexpr::Operation;
use metacat::theory::{RawTheorySet, Theory, TheorySet};
use open_hypergraphs::lax::OpenHypergraph;
use thiserror::Error;

use crate::{
    check::{CheckError, partial_definition_types},
    codegen::CodegenError,
    elaborate::ElaborateError,
    pass::{PassError, forget_closures::ForgetClosuresError},
    report::CompileReport,
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
    #[error(
        "definition `{theory}.{definition}` has closure type `=>` on its global interface; linear closure types are only allowed adjacent to CMC operations"
    )]
    ClosureOnGlobalInterface { theory: String, definition: String },
    #[error(transparent)]
    ForgetClosures(#[from] ForgetClosuresError),
    #[error(transparent)]
    Pass(#[from] PassError),
    #[error(transparent)]
    Codegen(#[from] CodegenError),
}

// TODO: Write a function `compile` which:
//
// - Elaborates input to include function names (finitary CMC)
// - Typechecks
// - Generates GPU codegen artifacts for each definition
// - Renders GPU source artifacts
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
pub fn compile(raw_theories: RawTheorySet) -> Result<CompileReport, CompileFailure> {
    let mut report = CompileReport::new(raw_theories);
    if let Err(cause) = compile_into(&mut report) {
        return Err(CompileFailure { report, cause });
    }
    Ok(report)
}

// Helper for `compile` which exists so `compile` can return
// `Result<CompileReport, CompileFailure>`
fn compile_into(report: &mut CompileReport) -> Result<(), CompileError> {
    let elaborated = crate::elaborate::elaborate(report.raw_theories.clone())?;
    report.elaborated = Some(elaborated.clone());

    let theory_set = TheorySet::from_raw(elaborated)?;
    report.theory_set = Some(theory_set.clone());

    // check is a special case pass; we catch the 'partial' check error and add a partial-check
    // diagram to output
    let definition_types = match crate::check::check(&theory_set) {
        Ok(definition_types) => definition_types,
        Err(error) => {
            report.partial_definition_types = partial_definition_types(&error);
            return Err(error.into());
        }
    };
    report.definition_types = Some(definition_types.clone());

    // don't allow `=>` types on global interfaces
    reject_closure_global_interfaces(&theory_set)?;

    let forgotten_closures = crate::pass::forget_closures::run(&theory_set, &definition_types)?;
    report.forgotten_closures = Some(forgotten_closures.clone());

    let boundary_sizes = crate::pass::record_boundary_sizes::run(&forgotten_closures)?;
    report.boundary_sizes = Some(boundary_sizes.clone());

    let unpacked_products = crate::pass::unpack_products::run(&boundary_sizes)?;
    report.unpacked_products = Some(unpacked_products.clone());

    let gpu_modules = crate::codegen::codegen(&unpacked_products)?;
    report.gpu_modules = Some(gpu_modules);

    Ok(())
}

fn reject_closure_global_interfaces(theory_set: &TheorySet) -> Result<(), CompileError> {
    for (theory_id, theory) in &theory_set.theories {
        let Theory::Theory { arrows, .. } = theory else {
            continue;
        };

        for (definition_name, arrow) in arrows {
            if arrow.definition.is_none() {
                continue;
            }

            if contains_closure_type_map(&arrow.type_maps.0)
                || contains_closure_type_map(&arrow.type_maps.1)
            {
                return Err(CompileError::ClosureOnGlobalInterface {
                    theory: theory_id.to_string(),
                    definition: definition_name.to_string(),
                });
            }
        }
    }

    Ok(())
}

fn contains_closure_type_map(type_map: &OpenHypergraph<(), Operation>) -> bool {
    type_map
        .hypergraph
        .edges
        .iter()
        .any(|op| op.as_str() == "=>")
}
