use std::collections::{BTreeMap, BTreeSet};

use hexpr::Operation;
use metacat::theory::{RawTheorySet, Theory, TheoryId, TheorySet};
use thiserror::Error;

use crate::{
    check::{CheckError, partial_definition_types},
    closure::theory::ConvertTheoryError,
    codegen::CodegenError,
    elaborate::ElaborateError,
    pass::{
        PassError, forget_closures::ForgetClosuresError, inline_definitions::InlineDefinitionsError,
    },
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
    #[error(transparent)]
    ClosureConversion(#[from] ConvertTheoryError),
    #[error(transparent)]
    InlineDefinitions(#[from] InlineDefinitionsError),
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

    let definitions_to_inline = closure_boundary_definitions(&theory_set);
    let theory_set = crate::pass::inline_definitions::run(&theory_set, &definitions_to_inline)?;
    report.theory_set = Some(theory_set.clone());

    let definition_types = match crate::check::check(&theory_set) {
        Ok(definition_types) => definition_types,
        Err(error) => {
            report.partial_definition_types = partial_definition_types(&error);
            return Err(error.into());
        }
    };
    report.definition_types = Some(definition_types.clone());

    let theory_set = convert_closures(&theory_set, &definition_types)?;
    report.theory_set = Some(theory_set.clone());

    // TODO: we have to re-check because convert_closures adds new arrows to the theory.
    // We need the full set of `definition_types`, so it's simpler just to re-check everything.
    // But this is kinda slow! We only really need to *incrementally* check anything that changed.
    let definition_types = match crate::check::check(&theory_set) {
        Ok(definition_types) => definition_types,
        Err(error) => {
            report.partial_definition_types = partial_definition_types(&error);
            return Err(error.into());
        }
    };
    report.definition_types = Some(definition_types.clone());

    // Compute out closures by bending wires
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

fn convert_closures(
    theory_set: &TheorySet,
    definition_types: &crate::check::DefinitionTypes,
) -> Result<TheorySet, CompileError> {
    let theory_ids = theory_set
        .theories
        .iter()
        .filter_map(|(theory_id, theory)| {
            matches!(theory, Theory::Theory { .. }).then_some(theory_id.clone())
        })
        .collect::<Vec<TheoryId>>();

    let mut converted = theory_set.clone();
    for theory_id in theory_ids {
        let theory =
            crate::closure::theory::convert_theory(theory_set, definition_types, &theory_id)?;
        converted.theories.insert(theory_id, theory);
    }
    Ok(converted)
}

fn closure_boundary_definitions(theory_set: &TheorySet) -> BTreeMap<TheoryId, BTreeSet<Operation>> {
    let mut output = BTreeMap::new();

    for (theory_id, theory) in &theory_set.theories {
        let Theory::Theory { arrows, .. } = theory else {
            continue;
        };

        let definitions = arrows
            .iter()
            .filter_map(|(definition_name, arrow)| {
                arrow.definition.as_ref()?;
                (contains_closure_type_map(&arrow.type_maps.0)
                    || contains_closure_type_map(&arrow.type_maps.1))
                .then_some(definition_name.clone())
            })
            .collect::<BTreeSet<_>>();

        if !definitions.is_empty() {
            output.insert(theory_id.clone(), definitions);
        }
    }

    output
}

fn contains_closure_type_map(type_map: &metacat::theory::Term) -> bool {
    type_map
        .hypergraph
        .edges
        .iter()
        .any(|op| op.as_str() == crate::stdlib::constants::FN_HOM_TYPE)
}
