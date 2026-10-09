mod progress;

pub(crate) use progress::timed;
pub use progress::{ProgressEvent, StageStatus, StageTiming};

use std::collections::{BTreeMap, BTreeSet};

use hexpr::Operation;
use metacat::theory::{Theory, TheoryId, TheorySet};
use thiserror::Error;

use crate::{
    check::{CheckError, partial_definition_types},
    closure::ConversionError,
    codegen::{CodegenError, CodegenKind, GpuDialect},
    elaborate::ElaborateError,
    pass::{
        PassError, forget_closures::ForgetClosuresError, inline_definitions::InlineDefinitionsError,
    },
    report::CompileReport,
    runtime::RuntimeModule,
};

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
    InlineDefinitions(#[from] InlineDefinitionsError),
    #[error(transparent)]
    ForgetClosures(#[from] ForgetClosuresError),
    #[error(transparent)]
    ClosureConversion(#[from] ConversionError),
    #[error(transparent)]
    Pass(#[from] PassError),
    #[error(transparent)]
    Codegen(#[from] CodegenError),
}

/// Compile one runtime module for the selected backend and dialect.
/// The caller retains pass diagnostics in `report`, including on failure.
pub fn compile(
    report: &mut CompileReport,
    codegen: CodegenKind,
    dialect: GpuDialect,
) -> Result<RuntimeModule, CompileError> {
    compile_with_progress(report, codegen, dialect, &mut |_| {})
}

/// Compile with live stage notifications. Timings are retained even without a callback.
pub fn compile_with_progress(
    report: &mut CompileReport,
    codegen: CodegenKind,
    dialect: GpuDialect,
    progress: &mut dyn FnMut(ProgressEvent),
) -> Result<RuntimeModule, CompileError> {
    report.timings.clear();
    let elaborated = timed(&mut report.timings, "Elaboration", progress, |_| {
        crate::elaborate::elaborate(report.raw_theories.clone())
    })?;
    report.elaborated = Some(elaborated.clone());

    let theory_set = timed(&mut report.timings, "Interpret theories", progress, |_| {
        TheorySet::from_raw(elaborated)
    })?;
    report.theory_set = Some(theory_set.clone());

    // check is a special case pass; we catch the 'partial' check error and add a partial-check
    // diagram to output
    let definition_types = match timed(&mut report.timings, "Typecheck", progress, |_| {
        crate::check::check(&theory_set)
    }) {
        Ok(definition_types) => definition_types,
        Err(error) => {
            report.partial_definition_types = partial_definition_types(&error);
            return Err(error.into());
        }
    };
    report.definition_types = Some(definition_types);

    let theory_set = timed(&mut report.timings, "Inline definitions", progress, |_| {
        let definitions_to_inline = closure_boundary_definitions(&theory_set);
        crate::pass::inline_definitions::run(&theory_set, &definitions_to_inline)
    })?;
    report.theory_set = Some(theory_set.clone());

    let definition_types = match timed(
        &mut report.timings,
        "Typecheck after inlining",
        progress,
        |_| crate::check::check(&theory_set),
    ) {
        Ok(definition_types) => definition_types,
        Err(error) => {
            report.partial_definition_types = partial_definition_types(&error);
            return Err(error.into());
        }
    };
    report.definition_types = Some(definition_types.clone());

    // Compute out closures by bending wires
    let forgotten_closures = timed(&mut report.timings, "Forget closures", progress, |_| {
        crate::pass::forget_closures::run(&theory_set, &definition_types)
    })?;
    report.forgotten_closures = Some(forgotten_closures.clone());

    let closure_conversion = timed(
        &mut report.timings,
        "Closure conversion",
        progress,
        |update| {
            crate::closure::run_with_progress(&theory_set, &forgotten_closures, codegen, update)
        },
    )?;
    report.closure_conversion = Some(closure_conversion);

    let converted_terms = &report
        .closure_conversion
        .as_ref()
        .expect("closure conversion was just recorded")
        .runtime_functions;
    let boundary_sizes = timed(
        &mut report.timings,
        "Record boundary sizes",
        progress,
        |_| crate::pass::record_boundary_sizes::run(converted_terms),
    )?;
    report.boundary_sizes = Some(boundary_sizes.clone());

    let unpacked_products = timed(&mut report.timings, "Unpack products", progress, |_| {
        crate::pass::unpack_products::run(&boundary_sizes)
    })?;
    report.unpacked_products = Some(unpacked_products.clone());

    Ok(timed(&mut report.timings, "Codegen", progress, |_| {
        crate::codegen::codegen(codegen, &unpacked_products, dialect)
    })?)
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
