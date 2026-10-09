//! Closure conversion over graphs produced by `forget_closures`.
//!
//! The conversion first inlines named calls with closure-bearing interfaces, then discovers
//! delimited control-flow regions, turns them into definitions, and replaces
//! them with explicit environments and function pointers.

use hexpr::Operation;
use metacat::theory::TheorySet;
use thiserror::Error;

use crate::{
    check::{CheckError, DefinitionTypes, PartialDefinitionTypes, partial_definition_types},
    pass::forget_closures::ClosureForgotten,
    report::TheoryTermMap,
};

/// Find regions by following closure domains to their codomains.
pub mod region;

/// Turn discovered regions into `closure.*` definitions and `name.closure.*` declarations.
pub mod definition;

mod context;
mod inline_named_calls;
mod region_conversion;
/// Replace regions with explicit environments, function pointers, and context operations.
pub mod replace;
mod schedule;

/// Complete output of closure conversion.
#[derive(Debug, Clone)]
pub struct Conversion {
    /// Closure-forgotten graph after closure-bearing named calls are inlined.
    pub closure_forgotten_definitions: TheoryTermMap<ClosureForgotten<Operation>>,
    /// Regions discovered in the closure-forgotten input.
    pub regions: region::ClosureRegionMap,
    /// Theory after inserting the generated `closure.*` and `name.closure.*` arrows.
    pub generated_theory: TheorySet,
    /// Independently checked node labels for `generated_theory`.
    pub generated_types: DefinitionTypes,
    /// Typed runtime functions cut out of the discovered regions.
    pub generated_functions: TheoryTermMap,
    /// Replacement graph before erasing context projections, retained for debugging.
    pub rewritten_definitions: TheoryTermMap,
    /// Final context-free closure-converted definitions used by downstream passes.
    pub runtime_functions: TheoryTermMap,
    /// Debug theory containing replaced definitions and context declarations.
    pub replacement_theory: TheorySet,
}

#[derive(Debug, Error)]
pub enum ConversionError {
    #[error(transparent)]
    FindRegions(#[from] region::FindRegionError),
    #[error(transparent)]
    DefineClosures(#[from] definition::DefineClosuresError),
    #[error("generated closure definition check failed: {error}")]
    CheckDefinitions {
        partial_definition_types: Option<PartialDefinitionTypes>,
        #[source]
        error: CheckError,
    },
    #[error(transparent)]
    ReplaceClosures(#[from] replace::ReplaceClosuresError),
    #[error(transparent)]
    EraseContexts(#[from] context::EraseContextsError),
    #[error(transparent)]
    InlineNamedCalls(#[from] inline_named_calls::InlineNamedCallsError),
}

/// Closure-convert graphs produced by `forget_closures` as one compiler pass.
///
/// Region discovery, generated-arrow construction, validation, and replacement
/// remain separate implementation modules, but callers receive one coherent
/// result which preserves every useful intermediate representation.
/// This entry point uses default stdlib primitives; the compiler selects the
/// experimental primitive policy explicitly through `run_with_progress`.
pub fn run(
    theory_set: &TheorySet,
    forgotten: &TheoryTermMap<ClosureForgotten<Operation>>,
) -> Result<Conversion, ConversionError> {
    run_with_progress(
        theory_set,
        forgotten,
        crate::codegen::CodegenKind::Default,
        &mut |_| {},
    )
}

pub(crate) fn run_with_progress(
    theory_set: &TheorySet,
    forgotten: &TheoryTermMap<ClosureForgotten<Operation>>,
    codegen: crate::codegen::CodegenKind,
    progress: &mut dyn FnMut(&str),
) -> Result<Conversion, ConversionError> {
    progress("Inlining named calls");
    // Forgetting exposes `name.f -> eval`. Inline the complete call adapter
    // when `f` has closures on its interface, before discovering regions.
    let inlined_definitions = inline_named_calls::run(theory_set, forgotten)?;
    let closure_forgotten_definitions = inlined_definitions.clone();

    // Both stdlibs declare their converted interfaces; only the name table differs.
    let primitives = match codegen {
        crate::codegen::CodegenKind::Default => replace::CONVERTED_PRIMITIVES,
        // TODO stdlib also might define callbacks without val wrapper
        crate::codegen::CodegenKind::Experimental => {
            crate::codegen::experimental::primitives::CONVERTED_PRIMITIVES
        }
    };
    let converted = region_conversion::run(theory_set, inlined_definitions, primitives, progress)?;
    let region_conversion::RegionConversion {
        terms: working,
        initial_regions: regions,
        theory: generated_theory,
        generated_functions,
    } = converted;

    // Validate the completed generated theory, finish primitive rewriting, and
    // erase compile-time context projections.
    progress("Validating completed generated theory");
    let generated_types = crate::check::check(&generated_theory).map_err(|error| {
        ConversionError::CheckDefinitions {
            partial_definition_types: partial_definition_types(&error),
            error,
        }
    })?;
    progress("Rewriting definitions and erasing contexts");
    let rewritten_definitions =
        replace::build_rewritten_definitions(&working, &generated_functions, primitives)?;
    let runtime_functions = context::erase(&rewritten_definitions)?;
    let replacement_theory = generated_theory.clone();

    Ok(Conversion {
        closure_forgotten_definitions,
        regions,
        generated_theory,
        generated_types,
        generated_functions,
        rewritten_definitions,
        runtime_functions,
        replacement_theory,
    })
}
