//! Closure conversion over graphs produced by `forget_closures`.
//!
//! The conversion is deliberately split into three stages: discover a delimited
//! control-flow region, turn that region into a definition, and replace the
//! original region with an explicit environment and function pointer.

use hexpr::Operation;
use metacat::theory::TheorySet;
use thiserror::Error;

use crate::{
    check::{CheckError, DefinitionTypes, PartialDefinitionTypes, partial_definition_types},
    pass::forget_closures::Region,
    report::TheoryTermMap,
};

/// Find regions by following closure domains to their codomains.
pub mod region;

/// Turn discovered regions into `closure.*` definitions and `name.closure.*` declarations.
pub mod definition;

mod context;
/// Replace regions with explicit environments, function pointers, and context operations.
pub mod replace;

/// Complete output of closure conversion, including its debugging snapshots.
#[derive(Debug, Clone)]
pub struct Conversion {
    /// Closure-forgotten graph on which conversion operates.
    pub closure_forgotten_definitions: TheoryTermMap<Region<Operation>>,
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
}

/// Closure-convert graphs produced by `forget_closures` as one compiler pass.
///
/// Region discovery, generated-arrow construction, validation, and replacement
/// remain separate implementation modules, but callers receive one coherent
/// result which preserves every useful intermediate representation.
pub fn run(
    theory_set: &TheorySet,
    forgotten: &TheoryTermMap<Region<Operation>>,
) -> Result<Conversion, ConversionError> {
    let regions = region::run(forgotten)?;
    let definition::DefinedClosures {
        generated_theory,
        generated_functions,
        definitions,
    } = definition::run(theory_set, forgotten, &regions)?;
    let generated_types = crate::check::check(&generated_theory).map_err(|error| {
        ConversionError::CheckDefinitions {
            partial_definition_types: partial_definition_types(&error),
            error,
        }
    })?;
    let replacement = replace::run(&generated_theory, &definitions, &regions)?;
    let rewritten_definitions = replacement.terms;
    let runtime_functions = context::erase(&rewritten_definitions)?;

    Ok(Conversion {
        closure_forgotten_definitions: forgotten.clone(),
        regions,
        generated_theory,
        generated_types,
        generated_functions,
        rewritten_definitions,
        runtime_functions,
        replacement_theory: replacement.theory_set,
    })
}
