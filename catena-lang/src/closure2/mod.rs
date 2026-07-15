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
    pub input: TheoryTermMap<Region<Operation>>,
    /// Regions discovered in the closure-forgotten input.
    pub regions: region::ClosureRegionMap,
    /// Theory after inserting the generated `closure.*` and `name.closure.*` arrows.
    pub definitions: TheorySet,
    /// Checked node labels for `definitions`, including generated closure bodies.
    pub definition_types: DefinitionTypes,
    /// Replacement graph before erasing context projections, retained for debugging.
    pub replacements: TheoryTermMap,
    /// Final context-free closure-converted definitions used by downstream passes.
    pub terms: TheoryTermMap,
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
    #[error("missing checked generated closure `{theory}.{definition}`")]
    MissingGeneratedClosureTypes { theory: String, definition: String },
    #[error("checked label count mismatch for generated closure `{theory}.{definition}`")]
    GeneratedClosureLabelCount { theory: String, definition: String },
    #[error("failed to quotient generated closure `{theory}.{definition}`: {error}")]
    GeneratedClosureQuotient {
        theory: String,
        definition: String,
        error: String,
    },
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
    let definitions = definition::run(theory_set, forgotten, &regions)?;
    let definition_types =
        crate::check::check(&definitions).map_err(|error| ConversionError::CheckDefinitions {
            partial_definition_types: partial_definition_types(&error),
            error,
        })?;
    let replacement = replace::run(&definitions, forgotten, &regions)?;
    let mut replacements = replacement.terms;

    for (theory_id, theory) in &definitions.theories {
        let metacat::theory::Theory::Theory { arrows, .. } = theory else {
            continue;
        };
        let checked = definition_types.get(theory_id);
        let output = replacements.entry(theory_id.clone()).or_default();
        for (operation, arrow) in arrows {
            if !operation.as_str().starts_with("closure.") {
                continue;
            }
            let Some(mut body) = arrow.definition.clone() else {
                continue;
            };
            body.quotient()
                .map_err(|error| ConversionError::GeneratedClosureQuotient {
                    theory: theory_id.to_string(),
                    definition: operation.to_string(),
                    error: format!("{error:?}"),
                })?;
            let labels = checked
                .and_then(|definitions| definitions.get(operation))
                .cloned()
                .ok_or_else(|| ConversionError::MissingGeneratedClosureTypes {
                    theory: theory_id.to_string(),
                    definition: operation.to_string(),
                })?;
            let body = body.with_nodes(|_| labels).ok_or_else(|| {
                ConversionError::GeneratedClosureLabelCount {
                    theory: theory_id.to_string(),
                    definition: operation.to_string(),
                }
            })?;
            output.insert(operation.clone(), body);
        }
    }

    let terms = context::erase(&replacements)?;

    Ok(Conversion {
        input: forgotten.clone(),
        regions,
        definitions,
        definition_types,
        replacements,
        terms,
        replacement_theory: replacement.theory_set,
    })
}
