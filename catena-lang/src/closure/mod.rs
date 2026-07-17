//! Closure conversion over graphs produced by `forget_closures`.
//!
//! The conversion is deliberately split into three stages: discover a delimited
//! control-flow region, turn that region into a definition, and replace the
//! original region with an explicit environment and function pointer.

use hexpr::Operation;
use metacat::theory::{Theory, TheorySet};
use thiserror::Error;

use crate::{
    check::{CheckError, DefinitionTypes, PartialDefinitionTypes, partial_definition_types},
    pass::forget_closures::ClosureForgotten,
    report::TheoryTermMap,
    stdlib::constants::FN_HOM_TYPE,
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
}

/// Closure-convert graphs produced by `forget_closures` as one compiler pass.
///
/// Region discovery, generated-arrow construction, validation, and replacement
/// remain separate implementation modules, but callers receive one coherent
/// result which preserves every useful intermediate representation.
pub fn run(
    theory_set: &TheorySet,
    forgotten: &TheoryTermMap<ClosureForgotten<Operation>>,
) -> Result<Conversion, ConversionError> {
    assert_closure_boundary_definitions_are_inlined(theory_set);

    let regions = region::run(forgotten)?;
    let definition::DefinedClosures {
        generated_theory,
        generated_functions,
        closure_contexts,
    } = definition::run(theory_set, forgotten, &regions)?;
    let generated_types = crate::check::check(&generated_theory).map_err(|error| {
        ConversionError::CheckDefinitions {
            partial_definition_types: partial_definition_types(&error),
            error,
        }
    })?;
    let replacement = replace::run(
        &generated_theory,
        forgotten,
        &generated_functions,
        &regions,
        &closure_contexts,
    )?;
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

/// Region discovery assumes that calls to definitions with closure-typed
/// interfaces have already been expanded by the early inlining pass.
fn assert_closure_boundary_definitions_are_inlined(theory_set: &TheorySet) {
    for (theory_id, theory) in &theory_set.theories {
        let Theory::Theory { arrows, .. } = theory else {
            continue;
        };

        for (definition_name, arrow) in arrows {
            if arrow.definition.is_none() {
                continue;
            }

            assert!(
                !contains_closure(&arrow.type_maps.0) && !contains_closure(&arrow.type_maps.1),
                "closure conversion requires closure-boundary definitions to be inlined first; `{theory_id}.{definition_name}` still has a closure on its global interface"
            );
        }
    }
}

fn contains_closure(type_map: &metacat::theory::Term) -> bool {
    type_map
        .hypergraph
        .edges
        .iter()
        .any(|operation| operation.as_str() == FN_HOM_TYPE)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use metacat::theory::{RawTheorySet, TheorySet};

    /// This is an internal stage invariant rather than a compile integration
    /// case: callers may not skip closure-boundary inlining before conversion.
    #[test]
    #[should_panic(
        expected = "closure conversion requires closure-boundary definitions to be inlined first"
    )]
    fn rejects_uninlined_closure_boundary_definitions() {
        let source = r#"
            (def program returns-closure :
              (bool val) -> ({1 (bool val)} =>)
            = ([captured.] ([.captured] defer)))
        "#;
        let raw = RawTheorySet::from_texts(crate::stdlib::sources().chain([source]))
            .expect("test theories should parse");
        let elaborated = crate::elaborate::elaborate(raw).expect("test theory should elaborate");
        let theory_set = TheorySet::from_raw(elaborated).expect("test theory should interpret");

        super::run(&theory_set, &BTreeMap::new())
            .expect("the inlining invariant should panic before conversion");
    }
}
