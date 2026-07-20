//! Closure conversion over graphs produced by `forget_closures`.
//!
//! The conversion first specializes explicit named evaluations, then discovers
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
mod named_eval;
/// Replace regions with explicit environments, function pointers, and context operations.
pub mod replace;

/// Complete output of closure conversion.
#[derive(Debug, Clone)]
pub struct Conversion {
    /// Closure-forgotten graph after named-evaluation specialization.
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
    NamedEval(#[from] named_eval::NamedEvalError),
    #[error("closure conversion made no progress while {markers} closure regions remain")]
    NoSpliceableRegion { markers: usize },
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
    let mut working = named_eval::run(theory_set, forgotten)?;
    let closure_forgotten_definitions = working.clone();
    let mut discovered = region::run(&working)?;
    let regions = discovered.clone();
    let mut generated_theory = theory_set.clone();
    let mut generated_functions = TheoryTermMap::new();

    loop {
        let marker_count = discovered
            .values()
            .flat_map(|definitions| definitions.values())
            .map(Vec::len)
            .sum::<usize>();
        if marker_count == 0 {
            break;
        }

        let selected = spliceable_regions(&working, &discovered);
        let selected_count = selected
            .values()
            .flat_map(|definitions| definitions.values())
            .map(Vec::len)
            .sum::<usize>();
        if selected_count == 0 {
            return Err(ConversionError::NoSpliceableRegion {
                markers: marker_count,
            });
        }

        let defined = definition::run(&generated_theory, &working, &selected)?;
        generated_theory = defined.generated_theory;
        merge_terms(&mut generated_functions, defined.generated_functions);

        // Validate each layer as soon as its generated declarations exist.
        crate::check::check(&generated_theory).map_err(|error| {
            ConversionError::CheckDefinitions {
                partial_definition_types: partial_definition_types(&error),
                error,
            }
        })?;

        let partial = replace::run_partial(
            &generated_theory,
            &working,
            &selected,
            &defined.closure_contexts,
        )?;
        generated_theory = partial.theory_set;
        working = partial.terms;
        replace::rewrite_ready_converted_primitives(&mut working);
        discovered = region::run(&working)?;
    }

    let generated_types = crate::check::check(&generated_theory).map_err(|error| {
        ConversionError::CheckDefinitions {
            partial_definition_types: partial_definition_types(&error),
            error,
        }
    })?;
    let replacement = replace::run(
        &generated_theory,
        &working,
        &generated_functions,
        &discovered,
        &Default::default(),
    )?;
    let mut rewritten_definitions = replacement.terms;
    replace::rewrite_all_converted_primitives(&mut rewritten_definitions);
    let runtime_functions = context::erase(&rewritten_definitions)?;

    Ok(Conversion {
        closure_forgotten_definitions,
        regions,
        generated_theory,
        generated_types,
        generated_functions,
        rewritten_definitions,
        runtime_functions,
        replacement_theory: replacement.theory_set,
    })
}

fn spliceable_regions(
    definitions: &TheoryTermMap<ClosureForgotten<Operation>>,
    discovered: &region::ClosureRegionMap,
) -> region::ClosureRegionMap {
    discovered
        .iter()
        .filter_map(|(theory, theory_regions)| {
            let selected = theory_regions
                .iter()
                .filter_map(|(definition, regions)| {
                    let term = &definitions[theory][definition];
                    let regions = regions
                        .iter()
                        .filter(|region| replace::region_is_spliceable(term, region))
                        .cloned()
                        .collect::<Vec<_>>();
                    (!regions.is_empty()).then_some((definition.clone(), regions))
                })
                .collect::<std::collections::BTreeMap<_, _>>();
            (!selected.is_empty()).then_some((theory.clone(), selected))
        })
        .collect()
}

fn merge_terms(output: &mut TheoryTermMap, next: TheoryTermMap) {
    for (theory, definitions) in next {
        output.entry(theory).or_default().extend(definitions);
    }
}

#[cfg(test)]
mod tests {
    use metacat::theory::{RawTheorySet, TheorySet};

    #[test]
    fn inlines_named_closure_boundary_evaluation_after_forgetting() {
        let source = r#"
            (def program apply-closure : ({1 (bool val)} =>) -> (bool val)
              = run)
            (def program use-named-closure : (bool val) -> (bool val)
              = ([captured.]
                  ([.captured] defer [inner.]
                    ({([.inner] defer) (name.apply-closure lift)} compose run))))
        "#;
        let raw = RawTheorySet::from_texts(crate::stdlib::sources().chain([source]))
            .expect("test theories should parse");
        let elaborated = crate::elaborate::elaborate(raw).expect("test theory should elaborate");
        let theory_set = TheorySet::from_raw(elaborated).expect("test theory should interpret");
        let types = crate::check::check(&theory_set).expect("test theory should check");
        let forgotten = crate::pass::forget_closures::run(&theory_set, &types)
            .expect("test theory should forget closures");
        let program = metacat::theory::TheoryId("program".parse().unwrap());
        let use_named: hexpr::Operation = "use-named-closure".parse().unwrap();
        assert!(
            forgotten[&program][&use_named]
                .hypergraph
                .edges
                .iter()
                .any(|edge| matches!(
                    edge,
                    crate::pass::forget_closures::ClosureForgotten::NamedEval {
                        definition,
                        ..
                    } if definition.as_str() == "apply-closure"
                ))
        );

        let conversion = super::run(&theory_set, &forgotten)
            .expect("named closure-boundary evaluation should specialize");
        assert!(
            conversion.closure_forgotten_definitions[&program][&use_named]
                .hypergraph
                .edges
                .iter()
                .all(|edge| !matches!(
                    edge,
                    crate::pass::forget_closures::ClosureForgotten::NamedEval { .. }
                ))
        );
        assert!(
            !conversion.runtime_functions[&program].contains_key(&"apply-closure".parse().unwrap())
        );
        assert!(
            conversion.runtime_functions[&program]
                .contains_key(&"use-named-closure".parse().unwrap())
        );
    }
}
