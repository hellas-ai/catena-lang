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
    // Phase 1: statically named calls are ordinary graph substitutions. This
    // happens before, and independently of, closure-region discovery.
    let specialized = named_eval::run(theory_set, forgotten)?;
    let closure_forgotten_definitions = specialized.clone();

    // Phase 2: discover and replace the actual ClosureMarker regions.
    let converted = convert_regions(theory_set, specialized)?;
    let working = converted.terms;
    let regions = converted.initial_regions;
    let final_regions = converted.final_regions;
    let generated_theory = converted.theory;
    let generated_functions = converted.generated_functions;

    // Phase 3: validate the completed generated theory, finish primitive
    // rewriting, and erase compile-time context projections.
    let generated_types = crate::check::check(&generated_theory).map_err(|error| {
        ConversionError::CheckDefinitions {
            partial_definition_types: partial_definition_types(&error),
            error,
        }
    })?;
    let no_closure_contexts = definition::ClosureContextMap::new();
    let replacement = replace::run(
        &generated_theory,
        &working,
        &generated_functions,
        &final_regions,
        &no_closure_contexts,
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

struct RegionConversion {
    terms: TheoryTermMap<ClosureForgotten<Operation>>,
    initial_regions: region::ClosureRegionMap,
    /// Contains no regions, but retains the theory/definition keys expected by
    /// the final replacement pass.
    final_regions: region::ClosureRegionMap,
    theory: TheorySet,
    generated_functions: TheoryTermMap,
}

fn convert_regions(
    theory_set: &TheorySet,
    mut terms: TheoryTermMap<ClosureForgotten<Operation>>,
) -> Result<RegionConversion, ConversionError> {
    let mut theory = theory_set.clone();
    let mut generated_functions = TheoryTermMap::new();
    let mut discovered = region::run(&terms)?;
    let initial_regions = discovered.clone();

    loop {
        let remaining = region_count(&discovered);
        if remaining == 0 {
            break;
        }

        // Only regions whose internals no longer contain another marker can be
        // cut out safely. Re-discovery after each round exposes the next layer.
        let selected = spliceable_regions(&terms, &discovered);
        if region_count(&selected) == 0 {
            return Err(ConversionError::NoSpliceableRegion { markers: remaining });
        }

        let defined = definition::run(&theory, &terms, &selected)?;
        theory = defined.generated_theory;
        merge_terms(&mut generated_functions, defined.generated_functions);

        // Each generated layer must typecheck before its declarations are used
        // to replace the selected markers.
        crate::check::check(&theory).map_err(|error| ConversionError::CheckDefinitions {
            partial_definition_types: partial_definition_types(&error),
            error,
        })?;

        let partial = replace::run_partial(&theory, &terms, &selected, &defined.closure_contexts)?;
        theory = partial.theory_set;
        terms = partial.terms;
        replace::rewrite_ready_converted_primitives(&mut terms);
        discovered = region::run(&terms)?;
    }

    Ok(RegionConversion {
        terms,
        initial_regions,
        final_regions: discovered,
        theory,
        generated_functions,
    })
}

fn region_count(regions: &region::ClosureRegionMap) -> usize {
    regions
        .values()
        .flat_map(|definitions| definitions.values())
        .map(Vec::len)
        .sum()
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
