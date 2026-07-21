//! Bottom-up conversion of `ClosureMarker` regions.
//!
//! A conversion round extracts every region that is currently self-contained.
//! Nested enclosing regions are rediscovered and handled by later rounds.

use std::collections::BTreeMap;

use hexpr::Operation;
use metacat::theory::TheorySet;

use super::{ConversionError, definition, region, replace};
use crate::{
    check::partial_definition_types, pass::forget_closures::ClosureForgotten, report::TheoryTermMap,
};

pub(super) struct RegionConversion {
    pub(super) terms: TheoryTermMap<ClosureForgotten<Operation>>,
    pub(super) initial_regions: region::ClosureRegionMap,
    pub(super) theory: TheorySet,
    pub(super) generated_functions: TheoryTermMap,
}

struct ConversionState {
    terms: TheoryTermMap<ClosureForgotten<Operation>>,
    theory: TheorySet,
    generated_functions: TheoryTermMap,
}

pub(super) fn run(
    theory_set: &TheorySet,
    terms: TheoryTermMap<ClosureForgotten<Operation>>,
) -> Result<RegionConversion, ConversionError> {
    let mut state = ConversionState {
        terms,
        theory: theory_set.clone(),
        generated_functions: TheoryTermMap::new(),
    };
    let mut discovered_regions = region::run(&state.terms)?;
    let initial_regions = discovered_regions.clone();

    while region_count(&discovered_regions) != 0 {
        // Step 1: choose the innermost/self-contained regions that can safely
        // be removed from the current graph.
        let ready_regions = regions_ready_for_extraction(&state.terms, &discovered_regions);
        require_progress(&discovered_regions, &ready_regions)?;

        // Step 2: generate and validate functions for this ready layer.
        let closure_contexts = define_ready_layer(&mut state, &ready_regions)?;
        validate_generated_theory(&state.theory)?;

        // Step 3: replace its markers with explicit
        // environment/function-pointer values.
        replace_ready_layer(&mut state, &ready_regions, &closure_contexts)?;

        // Step 4: discover the next enclosing layer in the rewritten graph.
        //
        // We intentionally do not build and maintain a region nesting tree.
        // Extraction deletes and unifies nodes, renumbers graph identifiers,
        // and can change an enclosing region's body and environment. Updating
        // a saved tree through those mutations is more complex and fragile
        // than rediscovering regions from the new graph.
        discovered_regions = region::run(&state.terms)?;
    }

    Ok(RegionConversion {
        terms: state.terms,
        initial_regions,
        theory: state.theory,
        generated_functions: state.generated_functions,
    })
}

fn regions_ready_for_extraction(
    definitions: &TheoryTermMap<ClosureForgotten<Operation>>,
    discovered_regions: &region::ClosureRegionMap,
) -> region::ClosureRegionMap {
    discovered_regions
        .iter()
        .filter_map(|(theory, theory_regions)| {
            let ready_definitions = theory_regions
                .iter()
                .filter_map(|(definition, regions)| {
                    let term = &definitions[theory][definition];
                    let ready_regions = regions
                        .iter()
                        .filter(|region| replace::region_is_ready_for_extraction(term, region))
                        .cloned()
                        .collect::<Vec<_>>();
                    (!ready_regions.is_empty()).then_some((definition.clone(), ready_regions))
                })
                .collect::<BTreeMap<_, _>>();
            (!ready_definitions.is_empty()).then_some((theory.clone(), ready_definitions))
        })
        .collect()
}

fn require_progress(
    discovered_regions: &region::ClosureRegionMap,
    ready_regions: &region::ClosureRegionMap,
) -> Result<(), ConversionError> {
    if region_count(ready_regions) == 0 {
        return Err(ConversionError::NoRegionReadyForExtraction {
            markers: region_count(discovered_regions),
        });
    }
    Ok(())
}

fn define_ready_layer(
    state: &mut ConversionState,
    ready_regions: &region::ClosureRegionMap,
) -> Result<definition::ClosureContextMap, ConversionError> {
    let defined = definition::run(&state.theory, &state.terms, ready_regions)?;
    state.theory = defined.generated_theory;
    merge_generated_functions(&mut state.generated_functions, defined.generated_functions);
    Ok(defined.closure_contexts)
}

fn validate_generated_theory(theory: &TheorySet) -> Result<(), ConversionError> {
    // Validate generated declarations before replacement starts depending on
    // their types and context projections.
    crate::check::check(theory).map_err(|error| ConversionError::CheckDefinitions {
        partial_definition_types: partial_definition_types(&error),
        error,
    })?;
    Ok(())
}

fn replace_ready_layer(
    state: &mut ConversionState,
    ready_regions: &region::ClosureRegionMap,
    closure_contexts: &definition::ClosureContextMap,
) -> Result<(), ConversionError> {
    let replaced =
        replace::run_partial(&state.theory, &state.terms, ready_regions, closure_contexts)?;
    state.theory = replaced.theory_set;
    state.terms = replaced.terms;
    replace::rewrite_ready_converted_primitives(&mut state.terms);
    Ok(())
}

fn region_count(regions: &region::ClosureRegionMap) -> usize {
    regions
        .values()
        .flat_map(|definitions| definitions.values())
        .map(Vec::len)
        .sum()
}

fn merge_generated_functions(output: &mut TheoryTermMap, next: TheoryTermMap) {
    for (theory, definitions) in next {
        output.entry(theory).or_default().extend(definitions);
    }
}
