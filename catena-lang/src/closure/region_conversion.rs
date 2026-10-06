//! Convert `ClosureMarker` regions in explicit dependency order.
//!
//! One ready region is converted at a time. Rewriting invalidates graph-local
//! node and edge identifiers, so regions and their dependencies are rebuilt
//! from the new graph after every step.

use hexpr::Operation;
use metacat::theory::{TheoryId, TheorySet};

use super::{ConversionError, definition, region, replace, schedule};
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
    next_generated_id: usize,
}

struct ScheduledRegion {
    theory: TheoryId,
    definition: Operation,
    region: region::ClosureRegion,
}

pub(super) fn run(
    theory_set: &TheorySet,
    terms: TheoryTermMap<ClosureForgotten<Operation>>,
    progress: &mut dyn FnMut(&str),
) -> Result<RegionConversion, ConversionError> {
    let mut state = ConversionState {
        terms,
        theory: theory_set.clone(),
        generated_functions: TheoryTermMap::new(),
        next_generated_id: 0,
    };
    progress("Discovering closure regions");
    let mut discovered_regions = region::run(&state.terms)?;
    let initial_regions = discovered_regions.clone();

    progress(&format!(
        "Converted 0 closures; {} regions remaining",
        region_count(&discovered_regions)
    ));
    while region_count(&discovered_regions) != 0 {
        progress("Scheduling next closure");
        // Step 1: build typed ordering dependencies and choose the first region with no unresolved prerequisites.
        let selected = schedule_next_region(&state.terms, &discovered_regions);
        let generated_id = state.next_generated_id;
        state.next_generated_id += 1;

        // Step 2: generate the selected function.
        progress(&format!(
            "Generating closure {} for {}.{}",
            generated_id, selected.theory, selected.definition
        ));
        let original_context_leaves = add_closure_and_name(&mut state, &selected, generated_id)?;
        // Temporarily skip checking the entire growing theory after every closure:
        // it repeatedly checks unchanged definitions and is a likely major cost.
        // closure::run_with_progress still validates the completed theory once.
        // TODO: validate only new/affected declarations here. Until then, malformed
        // generated declarations may fail during replacement before final validation.
        // progress("Validating generated theory");
        // validate_generated_theory(&state.theory)?;

        // Step 3: replace its marker with explicit
        // environment/function-pointer values.
        progress("Replacing closure region");
        replace_region_with_closure_representation(
            &mut state,
            &selected,
            &original_context_leaves,
            generated_id,
        )?;

        // Step 4: rediscover regions and rebuild dependencies. Extraction
        // deletes and unifies nodes, so the previous snapshot is now stale.
        progress("Rediscovering closure regions");
        discovered_regions = region::run(&state.terms)?;
        progress(&format!(
            "Converted {} closures; {} regions remaining",
            state.next_generated_id,
            region_count(&discovered_regions)
        ));
    }

    Ok(RegionConversion {
        terms: state.terms,
        initial_regions,
        theory: state.theory,
        generated_functions: state.generated_functions,
    })
}

fn schedule_next_region(
    definitions: &TheoryTermMap<ClosureForgotten<Operation>>,
    discovered_regions: &region::ClosureRegionMap,
) -> ScheduledRegion {
    for (theory, theory_regions) in discovered_regions {
        for (definition, regions) in theory_regions {
            if regions.is_empty() {
                continue;
            }
            let term = &definitions[theory][definition];
            let dependencies = schedule::analyze(term, regions);
            let region = dependencies.require_ready_region(regions);
            return ScheduledRegion {
                theory: theory.clone(),
                definition: definition.clone(),
                region: region.clone(),
            };
        }
    }

    unreachable!("the scheduler is only called while closure regions remain")
}

fn add_closure_and_name(
    state: &mut ConversionState,
    selected: &ScheduledRegion,
    generated_id: usize,
) -> Result<Vec<usize>, ConversionError> {
    let term = &state.terms[&selected.theory][&selected.definition];
    let defined = definition::define_closure_and_name(
        &state.theory,
        &selected.theory,
        &selected.definition,
        term,
        &selected.region,
        generated_id,
    )?;
    state.theory = defined.generated_theory;
    state
        .generated_functions
        .entry(selected.theory.clone())
        .or_default()
        .insert(defined.operation, defined.function);
    Ok(defined.original_context_leaves)
}

#[allow(
    dead_code,
    reason = "per-iteration validation is temporarily disabled above"
)]
fn validate_generated_theory(theory: &TheorySet) -> Result<(), ConversionError> {
    // Validate generated declarations before replacement starts depending on
    // their types and context projections.
    crate::check::check(theory).map_err(|error| ConversionError::CheckDefinitions {
        partial_definition_types: partial_definition_types(&error),
        error,
    })?;
    Ok(())
}

fn replace_region_with_closure_representation(
    state: &mut ConversionState,
    selected: &ScheduledRegion,
    original_context_leaves: &[usize],
    generated_id: usize,
) -> Result<(), ConversionError> {
    let replaced = replace::replace_region_with_closure_representation(
        &state.theory,
        &state.terms,
        &selected.theory,
        &selected.definition,
        &selected.region,
        original_context_leaves,
        generated_id,
    )?;
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
