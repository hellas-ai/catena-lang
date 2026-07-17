use catena_lang::pass::forget_closures::ClosureForgotten;

use crate::support::*;

/// A closure-free definition is a baseline for the whole compile pipeline: no
/// marker, region, or generated arrow should be introduced accidentally.
#[test]
fn closure_free_definition_is_unchanged() {
    assert!(regions("identity").is_empty());
    let forgotten = &conversion().closure_forgotten_definitions[&program()][&op("identity")];
    assert!(
        forgotten
            .hypergraph
            .edges
            .iter()
            .all(|edge| { !matches!(edge, ClosureForgotten::ClosureMarker) })
    );
    assert!(generated_with_prefix("closure.identity.").is_empty());
}

/// Closure construction, composition with a named arrow, and immediate `run`
/// are CMC plumbing. The full pipeline should erase that plumbing without
/// inventing a closure-conversion region around the identity computation.
#[test]
fn composed_defer_and_run_lower_without_a_region() {
    assert!(regions("deferred-identity").is_empty());
    assert_eq!(
        operation_count(final_term("deferred-identity"), "name.bool.id"),
        1
    );
    assert_eq!(operation_count(final_term("deferred-identity"), "eval"), 1);
    assert_eq!(
        operation_count(final_term("deferred-identity"), "compose"),
        0
    );
    assert_eq!(operation_count(final_term("deferred-identity"), "run"), 0);
    assert_fully_lowered("deferred-identity");
}

/// Named branches have no captured runtime values. Forgetting must delimit two
/// empty-environment regions and conversion must replace `bool.if` with one
/// explicit `bool.ifc` call backed by two generated bodies and names.
///
/// ```text
/// name f ──lift──┐                       env=() ─┐
/// name g ──lift──┼─ bool.if ──▶    name f* ─────┼─ bool.ifc
/// flag ──────────┘                name g* ──────┤
///                                  flag ───────┘
/// ```
#[test]
fn named_branches_become_explicit_function_pairs() {
    assert_eq!(regions("named-if").len(), 2);
    assert!(
        regions("named-if")
            .iter()
            .all(|region| region.environment.is_empty())
    );
    assert_eq!(generated_with_prefix("closure.named-if.").len(), 2);
    assert_eq!(generated_with_prefix("name.closure.named-if.").len(), 2);

    let final_graph = final_term("named-if");
    assert_eq!(operation_count(final_graph, "bool.if"), 0);
    assert_eq!(operation_count(final_graph, "bool.ifc"), 1);
    assert_fully_lowered("named-if");
}

/// Deferred values are the minimal captured case. Each branch region captures
/// exactly one value, while adjacent multi-edge regions must both survive the
/// replacement without stale graph identifiers.
///
/// ```text
/// lhs ──defer──┐                 lhs ─┬─ environment
/// rhs ──defer──┼─ bool.if  ─▶    rhs ─┤
/// flag ────────┘              names ──┼─ bool.ifc
///                                 flag ┘
/// ```
#[test]
fn captured_and_composed_branches_preserve_their_regions() {
    assert!(
        regions("captured-if")
            .iter()
            .all(|region| region.environment.len() == 1)
    );
    assert_eq!(regions("composed-if").len(), 2);

    for definition in ["captured-if", "composed-if"] {
        let final_graph = final_term(definition);
        assert_eq!(operation_count(final_graph, "bool.if"), 0);
        assert_eq!(operation_count(final_graph, "bool.ifc"), 1);
        assert_eq!(
            final_graph
                .hypergraph
                .edges
                .iter()
                .filter(|edge| edge
                    .operation
                    .as_str()
                    .starts_with(&format!("name.closure.{definition}.")))
                .count(),
            2
        );
        assert_fully_lowered(definition);
    }
}

/// This is the asymmetric form of the historical multiple-region regression:
/// replacing the larger left region must not leave stale positional IDs for the
/// smaller right region.
///
/// ```text
/// lhs ─ defer ─ name/eval ─ compose ─┐
/// rhs ─ defer ───────────────────────┼─ bool.if
/// flag ──────────────────────────────┘
/// ```
#[test]
fn asymmetric_parallel_regions_are_both_replaced() {
    let regions = regions("asymmetric-regions-if");
    assert_eq!(regions.len(), 2);
    assert_ne!(regions[0].marker, regions[1].marker);

    let final_graph = final_term("asymmetric-regions-if");
    assert_eq!(operation_count(final_graph, "bool.if"), 0);
    assert_eq!(operation_count(final_graph, "bool.ifc"), 1);
    assert_eq!(
        final_graph
            .hypergraph
            .edges
            .iter()
            .filter(|edge| edge
                .operation
                .as_str()
                .starts_with("name.closure.asymmetric-regions-if."))
            .count(),
        2
    );
    assert_fully_lowered("asymmetric-regions-if");
    assert_monogamous("asymmetric-regions-if");
}

/// Closure conversion and structural lowering must preserve linear use of every
/// runtime wire. These definitions intentionally have no ambient context, so a
/// strict monogamy check is appropriate for their complete final graphs.
#[test]
fn runtime_only_final_graphs_are_monogamous() {
    for definition in [
        "identity",
        "deferred-identity",
        "named-if",
        "captured-if",
        "composed-if",
        "asymmetric-regions-if",
    ] {
        assert_monogamous(definition);
    }
}
