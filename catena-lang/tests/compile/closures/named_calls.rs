use catena_lang::{
    pass::forget_closures::{self, ClosureForgotten, ClosureForgottenTerm},
    stdlib::constants::FN_HOM_TYPE,
};
use hexpr::Operation;
use metacat::tree::Tree;

use crate::support::*;

/// Inlining `forward-named-closures` exposes the named call in its forgotten
/// body. Processing must continue until both closure-bearing calls disappear.
#[test]
fn spliced_bodies_are_processed_until_no_closure_bearing_named_call_remains() {
    let before = forgotten_before_named_call_inlining();
    let before_program = &before[&program()];

    // Forgetting expands both closure arguments on the retained helper
    // boundaries: (1 => Bool), (1 => Bool), Bool becomes Bool, Bool, Bool.
    for helper in ["choose-named-closures", "forward-named-closures"] {
        assert_expanded_boundary(&before_program[&op(helper)], 3, 1);
    }

    // The outer call initially has its own two markers in the eval adapter.
    let before_entry = &before_program[&op("run-forwarded-named-closures")];
    assert_eq!(operation_count_before(before_entry, "eval"), 1);
    assert_eq!(
        operation_count_before(before_entry, "name.forward-named-closures"),
        1
    );
    assert_eq!(marker_edges(before_entry).len(), 2);

    let definitions = &conversion().closure_forgotten_definitions[&program()];
    let term = &definitions[&op("run-forwarded-named-closures")];
    assert_expanded_boundary(term, 3, 1);

    for operation in [
        "name.forward-named-closures",
        "name.choose-named-closures",
        "eval",
    ] {
        assert!(
            term.hypergraph.edges.iter().all(|edge| {
                !matches!(edge, ClosureForgotten::Operation(found)
                    if found.as_str() == operation)
            }),
            "post-forget inlining retained `{operation}`"
        );
    }

    assert!(!definitions.contains_key(&op("forward-named-closures")));
    assert!(!definitions.contains_key(&op("choose-named-closures")));
}

/// The forgotten callee body is spliced at its expanded boundary. Its own
/// adapters must still bracket both closure arguments before they enter the
/// ordinary `bool.if` operation.
#[test]
fn inlined_closure_arguments_are_fed_into_closure_markers() {
    let term = &conversion().closure_forgotten_definitions[&program()]
        [&op("run-forwarded-named-closures")];
    let bool_if = term
        .hypergraph
        .edges
        .iter()
        .position(|edge| {
            matches!(edge, ClosureForgotten::Operation(operation)
                if operation.as_str() == "bool.if")
        })
        .expect("the recursively inlined body should contain bool.if");
    let closure_inputs = &term.hypergraph.adjacency[bool_if].sources[..2];

    let markers = marker_edges(term);
    assert_eq!(markers.len(), 2);

    // Every surviving marker comes from the spliced choose-named-closures
    // body: its target is exactly one of bool.if's two closure inputs. The two
    // markers from the removed outer eval adapter therefore cannot remain.
    let mut marker_targets = markers
        .iter()
        .flat_map(|edge| &term.hypergraph.adjacency[*edge].targets)
        .map(|node| node.0)
        .collect::<Vec<_>>();
    let mut closure_inputs = closure_inputs.iter().map(|node| node.0).collect::<Vec<_>>();
    marker_targets.sort_unstable();
    closure_inputs.sort_unstable();
    assert_eq!(marker_targets, closure_inputs);

    // The marker inputs are the expanded behavioral wires, never another
    // opaque closure object.
    for marker in markers {
        assert!(
            term.hypergraph.adjacency[marker]
                .sources
                .iter()
                .all(|node| !contains_closure(&term.hypergraph.nodes[node.0]))
        );
    }
}

fn forgotten_before_named_call_inlining()
-> catena_lang::report::TheoryTermMap<ClosureForgotten<Operation>> {
    forget_closures::run(
        report().theory_set.as_ref().expect("compile checked above"),
        report()
            .definition_types
            .as_ref()
            .expect("compile checked above"),
    )
    .expect("fixture should forget closures")
}

fn assert_expanded_boundary(
    term: &ClosureForgottenTerm,
    expected_sources: usize,
    expected_targets: usize,
) {
    assert_eq!(term.sources.len(), expected_sources);
    assert_eq!(term.targets.len(), expected_targets);
    assert!(
        term.sources
            .iter()
            .chain(&term.targets)
            .all(|node| !contains_closure(&term.hypergraph.nodes[node.0]))
    );
}

fn contains_closure(object: &Tree<(), Operation>) -> bool {
    match object {
        Tree::Node(operation, _, _) if operation.as_str() == FN_HOM_TYPE => true,
        Tree::Node(_, _, children) => children.iter().any(contains_closure),
        Tree::Empty | Tree::Leaf(_, ()) => false,
    }
}

fn operation_count_before(term: &ClosureForgottenTerm, operation: &str) -> usize {
    term.hypergraph
        .edges
        .iter()
        .filter(|edge| {
            matches!(edge, ClosureForgotten::Operation(found) if found.as_str() == operation)
        })
        .count()
}

fn marker_edges(term: &ClosureForgottenTerm) -> Vec<usize> {
    term.hypergraph
        .edges
        .iter()
        .enumerate()
        .filter_map(|(edge, operation)| {
            matches!(operation, ClosureForgotten::ClosureMarker).then_some(edge)
        })
        .collect()
}
