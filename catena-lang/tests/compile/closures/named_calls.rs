use catena_lang::pass::forget_closures::ClosureForgotten;

use crate::support::*;

/// Inlining `forward-named-closures` exposes the named call in its forgotten
/// body. Processing must continue until both closure-bearing calls disappear.
#[test]
fn spliced_bodies_are_processed_until_no_closure_bearing_named_call_remains() {
    let definitions = &conversion().closure_forgotten_definitions[&program()];
    let term = &definitions[&op("run-forwarded-named-closures")];

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

    for closure_input in closure_inputs {
        let producers = term
            .hypergraph
            .adjacency
            .iter()
            .enumerate()
            .filter(|(_, boundary)| boundary.targets.contains(closure_input))
            .map(|(edge, _)| &term.hypergraph.edges[edge])
            .collect::<Vec<_>>();
        assert!(
            producers
                .iter()
                .any(|edge| matches!(edge, ClosureForgotten::ClosureMarker)),
            "expanded closure input w{} reaches bool.if without a closure marker",
            closure_input.0
        );
    }

    assert_eq!(
        term.hypergraph
            .edges
            .iter()
            .filter(|edge| matches!(edge, ClosureForgotten::ClosureMarker))
            .count(),
        2
    );
}
