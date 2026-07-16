use metacat::tree::Tree;

use crate::support::*;

/// Context leaves used by generated names remain wired to the corresponding
/// original leaves, even though each generated closure compacts its local
/// context to start at leaf zero.
///
/// ```text
/// use site:       Leaf(2) ─────────────▶ name.closure.*
/// generated body: local Leaf(0) ───────▶ closure.*
///                         same parameter, different numbering scope
/// ```
#[test]
fn generated_names_retain_original_context_wiring() {
    for (definition, original_leaf) in [("indexed-if", 0), ("sparse-context-if", 2)] {
        let term = final_term(definition);
        let names = term
            .hypergraph
            .edges
            .iter()
            .zip(&term.hypergraph.adjacency)
            .filter(|(edge, _)| {
                edge.operation
                    .as_str()
                    .starts_with(&format!("name.closure.{definition}."))
            })
            .collect::<Vec<_>>();
        assert_eq!(names.len(), 2);
        for (name, boundary) in names {
            assert_eq!(name.source_sizes, vec![1]);
            let [source] = boundary.sources.as_slice() else {
                panic!("generated name should receive one context wire");
            };
            assert!(
                matches!(term.hypergraph.nodes[source.0], Tree::Leaf(leaf, ()) if leaf == original_leaf)
            );
        }
        assert_fully_lowered(definition);
    }
}

/// Context-selected names evaluated before a closure boundary remain explicit,
/// but they do not enlarge the generated closure name's own context.
#[test]
fn pre_boundary_context_operations_are_not_recaptured() {
    for (definition, expected_evals) in [("mixed-context-if", 1), ("duplicate-context-if", 2)] {
        let term = final_term(definition);
        assert_eq!(operation_count(term, "name.u64-id-for-n"), expected_evals);
        assert_eq!(operation_count(term, "eval"), expected_evals);
        let generated_sources = term
            .hypergraph
            .edges
            .iter()
            .filter(|edge| {
                edge.operation
                    .as_str()
                    .starts_with(&format!("name.closure.{definition}."))
            })
            .map(|edge| edge.source_sizes.clone())
            .collect::<Vec<_>>();
        assert_eq!(generated_sources, vec![vec![], vec![]]);
        assert_fully_lowered(definition);
    }
}

/// Context required only inside a body must still be discovered. Multiple
/// sparse dependencies are deduplicated and ordered by their original leaves.
///
/// ```text
/// body dependencies:     Leaf(5), Leaf(2), Leaf(5)
/// compact local context: Leaf(0), Leaf(1)
/// name at use site:      Leaf(2), Leaf(5)
/// ```
#[test]
fn body_dependencies_form_a_minimal_ordered_context() {
    let internal = final_term("internal-context-if");
    assert_eq!(
        only_operation(internal, "bool.ifc").source_sizes,
        vec![2, 1, 0, 1, 1, 1]
    );
    assert!(internal.hypergraph.edges.iter().any(|edge| {
        edge.operation
            .as_str()
            .starts_with("name.closure.internal-context-if.")
            && edge.source_sizes == vec![1]
    }));

    let sparse = final_term("two-sparse-contexts-if");
    let (_, boundary) = sparse
        .hypergraph
        .edges
        .iter()
        .zip(&sparse.hypergraph.adjacency)
        .find(|(edge, _)| {
            edge.operation
                .as_str()
                .starts_with("name.closure.two-sparse-contexts-if.")
                && edge.source_sizes == vec![1, 1]
        })
        .expect("one generated name should require both sparse leaves");
    let sources = boundary
        .sources
        .iter()
        .map(|node| sparse.hypergraph.nodes[node.0].clone())
        .collect::<Vec<_>>();
    assert_eq!(sources, vec![Tree::Leaf(2, ()), Tree::Leaf(5, ())]);
    assert_fully_lowered("internal-context-if");
    assert_fully_lowered("two-sparse-contexts-if");
}
