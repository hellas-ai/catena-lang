use std::collections::BTreeMap;

use hexpr::Operation;
use metacat::{check::eval_type, dual::Dual, spiders::WithSpiders, theory::Term, tree::Tree};

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

/// Generated declarations and their concrete use sites must agree on types,
/// including context that appears only inside another type or only inside the
/// closure body. Declaration leaves are compact (`0..n`); use-site leaves are
/// instantiated back into the enclosing definition's context.
///
/// ```text
/// declaration: Leaf(0), Leaf(1) ─▶ name.closure.*
/// mapping:         0 ↦ 2, 1 ↦ 5
/// use site:    Leaf(2), Leaf(5) ─▶ name.closure.*
/// ```
#[test]
fn generated_declarations_match_context_uses() {
    let definitions = [
        "indexed-if",
        "sparse-context-if",
        "mixed-context-if",
        "duplicate-context-if",
        "internal-context-if",
        "two-sparse-contexts-if",
    ];

    for definition in definitions {
        // A generated closure is a context-indexed family of functions. Its
        // source and target type maps must therefore share the same context
        // domain, even when only one boundary mentions a dependent type.
        for (_, arrow) in generated_with_prefix(&format!("closure.{definition}.")) {
            assert_eq!(
                arrow.type_maps.0.sources, arrow.type_maps.1.sources,
                "generated closure for `{definition}` has mismatched type-map context domains"
            );
        }

        let rewritten = &conversion().rewritten_definitions[&program()][&op(definition)];
        for (name, boundary) in rewritten
            .hypergraph
            .edges
            .iter()
            .zip(&rewritten.hypergraph.adjacency)
            .filter(|(operation, _)| {
                operation
                    .as_str()
                    .starts_with(&format!("name.closure.{definition}."))
            })
        {
            let declaration = &generated_arrows()[name];
            let actual_sources = boundary
                .sources
                .iter()
                .map(|node| rewritten.hypergraph.nodes[node.0].clone())
                .collect::<Vec<_>>();
            let original_leaves = actual_sources
                .iter()
                .map(|object| match object {
                    Tree::Leaf(original, ()) => *original,
                    _ => panic!("generated name `{name}` received non-context input `{object:?}`"),
                })
                .collect::<Vec<_>>();
            let declared_sources = interface_types(&declaration.type_maps.0)
                .into_iter()
                .map(|object| instantiate_context(object, &original_leaves))
                .collect::<Vec<_>>();
            assert_eq!(
                actual_sources, declared_sources,
                "use of `{name}` does not match its declared source types"
            );
        }
    }
}

/// Evaluate a type-map graph into the object types at its target boundary.
/// Type maps use their source nodes as compact context variables.
fn interface_types(term: &Term) -> Vec<Tree<(), Operation>> {
    let mut term = term.clone();
    term.quotient().expect("type map should quotient");
    let values = eval_type(
        term.clone()
            .map_edges(|operation| WithSpiders::Operation(Dual::Fwd(operation))),
    )
    .expect("generated name type map should evaluate");
    let compact_by_source = term
        .sources
        .iter()
        .enumerate()
        .map(|(compact, node)| (node.0, compact))
        .collect::<BTreeMap<_, _>>();

    term.targets
        .iter()
        .map(|node| compact_type_map_leaves(&values[node.0], &compact_by_source))
        .collect()
}

fn compact_type_map_leaves(
    object: &Tree<(), Operation>,
    compact_by_source: &BTreeMap<usize, usize>,
) -> Tree<(), Operation> {
    match object {
        Tree::Empty => Tree::Empty,
        Tree::Leaf(node, annotation) => Tree::Leaf(
            *compact_by_source
                .get(node)
                .unwrap_or_else(|| panic!("type-map target depends on non-context node {node}")),
            *annotation,
        ),
        Tree::Node(operation, annotation, children) => Tree::Node(
            operation.clone(),
            *annotation,
            children
                .iter()
                .map(|child| compact_type_map_leaves(child, compact_by_source))
                .collect(),
        ),
    }
}

fn instantiate_context(
    object: Tree<(), Operation>,
    original_leaves: &[usize],
) -> Tree<(), Operation> {
    match object {
        Tree::Empty => Tree::Empty,
        Tree::Leaf(compact, annotation) => Tree::Leaf(
            *original_leaves
                .get(compact)
                .unwrap_or_else(|| panic!("declaration uses missing compact leaf {compact}")),
            annotation,
        ),
        Tree::Node(operation, annotation, children) => Tree::Node(
            operation,
            annotation,
            children
                .into_iter()
                .map(|child| instantiate_context(child, original_leaves))
                .collect(),
        ),
    }
}
