//! Preserve statically named evaluations while closures are forgotten.
//!
//! Before applying the forgetful functor, a directly connected `name.* ; lift`
//! pair is fused into a private operation. The functor maps that operation to a
//! [`ClosureForgotten::NamedEval`] whose inputs already distinguish captured
//! context from the fully forgotten call arguments.

use hexpr::Operation;
use metacat::tree::Tree;
use open_hypergraphs::{
    category::Arrow,
    lax::{EdgeId, NodeId, OpenHypergraph},
};

use super::{
    AnnotatedTerm, ClosureForgotten, ClosureForgottenTerm, Obj, closure_forgotten_boundaries,
    closure_forgotten_boundary, closure_parts, cup, op,
};
use crate::{
    prefixes::NAME_PREFIX,
    stdlib::constants::{FN_HOM_TYPE, LIFT},
};

const PREFIX: &str = "__catena_named_lift.";

/// Fuse the typed `name.* ; lift` shape before the functor handles its two
/// operations independently and loses the static callee identity.
pub(super) fn fuse_lifts(term: &mut AnnotatedTerm) {
    #[derive(Clone)]
    struct Pair {
        name: usize,
        lift: usize,
        pointer: NodeId,
        definition: String,
    }

    let mut pairs = Vec::new();
    for (lift, operation) in term.hypergraph.edges.iter().enumerate() {
        if operation.as_str() != LIFT {
            continue;
        }
        let boundary = &term.hypergraph.adjacency[lift];
        let ([pointer], [closure]) = (boundary.sources.as_slice(), boundary.targets.as_slice())
        else {
            continue;
        };
        if !closure_interface_contains_closure(&term.hypergraph.nodes[closure.0]) {
            continue;
        }
        if term
            .hypergraph
            .adjacency
            .iter()
            .enumerate()
            .filter(|(_, edge)| edge.sources.contains(pointer))
            .map(|(edge, _)| edge)
            .collect::<Vec<_>>()
            != [lift]
        {
            continue;
        }
        let Some((name, definition)) = term
            .hypergraph
            .adjacency
            .iter()
            .enumerate()
            .filter(|(_, edge)| edge.targets.contains(pointer))
            .find_map(|(edge, _)| {
                term.hypergraph.edges[edge]
                    .as_str()
                    .strip_prefix(NAME_PREFIX)
                    .map(|definition| (edge, definition.to_string()))
            })
        else {
            continue;
        };
        pairs.push(Pair {
            name,
            lift,
            pointer: *pointer,
            definition,
        });
    }

    for pair in &pairs {
        term.hypergraph.edges[pair.name] = op(&format!("{PREFIX}{}", pair.definition));
        term.hypergraph.adjacency[pair.name].targets =
            term.hypergraph.adjacency[pair.lift].targets.clone();
    }
    term.delete_edges(
        &pairs
            .iter()
            .map(|pair| EdgeId(pair.lift))
            .collect::<Vec<_>>(),
    );
    term.delete_nodes(&pairs.iter().map(|pair| pair.pointer).collect::<Vec<_>>());
}

/// Map the private fused operation. Returning `None` lets the ordinary
/// forget-closures operation mapping proceed.
pub(super) fn map_operation(
    operation: &Operation,
    source: &[Obj],
    target: &[Obj],
) -> Option<ClosureForgottenTerm> {
    let definition = operation.as_str().strip_prefix(PREFIX)?;
    let [closure] = target else {
        panic!("fused named lift should produce one closure");
    };
    let (domain, codomain) =
        closure_parts(closure).expect("fused named lift should produce a closure");
    let context = closure_forgotten_boundaries(source);
    let domain = closure_forgotten_boundary(domain);
    let codomain = closure_forgotten_boundary(codomain);

    // Keep one copy as the lifted closure's domain and pass the other to the
    // known callee. Context precedes call arguments on the NamedEval boundary.
    let mut prepare = cup(&domain).tensor(&OpenHypergraph::identity(context.clone()));
    let domain_len = domain.len();
    let context_len = context.len();
    let targets = prepare.targets.clone();
    prepare.targets = targets[..domain_len]
        .iter()
        .chain(&targets[2 * domain_len..])
        .chain(&targets[domain_len..2 * domain_len])
        .copied()
        .collect();

    let call = OpenHypergraph::singleton(
        ClosureForgotten::NamedEval {
            definition: definition
                .parse()
                .expect("fused named definition should parse"),
            context: context_len,
        },
        [context, domain.clone()].concat(),
        codomain,
    );
    let finish = OpenHypergraph::identity(domain).tensor(&call);
    Some(
        prepare
            .compose(&finish)
            .expect("fused named evaluation should compose"),
    )
}

fn closure_interface_contains_closure(object: &Obj) -> bool {
    let Some((domain, codomain)) = closure_parts(object) else {
        return false;
    };
    [domain, codomain].into_iter().any(object_contains_closure)
}

fn object_contains_closure(object: &Obj) -> bool {
    match object {
        Tree::Node(operation, _, _) if operation.as_str() == FN_HOM_TYPE => true,
        Tree::Node(_, _, children) => children.iter().any(object_contains_closure),
        Tree::Empty | Tree::Leaf(_, _) => false,
    }
}
