//! Preserve statically named evaluations while closures are forgotten.
//!
//! Before applying the forgetful functor, a directly connected `name.* ; lift`
//! pair is fused into a private operation. The functor maps that operation to a
//! [`ClosureForgotten::NamedEval`] whose inputs already distinguish the
//! specialization context from the fully forgotten call arguments.

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

const FUSED_NAMED_LIFT_PREFIX: &str = "__catena_named_lift.";

#[derive(Debug, Clone)]
struct NamedLiftSite {
    name_edge: EdgeId,
    lift_edge: EdgeId,
    pointer_node: NodeId,
    definition: String,
}

/// Fuse the typed `name.* ; lift` shape before the functor handles its two
/// operations independently and loses the static callee identity.
pub(super) fn preserve_named_lifts(term: &mut AnnotatedTerm) {
    let sites = find_named_lifts(term);

    // Replace `name.f ; lift` by one temporary typed edge. Mapping the two
    // original edges separately would erase the fact that the eval target is
    // statically `f`.
    for site in &sites {
        term.hypergraph.edges[site.name_edge.0] =
            op(&format!("{FUSED_NAMED_LIFT_PREFIX}{}", site.definition));
        term.hypergraph.adjacency[site.name_edge.0].targets =
            term.hypergraph.adjacency[site.lift_edge.0].targets.clone();
    }
    term.delete_edges(&sites.iter().map(|site| site.lift_edge).collect::<Vec<_>>());
    term.delete_nodes(
        &sites
            .iter()
            .map(|site| site.pointer_node)
            .collect::<Vec<_>>(),
    );
}

fn find_named_lifts(term: &AnnotatedTerm) -> Vec<NamedLiftSite> {
    term.hypergraph
        .edges
        .iter()
        .enumerate()
        .filter(|(_, operation)| operation.as_str() == LIFT)
        .filter_map(|(edge, _)| named_lift_at(term, EdgeId(edge)))
        .collect()
}

fn named_lift_at(term: &AnnotatedTerm, lift_edge: EdgeId) -> Option<NamedLiftSite> {
    let boundary = &term.hypergraph.adjacency[lift_edge.0];
    let ([pointer_node], [closure_node]) =
        (boundary.sources.as_slice(), boundary.targets.as_slice())
    else {
        return None;
    };

    // Ordinary lift lowering already handles first-order interfaces. This
    // preservation is needed only when its future arguments/results contain a
    // closure that ordinary eval adapters cannot reconstruct after forgetting.
    if !closure_interface_contains_closure(&term.hypergraph.nodes[closure_node.0])
        || !is_only_consumer(term, *pointer_node, lift_edge)
    {
        return None;
    }

    let (name_edge, definition) = named_pointer_producer(term, *pointer_node)?;
    Some(NamedLiftSite {
        name_edge,
        lift_edge,
        pointer_node: *pointer_node,
        definition,
    })
}

fn is_only_consumer(term: &AnnotatedTerm, node: NodeId, expected: EdgeId) -> bool {
    let mut consumers = term
        .hypergraph
        .adjacency
        .iter()
        .enumerate()
        .filter(|(_, boundary)| boundary.sources.contains(&node))
        .map(|(edge, _)| EdgeId(edge));
    consumers.next() == Some(expected) && consumers.next().is_none()
}

fn named_pointer_producer(term: &AnnotatedTerm, pointer: NodeId) -> Option<(EdgeId, String)> {
    term.hypergraph
        .adjacency
        .iter()
        .enumerate()
        .filter(|(_, boundary)| boundary.targets.contains(&pointer))
        .find_map(|(edge, _)| {
            term.hypergraph.edges[edge]
                .as_str()
                .strip_prefix(NAME_PREFIX)
                .map(|definition| (EdgeId(edge), definition.to_string()))
        })
}

/// Map the private fused operation. Returning `None` lets the ordinary
/// forget-closures operation mapping proceed.
pub(super) fn map_fused_lift(
    operation: &Operation,
    source: &[Obj],
    target: &[Obj],
) -> Option<ClosureForgottenTerm> {
    let definition = operation.as_str().strip_prefix(FUSED_NAMED_LIFT_PREFIX)?;
    Some(build_named_lift(definition, source, target))
}

fn build_named_lift(definition: &str, source: &[Obj], target: &[Obj]) -> ClosureForgottenTerm {
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
    let context_arity = context.len();
    prepare.targets = order_lift_outputs(&prepare.targets, domain.len());

    let call = OpenHypergraph::singleton(
        ClosureForgotten::NamedEval {
            definition: definition
                .parse()
                .expect("fused named definition should parse"),
            context_arity,
        },
        [context, domain.clone()].concat(),
        codomain,
    );
    let finish = OpenHypergraph::identity(domain).tensor(&call);
    prepare
        .compose(&finish)
        .expect("fused named evaluation should compose")
}

fn order_lift_outputs(outputs: &[NodeId], domain_arity: usize) -> Vec<NodeId> {
    // `cup(domain) ⊗ id(context)` initially produces:
    //
    //   [retained domain | call domain | context]
    //
    // The following composition expects:
    //
    //   [retained domain | context | call domain]
    let (retained_domain, rest) = outputs.split_at(domain_arity);
    let (call_domain, context) = rest.split_at(domain_arity);
    retained_domain
        .iter()
        .chain(context)
        .chain(call_domain)
        .copied()
        .collect()
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
