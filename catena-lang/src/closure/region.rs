//! Discover closure control-flow regions in a closure-forgotten graph.
//!
//! A `!closure` edge has two sources: its domain and codomain control-flow
//! endpoints. The core of a region is the directed slice that lies on a path
//! from the domain to the codomain.  Sources needed by that slice but produced
//! outside it form the captured environment.

use std::collections::{BTreeMap, VecDeque};

use hexpr::Operation;
use metacat::theory::TheoryId;
use open_hypergraphs::lax::{EdgeId, NodeId};
use thiserror::Error;

use crate::{
    pass::forget_closures::{ClosureForgotten, ClosureForgottenTerm},
    prefixes::NAME_PREFIX,
    report::TheoryTermMap,
};

/// Discovered regions grouped by theory and definition.
pub type ClosureRegionMap = BTreeMap<TheoryId, BTreeMap<Operation, Vec<ClosureRegion>>>;

/// A delimited closure body inside a graph produced by `forget_closures`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClosureRegion {
    /// The `!closure` edge which delimits this region.
    pub marker: EdgeId,
    /// The future argument of the closure body.
    pub domain: NodeId,
    /// The result of the closure body.
    pub codomain: NodeId,
    /// The opaque closure object consumed by the surrounding operation.
    pub closure: NodeId,
    /// Values entering the body from the surrounding graph.
    pub environment: Vec<NodeId>,
    /// Nodes belonging to the body, including its boundary.
    pub nodes: Vec<NodeId>,
    /// Body edges. The `!closure` marker itself is not included.
    pub edges: Vec<EdgeId>,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum FindRegionError {
    #[error(
        "closure marker e{edge} has {sources} sources and {targets} targets; expected two sources and one target"
    )]
    InvalidMarkerBoundary {
        edge: usize,
        sources: usize,
        targets: usize,
    },
}

/// Discover closure regions for every forgotten definition in a compiler stage.
pub fn run(
    terms: &TheoryTermMap<ClosureForgotten<Operation>>,
) -> Result<ClosureRegionMap, FindRegionError> {
    terms
        .iter()
        .map(|(theory, definitions)| {
            let regions = definitions
                .iter()
                .map(|(definition, term)| Ok((definition.clone(), find_regions(term)?)))
                .collect::<Result<BTreeMap<_, _>, FindRegionError>>()?;
            Ok((theory.clone(), regions))
        })
        .collect()
}

/// Find every closure region in marker-edge order.
///
/// For each marker with sources `(domain, codomain)`, region discovery proceeds
/// as follows:
///
/// 1. Walk forward from `domain`, without crossing any closure marker, and mark
///    every reachable edge.
/// 2. Walk backward from `codomain`, again without crossing markers, and mark
///    every edge from which the codomain is reachable.
/// 3. Intersect those two edge sets. An edge is in the control-flow body exactly
///    when it lies on a directed path from the closure domain to its codomain.
/// 4. Add `name.*` producers used by included `eval` edges. These are static
///    dependencies of the body rather than captured runtime values.
/// 5. Treat inputs of included edges that have no producer inside the body as
///    environment values. If forgetting `defer` removed the entire control
///    path, the unproduced codomain itself is the captured value.
///
/// The resulting region contains the intersected body, its static named
/// dependencies, and the environment boundary; the marker edge is only the
/// delimiter and is not part of the body.
pub fn find_regions(term: &ClosureForgottenTerm) -> Result<Vec<ClosureRegion>, FindRegionError> {
    let connectivity = Connectivity::new(term);
    term.hypergraph
        .edges
        .iter()
        .enumerate()
        .filter_map(|(index, edge)| {
            matches!(edge, ClosureForgotten::ClosureMarker).then_some(EdgeId(index))
        })
        .map(|marker| find_region(term, &connectivity, marker))
        .collect()
}

fn find_region(
    term: &ClosureForgottenTerm,
    connectivity: &Connectivity,
    marker: EdgeId,
) -> Result<ClosureRegion, FindRegionError> {
    let boundary = &term.hypergraph.adjacency[marker.0];
    let ([domain, codomain], [closure]) =
        (boundary.sources.as_slice(), boundary.targets.as_slice())
    else {
        return Err(FindRegionError::InvalidMarkerBoundary {
            edge: marker.0,
            sources: boundary.sources.len(),
            targets: boundary.targets.len(),
        });
    };

    let forward = reachable_forward(term, connectivity, *domain);
    let backward = reachable_backward(term, connectivity, *codomain);
    let discarded = reachable_backward_from_forward_sinks(term, connectivity, &forward);
    let mut included_edges = forward
        .iter()
        .zip(backward.iter().zip(discarded))
        .map(|(from_domain, (to_codomain, to_discard))| {
            *from_domain && (*to_codomain || to_discard)
        })
        .collect::<Vec<_>>();

    include_named_dependencies(term, connectivity, &mut included_edges);

    let environment = environment_nodes(term, connectivity, &included_edges, *domain, *codomain);
    let nodes = region_nodes(term, &included_edges, *domain, *codomain, &environment);
    let edges = included_edges
        .into_iter()
        .enumerate()
        .filter_map(|(index, included)| included.then_some(EdgeId(index)))
        .collect();

    Ok(ClosureRegion {
        marker,
        domain: *domain,
        codomain: *codomain,
        closure: *closure,
        environment,
        nodes,
        edges,
    })
}

/// Mark edges reachable in the ordinary graph direction from `start`.
fn reachable_forward(
    term: &ClosureForgottenTerm,
    connectivity: &Connectivity,
    start: NodeId,
) -> Vec<bool> {
    let mut reached_nodes = vec![false; term.hypergraph.nodes.len()];
    let mut reached_edges = vec![false; term.hypergraph.edges.len()];
    let mut pending = VecDeque::from([start]);
    reached_nodes[start.0] = true;

    while let Some(node) = pending.pop_front() {
        for &edge in &connectivity.consumers_by_node[node.0] {
            if is_marker(term, edge) || reached_edges[edge.0] {
                continue;
            }
            reached_edges[edge.0] = true;
            for &target in &term.hypergraph.adjacency[edge.0].targets {
                if !reached_nodes[target.0] {
                    reached_nodes[target.0] = true;
                    pending.push_back(target);
                }
            }
        }
    }

    reached_edges
}

/// Mark edges from which `end` is reachable in the ordinary graph direction.
fn reachable_backward(
    term: &ClosureForgottenTerm,
    connectivity: &Connectivity,
    end: NodeId,
) -> Vec<bool> {
    let mut reached_nodes = vec![false; term.hypergraph.nodes.len()];
    let mut reached_edges = vec![false; term.hypergraph.edges.len()];
    let mut pending = VecDeque::from([end]);
    reached_nodes[end.0] = true;

    while let Some(node) = pending.pop_front() {
        for &edge in &connectivity.producers_by_node[node.0] {
            if is_marker(term, edge) || reached_edges[edge.0] {
                continue;
            }
            reached_edges[edge.0] = true;
            for &source in &term.hypergraph.adjacency[edge.0].sources {
                if !reached_nodes[source.0] {
                    reached_nodes[source.0] = true;
                    pending.push_back(source);
                }
            }
        }
    }

    reached_edges
}

/// Mark forward-reachable branches which terminate without producing a value.
///
/// A closure may ignore all or part of its argument, for example:
///
/// ```text
/// domain ─> *.elim ─┬─> unit.elim
///                    `─> unit.elim
/// ```
///
/// These edges cannot reach the codomain, but they remain part of the closure
/// body. Start at reachable sink edges and walk backwards to recover the whole
/// discarded branch; the caller intersects the result with forward reachability
/// from the closure domain.
fn reachable_backward_from_forward_sinks(
    term: &ClosureForgottenTerm,
    connectivity: &Connectivity,
    forward: &[bool],
) -> Vec<bool> {
    let mut reached_nodes = vec![false; term.hypergraph.nodes.len()];
    let mut reached_edges = vec![false; term.hypergraph.edges.len()];
    let mut pending = VecDeque::new();

    for (index, reachable) in forward.iter().copied().enumerate() {
        if !reachable || !term.hypergraph.adjacency[index].targets.is_empty() {
            continue;
        }
        reached_edges[index] = true;
        for &source in &term.hypergraph.adjacency[index].sources {
            if !reached_nodes[source.0] {
                reached_nodes[source.0] = true;
                pending.push_back(source);
            }
        }
    }

    while let Some(node) = pending.pop_front() {
        for &edge in &connectivity.producers_by_node[node.0] {
            if is_marker(term, edge) || reached_edges[edge.0] {
                continue;
            }
            reached_edges[edge.0] = true;
            for &source in &term.hypergraph.adjacency[edge.0].sources {
                if !reached_nodes[source.0] {
                    reached_nodes[source.0] = true;
                    pending.push_back(source);
                }
            }
        }
    }

    reached_edges
}

/// Named function pointers are static dependencies of an `eval`, not runtime
/// closure captures. Keep their producer edges in the extracted body.
fn include_named_dependencies(
    term: &ClosureForgottenTerm,
    connectivity: &Connectivity,
    included_edges: &mut [bool],
) {
    loop {
        let mut changed = false;
        for edge_index in 0..included_edges.len() {
            if !included_edges[edge_index] {
                continue;
            }
            for source in &term.hypergraph.adjacency[edge_index].sources {
                for &producer in &connectivity.producers_by_node[source.0] {
                    if !included_edges[producer.0] && is_named_operation(term, producer) {
                        included_edges[producer.0] = true;
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
}

fn environment_nodes(
    term: &ClosureForgottenTerm,
    connectivity: &Connectivity,
    included_edges: &[bool],
    domain: NodeId,
    codomain: NodeId,
) -> Vec<NodeId> {
    let mut environment = vec![false; term.hypergraph.nodes.len()];

    for (edge_index, included) in included_edges.iter().copied().enumerate() {
        if !included {
            continue;
        }
        for &source in &term.hypergraph.adjacency[edge_index].sources {
            if source != domain && !has_included_producer(connectivity, included_edges, source) {
                environment[source.0] = true;
            }
        }
    }

    // `defer` disappears during forgetting. For an entirely captured closure
    // there is therefore no domain-to-codomain edge; its codomain is the value
    // stored in the environment.
    if codomain != domain && !has_included_producer(connectivity, included_edges, codomain) {
        environment[codomain.0] = true;
    }

    environment
        .into_iter()
        .enumerate()
        .filter_map(|(index, captured)| captured.then_some(NodeId(index)))
        .collect()
}

fn region_nodes(
    term: &ClosureForgottenTerm,
    included_edges: &[bool],
    domain: NodeId,
    codomain: NodeId,
    environment: &[NodeId],
) -> Vec<NodeId> {
    let mut included_nodes = vec![false; term.hypergraph.nodes.len()];
    included_nodes[domain.0] = true;
    included_nodes[codomain.0] = true;
    for &node in environment {
        included_nodes[node.0] = true;
    }
    for (edge_index, included) in included_edges.iter().copied().enumerate() {
        if !included {
            continue;
        }
        let edge = &term.hypergraph.adjacency[edge_index];
        for node in edge.sources.iter().chain(&edge.targets) {
            included_nodes[node.0] = true;
        }
    }
    included_nodes
        .into_iter()
        .enumerate()
        .filter_map(|(index, included)| included.then_some(NodeId(index)))
        .collect()
}

fn has_included_producer(
    connectivity: &Connectivity,
    included_edges: &[bool],
    node: NodeId,
) -> bool {
    connectivity.producers_by_node[node.0]
        .iter()
        .any(|producer| included_edges[producer.0])
}

fn is_marker(term: &ClosureForgottenTerm, edge: EdgeId) -> bool {
    matches!(
        term.hypergraph.edges[edge.0],
        ClosureForgotten::ClosureMarker
    )
}

fn is_named_operation(term: &ClosureForgottenTerm, edge: EdgeId) -> bool {
    matches!(
        &term.hypergraph.edges[edge.0],
        ClosureForgotten::Operation(operation) if operation.as_str().starts_with(NAME_PREFIX)
    )
}

struct Connectivity {
    producers_by_node: Vec<Vec<EdgeId>>,
    consumers_by_node: Vec<Vec<EdgeId>>,
}

impl Connectivity {
    fn new(term: &ClosureForgottenTerm) -> Self {
        let mut producers_by_node = vec![Vec::new(); term.hypergraph.nodes.len()];
        let mut consumers_by_node = vec![Vec::new(); term.hypergraph.nodes.len()];
        for (edge_index, boundary) in term.hypergraph.adjacency.iter().enumerate() {
            let edge = EdgeId(edge_index);
            for target in &boundary.targets {
                producers_by_node[target.0].push(edge);
            }
            for source in &boundary.sources {
                consumers_by_node[source.0].push(edge);
            }
        }
        Self {
            producers_by_node,
            consumers_by_node,
        }
    }
}

#[cfg(test)]
mod tests {
    use hexpr::Operation;
    use metacat::tree::Tree;

    use super::*;

    #[test]
    fn named_closure_contains_eval_and_name_without_runtime_environment() {
        let mut term = ClosureForgottenTerm::empty();
        let domain = term.new_node(obj("A"));
        let pointer = term.new_node(obj("pointer"));
        let codomain = term.new_node(obj("B"));
        let closure = term.new_node(obj("A=>B"));

        let name = term.new_edge(region_op("name.f"), (vec![], vec![pointer]));
        let eval = term.new_edge(region_op("eval"), (vec![domain, pointer], vec![codomain]));
        let marker = term.new_edge(
            ClosureForgotten::ClosureMarker,
            (vec![domain, codomain], vec![closure]),
        );

        let [region] = find_regions(&term).unwrap().try_into().unwrap();
        assert_eq!(region.marker, marker);
        assert_eq!(region.domain, domain);
        assert_eq!(region.codomain, codomain);
        assert_eq!(region.closure, closure);
        assert_eq!(region.environment, vec![]);
        assert_eq!(region.edges, vec![name, eval]);
    }

    #[test]
    fn operation_inputs_off_the_control_path_are_environment() {
        let mut term = ClosureForgottenTerm::empty();
        let domain = term.new_node(obj("A"));
        let captured = term.new_node(obj("E"));
        let codomain = term.new_node(obj("B"));
        let closure = term.new_node(obj("A=>B"));

        let body = term.new_edge(region_op("body"), (vec![domain, captured], vec![codomain]));
        term.new_edge(
            ClosureForgotten::ClosureMarker,
            (vec![domain, codomain], vec![closure]),
        );

        let [region] = find_regions(&term).unwrap().try_into().unwrap();
        assert_eq!(region.environment, vec![captured]);
        assert_eq!(region.edges, vec![body]);
    }

    #[test]
    fn deferred_value_is_captured_when_no_domain_to_codomain_path_exists() {
        let mut term = ClosureForgottenTerm::empty();
        let domain = term.new_node(obj("1"));
        let captured_codomain = term.new_node(obj("B"));
        let closure = term.new_node(obj("1=>B"));

        let discard = term.new_edge(region_op("unit.elim"), (vec![domain], vec![]));
        term.new_edge(
            ClosureForgotten::ClosureMarker,
            (vec![domain, captured_codomain], vec![closure]),
        );

        let [region] = find_regions(&term).unwrap().try_into().unwrap();
        assert_eq!(region.environment, vec![captured_codomain]);
        assert_eq!(region.edges, vec![discard]);
        assert_eq!(region.nodes, vec![domain, captured_codomain]);
    }

    #[test]
    fn malformed_marker_is_reported() {
        let mut term = ClosureForgottenTerm::empty();
        let only_source = term.new_node(obj("A"));
        term.new_edge(ClosureForgotten::ClosureMarker, (vec![only_source], vec![]));

        assert_eq!(
            find_regions(&term),
            Err(FindRegionError::InvalidMarkerBoundary {
                edge: 0,
                sources: 1,
                targets: 0,
            })
        );
    }

    fn obj(name: &str) -> Tree<(), Operation> {
        Tree::Node(op(name), 0, vec![])
    }

    fn region_op(name: &str) -> ClosureForgotten<Operation> {
        ClosureForgotten::Operation(op(name))
    }

    fn op(name: &str) -> Operation {
        name.parse().expect("test operation should parse")
    }
}
