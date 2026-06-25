use std::collections::BTreeSet;

use open_hypergraphs::lax::{EdgeId, Hyperedge, NodeId};
use thiserror::Error;

use crate::{check::AnnotatedTerm, closure::region::ClosureRegion};

#[derive(Debug, Error)]
pub enum RewriteRegionError {
    #[error("closure region rewriting requires a monogamous definition graph")]
    NonMonogamousDefinition,
    #[error("replacement source arity mismatch: expected {expected}, found {actual}")]
    SourceArity { expected: usize, actual: usize },
    #[error("region node n{node} is out of bounds")]
    RegionNodeOutOfBounds { node: usize },
    #[error("region edge e{edge} is out of bounds")]
    RegionEdgeOutOfBounds { edge: usize },
    #[error("region defer input n{wire} is out of bounds")]
    DeferInputOutOfBounds { wire: usize },
    #[error("replacement source n{wire} is out of bounds")]
    ReplacementSourceOutOfBounds { wire: usize },
    #[error("replacement source {index} type does not match region defer input type")]
    SourceTypeMismatch { index: usize },
    #[error("region boundary node n{wire} was deleted while removing region nodes")]
    DeletedBoundaryNode { wire: usize },
}

/// Replace an identified closure region with a caller-provided lowered term.
///
/// This removes the region's edges and non-`defer`-input nodes from
/// `definition`, appends `replacement`, identifies replacement sources with the
/// region's `defer` inputs, and replaces occurrences of the original closure
/// root in the outer target boundary with the replacement targets.
pub fn rewrite_region(
    definition: &AnnotatedTerm,
    region: &ClosureRegion,
    replacement: &AnnotatedTerm,
) -> Result<AnnotatedTerm, RewriteRegionError> {
    validate_region_bounds(definition, region)?;
    validate_monogamous(definition)?;
    validate_replacement(definition, region, replacement)?;

    let mut rewritten = definition.clone();
    let retained_edges = retained_edges(definition, &region.edges);
    rewritten.delete_edges(&region.edges);
    let deleted_nodes = non_defer_region_nodes(region);
    let node_map = delete_nodes_with_witness(&mut rewritten, &deleted_nodes);
    let defer_inputs = remap_boundary_nodes(&node_map, &region.defer_inputs)?;

    let (replacement_sources, replacement_targets) = rewritten.append(replacement.clone());
    for (region_source, replacement_source) in defer_inputs.into_iter().zip(replacement_sources) {
        rewritten.unify(region_source, replacement_source);
    }

    remap_remaining_edge_interfaces(
        &mut rewritten,
        definition,
        &retained_edges,
        &node_map,
        region.closure_wire,
        &replacement_targets,
    )?;
    rewritten.targets = remap_nodes_replacing(
        &node_map,
        &definition.targets,
        region.closure_wire,
        &replacement_targets,
    )?;

    Ok(rewritten)
}

fn validate_monogamous(definition: &AnnotatedTerm) -> Result<(), RewriteRegionError> {
    if definition.clone().to_strict().is_monogamous() {
        Ok(())
    } else {
        Err(RewriteRegionError::NonMonogamousDefinition)
    }
}

fn validate_region_bounds(
    definition: &AnnotatedTerm,
    region: &ClosureRegion,
) -> Result<(), RewriteRegionError> {
    for &node in &region.nodes {
        if node.0 >= definition.hypergraph.nodes.len() {
            return Err(RewriteRegionError::RegionNodeOutOfBounds { node: node.0 });
        }
    }
    for &edge in &region.edges {
        if edge.0 >= definition.hypergraph.edges.len() {
            return Err(RewriteRegionError::RegionEdgeOutOfBounds { edge: edge.0 });
        }
    }
    Ok(())
}

fn validate_replacement(
    definition: &AnnotatedTerm,
    region: &ClosureRegion,
    replacement: &AnnotatedTerm,
) -> Result<(), RewriteRegionError> {
    if replacement.sources.len() != region.defer_inputs.len() {
        return Err(RewriteRegionError::SourceArity {
            expected: region.defer_inputs.len(),
            actual: replacement.sources.len(),
        });
    }

    for (index, (&region_source, &replacement_source)) in region
        .defer_inputs
        .iter()
        .zip(&replacement.sources)
        .enumerate()
    {
        let region_type = definition.hypergraph.nodes.get(region_source.0).ok_or(
            RewriteRegionError::DeferInputOutOfBounds {
                wire: region_source.0,
            },
        )?;
        let replacement_type = replacement
            .hypergraph
            .nodes
            .get(replacement_source.0)
            .ok_or(RewriteRegionError::ReplacementSourceOutOfBounds {
                wire: replacement_source.0,
            })?;
        if region_type != replacement_type {
            return Err(RewriteRegionError::SourceTypeMismatch { index });
        }
    }

    Ok(())
}

fn non_defer_region_nodes(region: &ClosureRegion) -> Vec<NodeId> {
    let defer_inputs = region
        .defer_inputs
        .iter()
        .map(|node| node.0)
        .collect::<BTreeSet<_>>();
    region
        .nodes
        .iter()
        .copied()
        .filter(|node| !defer_inputs.contains(&node.0))
        .collect()
}

fn delete_nodes_with_witness(term: &mut AnnotatedTerm, nodes: &[NodeId]) -> Vec<Option<usize>> {
    let node_map = term.hypergraph.delete_nodes_witness(nodes);
    term.sources = term
        .sources
        .iter()
        .filter_map(|node| node_map[node.0].map(NodeId))
        .collect();
    term.targets = term
        .targets
        .iter()
        .filter_map(|node| node_map[node.0].map(NodeId))
        .collect();
    node_map
}

fn retained_edges(definition: &AnnotatedTerm, deleted_edges: &[EdgeId]) -> Vec<EdgeId> {
    let deleted = deleted_edges
        .iter()
        .map(|edge| edge.0)
        .collect::<BTreeSet<_>>();
    (0..definition.hypergraph.edges.len())
        .filter(|edge| !deleted.contains(edge))
        .map(EdgeId)
        .collect()
}

fn remap_remaining_edge_interfaces(
    rewritten: &mut AnnotatedTerm,
    original: &AnnotatedTerm,
    retained_edges: &[EdgeId],
    node_map: &[Option<usize>],
    replaced: NodeId,
    replacement_targets: &[NodeId],
) -> Result<(), RewriteRegionError> {
    for (new_edge, old_edge) in retained_edges.iter().enumerate() {
        let original_hyperedge = &original.hypergraph.adjacency[old_edge.0];
        rewritten.hypergraph.adjacency[new_edge] = Hyperedge {
            sources: remap_nodes_replacing(
                node_map,
                &original_hyperedge.sources,
                replaced,
                replacement_targets,
            )?,
            targets: remap_nodes_replacing(
                node_map,
                &original_hyperedge.targets,
                replaced,
                replacement_targets,
            )?,
        };
    }
    Ok(())
}

fn remap_nodes_replacing(
    node_map: &[Option<usize>],
    nodes: &[NodeId],
    replaced: NodeId,
    replacement_targets: &[NodeId],
) -> Result<Vec<NodeId>, RewriteRegionError> {
    let mut remapped = Vec::new();
    for &node in nodes {
        if node == replaced {
            remapped.extend_from_slice(replacement_targets);
        } else {
            remapped.push(remap_boundary_node(node_map, node)?);
        }
    }
    Ok(remapped)
}

fn remap_boundary_nodes(
    node_map: &[Option<usize>],
    nodes: &[NodeId],
) -> Result<Vec<NodeId>, RewriteRegionError> {
    nodes
        .iter()
        .map(|node| remap_boundary_node(node_map, *node))
        .collect()
}

fn remap_boundary_node(
    node_map: &[Option<usize>],
    node: NodeId,
) -> Result<NodeId, RewriteRegionError> {
    node_map
        .get(node.0)
        .and_then(|mapped| mapped.map(NodeId))
        .ok_or(RewriteRegionError::DeletedBoundaryNode { wire: node.0 })
}
