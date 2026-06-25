use std::collections::HashMap;

use open_hypergraphs::lax::NodeId;
use thiserror::Error;

use crate::{check::AnnotatedTerm, closure::region::ClosureRegion};

#[derive(Debug, Error)]
pub enum ExtractRegionError {
    #[error("region node n{node} is out of bounds")]
    NodeOutOfBounds { node: usize },
    #[error("region edge e{edge} is out of bounds")]
    EdgeOutOfBounds { edge: usize },
    #[error("region edge e{edge} references node n{node}, which is not in the region")]
    IncidentNodeOutsideRegion { edge: usize, node: usize },
    #[error("region closure wire n{wire} is not in the region")]
    ClosureWireOutsideRegion { wire: usize },
    #[error("region defer input n{wire} is not in the region")]
    DeferInputOutsideRegion { wire: usize },
}

/// Copy the identified closure region into a standalone annotated term.
///
/// The extracted term contains only the region's nodes and edges. Its source
/// interface is the region's recorded `defer` inputs, in that order, and its
/// target interface is the region's closure root.
pub fn extract_region(
    definition: &AnnotatedTerm,
    region: &ClosureRegion,
) -> Result<AnnotatedTerm, ExtractRegionError> {
    validate_region(definition, region)?;

    let mut extracted = AnnotatedTerm::empty();
    let mut node_map = HashMap::<NodeId, NodeId>::new();

    for &node in &region.nodes {
        let label = definition.hypergraph.nodes[node.0].clone();
        let copied = extracted.new_node(label);
        node_map.insert(node, copied);
    }

    for &edge in &region.edges {
        let hyperedge = &definition.hypergraph.adjacency[edge.0];
        let sources = remap_nodes(&node_map, edge.0, &hyperedge.sources)?;
        let targets = remap_nodes(&node_map, edge.0, &hyperedge.targets)?;
        extracted.new_edge(
            definition.hypergraph.edges[edge.0].clone(),
            (sources, targets),
        );
    }

    extracted.sources = remap_interface(&node_map, &region.defer_inputs, |wire| {
        ExtractRegionError::DeferInputOutsideRegion { wire }
    })?;
    extracted.targets = remap_interface(&node_map, &[region.closure_wire], |wire| {
        ExtractRegionError::ClosureWireOutsideRegion { wire }
    })?;

    Ok(extracted)
}

fn validate_region(
    definition: &AnnotatedTerm,
    region: &ClosureRegion,
) -> Result<(), ExtractRegionError> {
    for &node in &region.nodes {
        if node.0 >= definition.hypergraph.nodes.len() {
            return Err(ExtractRegionError::NodeOutOfBounds { node: node.0 });
        }
    }

    for &edge in &region.edges {
        if edge.0 >= definition.hypergraph.edges.len() {
            return Err(ExtractRegionError::EdgeOutOfBounds { edge: edge.0 });
        }
    }

    Ok(())
}

fn remap_nodes(
    node_map: &HashMap<NodeId, NodeId>,
    edge: usize,
    nodes: &[NodeId],
) -> Result<Vec<NodeId>, ExtractRegionError> {
    nodes
        .iter()
        .map(|node| {
            node_map
                .get(node)
                .copied()
                .ok_or(ExtractRegionError::IncidentNodeOutsideRegion { edge, node: node.0 })
        })
        .collect()
}

fn remap_interface(
    node_map: &HashMap<NodeId, NodeId>,
    nodes: &[NodeId],
    error: impl Fn(usize) -> ExtractRegionError,
) -> Result<Vec<NodeId>, ExtractRegionError> {
    nodes
        .iter()
        .map(|node| node_map.get(node).copied().ok_or_else(|| error(node.0)))
        .collect()
}
