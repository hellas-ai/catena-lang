use std::collections::BTreeSet;

use open_hypergraphs::lax::NodeId;
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
    #[error("replacement target arity mismatch: expected 1, found {actual}")]
    TargetArity { actual: usize },
    #[error("region defer input n{wire} is out of bounds")]
    DeferInputOutOfBounds { wire: usize },
    #[error("region closure wire n{wire} is out of bounds")]
    ClosureWireOutOfBounds { wire: usize },
    #[error("replacement source n{wire} is out of bounds")]
    ReplacementSourceOutOfBounds { wire: usize },
    #[error("replacement target n{wire} is out of bounds")]
    ReplacementTargetOutOfBounds { wire: usize },
    #[error("replacement source {index} type does not match region defer input type")]
    SourceTypeMismatch { index: usize },
    #[error("replacement target type does not match region closure type")]
    TargetTypeMismatch,
    #[error("region boundary node n{wire} was deleted while removing internal region nodes")]
    DeletedBoundaryNode { wire: usize },
}

/// Replace an identified closure region with a caller-provided replacement term.
///
/// This removes the region's internal edges and nodes from `definition`, appends
/// `replacement`, and identifies replacement sources with the region's `defer`
/// inputs and the replacement target with the region's closure root. The
/// top-level conversion pass will later provide the specific replacement body.
pub fn rewrite_region(
    definition: &AnnotatedTerm,
    region: &ClosureRegion,
    replacement: &AnnotatedTerm,
) -> Result<AnnotatedTerm, RewriteRegionError> {
    validate_region_bounds(definition, region)?;
    validate_monogamous(definition)?;
    validate_replacement(definition, region, replacement)?;

    let mut rewritten = definition.clone();
    rewritten.delete_edges(&region.edges);
    let deleted_nodes = internal_region_nodes(region);
    let node_map = delete_nodes_with_witness(&mut rewritten, &deleted_nodes);
    let defer_inputs = remap_boundary_nodes(&node_map, &region.defer_inputs)?;
    let closure_wire = remap_boundary_node(&node_map, region.closure_wire)?;

    let (replacement_sources, replacement_targets) = rewritten.append(replacement.clone());

    for (region_source, replacement_source) in defer_inputs.into_iter().zip(replacement_sources) {
        rewritten.unify(region_source, replacement_source);
    }
    rewritten.unify(closure_wire, replacement_targets[0]);

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

fn internal_region_nodes(region: &ClosureRegion) -> Vec<NodeId> {
    let boundary = region
        .defer_inputs
        .iter()
        .copied()
        .chain([region.closure_wire])
        .map(|node| node.0)
        .collect::<BTreeSet<_>>();
    region
        .nodes
        .iter()
        .copied()
        .filter(|node| !boundary.contains(&node.0))
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
    if replacement.targets.len() != 1 {
        return Err(RewriteRegionError::TargetArity {
            actual: replacement.targets.len(),
        });
    }

    let closure_type = definition
        .hypergraph
        .nodes
        .get(region.closure_wire.0)
        .ok_or(RewriteRegionError::ClosureWireOutOfBounds {
            wire: region.closure_wire.0,
        })?;
    if closure_type != &region.closure_type {
        return Err(RewriteRegionError::TargetTypeMismatch);
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

    let replacement_target = replacement.targets[0];
    let replacement_target_type = replacement
        .hypergraph
        .nodes
        .get(replacement_target.0)
        .ok_or(RewriteRegionError::ReplacementTargetOutOfBounds {
            wire: replacement_target.0,
        })?;
    if replacement_target_type != &region.closure_type {
        return Err(RewriteRegionError::TargetTypeMismatch);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use hexpr::Operation;
    use metacat::tree::Tree;
    use open_hypergraphs::lax::NodeId;

    use super::*;
    use crate::closure::region::Obj;

    #[test]
    fn rewrite_replaces_region_edges_with_replacement_edges() {
        let bool_value = obj("val", vec![obj("bool", vec![])]);
        let internal_type = obj("internal", vec![]);
        let closure_type = obj("=>", vec![obj("1", vec![]), bool_value.clone()]);

        let mut definition = AnnotatedTerm::empty();
        let captured = definition.new_node(bool_value.clone());
        let internal = definition.new_node(internal_type);
        let closure = definition.new_node(closure_type.clone());
        let prepare = definition.new_edge(op("prepare"), (vec![captured], vec![internal]));
        let defer = definition.new_edge(op("defer"), (vec![internal], vec![closure]));
        definition.sources = vec![captured];
        definition.targets = vec![closure];

        let region = ClosureRegion {
            closure_wire: closure,
            closure_type: closure_type.clone(),
            defer_inputs: vec![captured],
            nodes: vec![captured, internal, closure],
            edges: vec![prepare, defer],
        };

        let mut replacement = AnnotatedTerm::empty();
        let replacement_source = replacement.new_node(bool_value);
        let replacement_target = replacement.new_node(closure_type);
        replacement.new_edge(
            op("replacement"),
            (vec![replacement_source], vec![replacement_target]),
        );
        replacement.sources = vec![replacement_source];
        replacement.targets = vec![replacement_target];

        let rewritten = rewrite_region(&definition, &region, &replacement)
            .expect("region rewrite should succeed");

        assert_eq!(rewritten.hypergraph.edges, vec![op("replacement")]);
        assert_eq!(rewritten.hypergraph.nodes.len(), 4);
        assert_eq!(rewritten.targets, vec![NodeId(1)]);
        assert_eq!(rewritten.hypergraph.quotient.0, vec![NodeId(0), NodeId(1)]);
        assert_eq!(rewritten.hypergraph.quotient.1, vec![NodeId(2), NodeId(3)]);
    }

    #[test]
    fn rewrite_rejects_non_monogamous_definition() {
        let bool_value = obj("val", vec![obj("bool", vec![])]);
        let closure_type = obj("=>", vec![obj("1", vec![]), bool_value.clone()]);

        let mut definition = AnnotatedTerm::empty();
        let captured = definition.new_node(bool_value.clone());
        let closure = definition.new_node(closure_type.clone());
        let first = definition.new_edge(op("first"), (vec![captured], vec![closure]));
        let second = definition.new_edge(op("second"), (vec![captured], vec![closure]));
        definition.sources = vec![captured];
        definition.targets = vec![closure];

        let region = ClosureRegion {
            closure_wire: closure,
            closure_type: closure_type.clone(),
            defer_inputs: vec![captured],
            nodes: vec![captured, closure],
            edges: vec![first, second],
        };

        let mut replacement = AnnotatedTerm::empty();
        let replacement_source = replacement.new_node(bool_value);
        let replacement_target = replacement.new_node(closure_type);
        replacement.new_edge(
            op("replacement"),
            (vec![replacement_source], vec![replacement_target]),
        );
        replacement.sources = vec![replacement_source];
        replacement.targets = vec![replacement_target];

        let error =
            rewrite_region(&definition, &region, &replacement).expect_err("rewrite should fail");

        assert!(matches!(
            error,
            RewriteRegionError::NonMonogamousDefinition
        ));
    }

    fn obj(name: &str, children: Vec<Obj>) -> Obj {
        Tree::Node(op(name), 0, children)
    }

    fn op(name: &str) -> Operation {
        name.parse().expect("test operation should parse")
    }
}
