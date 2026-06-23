use thiserror::Error;

use crate::{check::AnnotatedTerm, closure::region::ClosureRegion};

#[derive(Debug, Error)]
pub enum RewriteRegionError {
    #[error("replacement source arity mismatch: expected {expected}, found {actual}")]
    SourceArity { expected: usize, actual: usize },
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
}

/// Replace an identified closure region with a caller-provided replacement term.
///
/// This removes the region's edges from `definition`, appends `replacement`, and
/// identifies replacement sources with the region's `defer` inputs and the
/// replacement target with the region's closure root. The top-level conversion
/// pass will later provide the specific replacement body.
pub fn rewrite_region(
    definition: &AnnotatedTerm,
    region: &ClosureRegion,
    replacement: &AnnotatedTerm,
) -> Result<AnnotatedTerm, RewriteRegionError> {
    validate_replacement(definition, region, replacement)?;

    let mut rewritten = definition.clone();
    rewritten.delete_edges(&region.edges);

    let (replacement_sources, replacement_targets) = rewritten.append(replacement.clone());

    for (&region_source, replacement_source) in region.defer_inputs.iter().zip(replacement_sources)
    {
        rewritten.unify(region_source, replacement_source);
    }
    rewritten.unify(region.closure_wire, replacement_targets[0]);

    Ok(rewritten)
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
        let closure_type = obj("=>", vec![obj("1", vec![]), bool_value.clone()]);

        let mut definition = AnnotatedTerm::empty();
        let captured = definition.new_node(bool_value.clone());
        let closure = definition.new_node(closure_type.clone());
        let defer = definition.new_edge(op("defer"), (vec![captured], vec![closure]));
        definition.targets = vec![closure];

        let region = ClosureRegion {
            closure_wire: closure,
            closure_type: closure_type.clone(),
            defer_inputs: vec![captured],
            nodes: vec![captured, closure],
            edges: vec![defer],
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
        assert_eq!(rewritten.targets, vec![closure]);
        assert_eq!(rewritten.hypergraph.quotient.0, vec![captured, closure]);
        assert_eq!(rewritten.hypergraph.quotient.1, vec![NodeId(2), NodeId(3)]);
    }

    fn obj(name: &str, children: Vec<Obj>) -> Obj {
        Tree::Node(op(name), 0, children)
    }

    fn op(name: &str) -> Operation {
        name.parse().expect("test operation should parse")
    }
}
