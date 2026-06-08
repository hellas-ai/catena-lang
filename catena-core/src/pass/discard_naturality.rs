//! Map `bound.eta` operations into compact-closed structure

use metacat::ssa::{SSAError, ssa};
use open_hypergraphs::lax::{EdgeId, NodeId, OpenHypergraph};

/// Recursively apply naturality of discarding to an open hypergraph
pub fn discard_naturality<O: Clone + PartialEq, A: Clone>(
    mut f: OpenHypergraph<O, A>,
) -> Result<OpenHypergraph<O, A>, SSAError> {
    // Mark all nodes as "dead"
    // Mark 'target' nodes as "live"
    // Propagate backwards (in reverse SSA order):
    //  - Any op with all 'dead' nodes becomes dead
    //  - Its sources all become dead too
    //
    // Delete all ops, nodes from graph which are dead
    let mut node_live = vec![false; f.hypergraph.nodes.len()];
    let mut edge_live = vec![false; f.hypergraph.edges.len()];

    // Boundaries are *always* live, because deleting them will change the morphism type
    for t in &f.targets {
        node_live[t.0] = true;
    }

    for t in &f.sources {
        node_live[t.0] = true;
    }

    let ssa = ssa(f.clone().to_strict()).unwrap();
    for op in ssa.iter().rev() {
        // op is live if any target is live
        let op_live = op.targets.iter().any(|i| node_live[i.0.0]);
        edge_live[op.edge_id.0] = op_live;

        for source in &op.sources {
            node_live[source.0.0] = true;
        }
    }

    // Delete dead nodes and ops
    // NOTE: we call the OpenHypergraph method which also handles *interfaces*
    f.delete_nodes(
        &node_live
            .iter()
            .enumerate()
            .filter_map(|(i, &b)| (!b).then_some(NodeId(i)))
            .collect::<Vec<NodeId>>(),
    );

    f.hypergraph.delete_edges(
        &edge_live
            .iter()
            .enumerate()
            .filter_map(|(i, &b)| (!b).then_some(EdgeId(i)))
            .collect::<Vec<EdgeId>>(),
    );

    Ok(f)
}
