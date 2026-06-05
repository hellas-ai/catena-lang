use open_hypergraphs::lax::NodeId;

use crate::compile::{analysis::wires::WireUses, graph_ops::Graph};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct BoundaryWires {
    pub(super) data_to_control: Vec<NodeId>,
    pub(super) control_to_data: Vec<NodeId>,
}

impl BoundaryWires {
    pub(super) fn from_graph(graph: &Graph) -> Self {
        let wire_uses = WireUses::from_graph(graph);
        let data_to_control = intersection(&wire_uses.control_inputs, &wire_uses.data_outputs);
        let control_to_data = intersection(&wire_uses.control_outputs, &wire_uses.data_inputs);

        Self {
            data_to_control,
            control_to_data,
        }
    }
}

fn intersection(left: &[NodeId], right: &[NodeId]) -> Vec<NodeId> {
    left.iter()
        .copied()
        .filter(|wire| right.contains(wire))
        .collect()
}
