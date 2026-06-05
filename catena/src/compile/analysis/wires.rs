use open_hypergraphs::lax::NodeId;

use crate::{
    compile::graph_ops::{
        Graph, operation_count, operation_inputs, operation_name, operation_outputs,
    },
    stdlib::operations::{OperationKind, operation_kind},
};

use super::partition::OperationId;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct WireUses {
    pub(super) data_inputs: Vec<NodeId>,
    pub(super) data_outputs: Vec<NodeId>,
    pub(super) control_inputs: Vec<NodeId>,
    pub(super) control_outputs: Vec<NodeId>,
}

impl WireUses {
    pub(super) fn from_graph(graph: &Graph) -> Self {
        let mut uses = Self {
            data_inputs: Vec::new(),
            data_outputs: Vec::new(),
            control_inputs: Vec::new(),
            control_outputs: Vec::new(),
        };

        for operation_id in operation_ids(graph) {
            if is_interleaved_control_operation(graph, operation_id) {
                push_unique_all(
                    &mut uses.control_inputs,
                    operation_inputs(graph, operation_id),
                );
                push_unique_all(
                    &mut uses.control_outputs,
                    operation_outputs(graph, operation_id),
                );
            } else {
                push_unique_all(&mut uses.data_inputs, operation_inputs(graph, operation_id));
                push_unique_all(
                    &mut uses.data_outputs,
                    operation_outputs(graph, operation_id),
                );
            }
        }

        uses
    }
}

pub(super) fn operation_ids(graph: &Graph) -> impl Iterator<Item = OperationId> {
    0..operation_count(graph)
}

pub(super) fn is_interleaved_control_operation(graph: &Graph, operation_id: OperationId) -> bool {
    matches!(
        operation_kind(operation_name(graph, operation_id)),
        OperationKind::InterleavedControl
    )
}

pub(super) fn assert_interleaved_control_operations_are_unary(graph: &Graph) {
    for operation_id in operation_ids(graph) {
        if !is_interleaved_control_operation(graph, operation_id) {
            continue;
        }

        let input_count = operation_inputs(graph, operation_id).count();
        let output_count = operation_outputs(graph, operation_id).count();
        assert!(
            input_count == 1 && output_count == 1,
            "analysis expects interleaved control operations to have arity 1 -> 1, but operation #{operation_id} `{}` has arity {input_count} -> {output_count}",
            operation_name(graph, operation_id)
        );
    }
}

pub(super) fn operation_wires(
    graph: &Graph,
    operation_id: OperationId,
) -> impl Iterator<Item = NodeId> {
    operation_inputs(graph, operation_id).chain(operation_outputs(graph, operation_id))
}

fn push_unique_all(target: &mut Vec<NodeId>, wires: impl IntoIterator<Item = NodeId>) {
    for wire in wires {
        if !target.contains(&wire) {
            target.push(wire);
        }
    }
}
