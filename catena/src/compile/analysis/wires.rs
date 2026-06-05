use open_hypergraphs::lax::NodeId;

use crate::{
    compile::graph_ops::{
        Graph, operation_count, operation_inputs, operation_name, operation_outputs,
    },
    stdlib::operations::{OperationKind, operation_kind},
};

use super::partition::OperationId;

pub(super) fn operation_ids(graph: &Graph) -> impl Iterator<Item = OperationId> {
    0..operation_count(graph)
}

pub(super) fn is_interleaved_control_operation(graph: &Graph, operation_id: OperationId) -> bool {
    matches!(
        operation_kind(operation_name(graph, operation_id)),
        OperationKind::InterleavedControl
    )
}

pub(super) fn is_interleaved_data_operation(graph: &Graph, operation_id: OperationId) -> bool {
    matches!(
        operation_kind(operation_name(graph, operation_id)),
        OperationKind::InterleavedData
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
