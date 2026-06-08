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

pub(super) fn operation_wires(
    graph: &Graph,
    operation_id: OperationId,
) -> impl Iterator<Item = NodeId> {
    operation_inputs(graph, operation_id).chain(operation_outputs(graph, operation_id))
}
