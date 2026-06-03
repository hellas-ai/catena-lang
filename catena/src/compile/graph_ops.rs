use hexpr::Operation;
use open_hypergraphs::{lax::NodeId, strict::vec::OpenHypergraph};

use crate::lang::Obj;

pub type Graph = OpenHypergraph<Obj, Operation>;

pub fn operation_count(graph: &Graph) -> usize {
    graph.h.x.0.len()
}

pub fn operation_name(graph: &Graph, operation_id: usize) -> &str {
    graph.h.x.0[operation_id].as_str()
}

pub fn operation_inputs(graph: &Graph, operation_id: usize) -> impl Iterator<Item = NodeId> {
    graph
        .h
        .s
        .clone()
        .into_iter()
        .nth(operation_id)
        .into_iter()
        .flat_map(|sources| sources.table.0.into_iter().map(NodeId))
}

pub fn operation_outputs(graph: &Graph, operation_id: usize) -> impl Iterator<Item = NodeId> {
    graph
        .h
        .t
        .clone()
        .into_iter()
        .nth(operation_id)
        .into_iter()
        .flat_map(|targets| targets.table.0.into_iter().map(NodeId))
}
