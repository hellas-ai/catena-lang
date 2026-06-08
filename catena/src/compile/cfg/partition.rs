use std::collections::{HashMap, HashSet};

use open_hypergraphs::lax::NodeId;

use crate::{
    compile::{
        cfg::wires::{
            is_interleaved_control_operation, is_interleaved_data_operation, operation_ids,
            operation_wires,
        },
        graph_ops::{Graph, operation_count, operation_inputs, operation_outputs},
    },
    union_find::UnionFind,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct OperationRegion {
    pub(super) kind: RegionKind,
    pub(super) operations: Vec<OperationId>,
}

pub(super) type OperationId = usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RegionKind {
    Data,
    InterleavedControl,
    Control,
    InterleavedData,
}

pub(super) fn partition_regions(graph: &Graph) -> Vec<OperationRegion> {
    let mut uf = UnionFind::new(operation_count(graph));
    let mut operations_by_wire = HashMap::<NodeId, Vec<OperationId>>::new();

    for operation_id in operation_ids(graph) {
        for wire in operation_wires(graph, operation_id) {
            operations_by_wire
                .entry(wire)
                .or_default()
                .push(operation_id);
        }
    }

    for operations in operations_by_wire.values() {
        if let Some((first, rest)) = operations.split_first() {
            for operation in rest {
                if data_operations_should_union(graph, *first, *operation) {
                    uf.union(*first, *operation);
                }
            }
        }
    }

    collect_regions(graph, uf, data_region_kind)
}

pub(super) fn partition_control_regions(graph: &Graph) -> Vec<OperationRegion> {
    let boundary_wires = graph_boundary_wires(graph);
    let mut uf = UnionFind::new(operation_count(graph));
    let mut producers_by_wire = HashMap::<NodeId, Vec<OperationId>>::new();
    let mut consumers_by_wire = HashMap::<NodeId, Vec<OperationId>>::new();

    for operation_id in operation_ids(graph) {
        for wire in operation_outputs(graph, operation_id) {
            producers_by_wire
                .entry(wire)
                .or_default()
                .push(operation_id);
        }

        for wire in operation_inputs(graph, operation_id) {
            consumers_by_wire
                .entry(wire)
                .or_default()
                .push(operation_id);
        }
    }

    for (wire, producers) in &producers_by_wire {
        let Some(consumers) = consumers_by_wire.get(wire) else {
            continue;
        };

        for producer in producers {
            for consumer in consumers {
                if control_operations_should_union(
                    graph,
                    *wire,
                    *producer,
                    *consumer,
                    &boundary_wires,
                ) {
                    uf.union(*producer, *consumer);
                }
            }
        }
    }

    collect_regions(graph, uf, control_region_kind)
}

fn collect_regions(
    graph: &Graph,
    mut uf: UnionFind,
    region_kind: impl Fn(&Graph, OperationId) -> RegionKind,
) -> Vec<OperationRegion> {
    let mut region_by_root = HashMap::<usize, usize>::new();
    let mut regions = Vec::<OperationRegion>::new();

    for operation_id in operation_ids(graph) {
        let root = uf.find(operation_id);
        let next_region = regions.len();
        let region_id = *region_by_root.entry(root).or_insert_with(|| {
            regions.push(OperationRegion {
                kind: region_kind(graph, operation_id),
                operations: Vec::new(),
            });
            next_region
        });
        regions[region_id].operations.push(operation_id);
    }

    regions
}

fn data_operations_should_union(graph: &Graph, left: OperationId, right: OperationId) -> bool {
    data_region_kind(graph, left) == data_region_kind(graph, right)
}

fn data_region_kind(graph: &Graph, operation_id: OperationId) -> RegionKind {
    if is_interleaved_control_operation(graph, operation_id) {
        RegionKind::InterleavedControl
    } else {
        RegionKind::Data
    }
}

fn control_operations_should_union(
    graph: &Graph,
    wire: NodeId,
    producer: OperationId,
    consumer: OperationId,
    boundary_wires: &HashSet<NodeId>,
) -> bool {
    if boundary_wires.contains(&wire) || producer == consumer {
        return false;
    }

    if control_region_kind(graph, producer) != control_region_kind(graph, consumer) {
        return false;
    }

    !is_branch_operation(graph, producer) && !is_merge_operation(graph, consumer)
}

fn control_region_kind(graph: &Graph, operation_id: OperationId) -> RegionKind {
    if is_interleaved_data_operation(graph, operation_id) {
        RegionKind::InterleavedData
    } else {
        RegionKind::Control
    }
}

fn is_branch_operation(graph: &Graph, operation_id: OperationId) -> bool {
    operation_outputs(graph, operation_id).count() > 1
}

fn is_merge_operation(graph: &Graph, operation_id: OperationId) -> bool {
    operation_inputs(graph, operation_id).count() > 1
}

fn graph_boundary_wires(graph: &Graph) -> HashSet<NodeId> {
    graph
        .s
        .table
        .0
        .iter()
        .chain(&graph.t.table.0)
        .copied()
        .map(NodeId)
        .collect()
}
