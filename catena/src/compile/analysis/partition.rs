use std::collections::HashMap;

use open_hypergraphs::lax::NodeId;

use crate::{
    compile::{
        analysis::wires::{is_interleaved_control_operation, operation_ids, operation_wires},
        graph_ops::{Graph, operation_count},
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
                if region_kind(graph, *first) == region_kind(graph, *operation) {
                    uf.union(*first, *operation);
                }
            }
        }
    }

    collect_regions(graph, uf)
}

fn collect_regions(graph: &Graph, mut uf: UnionFind) -> Vec<OperationRegion> {
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

fn region_kind(graph: &Graph, operation_id: OperationId) -> RegionKind {
    if is_interleaved_control_operation(graph, operation_id) {
        RegionKind::InterleavedControl
    } else {
        RegionKind::Data
    }
}
