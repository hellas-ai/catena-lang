use std::collections::HashMap;

use open_hypergraphs::lax::NodeId;

use crate::{
    compile::graph_ops::{
        Graph, operation_count, operation_inputs, operation_name, operation_outputs,
    },
    compile::{CompileGraph, CompileTheory, graph_render},
    stdlib::operations::{OperationKind, operation_kind},
    union_find::UnionFind,
};

pub fn render_analysis(graph: &CompileGraph) -> std::io::Result<Vec<u8>> {
    assert!(
        matches!(graph.theory, CompileTheory::Data),
        "analysis expects a data graph"
    );

    let boundary_wires = BoundaryWires::from_graph(&graph.graph);
    let _regions = partition_regions(&graph.graph, &boundary_wires);
    render_step(AnalysisStep::NormalizedGraph, graph)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AnalysisStep {
    NormalizedGraph,
}

fn render_step(step: AnalysisStep, graph: &CompileGraph) -> std::io::Result<Vec<u8>> {
    match step {
        AnalysisStep::NormalizedGraph => graph_render::nested_svg(graph),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BoundaryWires {
    data_to_control: Vec<NodeId>,
    control_to_data: Vec<NodeId>,
}

impl BoundaryWires {
    fn from_graph(graph: &Graph) -> Self {
        let wire_uses = WireUses::from_graph(graph);
        let data_to_control = intersection(&wire_uses.control_inputs, &wire_uses.data_outputs);
        let control_to_data = intersection(&wire_uses.control_outputs, &wire_uses.data_inputs);

        Self {
            data_to_control,
            control_to_data,
        }
    }

    fn contains(&self, wire: NodeId) -> bool {
        self.data_to_control.contains(&wire) || self.control_to_data.contains(&wire)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct OperationRegion {
    kind: RegionKind,
    operations: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RegionKind {
    Data,
    InterleavedControl,
}

fn partition_regions(graph: &Graph, boundary: &BoundaryWires) -> Vec<OperationRegion> {
    let mut uf = UnionFind::new(operation_count(graph));
    let mut operations_by_wire = HashMap::<NodeId, Vec<usize>>::new();

    for operation_id in 0..operation_count(graph) {
        for wire in operation_wires(graph, operation_id) {
            if !boundary.contains(wire) {
                operations_by_wire
                    .entry(wire)
                    .or_default()
                    .push(operation_id);
            }
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

    for operation_id in 0..operation_count(graph) {
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

fn region_kind(graph: &Graph, operation_id: usize) -> RegionKind {
    if is_interleaved_control_operation(graph, operation_id) {
        RegionKind::InterleavedControl
    } else {
        RegionKind::Data
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct WireUses {
    data_inputs: Vec<NodeId>,
    data_outputs: Vec<NodeId>,
    control_inputs: Vec<NodeId>,
    control_outputs: Vec<NodeId>,
}

impl WireUses {
    fn from_graph(graph: &Graph) -> Self {
        let mut uses = Self {
            data_inputs: Vec::new(),
            data_outputs: Vec::new(),
            control_inputs: Vec::new(),
            control_outputs: Vec::new(),
        };

        for operation_id in 0..operation_count(graph) {
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

fn is_interleaved_control_operation(graph: &Graph, operation_id: usize) -> bool {
    matches!(
        operation_kind(operation_name(graph, operation_id)),
        OperationKind::InterleavedControl
    )
}

fn operation_wires(graph: &Graph, operation_id: usize) -> impl Iterator<Item = NodeId> {
    operation_inputs(graph, operation_id).chain(operation_outputs(graph, operation_id))
}

fn push_unique_all(target: &mut Vec<NodeId>, wires: impl IntoIterator<Item = NodeId>) {
    for wire in wires {
        if !target.contains(&wire) {
            target.push(wire);
        }
    }
}

fn intersection(left: &[NodeId], right: &[NodeId]) -> Vec<NodeId> {
    left.iter()
        .copied()
        .filter(|wire| right.contains(wire))
        .collect()
}
