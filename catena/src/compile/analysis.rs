use open_hypergraphs::lax::NodeId;

use crate::{
    compile::graph_ops::{
        Graph, operation_count, operation_inputs, operation_name, operation_outputs,
    },
    compile::{CompileGraph, CompileTheory, graph_render},
    stdlib::operations::{OperationKind, operation_kind},
};

pub fn render_analysis(graph: &CompileGraph) -> std::io::Result<Vec<u8>> {
    assert!(
        matches!(graph.theory, CompileTheory::Data),
        "analysis expects a data graph"
    );

    let _boundary_wires = BoundaryWires::from_graph(&graph.graph);
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
