use crate::compile::{
    analysis::{
        DataRegionGraph,
        partition::{OperationRegion, RegionKind},
    },
    cfg::{BlockInstruction, Cfg, CfgNode, Transfer, render_cfg},
    graph_ops::{Graph, operation_inputs, operation_name, operation_outputs},
};

use super::ControlRegionGraph;

pub(super) fn render_flattened_cfg(
    graph: &Graph,
    regions: &[OperationRegion],
    control_region_graphs: &[ControlRegionGraph],
) -> Vec<u8> {
    let mut nodes = Vec::new();
    collect_data_graph_nodes(graph, regions, control_region_graphs, &mut nodes);
    let cfg = Cfg {
        entry: nodes.first().map(|node| node.id).unwrap_or(0),
        predecessors: vec![Vec::new(); nodes.len()],
        nodes,
    };
    render_cfg(&cfg).into_bytes()
}

fn collect_data_graph_nodes(
    graph: &Graph,
    regions: &[OperationRegion],
    control_region_graphs: &[ControlRegionGraph],
    nodes: &mut Vec<CfgNode>,
) {
    for region in regions {
        if matches!(region.kind, RegionKind::Data) {
            nodes.push(data_region_node(nodes.len(), graph, region));
        }
    }

    for control_region in control_region_graphs {
        collect_control_graph_nodes(control_region, nodes);
    }
}

fn collect_control_graph_nodes(control_region: &ControlRegionGraph, nodes: &mut Vec<CfgNode>) {
    for region in &control_region.regions {
        if matches!(region.kind, RegionKind::Control) {
            nodes.push(control_region_node(
                nodes.len(),
                &control_region.nested_graph.graph,
                region,
            ));
        }
    }

    for data_region in &control_region.data_region_graphs {
        collect_nested_data_graph_nodes(data_region, nodes);
    }
}

fn collect_nested_data_graph_nodes(data_region: &DataRegionGraph, nodes: &mut Vec<CfgNode>) {
    collect_data_graph_nodes(
        &data_region.nested_graph.graph,
        &data_region.regions,
        &data_region.control_region_graphs,
        nodes,
    );
}

fn data_region_node(id: usize, graph: &Graph, region: &OperationRegion) -> CfgNode {
    CfgNode {
        id,
        params: region_entry_wires(graph, region),
        block: region
            .operations
            .iter()
            .copied()
            .map(|operation_id| BlockInstruction {
                operation_id,
                operation: operation_name(graph, operation_id).to_string(),
                args: operation_inputs(graph, operation_id)
                    .map(|wire| wire.0)
                    .collect(),
                results: operation_outputs(graph, operation_id)
                    .map(|wire| wire.0)
                    .collect(),
            })
            .collect(),
        transfer: Transfer::Return(region_exit_wires(graph, region)),
    }
}

fn control_region_node(id: usize, graph: &Graph, region: &OperationRegion) -> CfgNode {
    CfgNode {
        id,
        params: region_entry_wires(graph, region),
        block: Vec::new(),
        transfer: Transfer::Return(region_exit_wires(graph, region)),
    }
}

fn region_entry_wires(graph: &Graph, region: &OperationRegion) -> Vec<usize> {
    unique_wires(
        region
            .operations
            .iter()
            .copied()
            .flat_map(|operation_id| operation_inputs(graph, operation_id).map(|wire| wire.0)),
    )
}

fn region_exit_wires(graph: &Graph, region: &OperationRegion) -> Vec<usize> {
    unique_wires(
        region
            .operations
            .iter()
            .copied()
            .flat_map(|operation_id| operation_outputs(graph, operation_id).map(|wire| wire.0)),
    )
}

fn unique_wires(wires: impl IntoIterator<Item = usize>) -> Vec<usize> {
    let mut unique = Vec::new();
    for wire in wires {
        if !unique.contains(&wire) {
            unique.push(wire);
        }
    }
    unique
}
