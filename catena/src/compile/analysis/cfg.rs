use std::collections::HashMap;

use crate::compile::{
    analysis::Layer,
    cfg::{BlockInstruction, Cfg, CfgEdge, CfgNode, Transfer},
    graph_ops::{Graph, operation_inputs, operation_name, operation_outputs},
};

use super::{
    cfg_render::render_analysis_cfg,
    region_graph::{RegionGraph, RegionGraphRegion, region_graph_with_regions},
};

pub(super) struct AnalysisCfg {
    pub(super) cfg: Cfg,
    pub(super) block_svg_paths: HashMap<usize, String>,
}

pub(super) fn render_cfg(root_layer: &Layer) -> Vec<u8> {
    let analysis_cfg = build_cfg(root_layer);
    render_analysis_cfg(&root_layer.graph, analysis_cfg)
}

fn build_cfg(root_layer: &Layer) -> AnalysisCfg {
    let region_graph = region_graph_with_regions(root_layer);
    let connectivity = RegionGraphConnectivity::new(&region_graph.graph);
    let nodes = region_graph
        .regions
        .iter()
        .enumerate()
        .map(|(node_id, region)| region_cfg_node(node_id, region, &connectivity))
        .collect::<Vec<_>>();

    assert_dense_unique_block_ids(&nodes);
    AnalysisCfg {
        cfg: Cfg {
            entry: connectivity.entry_node().unwrap_or(0),
            predecessors: predecessors(&nodes),
            nodes,
        },
        block_svg_paths: region_graph_block_annotations(&region_graph),
    }
}

fn region_cfg_node(
    node_id: usize,
    region: &RegionGraphRegion,
    connectivity: &RegionGraphConnectivity,
) -> CfgNode {
    let graph_sources = connectivity.operation_sources(node_id);
    let graph_targets = connectivity.operation_targets(node_id);
    CfgNode {
        id: node_id,
        params: region.inputs.clone(),
        block: region_block(&region.graph, &region.region),
        transfer: region_transfer(node_id, region, &graph_sources, graph_targets, connectivity),
    }
}

fn region_transfer(
    node_id: usize,
    region: &RegionGraphRegion,
    sources: &[usize],
    targets: Vec<usize>,
    connectivity: &RegionGraphConnectivity,
) -> Transfer {
    match (sources.len(), targets.len()) {
        (_, 0) => Transfer::Return(region.outputs.clone()),
        (_, 1) => goto_or_return(targets[0], &region.outputs, connectivity),
        (1, 2) => Transfer::If {
            condition: single_input(node_id, region),
            then_edge: edge_for_wire(targets[0], output_at(node_id, region, 0), connectivity),
            else_edge: edge_for_wire(targets[1], output_at(node_id, region, 1), connectivity),
        },
        _ => panic!(
            "unsupported region graph shape for n{node_id} {:?}: {} inputs -> {} outputs",
            region.kind,
            sources.len(),
            targets.len()
        ),
    }
}

fn goto_or_return(
    wire: usize,
    outputs: &[usize],
    connectivity: &RegionGraphConnectivity,
) -> Transfer {
    match connectivity.consumers(wire) {
        [] => Transfer::Return(outputs.to_vec()),
        [_] => Transfer::Goto(edge_for_wire(wire, outputs.to_vec(), connectivity)),
        consumers => panic!(
            "non-branching region graph wire w{wire} has {} consumers",
            consumers.len()
        ),
    }
}

fn edge_for_wire(wire: usize, args: Vec<usize>, connectivity: &RegionGraphConnectivity) -> CfgEdge {
    let consumers = connectivity.consumers(wire);
    let [target] = consumers else {
        panic!(
            "region graph wire w{wire} must have exactly one consumer; got {}",
            consumers.len()
        )
    };
    CfgEdge {
        target: *target,
        args,
    }
}

fn single_input(node_id: usize, region: &RegionGraphRegion) -> usize {
    let [input] = region.inputs.as_slice() else {
        panic!(
            "branching region n{node_id} {:?} must have one place-graph input; got {}",
            region.kind,
            region.inputs.len()
        )
    };
    *input
}

fn output_at(node_id: usize, region: &RegionGraphRegion, index: usize) -> Vec<usize> {
    let Some(output) = region.outputs.get(index).copied() else {
        panic!(
            "region n{node_id} {:?} must have output {index}; got {} outputs",
            region.kind,
            region.outputs.len()
        )
    };
    vec![output]
}

fn region_block(graph: &Graph, region: &crate::compile::analysis::Region) -> Vec<BlockInstruction> {
    region
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
        .collect()
}

struct RegionGraphConnectivity {
    sources_by_operation: Vec<Vec<usize>>,
    targets_by_operation: Vec<Vec<usize>>,
    consumers_by_wire: HashMap<usize, Vec<usize>>,
    producer_by_wire: HashMap<usize, usize>,
}

impl RegionGraphConnectivity {
    fn new(graph: &Graph) -> Self {
        let mut sources_by_operation = Vec::new();
        let mut targets_by_operation = Vec::new();
        let mut consumers_by_wire = HashMap::<usize, Vec<usize>>::new();
        let mut producer_by_wire = HashMap::<usize, usize>::new();

        for operation_id in 0..graph.h.x.0.len() {
            let sources = operation_inputs(graph, operation_id)
                .map(|wire| wire.0)
                .collect::<Vec<_>>();
            for source in &sources {
                consumers_by_wire
                    .entry(*source)
                    .or_default()
                    .push(operation_id);
            }

            let targets = operation_outputs(graph, operation_id)
                .map(|wire| wire.0)
                .collect::<Vec<_>>();
            for target in &targets {
                let previous = producer_by_wire.insert(*target, operation_id);
                assert!(
                    previous.is_none(),
                    "region graph wire w{target} has multiple producers"
                );
            }

            sources_by_operation.push(sources);
            targets_by_operation.push(targets);
        }

        Self {
            sources_by_operation,
            targets_by_operation,
            consumers_by_wire,
            producer_by_wire,
        }
    }

    fn operation_sources(&self, operation_id: usize) -> Vec<usize> {
        self.sources_by_operation[operation_id].clone()
    }

    fn operation_targets(&self, operation_id: usize) -> Vec<usize> {
        self.targets_by_operation[operation_id].clone()
    }

    fn consumers(&self, wire: usize) -> &[usize] {
        self.consumers_by_wire
            .get(&wire)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    fn entry_node(&self) -> Option<usize> {
        self.sources_by_operation
            .iter()
            .enumerate()
            .find(|(_, sources)| {
                sources
                    .iter()
                    .any(|source| !self.producer_by_wire.contains_key(source))
            })
            .map(|(operation_id, _)| operation_id)
    }
}

fn predecessors(nodes: &[CfgNode]) -> Vec<Vec<usize>> {
    let mut predecessors = vec![Vec::new(); nodes.len()];
    for node in nodes {
        for successor in successors(&node.transfer) {
            predecessors[successor].push(node.id);
        }
    }
    predecessors
}

fn successors(transfer: &Transfer) -> Vec<usize> {
    match transfer {
        Transfer::Goto(edge) => vec![edge.target],
        Transfer::If {
            then_edge,
            else_edge,
            ..
        } => vec![then_edge.target, else_edge.target],
        Transfer::Return(_) => Vec::new(),
    }
}

fn region_graph_block_annotations(region_graph: &RegionGraph) -> HashMap<usize, String> {
    region_graph
        .regions
        .iter()
        .enumerate()
        .map(|(node_id, region)| (node_id, region_path_annotation(&region.path, region.kind)))
        .collect()
}

fn region_path_annotation(
    path: &[usize],
    kind: crate::compile::analysis::partition::RegionKind,
) -> String {
    let path = path
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(".");
    format!("region.{path}.{}", region_kind_name(kind))
}

fn region_kind_name(kind: crate::compile::analysis::partition::RegionKind) -> &'static str {
    match kind {
        crate::compile::analysis::partition::RegionKind::Data => "data",
        crate::compile::analysis::partition::RegionKind::Control => "control",
        crate::compile::analysis::partition::RegionKind::InterleavedControl => {
            "interleaved-control"
        }
        crate::compile::analysis::partition::RegionKind::InterleavedData => "interleaved-data",
    }
}

fn assert_dense_unique_block_ids(nodes: &[CfgNode]) {
    let mut ids = nodes.iter().map(|node| node.id).collect::<Vec<_>>();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), nodes.len(), "cfg block ids must be unique");

    for (expected, id) in ids.into_iter().enumerate() {
        assert_eq!(id, expected, "cfg block ids must be dense after sorting");
    }
}
