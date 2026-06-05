use std::collections::{BTreeSet, HashMap};

use crate::compile::{
    analysis::Layer,
    cfg::{BlockInstruction, Cfg, CfgEdge, CfgNode, CfgOptions, Transfer},
    graph_ops::{Graph, operation_inputs, operation_name, operation_outputs},
};
use crate::stdlib::operations::{OperationKind, actual_operation_kind, actual_operation_name};

use super::{
    cfg_render::render_analysis_cfg,
    region_graph::{RegionGraph, RegionGraphRegion, region_graph_with_regions},
    value_equivalence::{ValueEquivalences, ValueProjection, value_equivalences},
};

pub(super) struct AnalysisCfg {
    pub(super) cfg: Cfg,
    pub(super) globals: Vec<usize>,
    pub(super) block_svg_paths: HashMap<usize, String>,
}

pub(super) fn render_cfg(root_layer: &Layer, options: CfgOptions) -> Vec<u8> {
    let analysis_cfg = build_cfg(root_layer, options);
    render_analysis_cfg(&root_layer.graph, analysis_cfg)
}

fn build_cfg(root_layer: &Layer, options: CfgOptions) -> AnalysisCfg {
    let region_graph = region_graph_with_regions(root_layer);
    let connectivity = RegionGraphConnectivity::new(&region_graph.graph);
    let value_equivalences = value_equivalences(root_layer);
    let nodes = region_graph
        .regions
        .iter()
        .enumerate()
        .map(|(node_id, region)| {
            region_cfg_node(node_id, region, &connectivity, &value_equivalences, options)
        })
        .collect::<Vec<_>>();

    assert_dense_unique_block_ids(&nodes);
    let globals = cfg_globals(root_layer, &nodes);
    AnalysisCfg {
        cfg: Cfg {
            entry: connectivity.entry_node().unwrap_or(0),
            predecessors: predecessors(&nodes),
            nodes,
        },
        globals,
        block_svg_paths: region_graph_block_annotations(&region_graph),
    }
}

fn cfg_globals(root_layer: &Layer, nodes: &[CfgNode]) -> Vec<usize> {
    let defined = root_layer
        .graph
        .s
        .table
        .iter()
        .copied()
        .chain(nodes.iter().flat_map(|node| node.params.iter().copied()))
        .chain(nodes.iter().flat_map(|node| {
            node.block
                .iter()
                .flat_map(|instruction| instruction.results.iter().copied())
        }))
        .collect::<BTreeSet<_>>();

    let mut used = BTreeSet::new();
    for node in nodes {
        for instruction in &node.block {
            used.extend(instruction.args.iter().copied());
        }
        used.extend(transfer_values(&node.transfer));
    }

    used.difference(&defined).copied().collect::<Vec<_>>()
}

fn transfer_values(transfer: &Transfer) -> Vec<usize> {
    match transfer {
        Transfer::Goto(edge) => edge.args.clone(),
        Transfer::If {
            condition,
            then_edge,
            else_edge,
        } => std::iter::once(*condition)
            .chain(then_edge.args.iter().copied())
            .chain(else_edge.args.iter().copied())
            .collect(),
        Transfer::Return(values) => values.clone(),
    }
}

fn region_cfg_node(
    node_id: usize,
    region: &RegionGraphRegion,
    connectivity: &RegionGraphConnectivity,
    value_equivalences: &ValueEquivalences,
    options: CfgOptions,
) -> CfgNode {
    let graph_sources = connectivity.operation_sources(node_id);
    let graph_targets = connectivity.operation_targets(node_id);
    CfgNode {
        id: node_id,
        params: region_params(region, value_equivalences, options),
        block: region_block(&region.graph, region, value_equivalences, options),
        transfer: region_transfer(
            node_id,
            region,
            &graph_sources,
            graph_targets,
            connectivity,
            value_equivalences,
            options,
        ),
    }
}

fn region_transfer(
    node_id: usize,
    region: &RegionGraphRegion,
    sources: &[usize],
    targets: Vec<usize>,
    connectivity: &RegionGraphConnectivity,
    value_equivalences: &ValueEquivalences,
    options: CfgOptions,
) -> Transfer {
    match (sources.len(), targets.len()) {
        (_, 0) => Transfer::Return(resolve_wires(
            &region.path,
            &region.outputs,
            value_equivalences,
            options,
        )),
        (_, 1) => goto_or_return(
            targets[0],
            resolve_wires(&region.path, &region.outputs, value_equivalences, options),
            transfer_args(&region.path, &region.outputs, value_equivalences, options),
            connectivity,
        ),
        (1, 2) => Transfer::If {
            condition: branch_condition(node_id, region, value_equivalences, options),
            then_edge: edge_for_wire(
                targets[0],
                transfer_output_at(node_id, region, 0, value_equivalences, options),
                connectivity,
            ),
            else_edge: edge_for_wire(
                targets[1],
                transfer_output_at(node_id, region, 1, value_equivalences, options),
                connectivity,
            ),
        },
        _ => panic!(
            "unsupported region graph shape for n{node_id} {:?}: {} inputs -> {} outputs",
            region.kind,
            sources.len(),
            targets.len()
        ),
    }
}

fn region_params(
    region: &RegionGraphRegion,
    value_equivalences: &ValueEquivalences,
    options: CfgOptions,
) -> Vec<usize> {
    if options.keep_monoidal_operations {
        resolve_wires(&region.path, &region.inputs, value_equivalences, options)
    } else {
        Vec::new()
    }
}

fn goto_or_return(
    wire: usize,
    return_values: Vec<usize>,
    edge_args: Vec<usize>,
    connectivity: &RegionGraphConnectivity,
) -> Transfer {
    match connectivity.consumers(wire) {
        [] => Transfer::Return(return_values),
        [_] => Transfer::Goto(edge_for_wire(wire, edge_args, connectivity)),
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

fn branch_condition(
    node_id: usize,
    region: &RegionGraphRegion,
    value_equivalences: &ValueEquivalences,
    options: CfgOptions,
) -> usize {
    let [input] = region.inputs.as_slice() else {
        panic!(
            "branching region n{node_id} {:?} must have one place-graph input; got {}",
            region.kind,
            region.inputs.len()
        )
    };
    if options.keep_monoidal_operations {
        return *input;
    }

    let projection = if region_has_operation(region, "distr") {
        vec![ValueProjection::Product(0), ValueProjection::Tag]
    } else if region_has_operation(region, "distl") {
        vec![ValueProjection::Product(1), ValueProjection::Tag]
    } else {
        Vec::new()
    };
    value_equivalences.resolve(&region.path, *input, &projection)
}

fn transfer_output_at(
    node_id: usize,
    region: &RegionGraphRegion,
    index: usize,
    value_equivalences: &ValueEquivalences,
    options: CfgOptions,
) -> Vec<usize> {
    let Some(output) = region.outputs.get(index).copied() else {
        panic!(
            "region n{node_id} {:?} must have output {index}; got {} outputs",
            region.kind,
            region.outputs.len()
        )
    };
    transfer_args(&region.path, &[output], value_equivalences, options)
}

fn transfer_args(
    path: &[usize],
    wires: &[usize],
    value_equivalences: &ValueEquivalences,
    options: CfgOptions,
) -> Vec<usize> {
    if options.keep_monoidal_operations {
        resolve_wires(path, wires, value_equivalences, options)
    } else {
        Vec::new()
    }
}

fn resolve_wires(
    path: &[usize],
    wires: &[usize],
    value_equivalences: &ValueEquivalences,
    options: CfgOptions,
) -> Vec<usize> {
    if options.keep_monoidal_operations {
        wires.to_vec()
    } else {
        wires
            .iter()
            .copied()
            .map(|wire| value_equivalences.resolve_wire(path, wire))
            .collect()
    }
}

fn region_block(
    graph: &Graph,
    region: &RegionGraphRegion,
    value_equivalences: &ValueEquivalences,
    options: CfgOptions,
) -> Vec<BlockInstruction> {
    region
        .region
        .operations
        .iter()
        .copied()
        .filter(|operation_id| {
            options.keep_monoidal_operations || !is_monoidal_operation(graph, *operation_id)
        })
        .map(|operation_id| BlockInstruction {
            operation_id,
            operation: operation_name(graph, operation_id).to_string(),
            args: operation_inputs(graph, operation_id)
                .map(|wire| {
                    if options.keep_monoidal_operations {
                        wire.0
                    } else {
                        value_equivalences.resolve_wire(&region.path, wire.0)
                    }
                })
                .collect(),
            results: operation_outputs(graph, operation_id)
                .map(|wire| {
                    if options.keep_monoidal_operations {
                        wire.0
                    } else {
                        value_equivalences.resolve_wire(&region.path, wire.0)
                    }
                })
                .collect(),
        })
        .collect()
}

fn is_monoidal_operation(graph: &Graph, operation_id: usize) -> bool {
    actual_operation_kind(operation_name(graph, operation_id)) == OperationKind::MonoidalStructure
}

fn region_has_operation(region: &RegionGraphRegion, operation: &str) -> bool {
    region
        .region
        .operations
        .iter()
        .copied()
        .any(|operation_id| {
            actual_operation_name(operation_name(&region.graph, operation_id)) == operation
        })
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
