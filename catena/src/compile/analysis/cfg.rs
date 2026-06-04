use std::{collections::HashMap, path::PathBuf};

use crate::compile::{
    analysis::{
        DataRegionGraph,
        partition::{OperationRegion, RegionKind},
    },
    cfg::{BlockInstruction, Cfg, CfgEdge, CfgNode, Transfer},
    graph_ops::{Graph, operation_inputs, operation_name, operation_outputs},
};

use super::{ControlRegionGraph, cfg_render::render_analysis_cfg};

pub(super) struct AnalysisCfg {
    pub(super) cfg: Cfg,
    pub(super) block_svg_paths: HashMap<usize, String>,
}

pub(super) fn render_cfg(
    graph: &Graph,
    regions: &[OperationRegion],
    control_region_graphs: &[ControlRegionGraph],
) -> Vec<u8> {
    let analysis_cfg = build_cfg(graph, regions, control_region_graphs);
    render_analysis_cfg(graph, analysis_cfg)
}

fn build_cfg(
    graph: &Graph,
    regions: &[OperationRegion],
    control_region_graphs: &[ControlRegionGraph],
) -> AnalysisCfg {
    let mut draft = collect_region_blocks(graph, regions, control_region_graphs);
    apply_transfer_shapes(&mut draft);
    let cfg = finalize_cfg(draft.nodes);
    AnalysisCfg {
        cfg,
        block_svg_paths: draft.block_svg_paths,
    }
}

fn collect_region_blocks(
    graph: &Graph,
    regions: &[OperationRegion],
    control_region_graphs: &[ControlRegionGraph],
) -> CfgDraft {
    let mut nodes = Vec::new();
    let mut block_svg_paths = HashMap::new();
    let mut pending_transfers = Vec::new();
    collect_data_graph_nodes(
        graph,
        regions,
        control_region_graphs,
        DataGraphArtifacts {
            resolved_svg: PathBuf::from("source.svg"),
            region_base: PathBuf::from("regions"),
            region_svgs_rendered: true,
            control_base: PathBuf::from("control-regions"),
        },
        false,
        &mut nodes,
        &mut block_svg_paths,
        &mut pending_transfers,
    );
    CfgDraft {
        nodes,
        block_svg_paths,
        pending_transfers,
    }
}

fn apply_transfer_shapes(draft: &mut CfgDraft) {
    apply_pending_transfers(
        &mut draft.nodes,
        std::mem::take(&mut draft.pending_transfers),
    );
}

fn finalize_cfg(nodes: Vec<CfgNode>) -> Cfg {
    assert_dense_unique_block_ids(&nodes);
    Cfg {
        entry: nodes.first().map(|node| node.id).unwrap_or(0),
        predecessors: vec![Vec::new(); nodes.len()],
        nodes,
    }
}

struct CfgDraft {
    nodes: Vec<CfgNode>,
    block_svg_paths: HashMap<usize, String>,
    pending_transfers: Vec<(usize, PendingTransfer)>,
}

fn collect_data_graph_nodes(
    graph: &Graph,
    regions: &[OperationRegion],
    control_region_graphs: &[ControlRegionGraph],
    artifacts: DataGraphArtifacts,
    in_control_context: bool,
    nodes: &mut Vec<CfgNode>,
    node_svg_paths: &mut HashMap<usize, String>,
    pending_transfers: &mut Vec<(usize, PendingTransfer)>,
) {
    let has_control_transfer = in_control_context || !control_region_graphs.is_empty();
    let boundary_context = DataBoundaryContext::new(graph, regions);
    for (region_index, region) in regions.iter().enumerate() {
        if matches!(region.kind, RegionKind::Data) {
            let id = nodes.len();
            node_svg_paths.insert(
                id,
                data_region_svg_path(region_index, region, &artifacts)
                    .display()
                    .to_string(),
            );
            push_data_region_node(
                id,
                graph,
                region,
                &boundary_context,
                has_control_transfer,
                nodes,
                pending_transfers,
            );
        }
    }

    for control_region in control_region_graphs {
        collect_control_graph_nodes(
            control_region,
            artifacts.control_base.clone(),
            nodes,
            node_svg_paths,
            pending_transfers,
        );
    }
}

fn collect_control_graph_nodes(
    control_region: &ControlRegionGraph,
    base: PathBuf,
    nodes: &mut Vec<CfgNode>,
    node_svg_paths: &mut HashMap<usize, String>,
    pending_transfers: &mut Vec<(usize, PendingTransfer)>,
) {
    let region_svgs_rendered = has_interleaved_data_regions(&control_region.regions);
    for (region_index, region) in control_region.regions.iter().enumerate() {
        if matches!(region.kind, RegionKind::Control) {
            let id = nodes.len();
            node_svg_paths.insert(
                id,
                control_region_svg_path(
                    control_region.region_index,
                    region_index,
                    region,
                    &base,
                    region_svgs_rendered,
                )
                .display()
                .to_string(),
            );
            push_control_region_node(
                id,
                &control_region.nested_graph.graph,
                region,
                nodes,
                pending_transfers,
            );
        }
    }

    let data_base = base.join(format!("{:03}-data-regions", control_region.region_index));
    for data_region in &control_region.data_region_graphs {
        collect_nested_data_graph_nodes(
            data_region,
            data_base.clone(),
            nodes,
            node_svg_paths,
            pending_transfers,
        );
    }
}

fn collect_nested_data_graph_nodes(
    data_region: &DataRegionGraph,
    base: PathBuf,
    nodes: &mut Vec<CfgNode>,
    node_svg_paths: &mut HashMap<usize, String>,
    pending_transfers: &mut Vec<(usize, PendingTransfer)>,
) {
    let region_svgs_rendered = has_interleaved_control_regions(&data_region.regions);
    collect_data_graph_nodes(
        &data_region.nested_graph.graph,
        &data_region.regions,
        &data_region.control_region_graphs,
        DataGraphArtifacts {
            resolved_svg: base.join(format!("{:03}-resolved.svg", data_region.region_index)),
            region_base: base.join(format!("{:03}-regions", data_region.region_index)),
            region_svgs_rendered,
            control_base: base.join(format!("{:03}-control-regions", data_region.region_index)),
        },
        true,
        nodes,
        node_svg_paths,
        pending_transfers,
    );
}

fn push_data_region_node(
    id: usize,
    graph: &Graph,
    region: &OperationRegion,
    boundary_context: &DataBoundaryContext,
    has_control_regions: bool,
    nodes: &mut Vec<CfgNode>,
    pending_transfers: &mut Vec<(usize, PendingTransfer)>,
) {
    let boundary = boundary_context.region_boundary(graph, region);
    nodes.push(CfgNode {
        id,
        params: boundary.inputs.clone(),
        block: region_block(graph, region),
        transfer: Transfer::Return(Vec::new()),
    });
    pending_transfers.push((
        id,
        data_region_transfer(id, region.operations.len(), &boundary, has_control_regions),
    ));
}

fn push_control_region_node(
    id: usize,
    graph: &Graph,
    region: &OperationRegion,
    nodes: &mut Vec<CfgNode>,
    pending_transfers: &mut Vec<(usize, PendingTransfer)>,
) {
    let boundary = RegionBoundary::new(graph, region);
    nodes.push(CfgNode {
        id,
        params: boundary.inputs.clone(),
        block: region_block(graph, region),
        transfer: Transfer::Return(Vec::new()),
    });
    pending_transfers.push((id, control_region_transfer(&boundary)));
}

fn region_block(graph: &Graph, region: &OperationRegion) -> Vec<BlockInstruction> {
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

#[derive(Debug, Clone)]
struct RegionBoundary {
    inputs: Vec<usize>,
    outputs: Vec<usize>,
}

#[derive(Debug, Clone)]
struct DataBoundaryContext {
    graph_sources: Vec<usize>,
    graph_targets: Vec<usize>,
    control_inputs: Vec<usize>,
    control_outputs: Vec<usize>,
}

impl DataBoundaryContext {
    fn new(graph: &Graph, regions: &[OperationRegion]) -> Self {
        let control_operations = regions
            .iter()
            .filter(|region| matches!(region.kind, RegionKind::InterleavedControl))
            .flat_map(|region| region.operations.iter().copied())
            .collect::<Vec<_>>();

        Self {
            graph_sources: graph.s.table.iter().copied().collect(),
            graph_targets: graph.t.table.iter().copied().collect(),
            control_inputs: unique_wires(
                control_operations.iter().copied().flat_map(|operation_id| {
                    operation_inputs(graph, operation_id).map(|wire| wire.0)
                }),
            ),
            control_outputs: unique_wires(control_operations.iter().copied().flat_map(
                |operation_id| operation_outputs(graph, operation_id).map(|wire| wire.0),
            )),
        }
    }

    fn region_boundary(&self, graph: &Graph, region: &OperationRegion) -> RegionBoundary {
        let consumed = region_consumed_wires(graph, region);
        let produced = region_produced_wires(graph, region);

        if self.control_inputs.is_empty() && self.control_outputs.is_empty() {
            return RegionBoundary::from_consumed_produced(graph, consumed, produced);
        }

        RegionBoundary {
            inputs: consumed
                .iter()
                .copied()
                .filter(|wire| {
                    self.graph_sources.contains(wire) || self.control_outputs.contains(wire)
                })
                .collect(),
            outputs: produced
                .iter()
                .copied()
                .filter(|wire| {
                    self.graph_targets.contains(wire) || self.control_inputs.contains(wire)
                })
                .collect(),
        }
    }
}

impl RegionBoundary {
    fn new(graph: &Graph, region: &OperationRegion) -> Self {
        let consumed = region_consumed_wires(graph, region);
        let produced = region_produced_wires(graph, region);
        Self::from_consumed_produced(graph, consumed, produced)
    }

    fn from_consumed_produced(graph: &Graph, consumed: Vec<usize>, produced: Vec<usize>) -> Self {
        let graph_sources = graph.s.table.iter().copied().collect::<Vec<_>>();
        let graph_targets = graph.t.table.iter().copied().collect::<Vec<_>>();

        Self {
            inputs: consumed
                .iter()
                .copied()
                .filter(|wire| !produced.contains(wire) || graph_sources.contains(wire))
                .collect(),
            outputs: produced
                .iter()
                .copied()
                .filter(|wire| !consumed.contains(wire) || graph_targets.contains(wire))
                .collect(),
        }
    }
}

fn region_consumed_wires(graph: &Graph, region: &OperationRegion) -> Vec<usize> {
    unique_wires(
        region
            .operations
            .iter()
            .copied()
            .flat_map(|operation_id| operation_inputs(graph, operation_id).map(|wire| wire.0)),
    )
}

fn region_produced_wires(graph: &Graph, region: &OperationRegion) -> Vec<usize> {
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

#[derive(Debug, Clone)]
enum PendingTransfer {
    Goto(Vec<usize>),
    If {
        condition: usize,
        then_args: Vec<usize>,
        else_args: Vec<usize>,
    },
    Return(Vec<usize>),
}

fn data_region_transfer(
    node_id: usize,
    operation_count: usize,
    boundary: &RegionBoundary,
    has_control_regions: bool,
) -> PendingTransfer {
    match (
        boundary.inputs.len(),
        boundary.outputs.len(),
        has_control_regions,
    ) {
        // Top-level data-only graph: the single data block is the whole CFG body.
        (_, _, false) => PendingTransfer::Return(boundary.outputs.clone()),
        // Data block enters control: boundary inputs collapse to one control token.
        (_, 1, true) => PendingTransfer::Goto(boundary.outputs.clone()),
        // Data block exits control: one control token expands to many top-level outputs.
        (1, _, true) if boundary.outputs.len() > 1 => {
            PendingTransfer::Return(boundary.outputs.clone())
        }
        _ => panic!(
            "unsupported data region shape for n{node_id} ({operation_count} operations): {} inputs -> {} outputs",
            boundary.inputs.len(),
            boundary.outputs.len()
        ),
    }
}

fn control_region_transfer(boundary: &RegionBoundary) -> PendingTransfer {
    match (boundary.inputs.len(), boundary.outputs.len()) {
        // Sequential control.
        (1, 1) => PendingTransfer::Goto(boundary.outputs.clone()),
        // Branching control.
        (1, 2) => PendingTransfer::If {
            condition: boundary.inputs[0],
            then_args: vec![boundary.outputs[0]],
            else_args: vec![boundary.outputs[1]],
        },
        // Merge control.
        (2, 1) => PendingTransfer::Goto(boundary.outputs.clone()),
        _ => panic!(
            "unsupported control region shape: {} inputs -> {} outputs",
            boundary.inputs.len(),
            boundary.outputs.len()
        ),
    }
}

fn apply_pending_transfers(
    nodes: &mut [CfgNode],
    pending_transfers: Vec<(usize, PendingTransfer)>,
) {
    for (source, pending_transfer) in pending_transfers {
        let transfer = match pending_transfer {
            PendingTransfer::Goto(args) => Transfer::Goto(CfgEdge {
                target: source,
                args,
            }),
            PendingTransfer::If {
                condition,
                then_args,
                else_args,
            } => Transfer::If {
                condition,
                then_edge: CfgEdge {
                    target: source,
                    args: then_args,
                },
                else_edge: CfgEdge {
                    target: source,
                    args: else_args,
                },
            },
            PendingTransfer::Return(args) => Transfer::Return(args),
        };
        let source_node = nodes
            .iter_mut()
            .find(|node| node.id == source)
            .expect("pending transfer source block must exist");
        source_node.transfer = transfer;
    }
}

fn assert_dense_unique_block_ids(nodes: &[CfgNode]) {
    let mut ids = nodes.iter().map(|node| node.id).collect::<Vec<_>>();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(
        ids.len(),
        nodes.len(),
        "flattened cfg block ids must be unique"
    );

    for (expected, id) in ids.into_iter().enumerate() {
        assert_eq!(
            id, expected,
            "flattened cfg block ids must be dense after sorting"
        );
    }
}

#[derive(Debug, Clone)]
struct DataGraphArtifacts {
    resolved_svg: PathBuf,
    region_base: PathBuf,
    region_svgs_rendered: bool,
    control_base: PathBuf,
}

fn data_region_svg_path(
    region_index: usize,
    region: &OperationRegion,
    artifacts: &DataGraphArtifacts,
) -> PathBuf {
    if artifacts.region_svgs_rendered {
        artifacts
            .region_base
            .join(region_svg_file_name(region_index, region.kind))
    } else {
        artifacts.resolved_svg.clone()
    }
}

fn control_region_svg_path(
    control_region_index: usize,
    region_index: usize,
    region: &OperationRegion,
    base: &std::path::Path,
    region_svgs_rendered: bool,
) -> PathBuf {
    if region_svgs_rendered {
        base.join(format!("{control_region_index:03}-regions"))
            .join(region_svg_file_name(region_index, region.kind))
    } else {
        base.join(format!("{control_region_index:03}-resolved.svg"))
    }
}

fn region_svg_file_name(region_index: usize, kind: RegionKind) -> String {
    format!("{region_index:03}-{}.svg", region_kind_file_name(kind))
}

fn region_kind_file_name(kind: RegionKind) -> &'static str {
    match kind {
        RegionKind::Data => "data",
        RegionKind::InterleavedControl => "control",
        RegionKind::Control => "native-control",
        RegionKind::InterleavedData => "interleaved-data",
    }
}

fn has_interleaved_data_regions(regions: &[OperationRegion]) -> bool {
    regions
        .iter()
        .any(|region| matches!(region.kind, RegionKind::InterleavedData))
}

fn has_interleaved_control_regions(regions: &[OperationRegion]) -> bool {
    regions
        .iter()
        .any(|region| matches!(region.kind, RegionKind::InterleavedControl))
}
