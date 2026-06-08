use std::collections::{BTreeSet, HashMap, HashSet};

use crate::compile::{
    CompileGraph,
    cfg::{BlockInstruction, Cfg, CfgArtifacts, CfgBuild, CfgEdge, CfgNode, CfgOptions, Transfer},
    graph_ops::{Graph, operation_inputs, operation_name, operation_outputs},
};
use crate::stdlib::operations::{OperationKind, actual_operation_kind, actual_operation_name};

use super::{
    layering::Layer,
    partition::RegionKind,
    region_graph::{RegionGraph, RegionGraphRegion},
    value_equivalence::{ValueEquivalences, ValueProjection},
};

pub(super) fn lower_region_graph_to_cfg(
    graph: &CompileGraph,
    root_layer: &Layer,
    region_graph: &RegionGraph,
    value_equivalences: &ValueEquivalences,
    wire_names: HashMap<usize, String>,
    options: CfgOptions,
) -> CfgBuild {
    let connectivity = RegionGraphConnectivity::new(&region_graph.graph);
    let mut nodes = region_graph
        .regions
        .iter()
        .enumerate()
        .map(|(node_id, region)| {
            region_cfg_node(node_id, region, &connectivity, &value_equivalences, options)
        })
        .collect::<Vec<_>>();
    let mut entry = connectivity.entry_node().unwrap_or(0);
    let mut block_svg_paths = region_graph_block_annotations(&region_graph);

    if !options.keep_control_flow_operations {
        (nodes, entry, block_svg_paths) = remove_empty_goto_blocks(nodes, entry, block_svg_paths);
    }

    assert_dense_unique_block_ids(&nodes);
    let globals = cfg_globals(root_layer, &nodes);
    let cfg = Cfg {
        entry,
        predecessors: predecessors(&nodes),
        nodes,
    };
    CfgBuild {
        artifacts: CfgArtifacts {
            graph: graph.clone(),
            layer: root_layer.clone(),
            cfg: cfg.clone(),
            globals: globals.clone(),
            wire_names: wire_names.clone(),
            block_svg_paths: block_svg_paths.clone(),
        },
        cfg,
        globals,
        wire_names,
        block_svg_paths,
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

fn remove_empty_goto_blocks(
    nodes: Vec<CfgNode>,
    entry: usize,
    block_svg_paths: HashMap<usize, String>,
) -> (Vec<CfgNode>, usize, HashMap<usize, String>) {
    let removable = nodes
        .iter()
        .filter(|node| removable_empty_goto_block(node))
        .map(|node| node.id)
        .collect::<HashSet<_>>();
    if removable.is_empty() {
        return (nodes, entry, block_svg_paths);
    }

    let redirected_entry = redirect_entry(entry, &nodes, &removable);
    let mut kept_nodes = nodes
        .iter()
        .filter(|node| !removable.contains(&node.id))
        .cloned()
        .map(|mut node| {
            node.transfer = redirect_transfer(node.transfer, &nodes, &removable);
            node
        })
        .collect::<Vec<_>>();

    let id_map = kept_nodes
        .iter()
        .enumerate()
        .map(|(new_id, node)| (node.id, new_id))
        .collect::<HashMap<_, _>>();

    for (new_id, node) in kept_nodes.iter_mut().enumerate() {
        node.id = new_id;
        remap_transfer_targets(&mut node.transfer, &id_map);
    }

    let block_svg_paths = block_svg_paths
        .into_iter()
        .filter_map(|(old_id, annotation)| id_map.get(&old_id).map(|new_id| (*new_id, annotation)))
        .collect();

    (kept_nodes, id_map[&redirected_entry], block_svg_paths)
}

fn removable_empty_goto_block(node: &CfgNode) -> bool {
    node.block.is_empty() && matches!(node.transfer, Transfer::Goto(_))
}

fn redirect_entry(entry: usize, nodes: &[CfgNode], removable: &HashSet<usize>) -> usize {
    let edge = redirect_edge(
        CfgEdge {
            target: entry,
            args: Vec::new(),
        },
        nodes,
        removable,
    );
    assert!(
        edge.args.is_empty(),
        "entry redirection through empty blocks cannot synthesize block arguments"
    );
    edge.target
}

fn redirect_transfer(
    transfer: Transfer,
    nodes: &[CfgNode],
    removable: &HashSet<usize>,
) -> Transfer {
    match transfer {
        Transfer::Goto(edge) => Transfer::Goto(redirect_edge(edge, nodes, removable)),
        Transfer::If {
            condition,
            then_edge,
            else_edge,
        } => Transfer::If {
            condition,
            then_edge: redirect_edge(then_edge, nodes, removable),
            else_edge: redirect_edge(else_edge, nodes, removable),
        },
        Transfer::Return(values) => Transfer::Return(values),
    }
}

fn redirect_edge(mut edge: CfgEdge, nodes: &[CfgNode], removable: &HashSet<usize>) -> CfgEdge {
    let mut seen = HashSet::new();
    while removable.contains(&edge.target) {
        let removed_node = edge.target;
        assert!(
            seen.insert(removed_node),
            "cycle while removing empty cfg block n{}",
            removed_node
        );
        let Transfer::Goto(next) = &nodes[removed_node].transfer else {
            unreachable!("only empty goto blocks are removable")
        };
        edge = CfgEdge {
            target: next.target,
            args: redirected_args(&nodes[removed_node], &edge.args, &next.args),
        };
    }
    edge
}

fn redirected_args(node: &CfgNode, incoming_args: &[usize], outgoing_args: &[usize]) -> Vec<usize> {
    match (incoming_args, outgoing_args) {
        (_, []) => Vec::new(),
        ([incoming], [_outgoing]) => vec![*incoming],
        ([], outgoing) => outgoing.to_vec(),
        (incoming, outgoing) if incoming == outgoing => incoming.to_vec(),
        _ => panic!(
            "cannot remove empty cfg block n{} with {} params, {} incoming args, and {} outgoing args",
            node.id,
            node.params.len(),
            incoming_args.len(),
            outgoing_args.len()
        ),
    }
}

fn remap_transfer_targets(transfer: &mut Transfer, id_map: &HashMap<usize, usize>) {
    match transfer {
        Transfer::Goto(edge) => remap_edge_target(edge, id_map),
        Transfer::If {
            then_edge,
            else_edge,
            ..
        } => {
            remap_edge_target(then_edge, id_map);
            remap_edge_target(else_edge, id_map);
        }
        Transfer::Return(_) => {}
    }
}

fn remap_edge_target(edge: &mut CfgEdge, id_map: &HashMap<usize, usize>) {
    edge.target = *id_map
        .get(&edge.target)
        .unwrap_or_else(|| panic!("missing remapped cfg node id for n{}", edge.target));
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
            (options.keep_monoidal_operations || !is_monoidal_operation(graph, *operation_id))
                && (options.keep_control_flow_operations
                    || !is_control_flow_operation(graph, *operation_id))
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

fn is_control_flow_operation(graph: &Graph, operation_id: usize) -> bool {
    actual_operation_kind(operation_name(graph, operation_id)) == OperationKind::ControlFlow
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

fn region_path_annotation(path: &[usize], kind: RegionKind) -> String {
    let path = path
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(".");
    format!("region.{path}.{}", region_kind_name(kind))
}

fn region_kind_name(kind: RegionKind) -> &'static str {
    match kind {
        RegionKind::Data => "data",
        RegionKind::Control => "control",
        RegionKind::InterleavedControl => "interleaved-control",
        RegionKind::InterleavedData => "interleaved-data",
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
