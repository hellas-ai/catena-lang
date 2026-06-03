use std::{collections::HashMap, io, path::PathBuf};

use open_hypergraphs::lax::NodeId;
use open_hypergraphs_dot::{Options, svg::to_svg_with};

use crate::{
    compile::graph_ops::{
        Graph, operation_count, operation_inputs, operation_name, operation_outputs,
    },
    compile::{CompileGraph, CompileTheory, graph_render::object_label},
    hypergraph::subgraph::{Subgraph, subgraph_from_operations},
    lang::Obj,
    stdlib::operations::{OperationKind, operation_kind},
    union_find::UnionFind,
};

pub fn render_analysis(graph: &CompileGraph) -> std::io::Result<Vec<u8>> {
    Ok(render_analysis_artifacts(graph)?
        .into_iter()
        .find(|artifact| artifact.path == PathBuf::from("normalized.svg"))
        .expect("analysis artifacts include normalized graph")
        .contents)
}

#[derive(Debug, Clone)]
pub struct AnalysisArtifact {
    pub path: PathBuf,
    pub contents: Vec<u8>,
}

pub fn render_analysis_artifacts(graph: &CompileGraph) -> std::io::Result<Vec<AnalysisArtifact>> {
    assert!(
        matches!(graph.theory, CompileTheory::Data),
        "analysis expects a data graph"
    );

    // I don't know if it is too strict, but I cannot imagine a case when it is not true
    // better fail early and loud if I am wrong!
    assert_interleaved_control_operations_are_unary(&graph.graph);
    let _boundary_wires = BoundaryWires::from_graph(&graph.graph);
    let regions = partition_regions(&graph.graph);
    let region_svgs = render_region_svgs(graph, &regions)?;
    let before_split = graph_svg(&graph.graph)?;
    let mut artifacts = vec![
        AnalysisArtifact {
            path: PathBuf::from("before-split.svg"),
            contents: before_split.clone(),
        },
        AnalysisArtifact {
            path: PathBuf::from("normalized.svg"),
            contents: before_split,
        },
    ];
    artifacts.extend(region_svgs.into_iter().map(|region| AnalysisArtifact {
        path: PathBuf::from("regions").join(region.file_name),
        contents: region.svg,
    }));
    Ok(artifacts)
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
struct OperationRegion {
    kind: RegionKind,
    operations: Vec<OperationId>,
}

type OperationId = usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RegionKind {
    Data,
    InterleavedControl,
}

#[derive(Debug, Clone)]
struct RegionSvg {
    file_name: String,
    svg: Vec<u8>,
}

fn render_region_svgs(
    parent: &CompileGraph,
    regions: &[OperationRegion],
) -> io::Result<Vec<RegionSvg>> {
    regions
        .iter()
        .enumerate()
        .map(|(region_index, region)| render_region_svg(parent, region_index, region))
        .collect()
}

fn render_region_svg(
    parent: &CompileGraph,
    region_index: usize,
    region: &OperationRegion,
) -> io::Result<RegionSvg> {
    let subgraph = subgraph_from_operations(&parent.graph.h, region.operations.iter().copied())
        .map_err(io::Error::other)?;
    Ok(RegionSvg {
        file_name: region_svg_file_name(region_index, region.kind),
        svg: subgraph_svg(&subgraph)?,
    })
}

fn graph_svg(graph: &Graph) -> io::Result<Vec<u8>> {
    let graph = open_hypergraphs::lax::OpenHypergraph::from_strict(graph.clone());
    to_svg_with(&graph, &dot_options()).map_err(io::Error::other)
}

fn subgraph_svg(subgraph: &Subgraph) -> io::Result<Vec<u8>> {
    graph_svg(&subgraph.open_graph())
}

fn dot_options() -> Options<Obj, hexpr::Operation> {
    let mut options = Options::default().lr();
    options.node_label = Box::new(object_label);
    options.edge_label = Box::new(|operation: &hexpr::Operation| operation.to_string());
    options
}

fn region_svg_file_name(region_index: usize, kind: RegionKind) -> String {
    format!("{region_index:03}-{}.svg", region_kind_name(kind))
}

fn region_kind_name(kind: RegionKind) -> &'static str {
    match kind {
        RegionKind::Data => "data",
        RegionKind::InterleavedControl => "control",
    }
}

fn partition_regions(graph: &Graph) -> Vec<OperationRegion> {
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

        for operation_id in operation_ids(graph) {
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

fn operation_ids(graph: &Graph) -> impl Iterator<Item = OperationId> {
    0..operation_count(graph)
}

fn is_interleaved_control_operation(graph: &Graph, operation_id: OperationId) -> bool {
    matches!(
        operation_kind(operation_name(graph, operation_id)),
        OperationKind::InterleavedControl
    )
}

fn assert_interleaved_control_operations_are_unary(graph: &Graph) {
    for operation_id in operation_ids(graph) {
        if !is_interleaved_control_operation(graph, operation_id) {
            continue;
        }

        let input_count = operation_inputs(graph, operation_id).count();
        let output_count = operation_outputs(graph, operation_id).count();
        assert!(
            input_count == 1 && output_count == 1,
            "analysis expects interleaved control operations to have arity 1 -> 1, but operation #{operation_id} `{}` has arity {input_count} -> {output_count}",
            operation_name(graph, operation_id)
        );
    }
}

fn operation_wires(graph: &Graph, operation_id: OperationId) -> impl Iterator<Item = NodeId> {
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
