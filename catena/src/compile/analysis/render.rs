use std::io;

use open_hypergraphs_dot::{Options, svg::to_svg_with};

use crate::{
    compile::{
        CompileGraph,
        analysis::partition::{OperationRegion, RegionKind},
        graph_ops::Graph,
        graph_render::object_label,
    },
    hypergraph::subgraph::{Subgraph, subgraph_from_operations},
    lang::Obj,
};

#[derive(Debug, Clone)]
pub(super) struct RegionSvg {
    pub(super) file_name: String,
    pub(super) svg: Vec<u8>,
}

pub(super) fn render_region_svgs(
    parent: &CompileGraph,
    regions: &[OperationRegion],
) -> io::Result<Vec<RegionSvg>> {
    render_graph_region_svgs(&parent.graph, regions)
}

pub(super) fn render_graph_region_svgs(
    graph: &Graph,
    regions: &[OperationRegion],
) -> io::Result<Vec<RegionSvg>> {
    regions
        .iter()
        .enumerate()
        .map(|(region_index, region)| render_region_svg(graph, region_index, region))
        .collect()
}

fn render_region_svg(
    graph: &Graph,
    region_index: usize,
    region: &OperationRegion,
) -> io::Result<RegionSvg> {
    let subgraph = subgraph_from_operations(&graph.h, region.operations.iter().copied())
        .map_err(io::Error::other)?;
    Ok(RegionSvg {
        file_name: region_svg_file_name(region_index, region.kind),
        svg: subgraph_svg(&subgraph)?,
    })
}

pub(super) fn graph_svg(graph: &Graph) -> io::Result<Vec<u8>> {
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
        RegionKind::Control => "native-control",
        RegionKind::InterleavedData => "interleaved-data",
    }
}
