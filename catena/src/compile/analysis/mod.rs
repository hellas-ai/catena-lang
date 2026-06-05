mod boundary;
mod partition;
mod render;
mod wires;

use std::path::PathBuf;

use crate::compile::{CompileGraph, CompileTheory};

use self::{
    boundary::BoundaryWires,
    partition::partition_regions,
    render::{graph_svg, render_region_svgs},
    wires::assert_interleaved_control_operations_are_unary,
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
    let mut artifacts = vec![AnalysisArtifact {
        path: PathBuf::from("before-split.svg"),
        contents: before_split.clone(),
    }];
    artifacts.extend(region_svgs.into_iter().map(|region| AnalysisArtifact {
        path: PathBuf::from("regions").join(region.file_name),
        contents: region.svg,
    }));
    Ok(artifacts)
}
