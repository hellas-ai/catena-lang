mod artifact_render;
mod build;
mod control_regions;
mod data_regions;
mod layering;
mod layers;
mod model;
mod nested_regions;
mod partition;
mod region_graph;
mod render;
mod value_equivalence;
mod wires;

use crate::compile::{CompileGraph, CompileTheory};

pub use artifact_render::CfgArtifact;
pub use model::{Cfg, CfgError, CfgOptions};
pub(crate) use model::{
    BlockInstruction, CfgEdge, CfgNode, CfgNodeId, Transfer, VariableId, variable_name,
};

use self::{
    artifact_render::render_cfg_artifacts as render_cfg_artifacts_for_layer,
    build::{build_cfg as build_layer_cfg, render_cfg as render_layer_cfg},
    layering::Layer,
    layers::root_layer,
    nested_regions::build_control_region_graphs,
    partition::partition_data_regions,
    wires::assert_interleaved_control_operations_are_unary,
};

fn layer(graph: &CompileGraph) -> Layer {
    assert!(
        matches!(graph.theory, CompileTheory::Data),
        "cfg construction expects a data graph"
    );
    assert_interleaved_control_operations_are_unary(&graph.graph);
    let regions = partition_data_regions(&graph.graph);
    let control_region_graphs = build_control_region_graphs(graph, &graph.graph, &regions);
    root_layer(graph.graph.clone(), &regions, &control_region_graphs)
}

pub fn build_cfg(graph: &CompileGraph, cfg_options: CfgOptions) -> Result<Cfg, CfgError> {
    if !matches!(graph.theory, CompileTheory::Data) {
        return Err(CfgError::UnsupportedTheory(graph.theory.clone()));
    }

    let layer = layer(graph);
    Ok(build_layer_cfg(&layer, graph.source_variable_names.clone(), cfg_options).cfg)
}

pub fn render_cfg(graph: &CompileGraph, cfg_options: CfgOptions) -> Result<Vec<u8>, CfgError> {
    if !matches!(graph.theory, CompileTheory::Data) {
        return Err(CfgError::UnsupportedTheory(graph.theory.clone()));
    }

    let layer = layer(graph);
    Ok(render_layer_cfg(
        &layer,
        graph.source_variable_names.clone(),
        cfg_options,
    ))
}

pub fn render_cfg_artifacts(
    graph: &CompileGraph,
    cfg_options: CfgOptions,
) -> std::io::Result<Vec<CfgArtifact>> {
    if !matches!(graph.theory, CompileTheory::Data) {
        return Err(std::io::Error::other(CfgError::UnsupportedTheory(
            graph.theory.clone(),
        )));
    }

    let layer = layer(graph);
    render_cfg_artifacts_for_layer(graph, &layer, cfg_options)
}
