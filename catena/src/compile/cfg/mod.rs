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
pub(crate) use model::{
    BlockInstruction, CfgEdge, CfgNode, CfgNodeId, Transfer, VariableId, variable_name,
};
pub use model::{Cfg, CfgArtifacts, CfgBuild, CfgError, CfgOptions};

use self::{
    artifact_render::render_cfg_artifacts as render_cfg_artifacts_for_layer,
    build::build_cfg_from_layer, layers::root_layer, nested_regions::build_control_region_graphs,
    partition::partition_data_regions, render::render_cfg_build,
    wires::assert_interleaved_control_operations_are_unary,
};

pub fn build_cfg(graph: &CompileGraph, cfg_options: CfgOptions) -> Result<CfgBuild, CfgError> {
    if !matches!(graph.theory, CompileTheory::Data) {
        return Err(CfgError::UnsupportedTheory(graph.theory.clone()));
    }

    assert_interleaved_control_operations_are_unary(&graph.graph);

    let data_regions = partition_data_regions(&graph.graph);
    let nested_control_regions = build_control_region_graphs(graph, &graph.graph, &data_regions);
    let root_layer = root_layer(graph.graph.clone(), &data_regions, &nested_control_regions);
    Ok(build_cfg_from_layer(
        graph,
        &root_layer,
        graph.source_variable_names.clone(),
        cfg_options,
    ))
}

pub fn render_cfg(cfg_build: &CfgBuild) -> Vec<u8> {
    render_cfg_build(cfg_build)
}

pub fn render_cfg_artifacts(cfg_artifacts: &CfgArtifacts) -> std::io::Result<Vec<CfgArtifact>> {
    render_cfg_artifacts_for_layer(cfg_artifacts)
}
