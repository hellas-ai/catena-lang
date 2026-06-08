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
    build::lower_region_graph_to_cfg, nested_regions::expand_nested_regions,
    partition::partition_regions, region_graph::lower_layer_to_region_graph,
    render::render_cfg_build, value_equivalence::compute_value_equivalences,
};

pub fn build_cfg(graph: &CompileGraph, cfg_options: CfgOptions) -> Result<CfgBuild, CfgError> {
    if !matches!(graph.theory, CompileTheory::Data) {
        return Err(CfgError::UnsupportedTheory(graph.theory.clone()));
    }

    let regions = partition_regions(&graph.graph);
    let layer = expand_nested_regions(graph, &regions);
    let region_graph = lower_layer_to_region_graph(&layer);
    let values = compute_value_equivalences(&region_graph);

    Ok(lower_region_graph_to_cfg(
        graph,
        &layer,
        &region_graph,
        &values,
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
