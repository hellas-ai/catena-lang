use crate::compile::{
    CompileGraph,
    cfg::{
        control_regions::{ControlRegionGraph, process_control_regions},
        data_regions::{DataRegionGraph, process_data_regions},
        layering::Layer,
        layers::root_layer,
        partition::{OperationRegion, partition_control_regions, partition_regions},
    },
    graph_ops::Graph,
};

const MAX_NESTED_REGION_DEPTH: usize = 16;

pub(super) fn expand_nested_regions(
    definition_context: &CompileGraph,
    regions: &[OperationRegion],
) -> Layer {
    let control_region_graphs = build_control_region_graphs_at_depth(
        definition_context,
        &definition_context.graph,
        regions,
        0,
    );
    root_layer(
        definition_context.graph.clone(),
        regions,
        &control_region_graphs,
    )
}

fn build_control_region_graphs_at_depth(
    definition_context: &CompileGraph,
    parent_graph: &Graph,
    regions: &[OperationRegion],
    depth: usize,
) -> Vec<ControlRegionGraph> {
    partition_control_region_graphs(
        definition_context,
        process_control_regions(definition_context, parent_graph, regions),
        depth,
    )
}

fn build_data_region_graphs_at_depth(
    definition_context: &CompileGraph,
    parent_graph: &Graph,
    regions: &[OperationRegion],
    depth: usize,
) -> Vec<DataRegionGraph> {
    partition_data_region_graphs(
        definition_context,
        process_data_regions(definition_context, parent_graph, regions),
        depth,
    )
}

fn partition_control_region_graphs(
    definition_context: &CompileGraph,
    control_region_graphs: Vec<ControlRegionGraph>,
    depth: usize,
) -> Vec<ControlRegionGraph> {
    control_region_graphs
        .into_iter()
        .map(|mut control_region| {
            control_region.regions = partition_control_regions(&control_region.nested_graph.graph);
            if depth < MAX_NESTED_REGION_DEPTH {
                control_region.data_region_graphs = build_data_region_graphs_at_depth(
                    definition_context,
                    &control_region.nested_graph.graph,
                    &control_region.regions,
                    depth + 1,
                );
            }
            control_region
        })
        .collect()
}

fn partition_data_region_graphs(
    definition_context: &CompileGraph,
    data_region_graphs: Vec<DataRegionGraph>,
    depth: usize,
) -> Vec<DataRegionGraph> {
    data_region_graphs
        .into_iter()
        .map(|mut data_region| {
            data_region.regions = partition_regions(&data_region.nested_graph.graph);
            if depth < MAX_NESTED_REGION_DEPTH {
                data_region.control_region_graphs = build_control_region_graphs_at_depth(
                    definition_context,
                    &data_region.nested_graph.graph,
                    &data_region.regions,
                    depth + 1,
                );
            }
            data_region
        })
        .collect()
}
