use crate::compile::{
    cfg::{
        control_regions::ControlRegionGraph,
        data_regions::DataRegionGraph,
        layering::{Layer, NestedGraph, NestingMorphism, Region},
        partition::{OperationRegion, RegionKind},
    },
    graph_ops::Graph,
};

pub(super) fn root_layer(
    graph: Graph,
    regions: &[OperationRegion],
    control_region_graphs: &[ControlRegionGraph],
) -> Layer {
    Layer {
        graph,
        regions: regions
            .iter()
            .enumerate()
            .map(|(index, region)| Region {
                index,
                kind: region.kind,
                operations: region.operations.clone(),
                expansion: control_region_graphs
                    .iter()
                    .find(|control_region| control_region.region_index == index)
                    .map(control_layer)
                    .map(Box::new),
            })
            .collect(),
        morphism_to_parent: None,
    }
}

fn control_layer(control_region: &ControlRegionGraph) -> Layer {
    Layer {
        graph: control_region.nested_graph.graph.clone(),
        regions: control_region
            .regions
            .iter()
            .enumerate()
            .map(|(index, region)| Region {
                index,
                kind: region.kind,
                operations: region.operations.clone(),
                expansion: matches!(region.kind, RegionKind::InterleavedData)
                    .then(|| data_expansion_for_region(control_region, index))
                    .flatten()
                    .map(Box::new),
            })
            .collect(),
        morphism_to_parent: Some(morphism_from_nested_graph(&control_region.nested_graph)),
    }
}

fn data_layer(data_region: &DataRegionGraph) -> Layer {
    Layer {
        graph: data_region.nested_graph.graph.clone(),
        regions: data_region
            .regions
            .iter()
            .enumerate()
            .map(|(index, region)| Region {
                index,
                kind: region.kind,
                operations: region.operations.clone(),
                expansion: matches!(region.kind, RegionKind::InterleavedControl)
                    .then(|| control_expansion_for_region(data_region, index))
                    .flatten()
                    .map(Box::new),
            })
            .collect(),
        morphism_to_parent: Some(morphism_from_nested_graph(&data_region.nested_graph)),
    }
}

fn data_expansion_for_region(
    control_region: &ControlRegionGraph,
    region_index: usize,
) -> Option<Layer> {
    control_region
        .data_region_graphs
        .iter()
        .find(|data_region| data_region.region_index == region_index)
        .map(data_layer)
}

fn control_expansion_for_region(
    data_region: &DataRegionGraph,
    region_index: usize,
) -> Option<Layer> {
    data_region
        .control_region_graphs
        .iter()
        .find(|control_region| control_region.region_index == region_index)
        .map(control_layer)
}

fn morphism_from_nested_graph(nested_graph: &NestedGraph) -> NestingMorphism {
    NestingMorphism {
        boundary_relation: nested_graph.boundary_relation.clone(),
    }
}
