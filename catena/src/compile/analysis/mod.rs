mod cfg;
mod cfg_render;
mod control_regions;
mod data_regions;
mod layering;
mod layers;
mod nested_regions;
mod partition;
mod region_graph;
mod render;
mod wires;

use std::{fmt::Write, path::PathBuf};

use crate::compile::{CompileGraph, CompileTheory};

use self::{
    cfg::render_cfg,
    layers::root_layer,
    nested_regions::build_control_region_graphs,
    partition::{OperationRegion, RegionKind, partition_data_regions},
    region_graph::{region_graph, region_graph_trace},
    render::{graph_svg, render_graph_region_svgs, render_region_svgs},
    wires::assert_interleaved_control_operations_are_unary,
};

pub use layering::{Layer, NestingMorphism, Region};

pub fn render_analysis(graph: &CompileGraph) -> std::io::Result<Vec<u8>> {
    Ok(render_analysis_artifacts(graph)?
        .into_iter()
        .find(|artifact| artifact.path == PathBuf::from("source.svg"))
        .expect("analysis artifacts include source graph")
        .contents)
}

#[derive(Debug, Clone)]
pub struct AnalysisArtifact {
    pub path: PathBuf,
    pub contents: Vec<u8>,
}

pub fn layer(graph: &CompileGraph) -> Layer {
    assert!(
        matches!(graph.theory, CompileTheory::Data),
        "analysis expects a data graph"
    );
    assert_interleaved_control_operations_are_unary(&graph.graph);
    let regions = partition_data_regions(&graph.graph);
    let control_region_graphs = build_control_region_graphs(graph, &graph.graph, &regions);
    root_layer(graph.graph.clone(), &regions, &control_region_graphs)
}

pub fn render_analysis_artifacts(graph: &CompileGraph) -> std::io::Result<Vec<AnalysisArtifact>> {
    assert!(
        matches!(graph.theory, CompileTheory::Data),
        "analysis expects a data graph"
    );

    // I don't know if it is too strict, but I cannot imagine a case when it is not true
    // better fail early and loud if I am wrong!
    assert_interleaved_control_operations_are_unary(&graph.graph);
    let regions = partition_data_regions(&graph.graph);
    let region_svgs = render_region_svgs(graph, &regions)?;
    let control_region_graphs = build_control_region_graphs(graph, &graph.graph, &regions);
    let layer = root_layer(graph.graph.clone(), &regions, &control_region_graphs);
    let source = graph_svg(&graph.graph)?;
    let region_graph = graph_svg(&region_graph(&layer))?;
    let region_graph_trace = region_graph_trace(&layer);
    let cfg = render_cfg(&layer);
    let mut artifacts = vec![
        analysis_index_artifact(graph, &layer),
        AnalysisArtifact {
            path: PathBuf::from("source.svg"),
            contents: source,
        },
        AnalysisArtifact {
            path: PathBuf::from("cfg.txt"),
            contents: cfg,
        },
        AnalysisArtifact {
            path: PathBuf::from("region-graph.svg"),
            contents: region_graph,
        },
        AnalysisArtifact {
            path: PathBuf::from("region-graph.txt"),
            contents: region_graph_trace,
        },
    ];
    artifacts.extend(region_svgs.into_iter().map(|region| AnalysisArtifact {
        path: PathBuf::from("regions").join(region.file_name),
        contents: region.svg,
    }));
    for region in &layer.regions {
        if let Some(expansion) = &region.expansion {
            render_layer_expansion_artifacts(
                &mut artifacts,
                PathBuf::from("control-regions"),
                region.index,
                expansion,
            )?;
        }
    }
    Ok(artifacts)
}

fn analysis_index_artifact(graph: &CompileGraph, layer: &Layer) -> AnalysisArtifact {
    let mut index = String::new();
    index.push_str("# Analysis\n\n");
    index.push_str("- [source graph](source.svg)\n");
    index.push_str("- [cfg](cfg.txt)\n");
    index.push_str("- [region graph](region-graph.svg)\n");
    index.push_str("- [region graph trace](region-graph.txt)\n");
    append_item(&mut index, 1, "partitions");
    append_source_regions_index(&mut index, graph, &layer.regions);
    append_item(&mut index, 1, "expansions");
    append_layer_expansion_index(&mut index, 2, PathBuf::from("control-regions"), layer);

    AnalysisArtifact {
        path: PathBuf::from("index.md"),
        contents: index.into_bytes(),
    }
}

fn render_layer_expansion_artifacts(
    artifacts: &mut Vec<AnalysisArtifact>,
    base: PathBuf,
    expansion_region_index: usize,
    layer: &Layer,
) -> std::io::Result<()> {
    artifacts.push(AnalysisArtifact {
        path: base.join(format!("{expansion_region_index:03}-resolved.svg")),
        contents: graph_svg(&layer.graph)?,
    });

    if has_interleaved_regions(&layer.regions) {
        artifacts.extend(
            render_graph_region_svgs(&layer.graph, &operation_regions(&layer.regions))?
                .into_iter()
                .map(|region| AnalysisArtifact {
                    path: base
                        .join(format!("{expansion_region_index:03}-regions"))
                        .join(region.file_name),
                    contents: region.svg,
                }),
        );
    }

    for region in &layer.regions {
        if let Some(expansion) = &region.expansion {
            render_layer_expansion_artifacts(
                artifacts,
                base.join(region_expansion_base_dir(
                    expansion_region_index,
                    region.kind,
                )),
                region.index,
                expansion,
            )?;
        }
    }

    Ok(())
}

fn append_layer_expansion_index(index: &mut String, depth: usize, base: PathBuf, layer: &Layer) {
    for region in &layer.regions {
        let Some(expansion) = &region.expansion else {
            continue;
        };
        let resolved = base.join(format!("{:03}-resolved.svg", region.index));
        append_linked_item(
            index,
            depth,
            &region_operations_label(&layer.graph, region),
            resolved,
        );
        append_item(
            index,
            depth + 1,
            &format!(
                "contains {}",
                operation_summary(
                    &expansion
                        .graph
                        .h
                        .x
                        .0
                        .0
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                )
            ),
        );
        append_layer_expansion_index(
            index,
            depth + 1,
            base.join(nested_expansion_base_dir(region.index, region.kind)),
            expansion,
        );
    }
}

fn append_source_regions_index(index: &mut String, graph: &CompileGraph, regions: &[Region]) {
    for region in regions {
        let file_name = format!(
            "{:03}-{}.svg",
            region.index,
            region_kind_file_name(region.kind)
        );
        append_linked_item(
            index,
            2,
            &format!(
                "{} ({})",
                region_operations_label(&graph.graph, region),
                operation_count_label(region.operations.len()),
            ),
            PathBuf::from("regions").join(file_name),
        );
    }
}

fn append_item(index: &mut String, depth: usize, label: &str) {
    writeln!(index, "{}- {label}", "  ".repeat(depth)).expect("write to string cannot fail");
}

fn append_linked_item(index: &mut String, depth: usize, label: &str, path: PathBuf) {
    writeln!(
        index,
        "{}- [{label}]({})",
        "  ".repeat(depth),
        path.display()
    )
    .expect("write to string cannot fail");
}

fn has_interleaved_regions(regions: &[Region]) -> bool {
    regions.iter().any(|region| {
        matches!(
            region.kind,
            RegionKind::InterleavedData | RegionKind::InterleavedControl
        )
    })
}

fn region_kind_file_name(kind: RegionKind) -> &'static str {
    match kind {
        RegionKind::Data => "data",
        RegionKind::InterleavedControl => "control",
        RegionKind::Control => "native-control",
        RegionKind::InterleavedData => "interleaved-data",
    }
}

fn operation_count_label(count: usize) -> String {
    if count == 1 {
        "1 operation".to_string()
    } else {
        format!("{count} operations")
    }
}

fn region_operations_label(graph: &crate::compile::graph_ops::Graph, region: &Region) -> String {
    region
        .operations
        .iter()
        .copied()
        .map(|operation| graph.h.x.0.0[operation].to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

fn operation_regions(regions: &[Region]) -> Vec<OperationRegion> {
    regions
        .iter()
        .map(|region| OperationRegion {
            kind: region.kind,
            operations: region.operations.clone(),
        })
        .collect()
}

fn region_expansion_base_dir(expansion_region_index: usize, kind: RegionKind) -> PathBuf {
    match kind {
        RegionKind::InterleavedControl => {
            PathBuf::from(format!("{expansion_region_index:03}-control-regions"))
        }
        RegionKind::InterleavedData => {
            PathBuf::from(format!("{expansion_region_index:03}-data-regions"))
        }
        RegionKind::Data | RegionKind::Control => {
            panic!("non-interleaved regions do not have expansion directories")
        }
    }
}

fn nested_expansion_base_dir(expansion_region_index: usize, kind: RegionKind) -> PathBuf {
    match kind {
        RegionKind::InterleavedControl => {
            PathBuf::from(format!("{expansion_region_index:03}-data-regions"))
        }
        RegionKind::InterleavedData => {
            PathBuf::from(format!("{expansion_region_index:03}-control-regions"))
        }
        RegionKind::Data | RegionKind::Control => {
            panic!("non-interleaved regions do not have expansion directories")
        }
    }
}

fn operation_summary(operations: &[String]) -> String {
    const MAX_OPERATIONS: usize = 8;
    let mut names = operations
        .iter()
        .take(MAX_OPERATIONS)
        .cloned()
        .collect::<Vec<_>>();
    if operations.len() > MAX_OPERATIONS {
        names.push(format!("... +{} more", operations.len() - MAX_OPERATIONS));
    }
    format!(
        "{} ({})",
        names.join(", "),
        operation_count_label(operations.len())
    )
}
