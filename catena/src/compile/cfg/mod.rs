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

use std::{fmt::Write, path::PathBuf};

use crate::compile::{CompileGraph, CompileTheory};

pub use layering::{Layer, NestingMorphism, Region};
pub use model::{
    BlockInstruction, Cfg, CfgEdge, CfgError, CfgNode, CfgNodeId, CfgOptions, OperationId,
    OperationName, Transfer, VariableId, VariableName, variable_name,
};

use self::{
    artifact_render::{graph_svg, render_graph_region_svgs, render_region_svgs},
    build::{build_cfg as build_analysis_cfg, render_cfg as render_analysis_cfg},
    layers::root_layer,
    nested_regions::build_control_region_graphs,
    partition::{OperationRegion, RegionKind, partition_data_regions},
    region_graph::{region_graph, region_graph_trace},
    value_equivalence::value_equivalence_trace,
    wires::assert_interleaved_control_operations_are_unary,
};

pub fn render_analysis(graph: &CompileGraph) -> std::io::Result<Vec<u8>> {
    Ok(render_analysis_artifacts(graph, CfgOptions::default())?
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
        "cfg analysis expects a data graph"
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
    Ok(build_analysis_cfg(&layer, graph.source_variable_names.clone(), cfg_options).cfg)
}

pub fn render_cfg(graph: &CompileGraph, cfg_options: CfgOptions) -> Result<Vec<u8>, CfgError> {
    if !matches!(graph.theory, CompileTheory::Data) {
        return Err(CfgError::UnsupportedTheory(graph.theory.clone()));
    }

    let layer = layer(graph);
    Ok(render_analysis_cfg(
        &layer,
        graph.source_variable_names.clone(),
        cfg_options,
    ))
}

pub fn render_analysis_artifacts(
    graph: &CompileGraph,
    cfg_options: CfgOptions,
) -> std::io::Result<Vec<AnalysisArtifact>> {
    assert!(
        matches!(graph.theory, CompileTheory::Data),
        "cfg analysis expects a data graph"
    );

    assert_interleaved_control_operations_are_unary(&graph.graph);
    let regions = partition_data_regions(&graph.graph);
    let region_svgs = render_region_svgs(graph, &regions)?;
    let control_region_graphs = build_control_region_graphs(graph, &graph.graph, &regions);
    let layer = root_layer(graph.graph.clone(), &regions, &control_region_graphs);
    let source = graph_svg(&graph.graph)?;
    let region_graph = graph_svg(&region_graph(&layer))?;
    let region_graph_trace = region_graph_trace(&layer);
    let value_equivalence_trace = value_equivalence_trace(&layer);
    let cfg = render_analysis_cfg(&layer, graph.source_variable_names.clone(), cfg_options);
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
        AnalysisArtifact {
            path: PathBuf::from("value-equivalence.txt"),
            contents: value_equivalence_trace,
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
    index.push_str("- [value equivalence](value-equivalence.txt)\n");
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
                    &expansion.graph,
                    &operation_regions(&expansion.regions),
                    usize::MAX
                )
            ),
        );
        append_layer_expansion_index(
            index,
            depth + 1,
            base.join(region_expansion_base_dir(region.index, region.kind)),
            expansion,
        );
    }
}

fn append_source_regions_index(index: &mut String, graph: &CompileGraph, regions: &[Region]) {
    for region in regions {
        let base = PathBuf::from("regions").join(format!(
            "{:03}-{}.svg",
            region.index,
            region_operations_file_stem(&graph.graph, region),
        ));
        append_linked_item(
            index,
            2,
            &region_operations_label(&graph.graph, region),
            base,
        );
        append_item(
            index,
            3,
            &format!("kind {}", region_kind_label(region.kind)),
        );
    }
}

fn append_linked_item(index: &mut String, depth: usize, label: &str, path: PathBuf) {
    append_item(index, depth, &format!("[{label}]({})", path.display()));
}

fn append_item(index: &mut String, depth: usize, label: &str) {
    writeln!(index, "{} {label}", "#".repeat(depth)).expect("write string");
}

fn region_expansion_base_dir(region_index: usize, kind: RegionKind) -> PathBuf {
    PathBuf::from(format!(
        "{region_index:03}-{}-expansion",
        region_kind_file_label(kind)
    ))
}

fn has_interleaved_regions(regions: &[Region]) -> bool {
    regions.iter().any(|region| {
        matches!(
            region.kind,
            RegionKind::InterleavedControl | RegionKind::InterleavedData
        )
    })
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

fn operation_summary(
    graph: &crate::compile::graph_ops::Graph,
    regions: &[OperationRegion],
    max_items: usize,
) -> String {
    let mut labels = regions
        .iter()
        .enumerate()
        .map(|(index, region)| {
            let operations = region
                .operations
                .iter()
                .map(|operation| crate::compile::graph_ops::operation_name(graph, *operation))
                .collect::<Vec<_>>()
                .join(", ");
            format!(
                "{} region {} [{}]",
                region_kind_label(region.kind),
                index,
                operations
            )
        })
        .collect::<Vec<_>>();
    if labels.len() > max_items {
        labels.truncate(max_items);
        labels.push("...".to_string());
    }
    labels.join("; ")
}

fn region_operations_label(graph: &crate::compile::graph_ops::Graph, region: &Region) -> String {
    format!(
        "{} region {} [{}]",
        region_kind_label(region.kind),
        region.index,
        region
            .operations
            .iter()
            .map(|operation| { crate::compile::graph_ops::operation_name(graph, *operation) })
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn region_operations_file_stem(
    graph: &crate::compile::graph_ops::Graph,
    region: &Region,
) -> String {
    let mut stem = format!(
        "{}-{}",
        region_kind_file_label(region.kind),
        region
            .operations
            .iter()
            .map(|operation| {
                sanitize_file_component(crate::compile::graph_ops::operation_name(
                    graph, *operation,
                ))
            })
            .collect::<Vec<_>>()
            .join("-"),
    );
    if stem.len() > 80 {
        stem.truncate(80);
    }
    stem
}

fn sanitize_file_component(label: &str) -> String {
    label
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '-'
            }
        })
        .collect()
}

fn region_kind_label(kind: RegionKind) -> &'static str {
    match kind {
        RegionKind::Data => "data",
        RegionKind::Control => "control",
        RegionKind::InterleavedControl => "interleaved control",
        RegionKind::InterleavedData => "interleaved data",
    }
}

fn region_kind_file_label(kind: RegionKind) -> &'static str {
    match kind {
        RegionKind::Data => "data",
        RegionKind::Control => "control",
        RegionKind::InterleavedControl => "interleaved-control",
        RegionKind::InterleavedData => "interleaved-data",
    }
}
