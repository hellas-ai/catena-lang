mod control_regions;
mod data_regions;
mod layering;
mod nested_regions;
mod partition;
mod render;
mod wires;

use std::{fmt::Write, path::PathBuf};

use crate::compile::{CompileGraph, CompileTheory};

use self::{
    nested_regions::build_control_region_graphs,
    partition::{
        OperationId, OperationRegion, RegionKind, SourceOperation, partition_data_regions,
    },
    render::{graph_svg, render_graph_region_svgs, render_region_svgs},
    wires::assert_interleaved_control_operations_are_unary,
};

pub use control_regions::ControlRegionGraph;
pub use data_regions::DataRegionGraph;
pub use layering::NestedGraph;

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

pub fn control_region_graphs(graph: &CompileGraph) -> Vec<ControlRegionGraph> {
    assert!(
        matches!(graph.theory, CompileTheory::Data),
        "analysis expects a data graph"
    );
    assert_interleaved_control_operations_are_unary(&graph.graph);
    let regions = partition_data_regions(&graph.graph);
    build_control_region_graphs(graph, &graph.graph, &regions)
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
    let source = graph_svg(&graph.graph)?;
    let mut artifacts = vec![
        analysis_index_artifact(graph, &regions, &control_region_graphs),
        AnalysisArtifact {
            path: PathBuf::from("source.svg"),
            contents: source,
        },
    ];
    artifacts.extend(region_svgs.into_iter().map(|region| AnalysisArtifact {
        path: PathBuf::from("regions").join(region.file_name),
        contents: region.svg,
    }));
    for control_region in &control_region_graphs {
        render_control_region_graph_artifacts(
            &mut artifacts,
            PathBuf::from("control-regions"),
            control_region,
        )?;
    }
    Ok(artifacts)
}

fn analysis_index_artifact(
    graph: &CompileGraph,
    regions: &[OperationRegion],
    control_region_graphs: &[ControlRegionGraph],
) -> AnalysisArtifact {
    let mut index = String::new();
    index.push_str("# Analysis\n\n");
    index.push_str("- [source graph](source.svg)\n");
    append_item(&mut index, 1, "partitions");
    append_source_regions_index(&mut index, graph, regions);
    append_item(&mut index, 1, "expansions");
    for control_region in control_region_graphs {
        append_control_region_index(
            &mut index,
            2,
            PathBuf::from("control-regions"),
            None,
            control_region,
        );
    }

    AnalysisArtifact {
        path: PathBuf::from("index.md"),
        contents: index.into_bytes(),
    }
}

fn render_control_region_graph_artifacts(
    artifacts: &mut Vec<AnalysisArtifact>,
    base: PathBuf,
    control_region: &ControlRegionGraph,
) -> std::io::Result<()> {
    artifacts.push(AnalysisArtifact {
        path: base.join(format!("{:03}-resolved.svg", control_region.region_index)),
        contents: graph_svg(&control_region.nested_graph.graph)?,
    });
    if has_interleaved_data_regions(&control_region.regions) {
        artifacts.extend(
            render_graph_region_svgs(&control_region.nested_graph.graph, &control_region.regions)?
                .into_iter()
                .map(|region| AnalysisArtifact {
                    path: base
                        .join(format!("{:03}-regions", control_region.region_index))
                        .join(region.file_name),
                    contents: region.svg,
                }),
        );
    }

    let data_base = base.join(format!("{:03}-data-regions", control_region.region_index));
    for data_region in &control_region.data_region_graphs {
        render_data_region_graph_artifacts(artifacts, data_base.clone(), data_region)?;
    }

    Ok(())
}

fn append_control_region_index(
    index: &mut String,
    depth: usize,
    base: PathBuf,
    opened_by: Option<&SourceOperation>,
    control_region: &ControlRegionGraph,
) {
    let resolved = base.join(format!("{:03}-resolved.svg", control_region.region_index));
    append_expanded_graph_node(
        index,
        depth,
        opened_by,
        &control_region.source_operations,
        &control_region.nested_graph,
        resolved,
    );

    let data_base = base.join(format!("{:03}-data-regions", control_region.region_index));
    for source_operation in &control_region.source_operations {
        for data_region in control_region
            .data_region_graphs
            .iter()
            .filter(|data_region| {
                region_belongs_to_parent_operation(
                    *data_region,
                    &control_region.nested_graph,
                    source_operation.id,
                )
            })
        {
            append_data_region_index(
                index,
                depth + 1,
                data_base.clone(),
                Some(source_operation),
                data_region,
            );
        }
    }
}

fn render_data_region_graph_artifacts(
    artifacts: &mut Vec<AnalysisArtifact>,
    base: PathBuf,
    data_region: &DataRegionGraph,
) -> std::io::Result<()> {
    artifacts.push(AnalysisArtifact {
        path: base.join(format!("{:03}-resolved.svg", data_region.region_index)),
        contents: graph_svg(&data_region.nested_graph.graph)?,
    });
    if has_interleaved_control_regions(&data_region.regions) {
        artifacts.extend(
            render_graph_region_svgs(&data_region.nested_graph.graph, &data_region.regions)?
                .into_iter()
                .map(|region| AnalysisArtifact {
                    path: base
                        .join(format!("{:03}-regions", data_region.region_index))
                        .join(region.file_name),
                    contents: region.svg,
                }),
        );
    }

    let control_base = base.join(format!("{:03}-control-regions", data_region.region_index));
    for control_region in &data_region.control_region_graphs {
        render_control_region_graph_artifacts(artifacts, control_base.clone(), control_region)?;
    }

    Ok(())
}

fn append_data_region_index(
    index: &mut String,
    depth: usize,
    base: PathBuf,
    opened_by: Option<&SourceOperation>,
    data_region: &DataRegionGraph,
) {
    let resolved = base.join(format!("{:03}-resolved.svg", data_region.region_index));
    append_expanded_graph_node(
        index,
        depth,
        opened_by,
        &data_region.source_operations,
        &data_region.nested_graph,
        resolved,
    );

    let control_base = base.join(format!("{:03}-control-regions", data_region.region_index));
    for source_operation in &data_region.source_operations {
        for control_region in data_region
            .control_region_graphs
            .iter()
            .filter(|control_region| {
                region_belongs_to_parent_operation(
                    *control_region,
                    &data_region.nested_graph,
                    source_operation.id,
                )
            })
        {
            append_control_region_index(
                index,
                depth + 1,
                control_base.clone(),
                Some(source_operation),
                control_region,
            );
        }
    }
}

fn append_source_regions_index(
    index: &mut String,
    graph: &CompileGraph,
    regions: &[OperationRegion],
) {
    for (region_index, region) in regions.iter().enumerate() {
        let file_name = format!(
            "{region_index:03}-{}.svg",
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

fn append_expanded_graph_node(
    index: &mut String,
    depth: usize,
    opened_by: Option<&SourceOperation>,
    source_operations: &[SourceOperation],
    nested_graph: &NestedGraph,
    path: PathBuf,
) {
    let relation = opened_by
        .map(|operation| format!(" opened by `{}`", operation.name))
        .unwrap_or_default();
    append_linked_item(
        index,
        depth,
        &format!(
            "{}{}",
            source_operation_list_label(source_operations),
            relation
        ),
        path,
    );
    append_item(
        index,
        depth + 1,
        &format!(
            "contains {}",
            operation_summary(
                &nested_graph
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

trait SourceRegionGraph {
    fn source_operations(&self) -> &[SourceOperation];
}

impl SourceRegionGraph for ControlRegionGraph {
    fn source_operations(&self) -> &[SourceOperation] {
        &self.source_operations
    }
}

impl SourceRegionGraph for DataRegionGraph {
    fn source_operations(&self) -> &[SourceOperation] {
        &self.source_operations
    }
}

fn region_belongs_to_parent_operation(
    region: &impl SourceRegionGraph,
    parent: &NestedGraph,
    parent_operation: OperationId,
) -> bool {
    region
        .source_operations()
        .first()
        .and_then(|source_operation| parent.parent_operations.get(source_operation.id))
        .is_some_and(|mapped_parent| *mapped_parent == parent_operation)
}

fn has_interleaved_data_regions(regions: &[OperationRegion]) -> bool {
    regions
        .iter()
        .any(|region| matches!(region.kind, RegionKind::InterleavedData))
}

fn has_interleaved_control_regions(regions: &[OperationRegion]) -> bool {
    regions
        .iter()
        .any(|region| matches!(region.kind, RegionKind::InterleavedControl))
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

fn region_operations_label(
    graph: &crate::compile::graph_ops::Graph,
    region: &OperationRegion,
) -> String {
    region
        .operations
        .iter()
        .copied()
        .map(|operation| graph.h.x.0.0[operation].to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

fn source_operation_list_label(source_operations: &[SourceOperation]) -> String {
    source_operations
        .iter()
        .map(|operation| operation.name.as_str())
        .collect::<Vec<_>>()
        .join(", ")
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
