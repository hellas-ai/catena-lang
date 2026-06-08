use std::{fmt::Write, io, path::PathBuf};

use open_hypergraphs_dot::{Options, svg::to_svg_with};

use crate::{
    compile::{
        CompileGraph,
        cfg::{
            CfgArtifacts,
            layering::{Layer, Region},
            partition::{OperationRegion, RegionKind},
            region_graph::{region_graph, region_graph_trace},
            render::render_cfg_parts,
            value_equivalence::value_equivalence_trace,
        },
        graph_ops::{Graph, operation_name},
        graph_render::object_label,
    },
    hypergraph::subgraph::{Subgraph, subgraph_from_operations},
    lang::Obj,
};

pub(super) fn render_cfg_artifacts(cfg_artifacts: &CfgArtifacts) -> io::Result<Vec<CfgArtifact>> {
    let graph = &cfg_artifacts.graph;
    let layer = &cfg_artifacts.layer;
    let regions = operation_regions(&layer.regions);
    let region_svgs = render_region_svgs(graph, &regions)?;
    let source = graph_svg(&graph.graph)?;
    let region_graph = graph_svg(&region_graph(layer))?;
    let region_graph_trace = region_graph_trace(layer);
    let value_equivalence_trace = value_equivalence_trace(layer);
    let cfg = render_cfg_parts(
        &cfg_artifacts.graph.graph,
        &cfg_artifacts.cfg,
        &cfg_artifacts.globals,
        &cfg_artifacts.wire_names,
        &cfg_artifacts.block_svg_paths,
    );
    let mut artifacts = vec![
        cfg_index_artifact(graph, layer),
        CfgArtifact {
            path: PathBuf::from("source.svg"),
            contents: source,
        },
        CfgArtifact {
            path: PathBuf::from("cfg.txt"),
            contents: cfg,
        },
        CfgArtifact {
            path: PathBuf::from("region-graph.svg"),
            contents: region_graph,
        },
        CfgArtifact {
            path: PathBuf::from("region-graph.txt"),
            contents: region_graph_trace,
        },
        CfgArtifact {
            path: PathBuf::from("value-equivalence.txt"),
            contents: value_equivalence_trace,
        },
    ];
    artifacts.extend(region_svgs.into_iter().map(|region| CfgArtifact {
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

#[derive(Debug, Clone)]
pub struct CfgArtifact {
    pub path: PathBuf,
    pub contents: Vec<u8>,
}

#[derive(Debug, Clone)]
pub(super) struct RegionSvg {
    pub(super) file_name: String,
    pub(super) svg: Vec<u8>,
}

fn render_region_svgs(
    parent: &CompileGraph,
    regions: &[OperationRegion],
) -> io::Result<Vec<RegionSvg>> {
    render_graph_region_svgs(&parent.graph, regions)
}

fn render_graph_region_svgs(
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

fn graph_svg(graph: &Graph) -> io::Result<Vec<u8>> {
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

fn cfg_index_artifact(graph: &CompileGraph, layer: &Layer) -> CfgArtifact {
    let mut index = String::new();
    index.push_str("# CFG Artifacts\n\n");
    index.push_str("- [source graph](source.svg)\n");
    index.push_str("- [cfg](cfg.txt)\n");
    index.push_str("- [region graph](region-graph.svg)\n");
    index.push_str("- [region graph trace](region-graph.txt)\n");
    index.push_str("- [value equivalence](value-equivalence.txt)\n");
    append_item(&mut index, 1, "partitions");
    append_source_regions_index(&mut index, graph, &layer.regions);
    append_item(&mut index, 1, "expansions");
    append_layer_expansion_index(&mut index, 2, PathBuf::from("control-regions"), layer);

    CfgArtifact {
        path: PathBuf::from("index.md"),
        contents: index.into_bytes(),
    }
}

fn render_layer_expansion_artifacts(
    artifacts: &mut Vec<CfgArtifact>,
    base: PathBuf,
    expansion_region_index: usize,
    layer: &Layer,
) -> io::Result<()> {
    artifacts.push(CfgArtifact {
        path: base.join(format!("{expansion_region_index:03}-resolved.svg")),
        contents: graph_svg(&layer.graph)?,
    });

    if has_interleaved_regions(&layer.regions) {
        artifacts.extend(
            render_graph_region_svgs(&layer.graph, &operation_regions(&layer.regions))?
                .into_iter()
                .map(|region| CfgArtifact {
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

fn operation_summary(graph: &Graph, regions: &[OperationRegion], max_items: usize) -> String {
    let mut labels = regions
        .iter()
        .enumerate()
        .map(|(index, region)| {
            let operations = region
                .operations
                .iter()
                .map(|operation| operation_name(graph, *operation))
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

fn region_operations_label(graph: &Graph, region: &Region) -> String {
    format!(
        "{} region {} [{}]",
        region_kind_label(region.kind),
        region.index,
        region
            .operations
            .iter()
            .map(|operation| operation_name(graph, *operation))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn region_operations_file_stem(graph: &Graph, region: &Region) -> String {
    let mut stem = format!(
        "{}-{}",
        region_kind_file_label(region.kind),
        region
            .operations
            .iter()
            .map(|operation| sanitize_file_component(operation_name(graph, *operation)))
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
