use std::{collections::HashMap, fmt, fs, io, path::Path};

use hexpr::Operation;
use metacat::theory::{Theory, TheoryId};
use metacat::tree::Tree;
use open_hypergraphs::lax::{NodeId, OpenHypergraph};
use open_hypergraphs_dot::{Options, svg::to_svg_with};

use crate::{
    closure::{Conversion, definition::closure_operation, region::ClosureRegion},
    pass::forget_closures::{ClosureForgotten, ClosureForgottenTerm},
    report::CompileReport,
};

/// Render a list of SVGs for each definition being compiled, one for each transformation phase.
pub fn dump_svgs(report: &CompileReport, dir: &Path) -> io::Result<()> {
    let Some(theory_set) = &report.theory_set else {
        return Ok(());
    };

    fs::create_dir_all(dir)?;

    for (theory_id, theory) in &theory_set.theories {
        let Theory::Theory { syntax, arrows } = theory else {
            continue;
        };
        let syntax_theory = theory_set
            .theories
            .get(syntax)
            .ok_or_else(|| invalid_data(format!("missing syntax theory `{syntax}`")))?;

        for (definition_name, arrow) in arrows {
            let Some(term) = &arrow.definition else {
                continue;
            };

            let definition_dir = dir.join(qualified_definition_dir(theory_id, definition_name));
            fs::create_dir_all(&definition_dir)?;

            let elaborated_svg = render_untyped_svg(term).map_err(|error| {
                io::Error::new(
                    error.kind(),
                    format!(
                        "failed to render elaborated svg for `{theory_id}.{definition_name}`: {error}"
                    ),
                )
            })?;
            let elaborated_path = definition_dir.join("elaborated.svg");
            fs::write(&elaborated_path, elaborated_svg).map_err(|error| {
                io::Error::new(
                    error.kind(),
                    format!("failed to write {}: {error}", elaborated_path.display()),
                )
            })?;
            dump_untyped_stage_hex(term, "elaborated", &definition_dir)?;

            if let Some(node_types) = report
                .definition_types
                .as_ref()
                .and_then(|defs| defs.get(theory_id))
                .and_then(|defs| defs.get(definition_name))
            {
                let svg = render_check_result_svg(term, node_types, syntax_theory).map_err(|error| {
                    io::Error::new(
                        error.kind(),
                        format!(
                            "failed to render checked svg for `{theory_id}.{definition_name}`: {error}"
                        ),
                    )
                })?;

                let checked_path = definition_dir.join("checked.svg");
                fs::write(&checked_path, svg).map_err(|error| {
                    io::Error::new(
                        error.kind(),
                        format!("failed to write {}: {error}", checked_path.display()),
                    )
                })?;
                dump_untyped_stage_hex(term, "checked", &definition_dir)?;
            }

            if let Some(node_types) = report
                .partial_definition_types
                .as_ref()
                .and_then(|defs| defs.get(theory_id))
                .and_then(|defs| defs.get(definition_name))
            {
                let svg = render_partial_check_result_svg(term, node_types, syntax_theory).map_err(|error| {
                    io::Error::new(
                        error.kind(),
                        format!(
                            "failed to render partial check svg for `{theory_id}.{definition_name}`: {error}"
                        ),
                    )
                })?;

                let checked_path = definition_dir.join("check_partial.svg");
                fs::write(&checked_path, svg).map_err(|error| {
                    io::Error::new(
                        error.kind(),
                        format!("failed to write {}: {error}", checked_path.display()),
                    )
                })?;
                dump_untyped_stage_hex(term, "check_partial", &definition_dir)?;
            }

            dump_typed_stage_svg(
                &report
                    .closure_conversion
                    .as_ref()
                    .map(|conversion| conversion.closure_forgotten_definitions.clone()),
                "forget_closures",
                theory_id,
                definition_name,
                syntax_theory,
                &definition_dir,
                region_to_hexpr_operation,
            )?;
            if let Some(conversion) = &report.closure_conversion {
                dump_closure_conversion_trace(
                    conversion,
                    theory_id,
                    definition_name,
                    syntax_theory,
                    &definition_dir,
                )?;
            }
            dump_typed_stage_svg(
                &report.boundary_sizes,
                "boundary_sizes",
                theory_id,
                definition_name,
                syntax_theory,
                &definition_dir,
                |op| op.operation.clone(),
            )?;
            dump_typed_stage_svg(
                &report.unpacked_products,
                "unpacked_products",
                theory_id,
                definition_name,
                syntax_theory,
                &definition_dir,
                |op| op.operation.clone(),
            )?;
        }
    }

    Ok(())
}

fn dump_closure_conversion_trace(
    conversion: &Conversion,
    theory_id: &TheoryId,
    definition_name: &Operation,
    syntax_theory: &Theory,
    definition_dir: &Path,
) -> io::Result<()> {
    let Some(term) = conversion
        .closure_forgotten_definitions
        .get(theory_id)
        .and_then(|definitions| definitions.get(definition_name))
    else {
        return Ok(());
    };
    let Some(regions) = conversion
        .regions
        .get(theory_id)
        .and_then(|definitions| definitions.get(definition_name))
    else {
        return Ok(());
    };
    if regions.is_empty() {
        return Ok(());
    }

    dump_typed_stage_svg(
        &Some(conversion.closure_forgotten_definitions.clone()),
        "closure_conversion_00_input",
        theory_id,
        definition_name,
        syntax_theory,
        definition_dir,
        region_to_hexpr_operation,
    )?;

    let overview = labelled_region_overview(term, regions, syntax_theory)?;
    let overview_path = definition_dir.join("closure_conversion_10_regions.svg");
    fs::write(&overview_path, overview).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!("failed to write {}: {error}", overview_path.display()),
        )
    })?;

    let generated_bodies = conversion
        .generated_functions
        .get(theory_id)
        .ok_or_else(|| invalid_data(format!("missing converted theory `{theory_id}`")))?;
    let mut index_document = format!(
        "# Closure conversion: `{theory_id}.{definition_name}`\n\n\
         1. [Closure-forgotten input](closure_conversion_00_input.svg) \
            ([hex](closure_conversion_00_input.hex))\n\
         2. [Colored region overview](closure_conversion_10_regions.svg)\n"
    );

    for (index, region) in regions.iter().enumerate() {
        let extracted = labelled_extracted_region(term, region, index, syntax_theory)?;
        let svg = to_svg_with(&extracted, &Options::default().display().lr())?;
        let svg = colorize_svg(
            svg,
            (0..extracted.hypergraph.nodes.len())
                .map(|node| (format!("n_{node}"), region_color(index))),
            (0..extracted.hypergraph.edges.len())
                .map(|edge| (format!("e_{edge}"), region_color(index))),
        )?;
        let region_stage = format!("closure_conversion_11_region_{index}");
        let path = definition_dir.join(format!("{region_stage}.svg"));
        fs::write(&path, svg).map_err(|error| {
            io::Error::new(
                error.kind(),
                format!("failed to write {}: {error}", path.display()),
            )
        })?;

        let closure_name = closure_operation(definition_name, region.closure);
        let closure_body = generated_bodies
            .get(&closure_name)
            .ok_or_else(|| invalid_data(format!("missing generated closure `{closure_name}`")))?;
        let closure_stage = format!("closure_conversion_20_closure_{index}");
        let closure_svg = render_typed_svg(closure_body, syntax_theory)?;
        let closure_path = definition_dir.join(format!("{closure_stage}.svg"));
        fs::write(&closure_path, closure_svg).map_err(|error| {
            io::Error::new(
                error.kind(),
                format!("failed to write {}: {error}", closure_path.display()),
            )
        })?;
        dump_typed_stage_hex(
            closure_body,
            &closure_stage,
            definition_dir,
            |operation| operation.clone(),
            syntax_theory,
        )?;

        index_document.push_str(&format!(
            "   - Region {index}: `{closure_name}` \
             ([region](closure_conversion_11_region_{index}.svg), \
              [body](closure_conversion_20_closure_{index}.svg), \
              [body hex](closure_conversion_20_closure_{index}.hex))\n"
        ));
    }

    dump_typed_stage_svg(
        &Some(conversion.rewritten_definitions.clone()),
        "closure_conversion_30_replacement",
        theory_id,
        definition_name,
        syntax_theory,
        definition_dir,
        |operation| operation.clone(),
    )?;
    index_document.push_str(
        "3. [Replacement with context projections](closure_conversion_30_replacement.svg) \
         ([hex](closure_conversion_30_replacement.hex))\n",
    );
    dump_typed_stage_svg(
        &Some(conversion.runtime_functions.clone()),
        "closure_conversion_40_output",
        theory_id,
        definition_name,
        syntax_theory,
        definition_dir,
        |operation| operation.clone(),
    )?;
    index_document.push_str(
        "4. [Final context-free output](closure_conversion_40_output.svg) \
         ([hex](closure_conversion_40_output.hex))\n",
    );
    let index_path = definition_dir.join("closure_conversion.md");
    fs::write(&index_path, index_document).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!("failed to write {}: {error}", index_path.display()),
        )
    })?;

    Ok(())
}

fn labelled_region_overview(
    term: &ClosureForgottenTerm,
    regions: &[ClosureRegion],
    syntax_theory: &Theory,
) -> io::Result<Vec<u8>> {
    let node_labels = term
        .hypergraph
        .nodes
        .iter()
        .enumerate()
        .map(|(node, ty)| {
            let mut roles = Vec::new();
            for (region_index, region) in regions.iter().enumerate() {
                let node = NodeId(node);
                if node == region.domain {
                    roles.push(format!("R{region_index}:domain"));
                }
                if node == region.codomain {
                    roles.push(format!("R{region_index}:codomain"));
                }
                if node == region.closure {
                    roles.push(format!("R{region_index}:closure"));
                }
                if region.environment.contains(&node) {
                    roles.push(format!("R{region_index}:env"));
                }
            }
            let roles = roles.join(", ");
            Ok(format!(
                "w{node}: {}\n{roles}",
                pretty_type(ty, syntax_theory)?
            ))
        })
        .collect::<io::Result<Vec<_>>>()?;

    let edge_labels = term
        .hypergraph
        .edges
        .iter()
        .enumerate()
        .map(|(edge, operation)| {
            let roles = regions
                .iter()
                .enumerate()
                .filter_map(|(region_index, region)| {
                    if region.marker.0 == edge {
                        Some(format!("R{region_index}:marker"))
                    } else if region.edges.iter().any(|body_edge| body_edge.0 == edge) {
                        Some(format!("R{region_index}:body"))
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join(", ");
            if roles.is_empty() {
                operation.to_string()
            } else {
                format!("{operation}\n{roles}")
            }
        })
        .collect::<Vec<_>>();

    let labelled = term
        .clone()
        .with_nodes(|_| node_labels)
        .and_then(|term| term.with_edges(|_| edge_labels))
        .ok_or_else(|| invalid_data("region overview label count mismatch".to_string()))?;
    let svg = to_svg_with(&labelled, &Options::default().display().lr())?;
    let node_colors = term
        .hypergraph
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(node, _)| {
            regions
                .iter()
                .position(|region| {
                    let node = NodeId(node);
                    node == region.domain
                        || node == region.codomain
                        || node == region.closure
                        || region.environment.contains(&node)
                })
                .map(|region| (format!("n_{node}"), region_color(region)))
        });
    let edge_colors = term
        .hypergraph
        .edges
        .iter()
        .enumerate()
        .filter_map(|(edge, _)| {
            regions
                .iter()
                .position(|region| {
                    region.marker.0 == edge
                        || region.edges.iter().any(|body_edge| body_edge.0 == edge)
                })
                .map(|region| (format!("e_{edge}"), region_color(region)))
        });
    colorize_svg(svg, node_colors, edge_colors)
}

fn labelled_extracted_region(
    term: &ClosureForgottenTerm,
    region: &ClosureRegion,
    region_index: usize,
    syntax_theory: &Theory,
) -> io::Result<OpenHypergraph<String, String>> {
    let mut extracted = OpenHypergraph::empty();
    let mut node_map = HashMap::new();

    for &node in &region.nodes {
        let ty = pretty_type(&term.hypergraph.nodes[node.0], syntax_theory)?;
        let mut roles = Vec::new();
        if node == region.domain {
            roles.push("domain");
        }
        if node == region.codomain {
            roles.push("codomain");
        }
        if region.environment.contains(&node) {
            roles.push("env");
        }
        let copied = extracted.new_node(format!(
            "R{region_index} w{}: {ty}\n{}",
            node.0,
            roles.join(", ")
        ));
        node_map.insert(node, copied);
    }

    for &edge in &region.edges {
        let boundary = &term.hypergraph.adjacency[edge.0];
        let sources = remap_region_nodes(&node_map, &boundary.sources)?;
        let targets = remap_region_nodes(&node_map, &boundary.targets)?;
        extracted.new_edge(
            format!("{}\nR{region_index}:body", term.hypergraph.edges[edge.0]),
            (sources, targets),
        );
    }

    extracted.sources = remap_region_nodes(&node_map, &region.environment)?;
    extracted.sources.push(node_map[&region.domain]);
    extracted.targets = vec![node_map[&region.codomain]];
    Ok(extracted)
}

fn remap_region_nodes(
    node_map: &HashMap<NodeId, NodeId>,
    nodes: &[NodeId],
) -> io::Result<Vec<NodeId>> {
    nodes
        .iter()
        .map(|node| {
            node_map
                .get(node)
                .copied()
                .ok_or_else(|| invalid_data(format!("region is missing node w{}", node.0)))
        })
        .collect()
}

const REGION_COLORS: [&str; 8] = [
    "#56B4E9", // sky blue
    "#E69F00", // orange
    "#009E73", // green
    "#CC79A7", // purple
    "#F0E442", // yellow
    "#D55E00", // vermilion
    "#0072B2", // blue
    "#B8E186", // light green
];

fn region_color(index: usize) -> &'static str {
    REGION_COLORS[index % REGION_COLORS.len()]
}

fn colorize_svg(
    svg: Vec<u8>,
    node_colors: impl IntoIterator<Item = (String, &'static str)>,
    edge_colors: impl IntoIterator<Item = (String, &'static str)>,
) -> io::Result<Vec<u8>> {
    let mut svg = String::from_utf8(svg)
        .map_err(|error| invalid_data(format!("Graphviz returned non-UTF-8 SVG: {error}")))?;
    for (element, color) in node_colors.into_iter().chain(edge_colors) {
        color_svg_group(&mut svg, &element, color);
    }
    Ok(svg.into_bytes())
}

fn color_svg_group(svg: &mut String, element: &str, color: &str) {
    let title = format!("<title>{element}</title>");
    let Some(title_start) = svg.find(&title) else {
        return;
    };
    let Some(group_start) = svg[..title_start].rfind("<g ") else {
        return;
    };
    let Some(relative_end) = svg[title_start..].find("</g>") else {
        return;
    };
    let group_end = title_start + relative_end + "</g>".len();
    let colored = svg[group_start..group_end]
        .replace("stroke=\"white\"", &format!("stroke=\"{color}\""))
        .replace("fill=\"white\"", &format!("fill=\"{color}\""));
    svg.replace_range(group_start..group_end, &colored);
}

fn dump_typed_stage_svg<A: Clone + fmt::Debug + fmt::Display + PartialEq>(
    theories: &Option<crate::report::TheoryTermMap<A>>,
    stage: &str,
    theory_id: &TheoryId,
    definition_name: &Operation,
    syntax_theory: &Theory,
    definition_dir: &Path,
    edge_to_operation: impl Fn(&A) -> Operation,
) -> io::Result<()> {
    let Some(transformed) = theories
        .as_ref()
        .and_then(|theories| theories.get(theory_id))
        .and_then(|defs| defs.get(definition_name))
    else {
        return Ok(());
    };

    let svg = render_typed_svg(transformed, syntax_theory).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!("failed to render {stage} svg for `{theory_id}.{definition_name}`: {error}"),
        )
    })?;
    let path = definition_dir.join(format!("{stage}.svg"));
    fs::write(&path, svg).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!("failed to write {}: {error}", path.display()),
        )
    })?;

    dump_typed_stage_hex(
        transformed,
        stage,
        definition_dir,
        edge_to_operation,
        syntax_theory,
    )
}

fn dump_untyped_stage_hex(
    term: &OpenHypergraph<(), Operation>,
    stage: &str,
    definition_dir: &Path,
) -> io::Result<()> {
    let term = term.clone().map_nodes(|_| Tree::Empty);
    write_stage_hex(&term, stage, definition_dir, "")
}

fn dump_typed_stage_hex<A: Clone>(
    term: &OpenHypergraph<metacat::tree::Tree<(), Operation>, A>,
    stage: &str,
    definition_dir: &Path,
    edge_to_operation: impl Fn(&A) -> Operation,
    syntax_theory: &Theory,
) -> io::Result<()> {
    let type_comments = type_comments(term, syntax_theory)?;
    let term = term.clone().map_edges(|edge| edge_to_operation(&edge));
    write_stage_hex(&term, stage, definition_dir, &type_comments)
}

fn write_stage_hex(
    term: &OpenHypergraph<metacat::tree::Tree<(), Operation>, Operation>,
    stage: &str,
    definition_dir: &Path,
    prefix: &str,
) -> io::Result<()> {
    let hexpr = crate::hexpr::term_to_hexpr(term);
    let path = definition_dir.join(format!("{stage}.hex"));
    fs::write(&path, format!("{prefix}{hexpr}\n")).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!("failed to write {}: {error}", path.display()),
        )
    })
}

fn region_to_hexpr_operation(region: &ClosureForgotten<Operation>) -> Operation {
    match region {
        ClosureForgotten::Operation(operation) => operation.clone(),
        ClosureForgotten::ClosureMarker => op("!closure"),
    }
}

fn type_comments(
    term: &OpenHypergraph<metacat::tree::Tree<(), Operation>, impl Clone>,
    syntax_theory: &Theory,
) -> io::Result<String> {
    term.hypergraph
        .nodes
        .iter()
        .enumerate()
        .map(|(index, ty)| {
            Ok(format!(
                "# w{index} : {}\n",
                pretty_type(ty, syntax_theory)?
            ))
        })
        .collect()
}

fn op(name: &str) -> Operation {
    name.parse().expect("generated operation should parse")
}

fn render_check_result_svg(
    term: &OpenHypergraph<(), Operation>,
    node_types: &[metacat::tree::Tree<(), Operation>],
    syntax_theory: &Theory,
) -> io::Result<Vec<u8>> {
    let labels: Vec<String> = node_types
        .iter()
        .map(|ty| pretty_type(ty, syntax_theory))
        .collect::<Result<_, _>>()?;
    render_labelled_svg(term, labels)
}

fn render_partial_check_result_svg(
    term: &OpenHypergraph<(), Operation>,
    node_types: &[Option<metacat::tree::Tree<(), Operation>>],
    syntax_theory: &Theory,
) -> io::Result<Vec<u8>> {
    let labels: Vec<String> = node_types
        .iter()
        .map(|ty| match ty {
            Some(ty) => pretty_type(ty, syntax_theory),
            None => Ok("?".to_string()),
        })
        .collect::<Result<_, _>>()?;
    render_labelled_svg(term, labels)
}

fn render_labelled_svg(
    term: &OpenHypergraph<(), Operation>,
    labels: Vec<String>,
) -> io::Result<Vec<u8>> {
    let mut term = term.clone();
    term.quotient().map_err(|error| {
        invalid_data(format!(
            "failed to quotient term for svg rendering: {error:?}"
        ))
    })?;

    let labelled = term
        .with_nodes(|_| labels)
        .ok_or_else(|| invalid_data("labels length mismatch".to_string()))?;
    to_svg_with(&labelled, &Options::default().display().lr())
}

fn render_untyped_svg(term: &OpenHypergraph<(), Operation>) -> io::Result<Vec<u8>> {
    let mut term = term.clone();
    term.quotient().map_err(|error| {
        invalid_data(format!(
            "failed to quotient term for svg rendering: {error:?}"
        ))
    })?;
    let labels = vec![String::new(); term.hypergraph.nodes.len()];
    let labelled = term
        .with_nodes(|_| labels)
        .ok_or_else(|| invalid_data("labels length mismatch".to_string()))?;
    to_svg_with(&labelled, &Options::default().display().lr())
}

fn render_typed_svg<A: Clone + fmt::Debug + fmt::Display + PartialEq>(
    term: &OpenHypergraph<metacat::tree::Tree<(), Operation>, A>,
    syntax_theory: &Theory,
) -> io::Result<Vec<u8>> {
    let mut term = term.clone();
    term.quotient().map_err(|error| {
        invalid_data(format!(
            "failed to quotient term for svg rendering: {error:?}"
        ))
    })?;

    let labels: Vec<String> = term
        .hypergraph
        .nodes
        .iter()
        .map(|ty| {
            ty.try_pretty(Some(&|op: &Operation| {
                syntax_theory.coarity_of(op).ok_or_else(|| {
                    invalid_data(format!("coarity lookup failed for operation `{op}`"))
                })
            }))
        })
        .collect::<Result<_, _>>()?;

    let labelled = term
        .with_nodes(|_| labels)
        .ok_or_else(|| invalid_data("labels length mismatch".to_string()))?;
    to_svg_with(&labelled, &Options::default().display().lr())
}

fn pretty_type(
    ty: &metacat::tree::Tree<(), Operation>,
    syntax_theory: &Theory,
) -> io::Result<String> {
    ty.try_pretty(Some(&|op: &Operation| {
        syntax_theory
            .coarity_of(op)
            .ok_or_else(|| invalid_data(format!("coarity lookup failed for operation `{op}`")))
    }))
}

fn qualified_definition_dir(theory_id: &TheoryId, definition_name: &Operation) -> String {
    format!("{theory_id}.{definition_name}")
}

fn invalid_data(message: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}
