use std::{fmt, fs, io, path::Path};

use hexpr::Operation;
use metacat::theory::{Theory, TheoryId};
use metacat::tree::Tree;
use open_hypergraphs::lax::OpenHypergraph;
use open_hypergraphs_dot::{Options, svg::to_svg_with};

use crate::{pass::forget_closures::Region, report::CompileReport};

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
                &report.forgotten_closures,
                "forget_closures",
                theory_id,
                definition_name,
                syntax_theory,
                &definition_dir,
                region_to_hexpr_operation,
            )?;
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

fn region_to_hexpr_operation(region: &Region<Operation>) -> Operation {
    match region {
        Region::Operation(operation) => operation.clone(),
        Region::Closure => op("!closure"),
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
