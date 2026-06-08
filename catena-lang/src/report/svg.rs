use std::{fs, io, path::Path};

use hexpr::Operation;
use metacat::theory::{Theory, TheoryId};
use open_hypergraphs::lax::OpenHypergraph;
use open_hypergraphs_dot::{Options, svg::to_svg_with};

use crate::report::CompileReport;

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
            }

            if let Some(transformed) = report
                .forgotten_closures
                .as_ref()
                .and_then(|theories| theories.get(theory_id))
                .and_then(|defs| defs.get(definition_name))
            {
                let forget_closures_svg = render_typed_svg(transformed, syntax_theory).map_err(|error| {
                    io::Error::new(
                        error.kind(),
                        format!(
                            "failed to render forget_closures svg for `{theory_id}.{definition_name}`: {error}"
                        ),
                    )
                })?;
                let forget_closures_path = definition_dir.join("forget_closures.svg");
                fs::write(&forget_closures_path, forget_closures_svg).map_err(|error| {
                    io::Error::new(
                        error.kind(),
                        format!(
                            "failed to write {}: {error}",
                            forget_closures_path.display()
                        ),
                    )
                })?;
            }
        }
    }

    Ok(())
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

fn render_typed_svg(
    term: &OpenHypergraph<metacat::tree::Tree<(), Operation>, Operation>,
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
