use std::collections::HashSet;

use hexpr::Operation;
use metacat::theory::{
    RawTheorySet, Theory, TheoryId, TheorySet,
    ast::{ParseRawError, RawTheory, RawTheoryArrow},
};
use open_hypergraphs::{
    category::Arrow,
    lax::{NodeId, OpenHypergraph},
};
use thiserror::Error;

use crate::compile::{
    config::{CompileConfig, TheoryExtension},
    lift::{LiftError, lift_with_tensor},
};

#[derive(Debug, Error)]
pub enum CompileLoadError {
    #[error(transparent)]
    RawParse(#[from] ParseRawError),
    #[error(transparent)]
    Load(#[from] metacat::theory::LoadError),
    #[error(transparent)]
    Lift(#[from] LiftError),
    #[error("unknown theory `{0}`")]
    UnknownTheory(String),
    #[error("invalid operation name `{0}`")]
    InvalidOperation(String),
}

/// Load a metacat theory source after adding Catena compile-extension arrows.
///
/// The returned [`TheorySet`] contains the original theories plus generated
/// lifted arrow declarations described by `config`. This allows definitions to
/// refer to extension arrows such as `data.copy` without declaring temporary
/// stubs in the source file.
pub fn load_extended_theory_set_from_text(
    source: &str,
    config: &CompileConfig,
) -> Result<TheorySet, CompileLoadError> {
    // The source may contain references to lifted arrows, such as
    // `data.copy`, before Catena has generated the corresponding declarations
    // in the target theory. A normal `TheorySet::from_text(source)` would fail
    // while interpreting definitions because those operations do not exist yet.
    let raw = RawTheorySet::from_text(source)?;

    // Build a definition-free scaffold so metacat resolves only theory names,
    // arrow names, and type maps. Catena can then inspect the base theories and
    // synthesize the missing lifted arrow declarations before loading the real
    // definitions below.
    //
    // TODO: consider whether metacat loading should expose a relaxed mode that
    // does not interpret/check definitions yet, so Catena can extend the raw
    // theory set without re-rendering this scaffold as text.
    let scaffold = TheorySet::from_text(&render_raw_theory_set(&raw, true))?;
    let lifted_declarations = lifted_declarations(&raw, &scaffold, config)?;

    if lifted_declarations.trim().is_empty() {
        Ok(TheorySet::from_text(source)?)
    } else {
        Ok(TheorySet::from_texts([
            source,
            lifted_declarations.as_str(),
        ])?)
    }
}

fn lifted_declarations(
    raw: &RawTheorySet,
    scaffold: &TheorySet,
    config: &CompileConfig,
) -> Result<String, CompileLoadError> {
    let syntax = theory(scaffold, config.syntax)?;
    let excluded_prefixes = config.lifted_prefixes();
    let mut theories = Vec::new();

    for extension in &config.extensions {
        let source = theory(scaffold, extension.source)?;
        let target = theory(scaffold, extension.target)?;
        let extended = lift_with_tensor(
            source,
            target,
            syntax,
            extension.prefix,
            extension.tensor,
            extension.unit,
            &excluded_prefixes,
        )?;
        let declarations = lifted_extension_declarations(raw, &extended, extension)?;
        if !declarations.is_empty() {
            theories.push(format!(
                "(theory {} {} {{\n{}\n}})",
                extension.target,
                config.syntax,
                declarations.join("\n")
            ));
        }
    }

    Ok(theories.join("\n\n"))
}

fn lifted_extension_declarations(
    raw: &RawTheorySet,
    extended: &Theory,
    extension: &TheoryExtension,
) -> Result<Vec<String>, CompileLoadError> {
    let existing = raw
        .theories
        .get(&operation(extension.target)?)
        .map(|theory| &theory.arrows);
    let Theory::Theory { arrows, .. } = extended else {
        return Ok(Vec::new());
    };

    let mut declarations = arrows
        .iter()
        .filter(|(name, _)| name.as_str().starts_with(&format!("{}.", extension.prefix)))
        .filter(|(name, _)| {
            existing
                .map(|arrows| !arrows.contains_key(*name))
                .unwrap_or(true)
        })
        .map(|(name, arrow)| {
            format!(
                "  (arr {} : {} -> {})",
                name,
                render_parseable_object_map(&arrow.type_maps.0),
                render_parseable_object_map(&arrow.type_maps.1)
            )
        })
        .collect::<Vec<_>>();
    declarations.sort();
    Ok(declarations)
}

fn render_raw_theory_set(raw: &RawTheorySet, strip_definitions: bool) -> String {
    raw.theories
        .values()
        .map(|theory| render_raw_theory(theory, strip_definitions))
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn render_raw_theory(theory: &RawTheory, strip_definitions: bool) -> String {
    let declarations = theory
        .arrows
        .values()
        .map(|arrow| render_raw_arrow(arrow, strip_definitions))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "(theory {} {} {{\n{}\n}})",
        theory.name, theory.syntax_category, declarations
    )
}

fn render_raw_arrow(arrow: &RawTheoryArrow, strip_definitions: bool) -> String {
    match (&arrow.definition, strip_definitions) {
        (Some(definition), false) => format!(
            "  (def {} : {} -> {} = {})",
            arrow.name, arrow.type_maps.0, arrow.type_maps.1, definition
        ),
        _ => format!(
            "  (arr {} : {} -> {})",
            arrow.name, arrow.type_maps.0, arrow.type_maps.1
        ),
    }
}

fn theory<'a>(set: &'a TheorySet, name: &str) -> Result<&'a Theory, CompileLoadError> {
    set.theories
        .get(&TheoryId(operation(name)?))
        .ok_or_else(|| CompileLoadError::UnknownTheory(name.to_string()))
}

fn operation(name: &str) -> Result<Operation, CompileLoadError> {
    name.parse()
        .map_err(|_| CompileLoadError::InvalidOperation(name.to_string()))
}

fn render_parseable_object_map(map: &OpenHypergraph<(), Operation>) -> String {
    let mut map = map.clone();
    let _ = map.quotient();
    let body = match map.target().len() {
        0 => "[]".to_string(),
        1 => render_parseable_node(&map, map.targets[0], &mut HashSet::new()),
        _ => {
            let targets = map
                .targets
                .iter()
                .map(|node| render_parseable_node(&map, *node, &mut HashSet::new()))
                .collect::<Vec<_>>();
            format!("{{{}}}", targets.join(" "))
        }
    };
    let all_sources = (0..map.source().len())
        .map(|index| format!("x{index}"))
        .collect::<Vec<_>>();
    let used_sources = used_source_vars(&map);
    if all_sources == used_sources {
        body
    } else {
        format!("({} {body})", render_spider(&all_sources, &used_sources))
    }
}

fn render_parseable_node(
    map: &OpenHypergraph<(), Operation>,
    node: NodeId,
    seen: &mut HashSet<NodeId>,
) -> String {
    if let Some(index) = map.sources.iter().position(|source| *source == node) {
        return format!("[x{index}]");
    }

    if !seen.insert(node) {
        return format!("[n{}]", node.0);
    }

    let rendered = producer_edge(map, node)
        .map(|edge_index| render_parseable_edge(map, edge_index, seen))
        .unwrap_or_else(|| format!("[n{}]", node.0));
    seen.remove(&node);
    rendered
}

fn render_parseable_edge(
    map: &OpenHypergraph<(), Operation>,
    edge_index: usize,
    seen: &mut HashSet<NodeId>,
) -> String {
    let op = &map.hypergraph.edges[edge_index];
    let adjacency = &map.hypergraph.adjacency[edge_index];
    if adjacency.sources.is_empty() {
        return op.to_string();
    }

    let direct_source_vars = adjacency
        .sources
        .iter()
        .map(|node| source_var(map, *node))
        .collect::<Option<Vec<_>>>();
    if let Some(input_vars) = direct_source_vars {
        let mut source_vars = Vec::new();
        for var in &input_vars {
            if !source_vars.contains(var) {
                source_vars.push(var.clone());
            }
        }
        return format!("({} {op})", render_spider(&source_vars, &input_vars));
    }

    let inputs = adjacency
        .sources
        .iter()
        .map(|node| render_parseable_node(map, *node, seen))
        .collect::<Vec<_>>();
    format!("({{{}}} {op})", inputs.join(" "))
}

fn producer_edge(map: &OpenHypergraph<(), Operation>, node: NodeId) -> Option<usize> {
    map.hypergraph
        .adjacency
        .iter()
        .position(|edge| edge.targets.contains(&node))
}

fn used_source_vars(map: &OpenHypergraph<(), Operation>) -> Vec<String> {
    let mut used = HashSet::new();
    for target in &map.targets {
        collect_used_source_vars(map, *target, &mut used, &mut HashSet::new());
    }
    (0..map.source().len())
        .map(|index| format!("x{index}"))
        .filter(|var| used.contains(var))
        .collect()
}

fn collect_used_source_vars(
    map: &OpenHypergraph<(), Operation>,
    node: NodeId,
    used: &mut HashSet<String>,
    seen: &mut HashSet<NodeId>,
) {
    if let Some(var) = source_var(map, node) {
        used.insert(var);
        return;
    }
    if !seen.insert(node) {
        return;
    }
    if let Some(edge_index) = producer_edge(map, node) {
        for source in &map.hypergraph.adjacency[edge_index].sources {
            collect_used_source_vars(map, *source, used, seen);
        }
    }
}

fn source_var(map: &OpenHypergraph<(), Operation>, node: NodeId) -> Option<String> {
    map.sources
        .iter()
        .position(|source| *source == node)
        .map(|index| format!("x{index}"))
}

fn render_spider(sources: &[String], targets: &[String]) -> String {
    if sources == targets {
        format!("[{}]", sources.join(" "))
    } else if targets.is_empty() {
        format!("[{} .]", sources.join(" "))
    } else {
        format!("[{} . {}]", sources.join(" "), targets.join(" "))
    }
}
