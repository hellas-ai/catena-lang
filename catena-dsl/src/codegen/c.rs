use std::collections::{BTreeMap, HashSet};

use catena::structured::{Primitive, StructuredProgram, ir::Stmt};
use hexpr::Operation;
use open_hypergraphs::lax::NodeId;
use thiserror::Error;

use crate::{codegen::types, report::AnnotatedTerm};

#[derive(Debug, Error)]
pub enum CRenderError {
    #[error("missing node type for `{0}`")]
    MissingNodeType(String),
    #[error("unsupported structured statement in C renderer")]
    UnsupportedStmt,
}

pub fn render_program(program: &StructuredProgram, term: &AnnotatedTerm) -> Result<String, CRenderError> {
    let mut term = term.clone();
    term.quotient().ok();
    let node_types = node_type_map(&term);
    let source_nodes: HashSet<_> = term.sources.iter().map(|n| n.0).collect();
    let mut declared = source_nodes
        .iter()
        .map(|node| node_var(NodeId(*node)))
        .collect::<HashSet<_>>();

    let mut out = String::new();
    out.push_str(runtime_prelude());
    out.push('\n');
    out.push_str(&format!("void {}(", program.entry.name));
    out.push_str(
        &program
            .entry
            .params
            .iter()
            .enumerate()
            .map(|(index, param)| {
                if index < term.sources.len() {
                    types::c_param_decl(&term.hypergraph.nodes[term.sources[index].0], &param.name, false)
                } else {
                    let target_index = index - term.sources.len();
                    types::c_param_decl(
                        &term.hypergraph.nodes[term.targets[target_index].0],
                        &param.name,
                        true,
                    )
                }
                .ok_or_else(|| CRenderError::MissingNodeType(param.name.clone()))
            })
            .collect::<Result<Vec<_>, _>>()?
            .join(", "),
    );
    out.push_str(") {\n");

    for stmt in &program.body {
        render_stmt(&mut out, stmt, &node_types, &mut declared)?;
    }

    out.push_str("}\n");
    Ok(out)
}

fn runtime_prelude() -> &'static str {
    r#"#include <stdint.h>

static inline uint8_t bool_not(uint8_t arg0) {
    return !arg0;
}

static inline uint8_t bool_or(uint8_t arg0, uint8_t arg1) {
    return arg0 || arg1;
}
"#
}

fn render_stmt(
    out: &mut String,
    stmt: &Stmt,
    node_types: &BTreeMap<String, metacat::tree::Tree<(), Operation>>,
    declared: &mut HashSet<String>,
) -> Result<(), CRenderError> {
    match stmt {
        Stmt::Primitive(primitive) => render_primitive(out, primitive, node_types, declared),
        Stmt::Assign { lhs, rhs } => {
            out.push_str(&format!("    {lhs} = {rhs};\n"));
            Ok(())
        }
        Stmt::Return => {
            out.push_str("    return;\n");
            Ok(())
        }
        _ => Err(CRenderError::UnsupportedStmt),
    }
}

fn render_primitive(
    out: &mut String,
    primitive: &Primitive,
    node_types: &BTreeMap<String, metacat::tree::Tree<(), Operation>>,
    declared: &mut HashSet<String>,
) -> Result<(), CRenderError> {
    for output in &primitive.outputs {
        if declared.insert(output.clone()) {
            let ty = node_types
                .get(output)
                .ok_or_else(|| CRenderError::MissingNodeType(output.clone()))?;
            let decl = types::c_local_decl(ty, output)
                .ok_or_else(|| CRenderError::MissingNodeType(output.clone()))?;
            out.push_str(&format!("    {};\n", decl));
        }
    }

    match primitive.name.as_str() {
        "bool.not" => {
            let [input] = primitive.inputs.as_slice() else { return Err(CRenderError::UnsupportedStmt) };
            let [output] = primitive.outputs.as_slice() else { return Err(CRenderError::UnsupportedStmt) };
            out.push_str(&format!("    {output} = !{input};\n"));
        }
        "bool.or" => {
            let [lhs, rhs] = primitive.inputs.as_slice() else { return Err(CRenderError::UnsupportedStmt) };
            let [output] = primitive.outputs.as_slice() else { return Err(CRenderError::UnsupportedStmt) };
            out.push_str(&format!("    {output} = {lhs} || {rhs};\n"));
        }
        "eval" => {
            let Some((func, args)) = primitive.inputs.split_last() else {
                return Err(CRenderError::UnsupportedStmt);
            };
            let [output] = primitive.outputs.as_slice() else { return Err(CRenderError::UnsupportedStmt) };
            out.push_str(&format!("    {output} = {func}({});\n", args.join(", ")));
        }
        _ if primitive.name.starts_with("name.") => {
            let [output] = primitive.outputs.as_slice() else { return Err(CRenderError::UnsupportedStmt) };
            let target = sanitize_ident(primitive.name.trim_start_matches("name."));
            out.push_str(&format!("    {output} = {target};\n"));
        }
        _ => {
            out.push_str(&format!("    /* TODO: lower `{}` */\n", primitive.name));
        }
    }

    Ok(())
}

fn node_type_map(term: &AnnotatedTerm) -> BTreeMap<String, metacat::tree::Tree<(), Operation>> {
    term.hypergraph
        .nodes
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, ty)| (node_var(NodeId(index)), ty))
        .collect()
}

fn node_var(node: NodeId) -> String {
    format!("x{}", node.0)
}

fn sanitize_ident(name: &str) -> String {
    let mut ident = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect::<String>();
    if ident.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        ident.insert(0, '_');
    }
    ident
}
