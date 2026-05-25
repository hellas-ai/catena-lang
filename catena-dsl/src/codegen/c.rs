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
    #[error("type `{0:?}` is unsupported in C")]
    UnsupportedType(metacat::tree::Tree<(), Operation>),
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
                .ok_or_else(|| {
                    let ty = if index < term.sources.len() {
                        &term.hypergraph.nodes[term.sources[index].0]
                    } else {
                        let target_index = index - term.sources.len();
                        &term.hypergraph.nodes[term.targets[target_index].0]
                    };
                    CRenderError::UnsupportedType(ty.clone())
                })
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

typedef uint8_t catena_unit_t;

static inline void bool_not(uint8_t arg0, uint8_t *out1) {
    *out1 = !arg0;
}

static inline void bool_or(uint8_t arg0, uint8_t arg1, uint8_t *out2) {
    *out2 = arg0 || arg1;
}

static inline void bool_and(uint8_t arg0, uint8_t arg1, uint8_t *out2) {
    *out2 = arg0 && arg1;
}

static inline void bool_id(uint8_t arg0, uint8_t *out1) {
    *out1 = arg0;
}

static inline void bool_copy(uint8_t arg0, uint8_t *out1, uint8_t *out2) {
    *out1 = arg0;
    *out2 = arg0;
}

static inline void bool_li(uint8_t arg0, uint8_t *out1) {
    *out1 = arg0;
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
                .ok_or_else(|| CRenderError::UnsupportedType(ty.clone()))?;
            out.push_str(&format!("    {};\n", decl));
        }
    }

    match primitive.name.as_str() {
        "bool.t" => {
            let [output] = primitive.outputs.as_slice() else { return Err(CRenderError::UnsupportedStmt) };
            out.push_str(&format!("    {output} = 1;\n"));
        }
        "bool.f" => {
            let [output] = primitive.outputs.as_slice() else { return Err(CRenderError::UnsupportedStmt) };
            out.push_str(&format!("    {output} = 0;\n"));
        }
        "bool.not" => {
            let [input] = primitive.inputs.as_slice() else { return Err(CRenderError::UnsupportedStmt) };
            let [output] = primitive.outputs.as_slice() else { return Err(CRenderError::UnsupportedStmt) };
            out.push_str(&format!("    {output} = !{input};\n"));
        }
        "bool.and" => {
            let [lhs, rhs] = primitive.inputs.as_slice() else { return Err(CRenderError::UnsupportedStmt) };
            let [output] = primitive.outputs.as_slice() else { return Err(CRenderError::UnsupportedStmt) };
            out.push_str(&format!("    {output} = {lhs} && {rhs};\n"));
        }
        "bool.or" => {
            let [lhs, rhs] = primitive.inputs.as_slice() else { return Err(CRenderError::UnsupportedStmt) };
            let [output] = primitive.outputs.as_slice() else { return Err(CRenderError::UnsupportedStmt) };
            out.push_str(&format!("    {output} = {lhs} || {rhs};\n"));
        }
        "bool.ifc" => {
            let [env_true, fn_true, env_false, fn_false, flag, arg] = primitive.inputs.as_slice() else {
                return Err(CRenderError::UnsupportedStmt);
            };
            let [output] = primitive.outputs.as_slice() else { return Err(CRenderError::UnsupportedStmt) };
            out.push_str(&format!(
                "    if ({flag}) {{ {fn_true}({env_true}, {arg}, &{output}); }} else {{ {fn_false}({env_false}, {arg}, &{output}); }}\n"
            ));
        }
        "unit.intro" => {
            // The singleton object is erased from the wire-level representation.
        }
        "eval" => {
            let Some((func, args)) = primitive.inputs.split_last() else {
                return Err(CRenderError::UnsupportedStmt);
            };
            let mut call_args = args.to_vec();
            let mut output_ptrs = primitive
                .outputs
                .iter()
                .map(|output| format!("&{output}"))
                .collect::<Vec<_>>();
            call_args.append(&mut output_ptrs);
            out.push_str(&format!("    {func}({});\n", call_args.join(", ")));
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
