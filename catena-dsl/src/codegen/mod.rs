//! CUDA codegen via [`catena::StructuredProgram`]
//!
//! For each annotated, transformed definition, we should have a number of definitions which
//! only contain the syntax defined in stdlib/base.
//!
//! We can now *compile* these into CUDA code via the [`catena::StructuredProgram`] interface.
//! Here's the spec:
//!
//! - Topologically sort all operations using metacat's SSA module (as in catena backend c codegen)
//! - For node i, use variable name "x{i}"
//! - All functions synthesized return 'void'; for inputs A, B and outputs C D, pass outputs as
//!   ptrs to be overwritten - see catena codegen module for example.
//! - Keep the mapping of catena-dsl types to CUDA types as its own file: for now, just pick one
//!   for bool, and synth a function type for (A -> B) fns.

mod types;

use std::collections::BTreeMap;

use catena::structured::{EntryPoint, Param, Primitive, Stmt, StructuredProgram};
use metacat::ssa::{SSAError, ssa};
use open_hypergraphs::lax::NodeId;
use thiserror::Error;

use crate::report::{AnnotatedTerm, StructuredProgramMap, TheoryTermMap};

/// Codegen for all functions, producing a map from definition names to [`StructuredProgram`]
pub fn codegen(terms: &TheoryTermMap) -> Result<StructuredProgramMap, CodegenError> {
    let mut programs = BTreeMap::new();

    for (theory_id, definitions) in terms {
        let mut theory_programs = BTreeMap::new();
        for (definition_name, term) in definitions {
            theory_programs.insert(
                definition_name.clone(),
                codegen_definition(&format!("{theory_id}.{definition_name}"), term)?,
            );
        }
        if !theory_programs.is_empty() {
            programs.insert(theory_id.clone(), theory_programs);
        }
    }

    Ok(programs)
}

#[derive(Debug, Error)]
pub enum CodegenError {
    #[error(transparent)]
    Ssa(#[from] SSAError),
    #[error("unsupported CUDA type for node {node}: {ty}")]
    UnsupportedType { node: usize, ty: String },
}

fn codegen_definition(
    qualified_name: &str,
    term: &AnnotatedTerm,
) -> Result<StructuredProgram, CodegenError> {
    let mut params = Vec::new();
    let mut body = Vec::new();

    for source in &term.sources {
        let node = source.0;
        params.push(Param {
            ty: types::param_type(&term.hypergraph.nodes[node], false).ok_or_else(|| {
                CodegenError::UnsupportedType {
                    node,
                    ty: format!("{:?}", term.hypergraph.nodes[node]),
                }
            })?,
            name: node_var(*source),
        });
    }

    for target in &term.targets {
        let node = target.0;
        params.push(Param {
            ty: types::param_type(&term.hypergraph.nodes[node], true).ok_or_else(|| {
                CodegenError::UnsupportedType {
                    node,
                    ty: format!("{:?}", term.hypergraph.nodes[node]),
                }
            })?,
            name: output_param(*target),
        });
    }

    for assignment in ssa(term.clone().to_strict())? {
        body.push(Stmt::Primitive(Primitive {
            name: assignment.op.to_string(),
            inputs: assignment
                .sources
                .iter()
                .map(|(node, _)| node_var(*node))
                .collect(),
            outputs: assignment
                .targets
                .iter()
                .map(|(node, _)| node_var(*node))
                .collect(),
            code: String::new(),
        }));
    }

    for target in &term.targets {
        body.push(Stmt::Assign {
            lhs: format!("*{}", output_param(*target)),
            rhs: node_var(*target),
        });
    }
    body.push(Stmt::Return);

    Ok(StructuredProgram {
        name: sanitize_ident(qualified_name),
        entry: EntryPoint {
            name: sanitize_ident(qualified_name),
            params,
        },
        body,
    })
}

fn node_var(node: NodeId) -> String {
    format!("x{}", node.0)
}

fn output_param(node: NodeId) -> String {
    format!("out_x{}", node.0)
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
