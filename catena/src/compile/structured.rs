use thiserror::Error;

use crate::{
    compile::{
        cfg,
        program::{Program, VariableId},
    },
    structured::{
        ir::{EntryPoint, Stmt, StructuredProgram},
        ramsey,
        ramsey::RamseyError,
    },
};

#[derive(Debug, Error)]
pub enum StructuredCompileError {
    #[error("failed to structure control graph: {0}")]
    Structure(#[from] RamseyError),
}

pub fn compile_structured_program(
    program: &Program,
) -> Result<StructuredProgram, StructuredCompileError> {
    let entry = program.entry_definition();
    let context = entry.context.clone();
    let body = ramsey::structure(entry.body.clone(), move |id| {
        context
            .variable(VariableId(id))
            .map(|variable| variable.name.clone())
            .unwrap_or_else(|| cfg::variable_name(id))
    })?;
    Ok(structured_program(&entry.name, body))
}

fn structured_program(entry: &str, body: Vec<Stmt>) -> StructuredProgram {
    StructuredProgram {
        name: sanitize_ident(entry),
        entry: EntryPoint {
            name: sanitize_ident(entry),
            params: Vec::new(),
        },
        body,
    }
}

fn sanitize_ident(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}
