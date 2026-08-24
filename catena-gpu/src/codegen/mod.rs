//! Scalar-only GPU code generation.

pub mod gpu;
pub mod lower_types;

use std::collections::BTreeMap;

use hexpr::Operation;
use metacat::{
    ssa::{SSAError, ssa},
    theory::TheoryId,
};
use open_hypergraphs::lax::NodeId;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{check::AnnotatedTerm, report::TheoryTermMap};
use lower_types::{CType, LowerTypeError, LoweredType, lower_type};

pub type GpuModuleMap = BTreeMap<Operation, GpuModule>;
type CodegenTerm = AnnotatedTerm<Operation>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuDialect {
    Hip,
    Cuda,
}

impl GpuDialect {
    pub(crate) fn runtime_header(self) -> &'static str {
        match self {
            Self::Hip => "hip/hip_runtime.h",
            Self::Cuda => "cuda_runtime.h",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuModule {
    pub name: String,
    pub source_name: Option<Operation>,
    pub entry: GpuFunction,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuFunction {
    pub name: String,
    pub sources: Vec<GpuVar>,
    pub targets: Vec<GpuVar>,
    pub assignments: Vec<GpuAssign>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuAssign {
    pub op: Operation,
    pub call_symbol: Option<String>,
    pub inputs: Vec<GpuVar>,
    pub outputs: Vec<GpuVar>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuVar {
    pub node: NodeId,
    pub name: String,
    pub lowered: LoweredType,
}

#[derive(Debug, Error)]
pub enum CodegenError {
    #[error(transparent)]
    Ssa(#[from] SSAError),
    #[error("failed to quotient term before codegen: {0:?}")]
    Quotient(open_hypergraphs::strict::vec::FiniteFunction),
    #[error(transparent)]
    LowerType(#[from] LowerTypeError),
}

pub fn codegen(terms: &TheoryTermMap) -> Result<GpuModuleMap, CodegenError> {
    let program = TheoryId("program".parse().unwrap());
    let Some(definitions) = terms.get(&program) else {
        return Ok(BTreeMap::new());
    };
    definitions
        .iter()
        .map(|(name, term)| {
            let symbol = sanitize_ident(&format!("program.{name}"));
            let module = codegen_definition(term, definitions, name.clone(), symbol)?;
            Ok((name.clone(), module))
        })
        .collect()
}

fn codegen_definition(
    term: &CodegenTerm,
    definitions: &BTreeMap<Operation, CodegenTerm>,
    source_name: Operation,
    symbol: String,
) -> Result<GpuModule, CodegenError> {
    let mut term = term.clone();
    term.quotient().map_err(CodegenError::Quotient)?;
    let sources = term
        .sources
        .iter()
        .map(|node| var(*node, &term))
        .filter_map(runtime_var)
        .collect::<Result<Vec<_>, _>>()?;
    let targets = term
        .targets
        .iter()
        .map(|node| var(*node, &term))
        .filter_map(runtime_var)
        .collect::<Result<Vec<_>, _>>()?;
    let mut assignments = Vec::new();
    for assignment in ssa(term.clone().to_strict())? {
        let op = assignment.op;
        if op.as_str().starts_with("name.") {
            continue;
        }
        let inputs = assignment
            .sources
            .iter()
            .map(|(node, _)| var(*node, &term))
            .filter_map(runtime_var)
            .collect::<Result<Vec<_>, _>>()?;
        let outputs = assignment
            .targets
            .iter()
            .map(|(node, _)| var(*node, &term))
            .filter_map(runtime_var)
            .collect::<Result<Vec<_>, _>>()?;
        if inputs.is_empty() && outputs.is_empty() {
            continue;
        }
        let call_symbol = definitions
            .contains_key(&op)
            .then(|| sanitize_ident(&format!("program.{op}")));
        assignments.push(GpuAssign {
            op,
            call_symbol,
            inputs,
            outputs,
        });
    }
    Ok(GpuModule {
        name: symbol.clone(),
        source_name: Some(source_name),
        entry: GpuFunction {
            name: symbol,
            sources,
            targets,
            assignments,
        },
    })
}

fn var(node: NodeId, term: &CodegenTerm) -> Result<GpuVar, LowerTypeError> {
    Ok(GpuVar {
        node,
        name: format!("x{}", node.0),
        lowered: lower_type(&term.hypergraph.nodes[node.0])?,
    })
}

fn runtime_var(value: Result<GpuVar, LowerTypeError>) -> Option<Result<GpuVar, LowerTypeError>> {
    match value {
        Ok(var) if matches!(var.lowered, LoweredType::Runtime(_)) => Some(Ok(var)),
        Ok(_) => None,
        Err(error) => Some(Err(error)),
    }
}

pub fn runtime_type(var: &GpuVar) -> Option<&CType> {
    match &var.lowered {
        LoweredType::Runtime(ty) => Some(ty),
        LoweredType::Erased => None,
    }
}

fn sanitize_ident(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}
