//! Scalar-only GPU code generation.

pub mod gpu;
pub mod lower_types;
mod ops;

use std::collections::{BTreeMap, HashMap};

use hexpr::Operation;
use metacat::{
    ssa::{SSAError, ssa},
    theory::TheoryId,
    tree::Tree,
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
    pub inputs: Vec<GpuValue>,
    pub outputs: Vec<GpuVar>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuValue {
    Var(GpuVar),
    FnSymbol(Operation),
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
    #[error("generated function name `{operation}` expected one output, found {actual}")]
    InvalidNameArity { operation: Operation, actual: usize },
    #[error("generated function name `{0}` has an invalid target")]
    InvalidNameTarget(String),
    #[error("structural product operation `{operation}` has incompatible runtime components")]
    InvalidProduct { operation: Operation },
    #[error("a function symbol reached the runtime output of `{0}`")]
    FunctionOutput(Operation),
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
    let function_symbols = direct_function_symbols(&term)?;
    let mut aliases = HashMap::<NodeId, Vec<GpuValue>>::new();
    let mut sources = Vec::new();
    for node in &term.sources {
        let components = vars(*node, &term)?;
        sources.extend(components.iter().cloned());
        aliases.insert(*node, components.into_iter().map(GpuValue::Var).collect());
    }
    let mut assignments = Vec::new();
    for assignment in ssa(term.clone().to_strict())? {
        let op = assignment.op;
        if op.as_str().starts_with("name.") {
            continue;
        }
        if op.as_str() == "*.intro" {
            let inputs = resolve_nodes(
                assignment.sources.iter().map(|(node, _)| *node),
                &aliases,
                &function_symbols,
                &term,
            )?;
            let [(target, _)] = assignment.targets.as_slice() else {
                return Err(CodegenError::InvalidProduct { operation: op });
            };
            aliases.insert(*target, inputs);
            continue;
        }
        if op.as_str() == "*.elim" {
            let [(source, _)] = assignment.sources.as_slice() else {
                return Err(CodegenError::InvalidProduct { operation: op });
            };
            let components = resolve_node(*source, &aliases, &function_symbols, &term)?;
            let mut offset = 0;
            for (target, _) in &assignment.targets {
                let count = vars(*target, &term)?.len();
                let Some(values) = components.get(offset..offset + count) else {
                    return Err(CodegenError::InvalidProduct {
                        operation: op.clone(),
                    });
                };
                aliases.insert(*target, values.to_vec());
                offset += count;
            }
            if offset != components.len() {
                return Err(CodegenError::InvalidProduct { operation: op });
            }
            continue;
        }
        let inputs = resolve_nodes(
            assignment.sources.iter().map(|(node, _)| *node),
            &aliases,
            &function_symbols,
            &term,
        )?;
        let mut outputs = Vec::new();
        for (node, _) in &assignment.targets {
            let components = vars(*node, &term)?;
            outputs.extend(components.iter().cloned());
            aliases.insert(*node, components.into_iter().map(GpuValue::Var).collect());
        }
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
    let targets = resolve_nodes(
        term.targets.iter().copied(),
        &aliases,
        &function_symbols,
        &term,
    )?
    .into_iter()
    .map(|value| match value {
        GpuValue::Var(var) => Ok(var),
        GpuValue::FnSymbol(_) => Err(CodegenError::FunctionOutput(source_name.clone())),
    })
    .collect::<Result<Vec<_>, _>>()?;
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

fn direct_function_symbols(term: &CodegenTerm) -> Result<HashMap<NodeId, Operation>, CodegenError> {
    let mut symbols = HashMap::new();
    for (edge_index, operation) in term.hypergraph.edges.iter().enumerate() {
        let Some(target) = operation.as_str().strip_prefix("name.") else {
            continue;
        };
        let adjacency = &term.hypergraph.adjacency[edge_index];
        let [node] = adjacency.targets.as_slice() else {
            return Err(CodegenError::InvalidNameArity {
                operation: operation.clone(),
                actual: adjacency.targets.len(),
            });
        };
        let target = target
            .parse()
            .map_err(|_| CodegenError::InvalidNameTarget(target.to_string()))?;
        symbols.insert(*node, target);
    }
    Ok(symbols)
}

fn vars(node: NodeId, term: &CodegenTerm) -> Result<Vec<GpuVar>, LowerTypeError> {
    let mut output = Vec::new();
    lower_components(
        node,
        &term.hypergraph.nodes[node.0],
        &format!("x{}", node.0),
        &mut output,
    )?;
    Ok(output)
}

fn lower_components(
    node: NodeId,
    ty: &Tree<(), Operation>,
    name: &str,
    output: &mut Vec<GpuVar>,
) -> Result<(), LowerTypeError> {
    if let Tree::Node(operation, _, children) = ty
        && operation.as_str() == "*"
    {
        for (index, child) in children.iter().enumerate() {
            lower_components(node, child, &format!("{name}_{index}"), output)?;
        }
        return Ok(());
    }
    let lowered = lower_type(ty)?;
    if let LoweredType::Runtime(_) = &lowered {
        output.push(GpuVar {
            node,
            name: name.into(),
            lowered,
        });
    }
    Ok(())
}

fn resolve_nodes(
    nodes: impl IntoIterator<Item = NodeId>,
    aliases: &HashMap<NodeId, Vec<GpuValue>>,
    function_symbols: &HashMap<NodeId, Operation>,
    term: &CodegenTerm,
) -> Result<Vec<GpuValue>, LowerTypeError> {
    let mut output = Vec::new();
    for node in nodes {
        output.extend(resolve_node(node, aliases, function_symbols, term)?);
    }
    Ok(output)
}

fn resolve_node(
    node: NodeId,
    aliases: &HashMap<NodeId, Vec<GpuValue>>,
    function_symbols: &HashMap<NodeId, Operation>,
    term: &CodegenTerm,
) -> Result<Vec<GpuValue>, LowerTypeError> {
    if let Some(symbol) = function_symbols.get(&node) {
        return Ok(vec![GpuValue::FnSymbol(symbol.clone())]);
    }
    if let Some(values) = aliases.get(&node) {
        return Ok(values.clone());
    }
    Ok(vars(node, term)?.into_iter().map(GpuValue::Var).collect())
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
