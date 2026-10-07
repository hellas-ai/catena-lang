//! Experimental GPU lowering of typed, closure-converted graphs.
//!
//! Only `codegen` and its error type cross the backend boundary. Lowering, IR,
//! and rendering are private to this module and its descendants.

mod gpu;
mod ir;
mod lower;
mod lower_types;
mod meta;
mod ops;
mod prelude;
mod smolcat_compat;
#[cfg(test)]
mod tests;
mod validate;
mod values;

use crate::{
    check::AnnotatedTerm,
    runtime::{GeneratedFunction, GpuDialect, RuntimeModule},
};
use hexpr::Operation;
use ir::*;
use lower_types::{children, node, runtime};
use metacat::{theory::TheoryId, tree::Tree};
use std::collections::BTreeMap;
use thiserror::Error;
use values::*;

#[derive(Debug, Error)]
pub enum CodegenError {
    #[error("experimental codegen: invalid GPU IR in `{function}`: {reason}")]
    InvalidIr { function: String, reason: String },
    #[error("experimental codegen: {0}")]
    Type(String),
    #[error("experimental codegen: type `{0:?}` has no runtime representation")]
    NoRuntimeRepresentation(Obj),
    #[error("experimental codegen does not support arrow `{0}`")]
    Unsupported(String),
    #[error("experimental codegen: invalid `{op}`: {reason}")]
    Invalid { op: String, reason: String },
    #[error("experimental codegen: recursive definition `{0}` requires an explicit fold")]
    Recursive(String),
    #[error("experimental codegen: SSA decomposition failed: {0}")]
    Ssa(#[from] metacat::ssa::SSAError),
}

fn invalid(op: &str, reason: impl Into<String>) -> CodegenError {
    CodegenError::Invalid {
        op: op.into(),
        reason: reason.into(),
    }
}

struct Template {
    term: AnnotatedTerm<crate::pass::record_boundary_sizes::OperationWithBoundarySizes<Operation>>,
}

struct Lowerer<'a> {
    templates: &'a BTreeMap<Operation, Template>,
    modules: Modules,
    body: Vec<Instruction>,
    place: Place,
    next: usize,
    next_type: usize,
    stack: Vec<Operation>,
    kernel_cache: Vec<(ops::launch::KernelKey, String)>,
}

pub(super) fn codegen(
    terms: &super::CodegenTermMap,
    dialect: GpuDialect,
) -> Result<RuntimeModule, CodegenError> {
    let modules = lower_to_ir(terms)?;
    Ok(render_runtime_module(&modules, dialect)?)
}

fn lower_to_ir(terms: &super::CodegenTermMap) -> Result<Modules, CodegenError> {
    let id = TheoryId("program".parse().unwrap());
    let mut templates = BTreeMap::new();
    if let Some(definitions) = terms.get(&id) {
        for (name, term) in definitions {
            templates.insert(name.clone(), Template { term: term.clone() });
        }
    }

    let mut lower = Lowerer::new(&templates);
    for (name, template) in &templates {
        // Library definitions and generic helpers are lowered at their call sites.
        if name.as_str().starts_with("stdlib.")
            || name.as_str().starts_with("core.")
            || name.as_str().starts_with("partial.")
            || name.as_str().starts_with("catena.")
            || name.as_str().starts_with("closure.")
            || name.as_str() == "evaluate"
        {
            continue;
        }
        let input_types = template
            .term
            .sources
            .iter()
            .map(|n| template.term.hypergraph.nodes[n.0].clone())
            .collect::<Vec<_>>();
        let output_types = template
            .term
            .targets
            .iter()
            .map(|n| template.term.hypergraph.nodes[n.0].clone())
            .collect::<Vec<_>>();
        if let Some(ty) = input_types
            .iter()
            .chain(&output_types)
            .find(|ty| has_abstract_runtime(ty))
        {
            return Err(CodegenError::NoRuntimeRepresentation(ty.clone()));
        }
        // Bare symbolic witnesses are erased, not unresolved runtime values.
        // Keep definitions with these generic interfaces available for call-site lowering.
        if input_types
            .iter()
            .chain(&output_types)
            .any(has_symbolic_witness)
        {
            continue;
        }
        if !input_types
            .iter()
            .chain(&output_types)
            .map(public_type)
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .all(|v| v)
        {
            continue;
        }
        lower.body.clear();
        let mut inputs = Vec::new();
        let values = input_types
            .iter()
            .map(|ty| lower.input(ty, &mut inputs))
            .collect::<Result<Vec<_>, _>>()?;
        let results = lower.call(name, values, &output_types)?;
        let mut outputs = Vec::new();
        for result in results {
            let mut fields = Vec::new();
            result.runtime_values(&mut fields);
            for field in fields {
                let ty = runtime(&field.ty)?
                    .ok_or_else(|| invalid(name.as_str(), "non-runtime output"))?;
                let variable = Variable {
                    name: format!("out{}", outputs.len()),
                    ty,
                };
                lower.body.push(Instruction::Assign(
                    format!("*{}", variable.name),
                    expr(&field)?.into(),
                ));
                outputs.push(variable);
            }
        }
        let abi_inputs = inputs
            .iter()
            .map(|v| v.ty.abi())
            .collect::<Option<Vec<_>>>();
        let abi_outputs = outputs
            .iter()
            .map(|v| v.ty.abi())
            .collect::<Option<Vec<_>>>();
        let (Some(abi_inputs), Some(abi_outputs)) = (abi_inputs, abi_outputs) else {
            return Err(invalid(
                name.as_str(),
                "entry-point interface is not supported by the runtime ABI",
            ));
        };
        let symbol = symbol(name.as_str());
        lower.modules.exports.push(GeneratedFunction {
            source_name: name.to_string(),
            symbol: symbol.clone(),
            inputs: abi_inputs,
            outputs: abi_outputs,
        });
        lower.modules.functions.insert(
            symbol.clone(),
            Function {
                symbol,
                inputs,
                outputs,
                body: std::mem::take(&mut lower.body),
            },
        );
    }
    validate::validate(&lower.modules)?;
    Ok(lower.modules)
}

fn public_type(ty: &Obj) -> Result<bool, CodegenError> {
    if let Some(fields) = children(ty, "*") {
        return Ok(fields
            .iter()
            .map(public_type)
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .all(|v| v));
    }
    Ok(runtime(ty)?.is_none_or(|t| t.abi().is_some()))
}

fn has_abstract_runtime(ty: &Obj) -> bool {
    if children(ty, "=>").is_some() || children(ty, "->").is_some() {
        return true;
    }
    if let Some(args) = children(ty, "*") {
        return args.iter().any(has_abstract_runtime);
    }
    let inner = children(ty, "val")
        .and_then(|a| a.first())
        .or_else(|| children(ty, ":").and_then(|a| a.get(1)));
    if let Some(inner) = inner {
        if matches!(inner, Tree::Leaf(..)) || children(inner, "->").is_some() {
            return true;
        }
        if let Some([_, element]) = children(inner, "stdlib.gpu.memory.type.Global") {
            return matches!(element, Tree::Leaf(..));
        }
        if let Some([_, _, _, _, _, _, _, _, element]) =
            children(inner, "stdlib.gpu.memory.type.Shared")
        {
            return matches!(element, Tree::Leaf(..));
        }
        let layout = children(inner, "stdlib.gpu.memory.type.SharedLayout")
            .and_then(|a| a.first())
            .or_else(|| {
                children(inner, "stdlib.gpu.memory.type.SharedMemory").and_then(|a| a.get(3))
            });
        if let Some(mut tail) = layout {
            while let Some([_, next]) = children(tail, "list.Cons") {
                tail = next;
            }
            return matches!(tail, Tree::Leaf(..));
        }
        if let Some([cap]) = children(inner, "mem") {
            return matches!(cap, Tree::Leaf(..));
        }
    }
    false
}

fn has_symbolic_witness(ty: &Obj) -> bool {
    matches!(ty, Tree::Leaf(..))
        || children(ty, "*").is_some_and(|fields| fields.iter().any(has_symbolic_witness))
}

fn render_runtime_module(
    modules: &Modules,
    dialect: GpuDialect,
) -> Result<RuntimeModule, CodegenError> {
    validate::validate(modules)?;
    Ok(RuntimeModule::new(
        dialect,
        gpu::render(modules, dialect),
        modules.exports.clone(),
    ))
}

fn symbol(name: &str) -> String {
    // Keep public symbols separate from prelude helpers and generated kernels.
    let mut result = "catena_fn_".to_string();
    for byte in name.bytes() {
        if byte.is_ascii_alphanumeric() {
            result.push(byte as char);
        } else {
            result.push_str(&format!("_{byte:02x}"));
        }
    }
    result
}

fn expr(value: &Value) -> Result<&str, CodegenError> {
    match &value.repr {
        Repr::Runtime(s) => Ok(s),
        _ => Err(CodegenError::Type(format!(
            "expected runtime value, got {:?}",
            value.ty
        ))),
    }
}

fn product(ty: Obj, fields: Vec<Value>) -> Value {
    Value {
        ty,
        repr: Repr::Product(fields),
    }
}

fn pack(mut values: Vec<Value>) -> Value {
    if values.is_empty() {
        return Value::erased(node("1", vec![]));
    }
    let mut value = values.pop().unwrap();
    while let Some(head) = values.pop() {
        value = product(
            node("*", vec![head.ty.clone(), value.ty.clone()]),
            vec![head, value],
        );
    }
    value
}

fn function_parts(ty: &Obj) -> Result<(&Obj, &Obj), CodegenError> {
    let inner = children(ty, "val").and_then(|a| a.first()).unwrap_or(ty);
    let Some([domain, codomain]) = children(inner, "->") else {
        return Err(invalid(
            "function reference",
            "expected a converted function type",
        ));
    };
    Ok((domain, codomain))
}
