use std::collections::{BTreeMap, BTreeSet};

use hexpr::Operation;

use crate::{
    check::AnnotatedTerm,
    codegen::{CodegenError, GpuValue},
    pass::record_boundary_sizes::OperationWithBoundarySizes,
};

pub(super) fn assignment(
    definitions: &BTreeMap<Operation, AnnotatedTerm<OperationWithBoundarySizes<Operation>>>,
    caller: &Operation,
    op: &Operation,
    inputs: &[GpuValue],
) -> Result<(), CodegenError> {
    match op.as_str() {
        "materializec"
        | "materializec.into"
        | "materializec.borrow"
        | "materializec.borrow-argmax-f32"
        | "materializec.borrow-reduce-f32"
        | "materializec.reduce-f32"
        | "materializec.reduce-f32-pair"
        | "materializec.softmax-f32" => materializec_producer(definitions, caller, inputs),
        _ => Ok(()),
    }
}

fn materializec_producer(
    definitions: &BTreeMap<Operation, AnnotatedTerm<OperationWithBoundarySizes<Operation>>>,
    caller: &Operation,
    inputs: &[GpuValue],
) -> Result<(), CodegenError> {
    for producer in inputs.iter().filter_map(|input| match input {
        GpuValue::FnSymbol(symbol) => Some(&symbol.target),
        GpuValue::Var(_) => None,
    }) {
        if let Some((containing, nested)) =
            first_materialize_op_in_call_chain(definitions, producer, &mut BTreeSet::new())
        {
            return Err(CodegenError::MaterializecProducerContainsMaterialize {
                caller: caller.clone(),
                producer: producer.clone(),
                containing,
                nested,
            });
        }
    }
    Ok(())
}

fn first_materialize_op_in_call_chain(
    definitions: &BTreeMap<Operation, AnnotatedTerm<OperationWithBoundarySizes<Operation>>>,
    definition: &Operation,
    visited: &mut BTreeSet<Operation>,
) -> Option<(Operation, Operation)> {
    if !visited.insert(definition.clone()) {
        return None;
    }
    let term = definitions.get(definition)?;
    for label in &term.hypergraph.edges {
        let op = &label.operation;
        if is_materialize_op(op) {
            return Some((definition.clone(), op.clone()));
        }
        if definitions.contains_key(op)
            && let Some(found) = first_materialize_op_in_call_chain(definitions, op, visited)
        {
            return Some(found);
        }
    }
    None
}

fn is_materialize_op(op: &Operation) -> bool {
    matches!(
        op.as_str(),
        "materialize"
            | "materializec"
            | "materializec.into"
            | "materializec.borrow"
            | "materializec.borrow-argmax-f32"
            | "materializec.borrow-topk-f32"
            | "materializec.borrow-reduce-f32"
            | "materializec.borrow-routed-bf16-gemv-pair"
            | "materializec.reduce-f32"
            | "materializec.reduce-f32-pair"
            | "materializec.softmax-f32"
            | "gpu.materialize"
    )
}
