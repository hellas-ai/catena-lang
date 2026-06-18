use std::collections::{BTreeMap, BTreeSet};

use hexpr::Operation;

use crate::{
    codegen::{CodegenError, GpuValue},
    report::AnnotatedTerm,
};

pub(super) fn assignment(
    definitions: &BTreeMap<Operation, AnnotatedTerm>,
    caller: &Operation,
    op: &Operation,
    inputs: &[GpuValue],
) -> Result<(), CodegenError> {
    match op.as_str() {
        "materializec" => materializec_producer(definitions, caller, inputs),
        _ => Ok(()),
    }
}

fn materializec_producer(
    definitions: &BTreeMap<Operation, AnnotatedTerm>,
    caller: &Operation,
    inputs: &[GpuValue],
) -> Result<(), CodegenError> {
    let Some(producer) = inputs.iter().find_map(|input| match input {
        GpuValue::FnSymbol(symbol) => Some(&symbol.target),
        GpuValue::Var(_) => None,
    }) else {
        return Ok(());
    };
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
    Ok(())
}

fn first_materialize_op_in_call_chain(
    definitions: &BTreeMap<Operation, AnnotatedTerm>,
    definition: &Operation,
    visited: &mut BTreeSet<Operation>,
) -> Option<(Operation, Operation)> {
    if !visited.insert(definition.clone()) {
        return None;
    }
    let term = definitions.get(definition)?;
    for op in &term.hypergraph.edges {
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
        "materialize" | "materializec" | "gpu.materialize"
    )
}
