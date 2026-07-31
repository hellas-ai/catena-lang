use std::collections::{BTreeMap, BTreeSet};

use hexpr::Operation;

use crate::{
    check::AnnotatedTerm,
    codegen::{CodegenError, GpuValue},
    pass::record_boundary_sizes::OperationWithBoundarySizes,
};

/// Validate restrictions imposed by operations that move execution to a GPU
/// kernel. Ordinary `materializec` stays unrestricted because its producer is
/// evaluated by the host loop.
pub(super) fn assignment(
    definitions: &BTreeMap<Operation, AnnotatedTerm<OperationWithBoundarySizes<Operation>>>,
    caller: &Operation,
    op: &Operation,
    input_sizes: &[usize],
    inputs: &[GpuValue],
) -> Result<(), CodegenError> {
    if op.as_str() != "gpu.materialize" {
        return Ok(());
    }

    let Some(producer) = last_component_function(input_sizes, inputs) else {
        return Ok(());
    };
    if let Some(containing) =
        first_gpu_materialize_in_call_chain(definitions, producer, &mut BTreeSet::new())
    {
        return Err(CodegenError::GpuMaterializeKernelContainsGpuMaterialize {
            caller: caller.clone(),
            producer: producer.clone(),
            containing,
        });
    }
    Ok(())
}

/// `gpu.materialize` receives `launch, size, kernel-env, kernel-fn` after
/// closure conversion. Select the final component so function symbols captured
/// in the kernel environment (for example matrix views) are not mistaken for
/// the kernel itself.
fn last_component_function<'a>(
    input_sizes: &[usize],
    inputs: &'a [GpuValue],
) -> Option<&'a Operation> {
    let (&function_size, prefix_sizes) = input_sizes.split_last()?;
    let function_start = prefix_sizes.iter().sum::<usize>();
    let function = inputs.get(function_start..function_start + function_size)?;
    let [GpuValue::FnSymbol(symbol)] = function else {
        return None;
    };
    Some(&symbol.target)
}

fn first_gpu_materialize_in_call_chain(
    definitions: &BTreeMap<Operation, AnnotatedTerm<OperationWithBoundarySizes<Operation>>>,
    definition: &Operation,
    visited: &mut BTreeSet<Operation>,
) -> Option<Operation> {
    if !visited.insert(definition.clone()) {
        return None;
    }
    let term = definitions.get(definition)?;
    for label in &term.hypergraph.edges {
        let op = &label.operation;
        if op.as_str() == "gpu.materialize" {
            return Some(definition.clone());
        }
        if definitions.contains_key(op)
            && let Some(found) = first_gpu_materialize_in_call_chain(definitions, op, visited)
        {
            return Some(found);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::{
        GpuVar,
        fn_ptrs::FnPtrSymbol,
        lower_types::{CType, LoweredType},
    };
    use open_hypergraphs::lax::NodeId;

    fn op(name: &str) -> Operation {
        name.parse().unwrap()
    }

    #[test]
    fn kernel_function_is_selected_after_function_valued_captures() {
        let inputs = [
            GpuValue::Var(GpuVar {
                node: NodeId(0),
                name: "launch".to_string(),
                lowered: LoweredType::Runtime(CType::Named("catena_launch_params_t".to_string())),
            }),
            GpuValue::Var(GpuVar {
                node: NodeId(1),
                name: "length".to_string(),
                lowered: LoweredType::Runtime(CType::U64),
            }),
            GpuValue::FnSymbol(FnPtrSymbol {
                target: op("captured-view"),
            }),
            GpuValue::FnSymbol(FnPtrSymbol {
                target: op("kernel"),
            }),
        ];

        assert_eq!(
            last_component_function(&[1, 1, 1, 1], &inputs),
            Some(&op("kernel"))
        );
    }
}
