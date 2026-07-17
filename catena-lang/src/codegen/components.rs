use crate::codegen::{
    GpuAssign, GpuValue, GpuVar, gpu::GpuRenderError, render_utils::sanitize_ident, runtime_type,
};

pub(in crate::codegen) type Component<'a> = &'a [GpuValue];
pub(in crate::codegen) type OutputComponent<'a> = &'a [GpuVar];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::codegen) struct ComponentValueCountError {
    pub actual: usize,
}

pub(in crate::codegen) fn input_components<'a>(
    assignment: &'a GpuAssign,
) -> Result<Vec<Component<'a>>, GpuRenderError> {
    let expected = assignment.input_sizes.iter().sum::<usize>();
    if expected != assignment.inputs.len() {
        return Err(GpuRenderError::InvalidFlattenedInputCount {
            op: assignment.op.clone(),
            expected,
            actual: assignment.inputs.len(),
        });
    }

    let mut offset = 0;
    Ok(assignment
        .input_sizes
        .iter()
        .map(|size| {
            let end = offset + *size;
            let component = &assignment.inputs[offset..end];
            offset = end;
            component
        })
        .collect())
}

pub(in crate::codegen) fn output_components(
    assignment: &GpuAssign,
) -> Result<Vec<OutputComponent<'_>>, GpuRenderError> {
    let expected = assignment.output_sizes.iter().sum::<usize>();
    if expected != assignment.outputs.len() {
        return Err(GpuRenderError::InvalidFlattenedOutputCount {
            op: assignment.op.clone(),
            expected,
            actual: assignment.outputs.len(),
        });
    }

    let mut offset = 0;
    Ok(assignment
        .output_sizes
        .iter()
        .map(|size| {
            let end = offset + *size;
            let component = &assignment.outputs[offset..end];
            offset = end;
            component
        })
        .collect())
}

pub(in crate::codegen) fn single_function(
    component: Component<'_>,
) -> Result<&GpuValue, ComponentValueCountError> {
    let functions = component
        .iter()
        .filter(|value| matches!(value, GpuValue::FnSymbol(_)))
        .collect::<Vec<_>>();
    let [function] = functions.as_slice() else {
        return Err(ComponentValueCountError {
            actual: functions.len(),
        });
    };
    Ok(*function)
}

pub(in crate::codegen) fn single_value(
    component: Component<'_>,
) -> Result<&GpuValue, ComponentValueCountError> {
    let values = runtime_values(component).collect::<Vec<_>>();
    let [value] = values.as_slice() else {
        return Err(ComponentValueCountError {
            actual: values.len(),
        });
    };
    Ok(*value)
}

pub(in crate::codegen) fn runtime_values(
    component: Component<'_>,
) -> impl Iterator<Item = &GpuValue> {
    component.iter().filter(|value| is_runtime_value(value))
}

pub(in crate::codegen) fn is_runtime_value(value: &GpuValue) -> bool {
    matches!(value, GpuValue::Var(var) if runtime_type(var).is_some())
}

pub(in crate::codegen) fn value_expr(value: &GpuValue) -> String {
    match value {
        GpuValue::Var(var) => var.name.clone(),
        GpuValue::FnSymbol(symbol) => sanitize_ident(&format!("program.{}", symbol.target)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::{
        GpuAssign, GpuVar,
        fn_ptrs::FnPtrSymbol,
        lower_types::{CType, LoweredType},
    };
    use hexpr::Operation;
    use open_hypergraphs::lax::NodeId;

    fn op(name: &str) -> Operation {
        name.parse().unwrap()
    }

    fn var(node: usize, name: &str) -> GpuValue {
        GpuValue::Var(output_var(node, name))
    }

    fn output_var(node: usize, name: &str) -> GpuVar {
        GpuVar {
            node: NodeId(node),
            name: name.to_string(),
            lowered: LoweredType::Runtime(CType::U64),
        }
    }

    fn fn_symbol(name: &str) -> GpuValue {
        GpuValue::FnSymbol(FnPtrSymbol { target: op(name) })
    }

    #[test]
    fn input_sizes_split_flattened_inputs_into_components() {
        let assignment = GpuAssign {
            op: op("f"),
            input_sizes: vec![1, 0, 2],
            output_sizes: vec![],
            call_symbol: None,
            inputs: vec![var(0, "a"), var(1, "b"), fn_symbol("g")],
            outputs: vec![],
        };

        let components = input_components(&assignment).unwrap();

        assert_eq!(components[0], [var(0, "a")]);
        assert!(components[1].is_empty());
        assert_eq!(components[2], [var(1, "b"), fn_symbol("g")]);
    }

    #[test]
    fn output_sizes_split_flattened_outputs_into_components() {
        let assignment = GpuAssign {
            op: op("f"),
            input_sizes: vec![],
            output_sizes: vec![2, 0, 1],
            call_symbol: None,
            inputs: vec![],
            outputs: vec![output_var(0, "a"), output_var(1, "b"), output_var(2, "c")],
        };

        let components = output_components(&assignment).unwrap();

        assert_eq!(components[0].len(), 2);
        assert!(components[1].is_empty());
        assert_eq!(components[2], [output_var(2, "c")]);
    }

    #[test]
    fn single_selectors_count_values_by_kind() {
        let component = [var(0, "a"), fn_symbol("g")];

        assert_eq!(single_value(&component).unwrap(), &var(0, "a"));
        assert_eq!(single_function(&component).unwrap(), &fn_symbol("g"));
    }
}
