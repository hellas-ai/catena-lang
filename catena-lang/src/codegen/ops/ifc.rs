//! `bool.ifc` selects and calls one of two closure-converted functions.

use crate::codegen::{
    GpuAssign, GpuValue,
    components::{
        Component, input_components, runtime_values, single_function, single_value, value_expr,
    },
    gpu::GpuRenderError,
    render_utils::invalid_outputs,
};

pub(in crate::codegen) fn render(
    out: &mut String,
    assignment: &GpuAssign,
) -> Result<(), GpuRenderError> {
    let [output] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 1));
    };
    let (true_env, true_fn, false_env, false_fn, flag, argument) = parts(assignment)?;

    let true_args = call_args(true_env, argument, output.name.as_str());
    let false_args = call_args(false_env, argument, output.name.as_str());
    out.push_str(&format!(
        "    if ({flag}) {{ {true_fn}({true_args}); }} else {{ {false_fn}({false_args}); }}\n",
        flag = value_expr(flag),
        true_fn = value_expr(true_fn),
        true_args = true_args.join(", "),
        false_fn = value_expr(false_fn),
        false_args = false_args.join(", "),
    ));
    Ok(())
}

type IfcParts<'a> = (
    Component<'a>,
    &'a GpuValue,
    Component<'a>,
    &'a GpuValue,
    &'a GpuValue,
    &'a GpuValue,
);

fn parts(assignment: &GpuAssign) -> Result<IfcParts<'_>, GpuRenderError> {
    let components = input_components(assignment)?;
    let [true_env, true_fn, false_env, false_fn, flag, argument] = components.as_slice() else {
        return Err(GpuRenderError::InvalidInputComponentCount {
            op: assignment.op.clone(),
            expected: 6,
            actual: components.len(),
        });
    };

    let true_fn = single_function(true_fn)
        .map_err(|error| invalid_component_count(assignment, "true_fn", error.actual))?;
    let false_fn = single_function(false_fn)
        .map_err(|error| invalid_component_count(assignment, "false_fn", error.actual))?;
    let flag = single_value(flag)
        .map_err(|error| invalid_component_count(assignment, "flag", error.actual))?;
    let argument = single_value(argument)
        .map_err(|error| invalid_component_count(assignment, "argument", error.actual))?;

    Ok((true_env, true_fn, false_env, false_fn, flag, argument))
}

fn invalid_component_count(
    assignment: &GpuAssign,
    component: &'static str,
    actual: usize,
) -> GpuRenderError {
    GpuRenderError::InvalidInputComponentValueCount {
        op: assignment.op.clone(),
        component,
        description: "single input value",
        expected: 1,
        actual,
    }
}

fn call_args(environment: Component<'_>, argument: &GpuValue, output: &str) -> Vec<String> {
    runtime_values(environment)
        .map(value_expr)
        .chain([value_expr(argument), format!("&{output}")])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::{
        GpuVar,
        fn_ptrs::FnPtrSymbol,
        lower_types::{CType, LoweredType},
    };
    use hexpr::Operation;
    use open_hypergraphs::lax::NodeId;

    fn op(name: &str) -> Operation {
        name.parse().unwrap()
    }

    fn var(node: usize, name: &str) -> GpuValue {
        GpuValue::Var(GpuVar {
            node: NodeId(node),
            name: name.to_string(),
            lowered: LoweredType::Runtime(CType::Bool),
        })
    }

    fn fn_symbol(name: &str) -> GpuValue {
        GpuValue::FnSymbol(FnPtrSymbol { target: op(name) })
    }

    #[test]
    fn zero_sized_environments_are_valid_components() {
        let assignment = GpuAssign {
            op: op("bool.ifc"),
            input_sizes: vec![0, 1, 0, 1, 1, 1],
            output_sizes: vec![1],
            call_symbol: None,
            inputs: vec![
                fn_symbol("true"),
                fn_symbol("false"),
                var(0, "flag"),
                var(1, "argument"),
            ],
            outputs: vec![GpuVar {
                node: NodeId(2),
                name: "output".to_string(),
                lowered: LoweredType::Runtime(CType::Bool),
            }],
        };

        let mut out = String::new();
        render(&mut out, &assignment).unwrap();

        assert!(out.contains("program_true(argument, &output)"));
        assert!(out.contains("program_false(argument, &output)"));
    }
}
