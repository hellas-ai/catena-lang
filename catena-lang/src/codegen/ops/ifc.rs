//! `bool.ifc` selects and calls one of two closure-converted functions.

use crate::codegen::{
    GpuAssign, GpuValue,
    components::{
        Component, input_components, output_components, single_function, single_value, value_expr,
    },
    gpu::{GpuRenderError, render_function_application},
};

pub(in crate::codegen) fn render(
    out: &mut String,
    assignment: &GpuAssign,
) -> Result<(), GpuRenderError> {
    let output_components = output_components(assignment)?;
    let [outputs] = output_components.as_slice() else {
        return Err(GpuRenderError::InvalidOutputComponentCount {
            op: assignment.op.clone(),
            expected: 1,
            actual: output_components.len(),
        });
    };
    let (true_env, true_fn, false_env, false_fn, flag, argument) = parts(assignment)?;

    let true_inputs = true_env
        .iter()
        .chain(argument.iter())
        .cloned()
        .collect::<Vec<_>>();
    let false_inputs = false_env
        .iter()
        .chain(argument.iter())
        .cloned()
        .collect::<Vec<_>>();
    out.push_str(&format!("    if ({}) {{\n", value_expr(flag)));
    render_function_application(out, "        ", true_fn, &true_inputs, outputs)?;
    out.push_str("    } else {\n");
    render_function_application(out, "        ", false_fn, &false_inputs, outputs)?;
    out.push_str("    }\n");
    Ok(())
}

type IfcParts<'a> = (
    Component<'a>,
    &'a GpuValue,
    Component<'a>,
    &'a GpuValue,
    &'a GpuValue,
    Component<'a>,
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

    fn output(node: usize, name: &str) -> GpuVar {
        GpuVar {
            node: NodeId(node),
            name: name.to_string(),
            lowered: LoweredType::Runtime(CType::Bool),
        }
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

    #[test]
    fn zero_sized_arguments_are_valid_components() {
        let assignment = GpuAssign {
            op: op("bool.ifc"),
            input_sizes: vec![1, 1, 1, 1, 1, 0],
            output_sizes: vec![1],
            call_symbol: None,
            inputs: vec![
                var(0, "true_env"),
                fn_symbol("true"),
                var(1, "false_env"),
                fn_symbol("false"),
                var(2, "flag"),
            ],
            outputs: vec![GpuVar {
                node: NodeId(3),
                name: "output".to_string(),
                lowered: LoweredType::Runtime(CType::Bool),
            }],
        };

        let mut out = String::new();
        render(&mut out, &assignment).unwrap();

        assert!(out.contains("program_true(true_env, &output)"));
        assert!(out.contains("program_false(false_env, &output)"));
    }

    #[test]
    fn product_results_pass_every_flattened_output_to_both_branches() {
        let assignment = GpuAssign {
            op: op("bool.ifc"),
            input_sizes: vec![2, 1, 2, 1, 1, 0],
            output_sizes: vec![2],
            call_symbol: None,
            inputs: vec![
                var(0, "true_env0"),
                var(1, "true_env1"),
                fn_symbol("true"),
                var(2, "false_env0"),
                var(3, "false_env1"),
                fn_symbol("false"),
                var(4, "flag"),
            ],
            outputs: vec![output(5, "output0"), output(6, "output1")],
        };

        let mut out = String::new();
        render(&mut out, &assignment).unwrap();

        assert!(out.contains("program_true(true_env0, true_env1, &output0, &output1)"));
        assert!(out.contains("program_false(false_env0, false_env1, &output0, &output1)"));
    }

    #[test]
    fn primitive_function_branches_render_inline() {
        let assignment = GpuAssign {
            op: op("bool.ifc"),
            input_sizes: vec![1, 1, 1, 1, 1, 1],
            output_sizes: vec![1],
            call_symbol: None,
            inputs: vec![
                var(0, "true_value"),
                fn_symbol("bool.and"),
                var(1, "false_value"),
                fn_symbol("bool.or"),
                var(2, "flag"),
                var(3, "argument"),
            ],
            outputs: vec![output(4, "output")],
        };

        let mut out = String::new();
        render(&mut out, &assignment).unwrap();

        assert!(out.contains("output = true_value && argument;"));
        assert!(out.contains("output = false_value || argument;"));
        assert!(!out.contains("program_bool_and"));
        assert!(!out.contains("program_bool_or"));
    }
}
