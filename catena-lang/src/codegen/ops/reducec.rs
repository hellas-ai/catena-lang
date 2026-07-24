//! `reducec` lowers a closure-converted reduction to a sequential fold.
//!
//! The lowered call shape is:
//!
//! ```text
//! zero, add_env..., add_fn, get_env..., get_fn, erased_witnesses..., n -> out
//! ```
//!
//! and the generated C++ is roughly:
//!
//! ```cpp
//! out = zero;
//! for (uint64_t i = 0; i < n; ++i) {
//!     A value;
//!     A next;
//!     get_fn(get_env..., i, &value);
//!     add_fn(add_env..., out, value, &next);
//!     out = next;
//! }
//! ```

use crate::codegen::{
    GpuAssign, GpuValue, GpuVar,
    components::{
        Component, input_components, is_runtime_value, output_components, single_function,
        single_value, value_expr,
    },
    gpu::{GpuRenderError, render_function_application},
    lower_types::{CType, LoweredType},
    render_utils::{c_type, invalid_outputs},
    runtime_type,
};

pub(in crate::codegen) fn render(
    out: &mut String,
    assignment: &GpuAssign,
) -> Result<(), GpuRenderError> {
    let (zero, add_env, add_fn, get_env, get_fn, n) = parts(assignment)?;
    let output_components = output_components(assignment)?;
    let [outputs] = output_components.as_slice() else {
        return Err(GpuRenderError::InvalidOutputComponentCount {
            op: assignment.op.clone(),
            expected: 1,
            actual: output_components.len(),
        });
    };
    if zero.len() != outputs.len() {
        return Err(GpuRenderError::InvalidInputComponentValueCount {
            op: assignment.op.clone(),
            component: "zero",
            description: "reducec zero input must match the accumulator arity",
            expected: outputs.len(),
            actual: zero.len(),
        });
    }
    if outputs.is_empty() {
        return Err(invalid_outputs(assignment, 1));
    }

    let i = format!("reduce_i_{}", outputs[0].name);
    let values = outputs
        .iter()
        .map(|output| {
            Ok((
                output,
                runtime_type(output).ok_or_else(|| GpuRenderError::ErasedType(output.clone()))?,
                format!("reduce_value_{}", output.name),
                format!("reduce_next_{}", output.name),
            ))
        })
        .collect::<Result<Vec<_>, GpuRenderError>>()?;

    // Keep the accumulator in the output slot itself. This makes the final
    // assignment to the function's out-pointer use the usual renderer path.
    out.push_str("    {\n");
    for ((output, _ty, _value, _next), zero_value) in values.iter().zip(zero.iter()) {
        out.push_str(&format!(
            "        {} = {};\n",
            output.name,
            value_expr(zero_value)
        ));
    }
    out.push_str(&format!(
        "        for (uint64_t {i} = 0; {i} < {n}; ++{i}) {{\n",
        n = value_expr(n),
    ));
    for (_output, ty, value, next) in &values {
        out.push_str(&format!("            {} {value};\n", c_type(ty)));
        out.push_str(&format!("            {} {next};\n", c_type(ty)));
    }

    // The producer closure is called with its environment and the current
    // index. Primitive function symbols are rendered inline by the same
    // primitive table used for direct assignments.
    let mut get_inputs = get_env.to_vec();
    get_inputs.push(GpuValue::Var(GpuVar {
        node: outputs[0].node,
        name: i.clone(),
        lowered: LoweredType::Runtime(CType::U64),
    }));
    let get_outputs = values
        .iter()
        .map(|(output, _ty, value, _next)| GpuVar {
            name: value.clone(),
            ..(*output).clone()
        })
        .collect::<Vec<_>>();
    render_function_application(out, "            ", get_fn, &get_inputs, &get_outputs)?;

    // The accumulator closure receives its environment, the current
    // accumulator, and the freshly produced element.
    let mut add_inputs = add_env.to_vec();
    add_inputs.extend(
        values
            .iter()
            .map(|(output, _ty, _value, _next)| GpuValue::Var((*output).clone())),
    );
    add_inputs.extend(get_outputs.iter().cloned().map(GpuValue::Var));
    let add_outputs = values
        .iter()
        .map(|(output, _ty, _value, next)| GpuVar {
            name: next.clone(),
            ..(*output).clone()
        })
        .collect::<Vec<_>>();
    render_function_application(out, "            ", add_fn, &add_inputs, &add_outputs)?;
    for (output, _ty, _value, next) in &values {
        out.push_str(&format!("            {} = {next};\n", output.name));
    }
    out.push_str("        }\n");
    out.push_str("    }\n");
    Ok(())
}

type ReducecParts<'a> = (
    Component<'a>,
    Component<'a>,
    &'a GpuValue,
    Component<'a>,
    &'a GpuValue,
    &'a GpuValue,
);

fn parts(assignment: &GpuAssign) -> Result<ReducecParts<'_>, GpuRenderError> {
    let components = input_components(assignment)?;
    let [zero, add_env, add_fn, get_env, get_fn, n] = components.as_slice() else {
        return Err(GpuRenderError::InvalidInputComponentCount {
            op: assignment.op.clone(),
            expected: 6,
            actual: components.len(),
        });
    };

    if zero.is_empty() {
        return Err(invalid_component_count(
            assignment,
            "zero",
            "runtime zero input",
            0,
        ));
    }
    if !zero.iter().all(|value| is_runtime_value(value)) {
        return Err(GpuRenderError::ErasedInputComponentValue {
            op: assignment.op.clone(),
            component: "zero",
            description: "reducec zero input must be runtime values",
        });
    }
    let add_fn = single_function(add_fn).map_err(|error| {
        invalid_component_count(assignment, "add_fn", "function symbol input", error.actual)
    })?;
    let get_fn = single_function(get_fn).map_err(|error| {
        invalid_component_count(assignment, "get_fn", "function symbol input", error.actual)
    })?;
    let n = single_value(n).map_err(|error| {
        invalid_component_count(assignment, "n", "runtime length input", error.actual)
    })?;

    Ok((zero, add_env, add_fn, get_env, get_fn, n))
}

fn invalid_component_count(
    assignment: &GpuAssign,
    component: &'static str,
    description: &'static str,
    actual: usize,
) -> GpuRenderError {
    GpuRenderError::InvalidInputComponentValueCount {
        op: assignment.op.clone(),
        component,
        description,
        expected: 1,
        actual,
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
        GpuValue::Var(GpuVar {
            node: NodeId(node),
            name: name.to_string(),
            lowered: LoweredType::Runtime(CType::U64),
        })
    }

    fn fn_symbol(name: &str) -> GpuValue {
        GpuValue::FnSymbol(FnPtrSymbol { target: op(name) })
    }

    #[test]
    fn input_sizes_group_flattened_reducec_environments() {
        let output = GpuVar {
            node: NodeId(9),
            name: "out".to_string(),
            lowered: LoweredType::Runtime(CType::U64),
        };
        let assignment = GpuAssign {
            op: op("reducec"),
            input_sizes: vec![1, 2, 1, 2, 1, 1],
            output_sizes: vec![1],
            call_symbol: None,
            inputs: vec![
                var(0, "zero"),
                var(1, "add_env0"),
                var(2, "add_env1"),
                fn_symbol("add"),
                var(3, "get_env0"),
                var(4, "get_env1"),
                fn_symbol("get"),
                var(5, "n"),
            ],
            outputs: vec![output],
        };

        let mut out = String::new();
        render(&mut out, &assignment).unwrap();

        assert!(out.contains("program_get(get_env0, get_env1, reduce_i_out, &reduce_value_out);"));
        assert!(
            out.contains(
                "program_add(add_env0, add_env1, out, reduce_value_out, &reduce_next_out);"
            )
        );
    }

    #[test]
    fn primitive_accumulator_function_renders_inline() {
        let output = GpuVar {
            node: NodeId(9),
            name: "out".to_string(),
            lowered: LoweredType::Runtime(CType::U64),
        };
        let assignment = GpuAssign {
            op: op("reducec"),
            input_sizes: vec![1, 0, 1, 0, 1, 1],
            output_sizes: vec![1],
            call_symbol: None,
            inputs: vec![
                var(0, "zero"),
                fn_symbol("u64.add"),
                fn_symbol("get"),
                var(1, "n"),
            ],
            outputs: vec![output],
        };

        let mut out = String::new();
        render(&mut out, &assignment).unwrap();

        assert!(out.contains("reduce_next_out = out + reduce_value_out;"));
        assert!(!out.contains("program_u64_add"));
    }

    #[test]
    fn product_accumulators_pass_every_flattened_output() {
        let outputs = vec![
            GpuVar {
                node: NodeId(5),
                name: "out0".to_string(),
                lowered: LoweredType::Runtime(CType::U64),
            },
            GpuVar {
                node: NodeId(6),
                name: "out1".to_string(),
                lowered: LoweredType::Runtime(CType::U64),
            },
        ];
        let assignment = GpuAssign {
            op: op("reducec"),
            input_sizes: vec![2, 0, 1, 0, 1, 1],
            output_sizes: vec![2],
            call_symbol: None,
            inputs: vec![
                var(0, "zero0"),
                var(1, "zero1"),
                fn_symbol("add"),
                fn_symbol("get"),
                var(2, "n"),
            ],
            outputs,
        };

        let mut out = String::new();
        render(&mut out, &assignment).unwrap();

        assert!(
            out.contains("program_get(reduce_i_out0, &reduce_value_out0, &reduce_value_out1);")
        );
        assert!(out.contains(
            "program_add(out0, out1, reduce_value_out0, reduce_value_out1, &reduce_next_out0, &reduce_next_out1);"
        ));
    }
}
