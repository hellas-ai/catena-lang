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
    GpuAssign, GpuValue,
    components::{
        Component, input_components, is_runtime_value, runtime_values, single_function,
        single_value, value_expr,
    },
    gpu::GpuRenderError,
    render_utils::{c_type, invalid_outputs},
    runtime_type,
};

pub(in crate::codegen) fn render(
    out: &mut String,
    assignment: &GpuAssign,
) -> Result<(), GpuRenderError> {
    let (zero, add_env, add_fn, get_env, get_fn, n) = parts(assignment)?;
    if zero.len() != assignment.outputs.len() {
        return Err(GpuRenderError::InvalidInputComponentValueCount {
            op: assignment.op.clone(),
            component: "zero",
            description: "reducec zero input must match the accumulator arity",
            expected: assignment.outputs.len(),
            actual: zero.len(),
        });
    }
    if assignment.outputs.is_empty() {
        return Err(invalid_outputs(assignment, 1));
    }

    let i = format!("reduce_i_{}", assignment.outputs[0].name);
    let values = assignment
        .outputs
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

    // The producer closure is called with its runtime environment, the current
    // index, and an out-pointer for the element at that index.
    let mut get_args = runtime_args(get_env);
    get_args.push(i.clone());
    get_args.extend(
        values
            .iter()
            .map(|(_output, _ty, value, _next)| format!("&{value}")),
    );
    out.push_str(&format!(
        "            {}({});\n",
        value_expr(get_fn),
        get_args.join(", ")
    ));

    // The accumulator closure receives its runtime environment, the current
    // accumulator, the freshly produced element, and an out-pointer for `next`.
    let mut add_args = runtime_args(add_env);
    add_args.extend(
        values
            .iter()
            .map(|(output, _ty, _value, _next)| output.name.clone()),
    );
    add_args.extend(
        values
            .iter()
            .map(|(_output, _ty, value, _next)| value.clone()),
    );
    add_args.extend(
        values
            .iter()
            .map(|(_output, _ty, _value, next)| format!("&{next}")),
    );
    out.push_str(&format!(
        "            {}({});\n",
        value_expr(add_fn),
        add_args.join(", ")
    ));
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

fn runtime_args(values: Component<'_>) -> Vec<String> {
    runtime_values(values).map(value_expr).collect()
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
}
