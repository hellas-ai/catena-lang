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
    gpu::GpuRenderError,
    render_utils::{c_type, invalid_outputs, sanitize_ident},
    runtime_type,
};

pub(in crate::codegen) fn render(
    out: &mut String,
    assignment: &GpuAssign,
) -> Result<(), GpuRenderError> {
    let [output] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 1));
    };
    let element = runtime_type(output).ok_or_else(|| GpuRenderError::ErasedType(output.clone()))?;
    let (zero, add_env, add_fn, get_env, get_fn, n) = parts(assignment)?;

    let i = format!("reduce_i_{}", output.name);
    let value = format!("reduce_value_{}", output.name);
    let next = format!("reduce_next_{}", output.name);

    // Keep the accumulator in the output slot itself. This makes the final
    // assignment to the function's out-pointer use the usual renderer path.
    out.push_str("    {\n");
    out.push_str(&format!(
        "        {} = {};\n",
        output.name,
        value_expr(zero)
    ));
    out.push_str(&format!(
        "        for (uint64_t {i} = 0; {i} < {n}; ++{i}) {{\n",
        n = value_expr(n),
    ));
    out.push_str(&format!("            {} {value};\n", c_type(element)));
    out.push_str(&format!("            {} {next};\n", c_type(element)));

    // The producer closure is called with its runtime environment, the current
    // index, and an out-pointer for the element at that index.
    let mut get_args = runtime_args(get_env);
    get_args.push(i.clone());
    get_args.push(format!("&{value}"));
    out.push_str(&format!(
        "            {}({});\n",
        value_expr(get_fn),
        get_args.join(", ")
    ));

    // The accumulator closure receives its runtime environment, the current
    // accumulator, the freshly produced element, and an out-pointer for `next`.
    let mut add_args = runtime_args(add_env);
    add_args.push(output.name.clone());
    add_args.push(value);
    add_args.push(format!("&{next}"));
    out.push_str(&format!(
        "            {}({});\n",
        value_expr(add_fn),
        add_args.join(", ")
    ));
    out.push_str(&format!("            {} = {next};\n", output.name));
    out.push_str("        }\n");
    out.push_str("    }\n");
    Ok(())
}

type ReducecParts<'a> = (
    &'a GpuValue,
    Vec<&'a GpuValue>,
    &'a GpuValue,
    Vec<&'a GpuValue>,
    &'a GpuValue,
    &'a GpuValue,
);

fn parts(assignment: &GpuAssign) -> Result<ReducecParts<'_>, GpuRenderError> {
    let [
        zero_group,
        add_env_group,
        add_fn_group,
        get_env_group,
        get_fn_group,
        len_group,
    ] = assignment.source_groups()
    else {
        return Err(GpuRenderError::InvalidReducecSourceGroupCount {
            actual: assignment.source_groups().len(),
        });
    };

    let add_fn_group_len = assignment.group_values(add_fn_group).len();
    if !matches!(
        assignment.single_group_value(add_fn_group),
        Some(GpuValue::FnSymbol(_))
    ) {
        return Err(GpuRenderError::InvalidReducecFunctionGroup {
            source_index: add_fn_group.source_index,
            actual: add_fn_group_len,
        });
    };

    let get_fn_group_len = assignment.group_values(get_fn_group).len();
    if !matches!(
        assignment.single_group_value(get_fn_group),
        Some(GpuValue::FnSymbol(_))
    ) {
        return Err(GpuRenderError::InvalidReducecFunctionGroup {
            source_index: get_fn_group.source_index,
            actual: get_fn_group_len,
        });
    };

    // The zero value must be a runtime value because it initializes the emitted
    // accumulator variable directly.
    let zero = assignment
        .single_group_value(zero_group)
        .ok_or(GpuRenderError::MissingReducecZero)?;
    if !is_runtime_value(zero) {
        return Err(GpuRenderError::ErasedReducecZero);
    }

    let add_env = assignment.group_values(add_env_group).iter().collect();
    let add_fn = assignment
        .single_group_value(add_fn_group)
        .expect("validated add function group");

    let get_env = assignment.group_values(get_env_group).iter().collect();
    let get_fn = assignment
        .single_group_value(get_fn_group)
        .expect("validated get function group");

    let len_values = assignment.group_values(len_group);
    let runtime_lengths = len_values
        .iter()
        .filter(|input| is_runtime_value(input))
        .collect::<Vec<_>>();
    let [n] = runtime_lengths.as_slice() else {
        return Err(GpuRenderError::InvalidReducecLengthCount {
            actual: runtime_lengths.len(),
        });
    };

    Ok((zero, add_env, add_fn, get_env, get_fn, n))
}

fn runtime_args(values: Vec<&GpuValue>) -> Vec<String> {
    values
        .into_iter()
        .filter(|value| is_runtime_value(value))
        .map(value_expr)
        .collect()
}

fn is_runtime_value(value: &GpuValue) -> bool {
    matches!(value, GpuValue::Var(var) if runtime_type(var).is_some())
}

fn value_expr(value: &GpuValue) -> String {
    match value {
        GpuValue::Var(var) => var.name.clone(),
        GpuValue::FnSymbol(symbol) => sanitize_ident(&format!("program.{}", symbol.target)),
    }
}
