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
    // assume there are only two function indices for now
    // we don't allow function pointers in context
    // a cleaner solution requires a refactoring of lowering that preserves type info
    let func_indices = assignment
        .inputs
        .iter()
        .enumerate()
        .filter_map(|(index, input)| matches!(input, GpuValue::FnSymbol(_)).then_some(index))
        .collect::<Vec<_>>();
    let [add_index, get_index] = func_indices.as_slice() else {
        return Err(GpuRenderError::InvalidReducecFunctionCount {
            actual: func_indices.len(),
        });
    };
    if *add_index == 0 {
        return Err(GpuRenderError::MissingReducecZero);
    }

    // The zero value must be a runtime value because it initializes the emitted
    // accumulator variable directly.
    let zero = &assignment.inputs[0];
    if !is_runtime_value(zero) {
        return Err(GpuRenderError::ErasedReducecZero);
    }

    // Everything between zero and the add function is the add closure's
    // environment. Erased values can appear here; they are filtered at call
    // rendering time.
    let add_env = assignment.inputs[1..*add_index].iter().collect::<Vec<_>>();
    let add_fn = &assignment.inputs[*add_index];

    // Everything between the add function and producer function is the
    // producer closure's environment.
    let get_env = assignment.inputs[*add_index + 1..*get_index]
        .iter()
        .collect::<Vec<_>>();
    let get_fn = &assignment.inputs[*get_index];

    // After the producer function, only one runtime value should remain: the
    // reduction length. Type-level witnesses in this suffix are erased.
    let trailing_runtime = assignment.inputs[*get_index + 1..]
        .iter()
        .filter(|input| is_runtime_value(input))
        .collect::<Vec<_>>();
    let [n] = trailing_runtime.as_slice() else {
        return Err(GpuRenderError::InvalidReducecLengthCount {
            actual: trailing_runtime.len(),
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
