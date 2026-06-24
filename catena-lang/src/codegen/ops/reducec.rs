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
    let components = input_components(assignment)?;
    let [zero, add_env, add_fn, get_env, get_fn, n] = components.as_slice() else {
        return Err(GpuRenderError::InvalidReducecSourceSizeCount {
            actual: assignment.source_sizes.len(),
        });
    };

    let [zero] = zero.as_slice() else {
        return Err(GpuRenderError::MissingReducecZero);
    };
    if !is_runtime_value(zero) {
        return Err(GpuRenderError::ErasedReducecZero);
    }

    let add_fn = single_function(add_fn)?;
    let get_fn = single_function(get_fn)?;

    let n_runtime = n
        .iter()
        .copied()
        .filter(|input| is_runtime_value(input))
        .collect::<Vec<_>>();
    let [n] = n_runtime.as_slice() else {
        return Err(GpuRenderError::InvalidReducecLengthCount {
            actual: n_runtime.len(),
        });
    };

    Ok((zero, add_env.clone(), add_fn, get_env.clone(), get_fn, n))
}

fn single_function<'a>(component: &[&'a GpuValue]) -> Result<&'a GpuValue, GpuRenderError> {
    let [function] = component else {
        return Err(GpuRenderError::InvalidReducecFunctionCount {
            actual: component
                .iter()
                .filter(|input| matches!(input, GpuValue::FnSymbol(_)))
                .count(),
        });
    };
    if !matches!(function, GpuValue::FnSymbol(_)) {
        return Err(GpuRenderError::InvalidReducecFunctionCount { actual: 0 });
    }
    Ok(function)
}

fn input_components<'a>(
    assignment: &'a GpuAssign,
) -> Result<Vec<Vec<&'a GpuValue>>, GpuRenderError> {
    let expected = assignment.source_sizes.iter().sum::<usize>();
    if expected != assignment.inputs.len() {
        return Err(GpuRenderError::InvalidReducecFlattenedInputCount {
            expected,
            actual: assignment.inputs.len(),
        });
    }

    let mut offset = 0;
    Ok(assignment
        .source_sizes
        .iter()
        .map(|size| {
            let end = offset + *size;
            let component = assignment.inputs[offset..end].iter().collect::<Vec<_>>();
            offset = end;
            component
        })
        .collect())
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
    fn source_sizes_group_flattened_reducec_environments() {
        let output = GpuVar {
            node: NodeId(9),
            name: "out".to_string(),
            lowered: LoweredType::Runtime(CType::U64),
        };
        let assignment = GpuAssign {
            op: op("reducec"),
            source_sizes: vec![1, 2, 1, 2, 1, 1],
            target_sizes: vec![1],
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
