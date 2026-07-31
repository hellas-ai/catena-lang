//! `materializec` builds an owned buffer by evaluating an array-as-a-function
//! producer at every index `0 <= i < len`.
//!
//! It is an ordinary sequential materialization. The host function emits
//! roughly:
//!
//! ```cpp
//! T *buf_data = nullptr;
//! catena_host_gpu_check(cudaMallocManaged((void **)&buf_data, len * sizeof(T)));
//! for (uint64_t i = 0; i < len; ++i) {
//!     T value;
//!     program_producer(env..., i, &value);
//!     buf_data[i] = value;
//! }
//! ```

use crate::codegen::{
    GpuAssign, GpuDialect, GpuValue, GpuVar,
    components::{Component, input_components, single_function, single_value, value_expr},
    gpu::{GpuRenderError, render_function_application},
    lower_types::{CType, LoweredType},
    render_utils::{c_type, invalid_outputs},
    runtime_type,
};

pub(in crate::codegen) fn render_call(
    out: &mut String,
    assignment: &GpuAssign,
    dialect: GpuDialect,
) -> Result<(), GpuRenderError> {
    let [output] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 1));
    };
    let CType::Pointer(element) =
        runtime_type(output).ok_or_else(|| GpuRenderError::ErasedType(output.clone()))?
    else {
        return Err(GpuRenderError::UnsupportedType(
            runtime_type(output).unwrap().clone(),
        ));
    };
    let (func, len, env) = parts(assignment)?;
    let len = value_expr(len);

    out.push_str(&format!(
        "    uint64_t {name}_len = {len};\n",
        name = output.name
    ));
    out.push_str(&format!(
        "    {} *{name}_data = nullptr;\n",
        c_type(element),
        name = output.name
    ));
    out.push_str(&format!(
        "    if ({name}_len != 0) {{\n",
        name = output.name
    ));
    out.push_str(&format!(
        "        catena_host_gpu_check({managed_alloc_fn}((void **)&{name}_data, {name}_len * sizeof({element})));\n",
        name = output.name,
        element = c_type(element),
        managed_alloc_fn = dialect.managed_alloc_fn(),
    ));
    let index_name = format!("{}_i", output.name);
    let value_name = format!("{}_value", output.name);
    out.push_str(&format!(
        "        for (uint64_t {index_name} = 0; {index_name} < {name}_len; ++{index_name}) {{\n",
        name = output.name,
    ));
    out.push_str(&format!("            {} {value_name};\n", c_type(element)));
    let mut producer_inputs = env.to_vec();
    producer_inputs.push(GpuValue::Var(GpuVar {
        node: output.node,
        name: index_name,
        lowered: LoweredType::Runtime(CType::U64),
    }));
    let producer_output = GpuVar {
        node: output.node,
        name: value_name.clone(),
        lowered: LoweredType::Runtime(element.as_ref().clone()),
    };
    render_function_application(
        out,
        "            ",
        func,
        &producer_inputs,
        &[producer_output],
    )?;
    out.push_str(&format!(
        "            {name}_data[{name}_i] = {value_name};\n",
        name = output.name,
    ));
    out.push_str("        }\n");
    out.push_str("    }\n");
    out.push_str(&format!("    {} = {}_data;\n", output.name, output.name));
    Ok(())
}

fn parts(assignment: &GpuAssign) -> Result<(&GpuValue, &GpuValue, Component<'_>), GpuRenderError> {
    // The lowered inputs follow the closure-converted call shape:
    //
    //     len, env..., producer_fn
    //
    // `env` may be a zero-size component when the source environment is unit.
    let components = input_components(assignment)?;
    let [len, env, func] = components.as_slice() else {
        return Err(GpuRenderError::InvalidInputComponentCount {
            op: assignment.op.clone(),
            expected: 3,
            actual: components.len(),
        });
    };

    let func = single_function(func).map_err(|error| {
        invalid_component_count(
            assignment,
            "producer_fn",
            "materializec producer function symbol",
            error.actual,
        )
    })?;
    let len = single_value(len).map_err(|error| {
        invalid_component_count(
            assignment,
            "len",
            "materializec runtime length input",
            error.actual,
        )
    })?;
    Ok((func, len, env))
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
    use crate::codegen::fn_ptrs::FnPtrSymbol;
    use hexpr::Operation;
    use open_hypergraphs::lax::NodeId;

    fn op(name: &str) -> Operation {
        name.parse().unwrap()
    }

    #[test]
    fn primitive_producer_function_renders_inline() {
        let assignment = GpuAssign {
            op: op("materializec"),
            input_sizes: vec![1, 0, 1],
            output_sizes: vec![1],
            call_symbol: None,
            inputs: vec![
                GpuValue::Var(GpuVar {
                    node: NodeId(0),
                    name: "len".to_string(),
                    lowered: LoweredType::Runtime(CType::U64),
                }),
                GpuValue::FnSymbol(FnPtrSymbol {
                    target: op("ix.to-u64"),
                }),
            ],
            outputs: vec![GpuVar {
                node: NodeId(1),
                name: "out".to_string(),
                lowered: LoweredType::Runtime(CType::Pointer(Box::new(CType::U64))),
            }],
        };

        let mut out = String::new();
        render_call(&mut out, &assignment, GpuDialect::Hip).unwrap();

        assert!(out.contains("for (uint64_t out_i = 0; out_i < out_len; ++out_i)"));
        assert!(out.contains("out_value = out_i;"));
        assert!(!out.contains("program_ix_to_u64"));
        assert!(!out.contains("<<<"));
        assert!(!out.contains("DeviceSynchronize"));
    }
}
