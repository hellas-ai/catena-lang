//! `materializec` builds an owned buffer by evaluating a array-as-a-function producer at every index `0 <= i < len`.
//! It is lowered as allocation plus a kernel launch, not as a
//! device-callable expression. The host wrapper emits roughly:
//!
//! ```cpp
//! T *buf_data = nullptr;
//! catena_host_gpu_check(cudaMallocManaged((void **)&buf_data, len * sizeof(T)));
//! materialize_kernel<<<dim3((len + 255) / 256), dim3(256)>>>(buf_data, len, env...);
//! catena_host_gpu_check(cudaDeviceSynchronize());
//! buf = buf_data;
//! ```
//!
//! The generated kernel is device code and assumes the producer is already device-callable and
//! allocation-free:
//!
//! ```cpp
//! uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
//! if (i >= len) { return; }
//! T value;
//! program_producer(env..., i, &value);
//! out[i] = value;
//! ```
//!

use crate::codegen::{
    GpuAssign, GpuDialect, GpuFunction, GpuValue,
    gpu::GpuRenderError,
    lower_types::CType,
    render_utils::{c_type, invalid_outputs, param_decl, sanitize_ident},
    runtime_type,
};

pub(in crate::codegen) fn render_kernel(
    out: &mut String,
    kernel_name: &str,
    assignment: &GpuAssign,
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
    let (func, _len, env) = parts(assignment)?;

    out.push_str(&format!(
        "__global__ void {kernel_name}({} *out, uint64_t len",
        c_type(element)
    ));
    for arg in &env {
        if let GpuValue::Var(var) = arg
            && runtime_type(var).is_some()
        {
            out.push_str(", ");
            out.push_str(&param_decl(var, false)?);
        }
    }
    out.push_str(") {\n");
    out.push_str("    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;\n");
    out.push_str("    if (i >= len) { return; }\n");
    out.push_str(&format!("    {} value;\n", c_type(element)));
    out.push_str("    ");
    out.push_str(&value_expr(func));
    out.push('(');
    let mut call_args = env
        .iter()
        .filter_map(|arg| match arg {
            GpuValue::Var(var) if runtime_type(var).is_some() => Some(var.name.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    call_args.push("i".to_string());
    call_args.push("&value".to_string());
    out.push_str(&call_args.join(", "));
    out.push_str(");\n");
    out.push_str("    out[i] = value;\n");
    out.push_str("}\n");
    Ok(())
}

pub(in crate::codegen) fn render_call(
    out: &mut String,
    function: &GpuFunction,
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
    let (_func, len, env) = parts(assignment)?;
    let len = value_expr(len);
    let kernel_name = kernel_name(&function.name, assignment)?;

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
        "    catena_host_gpu_check({managed_alloc_fn}((void **)&{name}_data, {name}_len * sizeof({element})));\n",
        name = output.name,
        element = c_type(element),
        managed_alloc_fn = dialect.managed_alloc_fn(),
    ));
    out.push_str(&format!(
        "    {kernel_name}<<<dim3(({name}_len + 255) / 256), dim3(256)>>>\n",
        name = output.name
    ));
    out.push_str(&format!(
        "        ({name}_data, {name}_len",
        name = output.name
    ));
    for arg in env {
        if let GpuValue::Var(var) = arg
            && runtime_type(var).is_some()
        {
            out.push_str(", ");
            out.push_str(&var.name);
        }
    }
    out.push_str(");\n");
    out.push_str(&format!(
        "    catena_host_gpu_check({synchronize_fn}());\n",
        synchronize_fn = dialect.synchronize_fn()
    ));
    out.push_str(&format!("    {} = {}_data;\n", output.name, output.name));
    Ok(())
}

pub(in crate::codegen) fn kernel_name(
    function_name: &str,
    assignment: &GpuAssign,
) -> Result<String, GpuRenderError> {
    let [output] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 1));
    };
    Ok(format!("materialize_{}_{}", function_name, output.name))
}

fn parts(assignment: &GpuAssign) -> Result<(&GpuValue, &GpuValue, Vec<&GpuValue>), GpuRenderError> {
    // The lowered inputs follow the closure-converted call shape:
    //
    //     env..., producer_fn, erased_witnesses..., len
    //
    // Values before the function symbol are the producer environment. After the function symbol,
    // `materializec` may still carry erased parameter-level inputs, so the length is recovered as
    // the single runtime value in that suffix.
    let func_index = assignment
        .inputs
        .iter()
        .position(|input| matches!(input, GpuValue::FnSymbol(_)))
        .ok_or(GpuRenderError::MissingMaterializecFunction)?;
    let func = &assignment.inputs[func_index];
    let env = assignment.inputs[..func_index].iter().collect::<Vec<_>>();
    let trailing_runtime = assignment.inputs[func_index + 1..]
        .iter()
        .filter(|input| matches!(input, GpuValue::Var(var) if runtime_type(var).is_some()))
        .collect::<Vec<_>>();
    let [len] = trailing_runtime.as_slice() else {
        if trailing_runtime.is_empty() {
            return Err(GpuRenderError::MissingMaterializecLength);
        }
        return Err(GpuRenderError::InvalidMaterializecLength {
            actual: trailing_runtime.len(),
        });
    };
    Ok((func, len, env))
}

fn value_expr(value: &GpuValue) -> String {
    match value {
        GpuValue::Var(var) => var.name.clone(),
        GpuValue::FnSymbol(symbol) => sanitize_ident(&format!("program.{}", symbol.target)),
    }
}
