//! `materializec` builds an owned buffer by evaluating a array-as-a-function producer at every index `0 <= i < len`.
//! It is lowered as allocation plus a kernel launch, not as a
//! device-callable expression. The host wrapper emits roughly:
//!
//! ```cpp
//! T *buf_data = nullptr;
//! catena_host_gpu_check(cudaMallocAsync((void **)&buf_data, len * sizeof(T), nullptr));
//! materialize_kernel<<<dim3((len + 255) / 256), dim3(256)>>>(buf_data, len, env...);
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

use crate::codegen::default::{
    GpuAssign, GpuDialect, GpuFunction, GpuValue, GpuVar,
    components::{
        Component, input_components, runtime_values, single_function, single_value, value_expr,
    },
    gpu::{GpuRenderError, render_function_application},
    lower_types::{CType, LoweredType},
    ops::gpu_support::{hip_gfx11_guard, hip_wmma_include_guard},
    ops::projection::ProjectionBackend,
    render_utils::{c_type, invalid_outputs, param_decl},
    runtime_type,
};

const ROUTED_MAX_EXPERTS: u64 = 256;
const ROUTED_MIN_ROUTES_PER_EXPERT: u64 = 4;

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
    for arg in runtime_values(env) {
        if let GpuValue::Var(var) = arg {
            out.push_str(", ");
            out.push_str(&param_decl(var, false)?);
        }
    }
    out.push_str(") {\n");
    out.push_str("    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;\n");
    out.push_str("    if (i >= len) { return; }\n");
    out.push_str(&format!("    {} value;\n", c_type(element)));
    let mut producer_inputs = env.to_vec();
    producer_inputs.push(GpuValue::Var(GpuVar {
        node: output.node,
        name: "i".to_string(),
        lowered: LoweredType::Runtime(CType::U64),
    }));
    let producer_output = GpuVar {
        node: output.node,
        name: "value".to_string(),
        lowered: LoweredType::Runtime(element.as_ref().clone()),
    };
    render_function_application(out, "    ", func, &producer_inputs, &[producer_output])?;
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
        "    if ({name}_len != 0) {{\n",
        name = output.name
    ));
    out.push_str(&format!(
        "        catena_host_gpu_check({device_alloc_async_fn}((void **)&{name}_data, {name}_len * sizeof({element}), nullptr));\n",
        name = output.name,
        element = c_type(element),
        device_alloc_async_fn = dialect.device_alloc_async_fn(),
    ));
    out.push_str(&format!(
        "        {kernel_name}<<<dim3(({name}_len + 255) / 256), dim3(256)>>>\n",
        name = output.name
    ));
    out.push_str(&format!(
        "            ({name}_data, {name}_len",
        name = output.name
    ));
    for arg in runtime_values(env) {
        if let GpuValue::Var(var) = arg {
            out.push_str(", ");
            out.push_str(&var.name);
        }
    }
    out.push_str(");\n");
    out.push_str("    }\n");
    out.push_str(&format!("    {} = {}_data;\n", output.name, output.name));
    Ok(())
}

pub(in crate::codegen) fn render_into_kernel(
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
    let (func, _buffer, _capacity, _offset, _len, env) = into_parts(assignment)?;

    out.push_str(&format!(
        "__global__ void {kernel_name}({} *out, uint64_t offset, uint64_t len",
        c_type(element)
    ));
    for arg in runtime_values(env) {
        if let GpuValue::Var(var) = arg {
            out.push_str(", ");
            out.push_str(&param_decl(var, false)?);
        }
    }
    out.push_str(") {\n");
    out.push_str("    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;\n");
    out.push_str("    if (i >= len) { return; }\n");
    out.push_str(&format!("    {} value;\n", c_type(element)));
    let mut producer_inputs = env.to_vec();
    producer_inputs.push(GpuValue::Var(GpuVar {
        node: output.node,
        name: "i".to_string(),
        lowered: LoweredType::Runtime(CType::U64),
    }));
    let producer_output = GpuVar {
        node: output.node,
        name: "value".to_string(),
        lowered: LoweredType::Runtime(element.as_ref().clone()),
    };
    render_function_application(out, "    ", func, &producer_inputs, &[producer_output])?;
    out.push_str("    out[offset + i] = value;\n");
    out.push_str("}\n");
    Ok(())
}

pub(in crate::codegen) fn render_into_call(
    out: &mut String,
    function: &GpuFunction,
    assignment: &GpuAssign,
    _dialect: GpuDialect,
) -> Result<(), GpuRenderError> {
    let [output] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 1));
    };
    let (_func, buffer, capacity, offset, len, env) = into_parts(assignment)?;
    let capacity = value_expr(capacity);
    let offset = value_expr(offset);
    let len = value_expr(len);
    let kernel_name = kernel_name(&function.name, assignment)?;

    out.push_str(&format!(
        "    catena_assert(({offset}) <= ({capacity}) && ({len}) <= ({capacity}) - ({offset}));\n"
    ));
    out.push_str(&format!("    if (({len}) != 0) {{\n"));
    out.push_str(&format!(
        "        {kernel_name}<<<dim3((({len}) + 255) / 256), dim3(256)>>>(\n"
    ));
    out.push_str(&format!(
        "            {}, {offset}, {len}",
        value_expr(buffer)
    ));
    for arg in runtime_values(env) {
        if let GpuValue::Var(var) = arg {
            out.push_str(", ");
            out.push_str(&var.name);
        }
    }
    out.push_str(");\n");
    out.push_str("    }\n");
    out.push_str(&format!("    {} = {};\n", output.name, value_expr(buffer)));
    Ok(())
}

pub(in crate::codegen) fn render_borrow_kernel(
    out: &mut String,
    kernel_name: &str,
    assignment: &GpuAssign,
) -> Result<(), GpuRenderError> {
    let [cache_output, output] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 2));
    };
    let CType::Pointer(cache_element) = runtime_type(cache_output)
        .ok_or_else(|| GpuRenderError::ErasedType(cache_output.clone()))?
    else {
        return Err(GpuRenderError::UnsupportedType(
            runtime_type(cache_output).unwrap().clone(),
        ));
    };
    let CType::Pointer(element) =
        runtime_type(output).ok_or_else(|| GpuRenderError::ErasedType(output.clone()))?
    else {
        return Err(GpuRenderError::UnsupportedType(
            runtime_type(output).unwrap().clone(),
        ));
    };
    let (func, _cache, _capacity, _len, env) = borrow_parts(assignment)?;

    out.push_str(&format!(
        "__global__ void {kernel_name}({} *borrowed, {} *out, uint64_t len",
        c_type(cache_element),
        c_type(element)
    ));
    for arg in runtime_values(env) {
        if let GpuValue::Var(var) = arg {
            out.push_str(", ");
            out.push_str(&param_decl(var, false)?);
        }
    }
    out.push_str(") {\n");
    out.push_str("    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;\n");
    out.push_str("    if (i >= len) { return; }\n");
    out.push_str(&format!("    {} value;\n", c_type(element)));
    let mut producer_inputs = vec![GpuValue::Var(GpuVar {
        node: cache_output.node,
        name: "borrowed".to_string(),
        lowered: LoweredType::Runtime(CType::Pointer(cache_element.clone())),
    })];
    producer_inputs.extend_from_slice(env);
    producer_inputs.push(GpuValue::Var(GpuVar {
        node: output.node,
        name: "i".to_string(),
        lowered: LoweredType::Runtime(CType::U64),
    }));
    let producer_output = GpuVar {
        node: output.node,
        name: "value".to_string(),
        lowered: LoweredType::Runtime(element.as_ref().clone()),
    };
    render_function_application(out, "    ", func, &producer_inputs, &[producer_output])?;
    out.push_str("    out[i] = value;\n");
    out.push_str("}\n");
    Ok(())
}

pub(in crate::codegen) fn render_borrow_call(
    out: &mut String,
    function: &GpuFunction,
    assignment: &GpuAssign,
    dialect: GpuDialect,
) -> Result<(), GpuRenderError> {
    let [cache_output, output] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 2));
    };
    let CType::Pointer(element) =
        runtime_type(output).ok_or_else(|| GpuRenderError::ErasedType(output.clone()))?
    else {
        return Err(GpuRenderError::UnsupportedType(
            runtime_type(output).unwrap().clone(),
        ));
    };
    let (_func, cache, _capacity, len, env) = borrow_parts(assignment)?;
    let len = value_expr(len);
    let kernel_name = kernel_name(&function.name, assignment)?;

    out.push_str(&format!("    uint64_t {}_len = {len};\n", output.name));
    out.push_str(&format!(
        "    {} *{}_data = nullptr;\n",
        c_type(element),
        output.name
    ));
    out.push_str(&format!("    if ({}_len != 0) {{\n", output.name));
    out.push_str(&format!(
        "        catena_host_gpu_check({}((void **)&{}_data, {}_len * sizeof({}), nullptr));\n",
        dialect.device_alloc_async_fn(),
        output.name,
        output.name,
        c_type(element)
    ));
    out.push_str(&format!(
        "        {kernel_name}<<<dim3(({}_len + 255) / 256), dim3(256)>>>(\n",
        output.name
    ));
    out.push_str(&format!(
        "            {}, {}_data, {}_len",
        value_expr(cache),
        output.name,
        output.name
    ));
    for arg in runtime_values(env) {
        if let GpuValue::Var(var) = arg {
            out.push_str(", ");
            out.push_str(&var.name);
        }
    }
    out.push_str(");\n");
    out.push_str("    }\n");
    out.push_str(&format!(
        "    {} = {};\n    {} = {}_data;\n",
        cache_output.name,
        value_expr(cache),
        output.name,
        output.name
    ));
    Ok(())
}

pub(in crate::codegen) fn render_reduce_f32_kernel(
    out: &mut String,
    kernel_name: &str,
    assignment: &GpuAssign,
) -> Result<(), GpuRenderError> {
    let [output] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 1));
    };
    let Some(CType::Pointer(element)) = runtime_type(output) else {
        return Err(GpuRenderError::ErasedType(output.clone()));
    };
    if !matches!(element.as_ref(), CType::F32 | CType::BF16) {
        return Err(GpuRenderError::UnsupportedType(
            runtime_type(output).unwrap().clone(),
        ));
    }
    let (term, epilogue, _output_len, _reduction_len, env) = reduce_f32_parts(assignment)?;

    out.push_str(&format!(
        "__global__ void {kernel_name}({} *out, uint64_t output_len, uint64_t reduction_len",
        c_type(element)
    ));
    for arg in runtime_values(env) {
        if let GpuValue::Var(var) = arg {
            out.push_str(", ");
            out.push_str(&param_decl(var, false)?);
        }
    }
    out.push_str(") {\n");
    out.push_str("    uint64_t output_index = (uint64_t)blockIdx.x;\n");
    out.push_str("    if (output_index >= output_len) { return; }\n");
    out.push_str("    extern __shared__ float reduction_terms[];\n");
    out.push_str(
        "    for (uint64_t reduction_index = (uint64_t)threadIdx.x; reduction_index < reduction_len; reduction_index += (uint64_t)blockDim.x) {\n",
    );
    out.push_str("        float term;\n");
    let output_index = GpuValue::Var(GpuVar {
        node: output.node,
        name: "output_index".to_string(),
        lowered: LoweredType::Runtime(CType::U64),
    });
    let reduction_index = GpuValue::Var(GpuVar {
        node: output.node,
        name: "reduction_index".to_string(),
        lowered: LoweredType::Runtime(CType::U64),
    });
    let mut term_inputs = env.to_vec();
    term_inputs.push(output_index.clone());
    term_inputs.push(reduction_index);
    let term_output = GpuVar {
        node: output.node,
        name: "term".to_string(),
        lowered: LoweredType::Runtime(CType::F32),
    };
    render_function_application(out, "        ", term, &term_inputs, &[term_output])?;
    out.push_str("        reduction_terms[reduction_index] = term;\n");
    out.push_str("    }\n");
    out.push_str("    __syncthreads();\n");
    out.push_str("    for (uint64_t stride = 1; stride < reduction_len; stride <<= 1) {\n");
    out.push_str("        uint64_t step = stride << 1;\n");
    out.push_str(
        "        for (uint64_t left = (uint64_t)threadIdx.x * step; left + stride < reduction_len; left += (uint64_t)blockDim.x * step) {\n",
    );
    out.push_str("            float left_value = reduction_terms[left];\n");
    out.push_str("            float right_value = reduction_terms[left + stride];\n");
    out.push_str("            reduction_terms[left] = left_value + right_value;\n");
    out.push_str("        }\n");
    out.push_str("        if (stride < 256) { __syncthreads(); } else { __syncwarp(); }\n");
    out.push_str("    }\n");
    out.push_str("    if (threadIdx.x == 0) {\n");
    out.push_str("        float reduction_sum = reduction_len == 0 ? 0.0f : reduction_terms[0];\n");
    out.push_str(&format!("        {} value;\n", c_type(element)));
    let mut epilogue_inputs = env.to_vec();
    epilogue_inputs.push(output_index);
    epilogue_inputs.push(GpuValue::Var(GpuVar {
        node: output.node,
        name: "reduction_sum".to_string(),
        lowered: LoweredType::Runtime(CType::F32),
    }));
    let epilogue_output = GpuVar {
        node: output.node,
        name: "value".to_string(),
        lowered: LoweredType::Runtime(element.as_ref().clone()),
    };
    render_function_application(
        out,
        "        ",
        epilogue,
        &epilogue_inputs,
        &[epilogue_output],
    )?;
    out.push_str("        out[output_index] = value;\n");
    out.push_str("    }\n");
    out.push_str("}\n");
    Ok(())
}

pub(in crate::codegen) fn render_reduce_f32_pair_kernel(
    out: &mut String,
    kernel_name: &str,
    assignment: &GpuAssign,
) -> Result<(), GpuRenderError> {
    let [left_output, right_output] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 2));
    };
    for output in [left_output, right_output] {
        let Some(CType::Pointer(element)) = runtime_type(output) else {
            return Err(GpuRenderError::ErasedType(output.clone()));
        };
        if element.as_ref() != &CType::F32 {
            return Err(GpuRenderError::UnsupportedType(
                runtime_type(output).unwrap().clone(),
            ));
        }
    }
    let (term, epilogue, _, _, env) = reduce_f32_pair_parts(assignment)?;

    out.push_str(&format!(
        "__global__ void {kernel_name}(float *left_out, float *right_out, uint64_t output_len, uint64_t reduction_len"
    ));
    for arg in runtime_values(env) {
        if let GpuValue::Var(var) = arg {
            out.push_str(", ");
            out.push_str(&param_decl(var, false)?);
        }
    }
    out.push_str(") {\n");
    out.push_str("    uint64_t output_index = (uint64_t)blockIdx.x;\n");
    out.push_str("    if (output_index >= output_len) { return; }\n");
    out.push_str("    extern __shared__ float reduction_terms[];\n");
    out.push_str("    float *left_terms = reduction_terms;\n");
    out.push_str("    float *right_terms = reduction_terms + reduction_len;\n");
    out.push_str("    for (uint64_t reduction_index = (uint64_t)threadIdx.x; reduction_index < reduction_len; reduction_index += (uint64_t)blockDim.x) {\n");
    out.push_str("        float left_term;\n");
    out.push_str("        float right_term;\n");
    let output_index = GpuValue::Var(GpuVar {
        node: left_output.node,
        name: "output_index".to_string(),
        lowered: LoweredType::Runtime(CType::U64),
    });
    let reduction_index = GpuValue::Var(GpuVar {
        node: left_output.node,
        name: "reduction_index".to_string(),
        lowered: LoweredType::Runtime(CType::U64),
    });
    let mut term_inputs = env.to_vec();
    term_inputs.push(output_index.clone());
    term_inputs.push(reduction_index);
    render_function_application(
        out,
        "        ",
        term,
        &term_inputs,
        &[
            GpuVar {
                node: left_output.node,
                name: "left_term".to_string(),
                lowered: LoweredType::Runtime(CType::F32),
            },
            GpuVar {
                node: right_output.node,
                name: "right_term".to_string(),
                lowered: LoweredType::Runtime(CType::F32),
            },
        ],
    )?;
    out.push_str("        left_terms[reduction_index] = left_term;\n");
    out.push_str("        right_terms[reduction_index] = right_term;\n");
    out.push_str("    }\n");
    out.push_str("    __syncthreads();\n");
    out.push_str("    for (uint64_t stride = 1; stride < reduction_len; stride <<= 1) {\n");
    out.push_str("        uint64_t step = stride << 1;\n");
    out.push_str("        for (uint64_t left = (uint64_t)threadIdx.x * step; left + stride < reduction_len; left += (uint64_t)blockDim.x * step) {\n");
    out.push_str("            left_terms[left] = left_terms[left] + left_terms[left + stride];\n");
    out.push_str(
        "            right_terms[left] = right_terms[left] + right_terms[left + stride];\n",
    );
    out.push_str("        }\n");
    out.push_str("        if (stride < 256) { __syncthreads(); } else { __syncwarp(); }\n");
    out.push_str("    }\n");
    out.push_str("    if (threadIdx.x == 0) {\n");
    out.push_str("        float left_sum = reduction_len == 0 ? 0.0f : left_terms[0];\n");
    out.push_str("        float right_sum = reduction_len == 0 ? 0.0f : right_terms[0];\n");
    out.push_str("        float left_value;\n");
    out.push_str("        float right_value;\n");
    let mut epilogue_inputs = env.to_vec();
    epilogue_inputs.push(output_index);
    epilogue_inputs.push(GpuValue::Var(GpuVar {
        node: left_output.node,
        name: "left_sum".to_string(),
        lowered: LoweredType::Runtime(CType::F32),
    }));
    epilogue_inputs.push(GpuValue::Var(GpuVar {
        node: right_output.node,
        name: "right_sum".to_string(),
        lowered: LoweredType::Runtime(CType::F32),
    }));
    render_function_application(
        out,
        "        ",
        epilogue,
        &epilogue_inputs,
        &[
            GpuVar {
                node: left_output.node,
                name: "left_value".to_string(),
                lowered: LoweredType::Runtime(CType::F32),
            },
            GpuVar {
                node: right_output.node,
                name: "right_value".to_string(),
                lowered: LoweredType::Runtime(CType::F32),
            },
        ],
    )?;
    out.push_str("        left_out[output_index] = left_value;\n");
    out.push_str("        right_out[output_index] = right_value;\n");
    out.push_str("    }\n");
    out.push_str("}\n");
    Ok(())
}

pub(in crate::codegen) fn render_borrow_reduce_f32_kernel(
    out: &mut String,
    kernel_name: &str,
    assignment: &GpuAssign,
) -> Result<(), GpuRenderError> {
    let [cache_output, output] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 2));
    };
    let CType::Pointer(cache_element) = runtime_type(cache_output)
        .ok_or_else(|| GpuRenderError::ErasedType(cache_output.clone()))?
    else {
        return Err(GpuRenderError::UnsupportedType(
            runtime_type(cache_output).unwrap().clone(),
        ));
    };
    let Some(CType::Pointer(output_element)) = runtime_type(output) else {
        return Err(GpuRenderError::ErasedType(output.clone()));
    };
    if !matches!(output_element.as_ref(), CType::F32 | CType::BF16) {
        return Err(GpuRenderError::UnsupportedType(
            runtime_type(output).unwrap().clone(),
        ));
    }
    let (term, epilogue, _cache, _capacity, _output_len, _reduction_len, env) =
        borrow_reduce_f32_parts(assignment)?;

    out.push_str(&format!(
        "__global__ void {kernel_name}({} *borrowed, {} *out, uint64_t output_len, uint64_t reduction_len",
        c_type(cache_element),
        c_type(output_element)
    ));
    for arg in runtime_values(env) {
        if let GpuValue::Var(var) = arg {
            out.push_str(", ");
            out.push_str(&param_decl(var, false)?);
        }
    }
    out.push_str(") {\n");
    out.push_str("    uint64_t output_index = (uint64_t)blockIdx.x;\n");
    out.push_str("    if (output_index >= output_len) { return; }\n");
    out.push_str("    extern __shared__ float reduction_terms[];\n");
    out.push_str(
        "    for (uint64_t reduction_index = (uint64_t)threadIdx.x; reduction_index < reduction_len; reduction_index += (uint64_t)blockDim.x) {\n",
    );
    out.push_str("        float term;\n");
    let borrowed = GpuValue::Var(GpuVar {
        node: cache_output.node,
        name: "borrowed".to_string(),
        lowered: LoweredType::Runtime(CType::Pointer(cache_element.clone())),
    });
    let output_index = GpuValue::Var(GpuVar {
        node: output.node,
        name: "output_index".to_string(),
        lowered: LoweredType::Runtime(CType::U64),
    });
    let reduction_index = GpuValue::Var(GpuVar {
        node: output.node,
        name: "reduction_index".to_string(),
        lowered: LoweredType::Runtime(CType::U64),
    });
    let mut term_inputs = vec![borrowed.clone()];
    term_inputs.extend_from_slice(env);
    term_inputs.push(output_index.clone());
    term_inputs.push(reduction_index);
    let term_output = GpuVar {
        node: output.node,
        name: "term".to_string(),
        lowered: LoweredType::Runtime(CType::F32),
    };
    render_function_application(out, "        ", term, &term_inputs, &[term_output])?;
    out.push_str("        reduction_terms[reduction_index] = term;\n");
    out.push_str("    }\n");
    out.push_str("    __syncthreads();\n");
    out.push_str("    for (uint64_t stride = 1; stride < reduction_len; stride <<= 1) {\n");
    out.push_str("        uint64_t step = stride << 1;\n");
    out.push_str(
        "        for (uint64_t left = (uint64_t)threadIdx.x * step; left + stride < reduction_len; left += (uint64_t)blockDim.x * step) {\n",
    );
    out.push_str(
        "            reduction_terms[left] = reduction_terms[left] + reduction_terms[left + stride];\n",
    );
    out.push_str("        }\n");
    out.push_str("        if (stride < 256) { __syncthreads(); } else { __syncwarp(); }\n");
    out.push_str("    }\n");
    out.push_str("    if (threadIdx.x == 0) {\n");
    out.push_str("        float reduction_sum = reduction_len == 0 ? 0.0f : reduction_terms[0];\n");
    out.push_str(&format!("        {} value;\n", c_type(output_element)));
    let mut epilogue_inputs = vec![borrowed];
    epilogue_inputs.extend_from_slice(env);
    epilogue_inputs.push(output_index);
    epilogue_inputs.push(GpuValue::Var(GpuVar {
        node: output.node,
        name: "reduction_sum".to_string(),
        lowered: LoweredType::Runtime(CType::F32),
    }));
    let epilogue_output = GpuVar {
        node: output.node,
        name: "value".to_string(),
        lowered: LoweredType::Runtime(output_element.as_ref().clone()),
    };
    render_function_application(
        out,
        "        ",
        epilogue,
        &epilogue_inputs,
        &[epilogue_output],
    )?;
    out.push_str("        out[output_index] = value;\n");
    out.push_str("    }\n");
    out.push_str("}\n");
    Ok(())
}

pub(in crate::codegen) fn render_borrow_routed_bf16_gemv_pair_kernel(
    out: &mut String,
    kernel_name: &str,
    assignment: &GpuAssign,
) -> Result<(), GpuRenderError> {
    let [input_output, selected_output, gate_output, up_output] = assignment.outputs.as_slice()
    else {
        return Err(invalid_outputs(assignment, 4));
    };
    for (output, expected) in [
        (input_output, CType::F32),
        (selected_output, CType::U64),
        (gate_output, CType::F32),
        (up_output, CType::F32),
    ] {
        let Some(CType::Pointer(element)) = runtime_type(output) else {
            return Err(GpuRenderError::ErasedType(output.clone()));
        };
        if element.as_ref() != &expected {
            return Err(GpuRenderError::UnsupportedType(
                runtime_type(output).unwrap().clone(),
            ));
        }
    }
    let _ = borrow_routed_bf16_gemv_pair_parts(assignment)?;

    out.push_str(&format!(
        "__global__ void {kernel_name}(const float *input, const catena_bf16_t *gate_weight, const catena_bf16_t *up_weight, const uint64_t *selected, float *gate_out, float *up_out, uint64_t output_len, uint64_t reduction_len, uint64_t slots, uint64_t output_features, uint64_t expert_count) {{\n"
    ));
    out.push_str("    uint64_t output_index = (uint64_t)blockIdx.x;\n");
    out.push_str("    if (output_index >= output_len) { return; }\n");
    out.push_str("    uint64_t active_width = slots * output_features;\n");
    out.push_str("    uint64_t row = output_index / active_width;\n");
    out.push_str("    uint64_t row_element = output_index % active_width;\n");
    out.push_str("    uint64_t slot = row_element / output_features;\n");
    out.push_str("    uint64_t output_feature = row_element % output_features;\n");
    out.push_str("    uint64_t expert = selected[row * slots + slot];\n");
    out.push_str("    catena_assert(expert < expert_count);\n");
    out.push_str("    uint64_t input_base = row * reduction_len;\n");
    out.push_str(
        "    uint64_t weight_base = (expert * output_features + output_feature) * reduction_len;\n",
    );
    out.push_str("    extern __shared__ float reduction_terms[];\n");
    out.push_str("    float *gate_terms = reduction_terms;\n");
    out.push_str("    float *up_terms = reduction_terms + reduction_len;\n");
    out.push_str("    for (uint64_t reduction_index = (uint64_t)threadIdx.x; reduction_index < reduction_len; reduction_index += (uint64_t)blockDim.x) {\n");
    out.push_str("        float input_value = input[input_base + reduction_index];\n");
    out.push_str("        gate_terms[reduction_index] = input_value * catena_bf16_to_f32(gate_weight[weight_base + reduction_index]);\n");
    out.push_str("        up_terms[reduction_index] = input_value * catena_bf16_to_f32(up_weight[weight_base + reduction_index]);\n");
    out.push_str("    }\n");
    out.push_str("    __syncthreads();\n");
    out.push_str("    for (uint64_t stride = 1; stride < reduction_len; stride <<= 1) {\n");
    out.push_str("        uint64_t step = stride << 1;\n");
    out.push_str("        for (uint64_t left = (uint64_t)threadIdx.x * step; left + stride < reduction_len; left += (uint64_t)blockDim.x * step) {\n");
    out.push_str("            gate_terms[left] = gate_terms[left] + gate_terms[left + stride];\n");
    out.push_str("            up_terms[left] = up_terms[left] + up_terms[left + stride];\n");
    out.push_str("        }\n");
    out.push_str("        if (stride < 256) { __syncthreads(); } else { __syncwarp(); }\n");
    out.push_str("    }\n");
    out.push_str("    if (threadIdx.x == 0) {\n");
    out.push_str("        gate_out[output_index] = reduction_len == 0 ? 0.0f : gate_terms[0];\n");
    out.push_str("        up_out[output_index] = reduction_len == 0 ? 0.0f : up_terms[0];\n");
    out.push_str("    }\n");
    out.push_str("}\n");
    Ok(())
}

pub(in crate::codegen) fn render_borrow_native_bf16_gemv_pair_kernel(
    out: &mut String,
    kernel_name: &str,
    assignment: &GpuAssign,
    dialect: GpuDialect,
) -> Result<(), GpuRenderError> {
    let [input_output, gate_output, up_output] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 3));
    };
    for output in [input_output, gate_output, up_output] {
        let Some(CType::Pointer(element)) = runtime_type(output) else {
            return Err(GpuRenderError::ErasedType(output.clone()));
        };
        if element.as_ref() != &CType::BF16 {
            return Err(GpuRenderError::UnsupportedType(
                runtime_type(output).unwrap().clone(),
            ));
        }
    }
    let _ = borrow_native_bf16_gemv_pair_parts(assignment)?;

    out.push_str(&format!(
        "__global__ void {kernel_name}(const catena_bf16_t *input, const catena_bf16_t *gate_weight, const catena_bf16_t *up_weight, catena_bf16_t *gate_out, catena_bf16_t *up_out, uint64_t output_len, uint64_t reduction_len, uint64_t input_row, uint64_t output_features) {{\n"
    ));
    out.push_str("    uint64_t output_index = (uint64_t)blockIdx.x;\n");
    out.push_str("    if (output_index >= output_len) { return; }\n");
    out.push_str("    uint64_t row = input_row + output_index / output_features;\n");
    out.push_str("    uint64_t output_feature = output_index % output_features;\n");
    out.push_str("    uint64_t input_base = row * reduction_len;\n");
    out.push_str("    uint64_t weight_base = output_feature * reduction_len;\n");
    out.push_str("    float gate_sum = 0.0f;\n    float up_sum = 0.0f;\n");
    out.push_str("    uint64_t pair_count = reduction_len >> 1;\n");
    if dialect == GpuDialect::Hip {
        out.push_str(&hip_gfx11_guard());
        out.push_str(
            "    typedef __bf16 catena_native_bf16x2_t __attribute__((ext_vector_type(2)));\n",
        );
        out.push_str("#endif\n");
    }
    out.push_str("    for (uint64_t pair = (uint64_t)threadIdx.x; pair < pair_count; pair += (uint64_t)blockDim.x) {\n");
    match dialect {
        GpuDialect::Hip => {
            out.push_str(&hip_gfx11_guard());
            out.push_str("        union { uint32_t bits; catena_native_bf16x2_t values; } input_pair, gate_pair, up_pair;\n");
            out.push_str("        __builtin_memcpy(&input_pair.bits, input + input_base + (pair << 1), sizeof(uint32_t));\n");
            out.push_str("        __builtin_memcpy(&gate_pair.bits, gate_weight + weight_base + (pair << 1), sizeof(uint32_t));\n");
            out.push_str("        __builtin_memcpy(&up_pair.bits, up_weight + weight_base + (pair << 1), sizeof(uint32_t));\n");
            out.push_str("        gate_sum = __builtin_amdgcn_fdot2_f32_bf16(input_pair.values, gate_pair.values, gate_sum, false);\n");
            out.push_str("        up_sum = __builtin_amdgcn_fdot2_f32_bf16(input_pair.values, up_pair.values, up_sum, false);\n");
            out.push_str("#else\n");
            out.push_str("        uint64_t index = pair << 1;\n");
            out.push_str("        float x0 = catena_bf16_to_f32(input[input_base + index]);\n");
            out.push_str("        float x1 = catena_bf16_to_f32(input[input_base + index + 1]);\n");
            out.push_str("        gate_sum += x0 * catena_bf16_to_f32(gate_weight[weight_base + index]) + x1 * catena_bf16_to_f32(gate_weight[weight_base + index + 1]);\n");
            out.push_str("        up_sum += x0 * catena_bf16_to_f32(up_weight[weight_base + index]) + x1 * catena_bf16_to_f32(up_weight[weight_base + index + 1]);\n");
            out.push_str("#endif\n");
        }
        GpuDialect::Cuda => {
            out.push_str("        uint64_t index = pair << 1;\n");
            out.push_str("        float x0 = catena_bf16_to_f32(input[input_base + index]);\n");
            out.push_str("        float x1 = catena_bf16_to_f32(input[input_base + index + 1]);\n");
            out.push_str("        gate_sum += x0 * catena_bf16_to_f32(gate_weight[weight_base + index]) + x1 * catena_bf16_to_f32(gate_weight[weight_base + index + 1]);\n");
            out.push_str("        up_sum += x0 * catena_bf16_to_f32(up_weight[weight_base + index]) + x1 * catena_bf16_to_f32(up_weight[weight_base + index + 1]);\n");
        }
    }
    out.push_str("    }\n");
    out.push_str("    if ((reduction_len & 1) != 0 && threadIdx.x == 0) {\n");
    out.push_str("        uint64_t index = reduction_len - 1;\n");
    out.push_str("        float x = catena_bf16_to_f32(input[input_base + index]);\n");
    out.push_str("        gate_sum += x * catena_bf16_to_f32(gate_weight[weight_base + index]);\n");
    out.push_str("        up_sum += x * catena_bf16_to_f32(up_weight[weight_base + index]);\n");
    out.push_str("    }\n");
    out.push_str(
        "    __shared__ float gate_partial[256];\n    __shared__ float up_partial[256];\n",
    );
    out.push_str("    gate_partial[threadIdx.x] = gate_sum;\n    up_partial[threadIdx.x] = up_sum;\n    __syncthreads();\n");
    out.push_str("    for (uint32_t stride = 128; stride != 0; stride >>= 1) {\n");
    out.push_str("        if (threadIdx.x < stride) { gate_partial[threadIdx.x] += gate_partial[threadIdx.x + stride]; up_partial[threadIdx.x] += up_partial[threadIdx.x + stride]; }\n");
    out.push_str("        __syncthreads();\n    }\n");
    out.push_str("    if (threadIdx.x == 0) { gate_out[output_index] = catena_bf16_from_f32(gate_partial[0]); up_out[output_index] = catena_bf16_from_f32(up_partial[0]); }\n");
    out.push_str("}\n");
    Ok(())
}

pub(in crate::codegen) fn render_borrow_routed_native_bf16_gemv_pair_kernel(
    out: &mut String,
    kernel_name: &str,
    assignment: &GpuAssign,
    dialect: GpuDialect,
) -> Result<(), GpuRenderError> {
    let [input_output, selected_output, gate_output, up_output] = assignment.outputs.as_slice()
    else {
        return Err(invalid_outputs(assignment, 4));
    };
    for (output, expected) in [
        (input_output, CType::BF16),
        (selected_output, CType::U64),
        (gate_output, CType::BF16),
        (up_output, CType::BF16),
    ] {
        let Some(CType::Pointer(element)) = runtime_type(output) else {
            return Err(GpuRenderError::ErasedType(output.clone()));
        };
        if element.as_ref() != &expected {
            return Err(GpuRenderError::UnsupportedType(
                runtime_type(output).unwrap().clone(),
            ));
        }
    }
    let _ = borrow_routed_bf16_gemv_pair_parts(assignment)?;

    out.push_str(&format!(
        "__global__ void {kernel_name}(const catena_bf16_t *input, const catena_bf16_t *gate_weight, const catena_bf16_t *up_weight, const uint64_t *selected, catena_bf16_t *gate_out, catena_bf16_t *up_out, uint64_t output_len, uint64_t reduction_len, uint64_t slots, uint64_t output_features, uint64_t expert_count) {{\n"
    ));
    out.push_str("    uint64_t output_index = (uint64_t)blockIdx.x + (uint64_t)blockIdx.y * (uint64_t)gridDim.x;\n");
    out.push_str("    if (output_index >= output_len) { return; }\n");
    out.push_str("    uint64_t active_width = slots * output_features;\n");
    out.push_str("    uint64_t row = output_index / active_width;\n");
    out.push_str("    uint64_t row_element = output_index % active_width;\n");
    out.push_str("    uint64_t slot = row_element / output_features;\n");
    out.push_str("    uint64_t output_feature = row_element % output_features;\n");
    out.push_str("    uint64_t expert = selected[row * slots + slot];\n");
    out.push_str("    catena_assert(expert < expert_count);\n");
    out.push_str("    uint64_t input_base = row * reduction_len;\n");
    out.push_str(
        "    uint64_t weight_base = (expert * output_features + output_feature) * reduction_len;\n",
    );
    out.push_str("    float gate_sum = 0.0f;\n    float up_sum = 0.0f;\n");
    out.push_str("    uint64_t pair_count = reduction_len >> 1;\n");
    if dialect == GpuDialect::Hip {
        out.push_str(&hip_gfx11_guard());
        out.push_str(
            "    typedef __bf16 catena_native_bf16x2_t __attribute__((ext_vector_type(2)));\n",
        );
        out.push_str("#endif\n");
    }
    out.push_str("    for (uint64_t pair = (uint64_t)threadIdx.x; pair < pair_count; pair += (uint64_t)blockDim.x) {\n");
    match dialect {
        GpuDialect::Hip => {
            out.push_str(&hip_gfx11_guard());
            out.push_str("        union { uint32_t bits; catena_native_bf16x2_t values; } input_pair, gate_pair, up_pair;\n");
            out.push_str("        __builtin_memcpy(&input_pair.bits, input + input_base + (pair << 1), sizeof(uint32_t));\n");
            out.push_str("        __builtin_memcpy(&gate_pair.bits, gate_weight + weight_base + (pair << 1), sizeof(uint32_t));\n");
            out.push_str("        __builtin_memcpy(&up_pair.bits, up_weight + weight_base + (pair << 1), sizeof(uint32_t));\n");
            out.push_str("        gate_sum = __builtin_amdgcn_fdot2_f32_bf16(input_pair.values, gate_pair.values, gate_sum, false);\n");
            out.push_str("        up_sum = __builtin_amdgcn_fdot2_f32_bf16(input_pair.values, up_pair.values, up_sum, false);\n");
            out.push_str("#else\n");
            out.push_str("        uint64_t index = pair << 1;\n");
            out.push_str("        float x0 = catena_bf16_to_f32(input[input_base + index]);\n");
            out.push_str("        float x1 = catena_bf16_to_f32(input[input_base + index + 1]);\n");
            out.push_str("        gate_sum += x0 * catena_bf16_to_f32(gate_weight[weight_base + index]) + x1 * catena_bf16_to_f32(gate_weight[weight_base + index + 1]);\n");
            out.push_str("        up_sum += x0 * catena_bf16_to_f32(up_weight[weight_base + index]) + x1 * catena_bf16_to_f32(up_weight[weight_base + index + 1]);\n");
            out.push_str("#endif\n");
        }
        GpuDialect::Cuda => {
            out.push_str("        uint64_t index = pair << 1;\n");
            out.push_str("        float x0 = catena_bf16_to_f32(input[input_base + index]);\n");
            out.push_str("        float x1 = catena_bf16_to_f32(input[input_base + index + 1]);\n");
            out.push_str("        gate_sum += x0 * catena_bf16_to_f32(gate_weight[weight_base + index]) + x1 * catena_bf16_to_f32(gate_weight[weight_base + index + 1]);\n");
            out.push_str("        up_sum += x0 * catena_bf16_to_f32(up_weight[weight_base + index]) + x1 * catena_bf16_to_f32(up_weight[weight_base + index + 1]);\n");
        }
    }
    out.push_str("    }\n");
    out.push_str("    if ((reduction_len & 1) != 0 && threadIdx.x == 0) {\n");
    out.push_str("        uint64_t index = reduction_len - 1;\n");
    out.push_str("        float x = catena_bf16_to_f32(input[input_base + index]);\n");
    out.push_str("        gate_sum += x * catena_bf16_to_f32(gate_weight[weight_base + index]);\n");
    out.push_str("        up_sum += x * catena_bf16_to_f32(up_weight[weight_base + index]);\n");
    out.push_str("    }\n");
    out.push_str(
        "    __shared__ float gate_partial[256];\n    __shared__ float up_partial[256];\n",
    );
    out.push_str("    gate_partial[threadIdx.x] = gate_sum;\n    up_partial[threadIdx.x] = up_sum;\n    __syncthreads();\n");
    out.push_str("    for (uint32_t stride = 128; stride != 0; stride >>= 1) {\n");
    out.push_str("        if (threadIdx.x < stride) { gate_partial[threadIdx.x] += gate_partial[threadIdx.x + stride]; up_partial[threadIdx.x] += up_partial[threadIdx.x + stride]; }\n");
    out.push_str("        __syncthreads();\n    }\n");
    out.push_str("    if (threadIdx.x == 0) { gate_out[output_index] = catena_bf16_from_f32(gate_partial[0]); up_out[output_index] = catena_bf16_from_f32(up_partial[0]); }\n");
    out.push_str("}\n");
    Ok(())
}

pub(in crate::codegen) fn render_borrow_routed_native_bf16_gemm_wmma_pair_kernels(
    out: &mut String,
    kernel_name: &str,
    assignment: &GpuAssign,
    dialect: GpuDialect,
) -> Result<(), GpuRenderError> {
    render_borrow_routed_native_bf16_gemv_pair_kernel(
        out,
        &format!("{kernel_name}_simt"),
        assignment,
        dialect,
    )?;
    if dialect != GpuDialect::Hip {
        return Ok(());
    }

    out.push_str(&hip_wmma_include_guard());
    let source = r#"
__global__ void __KERNEL___group(
    const uint64_t *selected, uint64_t route_count, uint64_t expert_count,
    uint64_t *offsets, uint64_t *permutation) {
    uint64_t expert = (uint64_t)blockIdx.x;
    uint32_t lane = threadIdx.x;
    uint64_t offset = 0;
    for (uint64_t route_base = 0; route_base < route_count; route_base += 32) {
        uint64_t route = route_base + lane;
        unsigned long long lower = __ballot(
            route < route_count && selected[route] < expert);
        offset += (uint64_t)__popcll(lower);
    }
    if (lane == 0) {
        offsets[expert] = offset;
    }
    uint64_t written = 0;
    for (uint64_t route_base = 0; route_base < route_count; route_base += 32) {
        uint64_t route = route_base + lane;
        bool matches = route < route_count && selected[route] == expert;
        unsigned long long match_mask = __ballot(matches);
        unsigned long long preceding = lane == 0 ? 0 : ((1ull << lane) - 1);
        if (matches) {
            permutation[offset + written + (uint64_t)__popcll(match_mask & preceding)] = route;
        }
        written += (uint64_t)__popcll(match_mask);
    }
    if (lane == 0 && expert + 1 == expert_count) {
        offsets[expert_count] = offset + written;
        catena_assert(offsets[expert_count] == route_count);
    }
}

__global__ void __KERNEL__(
    const catena_bf16_t *input, const catena_bf16_t *gate_weight,
    const catena_bf16_t *up_weight, const uint64_t *offsets,
    const uint64_t *permutation, catena_bf16_t *gate_out,
    catena_bf16_t *up_out, uint64_t reduction_len, uint64_t slots,
    uint64_t output_features) {
    using frag_a_t = rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, catena_bf16_t, rocwmma::row_major>;
    using frag_b_t = rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, catena_bf16_t, rocwmma::col_major>;
    using frag_c_t = rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float>;

    uint64_t expert = (uint64_t)blockIdx.y;
    uint64_t column_base = (uint64_t)blockIdx.x * 64;
    uint64_t route_begin = offsets[expert];
    uint64_t route_end = offsets[expert + 1];
    uint32_t wave = threadIdx.x / 32;
    uint32_t lane = threadIdx.x % 32;
    __shared__ catena_bf16_t tile_a[64 * 16];
    __shared__ catena_bf16_t tile_gate_b[64 * 16];
    __shared__ catena_bf16_t tile_up_b[64 * 16];
    __shared__ float tile_gate_c[4 * 16 * 16];
    __shared__ float tile_up_c[4 * 16 * 16];

    for (uint64_t route_base = route_begin; route_base < route_end; route_base += 64) {
        frag_a_t a;
        frag_b_t gate_b[4];
        frag_b_t up_b[4];
        frag_c_t gate_c[4];
        frag_c_t up_c[4];
        #pragma unroll
        for (uint32_t fragment = 0; fragment < 4; ++fragment) {
            rocwmma::fill_fragment(gate_c[fragment], 0.0f);
            rocwmma::fill_fragment(up_c[fragment], 0.0f);
        }
        for (uint64_t k_base = 0; k_base < reduction_len; k_base += 16) {
            for (uint32_t flat = threadIdx.x; flat < 64 * 16; flat += blockDim.x) {
                uint32_t tile_row = flat >> 4;
                uint32_t tile_column = flat & 15;
                uint64_t grouped_route = route_base + tile_row;
                uint64_t k = k_base + tile_column;
                if (grouped_route < route_end && k < reduction_len) {
                    uint64_t route = permutation[grouped_route];
                    tile_a[flat] = input[(route / slots) * reduction_len + k];
                } else {
                    tile_a[flat] = catena_bf16_from_f32(0.0f);
                }
                uint64_t column = column_base + tile_row;
                if (column < output_features && k < reduction_len) {
                    uint64_t weight_index =
                        (expert * output_features + column) * reduction_len + k;
                    tile_gate_b[flat] = gate_weight[weight_index];
                    tile_up_b[flat] = up_weight[weight_index];
                } else {
                    tile_gate_b[flat] = catena_bf16_from_f32(0.0f);
                    tile_up_b[flat] = catena_bf16_from_f32(0.0f);
                }
            }
            __syncthreads();
            rocwmma::load_matrix_sync(a, tile_a + wave * 16 * 16, 16);
            #pragma unroll
            for (uint32_t fragment = 0; fragment < 4; ++fragment) {
                rocwmma::load_matrix_sync(
                    gate_b[fragment], tile_gate_b + fragment * 16 * 16, 16);
                rocwmma::load_matrix_sync(
                    up_b[fragment], tile_up_b + fragment * 16 * 16, 16);
                rocwmma::mma_sync(
                    gate_c[fragment], a, gate_b[fragment], gate_c[fragment]);
                rocwmma::mma_sync(
                    up_c[fragment], a, up_b[fragment], up_c[fragment]);
            }
            __syncthreads();
        }
        #pragma unroll
        for (uint32_t fragment = 0; fragment < 4; ++fragment) {
            rocwmma::store_matrix_sync(
                tile_gate_c + wave * 16 * 16, gate_c[fragment], 16,
                rocwmma::mem_row_major);
            rocwmma::store_matrix_sync(
                tile_up_c + wave * 16 * 16, up_c[fragment], 16,
                rocwmma::mem_row_major);
            __syncthreads();
            for (uint32_t flat = lane; flat < 16 * 16; flat += 32) {
                uint64_t grouped_route = route_base + wave * 16 + (flat >> 4);
                uint64_t column = column_base + fragment * 16 + (flat & 15);
                if (grouped_route < route_end && column < output_features) {
                    uint64_t route = permutation[grouped_route];
                    gate_out[route * output_features + column] =
                        catena_bf16_from_f32(tile_gate_c[wave * 16 * 16 + flat]);
                    up_out[route * output_features + column] =
                        catena_bf16_from_f32(tile_up_c[wave * 16 * 16 + flat]);
                }
            }
            __syncthreads();
        }
    }
}

"#;
    out.push_str(&source.replace("__KERNEL__", kernel_name));
    out.push_str("#endif\n");
    Ok(())
}

pub(in crate::codegen) fn render_borrow_routed_native_bf16_gemv_residual_kernel(
    out: &mut String,
    kernel_name: &str,
    assignment: &GpuAssign,
    dialect: GpuDialect,
) -> Result<(), GpuRenderError> {
    let [active_output, selected_output, residual_output, output] = assignment.outputs.as_slice()
    else {
        return Err(invalid_outputs(assignment, 4));
    };
    for (value, expected) in [
        (active_output, CType::BF16),
        (selected_output, CType::U64),
        (residual_output, CType::BF16),
        (output, CType::BF16),
    ] {
        let Some(CType::Pointer(element)) = runtime_type(value) else {
            return Err(GpuRenderError::ErasedType(value.clone()));
        };
        if element.as_ref() != &expected {
            return Err(GpuRenderError::UnsupportedType(
                runtime_type(value).unwrap().clone(),
            ));
        }
    }
    let _ = borrow_routed_bf16_gemv_residual_parts(assignment)?;

    out.push_str(&format!("__global__ void {kernel_name}(const catena_bf16_t *active, const catena_bf16_t *down_weight, const uint64_t *selected, const catena_bf16_t *residual, catena_bf16_t *out, uint64_t output_len, uint64_t reduction_len, uint64_t slots, uint64_t intermediate, uint64_t output_features, uint64_t expert_count) {{\n"));
    out.push_str("    uint64_t output_index = (uint64_t)blockIdx.x + (uint64_t)blockIdx.y * (uint64_t)gridDim.x;\n");
    out.push_str("    if (output_index >= output_len) { return; }\n");
    out.push_str("    uint64_t row = output_index / output_features;\n");
    out.push_str("    uint64_t output_feature = output_index % output_features;\n");
    out.push_str("    uint64_t active_base = row * reduction_len;\n");
    out.push_str("    float sum = 0.0f;\n");
    out.push_str("    uint64_t pair_count = reduction_len >> 1;\n");
    if dialect == GpuDialect::Hip {
        out.push_str(&hip_gfx11_guard());
        out.push_str(
            "    typedef __bf16 catena_native_bf16x2_t __attribute__((ext_vector_type(2)));\n",
        );
        out.push_str("#endif\n");
    }
    out.push_str("    for (uint64_t pair = (uint64_t)threadIdx.x; pair < pair_count; pair += (uint64_t)blockDim.x) {\n");
    out.push_str("        uint64_t flat = pair << 1;\n");
    out.push_str("        uint64_t slot = flat / intermediate;\n");
    out.push_str("        uint64_t channel = flat % intermediate;\n");
    out.push_str("        uint64_t expert = selected[row * slots + slot];\n");
    out.push_str("        catena_assert(expert < expert_count);\n");
    out.push_str("        uint64_t weight_base = (expert * output_features + output_feature) * intermediate + channel;\n");
    match dialect {
        GpuDialect::Hip => {
            out.push_str(&hip_gfx11_guard());
            out.push_str("        if (channel + 1 < intermediate) {\n");
            out.push_str("            union { uint32_t bits; catena_native_bf16x2_t values; } active_pair, weight_pair;\n");
            out.push_str("            __builtin_memcpy(&active_pair.bits, active + active_base + flat, sizeof(uint32_t));\n");
            out.push_str("            __builtin_memcpy(&weight_pair.bits, down_weight + weight_base, sizeof(uint32_t));\n");
            out.push_str("            sum = __builtin_amdgcn_fdot2_f32_bf16(active_pair.values, weight_pair.values, sum, false);\n");
            out.push_str("        } else {\n");
            out.push_str("            sum += catena_bf16_to_f32(active[active_base + flat]) * catena_bf16_to_f32(down_weight[weight_base]);\n");
            out.push_str("            uint64_t next_slot = slot + 1;\n");
            out.push_str("            uint64_t next_expert = selected[row * slots + next_slot];\n");
            out.push_str("            uint64_t next_weight = (next_expert * output_features + output_feature) * intermediate;\n");
            out.push_str("            sum += catena_bf16_to_f32(active[active_base + flat + 1]) * catena_bf16_to_f32(down_weight[next_weight]);\n");
            out.push_str("        }\n");
            out.push_str("#else\n");
            out.push_str("        float x0 = catena_bf16_to_f32(active[active_base + flat]);\n");
            out.push_str("        float w0 = catena_bf16_to_f32(down_weight[weight_base]);\n");
            out.push_str("        uint64_t flat1 = flat + 1; uint64_t slot1 = flat1 / intermediate; uint64_t channel1 = flat1 % intermediate; uint64_t expert1 = selected[row * slots + slot1];\n");
            out.push_str("        uint64_t weight1 = (expert1 * output_features + output_feature) * intermediate + channel1;\n");
            out.push_str("        sum += x0 * w0 + catena_bf16_to_f32(active[active_base + flat1]) * catena_bf16_to_f32(down_weight[weight1]);\n");
            out.push_str("#endif\n");
        }
        GpuDialect::Cuda => {
            out.push_str("        float x0 = catena_bf16_to_f32(active[active_base + flat]);\n");
            out.push_str("        float w0 = catena_bf16_to_f32(down_weight[weight_base]);\n");
            out.push_str("        uint64_t flat1 = flat + 1; uint64_t slot1 = flat1 / intermediate; uint64_t channel1 = flat1 % intermediate; uint64_t expert1 = selected[row * slots + slot1];\n");
            out.push_str("        uint64_t weight1 = (expert1 * output_features + output_feature) * intermediate + channel1;\n");
            out.push_str("        sum += x0 * w0 + catena_bf16_to_f32(active[active_base + flat1]) * catena_bf16_to_f32(down_weight[weight1]);\n");
        }
    }
    out.push_str("    }\n");
    out.push_str("    if ((reduction_len & 1) != 0 && threadIdx.x == 0) {\n");
    out.push_str("        uint64_t flat = reduction_len - 1; uint64_t slot = flat / intermediate; uint64_t channel = flat % intermediate; uint64_t expert = selected[row * slots + slot];\n");
    out.push_str("        uint64_t weight_index = (expert * output_features + output_feature) * intermediate + channel;\n");
    out.push_str("        sum += catena_bf16_to_f32(active[active_base + flat]) * catena_bf16_to_f32(down_weight[weight_index]);\n    }\n");
    out.push_str("    __shared__ float partial[256];\n    partial[threadIdx.x] = sum;\n    __syncthreads();\n");
    out.push_str("    for (uint32_t stride = 128; stride != 0; stride >>= 1) { if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride]; __syncthreads(); }\n");
    out.push_str("    if (threadIdx.x == 0) out[output_index] = catena_bf16_from_f32(partial[0] + catena_bf16_to_f32(residual[output_index]));\n");
    out.push_str("}\n");
    Ok(())
}

pub(in crate::codegen) fn render_borrow_routed_native_bf16_gemm_wmma_residual_kernels(
    out: &mut String,
    kernel_name: &str,
    assignment: &GpuAssign,
    dialect: GpuDialect,
) -> Result<(), GpuRenderError> {
    render_borrow_routed_native_bf16_gemv_residual_kernel(
        out,
        &format!("{kernel_name}_simt"),
        assignment,
        dialect,
    )?;
    if dialect != GpuDialect::Hip {
        return Ok(());
    }

    out.push_str(&hip_wmma_include_guard());
    let source = r#"
__global__ void __KERNEL___group(
    const uint64_t *selected, uint64_t route_count, uint64_t expert_count,
    uint64_t *offsets, uint64_t *permutation) {
    uint64_t expert = (uint64_t)blockIdx.x;
    uint32_t lane = threadIdx.x;
    uint64_t offset = 0;
    for (uint64_t route_base = 0; route_base < route_count; route_base += 32) {
        uint64_t route = route_base + lane;
        unsigned long long lower = __ballot(
            route < route_count && selected[route] < expert);
        offset += (uint64_t)__popcll(lower);
    }
    if (lane == 0) {
        offsets[expert] = offset;
    }
    uint64_t written = 0;
    for (uint64_t route_base = 0; route_base < route_count; route_base += 32) {
        uint64_t route = route_base + lane;
        bool matches = route < route_count && selected[route] == expert;
        unsigned long long match_mask = __ballot(matches);
        unsigned long long preceding = lane == 0 ? 0 : ((1ull << lane) - 1);
        if (matches) {
            permutation[offset + written + (uint64_t)__popcll(match_mask & preceding)] = route;
        }
        written += (uint64_t)__popcll(match_mask);
    }
    if (lane == 0 && expert + 1 == expert_count) {
        offsets[expert_count] = offset + written;
        catena_assert(offsets[expert_count] == route_count);
    }
}

__global__ void __KERNEL__(
    const catena_bf16_t *active, const catena_bf16_t *down_weight,
    const uint64_t *offsets, const uint64_t *permutation,
    float *route_output, uint64_t intermediate, uint64_t output_features) {
    using frag_a_t = rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, catena_bf16_t, rocwmma::row_major>;
    using frag_b_t = rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, catena_bf16_t, rocwmma::col_major>;
    using frag_c_t = rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float>;

    uint64_t expert = (uint64_t)blockIdx.y;
    uint64_t column_base = (uint64_t)blockIdx.x * 128;
    uint64_t route_begin = offsets[expert];
    uint64_t route_end = offsets[expert + 1];
    uint32_t wave = threadIdx.x / 32;
    uint32_t lane = threadIdx.x % 32;
    __shared__ catena_bf16_t tile_a[64 * 16];
    __shared__ catena_bf16_t tile_b[128 * 16];
    __shared__ float tile_c[4 * 16 * 16];

    for (uint64_t route_base = route_begin; route_base < route_end; route_base += 64) {
        frag_a_t a;
        frag_b_t b[8];
        frag_c_t c[8];
        #pragma unroll
        for (uint32_t fragment = 0; fragment < 8; ++fragment) {
            rocwmma::fill_fragment(c[fragment], 0.0f);
        }
        for (uint64_t k_base = 0; k_base < intermediate; k_base += 16) {
            for (uint32_t flat = threadIdx.x; flat < 64 * 16; flat += blockDim.x) {
                uint32_t tile_row = flat >> 4;
                uint32_t tile_column = flat & 15;
                uint64_t grouped_route = route_base + tile_row;
                uint64_t k = k_base + tile_column;
                if (grouped_route < route_end && k < intermediate) {
                    uint64_t route = permutation[grouped_route];
                    tile_a[flat] = active[route * intermediate + k];
                } else {
                    tile_a[flat] = catena_bf16_from_f32(0.0f);
                }
            }
            for (uint32_t flat = threadIdx.x; flat < 128 * 16; flat += blockDim.x) {
                uint32_t tile_row = flat >> 4;
                uint32_t tile_column = flat & 15;
                uint64_t k = k_base + tile_column;
                uint64_t column = column_base + tile_row;
                if (column < output_features && k < intermediate) {
                    tile_b[flat] =
                        down_weight[(expert * output_features + column) * intermediate + k];
                } else {
                    tile_b[flat] = catena_bf16_from_f32(0.0f);
                }
            }
            __syncthreads();
            rocwmma::load_matrix_sync(a, tile_a + wave * 16 * 16, 16);
            #pragma unroll
            for (uint32_t fragment = 0; fragment < 8; ++fragment) {
                rocwmma::load_matrix_sync(
                    b[fragment], tile_b + fragment * 16 * 16, 16);
                rocwmma::mma_sync(c[fragment], a, b[fragment], c[fragment]);
            }
            __syncthreads();
        }
        #pragma unroll
        for (uint32_t fragment = 0; fragment < 8; ++fragment) {
            rocwmma::store_matrix_sync(
                tile_c + wave * 16 * 16, c[fragment], 16,
                rocwmma::mem_row_major);
            __syncthreads();
            for (uint32_t flat = lane; flat < 16 * 16; flat += 32) {
                uint64_t grouped_route = route_base + wave * 16 + (flat >> 4);
                uint64_t column = column_base + fragment * 16 + (flat & 15);
                if (grouped_route < route_end && column < output_features) {
                    uint64_t route = permutation[grouped_route];
                    route_output[route * output_features + column] =
                        tile_c[wave * 16 * 16 + flat];
                }
            }
            __syncthreads();
        }
    }
}

__global__ void __KERNEL___combine(
    const float *route_output, const catena_bf16_t *residual,
    catena_bf16_t *out, uint64_t output_len, uint64_t slots,
    uint64_t output_features) {
    uint64_t output_index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (output_index >= output_len) {
        return;
    }
    uint64_t row = output_index / output_features;
    uint64_t column = output_index % output_features;
    float sum = catena_bf16_to_f32(residual[output_index]);
    for (uint64_t slot = 0; slot < slots; ++slot) {
        sum += route_output[(row * slots + slot) * output_features + column];
    }
    out[output_index] = catena_bf16_from_f32(sum);
}
"#;
    out.push_str(&source.replace("__KERNEL__", kernel_name));
    out.push_str("#endif\n");
    Ok(())
}

pub(in crate::codegen) fn render_bf16_gemv_kernel(
    out: &mut String,
    kernel_name: &str,
    assignment: &GpuAssign,
    dialect: GpuDialect,
) -> Result<(), GpuRenderError> {
    let [output] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 1));
    };
    let Some(CType::Pointer(element)) = runtime_type(output) else {
        return Err(GpuRenderError::ErasedType(output.clone()));
    };
    if element.as_ref() != &CType::BF16 {
        return Err(GpuRenderError::UnsupportedType(
            runtime_type(output).unwrap().clone(),
        ));
    }
    let _ = bf16_gemv_parts(assignment)?;

    out.push_str(&format!(
        "__global__ void {kernel_name}(const catena_bf16_t *input, const catena_bf16_t *weight, catena_bf16_t *out, uint64_t output_len, uint64_t reduction_len, uint64_t input_row, uint64_t output_features) {{\n"
    ));
    out.push_str("    uint64_t output_index = (uint64_t)blockIdx.x;\n");
    out.push_str("    if (output_index >= output_len) { return; }\n");
    out.push_str("    uint64_t row = input_row + output_index / output_features;\n");
    out.push_str("    uint64_t output_feature = output_index % output_features;\n");
    out.push_str("    uint64_t input_base = row * reduction_len;\n");
    out.push_str("    uint64_t weight_base = output_feature * reduction_len;\n");
    out.push_str("    float sum = 0.0f;\n");
    out.push_str("    uint64_t pair_count = reduction_len >> 1;\n");
    if dialect == GpuDialect::Hip {
        out.push_str(&hip_gfx11_guard());
        out.push_str(
            "    typedef __bf16 catena_native_bf16x2_t __attribute__((ext_vector_type(2)));\n",
        );
        out.push_str("#endif\n");
    }
    out.push_str("    for (uint64_t pair = (uint64_t)threadIdx.x; pair < pair_count; pair += (uint64_t)blockDim.x) {\n");
    match dialect {
        GpuDialect::Hip => {
            out.push_str(&hip_gfx11_guard());
            out.push_str("        union { uint32_t bits; catena_native_bf16x2_t values; } input_pair, weight_pair;\n");
            out.push_str("        __builtin_memcpy(&input_pair.bits, input + input_base + (pair << 1), sizeof(uint32_t));\n");
            out.push_str("        __builtin_memcpy(&weight_pair.bits, weight + weight_base + (pair << 1), sizeof(uint32_t));\n");
            out.push_str("        sum = __builtin_amdgcn_fdot2_f32_bf16(input_pair.values, weight_pair.values, sum, false);\n");
            out.push_str("#else\n");
            out.push_str("        uint64_t index = pair << 1;\n");
            out.push_str("        sum += catena_bf16_to_f32(input[input_base + index]) * catena_bf16_to_f32(weight[weight_base + index]);\n");
            out.push_str("        sum += catena_bf16_to_f32(input[input_base + index + 1]) * catena_bf16_to_f32(weight[weight_base + index + 1]);\n");
            out.push_str("#endif\n");
        }
        GpuDialect::Cuda => {
            out.push_str("        uint64_t index = pair << 1;\n");
            out.push_str("        sum += catena_bf16_to_f32(input[input_base + index]) * catena_bf16_to_f32(weight[weight_base + index]);\n");
            out.push_str("        sum += catena_bf16_to_f32(input[input_base + index + 1]) * catena_bf16_to_f32(weight[weight_base + index + 1]);\n");
        }
    }
    out.push_str("    }\n");
    out.push_str("    if ((reduction_len & 1) != 0 && threadIdx.x == 0) {\n");
    out.push_str("        uint64_t index = reduction_len - 1;\n");
    out.push_str("        sum += catena_bf16_to_f32(input[input_base + index]) * catena_bf16_to_f32(weight[weight_base + index]);\n");
    out.push_str("    }\n");
    out.push_str("    __shared__ float partial[256];\n");
    out.push_str("    partial[threadIdx.x] = sum;\n");
    out.push_str("    __syncthreads();\n");
    out.push_str("    for (uint32_t stride = 128; stride != 0; stride >>= 1) {\n");
    out.push_str("        if (threadIdx.x < stride) { partial[threadIdx.x] += partial[threadIdx.x + stride]; }\n");
    out.push_str("        __syncthreads();\n");
    out.push_str("    }\n");
    out.push_str(
        "    if (threadIdx.x == 0) { out[output_index] = catena_bf16_from_f32(partial[0]); }\n",
    );
    out.push_str("}\n");
    Ok(())
}

fn render_bf16_gemm_wmma_kernel(out: &mut String, kernel_name: &str, dialect: GpuDialect) {
    let (namespace, include, matrix_a, matrix_b, accumulator, row_major, col_major, store_layout) =
        match dialect {
            GpuDialect::Hip => (
                "rocwmma",
                hip_wmma_include_guard(),
                "rocwmma::matrix_a",
                "rocwmma::matrix_b",
                "rocwmma::accumulator",
                "rocwmma::row_major",
                "rocwmma::col_major",
                "rocwmma::mem_row_major",
            ),
            GpuDialect::Cuda => (
                "nvcuda::wmma",
                "#include <mma.h>\n".to_owned(),
                "nvcuda::wmma::matrix_a",
                "nvcuda::wmma::matrix_b",
                "nvcuda::wmma::accumulator",
                "nvcuda::wmma::row_major",
                "nvcuda::wmma::col_major",
                "nvcuda::wmma::mem_row_major",
            ),
        };
    out.push_str(&include);
    out.push_str(&format!(
        "__global__ void {kernel_name}(const catena_bf16_t *input, const catena_bf16_t *weight, catena_bf16_t *out, uint64_t rows, uint64_t reduction_len, uint64_t input_row, uint64_t output_features) {{\n"
    ));
    out.push_str(&format!("    using frag_a_t = {namespace}::fragment<{matrix_a}, 16, 16, 16, catena_bf16_t, {row_major}>;\n"));
    out.push_str(&format!("    using frag_b_t = {namespace}::fragment<{matrix_b}, 16, 16, 16, catena_bf16_t, {col_major}>;\n"));
    out.push_str(&format!(
        "    using frag_c_t = {namespace}::fragment<{accumulator}, 16, 16, 16, float>;\n"
    ));
    out.push_str("    uint64_t row_base = (uint64_t)blockIdx.y * 128;\n    uint64_t column_base = (uint64_t)blockIdx.x * 128;\n    uint32_t wave = threadIdx.x / 32;\n    uint32_t lane = threadIdx.x % 32;\n");
    out.push_str("    __shared__ catena_bf16_t tile_a[128 * 16];\n    __shared__ catena_bf16_t tile_b[128 * 16];\n    __shared__ float tile_c[8 * 16 * 16];\n");
    out.push_str(&format!(
        "    frag_a_t a; frag_b_t b[8]; frag_c_t c[8];\n    #pragma unroll\n    for (uint32_t fragment = 0; fragment < 8; ++fragment) {{ {namespace}::fill_fragment(c[fragment], 0.0f); }}\n"
    ));
    out.push_str("    for (uint64_t k_base = 0; k_base < reduction_len; k_base += 16) {\n        for (uint32_t flat = threadIdx.x; flat < 128 * 16; flat += blockDim.x) {\n");
    out.push_str("            uint32_t tile_row = flat >> 4; uint32_t tile_column = flat & 15;\n            uint64_t row = row_base + tile_row; uint64_t k = k_base + tile_column;\n");
    out.push_str("            tile_a[flat] = (row < rows && k < reduction_len) ? input[(input_row + row) * reduction_len + k] : catena_bf16_from_f32(0.0f);\n");
    out.push_str("            uint64_t column = column_base + tile_row;\n            tile_b[flat] = (column < output_features && k < reduction_len) ? weight[column * reduction_len + k] : catena_bf16_from_f32(0.0f);\n");
    out.push_str(&format!("        }}\n        __syncthreads();\n        {namespace}::load_matrix_sync(a, tile_a + wave * 16 * 16, 16);\n        #pragma unroll\n        for (uint32_t fragment = 0; fragment < 8; ++fragment) {{\n            {namespace}::load_matrix_sync(b[fragment], tile_b + fragment * 16 * 16, 16);\n            {namespace}::mma_sync(c[fragment], a, b[fragment], c[fragment]);\n        }}\n        __syncthreads();\n    }}\n"));
    out.push_str(&format!("    #pragma unroll\n    for (uint32_t fragment = 0; fragment < 8; ++fragment) {{\n        {namespace}::store_matrix_sync(tile_c + wave * 16 * 16, c[fragment], 16, {store_layout});\n        __syncthreads();\n        for (uint32_t flat = lane; flat < 16 * 16; flat += 32) {{\n            uint64_t row = row_base + wave * 16 + (flat >> 4);\n            uint64_t column = column_base + fragment * 16 + (flat & 15);\n            if (row < rows && column < output_features) {{\n                out[row * output_features + column] = catena_bf16_from_f32(tile_c[wave * 16 * 16 + flat]);\n            }}\n        }}\n        __syncthreads();\n    }}\n}}\n"));
    if dialect == GpuDialect::Hip {
        out.push_str("#endif\n");
    }
}

/// Render an opt-in BF16 GEMM. HIP gfx11 and CUDA sm80+ cooperatively execute a
/// 128x128 output tile with eight waves/warps; the companion SIMT kernel handles
/// small inputs.
pub(in crate::codegen) fn render_bf16_gemm_wmma_kernels(
    out: &mut String,
    kernel_name: &str,
    assignment: &GpuAssign,
    dialect: GpuDialect,
) -> Result<(), GpuRenderError> {
    render_bf16_gemv_kernel(out, &format!("{kernel_name}_simt"), assignment, dialect)?;
    render_bf16_gemm_wmma_kernel(out, kernel_name, dialect);
    Ok(())
}

pub(in crate::codegen) fn render_borrow_native_bf16_gemm_wmma_pair_kernels(
    out: &mut String,
    kernel_name: &str,
    assignment: &GpuAssign,
    dialect: GpuDialect,
) -> Result<(), GpuRenderError> {
    render_borrow_native_bf16_gemv_pair_kernel(
        out,
        &format!("{kernel_name}_simt"),
        assignment,
        dialect,
    )?;
    render_bf16_gemm_wmma_kernel(out, &format!("{kernel_name}_gate"), dialect);
    render_bf16_gemm_wmma_kernel(out, &format!("{kernel_name}_up"), dialect);
    Ok(())
}

pub(in crate::codegen) fn render_blas_support(out: &mut String, dialect: GpuDialect) {
    out.push_str(match dialect {
        GpuDialect::Hip => {
            r#"#include <hipblas/hipblas.h>
#ifndef __HIP_DEVICE_COMPILE__
__host__ static inline void catena_host_blas_check(hipblasStatus_t status) {
    if (status != HIPBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "catena hipBLAS error: %s\n", hipblasStatusToString(status));
        fflush(stderr);
        __builtin_trap();
    }
}

struct catena_blas_context_t {
    hipblasHandle_t handle;

    catena_blas_context_t() : handle(nullptr) {
        catena_host_blas_check(hipblasCreate(&handle));
        catena_host_blas_check(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        catena_host_blas_check(hipblasSetStream(handle, nullptr));
    }

    ~catena_blas_context_t() {
        if (handle != nullptr) {
            (void)hipblasDestroy(handle);
        }
    }
};

__host__ static inline hipblasHandle_t catena_host_blas_handle() {
    static catena_blas_context_t context;
    return context.handle;
}

__host__ static inline void catena_host_blas_gemm(
    const catena_bf16_t *weight, const catena_bf16_t *input, catena_bf16_t *output,
    int output_features, int rows, int reduction_len, const float *alpha, const float *beta) {
    catena_host_blas_check(hipblasGemmEx(
        catena_host_blas_handle(), HIPBLAS_OP_T, HIPBLAS_OP_N,
        output_features, rows, reduction_len, alpha,
        weight, HIP_R_16BF, reduction_len, input, HIP_R_16BF, reduction_len,
        beta, output, HIP_R_16BF, output_features,
        HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
}
#endif
"#
        }
        GpuDialect::Cuda => {
            r#"#include <cublas_v2.h>
#ifndef __CUDA_ARCH__
__host__ static inline void catena_host_blas_check(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "catena cuBLAS error: %d\n", (int)status);
        fflush(stderr);
        __builtin_trap();
    }
}

struct catena_blas_context_t {
    cublasHandle_t handle;

    catena_blas_context_t() : handle(nullptr) {
        catena_host_blas_check(cublasCreate(&handle));
        catena_host_blas_check(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
        catena_host_blas_check(cublasSetStream(handle, nullptr));
    }

    ~catena_blas_context_t() {
        if (handle != nullptr) {
            (void)cublasDestroy(handle);
        }
    }
};

__host__ static inline cublasHandle_t catena_host_blas_handle() {
    static catena_blas_context_t context;
    return context.handle;
}

__host__ static inline void catena_host_blas_gemm(
    const catena_bf16_t *weight, const catena_bf16_t *input, catena_bf16_t *output,
    int output_features, int rows, int reduction_len, const float *alpha, const float *beta) {
    catena_host_blas_check(cublasGemmEx(
        catena_host_blas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
        output_features, rows, reduction_len, alpha,
        weight, CUDA_R_16BF, reduction_len, input, CUDA_R_16BF, reduction_len,
        beta, output, CUDA_R_16BF, output_features,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}
#endif
"#
        }
    });
}

pub(in crate::codegen) fn render_bf16_gemm_blas_kernels(
    out: &mut String,
    kernel_name: &str,
    assignment: &GpuAssign,
    dialect: GpuDialect,
) -> Result<(), GpuRenderError> {
    render_bf16_gemv_kernel(out, &format!("{kernel_name}_simt"), assignment, dialect)
}

pub(in crate::codegen) fn render_borrow_native_bf16_gemm_blas_pair_kernels(
    out: &mut String,
    kernel_name: &str,
    assignment: &GpuAssign,
    dialect: GpuDialect,
) -> Result<(), GpuRenderError> {
    render_borrow_native_bf16_gemv_pair_kernel(
        out,
        &format!("{kernel_name}_simt"),
        assignment,
        dialect,
    )
}

pub(in crate::codegen) fn render_borrow_argmax_f32_kernel(
    out: &mut String,
    kernel_name: &str,
    assignment: &GpuAssign,
) -> Result<(), GpuRenderError> {
    let [source_output, index_output] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 2));
    };
    let Some(CType::Pointer(source_element)) = runtime_type(source_output) else {
        return Err(GpuRenderError::ErasedType(source_output.clone()));
    };
    if !matches!(source_element.as_ref(), CType::F32 | CType::BF16) {
        return Err(GpuRenderError::UnsupportedType(
            runtime_type(source_output).unwrap().clone(),
        ));
    }
    let Some(CType::Pointer(index_element)) = runtime_type(index_output) else {
        return Err(GpuRenderError::ErasedType(index_output.clone()));
    };
    if index_element.as_ref() != &CType::U64 {
        return Err(GpuRenderError::UnsupportedType(
            runtime_type(index_output).unwrap().clone(),
        ));
    }
    let (_source, _len, _output_len) = borrow_argmax_f32_parts(assignment)?;

    out.push_str(&format!(
        "__global__ void {kernel_name}(const {} *data, uint64_t len, uint64_t *out) {{\n",
        c_type(source_element)
    ));
    out.push_str("    __shared__ uint32_t best_keys[256];\n");
    out.push_str("    __shared__ uint64_t best_indices[256];\n");
    out.push_str("    uint32_t best_key = 0;\n");
    out.push_str("    uint64_t best_index = UINT64_MAX;\n");
    out.push_str(
        "    for (uint64_t index = (uint64_t)threadIdx.x; index < len; index += (uint64_t)blockDim.x) {\n",
    );
    match source_element.as_ref() {
        CType::F32 => {
            out.push_str("        uint32_t bits = catena_f32_bitcast_u32(data[index]);\n");
            out.push_str("        uint32_t key = (bits & 0x80000000u) != 0 ? ~bits : (bits ^ 0x80000000u);\n");
        }
        CType::BF16 => {
            out.push_str(
                "        uint32_t bits = (uint32_t)catena_bf16_bitcast_u16(data[index]);\n",
            );
            out.push_str("        uint32_t key = (bits & 0x8000u) != 0 ? ((~bits) & 0xFFFFu) : (bits ^ 0x8000u);\n");
        }
        _ => unreachable!("source type was validated"),
    }
    out.push_str(
        "        if (best_index == UINT64_MAX || key > best_key || (key == best_key && index > best_index)) {\n",
    );
    out.push_str("            best_key = key;\n");
    out.push_str("            best_index = index;\n");
    out.push_str("        }\n");
    out.push_str("    }\n");
    out.push_str("    best_keys[threadIdx.x] = best_key;\n");
    out.push_str("    best_indices[threadIdx.x] = best_index;\n");
    out.push_str("    __syncthreads();\n");
    out.push_str("    for (uint32_t stride = 128; stride != 0; stride >>= 1) {\n");
    out.push_str("        if (threadIdx.x < stride) {\n");
    out.push_str("            uint64_t right_index = best_indices[threadIdx.x + stride];\n");
    out.push_str("            uint32_t right_key = best_keys[threadIdx.x + stride];\n");
    out.push_str("            uint64_t left_index = best_indices[threadIdx.x];\n");
    out.push_str("            uint32_t left_key = best_keys[threadIdx.x];\n");
    out.push_str(
        "            if (left_index == UINT64_MAX || (right_index != UINT64_MAX && (right_key > left_key || (right_key == left_key && right_index > left_index)))) {\n",
    );
    out.push_str("                best_keys[threadIdx.x] = right_key;\n");
    out.push_str("                best_indices[threadIdx.x] = right_index;\n");
    out.push_str("            }\n");
    out.push_str("        }\n");
    out.push_str("        __syncthreads();\n");
    out.push_str("    }\n");
    out.push_str("    if (threadIdx.x == 0) { out[0] = best_indices[0]; }\n");
    out.push_str("}\n");
    Ok(())
}

pub(in crate::codegen) fn render_borrow_argmax_f32_call(
    out: &mut String,
    function: &GpuFunction,
    assignment: &GpuAssign,
    dialect: GpuDialect,
) -> Result<(), GpuRenderError> {
    let [source_output, index_output] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 2));
    };
    let (source, len, output_len) = borrow_argmax_f32_parts(assignment)?;
    let len = value_expr(len);
    let output_len = value_expr(output_len);
    let kernel_name = kernel_name(&function.name, assignment)?;

    out.push_str(&format!("    catena_assert(({len}) != 0);\n"));
    out.push_str(&format!("    catena_assert(({output_len}) == 1);\n"));
    out.push_str(&format!(
        "    uint64_t *{}_data = nullptr;\n",
        index_output.name
    ));
    out.push_str(&format!(
        "    catena_host_gpu_check({}((void **)&{}_data, sizeof(uint64_t), nullptr));\n",
        dialect.device_alloc_async_fn(),
        index_output.name
    ));
    out.push_str(&format!(
        "    {kernel_name}<<<dim3(1), dim3(256)>>>(\n        {}, {len}, {}_data);\n",
        value_expr(source),
        index_output.name
    ));
    out.push_str(&format!(
        "    {} = {};\n    {} = {}_data;\n",
        source_output.name,
        value_expr(source),
        index_output.name,
        index_output.name
    ));
    Ok(())
}

pub(in crate::codegen) fn render_borrow_topk_f32_kernel(
    out: &mut String,
    kernel_name: &str,
    assignment: &GpuAssign,
) -> Result<(), GpuRenderError> {
    let [source_output, index_output] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 2));
    };
    let Some(CType::Pointer(source_element)) = runtime_type(source_output) else {
        return Err(GpuRenderError::ErasedType(source_output.clone()));
    };
    if source_element.as_ref() != &CType::F32 {
        return Err(GpuRenderError::UnsupportedType(
            runtime_type(source_output).unwrap().clone(),
        ));
    }
    let Some(CType::Pointer(index_element)) = runtime_type(index_output) else {
        return Err(GpuRenderError::ErasedType(index_output.clone()));
    };
    if index_element.as_ref() != &CType::U64 {
        return Err(GpuRenderError::UnsupportedType(
            runtime_type(index_output).unwrap().clone(),
        ));
    }
    let (_source, _len, _rows, _output_len, _columns, _k) = borrow_topk_f32_parts(assignment)?;

    out.push_str(&format!(
        "__global__ void {kernel_name}(const float *data, uint64_t rows, uint64_t columns, uint64_t k, uint64_t *out) {{\n"
    ));
    out.push_str("    uint64_t row = (uint64_t)blockIdx.x;\n");
    out.push_str("    if (row >= rows) { return; }\n");
    out.push_str("    uint64_t row_base = row * columns;\n");
    out.push_str("    if (columns <= 256) {\n");
    out.push_str("        __shared__ uint32_t sort_keys[256];\n");
    out.push_str("        __shared__ uint64_t sort_indices[256];\n");
    out.push_str("        uint32_t lane = (uint32_t)threadIdx.x;\n");
    out.push_str("        uint32_t candidate_key = 0;\n");
    out.push_str("        uint64_t candidate_index = UINT64_MAX;\n");
    out.push_str("        if ((uint64_t)lane < columns) {\n");
    out.push_str(
        "            uint32_t bits = catena_f32_bitcast_u32(data[row_base + (uint64_t)lane]);\n",
    );
    out.push_str("            if ((bits & 0x7FFFFFFFu) == 0) { bits = 0; }\n");
    out.push_str(
        "            candidate_key = (bits & 0x80000000u) != 0 ? ~bits : (bits ^ 0x80000000u);\n",
    );
    out.push_str("            candidate_index = (uint64_t)lane;\n");
    out.push_str("        }\n");
    out.push_str("        sort_keys[lane] = candidate_key;\n");
    out.push_str("        sort_indices[lane] = candidate_index;\n");
    out.push_str("        __syncthreads();\n");
    out.push_str("        for (uint32_t size = 2; size <= (uint32_t)blockDim.x; size <<= 1) {\n");
    out.push_str("            for (uint32_t stride = size >> 1; stride != 0; stride >>= 1) {\n");
    out.push_str("                uint32_t partner = lane ^ stride;\n");
    out.push_str("                if (partner > lane) {\n");
    out.push_str("                    uint32_t left_key = sort_keys[lane];\n");
    out.push_str("                    uint64_t left_index = sort_indices[lane];\n");
    out.push_str("                    uint32_t right_key = sort_keys[partner];\n");
    out.push_str("                    uint64_t right_index = sort_indices[partner];\n");
    out.push_str("                    bool left_better = left_index != UINT64_MAX && (right_index == UINT64_MAX || left_key > right_key || (left_key == right_key && left_index < right_index));\n");
    out.push_str("                    bool right_better = right_index != UINT64_MAX && (left_index == UINT64_MAX || right_key > left_key || (right_key == left_key && right_index < left_index));\n");
    out.push_str(
        "                    bool swap = (lane & size) == 0 ? right_better : left_better;\n",
    );
    out.push_str("                    if (swap) {\n");
    out.push_str("                        sort_keys[lane] = right_key;\n");
    out.push_str("                        sort_indices[lane] = right_index;\n");
    out.push_str("                        sort_keys[partner] = left_key;\n");
    out.push_str("                        sort_indices[partner] = left_index;\n");
    out.push_str("                    }\n");
    out.push_str("                }\n");
    out.push_str("                __syncthreads();\n");
    out.push_str("            }\n");
    out.push_str("        }\n");
    out.push_str(
        "        if ((uint64_t)lane < k) { out[row * k + (uint64_t)lane] = sort_indices[lane]; }\n",
    );
    out.push_str("        return;\n");
    out.push_str("    }\n");
    out.push_str("    if (threadIdx.x != 0) { return; }\n");
    out.push_str("    uint32_t best_keys[8];\n");
    out.push_str("    uint64_t best_indices[8];\n");
    out.push_str("    #pragma unroll\n");
    out.push_str("    for (uint32_t slot = 0; slot < 8; ++slot) { best_keys[slot] = 0; best_indices[slot] = UINT64_MAX; }\n");
    out.push_str("    for (uint64_t column = 0; column < columns; ++column) {\n");
    out.push_str("        uint32_t bits = catena_f32_bitcast_u32(data[row_base + column]);\n");
    out.push_str("        if ((bits & 0x7FFFFFFFu) == 0) { bits = 0; }\n");
    out.push_str("        uint32_t candidate_key = (bits & 0x80000000u) != 0 ? ~bits : (bits ^ 0x80000000u);\n");
    out.push_str("        uint64_t candidate_index = column;\n");
    out.push_str("        #pragma unroll\n");
    out.push_str("        for (uint32_t slot = 0; slot < 8; ++slot) {\n");
    out.push_str("            uint64_t incumbent_index = best_indices[slot];\n");
    out.push_str("            uint32_t incumbent_key = best_keys[slot];\n");
    out.push_str("            bool insert = candidate_index != UINT64_MAX && (incumbent_index == UINT64_MAX || candidate_key > incumbent_key || (candidate_key == incumbent_key && candidate_index < incumbent_index));\n");
    out.push_str("            if (insert) {\n");
    out.push_str("                best_keys[slot] = candidate_key;\n");
    out.push_str("                best_indices[slot] = candidate_index;\n");
    out.push_str("                candidate_key = incumbent_key;\n");
    out.push_str("                candidate_index = incumbent_index;\n");
    out.push_str("            }\n");
    out.push_str("        }\n");
    out.push_str("    }\n");
    out.push_str("    for (uint64_t slot = 0; slot < k; ++slot) { out[row * k + slot] = best_indices[slot]; }\n");
    out.push_str("}\n");
    Ok(())
}

pub(in crate::codegen) fn render_borrow_topk_f32_call(
    out: &mut String,
    function: &GpuFunction,
    assignment: &GpuAssign,
    dialect: GpuDialect,
) -> Result<(), GpuRenderError> {
    let [source_output, index_output] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 2));
    };
    let (source, len, rows, output_len, columns, k) = borrow_topk_f32_parts(assignment)?;
    let len = value_expr(len);
    let rows = value_expr(rows);
    let output_len = value_expr(output_len);
    let columns = value_expr(columns);
    let k = value_expr(k);
    let kernel_name = kernel_name(&function.name, assignment)?;

    out.push_str(&format!("    catena_assert(({columns}) != 0);\n"));
    out.push_str(&format!("    catena_assert(({k}) != 0 && ({k}) <= 8);\n"));
    out.push_str(&format!(
        "    catena_assert(({len}) % ({columns}) == 0 && ({rows}) == ({len}) / ({columns}));\n"
    ));
    out.push_str(&format!(
        "    catena_assert(({rows}) == 0 || ({k}) <= UINT64_MAX / ({rows}));\n"
    ));
    out.push_str(&format!(
        "    catena_assert(({output_len}) == ({rows}) * ({k}));\n"
    ));
    out.push_str(&format!(
        "    uint64_t *{}_data = nullptr;\n",
        index_output.name
    ));
    out.push_str(&format!("    if (({output_len}) != 0) {{\n"));
    out.push_str(&format!(
        "        catena_host_gpu_check({}((void **)&{}_data, ({output_len}) * sizeof(uint64_t), nullptr));\n",
        dialect.device_alloc_async_fn(),
        index_output.name
    ));
    out.push_str(&format!(
        "        {kernel_name}<<<dim3({rows}), dim3(({columns}) <= 128 ? 128 : (({columns}) <= 256 ? 256 : 128))>>>(\n            {}, {rows}, {columns}, {k}, {}_data);\n",
        value_expr(source),
        index_output.name
    ));
    out.push_str("    }\n");
    out.push_str(&format!(
        "    {} = {};\n    {} = {}_data;\n",
        source_output.name,
        value_expr(source),
        index_output.name,
        index_output.name
    ));
    Ok(())
}

pub(in crate::codegen) fn render_borrow_reduce_f32_call(
    out: &mut String,
    function: &GpuFunction,
    assignment: &GpuAssign,
    dialect: GpuDialect,
) -> Result<(), GpuRenderError> {
    let [cache_output, output] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 2));
    };
    let Some(CType::Pointer(output_element)) = runtime_type(output) else {
        return Err(GpuRenderError::ErasedType(output.clone()));
    };
    let (_term, _epilogue, cache, _capacity, output_len, reduction_len, env) =
        borrow_reduce_f32_parts(assignment)?;
    let output_len = value_expr(output_len);
    let reduction_len = value_expr(reduction_len);
    let kernel_name = kernel_name(&function.name, assignment)?;

    out.push_str(&format!(
        "    uint64_t {name}_len = {output_len};\n    {element} *{name}_data = nullptr;\n",
        name = output.name,
        element = c_type(output_element)
    ));
    out.push_str(&format!("    catena_assert(({reduction_len}) <= 16384);\n"));
    out.push_str(&format!("    if ({}_len != 0) {{\n", output.name));
    out.push_str(&format!(
        "        catena_host_gpu_check({}((void **)&{}_data, {}_len * sizeof({}), nullptr));\n",
        dialect.device_alloc_async_fn(),
        output.name,
        output.name,
        c_type(output_element)
    ));
    out.push_str(&format!(
        "        {kernel_name}<<<dim3({}_len), dim3(({reduction_len}) > 4096 ? 512 : (({reduction_len}) > 1024 ? 256 : (({reduction_len}) > 256 ? 128 : 64))), ({reduction_len}) * sizeof(float)>>>(\n            {}, {}_data, {}_len, {reduction_len}",
        output.name,
        value_expr(cache),
        output.name,
        output.name
    ));
    for arg in runtime_values(env) {
        if let GpuValue::Var(var) = arg {
            out.push_str(", ");
            out.push_str(&var.name);
        }
    }
    out.push_str(");\n");
    out.push_str("    }\n");
    out.push_str(&format!(
        "    {} = {};\n    {} = {}_data;\n",
        cache_output.name,
        value_expr(cache),
        output.name,
        output.name
    ));
    Ok(())
}

pub(in crate::codegen) fn render_borrow_routed_bf16_gemv_pair_call(
    out: &mut String,
    function: &GpuFunction,
    assignment: &GpuAssign,
    dialect: GpuDialect,
) -> Result<(), GpuRenderError> {
    let [input_output, selected_output, gate_output, up_output] = assignment.outputs.as_slice()
    else {
        return Err(invalid_outputs(assignment, 4));
    };
    let (
        input,
        input_capacity,
        gate_weight,
        gate_capacity,
        up_weight,
        up_capacity,
        selected,
        selected_capacity,
        output_len,
        reduction_len,
        slots,
        output_features,
    ) = borrow_routed_bf16_gemv_pair_parts(assignment)?;
    let input_capacity = value_expr(input_capacity);
    let gate_capacity = value_expr(gate_capacity);
    let up_capacity = value_expr(up_capacity);
    let selected_capacity = value_expr(selected_capacity);
    let output_len = value_expr(output_len);
    let reduction_len = value_expr(reduction_len);
    let slots = value_expr(slots);
    let output_features = value_expr(output_features);
    let kernel_name = kernel_name(&function.name, assignment)?;
    let prefix = &gate_output.name;

    out.push_str(&format!(
        "    catena_assert(({reduction_len}) != 0 && ({reduction_len}) <= 8192);\n"
    ));
    out.push_str(&format!(
        "    catena_assert(({slots}) != 0 && ({output_features}) != 0);\n"
    ));
    out.push_str(&format!(
        "    catena_assert(({input_capacity}) % ({reduction_len}) == 0);\n"
    ));
    out.push_str(&format!(
        "    uint64_t {prefix}_rows = ({input_capacity}) / ({reduction_len});\n"
    ));
    out.push_str(&format!(
        "    catena_assert({prefix}_rows == 0 || ({slots}) <= UINT64_MAX / {prefix}_rows);\n"
    ));
    out.push_str(&format!(
        "    catena_assert(({selected_capacity}) == {prefix}_rows * ({slots}));\n"
    ));
    out.push_str(&format!(
        "    catena_assert(({selected_capacity}) == 0 || ({output_features}) <= UINT64_MAX / ({selected_capacity}));\n"
    ));
    out.push_str(&format!(
        "    catena_assert(({output_len}) == ({selected_capacity}) * ({output_features}));\n"
    ));
    out.push_str(&format!(
        "    catena_assert(({output_features}) <= UINT64_MAX / ({reduction_len}));\n"
    ));
    out.push_str(&format!(
        "    uint64_t {prefix}_expert_stride = ({output_features}) * ({reduction_len});\n"
    ));
    out.push_str(&format!(
        "    catena_assert(({gate_capacity}) == ({up_capacity}));\n"
    ));
    out.push_str(&format!(
        "    catena_assert(({gate_capacity}) % {prefix}_expert_stride == 0);\n"
    ));
    out.push_str(&format!(
        "    uint64_t {prefix}_expert_count = ({gate_capacity}) / {prefix}_expert_stride;\n"
    ));
    out.push_str(&format!("    catena_assert({prefix}_expert_count != 0);\n"));
    out.push_str(&format!(
        "    float *{}_data = nullptr;\n    float *{}_data = nullptr;\n",
        gate_output.name, up_output.name
    ));
    out.push_str(&format!("    if (({output_len}) != 0) {{\n"));
    out.push_str(&format!(
        "        catena_host_gpu_check({}((void **)&{}_data, ({output_len}) * sizeof(float), nullptr));\n",
        dialect.device_alloc_async_fn(),
        gate_output.name
    ));
    out.push_str(&format!(
        "        catena_host_gpu_check({}((void **)&{}_data, ({output_len}) * sizeof(float), nullptr));\n",
        dialect.device_alloc_async_fn(),
        up_output.name
    ));
    out.push_str(&format!(
        "        {kernel_name}<<<dim3({output_len}), dim3(256), 2 * ({reduction_len}) * sizeof(float)>>>(\n            {}, {}, {}, {}, {}_data, {}_data, {output_len}, {reduction_len}, {slots}, {output_features}, {prefix}_expert_count);\n",
        value_expr(input),
        value_expr(gate_weight),
        value_expr(up_weight),
        value_expr(selected),
        gate_output.name,
        up_output.name,
    ));
    out.push_str("    }\n");
    out.push_str(&format!(
        "    {} = {};\n    {} = {};\n    {} = {}_data;\n    {} = {}_data;\n",
        input_output.name,
        value_expr(input),
        selected_output.name,
        value_expr(selected),
        gate_output.name,
        gate_output.name,
        up_output.name,
        up_output.name,
    ));
    Ok(())
}

pub(in crate::codegen) fn render_borrow_native_bf16_gemm_pair_call(
    out: &mut String,
    function: &GpuFunction,
    assignment: &GpuAssign,
    dialect: GpuDialect,
    backend: ProjectionBackend,
) -> Result<(), GpuRenderError> {
    let [input_output, gate_output, up_output] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 3));
    };
    let (
        input,
        input_capacity,
        gate_weight,
        gate_capacity,
        up_weight,
        up_capacity,
        output_len,
        reduction_len,
        input_row,
        output_features,
    ) = borrow_native_bf16_gemv_pair_parts(assignment)?;
    let input_capacity = value_expr(input_capacity);
    let gate_capacity = value_expr(gate_capacity);
    let up_capacity = value_expr(up_capacity);
    let output_len = value_expr(output_len);
    let reduction_len = value_expr(reduction_len);
    let input_row = value_expr(input_row);
    let output_features = value_expr(output_features);
    let kernel_name = kernel_name(&function.name, assignment)?;
    let prefix = &gate_output.name;

    out.push_str(&format!("    catena_assert(({reduction_len}) != 0);\n"));
    out.push_str(&format!(
        "    catena_assert(({input_capacity}) % ({reduction_len}) == 0);\n"
    ));
    out.push_str(&format!(
        "    uint64_t {prefix}_input_rows = ({input_capacity}) / ({reduction_len});\n"
    ));
    out.push_str(&format!("    catena_assert(({output_features}) != 0);\n"));
    out.push_str(&format!(
        "    catena_assert(({output_len}) % ({output_features}) == 0);\n"
    ));
    out.push_str(&format!(
        "    uint64_t {prefix}_rows = ({output_len}) / ({output_features});\n"
    ));
    out.push_str(&format!(
        "    catena_assert(({input_row}) <= {prefix}_input_rows && {prefix}_rows <= {prefix}_input_rows - ({input_row}));\n"
    ));
    out.push_str(&format!(
        "    catena_assert(({output_features}) <= UINT64_MAX / ({reduction_len}));\n"
    ));
    out.push_str(&format!(
        "    catena_assert(({gate_capacity}) == ({output_features}) * ({reduction_len}) && ({up_capacity}) == ({gate_capacity}));\n"
    ));
    out.push_str(&format!(
        "    catena_bf16_t *{}_data = nullptr;\n    catena_bf16_t *{}_data = nullptr;\n",
        gate_output.name, up_output.name
    ));
    out.push_str(&format!("    if (({output_len}) != 0) {{\n"));
    for output in [gate_output, up_output] {
        out.push_str(&format!(
            "        catena_host_gpu_check({}((void **)&{}_data, ({output_len}) * sizeof(catena_bf16_t), nullptr));\n",
            dialect.device_alloc_async_fn(),
            output.name
        ));
    }
    match backend {
        ProjectionBackend::Simt => {
            out.push_str(&format!(
                "        {kernel_name}<<<dim3({output_len}), dim3(256)>>>(\n            {}, {}, {}, {}_data, {}_data, {output_len}, {reduction_len}, {input_row}, {output_features});\n",
                value_expr(input),
                value_expr(gate_weight),
                value_expr(up_weight),
                gate_output.name,
                up_output.name,
            ));
        }
        ProjectionBackend::Wmma => {
            out.push_str(&format!(
                "        if ({prefix}_rows >= 64) {{\n            {kernel_name}_gate<<<dim3((({output_features}) + 127) / 128, ({prefix}_rows + 127) / 128), dim3(256)>>>(\n                {}, {}, {}_data, {prefix}_rows, {reduction_len}, {input_row}, {output_features});\n            {kernel_name}_up<<<dim3((({output_features}) + 127) / 128, ({prefix}_rows + 127) / 128), dim3(256)>>>(\n                {}, {}, {}_data, {prefix}_rows, {reduction_len}, {input_row}, {output_features});\n        }} else {{\n            {kernel_name}_simt<<<dim3({output_len}), dim3(256)>>>(\n                {}, {}, {}, {}_data, {}_data, {output_len}, {reduction_len}, {input_row}, {output_features});\n        }}\n",
                value_expr(input),
                value_expr(gate_weight),
                gate_output.name,
                value_expr(input),
                value_expr(up_weight),
                up_output.name,
                value_expr(input),
                value_expr(gate_weight),
                value_expr(up_weight),
                gate_output.name,
                up_output.name,
            ));
        }
        ProjectionBackend::Blas => {
            out.push_str(&format!(
                "        if ({prefix}_rows >= 64) {{\n            catena_assert(({output_features}) <= INT32_MAX && {prefix}_rows <= INT32_MAX && ({reduction_len}) <= INT32_MAX);\n            const float {prefix}_alpha = 1.0f;\n            const float {prefix}_beta = 0.0f;\n"
            ));
            for (weight, output) in [(gate_weight, gate_output), (up_weight, up_output)] {
                out.push_str(&format!(
                    "            catena_host_blas_gemm(\n                {}, {} + ({input_row}) * ({reduction_len}), {}_data,\n                (int)({output_features}), (int){prefix}_rows, (int)({reduction_len}),\n                &{prefix}_alpha, &{prefix}_beta);\n",
                    value_expr(weight),
                    value_expr(input),
                    output.name,
                ));
            }
            out.push_str(&format!(
                "        }} else {{\n            {kernel_name}_simt<<<dim3({output_len}), dim3(256)>>>(\n                {}, {}, {}, {}_data, {}_data, {output_len}, {reduction_len}, {input_row}, {output_features});\n        }}\n",
                value_expr(input),
                value_expr(gate_weight),
                value_expr(up_weight),
                gate_output.name,
                up_output.name,
            ));
        }
    }
    out.push_str("    }\n");
    out.push_str(&format!(
        "    {} = {};\n    {} = {}_data;\n    {} = {}_data;\n",
        input_output.name,
        value_expr(input),
        gate_output.name,
        gate_output.name,
        up_output.name,
        up_output.name,
    ));
    Ok(())
}

pub(in crate::codegen) fn render_borrow_routed_native_bf16_gemv_pair_call(
    out: &mut String,
    function: &GpuFunction,
    assignment: &GpuAssign,
    dialect: GpuDialect,
    backend: ProjectionBackend,
) -> Result<(), GpuRenderError> {
    let [input_output, selected_output, gate_output, up_output] = assignment.outputs.as_slice()
    else {
        return Err(invalid_outputs(assignment, 4));
    };
    let (
        input,
        input_capacity,
        gate_weight,
        gate_capacity,
        up_weight,
        up_capacity,
        selected,
        selected_capacity,
        output_len,
        reduction_len,
        slots,
        output_features,
    ) = borrow_routed_bf16_gemv_pair_parts(assignment)?;
    let input_capacity = value_expr(input_capacity);
    let gate_capacity = value_expr(gate_capacity);
    let up_capacity = value_expr(up_capacity);
    let selected_capacity = value_expr(selected_capacity);
    let output_len = value_expr(output_len);
    let reduction_len = value_expr(reduction_len);
    let slots = value_expr(slots);
    let output_features = value_expr(output_features);
    let kernel_name = kernel_name(&function.name, assignment)?;
    let prefix = &gate_output.name;

    out.push_str(&format!("    catena_assert(({reduction_len}) != 0);\n"));
    out.push_str(&format!(
        "    catena_assert(({slots}) != 0 && ({output_features}) != 0);\n"
    ));
    out.push_str(&format!(
        "    catena_assert(({input_capacity}) % ({reduction_len}) == 0);\n"
    ));
    out.push_str(&format!(
        "    uint64_t {prefix}_rows = ({input_capacity}) / ({reduction_len});\n"
    ));
    out.push_str(&format!(
        "    catena_assert(({selected_capacity}) == {prefix}_rows * ({slots}));\n"
    ));
    out.push_str(&format!(
        "    catena_assert(({output_len}) == ({selected_capacity}) * ({output_features}));\n"
    ));
    out.push_str(&format!(
        "    uint64_t {prefix}_expert_stride = ({output_features}) * ({reduction_len});\n"
    ));
    out.push_str(&format!("    catena_assert(({gate_capacity}) == ({up_capacity}) && ({gate_capacity}) % {prefix}_expert_stride == 0);\n"));
    out.push_str(&format!(
        "    uint64_t {prefix}_expert_count = ({gate_capacity}) / {prefix}_expert_stride;\n"
    ));
    out.push_str(&format!(
        "    catena_bf16_t *{}_data = nullptr;\n    catena_bf16_t *{}_data = nullptr;\n",
        gate_output.name, up_output.name
    ));
    out.push_str(&format!("    if (({output_len}) != 0) {{\n"));
    for output in [gate_output, up_output] {
        out.push_str(&format!("        catena_host_gpu_check({}((void **)&{}_data, ({output_len}) * sizeof(catena_bf16_t), nullptr));\n", dialect.device_alloc_async_fn(), output.name));
    }
    let simt_kernel_name = match backend {
        ProjectionBackend::Simt => kernel_name.clone(),
        ProjectionBackend::Wmma => format!("{kernel_name}_simt"),
        ProjectionBackend::Blas => unreachable!("routed BLAS projections are not supported"),
    };
    let simt_call = format!(
        "        const uint64_t {prefix}_max_blocks_x = UINT32_MAX / 256u;\n        uint64_t {prefix}_blocks_x = ({output_len}) < {prefix}_max_blocks_x ? ({output_len}) : {prefix}_max_blocks_x;\n        uint64_t {prefix}_blocks_y = ({output_len}) / {prefix}_blocks_x + ((({output_len}) % {prefix}_blocks_x) != 0);\n        catena_assert({prefix}_blocks_y <= UINT32_MAX);\n        {simt_kernel_name}<<<dim3((uint32_t){prefix}_blocks_x, (uint32_t){prefix}_blocks_y), dim3(256)>>>(\n            {}, {}, {}, {}, {}_data, {}_data, {output_len}, {reduction_len}, {slots}, {output_features}, {prefix}_expert_count);\n",
        value_expr(input),
        value_expr(gate_weight),
        value_expr(up_weight),
        value_expr(selected),
        gate_output.name,
        up_output.name,
    );
    if backend == ProjectionBackend::Wmma && dialect == GpuDialect::Hip {
        out.push_str(&format!(
            "        if ({prefix}_expert_count != 0 && {prefix}_expert_count <= {ROUTED_MAX_EXPERTS} && ({selected_capacity}) >= {prefix}_expert_count * {ROUTED_MIN_ROUTES_PER_EXPERT} && (({reduction_len}) & 15) == 0 && (({output_features}) & 15) == 0) {{\n            uint64_t *{prefix}_offsets = nullptr;\n            uint64_t *{prefix}_permutation = nullptr;\n            catena_host_gpu_check({alloc}((void **)&{prefix}_offsets, ({prefix}_expert_count + 1) * sizeof(uint64_t), nullptr));\n            catena_host_gpu_check({alloc}((void **)&{prefix}_permutation, ({selected_capacity}) * sizeof(uint64_t), nullptr));\n            {kernel_name}_group<<<dim3({prefix}_expert_count), dim3(32)>>>(\n                {}, {selected_capacity}, {prefix}_expert_count, {prefix}_offsets, {prefix}_permutation);\n            {kernel_name}<<<dim3((({output_features}) + 63) / 64, {prefix}_expert_count), dim3(128)>>>(\n                {}, {}, {}, {prefix}_offsets, {prefix}_permutation, {}_data, {}_data, {reduction_len}, {slots}, {output_features});\n            catena_host_gpu_check({free}({prefix}_offsets, nullptr));\n            catena_host_gpu_check({free}({prefix}_permutation, nullptr));\n        }} else {{\n{simt_call}        }}\n",
            value_expr(selected),
            value_expr(input),
            value_expr(gate_weight),
            value_expr(up_weight),
            gate_output.name,
            up_output.name,
            alloc = dialect.device_alloc_async_fn(),
            free = dialect.device_free_async_fn(),
        ));
    } else {
        out.push_str(&simt_call);
    }
    out.push_str("    }\n");
    out.push_str(&format!(
        "    {} = {};\n    {} = {};\n    {} = {}_data;\n    {} = {}_data;\n",
        input_output.name,
        value_expr(input),
        selected_output.name,
        value_expr(selected),
        gate_output.name,
        gate_output.name,
        up_output.name,
        up_output.name
    ));
    Ok(())
}

pub(in crate::codegen) fn render_borrow_routed_native_bf16_gemv_residual_call(
    out: &mut String,
    function: &GpuFunction,
    assignment: &GpuAssign,
    dialect: GpuDialect,
    backend: ProjectionBackend,
) -> Result<(), GpuRenderError> {
    let [active_output, selected_output, residual_output, output] = assignment.outputs.as_slice()
    else {
        return Err(invalid_outputs(assignment, 4));
    };
    let (
        active,
        active_capacity,
        down,
        down_capacity,
        selected,
        selected_capacity,
        residual,
        residual_capacity,
        output_len,
        reduction_len,
        slots,
        intermediate,
        output_features,
    ) = borrow_routed_bf16_gemv_residual_parts(assignment)?;
    let active_capacity = value_expr(active_capacity);
    let down_capacity = value_expr(down_capacity);
    let selected_capacity = value_expr(selected_capacity);
    let residual_capacity = value_expr(residual_capacity);
    let output_len = value_expr(output_len);
    let reduction_len = value_expr(reduction_len);
    let slots = value_expr(slots);
    let intermediate = value_expr(intermediate);
    let output_features = value_expr(output_features);
    let kernel_name = kernel_name(&function.name, assignment)?;
    let prefix = &output.name;
    out.push_str(&format!(
        "    catena_assert(({reduction_len}) == ({slots}) * ({intermediate}));\n"
    ));
    out.push_str(&format!(
        "    catena_assert(({active_capacity}) % ({reduction_len}) == 0);\n"
    ));
    out.push_str(&format!(
        "    uint64_t {prefix}_rows = ({active_capacity}) / ({reduction_len});\n"
    ));
    out.push_str(&format!(
        "    catena_assert(({selected_capacity}) == {prefix}_rows * ({slots}));\n"
    ));
    out.push_str(&format!("    catena_assert(({output_len}) == {prefix}_rows * ({output_features}) && ({residual_capacity}) == ({output_len}));\n"));
    out.push_str(&format!(
        "    uint64_t {prefix}_expert_stride = ({output_features}) * ({intermediate});\n"
    ));
    out.push_str(&format!(
        "    catena_assert(({down_capacity}) % {prefix}_expert_stride == 0);\n"
    ));
    out.push_str(&format!(
        "    uint64_t {prefix}_expert_count = ({down_capacity}) / {prefix}_expert_stride;\n"
    ));
    out.push_str(&format!(
        "    catena_assert(({output_features}) == 0 || ({selected_capacity}) <= UINT64_MAX / ({output_features}));\n"
    ));
    out.push_str(&format!(
        "    catena_bf16_t *{}_data = nullptr;\n",
        output.name
    ));
    out.push_str(&format!("    if (({output_len}) != 0) {{\n        catena_host_gpu_check({}((void **)&{}_data, ({output_len}) * sizeof(catena_bf16_t), nullptr));\n", dialect.device_alloc_async_fn(), output.name));
    let simt_kernel_name = match backend {
        ProjectionBackend::Simt => kernel_name.clone(),
        ProjectionBackend::Wmma => format!("{kernel_name}_simt"),
        ProjectionBackend::Blas => unreachable!("routed BLAS projections are not supported"),
    };
    let simt_call = format!(
        "        const uint64_t {prefix}_max_blocks_x = UINT32_MAX / 256u;\n        uint64_t {prefix}_blocks_x = ({output_len}) < {prefix}_max_blocks_x ? ({output_len}) : {prefix}_max_blocks_x;\n        uint64_t {prefix}_blocks_y = ({output_len}) / {prefix}_blocks_x + ((({output_len}) % {prefix}_blocks_x) != 0);\n        catena_assert({prefix}_blocks_y <= UINT32_MAX);\n        {simt_kernel_name}<<<dim3((uint32_t){prefix}_blocks_x, (uint32_t){prefix}_blocks_y), dim3(256)>>>(\n            {}, {}, {}, {}, {}_data, {output_len}, {reduction_len}, {slots}, {intermediate}, {output_features}, {prefix}_expert_count);\n",
        value_expr(active),
        value_expr(down),
        value_expr(selected),
        value_expr(residual),
        output.name,
    );
    if backend == ProjectionBackend::Wmma && dialect == GpuDialect::Hip {
        out.push_str(&format!(
            "        if ({prefix}_expert_count != 0 && {prefix}_expert_count <= {ROUTED_MAX_EXPERTS} && ({selected_capacity}) >= {prefix}_expert_count * {ROUTED_MIN_ROUTES_PER_EXPERT} && (({intermediate}) & 15) == 0 && (({output_features}) & 15) == 0) {{\n            uint64_t *{prefix}_offsets = nullptr;\n            uint64_t *{prefix}_permutation = nullptr;\n            float *{prefix}_route_output = nullptr;\n            uint64_t {prefix}_route_output_len = ({selected_capacity}) * ({output_features});\n            catena_host_gpu_check({alloc}((void **)&{prefix}_offsets, ({prefix}_expert_count + 1) * sizeof(uint64_t), nullptr));\n            catena_host_gpu_check({alloc}((void **)&{prefix}_permutation, ({selected_capacity}) * sizeof(uint64_t), nullptr));\n            catena_host_gpu_check({alloc}((void **)&{prefix}_route_output, {prefix}_route_output_len * sizeof(float), nullptr));\n            {kernel_name}_group<<<dim3({prefix}_expert_count), dim3(32)>>>(\n                {}, {selected_capacity}, {prefix}_expert_count, {prefix}_offsets, {prefix}_permutation);\n            {kernel_name}<<<dim3((({output_features}) + 127) / 128, {prefix}_expert_count), dim3(128)>>>(\n                {}, {}, {prefix}_offsets, {prefix}_permutation, {prefix}_route_output, {intermediate}, {output_features});\n            {kernel_name}_combine<<<dim3((({output_len}) + 255) / 256), dim3(256)>>>(\n                {prefix}_route_output, {}, {}_data, {output_len}, {slots}, {output_features});\n            catena_host_gpu_check({free}({prefix}_offsets, nullptr));\n            catena_host_gpu_check({free}({prefix}_permutation, nullptr));\n            catena_host_gpu_check({free}({prefix}_route_output, nullptr));\n        }} else {{\n{simt_call}        }}\n",
            value_expr(selected),
            value_expr(active),
            value_expr(down),
            value_expr(residual),
            output.name,
            alloc = dialect.device_alloc_async_fn(),
            free = dialect.device_free_async_fn(),
        ));
    } else {
        out.push_str(&simt_call);
    }
    out.push_str("    }\n");
    out.push_str(&format!(
        "    {} = {};\n    {} = {};\n    {} = {};\n    {} = {}_data;\n",
        active_output.name,
        value_expr(active),
        selected_output.name,
        value_expr(selected),
        residual_output.name,
        value_expr(residual),
        output.name,
        output.name
    ));
    Ok(())
}

pub(in crate::codegen) fn render_bf16_gemv_call(
    out: &mut String,
    function: &GpuFunction,
    assignment: &GpuAssign,
    dialect: GpuDialect,
) -> Result<(), GpuRenderError> {
    let [output] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 1));
    };
    let (
        input,
        input_capacity,
        weight,
        weight_capacity,
        output_len,
        reduction_len,
        input_row,
        output_features,
    ) = bf16_gemv_parts(assignment)?;
    let input_capacity = value_expr(input_capacity);
    let weight_capacity = value_expr(weight_capacity);
    let output_len = value_expr(output_len);
    let reduction_len = value_expr(reduction_len);
    let input_row = value_expr(input_row);
    let output_features = value_expr(output_features);
    let kernel_name = kernel_name(&function.name, assignment)?;

    out.push_str(&format!(
        "    uint64_t {name}_len = {output_len};\n    catena_bf16_t *{name}_data = nullptr;\n",
        name = output.name
    ));
    out.push_str(&format!("    catena_assert(({reduction_len}) != 0);\n"));
    out.push_str(&format!(
        "    catena_assert(({input_capacity}) % ({reduction_len}) == 0);\n"
    ));
    out.push_str(&format!(
        "    uint64_t {name}_input_rows = ({input_capacity}) / ({reduction_len});\n",
        name = output.name
    ));
    out.push_str(&format!("    catena_assert(({output_features}) != 0);\n"));
    out.push_str(&format!(
        "    catena_assert({name}_len % ({output_features}) == 0);\n",
        name = output.name
    ));
    out.push_str(&format!(
        "    uint64_t {name}_rows = {name}_len / ({output_features});\n",
        name = output.name
    ));
    out.push_str(&format!(
        "    catena_assert(({input_row}) <= {name}_input_rows && {name}_rows <= {name}_input_rows - ({input_row}));\n",
        name = output.name
    ));
    out.push_str(&format!(
        "    catena_assert(({output_features}) <= UINT64_MAX / ({reduction_len}));\n"
    ));
    out.push_str(&format!(
        "    catena_assert(({weight_capacity}) == ({output_features}) * ({reduction_len}));\n"
    ));
    out.push_str(&format!(
        "    if ({name}_len != 0) {{\n",
        name = output.name
    ));
    out.push_str(&format!(
        "        catena_host_gpu_check({}((void **)&{name}_data, {name}_len * sizeof(catena_bf16_t), nullptr));\n",
        dialect.device_alloc_async_fn(),
        name = output.name
    ));
    out.push_str(&format!(
        "        {kernel_name}<<<dim3({name}_len), dim3(256)>>>(\n            {}, {}, {name}_data, {name}_len, {reduction_len}, {input_row}, {output_features});\n",
        value_expr(input),
        value_expr(weight),
        name = output.name
    ));
    out.push_str("    }\n");
    out.push_str(&format!("    {} = {}_data;\n", output.name, output.name));
    Ok(())
}

pub(in crate::codegen) fn render_bf16_gemm_wmma_call(
    out: &mut String,
    function: &GpuFunction,
    assignment: &GpuAssign,
    dialect: GpuDialect,
) -> Result<(), GpuRenderError> {
    let [output] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 1));
    };
    let (
        input,
        input_capacity,
        weight,
        weight_capacity,
        output_len,
        reduction_len,
        input_row,
        output_features,
    ) = bf16_gemv_parts(assignment)?;
    let input_capacity = value_expr(input_capacity);
    let weight_capacity = value_expr(weight_capacity);
    let output_len = value_expr(output_len);
    let reduction_len = value_expr(reduction_len);
    let input_row = value_expr(input_row);
    let output_features = value_expr(output_features);
    let kernel_name = kernel_name(&function.name, assignment)?;
    let name = &output.name;
    out.push_str(&format!(
        "    uint64_t {name}_len = {output_len};\n    catena_bf16_t *{name}_data = nullptr;\n"
    ));
    out.push_str(&format!("    catena_assert(({reduction_len}) != 0);\n    catena_assert(({input_capacity}) % ({reduction_len}) == 0);\n    uint64_t {name}_input_rows = ({input_capacity}) / ({reduction_len});\n    catena_assert(({output_features}) != 0);\n    catena_assert({name}_len % ({output_features}) == 0);\n    uint64_t {name}_rows = {name}_len / ({output_features});\n"));
    out.push_str(&format!("    catena_assert(({input_row}) <= {name}_input_rows && {name}_rows <= {name}_input_rows - ({input_row}));\n    catena_assert(({output_features}) <= UINT64_MAX / ({reduction_len}));\n    catena_assert(({weight_capacity}) == ({output_features}) * ({reduction_len}));\n"));
    out.push_str(&format!("    if ({name}_len != 0) {{\n        catena_host_gpu_check({}((void **)&{name}_data, {name}_len * sizeof(catena_bf16_t), nullptr));\n", dialect.device_alloc_async_fn()));
    out.push_str(&format!("        if ({name}_rows >= 64) {{\n            {kernel_name}<<<dim3((({output_features}) + 127) / 128, ({name}_rows + 127) / 128), dim3(256)>>>(\n                {}, {}, {name}_data, {name}_rows, {reduction_len}, {input_row}, {output_features});\n        }} else\n", value_expr(input), value_expr(weight)));
    out.push_str(&format!("        {{\n            {kernel_name}_simt<<<dim3({name}_len), dim3(256)>>>(\n                {}, {}, {name}_data, {name}_len, {reduction_len}, {input_row}, {output_features});\n        }}\n    }}\n    {name} = {name}_data;\n", value_expr(input), value_expr(weight)));
    Ok(())
}

pub(in crate::codegen) fn render_bf16_gemm_blas_call(
    out: &mut String,
    function: &GpuFunction,
    assignment: &GpuAssign,
    dialect: GpuDialect,
) -> Result<(), GpuRenderError> {
    let [output] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 1));
    };
    let (
        input,
        input_capacity,
        weight,
        weight_capacity,
        output_len,
        reduction_len,
        input_row,
        output_features,
    ) = bf16_gemv_parts(assignment)?;
    let kernel_name = kernel_name(&function.name, assignment)?;
    let name = &output.name;

    out.push_str(&format!(
        r#"    uint64_t {name}_len = {output_len};
    catena_bf16_t *{name}_data = nullptr;
    catena_assert(({reduction_len}) != 0);
    catena_assert(({input_capacity}) % ({reduction_len}) == 0);
    uint64_t {name}_input_rows = ({input_capacity}) / ({reduction_len});
    catena_assert(({output_features}) != 0);
    catena_assert({name}_len % ({output_features}) == 0);
    uint64_t {name}_rows = {name}_len / ({output_features});
    catena_assert(({input_row}) <= {name}_input_rows && {name}_rows <= {name}_input_rows - ({input_row}));
    catena_assert(({output_features}) <= UINT64_MAX / ({reduction_len}));
    catena_assert(({weight_capacity}) == ({output_features}) * ({reduction_len}));
    if ({name}_len != 0) {{
        catena_host_gpu_check({device_alloc}((void **)&{name}_data, {name}_len * sizeof(catena_bf16_t), nullptr));
        if ({name}_rows >= 64) {{
            catena_assert(({output_features}) <= INT32_MAX && {name}_rows <= INT32_MAX && ({reduction_len}) <= INT32_MAX);
            const float {name}_alpha = 1.0f;
            const float {name}_beta = 0.0f;
            catena_host_blas_gemm(
                {weight}, {input} + ({input_row}) * ({reduction_len}), {name}_data,
                (int)({output_features}), (int){name}_rows, (int)({reduction_len}),
                &{name}_alpha, &{name}_beta);
        }} else {{
            {kernel_name}_simt<<<dim3({name}_len), dim3(256)>>>(
                {input}, {weight}, {name}_data, {name}_len, {reduction_len}, {input_row}, {output_features});
        }}
    }}
    {name} = {name}_data;
"#,
        input = value_expr(input),
        input_capacity = value_expr(input_capacity),
        weight = value_expr(weight),
        weight_capacity = value_expr(weight_capacity),
        output_len = value_expr(output_len),
        reduction_len = value_expr(reduction_len),
        input_row = value_expr(input_row),
        output_features = value_expr(output_features),
        device_alloc = dialect.device_alloc_async_fn(),
    ));
    Ok(())
}

pub(in crate::codegen) fn render_reduce_f32_call(
    out: &mut String,
    function: &GpuFunction,
    assignment: &GpuAssign,
    dialect: GpuDialect,
) -> Result<(), GpuRenderError> {
    let [output] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 1));
    };
    let Some(CType::Pointer(output_element)) = runtime_type(output) else {
        return Err(GpuRenderError::ErasedType(output.clone()));
    };
    let (term, _epilogue, output_len, reduction_len, env) = reduce_f32_parts(assignment)?;
    let _ = term;
    let output_len = value_expr(output_len);
    let reduction_len = value_expr(reduction_len);
    let kernel_name = kernel_name(&function.name, assignment)?;

    out.push_str(&format!(
        "    uint64_t {name}_len = {output_len};\n    {element} *{name}_data = nullptr;\n",
        name = output.name,
        element = c_type(output_element)
    ));
    out.push_str(&format!("    catena_assert(({reduction_len}) <= 16384);\n"));
    out.push_str(&format!("    if ({}_len != 0) {{\n", output.name));
    out.push_str(&format!(
        "        catena_host_gpu_check({}((void **)&{}_data, {}_len * sizeof({}), nullptr));\n",
        dialect.device_alloc_async_fn(),
        output.name,
        output.name,
        c_type(output_element)
    ));
    out.push_str(&format!(
        "        {kernel_name}<<<dim3({}_len), dim3(({reduction_len}) > 1024 ? 256 : 64), ({reduction_len}) * sizeof(float)>>>(\n            {}_data, {}_len, {reduction_len}",
        output.name, output.name, output.name
    ));
    for arg in runtime_values(env) {
        if let GpuValue::Var(var) = arg {
            out.push_str(", ");
            out.push_str(&var.name);
        }
    }
    out.push_str(");\n");
    out.push_str("    }\n");
    out.push_str(&format!("    {} = {}_data;\n", output.name, output.name));
    Ok(())
}

pub(in crate::codegen) fn render_reduce_f32_pair_call(
    out: &mut String,
    function: &GpuFunction,
    assignment: &GpuAssign,
    dialect: GpuDialect,
) -> Result<(), GpuRenderError> {
    let [left_output, right_output] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 2));
    };
    let (_, _, output_len, reduction_len, env) = reduce_f32_pair_parts(assignment)?;
    let output_len = value_expr(output_len);
    let reduction_len = value_expr(reduction_len);
    let kernel_name = kernel_name(&function.name, assignment)?;

    for output in [left_output, right_output] {
        out.push_str(&format!(
            "    uint64_t {name}_len = {output_len};\n    float *{name}_data = nullptr;\n",
            name = output.name
        ));
    }
    out.push_str(&format!("    catena_assert(({reduction_len}) <= 8192);\n"));
    out.push_str(&format!("    if ({output_len} != 0) {{\n"));
    for output in [left_output, right_output] {
        out.push_str(&format!(
            "        catena_host_gpu_check({}((void **)&{}_data, ({output_len}) * sizeof(float), nullptr));\n",
            dialect.device_alloc_async_fn(),
            output.name
        ));
    }
    out.push_str(&format!(
        "        {kernel_name}<<<dim3({output_len}), dim3(({reduction_len}) > 1024 ? 256 : 64), 2 * ({reduction_len}) * sizeof(float)>>>(\n            {}_data, {}_data, {output_len}, {reduction_len}",
        left_output.name, right_output.name
    ));
    for arg in runtime_values(env) {
        if let GpuValue::Var(var) = arg {
            out.push_str(", ");
            out.push_str(&var.name);
        }
    }
    out.push_str(");\n");
    out.push_str("    }\n");
    out.push_str(&format!(
        "    {} = {}_data;\n    {} = {}_data;\n",
        left_output.name, left_output.name, right_output.name, right_output.name
    ));
    Ok(())
}

pub(in crate::codegen) fn render_softmax_f32_kernel(
    out: &mut String,
    kernel_name: &str,
    assignment: &GpuAssign,
) -> Result<(), GpuRenderError> {
    let [output] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 1));
    };
    let Some(CType::Pointer(element)) = runtime_type(output) else {
        return Err(GpuRenderError::ErasedType(output.clone()));
    };
    if !matches!(element.as_ref(), CType::F32 | CType::BF16) {
        return Err(GpuRenderError::UnsupportedType(
            runtime_type(output).unwrap().clone(),
        ));
    }
    let (exponential, _buffer, _capacity, _columns) = softmax_f32_parts(assignment)?;

    out.push_str(&format!(
        "__global__ void {kernel_name}({} *data, uint64_t rows, uint64_t columns) {{\n",
        c_type(element)
    ));
    out.push_str("    uint64_t row = (uint64_t)blockIdx.x;\n");
    out.push_str("    if (row >= rows) { return; }\n");
    out.push_str("    uint64_t row_offset = row * columns;\n");
    out.push_str("    extern __shared__ float reduction_terms[];\n");
    out.push_str(
        "    for (uint64_t column = (uint64_t)threadIdx.x; column < columns; column += (uint64_t)blockDim.x) {\n",
    );
    let load = match element.as_ref() {
        CType::F32 => "data[row_offset + column]",
        CType::BF16 => "catena_bf16_to_f32(data[row_offset + column])",
        _ => unreachable!("element type was validated"),
    };
    out.push_str(&format!("        reduction_terms[column] = {load};\n"));
    out.push_str("    }\n");
    out.push_str("    __syncthreads();\n");
    out.push_str("    for (uint64_t stride = 1; stride < columns; stride <<= 1) {\n");
    out.push_str("        uint64_t step = stride << 1;\n");
    out.push_str(
        "        for (uint64_t left = (uint64_t)threadIdx.x * step; left + stride < columns; left += (uint64_t)blockDim.x * step) {\n",
    );
    out.push_str("            float left_value = reduction_terms[left];\n");
    out.push_str("            float right_value = reduction_terms[left + stride];\n");
    out.push_str(
        "            reduction_terms[left] = right_value > left_value ? right_value : left_value;\n",
    );
    out.push_str("        }\n");
    out.push_str("        __syncthreads();\n");
    out.push_str("    }\n");
    out.push_str("    float row_maximum = reduction_terms[0];\n");
    out.push_str("    __syncthreads();\n");
    out.push_str(
        "    for (uint64_t column = (uint64_t)threadIdx.x; column < columns; column += (uint64_t)blockDim.x) {\n",
    );
    out.push_str(&format!("        float score = {load};\n"));
    out.push_str("        float numerator;\n");
    out.push_str("        if (catena_f32_bitcast_u32(score) == 0xFF800000u) {\n");
    out.push_str("            numerator = 0.0f;\n");
    out.push_str("        } else {\n");
    out.push_str("            float shifted = score - row_maximum;\n");
    let exponential_input = GpuValue::Var(GpuVar {
        node: output.node,
        name: "shifted".to_string(),
        lowered: LoweredType::Runtime(CType::F32),
    });
    let exponential_output = GpuVar {
        node: output.node,
        name: "numerator".to_string(),
        lowered: LoweredType::Runtime(CType::F32),
    };
    render_function_application(
        out,
        "            ",
        exponential,
        &[exponential_input],
        &[exponential_output],
    )?;
    out.push_str("        }\n");
    out.push_str("        reduction_terms[column] = numerator;\n");
    if element.as_ref() == &CType::F32 {
        out.push_str("        data[row_offset + column] = numerator;\n");
    }
    out.push_str("    }\n");
    out.push_str("    __syncthreads();\n");
    out.push_str("    for (uint64_t stride = 1; stride < columns; stride <<= 1) {\n");
    out.push_str("        uint64_t step = stride << 1;\n");
    out.push_str(
        "        for (uint64_t left = (uint64_t)threadIdx.x * step; left + stride < columns; left += (uint64_t)blockDim.x * step) {\n",
    );
    out.push_str(
        "            reduction_terms[left] = reduction_terms[left] + reduction_terms[left + stride];\n",
    );
    out.push_str("        }\n");
    out.push_str("        __syncthreads();\n");
    out.push_str("    }\n");
    out.push_str("    float denominator = reduction_terms[0];\n");
    out.push_str(
        "    for (uint64_t column = (uint64_t)threadIdx.x; column < columns; column += (uint64_t)blockDim.x) {\n",
    );
    out.push_str("        uint64_t index = row_offset + column;\n");
    match element.as_ref() {
        CType::F32 => {
            out.push_str(
                "        data[index] = denominator == 0.0f ? 0.0f : data[index] / denominator;\n",
            );
        }
        CType::BF16 => {
            out.push_str("        float score = catena_bf16_to_f32(data[index]);\n");
            out.push_str("        float numerator;\n");
            out.push_str("        if (catena_f32_bitcast_u32(score) == 0xFF800000u) {\n");
            out.push_str("            numerator = 0.0f;\n");
            out.push_str("        } else {\n");
            out.push_str("            float shifted = score - row_maximum;\n");
            render_function_application(
                out,
                "            ",
                exponential,
                &[GpuValue::Var(GpuVar {
                    node: output.node,
                    name: "shifted".to_string(),
                    lowered: LoweredType::Runtime(CType::F32),
                })],
                &[GpuVar {
                    node: output.node,
                    name: "numerator".to_string(),
                    lowered: LoweredType::Runtime(CType::F32),
                }],
            )?;
            out.push_str("        }\n");
            out.push_str("        float probability = denominator == 0.0f ? 0.0f : numerator / denominator;\n");
            out.push_str("        data[index] = catena_bf16_from_f32(probability);\n");
        }
        _ => unreachable!("element type was validated"),
    }
    out.push_str("    }\n");
    out.push_str("}\n");
    Ok(())
}

pub(in crate::codegen) fn render_softmax_f32_call(
    out: &mut String,
    function: &GpuFunction,
    assignment: &GpuAssign,
) -> Result<(), GpuRenderError> {
    let [output] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 1));
    };
    let (_exponential, buffer, capacity, columns) = softmax_f32_parts(assignment)?;
    let capacity = value_expr(capacity);
    let columns = value_expr(columns);
    let kernel_name = kernel_name(&function.name, assignment)?;

    out.push_str(&format!("    catena_assert(({columns}) != 0);\n"));
    out.push_str(&format!("    catena_assert(({columns}) <= 16384);\n"));
    out.push_str(&format!(
        "    catena_assert(({capacity}) % ({columns}) == 0);\n"
    ));
    out.push_str(&format!("    if (({capacity}) != 0) {{\n"));
    out.push_str(&format!(
        "        {kernel_name}<<<dim3(({capacity}) / ({columns})), dim3(({columns}) > 1024 ? 256 : 64), ({columns}) * sizeof(float)>>>(\n            {}, ({capacity}) / ({columns}), {columns});\n",
        value_expr(buffer)
    ));
    out.push_str("    }\n");
    out.push_str(&format!("    {} = {};\n", output.name, value_expr(buffer)));
    Ok(())
}

pub(in crate::codegen) fn kernel_name(
    function_name: &str,
    assignment: &GpuAssign,
) -> Result<String, GpuRenderError> {
    let Some(output) = assignment.outputs.last() else {
        return Err(invalid_outputs(assignment, 1));
    };
    Ok(format!("materialize_{}_{}", function_name, output.name))
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

fn into_parts(
    assignment: &GpuAssign,
) -> Result<
    (
        &GpuValue,
        &GpuValue,
        &GpuValue,
        &GpuValue,
        &GpuValue,
        Component<'_>,
    ),
    GpuRenderError,
> {
    let components = input_components(assignment)?;
    let [buffer, capacity, offset, len, env, func] = components.as_slice() else {
        return Err(GpuRenderError::InvalidInputComponentCount {
            op: assignment.op.clone(),
            expected: 6,
            actual: components.len(),
        });
    };
    Ok((
        single_function(func).map_err(|error| {
            invalid_component_count(
                assignment,
                "producer_fn",
                "materializec.into producer function symbol",
                error.actual,
            )
        })?,
        single_runtime_input(assignment, "buffer", buffer)?,
        single_runtime_input(assignment, "capacity", capacity)?,
        single_runtime_input(assignment, "offset", offset)?,
        single_runtime_input(assignment, "len", len)?,
        env,
    ))
}

fn borrow_parts(
    assignment: &GpuAssign,
) -> Result<(&GpuValue, &GpuValue, &GpuValue, &GpuValue, Component<'_>), GpuRenderError> {
    let components = input_components(assignment)?;
    let [buffer, capacity, len, env, func] = components.as_slice() else {
        return Err(GpuRenderError::InvalidInputComponentCount {
            op: assignment.op.clone(),
            expected: 5,
            actual: components.len(),
        });
    };
    Ok((
        single_function(func).map_err(|error| {
            invalid_component_count(
                assignment,
                "producer_fn",
                "materializec.borrow producer function symbol",
                error.actual,
            )
        })?,
        single_runtime_input(assignment, "buffer", buffer)?,
        single_runtime_input(assignment, "capacity", capacity)?,
        single_runtime_input(assignment, "len", len)?,
        env,
    ))
}

fn borrow_reduce_f32_parts(
    assignment: &GpuAssign,
) -> Result<
    (
        &GpuValue,
        &GpuValue,
        &GpuValue,
        &GpuValue,
        &GpuValue,
        &GpuValue,
        Component<'_>,
    ),
    GpuRenderError,
> {
    let components = input_components(assignment)?;
    let [
        cache,
        capacity,
        output_len,
        reduction_len,
        env,
        term,
        epilogue,
    ] = components.as_slice()
    else {
        return Err(GpuRenderError::InvalidInputComponentCount {
            op: assignment.op.clone(),
            expected: 7,
            actual: components.len(),
        });
    };
    Ok((
        single_function(term).map_err(|error| {
            invalid_component_count(
                assignment,
                "term_fn",
                "materializec.borrow-reduce-f32 term function symbol",
                error.actual,
            )
        })?,
        single_function(epilogue).map_err(|error| {
            invalid_component_count(
                assignment,
                "epilogue_fn",
                "materializec.borrow-reduce-f32 epilogue function symbol",
                error.actual,
            )
        })?,
        single_runtime_input(assignment, "cache", cache)?,
        single_runtime_input(assignment, "capacity", capacity)?,
        single_runtime_input(assignment, "output_len", output_len)?,
        single_runtime_input(assignment, "reduction_len", reduction_len)?,
        env,
    ))
}

fn borrow_native_bf16_gemv_pair_parts(
    assignment: &GpuAssign,
) -> Result<
    (
        &GpuValue,
        &GpuValue,
        &GpuValue,
        &GpuValue,
        &GpuValue,
        &GpuValue,
        &GpuValue,
        &GpuValue,
        &GpuValue,
        &GpuValue,
    ),
    GpuRenderError,
> {
    let components = input_components(assignment)?;
    let [
        input,
        input_capacity,
        gate_weight,
        gate_capacity,
        up_weight,
        up_capacity,
        output_len,
        reduction_len,
        input_row,
        output_features,
    ] = components.as_slice()
    else {
        return Err(GpuRenderError::InvalidInputComponentCount {
            op: assignment.op.clone(),
            expected: 10,
            actual: components.len(),
        });
    };
    Ok((
        single_runtime_input(assignment, "input", input)?,
        single_runtime_input(assignment, "input_capacity", input_capacity)?,
        single_runtime_input(assignment, "gate_weight", gate_weight)?,
        single_runtime_input(assignment, "gate_capacity", gate_capacity)?,
        single_runtime_input(assignment, "up_weight", up_weight)?,
        single_runtime_input(assignment, "up_capacity", up_capacity)?,
        single_runtime_input(assignment, "output_len", output_len)?,
        single_runtime_input(assignment, "reduction_len", reduction_len)?,
        single_runtime_input(assignment, "input_row", input_row)?,
        single_runtime_input(assignment, "output_features", output_features)?,
    ))
}

fn borrow_routed_bf16_gemv_pair_parts(
    assignment: &GpuAssign,
) -> Result<
    (
        &GpuValue,
        &GpuValue,
        &GpuValue,
        &GpuValue,
        &GpuValue,
        &GpuValue,
        &GpuValue,
        &GpuValue,
        &GpuValue,
        &GpuValue,
        &GpuValue,
        &GpuValue,
    ),
    GpuRenderError,
> {
    let components = input_components(assignment)?;
    let [
        input,
        input_capacity,
        gate_weight,
        gate_capacity,
        up_weight,
        up_capacity,
        selected,
        selected_capacity,
        output_len,
        reduction_len,
        slots,
        output_features,
    ] = components.as_slice()
    else {
        return Err(GpuRenderError::InvalidInputComponentCount {
            op: assignment.op.clone(),
            expected: 12,
            actual: components.len(),
        });
    };
    Ok((
        single_runtime_input(assignment, "input", input)?,
        single_runtime_input(assignment, "input_capacity", input_capacity)?,
        single_runtime_input(assignment, "gate_weight", gate_weight)?,
        single_runtime_input(assignment, "gate_capacity", gate_capacity)?,
        single_runtime_input(assignment, "up_weight", up_weight)?,
        single_runtime_input(assignment, "up_capacity", up_capacity)?,
        single_runtime_input(assignment, "selected", selected)?,
        single_runtime_input(assignment, "selected_capacity", selected_capacity)?,
        single_runtime_input(assignment, "output_len", output_len)?,
        single_runtime_input(assignment, "reduction_len", reduction_len)?,
        single_runtime_input(assignment, "slots", slots)?,
        single_runtime_input(assignment, "output_features", output_features)?,
    ))
}

fn borrow_routed_bf16_gemv_residual_parts(
    assignment: &GpuAssign,
) -> Result<
    (
        &GpuValue,
        &GpuValue,
        &GpuValue,
        &GpuValue,
        &GpuValue,
        &GpuValue,
        &GpuValue,
        &GpuValue,
        &GpuValue,
        &GpuValue,
        &GpuValue,
        &GpuValue,
        &GpuValue,
    ),
    GpuRenderError,
> {
    let components = input_components(assignment)?;
    let [
        active,
        active_capacity,
        down,
        down_capacity,
        selected,
        selected_capacity,
        residual,
        residual_capacity,
        output_len,
        reduction_len,
        slots,
        intermediate,
        output_features,
    ] = components.as_slice()
    else {
        return Err(GpuRenderError::InvalidInputComponentCount {
            op: assignment.op.clone(),
            expected: 13,
            actual: components.len(),
        });
    };
    Ok((
        single_runtime_input(assignment, "active", active)?,
        single_runtime_input(assignment, "active_capacity", active_capacity)?,
        single_runtime_input(assignment, "down", down)?,
        single_runtime_input(assignment, "down_capacity", down_capacity)?,
        single_runtime_input(assignment, "selected", selected)?,
        single_runtime_input(assignment, "selected_capacity", selected_capacity)?,
        single_runtime_input(assignment, "residual", residual)?,
        single_runtime_input(assignment, "residual_capacity", residual_capacity)?,
        single_runtime_input(assignment, "output_len", output_len)?,
        single_runtime_input(assignment, "reduction_len", reduction_len)?,
        single_runtime_input(assignment, "slots", slots)?,
        single_runtime_input(assignment, "intermediate", intermediate)?,
        single_runtime_input(assignment, "output_features", output_features)?,
    ))
}

fn bf16_gemv_parts(
    assignment: &GpuAssign,
) -> Result<
    (
        &GpuValue,
        &GpuValue,
        &GpuValue,
        &GpuValue,
        &GpuValue,
        &GpuValue,
        &GpuValue,
        &GpuValue,
    ),
    GpuRenderError,
> {
    let components = input_components(assignment)?;
    let [
        input,
        input_capacity,
        weight,
        weight_capacity,
        output_len,
        reduction_len,
        input_row,
        output_features,
    ] = components.as_slice()
    else {
        return Err(GpuRenderError::InvalidInputComponentCount {
            op: assignment.op.clone(),
            expected: 8,
            actual: components.len(),
        });
    };
    Ok((
        single_runtime_input(assignment, "input", input)?,
        single_runtime_input(assignment, "input_capacity", input_capacity)?,
        single_runtime_input(assignment, "weight", weight)?,
        single_runtime_input(assignment, "weight_capacity", weight_capacity)?,
        single_runtime_input(assignment, "output_len", output_len)?,
        single_runtime_input(assignment, "reduction_len", reduction_len)?,
        single_runtime_input(assignment, "input_row", input_row)?,
        single_runtime_input(assignment, "output_features", output_features)?,
    ))
}

fn borrow_argmax_f32_parts(
    assignment: &GpuAssign,
) -> Result<(&GpuValue, &GpuValue, &GpuValue), GpuRenderError> {
    let components = input_components(assignment)?;
    let [source, len, output_len] = components.as_slice() else {
        return Err(GpuRenderError::InvalidInputComponentCount {
            op: assignment.op.clone(),
            expected: 3,
            actual: components.len(),
        });
    };
    Ok((
        single_runtime_input(assignment, "source", source)?,
        single_runtime_input(assignment, "len", len)?,
        single_runtime_input(assignment, "output_len", output_len)?,
    ))
}

fn borrow_topk_f32_parts(
    assignment: &GpuAssign,
) -> Result<
    (
        &GpuValue,
        &GpuValue,
        &GpuValue,
        &GpuValue,
        &GpuValue,
        &GpuValue,
    ),
    GpuRenderError,
> {
    let components = input_components(assignment)?;
    let [source, len, rows, output_len, columns, k] = components.as_slice() else {
        return Err(GpuRenderError::InvalidInputComponentCount {
            op: assignment.op.clone(),
            expected: 6,
            actual: components.len(),
        });
    };
    Ok((
        single_runtime_input(assignment, "source", source)?,
        single_runtime_input(assignment, "len", len)?,
        single_runtime_input(assignment, "rows", rows)?,
        single_runtime_input(assignment, "output_len", output_len)?,
        single_runtime_input(assignment, "columns", columns)?,
        single_runtime_input(assignment, "k", k)?,
    ))
}

fn reduce_f32_parts(
    assignment: &GpuAssign,
) -> Result<(&GpuValue, &GpuValue, &GpuValue, &GpuValue, Component<'_>), GpuRenderError> {
    let components = input_components(assignment)?;
    let [output_len, reduction_len, env, term, epilogue] = components.as_slice() else {
        return Err(GpuRenderError::InvalidInputComponentCount {
            op: assignment.op.clone(),
            expected: 5,
            actual: components.len(),
        });
    };
    Ok((
        single_function(term).map_err(|error| {
            invalid_component_count(
                assignment,
                "term_fn",
                "materializec.reduce-f32 term function symbol",
                error.actual,
            )
        })?,
        single_function(epilogue).map_err(|error| {
            invalid_component_count(
                assignment,
                "epilogue_fn",
                "materializec.reduce-f32 epilogue function symbol",
                error.actual,
            )
        })?,
        single_runtime_input(assignment, "output_len", output_len)?,
        single_runtime_input(assignment, "reduction_len", reduction_len)?,
        env,
    ))
}

fn reduce_f32_pair_parts(
    assignment: &GpuAssign,
) -> Result<(&GpuValue, &GpuValue, &GpuValue, &GpuValue, Component<'_>), GpuRenderError> {
    let components = input_components(assignment)?;
    let [output_len, reduction_len, env, term, epilogue] = components.as_slice() else {
        return Err(GpuRenderError::InvalidInputComponentCount {
            op: assignment.op.clone(),
            expected: 5,
            actual: components.len(),
        });
    };
    Ok((
        single_function(term).map_err(|error| {
            invalid_component_count(
                assignment,
                "term_fn",
                "materializec.reduce-f32-pair term function symbol",
                error.actual,
            )
        })?,
        single_function(epilogue).map_err(|error| {
            invalid_component_count(
                assignment,
                "epilogue_fn",
                "materializec.reduce-f32-pair epilogue function symbol",
                error.actual,
            )
        })?,
        single_runtime_input(assignment, "output_len", output_len)?,
        single_runtime_input(assignment, "reduction_len", reduction_len)?,
        env,
    ))
}

fn softmax_f32_parts(
    assignment: &GpuAssign,
) -> Result<(&GpuValue, &GpuValue, &GpuValue, &GpuValue), GpuRenderError> {
    let components = input_components(assignment)?;
    let [buffer, capacity, columns, exponential] = components.as_slice() else {
        return Err(GpuRenderError::InvalidInputComponentCount {
            op: assignment.op.clone(),
            expected: 4,
            actual: components.len(),
        });
    };
    Ok((
        single_function(exponential).map_err(|error| {
            invalid_component_count(
                assignment,
                "exponential_fn",
                "materializec.softmax-f32 exponential function symbol",
                error.actual,
            )
        })?,
        single_runtime_input(assignment, "buffer", buffer)?,
        single_runtime_input(assignment, "capacity", capacity)?,
        single_runtime_input(assignment, "columns", columns)?,
    ))
}

fn single_runtime_input<'a>(
    assignment: &GpuAssign,
    component: &'static str,
    values: Component<'a>,
) -> Result<&'a GpuValue, GpuRenderError> {
    single_value(values).map_err(|error| {
        invalid_component_count(assignment, component, "single runtime value", error.actual)
    })
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
    use crate::codegen::default::fn_ptrs::FnPtrSymbol;
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
        render_kernel(&mut out, "materialize_test", &assignment).unwrap();

        assert!(out.contains("value = i;"));
        assert!(!out.contains("program_ix_to_u64"));
    }

    #[test]
    fn optimized_projection_dispatch_contract_is_stable() {
        assert_eq!(ROUTED_MAX_EXPERTS, 256);
        assert_eq!(ROUTED_MIN_ROUTES_PER_EXPERT, 4);
    }
}
