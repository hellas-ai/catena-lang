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

use crate::codegen::{
    GpuAssign, GpuDialect, GpuFunction, GpuValue, GpuVar,
    components::{
        Component, input_components, runtime_values, single_function, single_value, value_expr,
    },
    gpu::{GpuRenderError, render_function_application},
    lower_types::{CType, LoweredType},
    render_utils::{c_type, invalid_outputs, param_decl},
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
    if element.as_ref() != &CType::F32 {
        return Err(GpuRenderError::UnsupportedType(
            runtime_type(output).unwrap().clone(),
        ));
    }
    let (term, epilogue, _output_len, _reduction_len, env) = reduce_f32_parts(assignment)?;

    out.push_str(&format!(
        "__global__ void {kernel_name}(float *out, uint64_t output_len, uint64_t reduction_len"
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
    out.push_str("        __syncthreads();\n");
    out.push_str("    }\n");
    out.push_str("    if (threadIdx.x == 0) {\n");
    out.push_str("        float reduction_sum = reduction_len == 0 ? 0.0f : reduction_terms[0];\n");
    out.push_str("        float value;\n");
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
        lowered: LoweredType::Runtime(CType::F32),
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
    if output_element.as_ref() != &CType::F32 {
        return Err(GpuRenderError::UnsupportedType(
            runtime_type(output).unwrap().clone(),
        ));
    }
    let (term, epilogue, _cache, _capacity, _output_len, _reduction_len, env) =
        borrow_reduce_f32_parts(assignment)?;

    out.push_str(&format!(
        "__global__ void {kernel_name}({} *borrowed, float *out, uint64_t output_len, uint64_t reduction_len",
        c_type(cache_element)
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
    out.push_str("        __syncthreads();\n");
    out.push_str("    }\n");
    out.push_str("    if (threadIdx.x == 0) {\n");
    out.push_str("        float reduction_sum = reduction_len == 0 ? 0.0f : reduction_terms[0];\n");
    out.push_str("        float value;\n");
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
        lowered: LoweredType::Runtime(CType::F32),
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
    let (_source, _len, _output_len) = borrow_argmax_f32_parts(assignment)?;

    out.push_str(&format!(
        "__global__ void {kernel_name}(const float *data, uint64_t len, uint64_t *out) {{\n"
    ));
    out.push_str("    __shared__ uint32_t best_keys[256];\n");
    out.push_str("    __shared__ uint64_t best_indices[256];\n");
    out.push_str("    uint32_t best_key = 0;\n");
    out.push_str("    uint64_t best_index = UINT64_MAX;\n");
    out.push_str(
        "    for (uint64_t index = (uint64_t)threadIdx.x; index < len; index += (uint64_t)blockDim.x) {\n",
    );
    out.push_str("        uint32_t bits = catena_f32_bitcast_u32(data[index]);\n");
    out.push_str(
        "        uint32_t key = (bits & 0x80000000u) != 0 ? ~bits : (bits ^ 0x80000000u);\n",
    );
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
    out.push_str(&format!("    catena_assert(({reduction_len}) <= 8192);\n"));
    out.push_str(&format!("    if ({}_len != 0) {{\n", output.name));
    out.push_str(&format!(
        "        catena_host_gpu_check({}((void **)&{}_data, {}_len * sizeof(float), nullptr));\n",
        dialect.device_alloc_async_fn(),
        output.name,
        output.name
    ));
    out.push_str(&format!(
        "        {kernel_name}<<<dim3({}_len), dim3(({reduction_len}) > 1024 ? 256 : (({reduction_len}) > 256 ? 128 : 64)), ({reduction_len}) * sizeof(float)>>>(\n            {}, {}_data, {}_len, {reduction_len}",
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

pub(in crate::codegen) fn render_reduce_f32_call(
    out: &mut String,
    function: &GpuFunction,
    assignment: &GpuAssign,
    dialect: GpuDialect,
) -> Result<(), GpuRenderError> {
    let [output] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 1));
    };
    let (term, _epilogue, output_len, reduction_len, env) = reduce_f32_parts(assignment)?;
    let _ = term;
    let output_len = value_expr(output_len);
    let reduction_len = value_expr(reduction_len);
    let kernel_name = kernel_name(&function.name, assignment)?;

    out.push_str(&format!(
        "    uint64_t {name}_len = {output_len};\n    float *{name}_data = nullptr;\n",
        name = output.name
    ));
    out.push_str(&format!("    catena_assert(({reduction_len}) <= 4096);\n"));
    out.push_str(&format!("    if ({}_len != 0) {{\n", output.name));
    out.push_str(&format!(
        "        catena_host_gpu_check({}((void **)&{}_data, {}_len * sizeof(float), nullptr));\n",
        dialect.device_alloc_async_fn(),
        output.name,
        output.name
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
    if element.as_ref() != &CType::F32 {
        return Err(GpuRenderError::UnsupportedType(
            runtime_type(output).unwrap().clone(),
        ));
    }
    let (exponential, _buffer, _capacity, _columns) = softmax_f32_parts(assignment)?;

    out.push_str(&format!(
        "__global__ void {kernel_name}(float *data, uint64_t rows, uint64_t columns) {{\n"
    ));
    out.push_str("    uint64_t row = (uint64_t)blockIdx.x;\n");
    out.push_str("    if (row >= rows) { return; }\n");
    out.push_str("    uint64_t row_offset = row * columns;\n");
    out.push_str("    extern __shared__ float reduction_terms[];\n");
    out.push_str(
        "    for (uint64_t column = (uint64_t)threadIdx.x; column < columns; column += (uint64_t)blockDim.x) {\n",
    );
    out.push_str("        reduction_terms[column] = data[row_offset + column];\n");
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
    out.push_str("        float score = data[row_offset + column];\n");
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
    out.push_str("        data[row_offset + column] = numerator;\n");
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
    out.push_str("        data[index] = denominator == 0.0f ? 0.0f : data[index] / denominator;\n");
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
    out.push_str(&format!("    catena_assert(({columns}) <= 8192);\n"));
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
        render_kernel(&mut out, "materialize_test", &assignment).unwrap();

        assert!(out.contains("value = i;"));
        assert!(!out.contains("program_ix_to_u64"));
    }
}
