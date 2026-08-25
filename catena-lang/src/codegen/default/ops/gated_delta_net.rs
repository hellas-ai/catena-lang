use crate::codegen::{
    GpuAssign, GpuDialect, GpuFunction, GpuValue,
    components::{input_components, output_components, single_value, value_expr},
    gpu::GpuRenderError,
    render_utils::{invalid_inputs, invalid_outputs},
};

type Parts<'a> = [&'a GpuValue; 18];

fn parts(assignment: &GpuAssign) -> Result<Parts<'_>, GpuRenderError> {
    let components = input_components(assignment)?;
    if components.len() != 18 {
        return Err(GpuRenderError::InvalidInputComponentCount {
            op: assignment.op.clone(),
            expected: 18,
            actual: components.len(),
        });
    }
    let mut values = Vec::with_capacity(18);
    for component in &components {
        values.push(single_value(component).map_err(|_| invalid_inputs(assignment, 18))?);
    }
    values
        .try_into()
        .map_err(|_| invalid_inputs(assignment, 18))
}

pub(in crate::codegen) fn kernel_name(
    function: &GpuFunction,
    assignment: &GpuAssign,
) -> Result<String, GpuRenderError> {
    let Some(output) = assignment.outputs.last() else {
        return Err(invalid_outputs(assignment, 7));
    };
    Ok(format!("gated_delta_net_{}_{}", function.name, output.name))
}

pub(in crate::codegen) fn render_kernel(out: &mut String, name: &str, dialect: GpuDialect) {
    let shuffle = match dialect {
        GpuDialect::Hip => "__shfl_down(value, offset, width)",
        GpuDialect::Cuda => "__shfl_down_sync(0xffffffffu, value, offset, width)",
    };
    let broadcast = match dialect {
        GpuDialect::Hip => "__shfl(prediction, 0, width)",
        GpuDialect::Cuda => "__shfl_sync(0xffffffffu, prediction, 0, width)",
    };
    out.push_str(&format!(r#"
__device__ __forceinline__ float {name}_sum(float value, int width) {{
    for (int offset = width / 2; offset > 0; offset /= 2) value += {shuffle};
    return value;
}}

__global__ void {name}(float *state, uint64_t state_offset,
    const float *q, const float *k, const float *v, const float *gate,
    const float *beta, float *result, uint64_t tokens,
    uint64_t key_heads, uint64_t value_heads, uint64_t dim) {{
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int width = blockDim.x;
    const uint64_t value_head = blockIdx.x;
    const int column = (int)blockIdx.z * blockDim.y + warp;
    if (value_head >= value_heads || column >= dim || lane >= width) return;
    const uint64_t key_head = value_head % key_heads;
    const uint64_t state_base = state_offset + (value_head * dim + column) * dim;
    float s[4];
    #pragma unroll
    for (int r = 0; r < 4; ++r) {{
        int i = lane + r * width;
        s[r] = i < dim ? state[state_base + i] : 0.0f;
    }}
    const float scale = rsqrtf((float)dim);
    for (uint64_t t = 0; t < tokens; ++t) {{
        const float decay = expf(gate[t * value_heads + value_head]);
        const float b = beta[t * value_heads + value_head];
        float prediction = 0.0f;
        #pragma unroll
        for (int r = 0; r < 4; ++r) {{
            int i = lane + r * width;
            if (i < dim) prediction += s[r] * k[(t * key_heads + key_head) * dim + i];
        }}
        prediction = {name}_sum(prediction, width);
        prediction = {broadcast};
        const float delta = (v[(t * value_heads + value_head) * dim + column] - decay * prediction) * b;
        float output = 0.0f;
        #pragma unroll
        for (int r = 0; r < 4; ++r) {{
            int i = lane + r * width;
            if (i < dim) {{
                s[r] = decay * s[r] + k[(t * key_heads + key_head) * dim + i] * delta;
                output += s[r] * q[(t * key_heads + key_head) * dim + i];
            }}
        }}
        output = {name}_sum(output, width);
        if (lane == 0) result[(t * value_heads + value_head) * dim + column] = output * scale;
    }}
    #pragma unroll
    for (int r = 0; r < 4; ++r) {{
        int i = lane + r * width;
        if (i < dim) state[state_base + i] = s[r];
    }}
}}
"#));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_a_single_token_scan_loop_for_hip() {
        let mut source = String::new();
        render_kernel(&mut source, "scan", GpuDialect::Hip);
        assert!(source.contains("for (uint64_t t = 0; t < tokens; ++t)"));
        assert!(source.contains("state[state_base + i] = s[r]"));
        assert!(source.contains("__shfl_down(value, offset, width)"));
        assert!(!source.contains("__shfl_down_sync"));
    }

    #[test]
    fn renders_cuda_synchronous_shuffle() {
        let mut source = String::new();
        render_kernel(&mut source, "scan", GpuDialect::Cuda);
        assert!(source.contains("__shfl_down_sync(0xffffffffu"));
        assert!(source.contains("__shfl_sync(0xffffffffu, prediction"));
    }
}

pub(in crate::codegen) fn render_call(
    out: &mut String,
    function: &GpuFunction,
    assignment: &GpuAssign,
    dialect: GpuDialect,
) -> Result<(), GpuRenderError> {
    let p = parts(assignment)?;
    let outputs = output_components(assignment)?;
    if outputs.len() != 7 || outputs.iter().any(|x| x.len() != 1) {
        return Err(invalid_outputs(assignment, 7));
    }
    let name = kernel_name(function, assignment)?;
    let e = |i| value_expr(p[i]);
    let result = &outputs[6][0].name;
    out.push_str(&format!(
        "    catena_assert(({kh}) != 0 && ({vh}) != 0 && ({vh}) % ({kh}) == 0);\n",
        kh = e(14),
        vh = e(15)
    ));
    out.push_str(&format!(
        "    catena_assert(({dim}) == 16 || ({dim}) == 32 || ({dim}) == 64 || ({dim}) == 128);\n",
        dim = e(16)
    ));
    out.push_str(&format!(
        "    catena_assert(({qcap}) == ({tokens}) * ({kh}) * ({dim}));\n",
        qcap = e(3),
        tokens = e(13),
        kh = e(14),
        dim = e(16)
    ));
    out.push_str(&format!(
        "    catena_assert(({kcap}) == ({qcap}) && ({vcap}) == ({tokens}) * ({vh}) * ({dim}));\n",
        kcap = e(5),
        qcap = e(3),
        vcap = e(7),
        tokens = e(13),
        vh = e(15),
        dim = e(16)
    ));
    out.push_str(&format!(
        "    catena_assert(({gcap}) == ({tokens}) * ({vh}) && ({bcap}) == ({gcap}));\n",
        gcap = e(9),
        bcap = e(11),
        tokens = e(13),
        vh = e(15)
    ));
    out.push_str(&format!("    catena_assert(({off}) <= ({scap}) && ({vh}) * ({dim}) * ({dim}) <= ({scap}) - ({off}));\n", off=e(12), scap=e(1), vh=e(15), dim=e(16)));
    out.push_str(&format!(
        "    float *{result}_data = nullptr;\n    if (({tokens}) != 0) {{\n",
        tokens = e(13)
    ));
    out.push_str(&format!("        catena_host_gpu_check({alloc}((void **)&{result}_data, ({tokens}) * ({vh}) * ({dim}) * sizeof(float), nullptr));\n", alloc=dialect.device_alloc_async_fn(), tokens=e(13), vh=e(15), dim=e(16)));
    out.push_str(&format!("        dim3 block(({dim}) < 32 ? ({dim}) : 32, 4);\n        dim3 grid(({vh}), 1, (({dim}) + 3) / 4);\n", dim=e(16), vh=e(15)));
    out.push_str(&format!("        {name}<<<grid, block>>>({state}, {off}, {q}, {k}, {v}, {gate}, {beta}, {result}_data, {tokens}, {kh}, {vh}, {dim});\n    }}\n", state=e(0), off=e(12), q=e(2), k=e(4), v=e(6), gate=e(8), beta=e(10), tokens=e(13), kh=e(14), vh=e(15), dim=e(16)));
    for (i, input_index) in [0usize, 2, 4, 6, 8, 10].iter().enumerate() {
        out.push_str(&format!(
            "    {} = {};\n",
            outputs[i][0].name,
            e(*input_index)
        ));
    }
    out.push_str(&format!("    {result} = {result}_data;\n"));
    Ok(())
}
