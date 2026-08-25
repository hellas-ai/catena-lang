use crate::codegen::{
    GpuAssign, GpuDialect, GpuFunction, GpuValue,
    components::{input_components, output_components, single_value, value_expr},
    gpu::GpuRenderError,
    render_utils::{invalid_inputs, invalid_outputs},
};

type Parts<'a> = [&'a GpuValue; 18];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::codegen) enum StorageType {
    F32,
    Bf16,
}

impl StorageType {
    fn c_type(self) -> &'static str {
        match self {
            Self::F32 => "float",
            Self::Bf16 => "catena_bf16_t",
        }
    }

    fn load(self, expression: &str) -> String {
        match self {
            Self::F32 => expression.to_owned(),
            Self::Bf16 => format!("catena_bf16_to_f32({expression})"),
        }
    }

    fn store(self, expression: &str) -> String {
        match self {
            Self::F32 => expression.to_owned(),
            Self::Bf16 => format!("catena_bf16_from_f32({expression})"),
        }
    }
}

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

pub(in crate::codegen) fn render_kernel(
    out: &mut String,
    name: &str,
    dialect: GpuDialect,
    storage: StorageType,
) {
    let shuffle = match dialect {
        GpuDialect::Hip => "__shfl_down(value, offset, width)",
        GpuDialect::Cuda => "__shfl_down_sync(0xffffffffu, value, offset, width)",
    };
    let broadcast = match dialect {
        GpuDialect::Hip => "__shfl(prediction, 0, width)",
        GpuDialect::Cuda => "__shfl_sync(0xffffffffu, prediction, 0, width)",
    };
    let storage_type = storage.c_type();
    let state_load = storage.load("state[state_base + i]");
    let key_load = storage.load("k[(t * key_heads + key_head) * dim + i]");
    let query_load = storage.load("q[(t * key_heads + key_head) * dim + i]");
    let value_load = storage.load("v[(t * value_heads + value_head) * dim + column]");
    let gate_load = storage.load("gate[t * value_heads + value_head]");
    let beta_load = storage.load("beta[t * value_heads + value_head]");
    let result_store = storage.store("output * scale");
    let state_store = storage.store("s[r]");
    out.push_str(&format!(
        r#"
__device__ __forceinline__ float {name}_sum(float value, int width) {{
    for (int offset = width / 2; offset > 0; offset /= 2) value += {shuffle};
    return value;
}}

__global__ void {name}({storage_type} *state, uint64_t state_offset,
    const {storage_type} *q, const {storage_type} *k, const {storage_type} *v,
    const {storage_type} *gate, const {storage_type} *beta, {storage_type} *result,
    uint64_t tokens,
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
        s[r] = i < dim ? {state_load} : 0.0f;
    }}
    const float scale = rsqrtf((float)dim);
    for (uint64_t t = 0; t < tokens; ++t) {{
        const float decay = expf({gate_load});
        const float b = {beta_load};
        float prediction = 0.0f;
        #pragma unroll
        for (int r = 0; r < 4; ++r) {{
            int i = lane + r * width;
            if (i < dim) prediction += s[r] * {key_load};
        }}
        prediction = {name}_sum(prediction, width);
        prediction = {broadcast};
        const float delta = ({value_load} - decay * prediction) * b;
        float output = 0.0f;
        #pragma unroll
        for (int r = 0; r < 4; ++r) {{
            int i = lane + r * width;
            if (i < dim) {{
                s[r] = decay * s[r] + {key_load} * delta;
                output += s[r] * {query_load};
            }}
        }}
        output = {name}_sum(output, width);
        if (lane == 0) result[(t * value_heads + value_head) * dim + column] = {result_store};
    }}
    #pragma unroll
    for (int r = 0; r < 4; ++r) {{
        int i = lane + r * width;
        if (i < dim) state[state_base + i] = {state_store};
    }}
}}
"#
    ));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_a_single_token_scan_loop_for_hip() {
        let mut source = String::new();
        render_kernel(&mut source, "scan", GpuDialect::Hip, StorageType::F32);
        assert!(source.contains("for (uint64_t t = 0; t < tokens; ++t)"));
        assert!(source.contains("state[state_base + i] = s[r]"));
        assert!(source.contains("__shfl_down(value, offset, width)"));
        assert!(!source.contains("__shfl_down_sync"));
    }

    #[test]
    fn renders_cuda_synchronous_shuffle() {
        let mut source = String::new();
        render_kernel(&mut source, "scan", GpuDialect::Cuda, StorageType::F32);
        assert!(source.contains("__shfl_down_sync(0xffffffffu"));
        assert!(source.contains("__shfl_sync(0xffffffffu, prediction"));
    }

    #[test]
    fn renders_bf16_storage_with_f32_recurrence() {
        let mut source = String::new();
        render_kernel(&mut source, "scan", GpuDialect::Hip, StorageType::Bf16);
        assert!(source.contains("catena_bf16_t *state"));
        assert!(source.contains("catena_bf16_to_f32(state[state_base + i])"));
        assert!(source.contains("float s[4]"));
        assert!(source.contains("catena_bf16_from_f32(output * scale)"));
        assert!(source.contains("catena_bf16_from_f32(s[r])"));
    }
}

pub(in crate::codegen) fn render_call(
    out: &mut String,
    function: &GpuFunction,
    assignment: &GpuAssign,
    dialect: GpuDialect,
    storage: StorageType,
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
        "    {storage_type} *{result}_data = nullptr;\n    if (({tokens}) != 0) {{\n",
        storage_type = storage.c_type(),
        tokens = e(13)
    ));
    out.push_str(&format!("        catena_host_gpu_check({alloc}((void **)&{result}_data, ({tokens}) * ({vh}) * ({dim}) * sizeof({storage_type}), nullptr));\n", alloc=dialect.device_alloc_async_fn(), tokens=e(13), vh=e(15), dim=e(16), storage_type=storage.c_type()));
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
