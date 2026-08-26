use crate::codegen::{
    GpuAssign, GpuDialect, GpuFunction, GpuValue,
    components::{input_components, output_components, single_value, value_expr},
    gpu::GpuRenderError,
    lower_types::CType,
    render_utils::{invalid_inputs, invalid_outputs},
    runtime_type,
};

type Parts<'a> = [&'a GpuValue; 12];

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

    fn element_type(self) -> CType {
        match self {
            Self::F32 => CType::F32,
            Self::Bf16 => CType::BF16,
        }
    }
}

fn parts(assignment: &GpuAssign) -> Result<Parts<'_>, GpuRenderError> {
    let components = input_components(assignment)?;
    if components.len() != 12 {
        return Err(GpuRenderError::InvalidInputComponentCount {
            op: assignment.op.clone(),
            expected: 12,
            actual: components.len(),
        });
    }
    let mut values = Vec::with_capacity(12);
    for component in &components {
        values.push(single_value(component).map_err(|_| invalid_inputs(assignment, 12))?);
    }
    values
        .try_into()
        .map_err(|_| invalid_inputs(assignment, 12))
}

pub(in crate::codegen) fn kernel_name(
    function: &GpuFunction,
    assignment: &GpuAssign,
) -> Result<String, GpuRenderError> {
    let Some(output) = assignment.outputs.last() else {
        return Err(invalid_outputs(assignment, 3));
    };
    Ok(format!("cached_context_{}_{}", function.name, output.name))
}

fn validate_outputs(assignment: &GpuAssign, storage: StorageType) -> Result<(), GpuRenderError> {
    let outputs = output_components(assignment)?;
    if outputs.len() != 3 || outputs.iter().any(|output| output.len() != 1) {
        return Err(invalid_outputs(assignment, 3));
    }
    let expected = storage.element_type();
    for output in outputs {
        let value = &output[0];
        let Some(CType::Pointer(element)) = runtime_type(value) else {
            return Err(GpuRenderError::ErasedType(value.clone()));
        };
        if element.as_ref() != &expected {
            return Err(GpuRenderError::UnsupportedType(
                runtime_type(value).unwrap().clone(),
            ));
        }
    }
    Ok(())
}

pub(in crate::codegen) fn render_kernel(
    out: &mut String,
    name: &str,
    assignment: &GpuAssign,
    storage: StorageType,
) -> Result<(), GpuRenderError> {
    let _ = parts(assignment)?;
    validate_outputs(assignment, storage)?;
    let element = storage.c_type();
    let probability = storage.load("probabilities[probability_base + key_position]");
    let value = storage.load("cache[value_channel_base + key_position * key_value_features]");
    let result = storage.store("sum");
    out.push_str(&format!(
        r#"__global__ void {name}(const {element} *cache, const {element} *probabilities,
    {element} *output, uint64_t sequence, uint64_t history,
    uint64_t token_capacity, uint64_t layer, uint64_t query_heads,
    uint64_t key_value_heads, uint64_t head_dim) {{
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t channel = (uint64_t)blockIdx.y * 32 + (uint64_t)threadIdx.x;
    uint64_t key_lane = (uint64_t)threadIdx.y;
    bool active = row < sequence * query_heads && channel < head_dim;
    uint64_t query_head = row % query_heads;
    uint64_t query_heads_per_key_value_head = query_heads / key_value_heads;
    uint64_t key_value_head = query_head / query_heads_per_key_value_head;
    uint64_t key_value_features = key_value_heads * head_dim;
    uint64_t plane_stride = token_capacity * key_value_features;
    uint64_t layer_stride = 2 * plane_stride;
    uint64_t value_channel_base = layer * layer_stride + plane_stride
        + key_value_head * head_dim + channel;
    uint64_t probability_base = row * history;
    float sum = 0.0f;
    if (active) {{
        for (uint64_t key_position = key_lane; key_position < history; key_position += 8) {{
            sum += {probability} * {value};
        }}
    }}
    __shared__ float partial[8][32];
    partial[threadIdx.y][threadIdx.x] = sum;
    __syncthreads();
    if (threadIdx.y < 4) partial[threadIdx.y][threadIdx.x] += partial[threadIdx.y + 4][threadIdx.x];
    __syncthreads();
    if (threadIdx.y < 2) partial[threadIdx.y][threadIdx.x] += partial[threadIdx.y + 2][threadIdx.x];
    __syncthreads();
    if (threadIdx.y == 0 && active) {{
        sum = partial[0][threadIdx.x] + partial[1][threadIdx.x];
        output[row * head_dim + channel] = {result};
    }}
}}
"#
    ));
    Ok(())
}

pub(in crate::codegen) fn render_call(
    out: &mut String,
    function: &GpuFunction,
    assignment: &GpuAssign,
    dialect: GpuDialect,
    storage: StorageType,
) -> Result<(), GpuRenderError> {
    let p = parts(assignment)?;
    validate_outputs(assignment, storage)?;
    let outputs = output_components(assignment)?;
    let e = |index| value_expr(p[index]);
    let cache_output = &outputs[0][0].name;
    let probabilities_output = &outputs[1][0].name;
    let output = &outputs[2][0].name;
    let name = kernel_name(function, assignment)?;
    let element = storage.c_type();

    out.push_str(&format!(
        "    catena_assert(({qh}) != 0 && ({kvh}) != 0 && ({qh}) % ({kvh}) == 0);\n",
        qh = e(9),
        kvh = e(10)
    ));
    out.push_str(&format!(
        "    catena_assert(({dim}) != 0 && ({dim}) <= 1024);\n",
        dim = e(11)
    ));
    out.push_str(&format!(
        "    catena_assert(({capacity}) != 0 && ({history}) <= ({capacity}));\n",
        history = e(6),
        capacity = e(7)
    ));
    out.push_str(&format!(
        "    catena_assert(({sequence}) == 0 || ({qh}) <= UINT64_MAX / ({sequence}));\n",
        sequence = e(5),
        qh = e(9)
    ));
    out.push_str(&format!(
        "    uint64_t {output}_rows = ({sequence}) * ({qh});\n",
        sequence = e(5),
        qh = e(9)
    ));
    out.push_str(&format!(
        "    catena_assert({output}_rows == 0 || ({history}) <= UINT64_MAX / {output}_rows);\n",
        history = e(6)
    ));
    out.push_str(&format!(
        "    catena_assert(({pcap}) == {output}_rows * ({history}));\n",
        pcap = e(3),
        history = e(6)
    ));
    out.push_str(&format!(
        "    catena_assert({output}_rows == 0 || ({dim}) <= UINT64_MAX / {output}_rows);\n",
        dim = e(11)
    ));
    out.push_str(&format!(
        "    catena_assert(({outcap}) == {output}_rows * ({dim}));\n",
        outcap = e(4),
        dim = e(11)
    ));
    out.push_str(&format!(
        "    catena_assert(({kvh}) <= UINT64_MAX / ({dim}));\n",
        kvh = e(10),
        dim = e(11)
    ));
    out.push_str(&format!(
        "    uint64_t {output}_kv_features = ({kvh}) * ({dim});\n",
        kvh = e(10),
        dim = e(11)
    ));
    out.push_str(&format!(
        "    catena_assert({output}_kv_features == 0 || ({capacity}) <= UINT64_MAX / {output}_kv_features);\n",
        capacity = e(7)
    ));
    out.push_str(&format!(
        "    uint64_t {output}_plane_stride = ({capacity}) * {output}_kv_features;\n",
        capacity = e(7)
    ));
    out.push_str(&format!(
        "    catena_assert({output}_plane_stride <= UINT64_MAX / 2);\n"
    ));
    out.push_str(&format!(
        "    uint64_t {output}_layer_stride = 2 * {output}_plane_stride;\n"
    ));
    out.push_str(&format!(
        "    catena_assert(({layer}) < UINT64_MAX && ({layer}) + 1 <= UINT64_MAX / {output}_layer_stride);\n",
        layer = e(8)
    ));
    out.push_str(&format!(
        "    catena_assert(({ccap}) >= (({layer}) + 1) * {output}_layer_stride);\n",
        ccap = e(1),
        layer = e(8)
    ));
    out.push_str(&format!(
        "    {element} *{output}_data = nullptr;\n    if (({outcap}) != 0) {{\n",
        outcap = e(4)
    ));
    out.push_str(&format!(
        "        catena_host_gpu_check({alloc}((void **)&{output}_data, ({outcap}) * sizeof({element}), nullptr));\n",
        alloc = dialect.device_alloc_async_fn(),
        outcap = e(4)
    ));
    out.push_str(&format!(
        "        {name}<<<dim3({output}_rows, (({dim}) + 31) / 32), dim3(32, 8)>>>({cache}, {probabilities}, {output}_data, {sequence}, {history}, {capacity}, {layer}, {qh}, {kvh}, {dim});\n    }}\n",
        dim = e(11),
        cache = e(0),
        probabilities = e(2),
        sequence = e(5),
        history = e(6),
        capacity = e(7),
        layer = e(8),
        qh = e(9),
        kvh = e(10)
    ));
    out.push_str(&format!(
        "    {cache_output} = {cache};\n    {probabilities_output} = {probabilities};\n    {output} = {output}_data;\n",
        cache = e(0),
        probabilities = e(2)
    ));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_coalesced_f32_value_traversal() {
        let mut source = String::new();
        let result = StorageType::F32.store("sum");
        let probability = StorageType::F32.load("probabilities[probability_base + key_position]");
        let value =
            StorageType::F32.load("cache[value_channel_base + key_position * key_value_features]");
        source.push_str(&format!("{probability}\n{value}\n{result}"));
        assert!(source.contains("key_position * key_value_features"));
        assert!(!source.contains("catena_bf16"));
    }

    #[test]
    fn renders_bf16_loads_with_f32_accumulation() {
        let probability = StorageType::Bf16.load("probabilities[probability_base + key_position]");
        let value =
            StorageType::Bf16.load("cache[value_channel_base + key_position * key_value_features]");
        let result = StorageType::Bf16.store("sum");
        assert!(probability.contains("catena_bf16_to_f32"));
        assert!(value.contains("catena_bf16_to_f32"));
        assert_eq!(result, "catena_bf16_from_f32(sum)");
    }
}
