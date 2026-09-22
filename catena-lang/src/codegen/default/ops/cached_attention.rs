use crate::codegen::{
    GpuAssign, GpuDialect, GpuFunction, GpuValue, GpuVar,
    components::{input_components, output_components, single_value, value_expr},
    gpu::GpuRenderError,
    lower_types::CType,
    render_utils::{invalid_inputs, invalid_outputs},
    runtime_type,
};

fn parts(assignment: &GpuAssign) -> Result<[&GpuValue; 13], GpuRenderError> {
    let components = input_components(assignment)?;
    if components.len() != 13 {
        return Err(invalid_inputs(assignment, 13));
    }
    components
        .iter()
        .map(|part| single_value(part).map_err(|_| invalid_inputs(assignment, 13)))
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|_| invalid_inputs(assignment, 13))
}

fn outputs(assignment: &GpuAssign) -> Result<[&GpuVar; 3], GpuRenderError> {
    let components = output_components(assignment)?;
    if components.len() != 3 || components.iter().any(|part| part.len() != 1) {
        return Err(invalid_outputs(assignment, 3));
    }
    let values: [&GpuVar; 3] = components
        .iter()
        .map(|part| &part[0])
        .collect::<Vec<_>>()
        .try_into()
        .map_err(|_| invalid_outputs(assignment, 3))?;
    for value in values {
        let Some(ty) = runtime_type(value) else {
            return Err(GpuRenderError::ErasedType(value.clone()));
        };
        if ty != &CType::Pointer(Box::new(CType::BF16)) {
            return Err(GpuRenderError::UnsupportedType(ty.clone()));
        }
    }
    Ok(values)
}

pub(in crate::codegen) fn kernel_name(
    function: &GpuFunction,
    assignment: &GpuAssign,
) -> Result<String, GpuRenderError> {
    Ok(format!(
        "cached_attention_{}_{}",
        function.name,
        outputs(assignment)?[2].name
    ))
}

pub(in crate::codegen) fn render_kernel(
    out: &mut String,
    name: &str,
    assignment: &GpuAssign,
    dialect: GpuDialect,
    matrix_core: bool,
) -> Result<(), GpuRenderError> {
    if matrix_core {
        out.push_str("#include <rocwmma/rocwmma.hpp>\n");
        render_kernel(out, name, assignment, dialect, false)?;
        render_wmma_kernel(out, name);
        return Ok(());
    }
    let _ = parts(assignment)?;
    let _ = outputs(assignment)?;
    let (shuffle_dot, shuffle_max, shuffle_sum, broadcast_max, broadcast_sum) = match dialect {
        GpuDialect::Hip => (
            "__shfl_down(dot, offset, 32)",
            "__shfl_down(tile_maximum, offset, 32)",
            "__shfl_down(tile_sum, offset, 32)",
            "__shfl(tile_maximum, 0, 32)",
            "__shfl(tile_sum, 0, 32)",
        ),
        GpuDialect::Cuda => (
            "__shfl_down_sync(0xffffffffu, dot, offset, 32)",
            "__shfl_down_sync(0xffffffffu, tile_maximum, offset, 32)",
            "__shfl_down_sync(0xffffffffu, tile_sum, offset, 32)",
            "__shfl_sync(0xffffffffu, tile_maximum, 0, 32)",
            "__shfl_sync(0xffffffffu, tile_sum, 0, 32)",
        ),
    };
    out.push_str(&format!(
        r#"__global__ void {name}_vector(
    const catena_bf16_t *cache, const catena_bf16_t *query,
    catena_bf16_t *output, uint64_t sequence, uint64_t history,
    uint64_t capacity, uint64_t layer, uint64_t query_heads,
    uint64_t key_value_heads, uint64_t dim, float scale) {{
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tid = (uint64_t)threadIdx.x;
    uint64_t lane = tid & 31;
    uint64_t key_lane = tid >> 5;
    uint64_t query_head = row % query_heads;
    uint64_t key_head = query_head / (query_heads / key_value_heads);
    uint64_t features = key_value_heads * dim;
    uint64_t plane = capacity * features;
    uint64_t key_base = 2 * layer * plane + key_head * dim;
    uint64_t value_base = key_base + plane;
    uint64_t causal_keys = history - sequence + row / query_heads + 1;
    __shared__ float scores[8];
    __shared__ float weights[8];
    __shared__ float alpha;
    __shared__ float denominator;
    float maximum = -INFINITY;
    float accumulator = 0.0f;
    if (tid == 0) denominator = 0.0f;
    for (uint64_t first = 0; first < causal_keys; first += 8) {{
        uint64_t key_position = first + key_lane;
        float dot = 0.0f;
        if (key_position < causal_keys) {{
            for (uint64_t channel = lane; channel < dim; channel += 32) {{
                dot += catena_bf16_to_f32(query[row * dim + channel]) *
                    catena_bf16_to_f32(cache[key_base + key_position * features + channel]);
            }}
        }}
        for (int offset = 16; offset > 0; offset >>= 1) {{
            dot += {shuffle_dot};
        }}
        if (lane == 0) scores[key_lane] = key_position < causal_keys ? dot * scale : -INFINITY;
        __syncthreads();
        if (tid == 0) {{
            float next_maximum = maximum;
            for (int k = 0; k < 8; ++k) next_maximum = fmaxf(next_maximum, scores[k]);
            alpha = expf(maximum - next_maximum);
            float next_denominator = denominator * alpha;
            for (int k = 0; k < 8; ++k) {{
                weights[k] = scores[k] == -INFINITY ? 0.0f : expf(scores[k] - next_maximum);
                next_denominator += weights[k];
            }}
            maximum = next_maximum;
            denominator = next_denominator;
        }}
        __syncthreads();
        if (tid < dim) {{
            accumulator *= alpha;
            for (uint64_t k = 0; k < 8 && first + k < causal_keys; ++k) {{
                accumulator += weights[k] * catena_bf16_to_f32(
                    cache[value_base + (first + k) * features + tid]);
            }}
        }}
        __syncthreads();
    }}
    if (tid < dim) output[row * dim + tid] = catena_bf16_from_f32(accumulator / denominator);
}}

__global__ void {name}_tiled(
    const catena_bf16_t *cache, const catena_bf16_t *query,
    catena_bf16_t *output, uint64_t sequence, uint64_t history,
    uint64_t capacity, uint64_t layer, uint64_t query_heads,
    uint64_t key_value_heads, uint64_t dim, float scale) {{
    constexpr uint64_t query_tile = 8;
    constexpr uint64_t key_tile = 32;
    constexpr uint64_t heads_per_tile = 2;
    uint64_t tid = (uint64_t)threadIdx.x;
    uint64_t lane = tid & 31;
    uint64_t wave = tid >> 5;
    uint64_t heads_per_key_head = query_heads / key_value_heads;
    uint64_t head_tiles = (heads_per_key_head + heads_per_tile - 1) / heads_per_tile;
    uint64_t key_head = (uint64_t)blockIdx.y / head_tiles;
    uint64_t head_in_group = ((uint64_t)blockIdx.y % head_tiles) * heads_per_tile;
    uint64_t query_start = (uint64_t)blockIdx.x * query_tile;
    uint64_t features = key_value_heads * dim;
    uint64_t plane = capacity * features;
    uint64_t key_base = 2 * layer * plane + key_head * dim;
    uint64_t value_base = key_base + plane;
    extern __shared__ unsigned char shared_bytes[];
    catena_bf16_t *queries = reinterpret_cast<catena_bf16_t *>(shared_bytes);
    catena_bf16_t *keys = queries + heads_per_tile * query_tile * dim;
    catena_bf16_t *values = keys + key_tile * dim;
    float *weights = reinterpret_cast<float *>(values + key_tile * dim);
    for (uint64_t index = tid; index < heads_per_tile * query_tile * dim; index += 256) {{
        uint64_t channel = index % dim;
        uint64_t row = index / dim;
        uint64_t token = row % query_tile;
        uint64_t local_head = row / query_tile;
        uint64_t query_head = key_head * heads_per_key_head + head_in_group + local_head;
        bool active = query_start + token < sequence &&
            head_in_group + local_head < heads_per_key_head;
        queries[index] = active
            ? query[((query_start + token) * query_heads + query_head) * dim + channel]
            : catena_bf16_from_f32(0.0f);
    }}
    __syncthreads();
    float maximum[2] = {{-INFINITY, -INFINITY}};
    float denominator[2] = {{0.0f, 0.0f}};
    float accumulator[2][8] = {{}};
    uint64_t tile_end = query_start + query_tile < sequence ? query_start + query_tile : sequence;
    uint64_t last_token = tile_end - 1;
    uint64_t block_causal_keys = history - sequence + last_token + 1;
    for (uint64_t first_key = 0; first_key < block_causal_keys; first_key += key_tile) {{
        for (uint64_t index = tid; index < key_tile * dim; index += 256) {{
            uint64_t key = index / dim;
            uint64_t channel = index % dim;
            uint64_t position = first_key + key;
            catena_bf16_t key_value = catena_bf16_from_f32(0.0f);
            catena_bf16_t value_value = catena_bf16_from_f32(0.0f);
            if (position < history) {{
                key_value = cache[key_base + position * features + channel];
                value_value = cache[value_base + position * features + channel];
            }}
            keys[channel * key_tile + key] = key_value;
            values[key * dim + channel] = value_value;
        }}
        __syncthreads();
        #pragma unroll
        for (uint64_t owned = 0; owned < 2; ++owned) {{
            uint64_t row = wave + owned * 8;
            uint64_t token = row % query_tile;
            uint64_t local_head = row / query_tile;
            bool active = query_start + token < sequence &&
                head_in_group + local_head < heads_per_key_head;
            uint64_t causal_keys = history - sequence + query_start + token + 1;
            uint64_t key_position = first_key + lane;
            float dot = 0.0f;
            if (active && key_position < causal_keys) {{
                #pragma unroll 4
                for (uint64_t channel = 0; channel < dim; ++channel) {{
                    dot += catena_bf16_to_f32(queries[row * dim + channel]) *
                        catena_bf16_to_f32(keys[channel * key_tile + lane]);
                }}
                dot *= scale;
            }} else {{
                dot = -INFINITY;
            }}
            float tile_maximum = dot;
            for (int offset = 16; offset > 0; offset >>= 1) {{
                tile_maximum = fmaxf(tile_maximum, {shuffle_max});
            }}
            tile_maximum = {broadcast_max};
            float next_maximum = fmaxf(maximum[owned], tile_maximum);
            float alpha = expf(maximum[owned] - next_maximum);
            float weight = dot == -INFINITY ? 0.0f : expf(dot - next_maximum);
            weights[row * key_tile + lane] = weight;
            __syncwarp();
            float tile_sum = weight;
            for (int offset = 16; offset > 0; offset >>= 1) {{
                tile_sum += {shuffle_sum};
            }}
            tile_sum = {broadcast_sum};
            denominator[owned] = denominator[owned] * alpha + tile_sum;
            maximum[owned] = next_maximum;
            if (active) {{
                #pragma unroll
                for (uint64_t part = 0; part < 8; ++part) {{
                    uint64_t channel = lane + part * 32;
                    if (channel < dim) {{
                        float contribution = 0.0f;
                        #pragma unroll
                        for (uint64_t key = 0; key < key_tile; ++key) {{
                            contribution += weights[row * key_tile + key] *
                                catena_bf16_to_f32(values[key * dim + channel]);
                        }}
                        accumulator[owned][part] = accumulator[owned][part] * alpha + contribution;
                    }}
                }}
            }}
        }}
        __syncthreads();
    }}
    #pragma unroll
    for (uint64_t owned = 0; owned < 2; ++owned) {{
        uint64_t row = wave + owned * 8;
        uint64_t token = row % query_tile;
        uint64_t local_head = row / query_tile;
        uint64_t query_head = key_head * heads_per_key_head + head_in_group + local_head;
        bool active = query_start + token < sequence &&
            head_in_group + local_head < heads_per_key_head;
        if (active) {{
            #pragma unroll
            for (uint64_t part = 0; part < 8; ++part) {{
                uint64_t channel = lane + part * 32;
                if (channel < dim) {{
                    uint64_t output_row = (query_start + token) * query_heads + query_head;
                    output[output_row * dim + channel] =
                        catena_bf16_from_f32(accumulator[owned][part] / denominator[owned]);
                }}
            }}
        }}
    }}
}}
"#
    ));
    Ok(())
}

fn render_wmma_kernel(out: &mut String, name: &str) {
    out.push_str(&format!(
        r#"__global__ void {name}_wmma(
    const catena_bf16_t *cache, const catena_bf16_t *query,
    catena_bf16_t *output, uint64_t sequence, uint64_t history,
    uint64_t capacity, uint64_t layer, uint64_t query_heads,
    uint64_t key_value_heads, uint64_t dim, float scale) {{
    constexpr uint64_t query_tile = 8;
    constexpr uint64_t key_tile = 16;
    constexpr uint64_t heads_per_tile = 2;
    uint64_t tid = (uint64_t)threadIdx.x;
    uint64_t lane = tid & 31;
    uint64_t wave = tid >> 5;
    uint64_t heads_per_key_head = query_heads / key_value_heads;
    uint64_t head_tiles = (heads_per_key_head + heads_per_tile - 1) / heads_per_tile;
    uint64_t key_head = (uint64_t)blockIdx.y / head_tiles;
    uint64_t head_in_group = ((uint64_t)blockIdx.y % head_tiles) * heads_per_tile;
    uint64_t query_start = (uint64_t)blockIdx.x * query_tile;
    uint64_t features = key_value_heads * dim;
    uint64_t plane = capacity * features;
    uint64_t key_base = 2 * layer * plane + key_head * dim;
    uint64_t value_base = key_base + plane;
    extern __shared__ unsigned char shared_bytes[];
    catena_bf16_t *queries = reinterpret_cast<catena_bf16_t *>(shared_bytes);
    catena_bf16_t *keys = queries + heads_per_tile * query_tile * dim;
    catena_bf16_t *values = keys + key_tile * dim;
    float *scores = reinterpret_cast<float *>(values + key_tile * dim);
    float *weights = scores + heads_per_tile * query_tile * key_tile;
    for (uint64_t index = tid; index < heads_per_tile * query_tile * dim; index += 256) {{
        uint64_t channel = index % dim;
        uint64_t row = index / dim;
        uint64_t token = row % query_tile;
        uint64_t local_head = row / query_tile;
        uint64_t query_head = key_head * heads_per_key_head + head_in_group + local_head;
        bool active = query_start + token < sequence &&
            head_in_group + local_head < heads_per_key_head;
        queries[index] = active
            ? query[((query_start + token) * query_heads + query_head) * dim + channel]
            : catena_bf16_from_f32(0.0f);
    }}
    __syncthreads();
    float maximum[2] = {{-INFINITY, -INFINITY}};
    float denominator[2] = {{0.0f, 0.0f}};
    float accumulator[2][8] = {{}};
    uint64_t tile_end = query_start + query_tile < sequence ? query_start + query_tile : sequence;
    uint64_t block_causal_keys = history - sequence + tile_end;
    for (uint64_t first_key = 0; first_key < block_causal_keys; first_key += key_tile) {{
        for (uint64_t index = tid; index < key_tile * dim; index += 256) {{
            uint64_t key = index / dim;
            uint64_t channel = index % dim;
            uint64_t position = first_key + key;
            keys[index] = position < history
                ? cache[key_base + position * features + channel]
                : catena_bf16_from_f32(0.0f);
            values[index] = position < history
                ? cache[value_base + position * features + channel]
                : catena_bf16_from_f32(0.0f);
        }}
        __syncthreads();
        if (wave == 0) {{
            using a_t = rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16,
                catena_bf16_t, rocwmma::row_major>;
            using b_t = rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16,
                catena_bf16_t, rocwmma::col_major>;
            using c_t = rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float>;
            a_t a;
            b_t b;
            c_t c;
            rocwmma::fill_fragment(c, 0.0f);
            for (uint64_t first_channel = 0; first_channel < dim; first_channel += 16) {{
                rocwmma::load_matrix_sync(a, queries + first_channel, dim);
                rocwmma::load_matrix_sync(b, keys + first_channel, dim);
                rocwmma::mma_sync(c, a, b, c);
            }}
            rocwmma::store_matrix_sync(scores, c, key_tile, rocwmma::mem_row_major);
        }}
        __syncthreads();
        #pragma unroll
        for (uint64_t owned = 0; owned < 2; ++owned) {{
            uint64_t row = wave + owned * 8;
            uint64_t token = row % query_tile;
            uint64_t local_head = row / query_tile;
            bool active = query_start + token < sequence &&
                head_in_group + local_head < heads_per_key_head;
            uint64_t causal_keys = history - sequence + query_start + token + 1;
            uint64_t key_position = first_key + lane;
            float score = active && lane < key_tile && key_position < causal_keys
                ? scores[row * key_tile + lane] * scale
                : -INFINITY;
            float tile_maximum = score;
            for (int offset = 16; offset > 0; offset >>= 1) {{
                tile_maximum = fmaxf(tile_maximum, __shfl_down(tile_maximum, offset, 32));
            }}
            tile_maximum = __shfl(tile_maximum, 0, 32);
            float next_maximum = fmaxf(maximum[owned], tile_maximum);
            float alpha = expf(maximum[owned] - next_maximum);
            float weight = score == -INFINITY ? 0.0f : expf(score - next_maximum);
            if (lane < key_tile) weights[row * key_tile + lane] = weight;
            __syncwarp();
            float tile_sum = weight;
            for (int offset = 16; offset > 0; offset >>= 1) {{
                tile_sum += __shfl_down(tile_sum, offset, 32);
            }}
            tile_sum = __shfl(tile_sum, 0, 32);
            denominator[owned] = denominator[owned] * alpha + tile_sum;
            maximum[owned] = next_maximum;
            if (active) {{
                #pragma unroll
                for (uint64_t part = 0; part < 8; ++part) {{
                    uint64_t channel = lane + part * 32;
                    if (channel < dim) {{
                        float contribution = 0.0f;
                        #pragma unroll
                        for (uint64_t key = 0; key < key_tile; ++key) {{
                            contribution += weights[row * key_tile + key] *
                                catena_bf16_to_f32(values[key * dim + channel]);
                        }}
                        accumulator[owned][part] = accumulator[owned][part] * alpha + contribution;
                    }}
                }}
            }}
        }}
        __syncthreads();
    }}
    #pragma unroll
    for (uint64_t owned = 0; owned < 2; ++owned) {{
        uint64_t row = wave + owned * 8;
        uint64_t token = row % query_tile;
        uint64_t local_head = row / query_tile;
        uint64_t query_head = key_head * heads_per_key_head + head_in_group + local_head;
        bool active = query_start + token < sequence &&
            head_in_group + local_head < heads_per_key_head;
        if (active) {{
            #pragma unroll
            for (uint64_t part = 0; part < 8; ++part) {{
                uint64_t channel = lane + part * 32;
                if (channel < dim) {{
                    uint64_t output_row = (query_start + token) * query_heads + query_head;
                    output[output_row * dim + channel] =
                        catena_bf16_from_f32(accumulator[owned][part] / denominator[owned]);
                }}
            }}
        }}
    }}
}}
"#
    ));
}

pub(in crate::codegen) fn render_call(
    out: &mut String,
    function: &GpuFunction,
    assignment: &GpuAssign,
    dialect: GpuDialect,
    matrix_core: bool,
) -> Result<(), GpuRenderError> {
    let parts = parts(assignment)?;
    let outputs = outputs(assignment)?;
    let e = |index| value_expr(parts[index]);
    let name = kernel_name(function, assignment)?;
    let output = &outputs[2].name;
    let launch = |kernel: &str, shared: &str| {
        format!(
            "{kernel}<<<dim3((({sequence}) + 7) / 8, ({kvh}) * {output}_head_tiles), dim3(256), {output}_{shared}_shared_bytes>>>({cache}, {query}, {output}_data, {sequence}, {history}, {capacity}, {layer}, {qh}, {kvh}, {dim}, {scale});",
            sequence = e(5),
            kvh = e(10),
            cache = e(0),
            query = e(2),
            history = e(6),
            capacity = e(7),
            layer = e(8),
            qh = e(9),
            dim = e(11),
            scale = e(12),
        )
    };
    let simt_launch = launch(&format!("{name}_tiled"), "simt");
    let prefill_launch = if matrix_core {
        let wmma_launch = launch(&format!("{name}_wmma"), "wmma");
        format!(
            "if (({sequence}) < 1024 || ({dim}) % 16 != 0) {{ {simt_launch} }} else {{ {wmma_launch} }}",
            sequence = e(5),
            dim = e(11),
        )
    } else {
        simt_launch
    };
    out.push_str(&format!(
        "    catena_assert(({qh}) != 0 && ({kvh}) != 0 && ({qh}) % ({kvh}) == 0);\n\
         catena_assert(({dim}) != 0 && ({dim}) <= 256);\n\
         catena_assert(({capacity}) != 0 && ({sequence}) <= ({history}) && ({history}) <= ({capacity}));\n\
         catena_assert(({sequence}) == 0 || ({qh}) <= UINT64_MAX / ({sequence}));\n\
         uint64_t {output}_rows = ({sequence}) * ({qh});\n\
         catena_assert({output}_rows == 0 || ({dim}) <= UINT64_MAX / {output}_rows);\n\
         catena_assert(({query_count}) == {output}_rows * ({dim}));\n\
         catena_assert(({output_count}) == ({query_count}));\n\
         catena_assert(({kvh}) <= UINT64_MAX / ({dim}));\n\
         uint64_t {output}_features = ({kvh}) * ({dim});\n\
         catena_assert({output}_features == 0 || ({capacity}) <= UINT64_MAX / {output}_features);\n\
         uint64_t {output}_plane = ({capacity}) * {output}_features;\n\
         catena_assert({output}_plane <= UINT64_MAX / 2);\n\
         uint64_t {output}_stride = 2 * {output}_plane;\n\
         catena_assert(({layer}) < UINT64_MAX && ({layer}) + 1 <= UINT64_MAX / {output}_stride);\n\
         catena_assert(({cache_count}) >= (({layer}) + 1) * {output}_stride);\n\
         uint64_t {output}_heads_per_key_head = ({qh}) / ({kvh});\n\
         uint64_t {output}_head_tiles = ({output}_heads_per_key_head + 1) / 2;\n\
         catena_assert({output}_head_tiles == 0 || ({kvh}) <= UINT64_MAX / {output}_head_tiles);\n\
         size_t {output}_simt_shared_bytes = (size_t)80 * ({dim}) * sizeof(catena_bf16_t) + (size_t)16 * 32 * sizeof(float);\n\
         size_t {output}_wmma_shared_bytes = (size_t)48 * ({dim}) * sizeof(catena_bf16_t) + (size_t)16 * 32 * sizeof(float);\n\
         catena_bf16_t *{output}_data = nullptr;\n\
         if (({output_count}) != 0) {{\n\
             catena_host_gpu_check({alloc}((void **)&{output}_data, ({output_count}) * sizeof(catena_bf16_t), nullptr));\n\
             if (({sequence}) < 8) {{\n\
                 {name}_vector<<<dim3({output}_rows), dim3(256)>>>({cache}, {query}, {output}_data, {sequence}, {history}, {capacity}, {layer}, {qh}, {kvh}, {dim}, {scale});\n\
             }} else {{\n\
                 {prefill_launch}\n\
             }}\n\
         }}\n\
         {cache_out} = {cache};\n\
         {query_out} = {query};\n\
         {output} = {output}_data;\n",
        cache = e(0), cache_count = e(1), query = e(2), query_count = e(3),
        output_count = e(4), sequence = e(5), history = e(6), capacity = e(7),
        layer = e(8), qh = e(9), kvh = e(10), dim = e(11), scale = e(12),
        alloc = dialect.device_alloc_async_fn(), cache_out = outputs[0].name,
        query_out = outputs[1].name,
        prefill_launch = prefill_launch,
    ));
    Ok(())
}
