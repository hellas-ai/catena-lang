use super::super::gpu::{GpuRenderError, value_expr};
use crate::codegen::{GpuAssign, GpuFunction, GpuValue};

pub fn kernel_name(function: &GpuFunction, assignment_index: usize) -> String {
    format!("{}_launch_{assignment_index}", function.name)
}

pub fn render_kernel(
    output: &mut String,
    function: &GpuFunction,
    assignment_index: usize,
    assignment: &GpuAssign,
) -> Result<(), GpuRenderError> {
    let LaunchInputs {
        length: _,
        grid: _,
        buffer: _,
        kernel_argument: _,
        scheduling: _,
        kernel,
    } = inputs(assignment)?;
    let wrapper = kernel_name(function, assignment_index);
    output.push_str(&format!(
        "extern \"C\" __global__ void {wrapper}(uint64_t length, uint64_t *buffer, uint64_t kernel_argument, catena_scheduling_t scheduling) {{\n"
    ));
    output.push_str("    uint64_t global_x = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;\n");
    output.push_str("    uint64_t global_y = (uint64_t)blockIdx.y * blockDim.y + threadIdx.y;\n");
    output.push_str("    uint64_t global_z = (uint64_t)blockIdx.z * blockDim.z + threadIdx.z;\n");
    output.push_str("    uint64_t global_width = (uint64_t)gridDim.x * blockDim.x;\n");
    output.push_str("    uint64_t global_height = (uint64_t)gridDim.y * blockDim.y;\n");
    output.push_str("    uint64_t global_linear_id = (global_z * global_height + global_y) * global_width + global_x;\n");
    output.push_str("    uint64_t in_block_linear_id = ((uint64_t)threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;\n");
    output.push_str("    uint64_t block_linear_id = ((uint64_t)blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;\n");
    output.push_str(
        "    catena_thread_t thread = { global_linear_id, in_block_linear_id, block_linear_id };\n",
    );
    output.push_str("    catena_scheduling_t scheduling_after;\n");
    output.push_str(&format!(
        "    {}(length, buffer, kernel_argument, thread, scheduling, &scheduling_after);\n",
        value_expr(kernel),
    ));
    output.push_str("}\n");
    Ok(())
}

pub fn render_call(
    output: &mut String,
    function: &GpuFunction,
    assignment_index: usize,
    assignment: &GpuAssign,
) -> Result<(), GpuRenderError> {
    let LaunchInputs {
        length,
        grid,
        buffer,
        kernel_argument,
        scheduling,
        kernel: _,
    } = inputs(assignment)?;
    let [result] = assignment.outputs.as_slice() else {
        return Err(GpuRenderError::InvalidLaunch(
            "expected one global-buffer output",
        ));
    };
    let wrapper = kernel_name(function, assignment_index);
    let grid = value_expr(grid);
    output.push_str(&format!(
        "    {wrapper}<<<dim3({grid}.grid_dim.x, {grid}.grid_dim.y, {grid}.grid_dim.z), dim3({grid}.block_dim.x, {grid}.block_dim.y, {grid}.block_dim.z)>>>(\n        {}, {}, {}, {});\n",
        value_expr(length),
        value_expr(buffer),
        value_expr(kernel_argument),
        value_expr(scheduling),
    ));
    output.push_str("    catena_gpu_synchronize();\n");
    output.push_str(&format!("    {} = {};\n", result.name, value_expr(buffer)));
    Ok(())
}

struct LaunchInputs<'a> {
    length: &'a GpuValue,
    grid: &'a GpuValue,
    buffer: &'a GpuValue,
    kernel_argument: &'a GpuValue,
    scheduling: &'a GpuValue,
    kernel: &'a GpuValue,
}

fn inputs(assignment: &GpuAssign) -> Result<LaunchInputs<'_>, GpuRenderError> {
    let [length, grid, buffer, kernel_argument, scheduling, kernel] = assignment.inputs.as_slice()
    else {
        return Err(GpuRenderError::InvalidLaunch(
            "expected length, grid, buffer, argument, scheduling, and kernel",
        ));
    };
    if !matches!(kernel, GpuValue::FnSymbol(_)) {
        return Err(GpuRenderError::InvalidLaunch(
            "kernel input must be a direct function name",
        ));
    }
    Ok(LaunchInputs {
        length,
        grid,
        buffer,
        kernel_argument,
        scheduling,
        kernel,
    })
}
