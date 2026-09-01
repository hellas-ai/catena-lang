use super::super::gpu::{GpuRenderError, c_type, value_expr};
use crate::codegen::{GpuAssign, GpuFunction, GpuModuleMap, GpuValue, runtime_type};

pub fn kernel_name(function: &GpuFunction, assignment_index: usize) -> String {
    format!("{}_launch_{assignment_index}", function.name)
}

pub fn render_kernel(
    output: &mut String,
    modules: &GpuModuleMap,
    function: &GpuFunction,
    assignment_index: usize,
    assignment: &GpuAssign,
) -> Result<(), GpuRenderError> {
    let LaunchInputs {
        grid: _,
        kernel_arguments,
        kernel,
    } = inputs(assignment)?;
    let GpuValue::FnSymbol(kernel_symbol) = kernel else {
        return Err(GpuRenderError::InvalidLaunch(
            "kernel input must be a direct function name",
        ));
    };
    let kernel_function = modules
        .get(kernel_symbol)
        .map(|module| &module.entry)
        .ok_or(GpuRenderError::InvalidLaunch(
            "kernel definition is missing",
        ))?;
    let wrapper = kernel_name(function, assignment_index);
    let mut parameters = Vec::new();
    for (index, argument) in kernel_arguments.iter().enumerate() {
        let GpuValue::Var(argument) = argument else {
            return Err(GpuRenderError::InvalidLaunch(
                "kernel arguments must be runtime values",
            ));
        };
        parameters.push(format!(
            "{} kernel_argument_{index}",
            c_type(runtime_type(argument).expect("GPU variables have runtime types"))
        ));
    }
    output.push_str(&format!(
        "extern \"C\" __global__ void {wrapper}({}) {{\n",
        parameters.join(", ")
    ));
    output.push_str("    uint64_t global_x = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;\n");
    output.push_str("    uint64_t global_y = (uint64_t)blockIdx.y * blockDim.y + threadIdx.y;\n");
    output.push_str("    uint64_t global_z = (uint64_t)blockIdx.z * blockDim.z + threadIdx.z;\n");
    output.push_str("    catena_ix_t global_index = { global_x, global_y, global_z };\n");
    output.push_str("    catena_ix_t in_block_index = { (uint64_t)threadIdx.x, (uint64_t)threadIdx.y, (uint64_t)threadIdx.z };\n");
    output.push_str("    catena_ix_t block_index = { (uint64_t)blockIdx.x, (uint64_t)blockIdx.y, (uint64_t)blockIdx.z };\n");
    output
        .push_str("    catena_thread_t thread = { global_index, in_block_index, block_index };\n");
    for (index, result) in kernel_function.targets.iter().enumerate() {
        output.push_str(&format!(
            "    {} kernel_result_{index};\n",
            c_type(runtime_type(result).expect("GPU variables have runtime types"))
        ));
    }
    let mut arguments = kernel_arguments
        .iter()
        .enumerate()
        .map(|(index, _)| format!("kernel_argument_{index}"))
        .collect::<Vec<_>>();
    arguments.push("thread".to_string());
    arguments.extend(
        kernel_function
            .targets
            .iter()
            .enumerate()
            .map(|(index, _)| format!("&kernel_result_{index}")),
    );
    output.push_str(&format!(
        "    {}({});\n",
        value_expr(kernel),
        arguments.join(", "),
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
        grid,
        kernel_arguments,
        kernel: _,
    } = inputs(assignment)?;
    if assignment.outputs.len() != kernel_arguments.len() {
        return Err(GpuRenderError::InvalidLaunch(
            "kernel arguments must be returned unchanged",
        ));
    }
    let wrapper = kernel_name(function, assignment_index);
    let grid = value_expr(grid);
    let arguments = kernel_arguments.iter().map(value_expr).collect::<Vec<_>>();
    output.push_str(&format!(
        "    {wrapper}<<<dim3({grid}.grid_dim.x, {grid}.grid_dim.y, {grid}.grid_dim.z), dim3({grid}.block_dim.x, {grid}.block_dim.y, {grid}.block_dim.z)>>>(\n        {});\n",
        arguments.join(", "),
    ));
    output.push_str("    catena_gpu_synchronize();\n");
    for (result, argument) in assignment.outputs.iter().zip(kernel_arguments) {
        output.push_str(&format!(
            "    {} = {};\n",
            result.name,
            value_expr(argument)
        ));
    }
    Ok(())
}

struct LaunchInputs<'a> {
    grid: &'a GpuValue,
    kernel_arguments: &'a [GpuValue],
    kernel: &'a GpuValue,
}

fn inputs(assignment: &GpuAssign) -> Result<LaunchInputs<'_>, GpuRenderError> {
    let [grid, kernel_arguments @ .., kernel] = assignment.inputs.as_slice() else {
        return Err(GpuRenderError::InvalidLaunch(
            "expected grid, kernel arguments, and kernel",
        ));
    };
    if !matches!(kernel, GpuValue::FnSymbol(_)) {
        return Err(GpuRenderError::InvalidLaunch(
            "kernel input must be a direct function name",
        ));
    }
    Ok(LaunchInputs {
        grid,
        kernel_arguments,
        kernel,
    })
}
