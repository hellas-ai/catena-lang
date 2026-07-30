use super::*;

pub(super) fn render_kernel(
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
    let parts = materialize_parts(assignment)?;
    let args = runtime_values(parts.kernel_env).collect::<Vec<_>>();

    out.push_str(&format!(
        "__global__ void {kernel_name}({} *out",
        c_type(element)
    ));
    for arg in &args {
        if let GpuValue::Var(var) = arg {
            if runtime_type(var).is_some() {
                out.push_str(", ");
                out.push_str(&param_decl(var, false)?);
            }
        }
    }
    out.push_str(") {\n");
    out.push_str("    uint64_t global_x = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;\n");
    out.push_str("    uint64_t global_y = (uint64_t)blockIdx.y * blockDim.y + threadIdx.y;\n");
    out.push_str("    uint64_t global_width = (uint64_t)gridDim.x * blockDim.x;\n");
    out.push_str("    uint64_t thread_id = global_y * global_width + global_x;\n");
    out.push_str(&format!("    {} value;\n", c_type(element)));
    let thread_id = GpuValue::Var(GpuVar {
        node: output.node,
        name: "thread_id".to_string(),
        lowered: LoweredType::Runtime(CType::U64),
    });
    let mut kernel_inputs = parts.kernel_env.to_vec();
    kernel_inputs.push(thread_id);
    let kernel_outputs = [GpuVar {
        node: output.node,
        name: "value".to_string(),
        lowered: LoweredType::Runtime(element.as_ref().clone()),
    }];
    render_function_application(out, "    ", parts.kernel, &kernel_inputs, &kernel_outputs)?;
    out.push_str("    out[thread_id] = value;\n");
    out.push_str("}\n");
    Ok(())
}

pub(super) fn render_call(
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
    let parts = materialize_parts(assignment)?;
    let args = runtime_values(parts.kernel_env).collect::<Vec<_>>();
    let launch = value_expr(parts.launch);
    let element_count = value_expr(parts.element_count);
    let kernel_name = kernel_name(&function.name, assignment)?;

    out.push_str(&format!(
        "    {} *{name}_data = nullptr;\n",
        c_type(element),
        name = output.name
    ));
    out.push_str(&format!(
        "    catena_host_gpu_check({managed_alloc_fn}((void **)&{name}_data, {element_count} * sizeof({element})));\n",
        name = output.name,
        element = c_type(element),
        managed_alloc_fn = dialect.managed_alloc_fn(),
    ));
    out.push_str(&format!(
        "    {kernel_name}<<<dim3({launch}.grid_dim.x, {launch}.grid_dim.y, {launch}.grid_dim.z), dim3({launch}.block_dim.x, {launch}.block_dim.y, {launch}.block_dim.z)>>>\n"
    ));
    out.push_str(&format!("        ({name}_data", name = output.name));
    for arg in args {
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
        synchronize_fn = dialect.synchronize_fn(),
    ));
    out.push_str(&format!("    {} = {}_data;\n", output.name, output.name));
    Ok(())
}

struct MaterializeParts<'a> {
    launch: &'a GpuValue,
    element_count: &'a GpuValue,
    kernel: &'a GpuValue,
    kernel_env: Component<'a>,
}

fn materialize_parts(assignment: &GpuAssign) -> Result<MaterializeParts<'_>, GpuRenderError> {
    // Closure conversion preserves the three source components:
    // launch parameters, element count, and the kernel closure.
    let components = input_components(assignment)?;
    let [launch, element_count, kernel] = components.as_slice() else {
        return Err(GpuRenderError::InvalidInputComponentCount {
            op: assignment.op.clone(),
            expected: 3,
            actual: components.len(),
        });
    };
    Ok(MaterializeParts {
        launch: materialize_single_value(assignment, launch, "launch", "launch parameters")?,
        element_count: materialize_single_value(
            assignment,
            element_count,
            "element count",
            "output element count",
        )?,
        kernel: materialize_single_function(
            assignment,
            kernel,
            "kernel",
            "value kernel function symbol",
        )?,
        kernel_env: kernel,
    })
}

fn materialize_single_value<'a>(
    assignment: &GpuAssign,
    component: Component<'a>,
    component_name: &'static str,
    description: &'static str,
) -> Result<&'a GpuValue, GpuRenderError> {
    single_value(component).map_err(|error| GpuRenderError::InvalidInputComponentValueCount {
        op: assignment.op.clone(),
        component: component_name,
        description,
        expected: 1,
        actual: error.actual,
    })
}

fn materialize_single_function<'a>(
    assignment: &GpuAssign,
    component: Component<'a>,
    component_name: &'static str,
    description: &'static str,
) -> Result<&'a GpuValue, GpuRenderError> {
    single_function(component).map_err(|error| GpuRenderError::InvalidInputComponentValueCount {
        op: assignment.op.clone(),
        component: component_name,
        description,
        expected: 1,
        actual: error.actual,
    })
}

pub(super) fn kernel_name(
    function_name: &str,
    assignment: &GpuAssign,
) -> Result<String, GpuRenderError> {
    let [output] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 1));
    };
    Ok(format!("materialize_{}_{}", function_name, output.name))
}
