use super::*;

pub(super) fn render(out: &mut String, assignment: &GpuAssign) -> Result<bool, GpuRenderError> {
    match assignment.op.as_str() {
        "gpu.launch_params.2d-2d.intro" => render_launch_params(out, assignment)?,
        _ => return Ok(false),
    }
    Ok(true)
}

fn render_launch_params(out: &mut String, assignment: &GpuAssign) -> Result<(), GpuRenderError> {
    let components = input_components(assignment)?;
    let [grid, block] = components.as_slice() else {
        return Err(GpuRenderError::InvalidInputComponentCount {
            op: assignment.op.clone(),
            expected: 2,
            actual: components.len(),
        });
    };
    let grid = runtime_values(grid).collect::<Vec<_>>();
    let block = runtime_values(block).collect::<Vec<_>>();
    if grid.len() != 2 {
        return Err(invalid_runtime_inputs(assignment, 2, grid.len()));
    }
    if block.len() != 2 {
        return Err(invalid_runtime_inputs(assignment, 2, block.len()));
    }
    let outputs = runtime_outputs(assignment);
    let [output] = outputs.as_slice() else {
        return Err(invalid_runtime_outputs(assignment, 1, outputs.len()));
    };

    render_dim3_fields(out, &output.name, "grid_dim", &grid);
    render_dim3_fields(out, &output.name, "block_dim", &block);
    Ok(())
}

fn render_dim3_fields(out: &mut String, output: &str, field: &str, axes: &[&GpuValue]) {
    for (axis, value) in ["x", "y"].into_iter().zip(axes) {
        out.push_str(&format!(
            "    {output}.{field}.{axis} = (uint32_t){};\n",
            value_expr(value)
        ));
    }
    out.push_str(&format!("    {output}.{field}.z = 1U;\n"));
}

fn runtime_outputs(assignment: &GpuAssign) -> Vec<&GpuVar> {
    assignment
        .outputs
        .iter()
        .filter(|output| runtime_type(output).is_some())
        .collect()
}

fn invalid_runtime_inputs(
    assignment: &GpuAssign,
    expected: usize,
    actual: usize,
) -> GpuRenderError {
    GpuRenderError::InvalidInputComponentValueCount {
        op: assignment.op.clone(),
        component: "runtime inputs",
        description: "runtime values",
        expected,
        actual,
    }
}

fn invalid_runtime_outputs(
    assignment: &GpuAssign,
    expected: usize,
    actual: usize,
) -> GpuRenderError {
    GpuRenderError::InvalidOutputComponentCount {
        op: assignment.op.clone(),
        expected,
        actual,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use open_hypergraphs::lax::NodeId;

    fn op(name: &str) -> Operation {
        name.parse().unwrap()
    }

    fn var(node: usize, name: &str, ty: CType) -> GpuVar {
        GpuVar {
            node: NodeId(node),
            name: name.to_string(),
            lowered: LoweredType::Runtime(ty),
        }
    }

    #[test]
    fn structural_dimensions_construct_dim3_only_at_the_launch_boundary() {
        let grid_x = var(0, "grid_x", CType::U64);
        let grid_y = var(1, "grid_y", CType::U64);
        let block_x = var(2, "block_x", CType::U64);
        let block_y = var(3, "block_y", CType::U64);
        let params = var(4, "params", CType::Named("catena_launch_params_t".into()));
        let assignment = GpuAssign {
            op: op("gpu.launch_params.2d-2d.intro"),
            input_sizes: vec![2, 2],
            output_sizes: vec![1],
            call_symbol: None,
            inputs: vec![
                GpuValue::Var(grid_x),
                GpuValue::Var(grid_y),
                GpuValue::Var(block_x),
                GpuValue::Var(block_y),
            ],
            outputs: vec![params],
        };

        let mut out = String::new();
        assert!(render(&mut out, &assignment).unwrap());
        assert!(out.contains("params.grid_dim.x = (uint32_t)grid_x;"));
        assert!(out.contains("params.grid_dim.y = (uint32_t)grid_y;"));
        assert!(out.contains("params.block_dim.x = (uint32_t)block_x;"));
        assert!(out.contains("params.block_dim.y = (uint32_t)block_y;"));
    }
}
