use super::*;

pub(super) fn render(out: &mut String, assignment: &GpuAssign) -> Result<bool, GpuRenderError> {
    match assignment.op.as_str() {
        "gpu.shared.alloc" => render_alloc(out, assignment)?,
        "gpu.shared.row-major.cooperative-loadc" => {
            render_row_major_cooperative_load(out, assignment)?
        }
        "gpu.shared.materialize" => render_runtime_identity(out, assignment)?,
        "gpu.sync" => out.push_str("    __syncthreads();\n"),
        _ => return Ok(false),
    }
    Ok(true)
}

fn render_alloc(out: &mut String, assignment: &GpuAssign) -> Result<(), GpuRenderError> {
    let inputs = runtime_inputs(assignment);
    let outputs = runtime_outputs(assignment);
    let [len] = inputs.as_slice() else {
        return Err(invalid_runtime_inputs(assignment, 1, inputs.len()));
    };
    let [slot] = outputs.as_slice() else {
        return Err(invalid_runtime_outputs(assignment, 1, outputs.len()));
    };
    let CType::Pointer(element) =
        runtime_type(slot).ok_or_else(|| GpuRenderError::ErasedType((*slot).clone()))?
    else {
        return Err(GpuRenderError::UnsupportedType(
            runtime_type(slot).unwrap().clone(),
        ));
    };
    out.push_str(&format!(
        "    __shared__ {} {}_storage[CATENA_GPU_MAX_SHARED_F32_SLOT_ELEMENTS];\n",
        c_type(element),
        slot.name
    ));
    out.push_str(&format!(
        "    catena_assert({} <= CATENA_GPU_MAX_SHARED_F32_SLOT_ELEMENTS);\n",
        value_expr(len)
    ));
    out.push_str(&format!("    {} = {}_storage;\n", slot.name, slot.name));
    Ok(())
}

fn render_row_major_cooperative_load(
    out: &mut String,
    assignment: &GpuAssign,
) -> Result<(), GpuRenderError> {
    let components = input_components(assignment)?;
    let [
        _,
        _,
        slot_component,
        rows_component,
        cols_component,
        source_environment,
        source_function,
    ] = components.as_slice()
    else {
        return Err(GpuRenderError::InvalidInputComponentCount {
            op: assignment.op.clone(),
            expected: 7,
            actual: components.len(),
        });
    };
    let slot = single_runtime_value(assignment, *slot_component, "slot")?;
    let _rows = single_runtime_value(assignment, *rows_component, "rows")?;
    let cols = single_runtime_value(assignment, *cols_component, "cols")?;
    let source = single_function(*source_function).map_err(|error| {
        GpuRenderError::InvalidInputComponentValueCount {
            op: assignment.op.clone(),
            component: "source function",
            description: "source view function symbol",
            expected: 1,
            actual: error.actual,
        }
    })?;
    let outputs = runtime_outputs(assignment);
    let [next_slot] = outputs.as_slice() else {
        return Err(invalid_runtime_outputs(assignment, 1, outputs.len()));
    };
    let CType::Pointer(element) =
        runtime_type(next_slot).ok_or_else(|| GpuRenderError::ErasedType((*next_slot).clone()))?
    else {
        return Err(GpuRenderError::UnsupportedType(
            runtime_type(next_slot).unwrap().clone(),
        ));
    };

    let row_name = format!("{}_row", next_slot.name);
    let col_name = format!("{}_col", next_slot.name);
    let value_name = format!("{}_value", next_slot.name);
    out.push_str(&format!(
        "    uint64_t {row_name} = (uint64_t)threadIdx.y;\n"
    ));
    out.push_str(&format!(
        "    uint64_t {col_name} = (uint64_t)threadIdx.x;\n"
    ));
    out.push_str(&format!("    {} {value_name};\n", c_type(element)));
    let row = synthetic_var(next_slot, &row_name, CType::U64);
    let col = synthetic_var(next_slot, &col_name, CType::U64);
    let value = synthetic_var(next_slot, &value_name, element.as_ref().clone());
    let mut source_inputs = runtime_values(*source_environment)
        .cloned()
        .collect::<Vec<_>>();
    source_inputs.push(GpuValue::Var(row));
    source_inputs.push(GpuValue::Var(col));
    render_function_application(
        out,
        "    ",
        source,
        &source_inputs,
        std::slice::from_ref(&value),
    )?;
    out.push_str(&format!(
        "    {}[{row_name} * {} + {col_name}] = {value_name};\n",
        value_expr(slot),
        value_expr(cols)
    ));
    out.push_str(&format!("    {} = {};\n", next_slot.name, value_expr(slot)));
    Ok(())
}

fn render_runtime_identity(out: &mut String, assignment: &GpuAssign) -> Result<(), GpuRenderError> {
    let inputs = runtime_inputs(assignment);
    let outputs = runtime_outputs(assignment);
    if inputs.len() != outputs.len() {
        return Err(invalid_runtime_outputs(
            assignment,
            inputs.len(),
            outputs.len(),
        ));
    }
    for (input, output) in inputs.iter().zip(outputs) {
        out.push_str(&format!("    {} = {};\n", output.name, value_expr(input)));
    }
    Ok(())
}

fn single_runtime_value<'a>(
    assignment: &GpuAssign,
    component: Component<'a>,
    component_name: &'static str,
) -> Result<&'a GpuValue, GpuRenderError> {
    single_value(component).map_err(|error| GpuRenderError::InvalidInputComponentValueCount {
        op: assignment.op.clone(),
        component: component_name,
        description: "runtime value",
        expected: 1,
        actual: error.actual,
    })
}

fn synthetic_var(reference: &GpuVar, name: &str, ty: CType) -> GpuVar {
    GpuVar {
        node: reference.node,
        name: name.to_string(),
        lowered: LoweredType::Runtime(ty),
    }
}

fn runtime_inputs(assignment: &GpuAssign) -> Vec<&GpuValue> {
    assignment
        .inputs
        .iter()
        .filter(|input| match input {
            GpuValue::Var(var) => runtime_type(var).is_some(),
            GpuValue::FnSymbol(_) => true,
        })
        .collect()
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
    use crate::codegen::fn_ptrs::FnPtrSymbol;
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

    fn erased_var(node: usize, name: &str) -> GpuVar {
        GpuVar {
            node: NodeId(node),
            name: name.to_string(),
            lowered: LoweredType::Erased,
        }
    }

    #[test]
    fn shared_alloc_creates_one_block_local_array_per_slot() {
        let state = erased_var(0, "state");
        let len = var(0, "len", CType::U64);
        let slot = var(1, "slot", CType::Pointer(Box::new(CType::F32)));
        let assignment = GpuAssign {
            op: op("gpu.shared.alloc"),
            input_sizes: vec![1, 1],
            output_sizes: vec![0, 0, 1],
            call_symbol: None,
            inputs: vec![GpuValue::Var(state), GpuValue::Var(len)],
            outputs: vec![slot],
        };

        let mut out = String::new();
        assert!(render(&mut out, &assignment).unwrap());
        assert!(out.contains("__shared__ float slot_storage"));
        assert!(out.contains("catena_assert(len <= CATENA_GPU_MAX_SHARED_F32_SLOT_ELEMENTS);"));
        assert!(out.contains("slot = slot_storage;"));
    }

    #[test]
    fn cooperative_load_calls_the_view_for_the_current_block_thread() {
        let state = erased_var(0, "state");
        let allocated = erased_var(1, "allocated");
        let slot = var(0, "slot", CType::Pointer(Box::new(CType::F32)));
        let rows = var(1, "rows", CType::U64);
        let cols = var(2, "cols", CType::U64);
        let capture = var(3, "capture", CType::Pointer(Box::new(CType::F32)));
        let next_slot = var(4, "next_slot", CType::Pointer(Box::new(CType::F32)));
        let assignment = GpuAssign {
            op: op("gpu.shared.row-major.cooperative-loadc"),
            input_sizes: vec![1, 1, 1, 1, 1, 1, 1],
            output_sizes: vec![0, 1],
            call_symbol: None,
            inputs: vec![
                GpuValue::Var(state),
                GpuValue::Var(allocated),
                GpuValue::Var(slot),
                GpuValue::Var(rows),
                GpuValue::Var(cols),
                GpuValue::Var(capture),
                GpuValue::FnSymbol(FnPtrSymbol {
                    target: op("tile-view"),
                }),
            ],
            outputs: vec![next_slot],
        };

        let mut out = String::new();
        assert!(render(&mut out, &assignment).unwrap());
        assert!(out.contains("next_slot_row = (uint64_t)threadIdx.y;"));
        assert!(out.contains("next_slot_col = (uint64_t)threadIdx.x;"));
        assert!(out.contains(
            "program_tile_view(capture, next_slot_row, next_slot_col, &next_slot_value);"
        ));
        assert!(out.contains("slot[next_slot_row * cols + next_slot_col] = next_slot_value;"));
        assert!(out.contains("next_slot = slot;"));
    }

    #[test]
    fn gpu_sync_is_a_block_barrier() {
        let state = erased_var(0, "state");
        let next_state = erased_var(1, "next_state");
        let assignment = GpuAssign {
            op: op("gpu.sync"),
            input_sizes: vec![1],
            output_sizes: vec![1],
            call_symbol: None,
            inputs: vec![GpuValue::Var(state)],
            outputs: vec![next_state],
        };

        let mut out = String::new();
        assert!(render(&mut out, &assignment).unwrap());
        assert_eq!(out, "    __syncthreads();\n");
    }
}
