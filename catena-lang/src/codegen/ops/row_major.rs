use crate::codegen::{
    GpuAssign, GpuValue,
    components::value_expr,
    gpu::GpuRenderError,
    render_utils::{invalid_inputs, invalid_outputs},
    runtime_type,
};

pub(in crate::codegen) fn render_index(
    out: &mut String,
    assignment: &GpuAssign,
) -> Result<(), GpuRenderError> {
    let inputs = runtime_inputs(assignment);
    let [cols, row, col] = inputs.as_slice() else {
        return Err(invalid_inputs(assignment, 3));
    };
    let [output] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 1));
    };
    out.push_str(&format!(
        "    {} = {} * {} + {};\n",
        output.name,
        value_expr(row),
        value_expr(cols),
        value_expr(col)
    ));
    Ok(())
}

pub(in crate::codegen) fn render_row(
    out: &mut String,
    assignment: &GpuAssign,
) -> Result<(), GpuRenderError> {
    let inputs = runtime_inputs(assignment);
    let [cols, flat] = inputs.as_slice() else {
        return Err(invalid_inputs(assignment, 2));
    };
    let [output] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 1));
    };
    out.push_str(&format!(
        "    {} = {} / {};\n",
        output.name,
        value_expr(flat),
        value_expr(cols)
    ));
    Ok(())
}

pub(in crate::codegen) fn render_col(
    out: &mut String,
    assignment: &GpuAssign,
) -> Result<(), GpuRenderError> {
    let inputs = runtime_inputs(assignment);
    let [cols, flat] = inputs.as_slice() else {
        return Err(invalid_inputs(assignment, 2));
    };
    let [output] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 1));
    };
    out.push_str(&format!(
        "    {} = {} % {};\n",
        output.name,
        value_expr(flat),
        value_expr(cols)
    ));
    Ok(())
}

fn runtime_inputs(assignment: &GpuAssign) -> Vec<&GpuValue> {
    assignment
        .inputs
        .iter()
        .filter(|value| matches!(value, GpuValue::Var(var) if runtime_type(var).is_some()))
        .collect()
}
