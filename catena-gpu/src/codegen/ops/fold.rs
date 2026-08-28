use super::super::gpu::{GpuRenderError, c_type, value_expr};
use crate::codegen::{GpuAssign, GpuValue, runtime_type};

pub fn render(output: &mut String, assignment: &GpuAssign) -> Result<(), GpuRenderError> {
    let [head @ .., body] = assignment.inputs.as_slice() else {
        return Err(GpuRenderError::InvalidFold("missing initial state"));
    };
    if !matches!(body, GpuValue::FnSymbol(_)) {
        return Err(GpuRenderError::InvalidFold(
            "body must be a direct function name",
        ));
    }
    let Some((count, initial_state)) = head.split_last() else {
        return Err(GpuRenderError::InvalidFold("missing count"));
    };
    if assignment.outputs.is_empty() {
        return Err(GpuRenderError::InvalidFold("state must not be empty"));
    }
    if initial_state.len() != assignment.outputs.len() {
        return Err(GpuRenderError::InvalidFold(
            "initial and final state must have the same runtime shape",
        ));
    }

    let loop_index = format!("fold_index_{}", assignment.outputs[0].name);
    output.push_str("    {\n");
    for (state, initial) in assignment.outputs.iter().zip(initial_state) {
        output.push_str(&format!(
            "        {} = {};\n",
            state.name,
            value_expr(initial)
        ));
    }
    output.push_str(&format!(
        "        for (uint64_t {loop_index} = 0; {loop_index} < {}; ++{loop_index}) {{\n",
        value_expr(count)
    ));

    let next_state = assignment
        .outputs
        .iter()
        .map(|state| format!("fold_next_{}", state.name))
        .collect::<Vec<_>>();
    for (state, next) in assignment.outputs.iter().zip(&next_state) {
        output.push_str(&format!(
            "            {} {next};\n",
            c_type(runtime_type(state).expect("fold state has a runtime type"))
        ));
    }

    let mut arguments = vec![format!("catena_ix_t{{ {loop_index}, 0, 0 }}")];
    arguments.extend(assignment.outputs.iter().map(|state| state.name.clone()));
    arguments.extend(next_state.iter().map(|next| format!("&{next}")));
    output.push_str(&format!(
        "            {}({});\n",
        value_expr(body),
        arguments.join(", ")
    ));
    for (state, next) in assignment.outputs.iter().zip(&next_state) {
        output.push_str(&format!("            {} = {next};\n", state.name));
    }
    output.push_str("        }\n");
    output.push_str("    }\n");
    Ok(())
}
