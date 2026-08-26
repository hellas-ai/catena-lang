use super::super::gpu::{GpuRenderError, invalid_arity, value_expr};
use crate::codegen::GpuAssign;

pub fn render(output: &mut String, assignment: &GpuAssign) -> Result<bool, GpuRenderError> {
    match assignment.op.as_str() {
        "gpu.scheduling.linear" => {
            if !assignment.inputs.is_empty() || assignment.outputs.len() != 1 {
                return Err(invalid_arity(assignment, 0, 1));
            }
            output.push_str(&format!(
                "    {} = {{ CATENA_SCHEDULING_LINEAR }};\n",
                assignment.outputs[0].name
            ));
        }
        operation @ ("gpu.scheduling.can-own" | "gpu.scheduling.can-read") => {
            let [schedule, thread, cell] = assignment.inputs.as_slice() else {
                return Err(invalid_arity(assignment, 3, 4));
            };
            let [schedule_after, thread_after, cell_after, decision] =
                assignment.outputs.as_slice()
            else {
                return Err(invalid_arity(assignment, 3, 4));
            };
            let permission_test = if operation == "gpu.scheduling.can-own" {
                "== CATENA_CELL_OWNED"
            } else {
                ">= CATENA_CELL_READABLE"
            };
            output.push_str(&format!(
                "    {} = {};\n    {} = {};\n    {} = {};\n    {} = (catena_scheduling_resolve({}, {}, {}) {permission_test});\n",
                schedule_after.name,
                value_expr(schedule),
                thread_after.name,
                value_expr(thread),
                cell_after.name,
                value_expr(cell),
                decision.name,
                value_expr(schedule),
                value_expr(thread),
                value_expr(cell),
            ));
        }
        _ => return Ok(false),
    }
    Ok(true)
}
