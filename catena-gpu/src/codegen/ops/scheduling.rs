use super::super::gpu::{GpuRenderError, invalid_arity, value_expr};
use crate::codegen::GpuAssign;

pub fn render(output: &mut String, assignment: &GpuAssign) -> Result<bool, GpuRenderError> {
    match assignment.op.as_str() {
        "gpu.scheduling.forget" => {
            if assignment.inputs.len() != 1 || !assignment.outputs.is_empty() {
                return Err(invalid_arity(assignment, 1, 0));
            }
        }
        operation @ ("gpu.scheduling.own-each"
        | "gpu.scheduling.read-all"
        | "gpu.scheduling.2d.row-major.read-all") => {
            if !assignment.inputs.is_empty() || assignment.outputs.len() != 1 {
                return Err(invalid_arity(assignment, 0, 1));
            }
            let kind = if operation == "gpu.scheduling.own-each" {
                "CATENA_SCHEDULING_OWN_EACH"
            } else {
                "CATENA_SCHEDULING_READ_ALL"
            };
            output.push_str(&format!(
                "    {} = {{ {kind}, 0 }};\n",
                assignment.outputs[0].name,
            ));
        }
        "gpu.scheduling.2d.row-major.own-each" => {
            let [column_count] = assignment.inputs.as_slice() else {
                return Err(invalid_arity(assignment, 1, 1));
            };
            let [scheduling] = assignment.outputs.as_slice() else {
                return Err(invalid_arity(assignment, 1, 1));
            };
            output.push_str(&format!(
                "    {} = {{ CATENA_SCHEDULING_2D_ROW_MAJOR_OWN, {} }};\n",
                scheduling.name,
                value_expr(column_count),
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
