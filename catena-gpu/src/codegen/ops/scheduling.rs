use super::super::gpu::{GpuRenderError, invalid_arity, value_expr};
use crate::codegen::GpuAssign;

pub fn render(output: &mut String, assignment: &GpuAssign) -> Result<bool, GpuRenderError> {
    match assignment.op.as_str() {
        "gpu.scheduling.own-each" => {
            if !assignment.inputs.is_empty() || assignment.outputs.len() != 1 {
                return Err(invalid_arity(assignment, 0, 1));
            }
            output.push_str(&format!(
                "    {} = {{ CATENA_SCHEDULING_OWN_EACH, 0 }};\n",
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
        "gpu.scheduling.can-own" => {
            let [schedule, thread, cell] = assignment.inputs.as_slice() else {
                return Err(invalid_arity(assignment, 3, 4));
            };
            let [schedule_after, thread_after, cell_after, decision] =
                assignment.outputs.as_slice()
            else {
                return Err(invalid_arity(assignment, 3, 4));
            };
            output.push_str(&format!(
                "    {} = {};\n    {} = {};\n    {} = {};\n    {} = (catena_scheduling_resolve({}, {}, {}) == CATENA_CELL_OWNED);\n",
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
