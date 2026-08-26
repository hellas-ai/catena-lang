use super::super::gpu::{GpuRenderError, invalid_arity, value_expr};
use crate::codegen::GpuAssign;

pub fn render(output: &mut String, assignment: &GpuAssign) -> Result<bool, GpuRenderError> {
    match assignment.op.as_str() {
        "gpu.global.forget" => {
            if assignment.inputs.len() != 1 || !assignment.outputs.is_empty() {
                return Err(invalid_arity(assignment, 1, 0));
            }
        }
        "mem.cast.u64" => {
            let [memory] = assignment.inputs.as_slice() else {
                return Err(invalid_arity(assignment, 1, 2));
            };
            let [length, buffer] = assignment.outputs.as_slice() else {
                return Err(invalid_arity(assignment, 1, 2));
            };
            output.push_str(&format!(
                "    {length} = {memory}.len / sizeof(uint64_t);\n    {buffer} = (uint64_t *){memory}.data;\n",
                length = length.name,
                buffer = buffer.name,
                memory = value_expr(memory),
            ));
        }
        "gpu.global.to-mem" => {
            let [length, buffer] = assignment.inputs.as_slice() else {
                return Err(invalid_arity(assignment, 2, 1));
            };
            let [memory] = assignment.outputs.as_slice() else {
                return Err(invalid_arity(assignment, 2, 1));
            };
            output.push_str(&format!(
                "    {memory}.data = (void *){buffer};\n    {memory}.len = {length} * sizeof(uint64_t);\n",
                memory = memory.name,
                buffer = value_expr(buffer),
                length = value_expr(length),
            ));
        }
        "gpu.global.write" => {
            let [_thread, buffer, cell, value] = assignment.inputs.as_slice() else {
                return Err(invalid_arity(assignment, 4, 0));
            };
            if !assignment.outputs.is_empty() {
                return Err(invalid_arity(assignment, 4, 0));
            }
            output.push_str(&format!(
                "    {}[{}] = {};\n",
                value_expr(buffer),
                value_expr(cell),
                value_expr(value),
            ));
        }
        "gpu.global.read" => {
            let [buffer, cell] = assignment.inputs.as_slice() else {
                return Err(invalid_arity(assignment, 2, 2));
            };
            let [buffer_after_read, value] = assignment.outputs.as_slice() else {
                return Err(invalid_arity(assignment, 2, 2));
            };
            output.push_str(&format!(
                "    {} = {};\n    {} = {}[{}];\n",
                buffer_after_read.name,
                value_expr(buffer),
                value.name,
                value_expr(buffer),
                value_expr(cell),
            ));
        }
        _ => return Ok(false),
    }
    Ok(true)
}
