use super::super::gpu::{GpuRenderError, c_type, invalid_arity, value_expr};
use crate::codegen::{GpuAssign, runtime_type};

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
                "    {}[{}.first] = {};\n",
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
                "    {} = {};\n    {} = {}[{}.first];\n",
                buffer_after_read.name,
                value_expr(buffer),
                value.name,
                value_expr(buffer),
                value_expr(cell),
            ));
        }
        "gpu.shared.write" => {
            let [thread, block, buffer, cell, value] = assignment.inputs.as_slice() else {
                return Err(invalid_arity(assignment, 5, 4));
            };
            let [thread_after, block_after, buffer_after, cell_after] =
                assignment.outputs.as_slice()
            else {
                return Err(invalid_arity(assignment, 5, 4));
            };
            output.push_str(&format!(
                "    {} = {};\n    {} = {};\n    {}[{}.first] = {};\n    {} = {};\n    {} = {};\n",
                thread_after.name,
                value_expr(thread),
                block_after.name,
                value_expr(block),
                value_expr(buffer),
                value_expr(cell),
                value_expr(value),
                buffer_after.name,
                value_expr(buffer),
                cell_after.name,
                value_expr(cell),
            ));
        }
        "gpu.shared.read" => {
            let [thread, block, buffer, cell] = assignment.inputs.as_slice() else {
                return Err(invalid_arity(assignment, 4, 5));
            };
            let [thread_after, block_after, buffer_after, cell_after, value] =
                assignment.outputs.as_slice()
            else {
                return Err(invalid_arity(assignment, 4, 5));
            };
            output.push_str(&format!(
                "    {} = {};\n    {} = {};\n    {} = {};\n    {} = {};\n    {} = {}[{}.first];\n",
                thread_after.name,
                value_expr(thread),
                block_after.name,
                value_expr(block),
                buffer_after.name,
                value_expr(buffer),
                cell_after.name,
                value_expr(cell),
                value.name,
                value_expr(buffer),
                value_expr(cell),
            ));
        }
        "gpu.shared.layout.open-pair" => {
            let [layout, thread, block] = assignment.inputs.as_slice() else {
                return Err(invalid_arity(assignment, 3, 4));
            };
            let [thread_after, block_after, first, second] = assignment.outputs.as_slice() else {
                return Err(invalid_arity(assignment, 3, 4));
            };
            let first_type = c_type(
                runtime_type(first)
                    .ok_or_else(|| GpuRenderError::UnsupportedOp(assignment.op.clone()))?,
            );
            let second_type = c_type(
                runtime_type(second)
                    .ok_or_else(|| GpuRenderError::UnsupportedOp(assignment.op.clone()))?,
            );
            output.push_str(&format!(
                "    {} = {};\n    {} = {};\n    {} = ({}){}.shared;\n    {} = ({})({}.shared + {} / 2);\n",
                thread_after.name,
                value_expr(thread),
                block_after.name,
                value_expr(block),
                first.name,
                first_type,
                value_expr(thread),
                second.name,
                second_type,
                value_expr(thread),
                value_expr(layout),
            ));
        }
        "gpu.shared.layout.close-pair" => {
            let [thread, _block, _first, _second, _cell] = assignment.inputs.as_slice() else {
                return Err(invalid_arity(assignment, 5, 1));
            };
            let [thread_after] = assignment.outputs.as_slice() else {
                return Err(invalid_arity(assignment, 5, 1));
            };
            output.push_str(&format!(
                "    {} = {};\n",
                thread_after.name,
                value_expr(thread),
            ));
        }
        "gpu.sync" => {
            let [block] = assignment.inputs.as_slice() else {
                return Err(invalid_arity(assignment, 1, 1));
            };
            let [block_after] = assignment.outputs.as_slice() else {
                return Err(invalid_arity(assignment, 1, 1));
            };
            output.push_str(&format!(
                "    catena_block_sync();\n    {} = {};\n",
                block_after.name,
                value_expr(block),
            ));
        }
        _ => return Ok(false),
    }
    Ok(true)
}
