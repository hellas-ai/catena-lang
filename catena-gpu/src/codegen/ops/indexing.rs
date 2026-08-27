use super::super::gpu::{GpuRenderError, invalid_arity, value_expr};
use crate::codegen::GpuAssign;

pub fn render(output: &mut String, assignment: &GpuAssign) -> Result<bool, GpuRenderError> {
    match assignment.op.as_str() {
        "ix.forget" => {
            if assignment.inputs.len() != 1 || !assignment.outputs.is_empty() {
                return Err(invalid_arity(assignment, 1, 0));
            }
        }
        "gpu.thread.in-grid.index" => {
            let [thread] = assignment.inputs.as_slice() else {
                return Err(invalid_arity(assignment, 1, 2));
            };
            let [thread_after, index] = assignment.outputs.as_slice() else {
                return Err(invalid_arity(assignment, 1, 2));
            };
            output.push_str(&format!(
                "    {} = {};\n    {} = {}.global_index;\n",
                thread_after.name,
                value_expr(thread),
                index.name,
                value_expr(thread),
            ));
        }
        "gpu.thread.in-block.index" => {
            let [thread] = assignment.inputs.as_slice() else {
                return Err(invalid_arity(assignment, 1, 2));
            };
            let [thread_after, index] = assignment.outputs.as_slice() else {
                return Err(invalid_arity(assignment, 1, 2));
            };
            output.push_str(&format!(
                "    {} = {};\n    {} = {}.in_block_index;\n",
                thread_after.name,
                value_expr(thread),
                index.name,
                value_expr(thread),
            ));
        }
        "gpu.thread.block" => {
            let [thread] = assignment.inputs.as_slice() else {
                return Err(invalid_arity(assignment, 1, 2));
            };
            let [thread_after, block] = assignment.outputs.as_slice() else {
                return Err(invalid_arity(assignment, 1, 2));
            };
            output.push_str(&format!(
                "    {} = {};\n    {} = {{ {}.block_index }};\n",
                thread_after.name,
                value_expr(thread),
                block.name,
                value_expr(thread),
            ));
        }
        "gpu.block.index" => {
            let [block] = assignment.inputs.as_slice() else {
                return Err(invalid_arity(assignment, 1, 2));
            };
            let [block_after, index] = assignment.outputs.as_slice() else {
                return Err(invalid_arity(assignment, 1, 2));
            };
            output.push_str(&format!(
                "    {} = {};\n    {} = {}.index;\n",
                block_after.name,
                value_expr(block),
                index.name,
                value_expr(block),
            ));
        }
        "ix.to-u64" => {
            let [index] = assignment.inputs.as_slice() else {
                return Err(invalid_arity(assignment, 1, 2));
            };
            let [index_after, offset] = assignment.outputs.as_slice() else {
                return Err(invalid_arity(assignment, 1, 2));
            };
            let index = value_expr(index);
            output.push_str(&format!(
                "    {} = {index};\n    {} = {index}.linear;\n",
                index_after.name, offset.name,
            ));
        }
        "ix.2d.intro" => {
            let [first, second] = assignment.inputs.as_slice() else {
                return Err(invalid_arity(assignment, 2, 1));
            };
            let [index] = assignment.outputs.as_slice() else {
                return Err(invalid_arity(assignment, 2, 1));
            };
            output.push_str(&format!(
                "    {} = {{ {}.linear, {}.linear, 0, 0 }};\n",
                index.name,
                value_expr(first),
                value_expr(second),
            ));
        }
        "ix.2d.elim" => {
            let [index] = assignment.inputs.as_slice() else {
                return Err(invalid_arity(assignment, 1, 2));
            };
            let [first, second] = assignment.outputs.as_slice() else {
                return Err(invalid_arity(assignment, 1, 2));
            };
            let index = value_expr(index);
            output.push_str(&format!(
                "    {} = {{ {index}.first, 0, 0, {index}.first }};\n    {} = {{ {index}.second, 0, 0, {index}.second }};\n",
                first.name, second.name,
            ));
        }
        "u64.to-ix" => {
            let [candidate, _bound] = assignment.inputs.as_slice() else {
                return Err(invalid_arity(assignment, 2, 1));
            };
            let [index] = assignment.outputs.as_slice() else {
                return Err(invalid_arity(assignment, 2, 1));
            };
            output.push_str(&format!(
                "    {} = {{ {}, 0, 0, {} }};\n",
                index.name,
                value_expr(candidate),
                value_expr(candidate),
            ));
        }
        _ => return Ok(false),
    }
    Ok(true)
}
