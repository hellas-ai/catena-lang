//! GPU lowering for the generic parallel execution primitives.
//!
//! The topology remains visible in the lowered context type:
//!
//! ```text
//! grid context  + materializec  -> launch one GPU kernel
//! block context + materializec  -> cooperate in the current GPU block
//! ```
//!
//! Buffer modifiers (`writable` and `pending`) have already lowered to the
//! underlying pointer. The operations here preserve that pointer while adding
//! allocation, launch, or synchronization behavior.

use crate::codegen::{
    GpuAssign, GpuDialect, GpuFunction, GpuValue, GpuVar,
    components::{input_components, runtime_values, single_function, single_value, value_expr},
    gpu::{GpuRenderError, render_function_application},
    lower_types::{CType, LoweredType},
    render_utils::{c_type, invalid_inputs, invalid_outputs, param_decl},
    runtime_type,
};

const GRID_CONTEXT: &str = "catena_gpu_grid_context_t";
const BLOCK_CONTEXT: &str = "catena_gpu_block_context_t";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::codegen) enum ContextKind {
    Grid,
    Block,
}

pub(in crate::codegen) fn context_kind(assignment: &GpuAssign) -> Option<ContextKind> {
    assignment.inputs.iter().find_map(|input| {
        let GpuValue::Var(var) = input else {
            return None;
        };
        match runtime_type(var) {
            Some(CType::Named(name)) if name == GRID_CONTEXT => Some(ContextKind::Grid),
            Some(CType::Named(name)) if name == BLOCK_CONTEXT => Some(ContextKind::Block),
            _ => None,
        }
    })
}

pub(in crate::codegen) fn render_shape_2d(
    out: &mut String,
    assignment: &GpuAssign,
) -> Result<(), GpuRenderError> {
    let [x, y] = assignment.inputs.as_slice() else {
        return Err(invalid_inputs(assignment, 2));
    };
    let [shape] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 1));
    };
    out.push_str(&format!(
        "    {} = {{ (uint32_t){}, (uint32_t){}, 1 }};\n",
        shape.name,
        value_expr(x),
        value_expr(y)
    ));
    Ok(())
}

pub(in crate::codegen) fn render_grid_2d(
    out: &mut String,
    assignment: &GpuAssign,
) -> Result<(), GpuRenderError> {
    let [grid_x, grid_y, block_x, block_y] = assignment.inputs.as_slice() else {
        return Err(invalid_inputs(assignment, 4));
    };
    let [level] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 1));
    };
    out.push_str(&format!(
        "    {} = {{ {{(uint32_t){}, (uint32_t){}, 1}}, {{(uint32_t){}, (uint32_t){}, 1}} }};\n",
        level.name,
        value_expr(grid_x),
        value_expr(grid_y),
        value_expr(block_x),
        value_expr(block_y)
    ));
    Ok(())
}

pub(in crate::codegen) fn render_plan(
    out: &mut String,
    assignment: &GpuAssign,
) -> Result<(), GpuRenderError> {
    let [level, group_shape, worker_shape] = assignment.inputs.as_slice() else {
        return Err(invalid_inputs(assignment, 3));
    };
    let [plan] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 1));
    };
    out.push_str(&format!(
        "    {} = {{ {}, {} }};\n",
        plan.name,
        value_expr(group_shape),
        value_expr(worker_shape)
    ));
    out.push_str(&format!("    (void){};\n", value_expr(level)));
    Ok(())
}

pub(in crate::codegen) fn render_root_types(
    out: &mut String,
    assignment: &GpuAssign,
) -> Result<(), GpuRenderError> {
    let [plan] = assignment.inputs.as_slice() else {
        return Err(invalid_inputs(assignment, 1));
    };
    let [typed_plan, worker] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 2));
    };
    let plan = value_expr(plan);
    out.push_str(&format!("    {} = {};\n", typed_plan.name, plan));
    out.push_str(&format!("    {} = {{ {}, 0 }};\n", worker.name, plan));
    Ok(())
}

pub(in crate::codegen) fn render_schedule(
    out: &mut String,
    assignment: &GpuAssign,
) -> Result<(), GpuRenderError> {
    let [plan] = assignment.inputs.as_slice() else {
        return Err(invalid_inputs(assignment, 1));
    };
    let [context] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 1));
    };
    out.push_str(&format!(
        "    {} = {{ {} }};\n",
        context.name,
        value_expr(plan)
    ));
    Ok(())
}

pub(in crate::codegen) fn render_scope_worker_to_block(
    out: &mut String,
    assignment: &GpuAssign,
) -> Result<(), GpuRenderError> {
    let [worker] = assignment.inputs.as_slice() else {
        return Err(invalid_inputs(assignment, 1));
    };
    let [block_worker] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 1));
    };
    out.push_str(&format!(
        "    {} = {{ {}.launch, catena_block_worker_index() }};\n",
        block_worker.name,
        value_expr(worker)
    ));
    Ok(())
}

pub(in crate::codegen) fn render_worker_context(
    out: &mut String,
    assignment: &GpuAssign,
) -> Result<(), GpuRenderError> {
    let [worker] = assignment.inputs.as_slice() else {
        return Err(invalid_inputs(assignment, 1));
    };
    let [context, preserved_worker] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 2));
    };
    let worker = value_expr(worker);
    out.push_str(&format!(
        "    {} = {{ {}.launch }};\n",
        context.name, worker
    ));
    out.push_str(&format!("    {} = {};\n", preserved_worker.name, worker));
    Ok(())
}

pub(in crate::codegen) fn render_worker_index(
    out: &mut String,
    assignment: &GpuAssign,
) -> Result<(), GpuRenderError> {
    let [_context, worker] = assignment.inputs.as_slice() else {
        return Err(invalid_inputs(assignment, 2));
    };
    let [index] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 1));
    };
    out.push_str(&format!(
        "    {} = {}.index;\n",
        index.name,
        value_expr(worker)
    ));
    Ok(())
}

pub(in crate::codegen) fn render_allocate(
    out: &mut String,
    assignment: &GpuAssign,
    dialect: GpuDialect,
) -> Result<(), GpuRenderError> {
    let [context, len] = assignment.inputs.as_slice() else {
        return Err(invalid_inputs(assignment, 2));
    };
    let [next_context, buffer] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 2));
    };
    let CType::Pointer(element) =
        runtime_type(buffer).ok_or_else(|| GpuRenderError::ErasedType(buffer.clone()))?
    else {
        return Err(GpuRenderError::UnsupportedType(
            runtime_type(buffer).unwrap().clone(),
        ));
    };
    out.push_str(&format!(
        "    {} = {};\n",
        next_context.name,
        value_expr(context)
    ));

    match context_kind(assignment) {
        Some(ContextKind::Grid) => {
            out.push_str(&format!("    {} = nullptr;\n", buffer.name));
            out.push_str(&format!(
                "    if ({} != 0) catena_host_gpu_check({}((void **)&{}, {} * sizeof({})));\n",
                value_expr(len),
                dialect.device_alloc_fn(),
                buffer.name,
                value_expr(len),
                c_type(element)
            ));
        }
        Some(ContextKind::Block) => {
            // A static per-allocation shared arena keeps allocation local to
            // the current block. The bound is checked at runtime until shared
            // memory sizing becomes part of the launch plan.
            out.push_str(&format!(
                "#ifdef {}\n    __shared__ {} {}_storage[CATENA_BLOCK_BUFFER_CAPACITY];\n    catena_assert({} <= CATENA_BLOCK_BUFFER_CAPACITY);\n    {} = {}_storage;\n#else\n    {} = nullptr;\n#endif\n",
                dialect.device_compile_guard(),
                c_type(element),
                buffer.name,
                value_expr(len),
                buffer.name,
                buffer.name,
                buffer.name,
            ));
        }
        None => {
            return Err(GpuRenderError::UnsupportedType(
                runtime_type(next_context).unwrap().clone(),
            ));
        }
    }
    Ok(())
}

pub(in crate::codegen) fn render_synchronize(
    out: &mut String,
    assignment: &GpuAssign,
    dialect: GpuDialect,
) -> Result<(), GpuRenderError> {
    let runtime_inputs = assignment
        .inputs
        .iter()
        .filter(|input| matches!(input, GpuValue::Var(var) if runtime_type(var).is_some()))
        .collect::<Vec<_>>();
    let runtime_outputs = assignment
        .outputs
        .iter()
        .filter(|output| runtime_type(output).is_some())
        .collect::<Vec<_>>();
    if runtime_inputs.len() != runtime_outputs.len() {
        return Err(invalid_outputs(assignment, runtime_inputs.len()));
    }

    match context_kind(assignment) {
        Some(ContextKind::Grid) => out.push_str(&format!(
            "    catena_host_gpu_check({}());\n",
            dialect.synchronize_fn()
        )),
        Some(ContextKind::Block) => out.push_str("    catena_block_barrier();\n"),
        None => {}
    }
    for (input, output) in runtime_inputs.into_iter().zip(runtime_outputs) {
        out.push_str(&format!("    {} = {};\n", output.name, value_expr(input)));
    }
    Ok(())
}

pub(in crate::codegen) fn render_materialize_kernel(
    out: &mut String,
    kernel_name: &str,
    assignment: &GpuAssign,
) -> Result<(), GpuRenderError> {
    if context_kind(assignment) != Some(ContextKind::Grid) {
        return Ok(());
    }
    let parts = materialize_parts(assignment)?;
    let CType::Pointer(element) = runtime_type(parts.buffer_var)
        .ok_or_else(|| GpuRenderError::ErasedType(parts.buffer_var.clone()))?
    else {
        return Err(GpuRenderError::UnsupportedType(
            runtime_type(parts.buffer_var).unwrap().clone(),
        ));
    };

    out.push_str(&format!(
        "__global__ void {kernel_name}({} *out, catena_gpu_grid_context_t context",
        c_type(element)
    ));
    for value in runtime_values(parts.environment) {
        if let GpuValue::Var(var) = value {
            out.push_str(", ");
            out.push_str(&param_decl(var, false)?);
        }
    }
    out.push_str(") {\n");
    out.push_str("    uint64_t global_x = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;\n");
    out.push_str("    uint64_t global_y = (uint64_t)blockIdx.y * blockDim.y + threadIdx.y;\n");
    out.push_str(
        "    uint64_t index = global_y * ((uint64_t)gridDim.x * blockDim.x) + global_x;\n",
    );
    out.push_str("    catena_gpu_grid_worker_t worker = { context.launch, index };\n");
    out.push_str(&format!("    {} value;\n", c_type(element)));
    let mut producer_inputs = parts.environment.to_vec();
    producer_inputs.push(GpuValue::Var(GpuVar {
        node: parts.buffer_var.node,
        name: "worker".to_string(),
        lowered: LoweredType::Runtime(CType::Named("catena_gpu_grid_worker_t".to_string())),
    }));
    producer_inputs.push(GpuValue::Var(GpuVar {
        node: parts.buffer_var.node,
        name: "index".to_string(),
        lowered: LoweredType::Runtime(CType::U64),
    }));
    render_function_application(
        out,
        "    ",
        parts.function,
        &producer_inputs,
        &[GpuVar {
            node: parts.buffer_var.node,
            name: "value".to_string(),
            lowered: LoweredType::Runtime(element.as_ref().clone()),
        }],
    )?;
    out.push_str("    out[index] = value;\n");
    out.push_str("}\n");
    Ok(())
}

pub(in crate::codegen) fn render_materialize(
    out: &mut String,
    function: &GpuFunction,
    assignment: &GpuAssign,
) -> Result<(), GpuRenderError> {
    let parts = materialize_parts(assignment)?;
    let [next_context, pending_buffer] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 2));
    };
    match context_kind(assignment) {
        Some(ContextKind::Grid) => {
            let kernel_name = materialize_kernel_name(&function.name, assignment)?;
            let context = value_expr(parts.context);
            out.push_str(&format!(
                "    {kernel_name}<<<dim3({context}.launch.grid_dim.x, {context}.launch.grid_dim.y, {context}.launch.grid_dim.z), dim3({context}.launch.block_dim.x, {context}.launch.block_dim.y, {context}.launch.block_dim.z)>>>
                    ({}, {}",
                value_expr(parts.buffer),
                context,
            ));
            for value in runtime_values(parts.environment) {
                out.push_str(", ");
                out.push_str(&value_expr(value));
            }
            out.push_str(");\n");
        }
        Some(ContextKind::Block) => {
            let suffix = &parts.buffer_var.name;
            let index = format!("parallel_index_{suffix}");
            let worker = format!("parallel_worker_{suffix}");
            let value = format!("parallel_value_{suffix}");
            out.push_str(&format!(
                "    uint64_t {index} = catena_block_worker_index();\n"
            ));
            out.push_str(&format!(
                "    catena_gpu_block_worker_t {worker} = {{ {}.launch, {index} }};\n",
                value_expr(parts.context)
            ));
            let CType::Pointer(element) = runtime_type(parts.buffer_var)
                .ok_or_else(|| GpuRenderError::ErasedType(parts.buffer_var.clone()))?
            else {
                return Err(GpuRenderError::UnsupportedType(
                    runtime_type(parts.buffer_var).unwrap().clone(),
                ));
            };
            out.push_str(&format!("    {} {value};\n", c_type(element)));
            let mut producer_inputs = parts.environment.to_vec();
            producer_inputs.push(GpuValue::Var(GpuVar {
                node: parts.buffer_var.node,
                name: worker,
                lowered: LoweredType::Runtime(CType::Named(
                    "catena_gpu_block_worker_t".to_string(),
                )),
            }));
            producer_inputs.push(GpuValue::Var(GpuVar {
                node: parts.buffer_var.node,
                name: index.clone(),
                lowered: LoweredType::Runtime(CType::U64),
            }));
            render_function_application(
                out,
                "    ",
                parts.function,
                &producer_inputs,
                &[GpuVar {
                    node: parts.buffer_var.node,
                    name: value.clone(),
                    lowered: LoweredType::Runtime(element.as_ref().clone()),
                }],
            )?;
            out.push_str(&format!(
                "    {}[{index}] = {value};\n",
                value_expr(parts.buffer),
            ));
        }
        None => {
            return Err(GpuRenderError::UnsupportedType(
                runtime_type(next_context).unwrap().clone(),
            ));
        }
    }
    out.push_str(&format!(
        "    {} = {};\n    {} = {};\n",
        next_context.name,
        value_expr(parts.context),
        pending_buffer.name,
        value_expr(parts.buffer)
    ));
    Ok(())
}

pub(in crate::codegen) fn materialize_kernel_name(
    function_name: &str,
    assignment: &GpuAssign,
) -> Result<String, GpuRenderError> {
    let Some(buffer) = assignment.outputs.get(1) else {
        return Err(invalid_outputs(assignment, 2));
    };
    Ok(format!(
        "parallel_materialize_{function_name}_{}",
        buffer.name
    ))
}

struct MaterializeParts<'a> {
    context: &'a GpuValue,
    buffer: &'a GpuValue,
    buffer_var: &'a GpuVar,
    environment: &'a [GpuValue],
    function: &'a GpuValue,
}

fn materialize_parts(assignment: &GpuAssign) -> Result<MaterializeParts<'_>, GpuRenderError> {
    let components = input_components(assignment)?;
    let [context, buffer, environment, function] = components.as_slice() else {
        return Err(GpuRenderError::InvalidInputComponentCount {
            op: assignment.op.clone(),
            expected: 4,
            actual: components.len(),
        });
    };
    let context = single_value(context)
        .map_err(|error| component_error(assignment, "context", error.actual))?;
    let buffer = single_value(buffer)
        .map_err(|error| component_error(assignment, "buffer", error.actual))?;
    let GpuValue::Var(buffer_var) = buffer else {
        unreachable!("single runtime value must be a variable")
    };
    let function = single_function(function)
        .map_err(|error| component_error(assignment, "producer", error.actual))?;
    Ok(MaterializeParts {
        context,
        buffer,
        buffer_var,
        environment,
        function,
    })
}

fn component_error(
    assignment: &GpuAssign,
    component: &'static str,
    actual: usize,
) -> GpuRenderError {
    GpuRenderError::InvalidInputComponentValueCount {
        op: assignment.op.clone(),
        component,
        description: "runtime value",
        expected: 1,
        actual,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::fn_ptrs::FnPtrSymbol;
    use hexpr::Operation;
    use open_hypergraphs::lax::NodeId;

    #[test]
    fn grid_materialize_emits_a_kernel_launch() {
        let assignment = materialize_assignment(GRID_CONTEXT);
        let function = empty_function();
        let mut kernel = String::new();
        let mut call = String::new();

        render_materialize_kernel(&mut kernel, "parallel_kernel", &assignment).unwrap();
        render_materialize(&mut call, &function, &assignment).unwrap();

        assert!(kernel.contains("__global__ void parallel_kernel"));
        assert!(kernel.contains("catena_gpu_grid_worker_t worker"));
        assert!(call.contains("parallel_materialize_test_out<<<"));
    }

    #[test]
    fn block_materialize_runs_the_producer_in_the_current_block() {
        let assignment = materialize_assignment(BLOCK_CONTEXT);
        let function = empty_function();
        let mut kernel = String::new();
        let mut call = String::new();

        render_materialize_kernel(&mut kernel, "unused", &assignment).unwrap();
        render_materialize(&mut call, &function, &assignment).unwrap();

        assert!(kernel.is_empty());
        assert!(call.contains("parallel_index_buffer = catena_block_worker_index()"));
        assert!(call.contains("buffer[parallel_index_buffer] = parallel_value_buffer"));
        assert!(!call.contains("<<<"));
    }

    fn materialize_assignment(context_type: &str) -> GpuAssign {
        GpuAssign {
            op: op("parallel.materializec"),
            input_sizes: vec![1, 1, 0, 1],
            output_sizes: vec![1, 1],
            call_symbol: None,
            inputs: vec![
                value(0, "context", CType::Named(context_type.to_string())),
                value(1, "buffer", CType::Pointer(Box::new(CType::F32))),
                GpuValue::FnSymbol(FnPtrSymbol {
                    target: op("producer"),
                }),
            ],
            outputs: vec![
                var(2, "next_context", CType::Named(context_type.to_string())),
                var(3, "out", CType::Pointer(Box::new(CType::F32))),
            ],
        }
    }

    fn empty_function() -> GpuFunction {
        GpuFunction {
            name: "test".to_string(),
            sources: vec![],
            targets: vec![],
            assignments: vec![],
        }
    }

    fn value(node: usize, name: &str, ty: CType) -> GpuValue {
        GpuValue::Var(var(node, name, ty))
    }

    fn var(node: usize, name: &str, ty: CType) -> GpuVar {
        GpuVar {
            node: NodeId(node),
            name: name.to_string(),
            lowered: LoweredType::Runtime(ty),
        }
    }

    fn op(name: &str) -> Operation {
        name.parse().expect("test operation should parse")
    }
}
