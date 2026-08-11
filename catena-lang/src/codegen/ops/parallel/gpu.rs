//! CUDA/HIP lowering for generic parallel execution primitives.
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

use hexpr::Operation;
use metacat::tree::Tree;

use crate::codegen::{
    GpuAssign, GpuDialect, GpuFunction, GpuValue, GpuVar,
    components::{runtime_values, value_expr},
    gpu::{GpuRenderError, render_function_application},
    lower_types::{CType, LowerTypeError, LoweredType},
    render_utils::{c_type, invalid_inputs, invalid_outputs, param_decl},
    runtime_type,
};

use super::{materialize_parts, preserved_runtime_values};

const GRID_HOST: &str = "catena_gpu_grid_host_t";
const GRID_WORKER: &str = "catena_gpu_grid_worker_t";
const BLOCK_WORKER: &str = "catena_gpu_block_worker_t";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::codegen) enum RunnerKind {
    GridHost,
    BlockWorker,
}

/// Whether this backend supplies a concrete representation for the parallel
/// type constructor. Generic parallel decoding does not choose these types.
pub(in crate::codegen) fn is_runtime_type(name: &str) -> bool {
    matches!(name, "host" | "worker")
}

/// Choose the CUDA/HIP runtime representation of a parallel type.
pub(in crate::codegen) fn lower_runtime_type(
    name: &str,
    children: &[Tree<(), Operation>],
) -> Result<CType, LowerTypeError> {
    let (expected, c_name) = match name {
        "host" => (3, scoped_runtime_name(name, children)?),
        "worker" => (4, scoped_runtime_name(name, children)?),
        _ => unreachable!("checked by is_runtime_type"),
    };
    if children.len() != expected {
        return Err(LowerTypeError::InvalidArity {
            name: name.to_string(),
            expected,
            actual: children.len(),
        });
    }
    Ok(CType::Named(c_name.to_string()))
}

fn scoped_runtime_name<'a>(
    type_name: &str,
    children: &'a [Tree<(), Operation>],
) -> Result<&'a str, LowerTypeError> {
    let Some(Tree::Node(level, _, _)) = children.first() else {
        return Err(no_runtime_representation(type_name, children));
    };
    match (type_name, level.as_str()) {
        ("host", "gpu.grid.2d") => Ok(GRID_HOST),
        ("worker", "gpu.grid.2d") => Ok(GRID_WORKER),
        ("worker", "gpu.block.2d") => Ok(BLOCK_WORKER),
        _ => Err(no_runtime_representation(type_name, children)),
    }
}

fn no_runtime_representation(type_name: &str, children: &[Tree<(), Operation>]) -> LowerTypeError {
    LowerTypeError::NoRuntimeRepresentation(Tree::Node(
        type_name.parse().expect("type name should parse"),
        0,
        children.to_vec(),
    ))
}

pub(in crate::codegen) fn runner_kind(assignment: &GpuAssign) -> Option<RunnerKind> {
    assignment.inputs.iter().find_map(|input| {
        let GpuValue::Var(var) = input else {
            return None;
        };
        match runtime_type(var) {
            Some(CType::Named(name)) if name == GRID_HOST => Some(RunnerKind::GridHost),
            Some(CType::Named(name)) if name == BLOCK_WORKER => Some(RunnerKind::BlockWorker),
            _ => None,
        }
    })
}

pub(in crate::codegen) fn render_grid_2d(
    out: &mut String,
    assignment: &GpuAssign,
) -> Result<(), GpuRenderError> {
    let inputs = runtime_values(&assignment.inputs).collect::<Vec<_>>();
    let [_, _, _, _] = inputs.as_slice() else {
        return Err(invalid_inputs(assignment, 4));
    };
    let outputs = assignment
        .outputs
        .iter()
        .filter(|output| runtime_type(output).is_some())
        .collect::<Vec<_>>();
    let [_, _, _, _] = outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 4));
    };
    for (input, output) in inputs.into_iter().zip(outputs) {
        out.push_str(&format!("    {} = {};\n", output.name, value_expr(input)));
    }
    Ok(())
}

pub(in crate::codegen) fn render_block_2d(
    out: &mut String,
    assignment: &GpuAssign,
) -> Result<(), GpuRenderError> {
    let inputs = runtime_values(&assignment.inputs).collect::<Vec<_>>();
    let [_, _] = inputs.as_slice() else {
        return Err(invalid_inputs(assignment, 2));
    };
    let outputs = assignment
        .outputs
        .iter()
        .filter(|output| runtime_type(output).is_some())
        .collect::<Vec<_>>();
    let [_, _] = outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 2));
    };
    for (input, output) in inputs.into_iter().zip(outputs) {
        out.push_str(&format!("    {} = {};\n", output.name, value_expr(input)));
    }
    Ok(())
}

pub(in crate::codegen) fn render_host(
    out: &mut String,
    assignment: &GpuAssign,
) -> Result<(), GpuRenderError> {
    let inputs = runtime_values(&assignment.inputs).collect::<Vec<_>>();
    let [grid_x, grid_y, block_x, block_y] = inputs.as_slice() else {
        return Err(invalid_inputs(assignment, 4));
    };
    let [context] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 1));
    };
    out.push_str(&format!(
        "    {} = {{ {{ {{(uint32_t){}, (uint32_t){}, 1}}, {{(uint32_t){}, (uint32_t){}, 1}} }} }};\n",
        context.name,
        value_expr(grid_x),
        value_expr(grid_y),
        value_expr(block_x),
        value_expr(block_y)
    ));
    Ok(())
}

pub(in crate::codegen) fn render_worker_index(
    out: &mut String,
    assignment: &GpuAssign,
) -> Result<(), GpuRenderError> {
    let [worker] = assignment.inputs.as_slice() else {
        return Err(invalid_inputs(assignment, 1));
    };
    let [preserved_worker, x, y] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 3));
    };
    let GpuValue::Var(worker_var) = worker else {
        return Err(GpuRenderError::UnsupportedOp(assignment.op.clone()));
    };
    let worker = value_expr(worker);
    out.push_str(&format!("    {} = {};\n", preserved_worker.name, worker));
    let width = match runtime_type(worker_var) {
        Some(CType::Named(name)) if name == GRID_WORKER => {
            format!("((uint64_t){worker}.launch.grid_dim.x * {worker}.launch.block_dim.x)")
        }
        Some(CType::Named(name)) if name == BLOCK_WORKER => {
            format!("(uint64_t){worker}.launch.block_dim.x")
        }
        _ => return Err(GpuRenderError::UnsupportedOp(assignment.op.clone())),
    };
    out.push_str(&format!(
        "    {} = {}.index % {};\n    {} = {}.index / {};\n",
        x.name, worker, width, y.name, worker, width
    ));
    Ok(())
}

pub(in crate::codegen) fn render_worker(
    out: &mut String,
    assignment: &GpuAssign,
) -> Result<(), GpuRenderError> {
    let inputs = runtime_values(&assignment.inputs).collect::<Vec<_>>();
    let [parent_worker, child_x, child_y] = inputs.as_slice() else {
        return Err(invalid_inputs(assignment, 3));
    };
    let [child_worker] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 1));
    };
    out.push_str(&format!(
        "    {0} = {{ {1}.launch, {3} * (uint64_t){1}.launch.block_dim.x + {2} }};\n",
        child_worker.name,
        value_expr(parent_worker),
        value_expr(child_x),
        value_expr(child_y)
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
    let [next_runner, buffer] = assignment.outputs.as_slice() else {
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
        next_runner.name,
        value_expr(context)
    ));

    match runner_kind(assignment) {
        Some(RunnerKind::GridHost) => {
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
        Some(RunnerKind::BlockWorker) => {
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
                runtime_type(next_runner).unwrap().clone(),
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
    let preserved = preserved_runtime_values(assignment)?;

    match runner_kind(assignment) {
        Some(RunnerKind::GridHost) => out.push_str(&format!(
            "    catena_host_gpu_check({}());\n",
            dialect.synchronize_fn()
        )),
        Some(RunnerKind::BlockWorker) => out.push_str("    catena_block_barrier();\n"),
        None => {}
    }
    for (input, output) in preserved.inputs.into_iter().zip(preserved.outputs) {
        out.push_str(&format!("    {} = {};\n", output.name, value_expr(input)));
    }
    Ok(())
}

pub(in crate::codegen) fn render_materialize_kernel(
    out: &mut String,
    kernel_name: &str,
    assignment: &GpuAssign,
) -> Result<(), GpuRenderError> {
    if runner_kind(assignment) != Some(RunnerKind::GridHost) {
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
        "__global__ void {kernel_name}({} *out, catena_gpu_grid_host_t context",
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
    let [next_runner, pending_buffer] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 2));
    };
    match runner_kind(assignment) {
        Some(RunnerKind::GridHost) => {
            let kernel_name = materialize_kernel_name(&function.name, assignment)?;
            let context = value_expr(parts.runner);
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
        Some(RunnerKind::BlockWorker) => {
            let suffix = &parts.buffer_var.name;
            let index = format!("parallel_index_{suffix}");
            let worker = value_expr(parts.runner);
            let value = format!("parallel_value_{suffix}");
            out.push_str(&format!("    uint64_t {index} = {worker}.index;\n"));
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
                runtime_type(next_runner).unwrap().clone(),
            ));
        }
    }
    out.push_str(&format!(
        "    {} = {};\n    {} = {};\n",
        next_runner.name,
        value_expr(parts.runner),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::fn_ptrs::FnPtrSymbol;
    use hexpr::Operation;
    use open_hypergraphs::lax::NodeId;

    #[test]
    fn grid_materialize_emits_a_kernel_launch() {
        let assignment = materialize_assignment(GRID_HOST);
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
        let assignment = materialize_assignment(BLOCK_WORKER);
        let function = empty_function();
        let mut kernel = String::new();
        let mut call = String::new();

        render_materialize_kernel(&mut kernel, "unused", &assignment).unwrap();
        render_materialize(&mut call, &function, &assignment).unwrap();

        assert!(kernel.is_empty());
        assert!(call.contains("parallel_index_buffer = runner.index"));
        assert!(call.contains("buffer[parallel_index_buffer] = parallel_value_buffer"));
        assert!(!call.contains("<<<"));
    }

    fn materialize_assignment(runner_type: &str) -> GpuAssign {
        GpuAssign {
            op: op("parallel.materializec"),
            input_sizes: vec![1, 1, 0, 1],
            output_sizes: vec![1, 1],
            call_symbol: None,
            inputs: vec![
                value(0, "runner", CType::Named(runner_type.to_string())),
                value(1, "buffer", CType::Pointer(Box::new(CType::F32))),
                GpuValue::FnSymbol(FnPtrSymbol {
                    target: op("producer"),
                }),
            ],
            outputs: vec![
                var(2, "next_runner", CType::Named(runner_type.to_string())),
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
