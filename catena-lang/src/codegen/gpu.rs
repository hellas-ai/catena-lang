use std::collections::HashSet;

use hexpr::Operation;
use thiserror::Error;

use crate::codegen::{
    GpuAssign, GpuFunction, GpuModule, GpuModuleMap, GpuValue, GpuVar, lower_types::CType,
    prelude::GPU_PRELUDE, runtime_type,
};

#[derive(Debug, Error)]
pub enum GpuRenderError {
    #[error("type `{0:?}` is erased and cannot be rendered here")]
    ErasedType(GpuVar),
    #[error("type `{0:?}` is unsupported in GPU renderer")]
    UnsupportedType(CType),
    #[error("unsupported GPU operation `{0}`")]
    UnsupportedOp(Operation),
    #[error("operation `{op}` expected {expected} inputs, found {actual}")]
    InvalidInputCount {
        op: Operation,
        expected: usize,
        actual: usize,
    },
    #[error("operation `{op}` expected {expected} outputs, found {actual}")]
    InvalidOutputCount {
        op: Operation,
        expected: usize,
        actual: usize,
    },
    #[error("gpu.materialize is missing launch params")]
    MissingMaterializeLaunchParams,
    #[error("gpu.materialize is missing function input")]
    MissingMaterializeFunction,
}

/// Render a single GPU dataflow module as standalone HIP/C++ source.
///
/// Codegen has already produced the semantic `GpuModule`; this renderer is responsible only for
/// turning that artifact into text. It still contains primitive-specific lowering for the current
/// small backend surface, but it should not inspect the original Catena graph or report state.
pub fn render_module(module: &GpuModule) -> Result<String, GpuRenderError> {
    let mut out = String::new();
    out.push_str(GPU_PRELUDE);
    out.push('\n');
    render_module_body(&mut out, module)?;
    Ok(out)
}

/// Render all generated GPU modules into one HIP/C++ translation unit.
pub fn render_modules(modules: &GpuModuleMap) -> Result<String, GpuRenderError> {
    let mut out = String::new();
    out.push_str(GPU_PRELUDE);
    out.push('\n');

    for module in modules.values() {
        render_function_decl(&mut out, &module.entry)?;
    }
    if !modules.is_empty() {
        out.push('\n');
    }

    for module in modules.values() {
        render_module_body(&mut out, module)?;
        out.push('\n');
    }

    Ok(out)
}

fn render_module_body(out: &mut String, module: &GpuModule) -> Result<(), GpuRenderError> {
    // Materialization is represented in the dataflow body as one assignment, but HIP needs an
    // auxiliary `__global__` kernel in addition to the host wrapper function.
    for assignment in &module.entry.assignments {
        if assignment.op.as_str() == "gpu.materialize" {
            render_materialize_kernel(
                out,
                &materialize_kernel_name(&module.entry.name, assignment)?,
                assignment,
            )?;
            out.push('\n');
        }
    }

    // The entry function is the ordinary host-callable wrapper for this definition.
    render_function(out, &module.entry)?;
    Ok(())
}

fn render_function_decl(out: &mut String, function: &GpuFunction) -> Result<(), GpuRenderError> {
    out.push_str(&function_signature(function)?);
    out.push_str(";\n");
    Ok(())
}

fn render_function(out: &mut String, function: &GpuFunction) -> Result<(), GpuRenderError> {
    out.push_str(&function_signature(function)?);
    out.push_str(" {\n");
    let mut declared = function
        .sources
        .iter()
        .map(|source| source.name.clone())
        .collect::<HashSet<_>>();

    for assignment in &function.assignments {
        for output in &assignment.outputs {
            if runtime_type(output).is_some() && declared.insert(output.name.clone()) {
                out.push_str(&format!("    {};\n", local_decl(output)?));
            }
        }
        render_assignment(out, function, assignment)?;
    }

    // Multiple outputs are returned by writing computed target wires into pointer parameters.
    for result in &function.targets {
        out.push_str(&format!("    *out_{} = {};\n", result.name, result.name));
    }
    out.push_str("    return;\n");
    out.push_str("}\n");
    Ok(())
}

fn function_signature(function: &GpuFunction) -> Result<String, GpuRenderError> {
    let qualifier = if function
        .assignments
        .iter()
        .any(|assignment| assignment.op.as_str() == "gpu.materialize")
    {
        ""
    } else {
        "__host__ __device__ "
    };
    let mut signature = format!("extern \"C\" {qualifier}void {}(", function.name);
    let mut params = Vec::new();

    // Sources render as ordinary parameters; targets render as output-pointer parameters.
    for source in &function.sources {
        params.push(param_decl(source, false)?);
    }
    for target in &function.targets {
        params.push(param_decl(target, true)?);
    }
    signature.push_str(&params.join(", "));
    signature.push(')');
    Ok(signature)
}

fn render_assignment(
    out: &mut String,
    function: &GpuFunction,
    assignment: &GpuAssign,
) -> Result<(), GpuRenderError> {
    if let Some(symbol) = &assignment.call_symbol {
        return render_call(out, symbol, assignment);
    }

    match assignment.op.as_str() {
        "bool.t" => {
            let [] = assignment.inputs.as_slice() else {
                return Err(invalid_inputs(assignment, 0));
            };
            let [output] = assignment.outputs.as_slice() else {
                return Err(invalid_outputs(assignment, 1));
            };
            out.push_str(&format!("    {} = 1;\n", output.name));
        }
        "bool.f" => {
            let [] = assignment.inputs.as_slice() else {
                return Err(invalid_inputs(assignment, 0));
            };
            let [output] = assignment.outputs.as_slice() else {
                return Err(invalid_outputs(assignment, 1));
            };
            out.push_str(&format!("    {} = 0;\n", output.name));
        }
        "bool.not" => {
            let [input] = assignment.inputs.as_slice() else {
                return Err(invalid_inputs(assignment, 1));
            };
            let [output] = assignment.outputs.as_slice() else {
                return Err(invalid_outputs(assignment, 1));
            };
            out.push_str(&format!("    {} = !{};\n", output.name, value_expr(input)));
        }
        "bool.and" => render_binary_bool(out, assignment, "&&")?,
        "bool.or" => render_binary_bool(out, assignment, "||")?,
        "bool.ifc" => render_bool_ifc(out, assignment)?,
        "unit.intro" => {}
        "ax-mp" | "assert-then" | ":.forget" => {}
        "assert" => render_assert(out, assignment)?,
        "u64.zero" => render_u64_zero(out, assignment)?,
        "u64.gt" => render_u64_gt(out, assignment)?,
        "mem.cast.u64" => render_mem_cast_u64(out, assignment)?,
        "ix.zero" => render_ix_zero(out, assignment)?,
        "ix" => render_ix(out, assignment)?,
        "eval" => render_eval(out, assignment)?,
        "gpu.materialize" => render_materialize_call(out, function, assignment)?,
        op => {
            return Err(GpuRenderError::UnsupportedOp(
                op.parse().unwrap_or_else(|_| assignment.op.clone()),
            ));
        }
    }
    Ok(())
}

fn render_assert(out: &mut String, assignment: &GpuAssign) -> Result<(), GpuRenderError> {
    let [input] = assignment.inputs.as_slice() else {
        return Err(invalid_inputs(assignment, 1));
    };
    let [_proof] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 1));
    };
    out.push_str(&format!("    catena_assert({});\n", value_expr(input)));
    Ok(())
}

fn render_u64_zero(out: &mut String, assignment: &GpuAssign) -> Result<(), GpuRenderError> {
    let [] = assignment.inputs.as_slice() else {
        return Err(invalid_inputs(assignment, 0));
    };
    let [output] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 1));
    };
    out.push_str(&format!("    {} = 0;\n", output.name));
    Ok(())
}

fn render_u64_gt(out: &mut String, assignment: &GpuAssign) -> Result<(), GpuRenderError> {
    let [lhs, rhs] = assignment.inputs.as_slice() else {
        return Err(invalid_inputs(assignment, 2));
    };
    let [flag, _true_witness, _false_witness] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 3));
    };
    out.push_str(&format!(
        "    {} = {} > {};\n",
        flag.name,
        value_expr(lhs),
        value_expr(rhs)
    ));
    Ok(())
}

fn render_mem_cast_u64(out: &mut String, assignment: &GpuAssign) -> Result<(), GpuRenderError> {
    let [input] = assignment.inputs.as_slice() else {
        return Err(invalid_inputs(assignment, 1));
    };
    let [len, buffer] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 2));
    };
    out.push_str(&format!(
        "    {len} = {mem}.len / sizeof(uint64_t);\n    {buf} = (uint64_t *){mem}.data;\n",
        len = len.name,
        buf = buffer.name,
        mem = value_expr(input)
    ));
    Ok(())
}

fn render_ix_zero(out: &mut String, assignment: &GpuAssign) -> Result<(), GpuRenderError> {
    let [_proof] = assignment.inputs.as_slice() else {
        return Err(invalid_inputs(assignment, 1));
    };
    let [output] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 1));
    };
    out.push_str(&format!("    {} = 0;\n", output.name));
    Ok(())
}

fn render_ix(out: &mut String, assignment: &GpuAssign) -> Result<(), GpuRenderError> {
    let [index, buffer] = assignment.inputs.as_slice() else {
        return Err(invalid_inputs(assignment, 2));
    };
    let [output] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 1));
    };
    out.push_str(&format!(
        "    {} = {}[{}];\n",
        output.name,
        value_expr(buffer),
        value_expr(index)
    ));
    Ok(())
}

fn render_call(
    out: &mut String,
    symbol: &str,
    assignment: &GpuAssign,
) -> Result<(), GpuRenderError> {
    let mut call_args = assignment
        .inputs
        .iter()
        .filter_map(|input| match input {
            GpuValue::Var(var) if runtime_type(var).is_some() => Some(var.name.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    call_args.extend(
        assignment
            .outputs
            .iter()
            .filter(|output| runtime_type(output).is_some())
            .map(|output| format!("&{}", output.name)),
    );
    out.push_str(&format!("    {symbol}({});\n", call_args.join(", ")));
    Ok(())
}

fn render_binary_bool(
    out: &mut String,
    assignment: &GpuAssign,
    operator: &str,
) -> Result<(), GpuRenderError> {
    let [lhs, rhs] = assignment.inputs.as_slice() else {
        return Err(invalid_inputs(assignment, 2));
    };
    let [output] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 1));
    };
    out.push_str(&format!(
        "    {} = {} {operator} {};\n",
        output.name,
        value_expr(lhs),
        value_expr(rhs)
    ));
    Ok(())
}

fn render_bool_ifc(out: &mut String, assignment: &GpuAssign) -> Result<(), GpuRenderError> {
    let [env_true, fn_true, env_false, fn_false, flag, arg] = assignment.inputs.as_slice() else {
        return Err(invalid_inputs(assignment, 6));
    };
    let [output] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 1));
    };
    out.push_str(&format!(
        "    if ({flag}) {{ {fn_true}({env_true}, {arg}, &{output}); }} else {{ {fn_false}({env_false}, {arg}, &{output}); }}\n",
        flag = value_expr(flag),
        fn_true = callable_expr(fn_true),
        env_true = value_expr(env_true),
        arg = value_expr(arg),
        output = output.name,
        fn_false = callable_expr(fn_false),
        env_false = value_expr(env_false),
    ));
    Ok(())
}

fn render_eval(out: &mut String, assignment: &GpuAssign) -> Result<(), GpuRenderError> {
    let Some((func, args)) = assignment.inputs.split_last() else {
        return Err(invalid_inputs(assignment, 1));
    };
    let mut call_args = args.iter().map(value_expr).collect::<Vec<_>>();
    call_args.extend(
        assignment
            .outputs
            .iter()
            .filter(|output| runtime_type(output).is_some())
            .map(|output| format!("&{}", output.name)),
    );
    out.push_str(&format!(
        "    {}({});\n",
        callable_expr(func),
        call_args.join(", ")
    ));
    Ok(())
}

fn render_materialize_kernel(
    out: &mut String,
    kernel_name: &str,
    assignment: &GpuAssign,
) -> Result<(), GpuRenderError> {
    let [output] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 1));
    };
    let CType::Pointer(element) =
        runtime_type(output).ok_or_else(|| GpuRenderError::ErasedType(output.clone()))?
    else {
        return Err(GpuRenderError::UnsupportedType(
            runtime_type(output).unwrap().clone(),
        ));
    };
    let (_, func, args) = materialize_parts(assignment)?;

    out.push_str(&format!(
        "__global__ void {kernel_name}({} *out, uint64_t len",
        c_type(element)
    ));
    for arg in &args {
        if let GpuValue::Var(var) = arg {
            if runtime_type(var).is_some() {
                out.push_str(", ");
                out.push_str(&param_decl(var, false)?);
            }
        }
    }
    out.push_str(") {\n");
    out.push_str("    uint64_t thread_id = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;\n");
    out.push_str("    if (thread_id >= len) { return; }\n");
    out.push_str("    catena_gpu_env_t env = { thread_id };\n");
    out.push_str("    catena_gpu_state_t state = 0;\n");
    out.push_str("    catena_gpu_state_t next_state = 0;\n");
    out.push_str(&format!("    {} value;\n", c_type(element)));
    out.push_str("    ");
    out.push_str(&callable_expr(func));
    out.push_str("(env, state");
    for arg in args {
        if let GpuValue::Var(var) = arg
            && runtime_type(var).is_some()
        {
            out.push_str(", ");
            out.push_str(&var.name);
        }
    }
    out.push_str(", &next_state, &value);\n");
    out.push_str("    out[thread_id] = value;\n");
    out.push_str("}\n");
    Ok(())
}

fn render_materialize_call(
    out: &mut String,
    function: &GpuFunction,
    assignment: &GpuAssign,
) -> Result<(), GpuRenderError> {
    let [output] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 1));
    };
    let CType::Pointer(element) =
        runtime_type(output).ok_or_else(|| GpuRenderError::ErasedType(output.clone()))?
    else {
        return Err(GpuRenderError::UnsupportedType(
            runtime_type(output).unwrap().clone(),
        ));
    };
    let (launch, _func, args) = materialize_parts(assignment)?;
    let launch = value_expr(launch);
    let kernel_name = materialize_kernel_name(&function.name, assignment)?;

    out.push_str(&format!(
        "    uint64_t {name}_len = catena_launch_len({launch});\n",
        name = output.name
    ));
    out.push_str(&format!(
        "    {} *{name}_data = nullptr;\n",
        c_type(element),
        name = output.name
    ));
    out.push_str(&format!(
        "    catena_hip_check(hipMallocManaged((void **)&{name}_data, {name}_len * sizeof({element})));\n",
        name = output.name,
        element = c_type(element)
    ));
    out.push_str(&format!(
        "    {kernel_name}<<<dim3({launch}.grid_dim.x, {launch}.grid_dim.y, {launch}.grid_dim.z), dim3({launch}.block_dim.x, {launch}.block_dim.y, {launch}.block_dim.z)>>>\n"
    ));
    out.push_str(&format!(
        "        ({name}_data, {name}_len",
        name = output.name
    ));
    for arg in args {
        if let GpuValue::Var(var) = arg
            && runtime_type(var).is_some()
        {
            out.push_str(", ");
            out.push_str(&var.name);
        }
    }
    out.push_str(");\n");
    out.push_str(&format!("    {} = {}_data;\n", output.name, output.name));
    Ok(())
}

fn materialize_parts(
    assignment: &GpuAssign,
) -> Result<(&GpuValue, &GpuValue, Vec<&GpuValue>), GpuRenderError> {
    let launch = assignment
        .inputs
        .iter()
        .find(|input| {
            matches!(
                input,
                GpuValue::Var(var)
                    if matches!(runtime_type(var), Some(CType::Named(name)) if name == "catena_launch_params_t")
            )
        })
        .ok_or(GpuRenderError::MissingMaterializeLaunchParams)?;
    let func = assignment
        .inputs
        .iter()
        .find(|input| matches!(input, GpuValue::FnSymbol(_)))
        .ok_or(GpuRenderError::MissingMaterializeFunction)?;
    let args = assignment
        .inputs
        .iter()
        .filter(|input| !std::ptr::eq(*input, launch) && !std::ptr::eq(*input, func))
        .collect();
    Ok((launch, func, args))
}

fn materialize_kernel_name(
    function_name: &str,
    assignment: &GpuAssign,
) -> Result<String, GpuRenderError> {
    let [output] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 1));
    };
    Ok(format!("materialize_{}_{}", function_name, output.name))
}

fn param_decl(var: &GpuVar, by_pointer: bool) -> Result<String, GpuRenderError> {
    let ty = runtime_type(var).ok_or_else(|| GpuRenderError::ErasedType(var.clone()))?;
    if by_pointer {
        Ok(format!("{} *out_{}", c_type(ty), var.name))
    } else {
        Ok(format!("{} {}", c_type(ty), var.name))
    }
}

fn local_decl(var: &GpuVar) -> Result<String, GpuRenderError> {
    let ty = runtime_type(var).ok_or_else(|| GpuRenderError::ErasedType(var.clone()))?;
    Ok(format!("{} {}", c_type(ty), var.name))
}

fn c_type(ty: &CType) -> String {
    match ty {
        CType::Unit => "catena_unit_t".to_string(),
        CType::Bool => "uint8_t".to_string(),
        CType::U64 => "uint64_t".to_string(),
        CType::F32 => "float".to_string(),
        CType::Pointer(inner) => format!("{} *", c_type(inner)),
        CType::Named(name) => name.clone(),
    }
}

fn value_expr(value: &GpuValue) -> String {
    match value {
        GpuValue::Var(var) => var.name.clone(),
        GpuValue::FnSymbol(symbol) => sanitize_ident(symbol.target.as_str()),
    }
}

fn callable_expr(value: &GpuValue) -> String {
    value_expr(value)
}

fn invalid_inputs(assignment: &GpuAssign, expected: usize) -> GpuRenderError {
    GpuRenderError::InvalidInputCount {
        op: assignment.op.clone(),
        expected,
        actual: assignment.inputs.len(),
    }
}

fn invalid_outputs(assignment: &GpuAssign, expected: usize) -> GpuRenderError {
    GpuRenderError::InvalidOutputCount {
        op: assignment.op.clone(),
        expected,
        actual: assignment.outputs.len(),
    }
}

fn sanitize_ident(name: &str) -> String {
    let mut ident = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect::<String>();
    if ident.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        ident.insert(0, '_');
    }
    ident
}
