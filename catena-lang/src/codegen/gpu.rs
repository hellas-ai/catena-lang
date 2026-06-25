use std::collections::HashSet;

use hexpr::Operation;
use thiserror::Error;

use crate::codegen::{
    GpuAssign, GpuDialect, GpuFunction, GpuModule, GpuModuleMap, GpuValue, GpuVar,
    lower_types::CType,
    ops::{ifc, materializec, reducec},
    prelude::render_gpu_prelude,
    render_utils::{c_type, invalid_inputs, invalid_outputs, param_decl, sanitize_ident},
    runtime_type,
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
    #[error("operation `{op}` expected {expected} input components, found {actual}")]
    InvalidInputComponentCount {
        op: Operation,
        expected: usize,
        actual: usize,
    },
    #[error(
        "operation `{op}` input sizes account for {expected} flattened inputs, but assignment has {actual}"
    )]
    InvalidFlattenedInputCount {
        op: Operation,
        expected: usize,
        actual: usize,
    },
    #[error(
        "operation `{op}` expected {expected} {description} in input component `{component}`, found {actual}"
    )]
    InvalidInputComponentValueCount {
        op: Operation,
        component: &'static str,
        description: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("operation `{op}` input component `{component}` is erased: {description}")]
    ErasedInputComponentValue {
        op: Operation,
        component: &'static str,
        description: &'static str,
    },
    #[error("gpu.materialize is missing launch params")]
    MissingMaterializeLaunchParams,
    #[error("gpu.materialize is missing function input")]
    MissingMaterializeFunction,
    #[error("invalid integer constant operation `{op}`")]
    InvalidIntegerConstant { op: Operation },
}

/// Render a single GPU dataflow module as standalone GPU-flavored C++ source.
///
/// Codegen has already produced the semantic `GpuModule`; this renderer is responsible only for
/// turning that artifact into text. It still contains primitive-specific lowering for the current
/// small backend surface, but it should not inspect the original Catena graph or report state.
pub fn render_module(module: &GpuModule, dialect: GpuDialect) -> Result<String, GpuRenderError> {
    let mut out = String::new();
    out.push_str(&render_gpu_prelude(dialect));
    out.push('\n');
    render_module_body(&mut out, module, dialect)?;
    Ok(out)
}

/// Render all generated GPU modules into one GPU-flavored C++ translation unit.
pub fn render_modules(
    modules: &GpuModuleMap,
    dialect: GpuDialect,
) -> Result<String, GpuRenderError> {
    let mut out = String::new();
    out.push_str(&render_gpu_prelude(dialect));
    out.push('\n');

    for module in modules.values() {
        render_function_decl(&mut out, &module.entry, dialect)?;
    }
    if !modules.is_empty() {
        out.push('\n');
    }

    for module in modules.values() {
        render_module_body(&mut out, module, dialect)?;
        out.push('\n');
    }

    Ok(out)
}

fn render_module_body(
    out: &mut String,
    module: &GpuModule,
    dialect: GpuDialect,
) -> Result<(), GpuRenderError> {
    // Materialization is represented in the dataflow body as one assignment, but GPU codegen needs an
    // auxiliary `__global__` kernel in addition to the host wrapper function.
    for assignment in &module.entry.assignments {
        if assignment.op.as_str() == "gpu.materialize" {
            render_materialize_kernel(
                out,
                &materialize_kernel_name(&module.entry.name, assignment)?,
                assignment,
            )?;
            out.push('\n');
        } else if assignment.op.as_str() == "materializec" {
            materializec::render_kernel(
                out,
                &materializec::kernel_name(&module.entry.name, assignment)?,
                assignment,
            )?;
            out.push('\n');
        }
    }

    // The entry function is the ordinary host-callable wrapper for this definition.
    render_function(out, &module.entry, dialect)?;
    Ok(())
}

fn render_function_decl(
    out: &mut String,
    function: &GpuFunction,
    dialect: GpuDialect,
) -> Result<(), GpuRenderError> {
    if function_is_host_only(function) {
        out.push_str(&format!("#ifndef {}\n", dialect.device_compile_guard()));
    }
    out.push_str(&function_signature(function)?);
    out.push_str(";\n");
    if function_is_host_only(function) {
        out.push_str("#endif\n");
    }
    Ok(())
}

fn render_function(
    out: &mut String,
    function: &GpuFunction,
    dialect: GpuDialect,
) -> Result<(), GpuRenderError> {
    if function_is_host_only(function) {
        out.push_str(&format!("#ifndef {}\n", dialect.device_compile_guard()));
    }
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
        render_assignment(out, function, assignment, dialect)?;
    }

    // Multiple outputs are returned by writing computed target wires into pointer parameters.
    for result in &function.targets {
        out.push_str(&format!("    *out_{} = {};\n", result.name, result.name));
    }
    out.push_str("    return;\n");
    out.push_str("}\n");
    if function_is_host_only(function) {
        out.push_str("#endif\n");
    }
    Ok(())
}

fn function_is_host_only(function: &GpuFunction) -> bool {
    function
        .assignments
        .iter()
        .any(|assignment| matches!(assignment.op.as_str(), "gpu.materialize" | "materializec"))
}

fn function_signature(function: &GpuFunction) -> Result<String, GpuRenderError> {
    let qualifier = if function_is_host_only(function) {
        "__host__ "
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
    dialect: GpuDialect,
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
        "bool.ifc" => ifc::render(out, assignment)?,
        "unit.intro" => {}
        "ax-mp" | "assert-then" | ":.forget" | ":.param" => {}
        "assert" => render_assert(out, assignment)?,
        "u64.zero" => render_u64_zero(out, assignment)?,
        "u64.one" => render_u64_one(out, assignment)?,
        "u64.add" => render_binary(out, assignment, "+")?,
        "u64.mul" => render_binary(out, assignment, "*")?,
        "u32.one" => render_u64_one(out, assignment)?,
        "u32.add" => render_binary(out, assignment, "+")?,
        "u32.sub" => render_binary(out, assignment, "-")?,
        "u32.and" => render_binary(out, assignment, "&")?,
        "u32.or" => render_binary(out, assignment, "|")?,
        "u32.xor" => render_binary(out, assignment, "^")?,
        "u32.eq" => render_binary_bool(out, assignment, "==")?,
        "u32.ne" => render_binary_bool(out, assignment, "!=")?,
        "u32.lt" => render_binary_bool(out, assignment, "<")?,
        "u32.gt" => render_binary_bool(out, assignment, ">")?,
        "u32.lte" => render_binary_bool(out, assignment, "<=")?,
        "u32.gte" => render_binary_bool(out, assignment, ">=")?,
        "u32.mul" => render_binary(out, assignment, "*")?,
        "u32.shl" => render_binary(out, assignment, "<<")?,
        "u32.shr" => render_binary(out, assignment, ">>")?,
        "u32.not" => render_unary_prefix(out, assignment, "~")?,
        "u32.to-f32" => render_u32_cast_to_f32(out, assignment)?,
        "u32.bitcast-f32" => render_u32_bitcast_f32(out, assignment)?,
        "u64.eq" => render_u64_eq(out, assignment)?,
        "u64.gt" => render_u64_gt(out, assignment)?,
        "mem.cast.u64" => render_mem_cast_u64(out, assignment)?,
        "buf.u64.cast-same-length" => render_buf_u64_cast_same_length(out, assignment)?,
        "buf.to-mem" => render_buf_to_mem(out, assignment)?,
        "f32.one" => render_f32_one(out, assignment)?,
        "f32.add" => render_binary(out, assignment, "+")?,
        "f32.sub" => render_binary(out, assignment, "-")?,
        "f32.neg" => render_f32_neg(out, assignment)?,
        "f32.mul" => render_binary(out, assignment, "*")?,
        "f32.fma" => render_f32_fma(out, assignment)?,
        "f32.div" => render_binary(out, assignment, "/")?,
        "f32.eq" => render_binary_bool(out, assignment, "==")?,
        "f32.ne" => render_binary_bool(out, assignment, "!=")?,
        "f32.lt" => render_binary_bool(out, assignment, "<")?,
        "f32.gt" => render_binary_bool(out, assignment, ">")?,
        "f32.lte" => render_binary_bool(out, assignment, "<=")?,
        "f32.gte" => render_binary_bool(out, assignment, ">=")?,
        "f32.select" | "u32.select" | "u64.select" | "bool.select" => {
            render_select(out, assignment)?
        }
        "f32.round-to-u32" => render_f32_round_to_u32(out, assignment)?,
        "f32.bitcast-u32" => render_f32_bitcast_u32(out, assignment)?,
        "ix.zero" => render_ix_zero(out, assignment)?,
        "ix" => render_ix(out, assignment)?,
        "eval" => render_eval(out, assignment)?,
        "reducec" => reducec::render(out, assignment)?,
        "gpu.materialize" => render_materialize_call(out, function, assignment, dialect)?,
        "materializec" => materializec::render_call(out, function, assignment, dialect)?,
        op if op.starts_with("const.u64.") => {
            render_int_const(out, assignment, "const.u64.", "ULL")?
        }
        op if op.starts_with("const.u32.") => render_int_const(out, assignment, "const.u32.", "U")?,
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

fn render_u64_one(out: &mut String, assignment: &GpuAssign) -> Result<(), GpuRenderError> {
    let [] = assignment.inputs.as_slice() else {
        return Err(invalid_inputs(assignment, 0));
    };
    let [output] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 1));
    };
    out.push_str(&format!("    {} = 1;\n", output.name));
    Ok(())
}

fn render_f32_one(out: &mut String, assignment: &GpuAssign) -> Result<(), GpuRenderError> {
    let [] = assignment.inputs.as_slice() else {
        return Err(invalid_inputs(assignment, 0));
    };
    let [output] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 1));
    };
    out.push_str(&format!("    {} = 1.0;\n", output.name));
    Ok(())
}

fn render_u32_cast_to_f32(out: &mut String, assignment: &GpuAssign) -> Result<(), GpuRenderError> {
    let [input] = assignment.inputs.as_slice() else {
        return Err(invalid_inputs(assignment, 1));
    };
    let [output] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 1));
    };
    out.push_str(&format!(
        "    {} = (float)({});\n",
        output.name,
        value_expr(input)
    ));
    Ok(())
}

fn render_u32_bitcast_f32(out: &mut String, assignment: &GpuAssign) -> Result<(), GpuRenderError> {
    let [input] = assignment.inputs.as_slice() else {
        return Err(invalid_inputs(assignment, 1));
    };
    let [output] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 1));
    };
    out.push_str(&format!(
        "    {} = catena_u32_bitcast_f32({});\n",
        output.name,
        value_expr(input)
    ));
    Ok(())
}

fn render_f32_neg(out: &mut String, assignment: &GpuAssign) -> Result<(), GpuRenderError> {
    let [input] = assignment.inputs.as_slice() else {
        return Err(invalid_inputs(assignment, 1));
    };
    let [output] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 1));
    };
    out.push_str(&format!("    {} = -{};\n", output.name, value_expr(input)));
    Ok(())
}

fn render_f32_fma(out: &mut String, assignment: &GpuAssign) -> Result<(), GpuRenderError> {
    let [multiplicand, multiplier, addend] = assignment.inputs.as_slice() else {
        return Err(invalid_inputs(assignment, 3));
    };
    let [output] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 1));
    };
    out.push_str(&format!(
        "    {} = fmaf({}, {}, {});\n",
        output.name,
        value_expr(multiplicand),
        value_expr(multiplier),
        value_expr(addend)
    ));
    Ok(())
}

fn render_unary_prefix(
    out: &mut String,
    assignment: &GpuAssign,
    operator: &str,
) -> Result<(), GpuRenderError> {
    let [input] = assignment.inputs.as_slice() else {
        return Err(invalid_inputs(assignment, 1));
    };
    let [output] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 1));
    };
    out.push_str(&format!(
        "    {} = {operator}{};\n",
        output.name,
        value_expr(input)
    ));
    Ok(())
}

fn render_select(out: &mut String, assignment: &GpuAssign) -> Result<(), GpuRenderError> {
    let [flag, when_true, when_false] = assignment.inputs.as_slice() else {
        return Err(invalid_inputs(assignment, 3));
    };
    let [output] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 1));
    };
    out.push_str(&format!(
        "    {} = {} ? {} : {};\n",
        output.name,
        value_expr(flag),
        value_expr(when_true),
        value_expr(when_false)
    ));
    Ok(())
}

fn render_f32_round_to_u32(out: &mut String, assignment: &GpuAssign) -> Result<(), GpuRenderError> {
    let [input] = assignment.inputs.as_slice() else {
        return Err(invalid_inputs(assignment, 1));
    };
    let [output] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 1));
    };
    out.push_str(&format!(
        "    {} = (uint32_t)nearbyintf({});\n",
        output.name,
        value_expr(input)
    ));
    Ok(())
}

fn render_f32_bitcast_u32(out: &mut String, assignment: &GpuAssign) -> Result<(), GpuRenderError> {
    let [input] = assignment.inputs.as_slice() else {
        return Err(invalid_inputs(assignment, 1));
    };
    let [output] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 1));
    };
    out.push_str(&format!(
        "    {} = catena_f32_bitcast_u32({});\n",
        output.name,
        value_expr(input)
    ));
    Ok(())
}

fn render_int_const(
    out: &mut String,
    assignment: &GpuAssign,
    prefix: &str,
    suffix: &str,
) -> Result<(), GpuRenderError> {
    let [] = assignment.inputs.as_slice() else {
        return Err(invalid_inputs(assignment, 0));
    };
    let [output] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 1));
    };
    let value = parse_int_const(&assignment.op, prefix).ok_or_else(|| {
        GpuRenderError::InvalidIntegerConstant {
            op: assignment.op.clone(),
        }
    })?;
    out.push_str(&format!("    {} = {value}{suffix};\n", output.name));
    Ok(())
}

fn parse_int_const(op: &Operation, prefix: &str) -> Option<u64> {
    let literal = op.as_str().strip_prefix(prefix)?;
    let literal = literal.replace('_', "");
    let hex = literal.strip_prefix("0x")?;
    u64::from_str_radix(hex, 16).ok()
}

fn render_binary(
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

fn render_u64_eq(out: &mut String, assignment: &GpuAssign) -> Result<(), GpuRenderError> {
    let [lhs, rhs] = assignment.inputs.as_slice() else {
        return Err(invalid_inputs(assignment, 2));
    };
    let [flag, _true_witness, _false_witness] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 3));
    };
    out.push_str(&format!(
        "    {} = {} == {};\n",
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

fn render_buf_u64_cast_same_length(
    out: &mut String,
    assignment: &GpuAssign,
) -> Result<(), GpuRenderError> {
    let [buffer, _same_length] = assignment.inputs.as_slice() else {
        return Err(invalid_inputs(assignment, 2));
    };
    let [output] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 1));
    };
    out.push_str(&format!("    {} = {};\n", output.name, value_expr(buffer)));
    Ok(())
}

fn render_buf_to_mem(out: &mut String, assignment: &GpuAssign) -> Result<(), GpuRenderError> {
    let [len, buffer] = assignment.inputs.as_slice() else {
        return Err(invalid_inputs(assignment, 2));
    };
    let [output] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 1));
    };
    let GpuValue::Var(buffer) = buffer else {
        return Err(GpuRenderError::UnsupportedOp(assignment.op.clone()));
    };
    let CType::Pointer(element) =
        runtime_type(buffer).ok_or_else(|| GpuRenderError::ErasedType(buffer.clone()))?
    else {
        return Err(GpuRenderError::UnsupportedType(
            runtime_type(buffer).unwrap().clone(),
        ));
    };
    out.push_str(&format!(
        "    {mem}.data = (void *){buf};\n    {mem}.len = {len} * sizeof({element});\n",
        mem = output.name,
        buf = buffer.name,
        len = value_expr(len),
        element = c_type(element),
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
    dialect: GpuDialect,
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
        "    catena_host_gpu_check({managed_alloc_fn}((void **)&{name}_data, {name}_len * sizeof({element})));\n",
        name = output.name,
        element = c_type(element),
        managed_alloc_fn = dialect.managed_alloc_fn(),
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

fn local_decl(var: &GpuVar) -> Result<String, GpuRenderError> {
    let ty = runtime_type(var).ok_or_else(|| GpuRenderError::ErasedType(var.clone()))?;
    Ok(format!("{} {}", c_type(ty), var.name))
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

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::BTreeMap;

    use crate::codegen::{
        fn_ptrs::FnPtrSymbol,
        lower_types::{CType, LoweredType},
    };
    use open_hypergraphs::lax::NodeId;

    fn op(name: &str) -> Operation {
        name.parse().unwrap()
    }

    fn var(node: usize, name: &str, ty: CType) -> GpuVar {
        GpuVar {
            node: NodeId(node),
            name: name.to_string(),
            lowered: LoweredType::Runtime(ty),
        }
    }

    #[test]
    fn materializing_host_wrappers_are_hidden_from_hip_device_parse() {
        let len = var(0, "len", CType::U64);
        let out = var(1, "out", CType::Pointer(Box::new(CType::U64)));
        let value = var(2, "value", CType::U64);
        let index = var(3, "i", CType::U64);

        let materialize = GpuModule {
            name: "program_materialize".to_string(),
            source_name: Some(op("materialize")),
            entry: GpuFunction {
                name: "program_materialize".to_string(),
                sources: vec![len.clone()],
                targets: vec![out.clone()],
                assignments: vec![GpuAssign {
                    op: op("materializec"),
                    input_sizes: vec![0, 1, 1],
                    output_sizes: Vec::new(),
                    call_symbol: None,
                    inputs: vec![
                        GpuValue::FnSymbol(FnPtrSymbol {
                            target: op("program.producer"),
                        }),
                        GpuValue::Var(len),
                    ],
                    outputs: vec![out],
                }],
            },
        };
        let producer = GpuModule {
            name: "program_producer".to_string(),
            source_name: Some(op("producer")),
            entry: GpuFunction {
                name: "program_producer".to_string(),
                sources: vec![index],
                targets: vec![value.clone()],
                assignments: vec![GpuAssign {
                    op: op("u64.one"),
                    input_sizes: Vec::new(),
                    output_sizes: Vec::new(),
                    call_symbol: None,
                    inputs: vec![],
                    outputs: vec![value],
                }],
            },
        };

        let modules =
            BTreeMap::from([(op("materialize"), materialize), (op("producer"), producer)]);
        let source = render_modules(&modules, GpuDialect::Hip).unwrap();

        assert!(source.contains(
            "#ifndef __HIP_DEVICE_COMPILE__\nextern \"C\" __host__ void program_materialize"
        ));
        assert!(source.contains(
            "#ifndef __HIP_DEVICE_COMPILE__\nextern \"C\" __host__ void program_materialize(uint64_t len, uint64_t * *out_out) {"
        ));
        assert!(source.contains("catena_host_gpu_check(hipMallocManaged"));
        assert!(source.contains("catena_host_gpu_check(hipDeviceSynchronize"));
        assert!(!source.contains("catena_gpu_check"));
    }
}
