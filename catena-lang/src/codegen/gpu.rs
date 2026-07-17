use std::collections::{BTreeMap, HashSet};

use hexpr::Operation;
use thiserror::Error;

use crate::codegen::{
    GpuAssign, GpuDialect, GpuFunction, GpuModule, GpuModuleMap, GpuValue, GpuVar,
    components::value_expr,
    components::{input_components, single_value, value_expr},
    gpu_placement::{
        GpuFunctionPlacement, direct_function_placement, function_placement, function_placements,
    },
    lower_types::CType,
    ops::{ifc, materializec, reducec, row_major},
    prelude::render_gpu_prelude,
    render_utils::{c_type, invalid_inputs, invalid_outputs, param_decl},
    runtime_type,
};
use crate::prefixes::{CONST_U32_PREFIX, CONST_U64_PREFIX};

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
    #[error("operation `{op}` expected {expected} output components, found {actual}")]
    InvalidOutputComponentCount {
        op: Operation,
        expected: usize,
        actual: usize,
    },
    #[error(
        "operation `{op}` output sizes account for {expected} flattened outputs, but assignment has {actual}"
    )]
    InvalidFlattenedOutputCount {
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
    render_module_body(
        &mut out,
        module,
        dialect,
        direct_function_placement(&module.entry),
    )?;
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
    let placements = function_placements(modules);

    for module in modules.values() {
        render_function_decl(
            &mut out,
            &module.entry,
            dialect,
            function_placement(&placements, &module.entry.name),
        )?;
    }
    if !modules.is_empty() {
        out.push('\n');
    }

    for module in modules.values() {
        render_module_body(
            &mut out,
            module,
            dialect,
            function_placement(&placements, &module.entry.name),
        )?;
        out.push('\n');
    }

    Ok(out)
}

fn render_module_body(
    out: &mut String,
    module: &GpuModule,
    dialect: GpuDialect,
    placement: GpuFunctionPlacement,
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
    render_function(out, &module.entry, dialect, placement)?;
    Ok(())
}

fn render_function_decl(
    out: &mut String,
    function: &GpuFunction,
    dialect: GpuDialect,
    placement: GpuFunctionPlacement,
) -> Result<(), GpuRenderError> {
    if placement.is_host_only() {
        out.push_str(&format!("#ifndef {}\n", dialect.device_compile_guard()));
    }
    out.push_str(&function_signature(function, placement)?);
    out.push_str(";\n");
    if placement.is_host_only() {
        out.push_str("#endif\n");
    }
    Ok(())
}

fn render_function(
    out: &mut String,
    function: &GpuFunction,
    dialect: GpuDialect,
    placement: GpuFunctionPlacement,
) -> Result<(), GpuRenderError> {
    if placement.is_host_only() {
        out.push_str(&format!("#ifndef {}\n", dialect.device_compile_guard()));
    }
    out.push_str(&function_signature(function, placement)?);
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
    for (result, output_param) in function
        .targets
        .iter()
        .zip(output_param_names(&function.targets))
    {
        out.push_str(&format!("    *{output_param} = {};\n", result.name));
    }
    out.push_str("    return;\n");
    out.push_str("}\n");
    if placement.is_host_only() {
        out.push_str("#endif\n");
    }
    Ok(())
}

fn function_signature(
    function: &GpuFunction,
    placement: GpuFunctionPlacement,
) -> Result<String, GpuRenderError> {
    let qualifier = match placement {
        GpuFunctionPlacement::HostOnly => "__host__ ",
        GpuFunctionPlacement::HostAndDevice => "__host__ __device__ ",
    };
    let mut signature = format!("extern \"C\" {qualifier}void {}(", function.name);
    let mut params = Vec::new();

    // Sources render as ordinary parameters; targets render as output-pointer parameters.
    for source in &function.sources {
        params.push(param_decl(source, false)?);
    }
    for (target, output_param) in function
        .targets
        .iter()
        .zip(output_param_names(&function.targets))
    {
        let ty = runtime_type(target).ok_or_else(|| GpuRenderError::ErasedType(target.clone()))?;
        params.push(format!("{} *{}", c_type(ty), output_param));
    }
    signature.push_str(&params.join(", "));
    signature.push(')');
    Ok(signature)
}

fn output_param_names(targets: &[GpuVar]) -> Vec<String> {
    let counts = targets
        .iter()
        .fold(BTreeMap::<&str, usize>::new(), |mut counts, target| {
            *counts.entry(target.name.as_str()).or_default() += 1;
            counts
        });
    let mut seen = BTreeMap::<&str, usize>::new();
    targets
        .iter()
        .map(|target| {
            let name = target.name.as_str();
            if counts[name] == 1 {
                format!("out_{name}")
            } else {
                let index = seen.entry(name).or_default();
                let output_name = format!("out_{name}_{index}");
                *index += 1;
                output_name
            }
        })
        .collect()
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
        "ax-mp" | "assert-then" | ":.param" => {}
        ":.ty" => render_ty_ascription(out, assignment)?,
        ":.forget" => render_forget(out, assignment)?,
        "assert" => render_assert(out, assignment)?,
        "u64.zero" => render_u64_zero(out, assignment)?,
        "u64.one" => render_u64_one(out, assignment)?,
        "u64.add" => render_binary(out, assignment, "+")?,
        "u64.sub" => render_binary(out, assignment, "-")?,
        "u64.ne" => render_binary_bool(out, assignment, "!=")?,
        "u64.lt" => render_binary_bool(out, assignment, "<")?,
        "u64.lte" => render_binary_bool(out, assignment, "<=")?,
        "u64.gte" => render_binary_bool(out, assignment, ">=")?,
        "u64.mul" => render_binary(out, assignment, "*")?,
        "u64.name" => render_forget(out, assignment)?,
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
        "u64.to-f32" => render_u64_cast_to_f32(out, assignment)?,
        "u32.bitcast-f32" => render_u32_bitcast_f32(out, assignment)?,
        "u64.eq" => render_u64_eq(out, assignment)?,
        "u64.gt" => render_u64_gt(out, assignment)?,
        "mem.cast.u64" => render_mem_cast_u64(out, assignment)?,
        "mem.cast.f32" => render_mem_cast_f32(out, assignment)?,
        "buf.u64.cast-same-length" => render_buf_u64_cast_same_length(out, assignment)?,
        "buf.f32.cast-same-length" => render_buf_f32_cast_same_length(out, assignment)?,
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
        "ix.to-u64" => render_forget(out, assignment)?,
        "ix.zero" => render_ix_zero(out, assignment)?,
        "ix" => render_ix(out, assignment)?,
        "row-major-index" => row_major::render_index(out, assignment)?,
        "row-major-row" => row_major::render_row(out, assignment)?,
        "row-major-col" => row_major::render_col(out, assignment)?,
        "ix.to-u64" => render_forget(out, assignment)?,
        "u64.to-ix" => render_u64_to_ix(out, assignment)?,
        "eval" => render_eval(out, assignment)?,
        "reducec" => reducec::render(out, assignment)?,
        "gpu.materialize" => render_materialize_call(out, function, assignment, dialect)?,
        "materializec" => materializec::render_call(out, function, assignment, dialect)?,
        op if op.starts_with(CONST_U64_PREFIX) => {
            render_int_const(out, assignment, CONST_U64_PREFIX, "ULL")?
        }
        op if op.starts_with(CONST_U32_PREFIX) => {
            render_int_const(out, assignment, CONST_U32_PREFIX, "U")?
        }
        _op if is_erased_only_assignment(assignment) => {}
        op => {
            return Err(GpuRenderError::UnsupportedOp(
                op.parse().unwrap_or_else(|_| assignment.op.clone()),
            ));
        }
    }
    Ok(())
}

fn is_erased_only_assignment(assignment: &GpuAssign) -> bool {
    assignment
        .inputs
        .iter()
        .all(|value| matches!(value, GpuValue::Var(var) if runtime_type(var).is_none()))
        && assignment
            .outputs
            .iter()
            .all(|var| runtime_type(var).is_none())
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

fn render_forget(out: &mut String, assignment: &GpuAssign) -> Result<(), GpuRenderError> {
    let [input] = assignment.inputs.as_slice() else {
        return Err(invalid_inputs(assignment, 1));
    };
    let [output] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 1));
    };
    out.push_str(&format!("    {} = {};\n", output.name, value_expr(input)));
    Ok(())
}

fn render_ty_ascription(out: &mut String, assignment: &GpuAssign) -> Result<(), GpuRenderError> {
    let runtime_inputs = assignment
        .inputs
        .iter()
        .filter_map(|input| match input {
            GpuValue::Var(var) if runtime_type(var).is_some() => Some(var),
            _ => None,
        })
        .collect::<Vec<_>>();
    let [input] = runtime_inputs.as_slice() else {
        return Err(GpuRenderError::InvalidInputComponentValueCount {
            op: assignment.op.clone(),
            component: "runtime input",
            description: "runtime values",
            expected: 1,
            actual: runtime_inputs.len(),
        });
    };

    for output in assignment
        .outputs
        .iter()
        .filter(|output| runtime_type(output).is_some())
    {
        out.push_str(&format!("    {} = {};\n", output.name, input.name));
    }
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

fn render_u64_cast_to_f32(out: &mut String, assignment: &GpuAssign) -> Result<(), GpuRenderError> {
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

fn render_mem_cast_f32(out: &mut String, assignment: &GpuAssign) -> Result<(), GpuRenderError> {
    let [input] = assignment.inputs.as_slice() else {
        return Err(invalid_inputs(assignment, 1));
    };
    let [len, buffer] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 2));
    };
    out.push_str(&format!(
        "    {len} = {mem}.len / sizeof(float);\n    {buf} = (float *){mem}.data;\n",
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

fn render_buf_f32_cast_same_length(
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

fn render_u64_to_ix(out: &mut String, assignment: &GpuAssign) -> Result<(), GpuRenderError> {
    let components = input_components(assignment)?;
    let [index, _bound, _proof] = components.as_slice() else {
        return Err(GpuRenderError::InvalidInputComponentCount {
            op: assignment.op.clone(),
            expected: 3,
            actual: components.len(),
        });
    };
    let index =
        single_value(index).map_err(|error| GpuRenderError::InvalidInputComponentValueCount {
            op: assignment.op.clone(),
            component: "index",
            description: "runtime value",
            expected: 1,
            actual: error.actual,
        })?;
    let [output] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 1));
    };
    out.push_str(&format!(
        "    {output} = {index};\n",
        output = output.name,
        index = value_expr(index)
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
    if let GpuValue::FnSymbol(symbol) = func
        && render_primitive_eval(out, assignment, &symbol.target, args)?
    {
        return Ok(());
    }
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
        value_expr(func),
        call_args.join(", ")
    ));
    Ok(())
}

fn render_primitive_eval(
    out: &mut String,
    assignment: &GpuAssign,
    target: &Operation,
    args: &[GpuValue],
) -> Result<bool, GpuRenderError> {
    let primitive = GpuAssign {
        op: target.clone(),
        input_sizes: Vec::new(),
        output_sizes: assignment.output_sizes.clone(),
        call_symbol: None,
        inputs: args.to_vec(),
        outputs: assignment.outputs.clone(),
    };
    match target.as_str() {
        "f32.add" => render_binary(out, &primitive, "+")?,
        "f32.mul" => render_binary(out, &primitive, "*")?,
        "row-major-index" => row_major::render_index(out, &primitive)?,
        "row-major-row" => row_major::render_row(out, &primitive)?,
        "row-major-col" => row_major::render_col(out, &primitive)?,
        _ => return Ok(false),
    }
    Ok(true)
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
    out.push_str(&value_expr(func));
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

    fn erased_var(node: usize, name: &str) -> GpuVar {
        GpuVar {
            node: NodeId(node),
            name: name.to_string(),
            lowered: LoweredType::Erased,
        }
    }

    #[test]
    fn ty_ascription_preserves_runtime_value_output() {
        let input = var(0, "x0", CType::U64);
        let runtime_output = var(1, "x1", CType::U64);
        let type_output = erased_var(2, "x2");
        let module = GpuModule {
            name: "program_ascribed".to_string(),
            source_name: Some(op("ascribed")),
            entry: GpuFunction {
                name: "program_ascribed".to_string(),
                sources: vec![input.clone()],
                targets: vec![runtime_output.clone()],
                assignments: vec![GpuAssign {
                    op: op(":.ty"),
                    input_sizes: vec![1],
                    output_sizes: vec![1, 1],
                    call_symbol: None,
                    inputs: vec![GpuValue::Var(input)],
                    outputs: vec![runtime_output, type_output],
                }],
            },
        };

        let source = render_module(&module, GpuDialect::Hip).unwrap();

        assert!(source.contains("uint64_t x1;"));
        assert!(source.contains("    x1 = x0;\n"));
        assert!(!source.contains("uint64_t x2;"));
    }

    #[test]
    fn erased_only_helper_assignment_renders_as_noop() {
        let input = erased_var(0, "x0");
        let output_a = erased_var(1, "x1");
        let output_b = erased_var(2, "x2");
        let module = GpuModule {
            name: "program_type_helper".to_string(),
            source_name: Some(op("type-helper")),
            entry: GpuFunction {
                name: "program_type_helper".to_string(),
                sources: Vec::new(),
                targets: Vec::new(),
                assignments: vec![GpuAssign {
                    op: op("type-helper"),
                    input_sizes: vec![1],
                    output_sizes: vec![1, 1],
                    call_symbol: None,
                    inputs: vec![GpuValue::Var(input)],
                    outputs: vec![output_a, output_b],
                }],
            },
        };

        let source = render_module(&module, GpuDialect::Hip).unwrap();

        assert!(source.contains("void program_type_helper()"));
        assert!(!source.contains("type-helper"));
    }

    #[test]
    fn row_major_layout_operations_render_index_arithmetic() {
        let cols = var(0, "cols", CType::U64);
        let row = var(1, "row", CType::U64);
        let col = var(2, "col", CType::U64);
        let flat = var(3, "flat", CType::U64);
        let recovered_row = var(4, "recovered_row", CType::U64);
        let recovered_col = var(5, "recovered_col", CType::U64);
        let erased = erased_var(6, "shape");
        let module = GpuModule {
            name: "program_row_major".to_string(),
            source_name: Some(op("row-major-test")),
            entry: GpuFunction {
                name: "program_row_major".to_string(),
                sources: vec![cols.clone(), row.clone(), col.clone()],
                targets: vec![flat.clone(), recovered_row.clone(), recovered_col.clone()],
                assignments: vec![
                    GpuAssign {
                        op: op("row-major-index"),
                        input_sizes: vec![1, 2],
                        output_sizes: vec![1],
                        call_symbol: None,
                        inputs: vec![
                            GpuValue::Var(cols.clone()),
                            GpuValue::Var(erased),
                            GpuValue::Var(row),
                            GpuValue::Var(col),
                        ],
                        outputs: vec![flat.clone()],
                    },
                    GpuAssign {
                        op: op("row-major-row"),
                        input_sizes: vec![1, 1],
                        output_sizes: vec![1],
                        call_symbol: None,
                        inputs: vec![GpuValue::Var(cols.clone()), GpuValue::Var(flat.clone())],
                        outputs: vec![recovered_row],
                    },
                    GpuAssign {
                        op: op("row-major-col"),
                        input_sizes: vec![1, 1],
                        output_sizes: vec![1],
                        call_symbol: None,
                        inputs: vec![GpuValue::Var(cols), GpuValue::Var(flat)],
                        outputs: vec![recovered_col],
                    },
                ],
            },
        };

        let source = render_module(&module, GpuDialect::Hip).unwrap();

        assert!(source.contains("    flat = row * cols + col;\n"));
        assert!(source.contains("    recovered_row = flat / cols;\n"));
        assert!(source.contains("    recovered_col = flat % cols;\n"));
    }

    #[test]
    fn eval_of_primitive_function_symbol_renders_inline() {
        let cols = var(0, "cols", CType::U64);
        let row = var(1, "row", CType::U64);
        let col = var(2, "col", CType::U64);
        let flat = var(3, "flat", CType::U64);
        let module = GpuModule {
            name: "program_eval_primitive".to_string(),
            source_name: Some(op("eval-primitive-test")),
            entry: GpuFunction {
                name: "program_eval_primitive".to_string(),
                sources: vec![cols.clone(), row.clone(), col.clone()],
                targets: vec![flat.clone()],
                assignments: vec![GpuAssign {
                    op: op("eval"),
                    input_sizes: vec![1, 1, 1, 0],
                    output_sizes: vec![1],
                    call_symbol: None,
                    inputs: vec![
                        GpuValue::Var(cols),
                        GpuValue::Var(row),
                        GpuValue::Var(col),
                        GpuValue::FnSymbol(FnPtrSymbol {
                            target: op("row-major-index"),
                        }),
                    ],
                    outputs: vec![flat],
                }],
            },
        };

        let source = render_module(&module, GpuDialect::Hip).unwrap();

        assert!(source.contains("    flat = row * cols + col;\n"));
        assert!(!source.contains("program_row_major_index"));
    }

    #[test]
    fn duplicate_target_nodes_get_distinct_output_parameter_names() {
        let value = var(0, "x0", CType::U64);
        let module = GpuModule {
            name: "program_index_copy".to_string(),
            source_name: Some(op("ix.copy")),
            entry: GpuFunction {
                name: "program_index_copy".to_string(),
                sources: vec![value.clone()],
                targets: vec![value.clone(), value],
                assignments: Vec::new(),
            },
        };

        let source = render_module(&module, GpuDialect::Hip).unwrap();

        assert!(source.contains(
            "extern \"C\" __host__ __device__ void program_index_copy(uint64_t x0, uint64_t *out_x0_0, uint64_t *out_x0_1)"
        ));
        assert!(source.contains("    *out_x0_0 = x0;\n"));
        assert!(source.contains("    *out_x0_1 = x0;\n"));
        assert!(!source.contains("uint64_t *out_x0, uint64_t *out_x0"));
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
