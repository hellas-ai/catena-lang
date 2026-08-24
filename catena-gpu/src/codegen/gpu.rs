use std::collections::BTreeSet;

use hexpr::Operation;
use thiserror::Error;

use super::{
    GpuAssign, GpuDialect, GpuFunction, GpuModuleMap, GpuVar, lower_types::CType, runtime_type,
};

#[derive(Debug, Error)]
pub enum GpuRenderError {
    #[error("unsupported GPU operation `{0}`")]
    UnsupportedOp(Operation),
    #[error("operation `{op}` expected {expected} inputs and {outputs} outputs")]
    InvalidArity {
        op: Operation,
        expected: usize,
        outputs: usize,
    },
}

pub fn render_modules(
    modules: &GpuModuleMap,
    dialect: GpuDialect,
) -> Result<String, GpuRenderError> {
    let mut output = format!(
        "#include <{}>\n#include <stdint.h>\n#include <math.h>\n\n",
        dialect.runtime_header()
    );
    for module in modules.values() {
        output.push_str(&signature(&module.entry));
        output.push_str(";\n");
    }
    output.push('\n');
    for module in modules.values() {
        render_function(&mut output, &module.entry)?;
        output.push('\n');
    }
    Ok(output)
}

fn signature(function: &GpuFunction) -> String {
    let mut params = function.sources.iter().map(param).collect::<Vec<_>>();
    params.extend(
        function
            .targets
            .iter()
            .enumerate()
            .map(|(index, var)| format!("{} *out_{index}", c_type(runtime_type(var).unwrap()))),
    );
    format!(
        "extern \"C\" __host__ void {}({})",
        function.name,
        params.join(", ")
    )
}

fn param(var: &GpuVar) -> String {
    format!("{} {}", c_type(runtime_type(var).unwrap()), var.name)
}

fn c_type(ty: &CType) -> &'static str {
    match ty {
        CType::Bool => "uint8_t",
        CType::U32 => "uint32_t",
        CType::U64 => "uint64_t",
        CType::F32 => "float",
    }
}

fn render_function(output: &mut String, function: &GpuFunction) -> Result<(), GpuRenderError> {
    output.push_str(&signature(function));
    output.push_str(" {\n");
    let mut declared = function
        .sources
        .iter()
        .map(|var| var.name.clone())
        .collect::<BTreeSet<_>>();
    for assignment in &function.assignments {
        for var in &assignment.outputs {
            if declared.insert(var.name.clone()) {
                output.push_str(&format!(
                    "    {} {};\n",
                    c_type(runtime_type(var).unwrap()),
                    var.name
                ));
            }
        }
        render_assignment(output, assignment)?;
    }
    for (index, target) in function.targets.iter().enumerate() {
        output.push_str(&format!("    *out_{index} = {};\n", target.name));
    }
    output.push_str("}\n");
    Ok(())
}

fn render_assignment(output: &mut String, assignment: &GpuAssign) -> Result<(), GpuRenderError> {
    if let Some(symbol) = &assignment.call_symbol {
        let mut args = assignment
            .inputs
            .iter()
            .map(|var| var.name.clone())
            .collect::<Vec<_>>();
        args.extend(
            assignment
                .outputs
                .iter()
                .map(|var| format!("&{}", var.name)),
        );
        output.push_str(&format!("    {symbol}({});\n", args.join(", ")));
        return Ok(());
    }
    let op = assignment.op.as_str();
    match op {
        "bool.t" => unary_output(output, assignment, "1", 0),
        "bool.f" | "u64.zero" | "u32.zero" => unary_output(output, assignment, "0", 0),
        "u64.one" | "u32.one" | "f32.one" => unary_output(output, assignment, "1", 0),
        "bool.not" => unary(output, assignment, "!"),
        "u32.not" => unary(output, assignment, "~"),
        "f32.neg" => unary(output, assignment, "-"),
        "bool.and" => binary(output, assignment, "&&"),
        "bool.or" => binary(output, assignment, "||"),
        "u64.add" | "u32.add" | "f32.add" => binary(output, assignment, "+"),
        "u64.sub" | "u32.sub" | "f32.sub" => binary(output, assignment, "-"),
        "u64.mul" | "u32.mul" | "f32.mul" => binary(output, assignment, "*"),
        "u64.div" | "f32.div" => binary(output, assignment, "/"),
        "u64.rem" => binary(output, assignment, "%"),
        "u32.and" => binary(output, assignment, "&"),
        "u32.or" => binary(output, assignment, "|"),
        "u32.xor" => binary(output, assignment, "^"),
        "u32.shl" => binary(output, assignment, "<<"),
        "u32.shr" => binary(output, assignment, ">>"),
        "u64.ne" | "u32.ne" | "f32.ne" => binary(output, assignment, "!="),
        "u32.eq" | "f32.eq" => binary(output, assignment, "=="),
        "u64.lt" | "u32.lt" | "f32.lt" => binary(output, assignment, "<"),
        "u32.gt" | "f32.gt" => binary(output, assignment, ">"),
        "u64.lte" | "u32.lte" | "f32.lte" => binary(output, assignment, "<="),
        "u64.gte" | "u32.gte" | "f32.gte" => binary(output, assignment, ">="),
        "bool.select" | "u64.select" | "u32.select" | "f32.select" => select(output, assignment),
        "u32.to-f32" => cast(output, assignment, "float"),
        "u64.to-f32" => cast(output, assignment, "float"),
        "u32.bitcast-f32" => bitcast(output, assignment, "float", "uint32_t"),
        "f32.bitcast-u32" => bitcast(output, assignment, "uint32_t", "float"),
        "f32.round-to-u32" => call1(output, assignment, "(uint32_t)roundf"),
        "f32.fma" => call3(output, assignment, "fmaf"),
        _ => Err(GpuRenderError::UnsupportedOp(assignment.op.clone())),
    }
}

fn unary_output(
    output: &mut String,
    a: &GpuAssign,
    expression: &str,
    inputs: usize,
) -> Result<(), GpuRenderError> {
    if a.inputs.len() != inputs || a.outputs.len() != 1 {
        return Err(arity(a, inputs));
    }
    output.push_str(&format!("    {} = {expression};\n", a.outputs[0].name));
    Ok(())
}

fn unary(output: &mut String, a: &GpuAssign, operator: &str) -> Result<(), GpuRenderError> {
    unary_output(output, a, &format!("{operator}{}", input(a, 0)?), 1)
}

fn binary(output: &mut String, a: &GpuAssign, operator: &str) -> Result<(), GpuRenderError> {
    unary_output(
        output,
        a,
        &format!("{} {operator} {}", input(a, 0)?, input(a, 1)?),
        2,
    )
}

fn select(output: &mut String, a: &GpuAssign) -> Result<(), GpuRenderError> {
    unary_output(
        output,
        a,
        &format!("{} ? {} : {}", input(a, 0)?, input(a, 1)?, input(a, 2)?),
        3,
    )
}

fn cast(output: &mut String, a: &GpuAssign, ty: &str) -> Result<(), GpuRenderError> {
    unary_output(output, a, &format!("({ty}){}", input(a, 0)?), 1)
}

fn call1(output: &mut String, a: &GpuAssign, function: &str) -> Result<(), GpuRenderError> {
    unary_output(output, a, &format!("{function}({})", input(a, 0)?), 1)
}

fn call3(output: &mut String, a: &GpuAssign, function: &str) -> Result<(), GpuRenderError> {
    unary_output(
        output,
        a,
        &format!(
            "{function}({}, {}, {})",
            input(a, 0)?,
            input(a, 1)?,
            input(a, 2)?
        ),
        3,
    )
}

fn bitcast(output: &mut String, a: &GpuAssign, to: &str, from: &str) -> Result<(), GpuRenderError> {
    if a.inputs.len() != 1 || a.outputs.len() != 1 {
        return Err(arity(a, 1));
    }
    output.push_str(&format!(
        "    {{ union {{ {from} from; {to} to; }} bits = {{ {} }}; {} = bits.to; }}\n",
        a.inputs[0].name, a.outputs[0].name
    ));
    Ok(())
}

fn input(a: &GpuAssign, index: usize) -> Result<&str, GpuRenderError> {
    a.inputs
        .get(index)
        .map(|var| var.name.as_str())
        .ok_or_else(|| arity(a, index + 1))
}

fn arity(a: &GpuAssign, expected: usize) -> GpuRenderError {
    GpuRenderError::InvalidArity {
        op: a.op.clone(),
        expected,
        outputs: a.outputs.len(),
    }
}
