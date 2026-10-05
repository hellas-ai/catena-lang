use super::super::{
    CodegenError, Lowerer, expr, invalid,
    ir::*,
    lower_types::{CType, children, concrete, runtime},
    values::*,
};
use super::{results, runtime_args, single_output};

pub(super) fn lower(
    l: &mut Lowerer<'_>,
    op: &str,
    args: &[Value],
    outputs: &[Obj],
) -> Result<Vec<Value>, CodegenError> {
    if matches!(
        op,
        "numeric.u32_subtype_u64"
            | "stdlib.numeric.integral.int"
            | "stdlib.numeric.integral.u32"
            | "stdlib.numeric.integral.u64"
            | "stdlib.numeric.numeric.int"
            | "stdlib.numeric.numeric.u32"
            | "stdlib.numeric.numeric.u64"
            | "stdlib.numeric.numeric.f32"
            | "stdlib.numeric.signed.int"
            | "stdlib.numeric.signed.f32"
    ) {
        return outputs.iter().map(|t| l.erased(t)).collect();
    }
    let ty = single_output(outputs)?;
    let out = runtime(&ty)?.unwrap();
    let values = runtime_args(args);
    let xs = values.iter().map(expr).collect::<Result<Vec<_>, _>>()?;
    let name = op.strip_prefix("stdlib.numeric.").unwrap_or(op);
    if op.starts_with("bool.")
        && (out != CType::Bool
            || values
                .iter()
                .any(|v| runtime(&v.ty).ok().flatten() != Some(CType::Bool)))
    {
        return Err(invalid(op, "boolean operand or result type mismatch"));
    }
    let expression = if let Some(constant) = op
        .strip_prefix("type.const.")
        .or_else(|| op.strip_prefix("const."))
    {
        let (kind, hex) = constant
            .split_once(".0x")
            .ok_or_else(|| invalid(op, "invalid integer constant"))?;
        let (width, expected) = match kind {
            "u32" => (8, CType::U32),
            "u64" => (16, CType::U64),
            _ => return Err(CodegenError::Unsupported(op.into())),
        };
        if !xs.is_empty()
            || out != expected
            || hex.len() != width
            || !hex.bytes().all(|b| b.is_ascii_hexdigit())
        {
            return Err(invalid(op, "invalid integer constant type or width"));
        }
        format!("UINT{}_C(0x{hex})", width * 4)
    } else {
        match (name, xs.as_slice()) {
            ("bool.t", []) => "1".into(),
            ("bool.f", []) => "0".into(),
            ("bool.not", [x]) => format!("!{x}"),
            ("bool.and", [x, y]) => format!("{x} && {y}"),
            ("bool.or", [x, y]) => format!("{x} || {y}"),
            ("zero", []) if out.numeric() => "0".into(),
            ("numeric_size", []) if out == CType::U32 => {
                let element = args
                    .iter()
                    .find_map(|v| {
                        children(&v.ty, "|-")
                            .and_then(|p| p.first())
                            .and_then(|p| children(p, "stdlib.numeric.prop.Numeric"))
                            .and_then(|p| p.first())
                    })
                    .ok_or_else(|| invalid(op, "missing Numeric evidence"))?;
                format!("sizeof({})", concrete(element)?.c_name())
            }
            ("coerce", [x]) if runtime(&values[0].ty)? == Some(CType::U32) && out == CType::U64 => {
                format!("uint64_t({x})")
            }
            ("negate", [x]) if out == CType::F32 && runtime(&values[0].ty)? == Some(CType::F32) => {
                format!("-{x}")
            }
            (
                "add" | "subtract" | "multiply" | "divide" | "remainder" | "ceil_div"
                | "ceil_div_u32" | "equal" | "notequal" | "less" | "lessequal" | "greater"
                | "greaterequal" | "min" | "max",
                [x, y],
            ) => {
                let input = runtime(&values[0].ty)?.unwrap();
                if !input.numeric() || runtime(&values[1].ty)? != Some(input.clone()) {
                    return Err(invalid(op, "numeric operand type mismatch"));
                }
                let comparison = matches!(
                    name,
                    "equal" | "notequal" | "less" | "lessequal" | "greater" | "greaterequal"
                );
                if out
                    != if comparison {
                        CType::Bool
                    } else {
                        input.clone()
                    }
                {
                    return Err(invalid(op, "numeric result type mismatch"));
                }
                if matches!(name, "remainder" | "ceil_div" | "ceil_div_u32") && !input.integral() {
                    return Err(invalid(op, "requires integral operands"));
                }
                if matches!(name, "divide" | "remainder" | "ceil_div" | "ceil_div_u32")
                    && input.integral()
                {
                    l.body.push(Instruction::Assert {
                        condition: format!("{y} != 0"),
                    });
                }
                match name {
                    "min" => format!("({x} < {y} ? {x} : {y})"),
                    "max" => format!("({x} > {y} ? {x} : {y})"),
                    "ceil_div" | "ceil_div_u32" => format!("({x} / {y} + ({x} % {y} != 0))"),
                    _ => {
                        let operator = match name {
                            "add" => "+",
                            "subtract" => "-",
                            "multiply" => "*",
                            "divide" => "/",
                            "remainder" => "%",
                            "equal" => "==",
                            "notequal" => "!=",
                            "less" => "<",
                            "lessequal" => "<=",
                            "greater" => ">",
                            "greaterequal" => ">=",
                            _ => unreachable!(),
                        };
                        // Avoid signed C++ integer promotion for u16 multiplication.
                        if input == CType::U16 {
                            format!("(uint32_t({x}) {operator} uint32_t({y}))")
                        } else {
                            format!("({x} {operator} {y})")
                        }
                    }
                }
            }
            _ => return Err(CodegenError::Unsupported(op.into())),
        }
    };
    let value = l.emit(&ty, expression)?;
    results(l, outputs, vec![value])
}
