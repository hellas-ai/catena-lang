//! Scalar bit and conversion primitives used by the exported math definitions.
use super::super::{
    CodegenError, Lowerer, expr, invalid,
    ir::*,
    lower_types::{CType, runtime},
    values::*,
};
use super::{results, runtime_args, single_output};

pub(super) fn lower(
    l: &mut Lowerer<'_>,
    op: &str,
    args: &[Value],
    outputs: &[Obj],
) -> Result<Vec<Value>, CodegenError> {
    let (inputs, output, helper) = match op {
        "stdlib.math.f32_bits" => (vec![CType::F32], CType::U32, "catena_f32_bitcast_u32"),
        "stdlib.math.from_bits" => (vec![CType::U32], CType::F32, "catena_u32_bitcast_f32"),
        "stdlib.math.round_to_u32" => (vec![CType::F32], CType::U32, "catena_round_to_u32"),
        "stdlib.math.shift_right" => (vec![CType::U32, CType::U32], CType::U32, ""),
        _ => return Err(CodegenError::Unsupported(op.into())),
    };
    let values = runtime_args(args);
    let ty = single_output(outputs)?;
    if values.len() != inputs.len()
        || runtime(&ty)? != Some(output)
        || values
            .iter()
            .zip(inputs)
            .any(|(value, expected)| runtime(&value.ty).ok().flatten() != Some(expected))
    {
        return Err(invalid(op, "math operand or result type mismatch"));
    }
    let xs = values.iter().map(expr).collect::<Result<Vec<_>, _>>()?;
    let expression = if op == "stdlib.math.shift_right" {
        l.body.push(Instruction::Assert {
            condition: format!("{} < 32", xs[1]),
        });
        format!("({} >> {})", xs[0], xs[1])
    } else {
        format!("{helper}({})", xs[0])
    };
    let value = l.emit(&ty, expression)?;
    results(l, outputs, vec![value])
}
