//! BF16 operations retain BF16 operands/results. Packed dot is native BF16/F32.
use super::super::{
    CodegenError, Lowerer, expr, invalid,
    ir::Place,
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
    let ty = single_output(outputs)?;
    let out = runtime(&ty)?.ok_or_else(|| invalid(op, "expected runtime result"))?;
    let args = runtime_args(args);
    let types = args
        .iter()
        .map(|a| runtime(&a.ty))
        .collect::<Result<Vec<_>, _>>()?;
    let xs = args.iter().map(expr).collect::<Result<Vec<_>, _>>()?;
    let b = Some(CType::Bf16);
    let p = Some(CType::Bf16x2);
    if op.starts_with("wmma.") {
        if l.place != Place::Device {
            return Err(invalid(op, "WMMA requires device execution"));
        }
        l.modules.wmma = true;
    }
    let (expected, inputs, expression) = match op {
        "wmma.bf16x16.pack" => (
            CType::WmmaBf16x16,
            vec![p.clone(); 8],
            format!("catena_wmma_pack({})", xs.join(",")),
        ),
        "wmma.f32x8.pack" => (
            CType::WmmaF32x8,
            vec![Some(CType::F32); 8],
            format!("catena_wmma_f32x8_t{{{{{}}}}}", xs.join(",")),
        ),
        "wmma.f32x8.zero" => (CType::WmmaF32x8, vec![], "catena_wmma_zero()".into()),
        "wmma.f32x8.get" => (
            CType::F32,
            vec![Some(CType::WmmaF32x8), Some(CType::U32)],
            format!("catena_wmma_get({})", xs.join(",")),
        ),
        "wmma.mma-bf16-f32" => (
            CType::WmmaF32x8,
            vec![
                Some(CType::WmmaBf16x16),
                Some(CType::WmmaBf16x16),
                Some(CType::WmmaF32x8),
            ],
            format!("catena_wmma_mma({})", xs.join(",")),
        ),
        "bf16.to-f32" => (
            CType::F32,
            vec![b.clone()],
            format!("catena_bf16_to_f32({})", xs.join(",")),
        ),
        "f32.to-bf16" => (
            CType::Bf16,
            vec![Some(CType::F32)],
            format!("catena_bf16_from_f32({})", xs.join(",")),
        ),
        "bf16x2.pack" => (
            CType::Bf16x2,
            vec![b.clone(); 2],
            format!("catena_bf16_pack({})", xs.join(",")),
        ),
        "bf16x2.low" | "bf16x2.high" => (
            CType::Bf16,
            vec![p.clone()],
            format!(
                "catena_bf16_{}({})",
                op.trim_start_matches("bf16x2."),
                xs.join(",")
            ),
        ),
        "bf16x2.dot-f32" => {
            if l.place != Place::Device {
                return Err(invalid(op, "packed dot requires device execution"));
            }
            l.modules.bf16_dot = true;
            (
                CType::F32,
                vec![p.clone(), p, Some(CType::F32)],
                format!("catena_bf16_dot({})", xs.join(",")),
            )
        }
        _ => return Err(CodegenError::Unsupported(op.into())),
    };
    if types != inputs || out != expected {
        return Err(invalid(op, "BF16 operand or result type mismatch"));
    }
    l.modules.bf16 = true;
    let v = l.emit(&ty, expression)?;
    results(l, outputs, vec![v])
}
