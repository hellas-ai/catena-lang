//! NVIDIA fragments preserve PTX register order without layout conversions.
use super::super::{
    CodegenError, Lowerer, expr, invalid,
    ir::{Instruction, Place, Variable},
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
    if l.place != Place::Device {
        return Err(invalid(op, "NVIDIA MMA requires device execution"));
    }
    let ty = single_output(outputs)?;
    let out = runtime(&ty)?.ok_or_else(|| invalid(op, "expected runtime result"))?;
    let args = runtime_args(args);
    let types = args
        .iter()
        .map(|a| runtime(&a.ty))
        .collect::<Result<Vec<_>, _>>()?;
    let xs = args.iter().map(expr).collect::<Result<Vec<_>, _>>()?;
    use CType::*;
    let (expected, inputs, expression) = match op {
        "mma.m16n8k16.bf16.a.pack" => (
            MmaBf16A,
            vec![Some(Bf16x2); 4],
            format!("catena_mma_a_pack({})", xs.join(",")),
        ),
        "mma.m16n8k16.bf16.b.pack" => (
            MmaBf16B,
            vec![Some(Bf16x2); 2],
            format!("catena_mma_b_pack({})", xs.join(",")),
        ),
        "mma.m16n8k16.f32.acc.pack" => (
            MmaF32Acc,
            vec![Some(F32); 4],
            format!("catena_mma_f32_acc_t{{{{{}}}}}", xs.join(",")),
        ),
        "mma.m16n8k16.f32.acc.zero" => (
            MmaF32Acc,
            vec![],
            "catena_mma_f32_acc_t{{0.0f,0.0f,0.0f,0.0f}}".into(),
        ),
        "mma.m16n8k16.f32.acc.get" => (
            F32,
            vec![Some(MmaF32Acc), Some(U32)],
            format!("catena_mma_acc_get({})", xs.join(",")),
        ),
        "mma.m16n8k16.bf16-f32" => (
            MmaF32Acc,
            vec![Some(MmaBf16A), Some(MmaBf16B), Some(MmaF32Acc)],
            String::new(),
        ),
        _ => return Err(CodegenError::Unsupported(op.into())),
    };
    if types != inputs || out != expected {
        return Err(invalid(op, "NVIDIA MMA operand or result type mismatch"));
    }
    l.modules.bf16 = true;
    l.modules.nvidia_mma = true;
    let v = if op == "mma.m16n8k16.bf16-f32" {
        let name = l.fresh("v");
        let variable = |name: &str, ty| Variable {
            name: name.into(),
            ty,
        };
        l.body.push(Instruction::NvidiaMma {
            result: variable(&name, MmaF32Acc),
            a: variable(xs[0], MmaBf16A),
            b: variable(xs[1], MmaBf16B),
            accumulator: variable(xs[2], MmaF32Acc),
        });
        Value {
            ty,
            repr: Repr::Runtime(name),
        }
    } else {
        l.emit(&ty, expression)?
    };
    results(l, outputs, vec![v])
}
