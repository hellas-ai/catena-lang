//! Explicit primitive dispatch. Never erase an unknown arrow merely because its
//! result is a proof: assertions, writes and barriers have erased results too.
mod barriers;
pub(super) mod control;
mod geometry;
pub(super) mod launch;
mod memory;
mod numeric;
mod runtime;
pub(super) mod structure;

use super::{
    CodegenError, Lowerer, invalid,
    ir::*,
    lower_types::{children, runtime as representation},
    values::*,
};

pub(super) fn lower(
    l: &mut Lowerer<'_>,
    op: &str,
    args: Vec<Value>,
    outputs: &[Obj],
) -> Result<Vec<Value>, CodegenError> {
    if op.starts_with("stdlib.numeric.")
        || op.starts_with("type.const.")
        || op.starts_with("const.")
        || op.starts_with("bool.")
        || op == "numeric.u32_subtype_u64"
    {
        return numeric::lower(l, op, &args, outputs);
    }
    if op.starts_with("stdlib.gpu.geometry.") {
        return geometry::lower(l, op, &args, outputs);
    }
    if op.starts_with("stdlib.gpu.memory.") {
        return memory::lower(l, op, &args, outputs);
    }
    if op == "stdlib.gpu.barriers.sync" {
        return barriers::lower(l, op, &args, outputs);
    }
    if op.starts_with("runtime.global_") {
        return runtime::lower(l, op, &args, outputs);
    }
    match op {
        "stdlib.assert.assert_true" => {
            let [value] = args.as_slice() else {
                return Err(invalid(op, "assert_true requires one condition"));
            };
            l.body.push(Instruction::Assert {
                condition: super::expr(value)?.into(),
            });
            outputs.iter().map(|ty| l.erased(ty)).collect()
        }
        "ax-mp"
        | "stdlib.assert.ax_mp"
        | "stdlib.prelude.and_intro"
        | "stdlib.prelude.and_elim"
        | "stdlib.prelude.and_intro.applied.2"
        | "stdlib.prelude.and_elim.applied.2"
        | "stdlib.prelude.truth" => {
            for arg in &args {
                l.erased(&arg.ty)?;
            }
            outputs.iter().map(|ty| l.erased(ty)).collect()
        }
        _ => Err(CodegenError::Unsupported(op.into())),
    }
}

pub(super) fn runtime_args(args: &[Value]) -> Vec<Value> {
    let mut result = Vec::new();
    for arg in args {
        arg.runtime_values(&mut result);
    }
    result
}

/// Fill a primitive's structured results with concrete runtime expressions.
pub(super) fn results(
    l: &Lowerer<'_>,
    outputs: &[Obj],
    values: Vec<Value>,
) -> Result<Vec<Value>, CodegenError> {
    let mut values = values.into_iter();
    let result = outputs
        .iter()
        .map(|ty| l.argument(ty, &mut values))
        .collect::<Result<_, _>>()?;
    if values.next().is_some() {
        return Err(CodegenError::Type("too many primitive results".into()));
    }
    Ok(result)
}

pub(super) fn single_output(outputs: &[Obj]) -> Result<Obj, CodegenError> {
    fn visit(ty: &Obj, found: &mut Vec<Obj>) -> Result<(), CodegenError> {
        if let Some(fields) = children(ty, "*") {
            for field in fields {
                visit(field, found)?;
            }
        } else if representation(ty)?.is_some() {
            found.push(ty.clone());
        }
        Ok(())
    }
    let mut found = vec![];
    for ty in outputs {
        visit(ty, &mut found)?;
    }
    match found.as_slice() {
        [ty] => Ok(ty.clone()),
        _ => Err(CodegenError::Type(format!(
            "expected one runtime result, got {}",
            found.len()
        ))),
    }
}
