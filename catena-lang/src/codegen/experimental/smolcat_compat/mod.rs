//! Smolcat symbolic-witness operations, separate from meta projections.
mod symbols;

use super::{
    CodegenError, Lowerer,
    values::{Obj, Value},
};

pub(super) fn lower(
    l: &mut Lowerer<'_>,
    op: &str,
    args: &[Value],
    outputs: &[Obj],
) -> Result<Vec<Value>, CodegenError> {
    let mut flat = Vec::new();
    for arg in args {
        arg.clone().flatten(&mut flat);
    }
    let args = flat.as_slice();
    match op {
        "smolcat.callback.symbol" | "smolcat.value.named" | "smolcat.value.named_at" => {
            symbols::lower(l, op, args, outputs)
        }
        _ => Err(CodegenError::Unsupported(op.into())),
    }
}
