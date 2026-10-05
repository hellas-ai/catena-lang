//! Smolcat witness operations and legacy proof-family bridges.
mod family_bridges;
mod symbols;

use super::{
    CodegenError, Lowerer,
    values::{Obj, Value},
};

pub(crate) fn lower(
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
    if op.starts_with("smolcat.apply.") {
        return family_bridges::lower(l, op, args, outputs);
    }
    match op {
        "smolcat.callback.symbol" | "smolcat.value.named" | "smolcat.value.named_at" => {
            symbols::lower(l, op, args, outputs)
        }
        _ => Err(CodegenError::Unsupported(op.into())),
    }
}
