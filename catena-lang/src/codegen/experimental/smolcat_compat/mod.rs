//! Temporary exporter compatibility, deliberately separate from language primitives.
mod family_bridges;
mod meta;
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
) -> Result<Option<Vec<Value>>, CodegenError> {
    if op.starts_with("meta.") {
        return meta::lower(l, op, args, outputs).map(Some);
    }
    if op.starts_with("smolcat.apply.") {
        return family_bridges::lower(l, op, args, outputs).map(Some);
    }
    match op {
        "smolcat.callback.symbol" | "smolcat.value.named" => {
            symbols::lower(l, op, args, outputs).map(Some)
        }
        _ if op.starts_with("smolcat.") => Err(CodegenError::Unsupported(op.into())),
        _ => Ok(None),
    }
}
