//! Exporter meta projections forward their data input prefix, then expose
//! symbolic context. Do not infer runtime meaning from their generated names.
use super::super::{
    CodegenError, Lowerer, invalid,
    values::{Obj, Repr, Value},
};

pub(super) fn lower(
    l: &Lowerer<'_>,
    op: &str,
    args: &[Value],
    outputs: &[Obj],
) -> Result<Vec<Value>, CodegenError> {
    let prefix = args
        .iter()
        .zip(outputs)
        .take_while(|(v, t)| &v.ty == *t)
        .count();
    // Explicit symbolic inputs can be consumed/reordered by *.arguments arrows.
    if args[prefix..]
        .iter()
        .any(|v| !matches!(v.repr, Repr::Erased))
    {
        return Err(invalid(
            op,
            "Smolcat meta projection must preserve its typed data prefix",
        ));
    }
    let mut result = args[..prefix].to_vec();
    for ty in &outputs[prefix..] {
        result.push(l.erased(ty).map_err(|_| {
            invalid(
                op,
                "Smolcat meta projection adds a runtime value or closure",
            )
        })?);
    }
    Ok(result)
}
