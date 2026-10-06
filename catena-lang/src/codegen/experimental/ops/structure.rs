//! Preserve runtime values across type ascriptions.
use super::super::{CodegenError, Lowerer, invalid, values::*};
pub(in crate::codegen::experimental) fn lower(
    l: &mut Lowerer<'_>,
    op: &str,
    args: Vec<Value>,
    outputs: &[Obj],
) -> Result<Vec<Value>, CodegenError> {
    match op {
        ":.forget" | ":.ty" | ":.param" => {
            let [value] = args.as_slice() else {
                return Err(invalid(op, "expected one value"));
            };
            let mut results = Vec::new();
            for (i, ty) in outputs.iter().enumerate() {
                if i == 0 && op != ":.param" {
                    let mut v = value.clone();
                    v.ty = ty.clone();
                    results.push(v);
                } else {
                    results.push(l.erased(ty)?);
                }
            }
            Ok(results)
        }
        _ => unreachable!("unexpected structural operation: {op}"),
    }
}
