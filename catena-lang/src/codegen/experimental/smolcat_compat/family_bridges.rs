//! The unusually long exporter-generated family names are a temporary wire
//! format. Programs generate further names from their propositions. Accept only
//! pack/unpack bridges whose checked types preserve the complete proposition.
use super::super::{
    CodegenError, Lowerer, invalid,
    lower_types::children,
    values::{Obj, Value},
};

pub(super) fn lower(
    l: &Lowerer<'_>,
    op: &str,
    args: &[Value],
    outputs: &[Obj],
) -> Result<Vec<Value>, CodegenError> {
    if !op.starts_with("smolcat.apply.2.") || !(op.ends_with(".pack") || op.ends_with(".unpack")) {
        return Err(CodegenError::Unsupported(op.into()));
    }
    let ([value], [ty]) = (args, outputs) else {
        return Err(invalid(
            op,
            "family bridge must have one proof input and output",
        ));
    };
    let (plain, applied) = if op.ends_with(".pack") {
        (&value.ty, ty)
    } else {
        (ty, &value.ty)
    };
    let Some([payload, _, _]) = children(applied, "smolcat.family.Apply.2") else {
        return Err(invalid(op, "expected an applied proof family"));
    };
    if payload != plain || !matches!(children(plain, "|-"), Some([_])) {
        return Err(invalid(
            op,
            "family bridge must preserve its complete proposition",
        ));
    }
    Ok(vec![l.erased(ty)?])
}
