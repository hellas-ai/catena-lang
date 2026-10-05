use super::super::{CodegenError, Lowerer, invalid, ir::*, values::*};

pub(super) fn lower(
    l: &mut Lowerer<'_>,
    op: &str,
    _args: &[Value],
    outputs: &[Obj],
) -> Result<Vec<Value>, CodegenError> {
    if l.place != Place::Device {
        return Err(invalid(op, "barrier requires device execution"));
    }
    l.body.push(Instruction::Sync);
    outputs.iter().map(|ty| l.erased(ty)).collect()
}
