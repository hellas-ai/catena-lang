//! Lower converted function/environment pairs into structured control flow.
use super::super::{
    CodegenError, Lowerer, expr, invalid,
    ir::*,
    lower_types::{node, runtime as representation},
    ops::{results, runtime_args},
    values::*,
};
pub(in crate::codegen::experimental) fn lower(
    l: &mut Lowerer<'_>,
    op: &str,
    args: Vec<Value>,
    outputs: &[Obj],
) -> Result<Vec<Value>, CodegenError> {
    match op {
        "core.if" | "core.if_guarded" => conditional(l, op, &args, outputs),
        "core.fold.bounded"
        | "core.fold.trace"
        | "core.fold.bounded_u64"
        | "core.fold.trace_u64" => fold(l, op, &args, outputs),
        _ => unreachable!("unexpected callback operation: {op}"),
    }
}

fn conditional(
    l: &mut Lowerer<'_>,
    op: &str,
    args: &[Value],
    outputs: &[Obj],
) -> Result<Vec<Value>, CodegenError> {
    if args.len() < 7 {
        return Err(invalid(op, "missing branch operands"));
    }
    let condition = expr(&args[0])?.to_owned();
    let input = runtime_args(&[args[1].clone(), args[6].clone()]);
    let outer = std::mem::take(&mut l.body);
    let mut bodies = vec![];
    let mut branches = vec![];
    for pair in args[2..6].chunks_exact(2) {
        let result = l.callback(&pair[1], &pair[0], input.clone())?;
        branches.push(runtime_args(&result));
        bodies.push(std::mem::take(&mut l.body));
    }
    l.body = outer;
    if branches[0].len() != branches[1].len() {
        return Err(invalid(op, "branch result shapes differ"));
    }
    let mut merged = vec![];
    for (a, b) in branches[0].iter().zip(&branches[1]) {
        if representation(&a.ty)? != representation(&b.ty)? {
            return Err(invalid(op, "branch result types differ"));
        }
        let result = l.emit(&a.ty, "{}")?;
        bodies[0].push(Instruction::Assign(expr(&result)?.into(), expr(a)?.into()));
        bodies[1].push(Instruction::Assign(expr(&result)?.into(), expr(b)?.into()));
        merged.push(result);
    }
    let no = bodies.pop().unwrap();
    let yes = bodies.pop().unwrap();
    l.body.push(Instruction::If { condition, yes, no });
    results(l, outputs, merged)
}

fn fold(
    l: &mut Lowerer<'_>,
    op: &str,
    args: &[Value],
    outputs: &[Obj],
) -> Result<Vec<Value>, CodegenError> {
    if args.len() < 5 {
        return Err(invalid(op, "missing fold operands"));
    }
    let index_type = if op.ends_with("_u64") { "u64" } else { "u32" };
    let index_ty = node("val", vec![node(index_type, vec![])]);
    if representation(&args[0].ty)? != representation(&index_ty)? {
        return Err(invalid(op, "fold bound type mismatch"));
    }
    let end = expr(&args[0])?.to_owned();
    let initial = runtime_args(&[args[1].clone(), args[4].clone()]);
    let mut state = vec![];
    for v in initial {
        state.push(l.emit(&v.ty, expr(&v)?)?);
    }
    let outer = std::mem::take(&mut l.body);
    let index = l.fresh("index");
    let mut fields = vec![Value {
        ty: index_ty.clone(),
        repr: Repr::Runtime(index.clone()),
    }];
    fields.extend(state.clone());
    let result = l.callback(&args[3], &args[2], fields)?;
    let next = runtime_args(&result);
    if next.len() != state.len() {
        return Err(invalid(op, "fold state shape changed"));
    }
    // Snapshot all fields before assigning, so state permutations are simultaneous.
    let mut snapshots = vec![];
    for (old, new) in state.iter().zip(&next) {
        if representation(&old.ty)? != representation(&new.ty)? {
            return Err(invalid(op, "fold state type changed"));
        }
        snapshots.push(l.emit(&old.ty, expr(new)?)?);
    }
    for (old, new) in state.iter().zip(&snapshots) {
        l.body
            .push(Instruction::Assign(expr(old)?.into(), expr(new)?.into()));
    }
    let body = std::mem::replace(&mut l.body, outer);
    l.body.push(Instruction::For {
        index,
        index_type: representation(&index_ty)?.unwrap(),
        end,
        body,
    });
    results(l, outputs, state)
}
