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
    let [
        condition,
        context,
        yes_env,
        yes_fn,
        no_env,
        no_fn,
        contracts,
        extra @ ..,
    ] = args
    else {
        return Err(invalid(op, "invalid conditional operands"));
    };
    match (op, extra) {
        ("core.if", []) => {}
        ("core.if_guarded", [permission]) => {
            l.erased(&permission.ty)?;
        }
        _ => return Err(invalid(op, "invalid conditional operands")),
    }
    let condition = expr(condition)?.to_owned();
    let input = runtime_args(&[context.clone(), contracts.clone()]);
    let outer = std::mem::take(&mut l.body);
    let mut bodies = vec![];
    let mut branches = vec![];
    for (environment, function) in [(yes_env, yes_fn), (no_env, no_fn)] {
        let result = l.callback(function, environment, input.clone())?;
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
    let (bound, initial, environment, function, invariant) = match (op, args) {
        (
            "core.fold.bounded" | "core.fold.bounded_u64",
            [bound, initial, environment, function, invariant],
        ) => (bound, initial, environment, function, invariant),
        (
            "core.fold.trace" | "core.fold.trace_u64",
            [bound, initial, environment, function, invariant, trace],
        ) => {
            l.erased(&trace.ty)?;
            (bound, initial, environment, function, invariant)
        }
        _ => return Err(invalid(op, "invalid fold operands")),
    };
    let index_type = if op.ends_with("_u64") { "u64" } else { "u32" };
    let index_ty = node("val", vec![node(index_type, vec![])]);
    if representation(&bound.ty)? != representation(&index_ty)? {
        return Err(invalid(op, "fold bound type mismatch"));
    }
    let end = expr(bound)?.to_owned();
    let initial = runtime_args(&[initial.clone(), invariant.clone()]);
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
    let result = l.callback(function, environment, fields)?;
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
