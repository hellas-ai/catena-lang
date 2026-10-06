use super::*;

fn scalar(name: &str) -> Obj {
    node("val", vec![node(name, vec![])])
}
fn proof() -> Obj {
    node("|-", vec![node("test.proposition", vec![])])
}
fn value(ty: Obj, name: &str) -> Value {
    Value {
        ty,
        repr: Repr::Runtime(name.into()),
    }
}
fn lowerer<'a>(templates: &'a BTreeMap<Operation, Template>) -> Lowerer<'a> {
    Lowerer::new(templates)
}

#[test]
fn generic_entrypoint_interfaces_fail_during_codegen() {
    use crate::pass::record_boundary_sizes::OperationWithBoundarySizes;
    use open_hypergraphs::lax::OpenHypergraph;

    let generic = node("val", vec![Tree::Leaf(0, ())]);
    for ty in [generic.clone(), node("*", vec![scalar("u32"), generic])] {
        for is_input in [true, false] {
            let (sources, targets) = if is_input {
                (vec![ty.clone()], vec![])
            } else {
                (vec![], vec![ty.clone()])
            };
            let term = OpenHypergraph::singleton(
                OperationWithBoundarySizes {
                    operation: "meta.test".parse().unwrap(),
                    source_sizes: vec![1; sources.len()],
                    target_sizes: vec![1; targets.len()],
                },
                sources,
                targets,
            );
            let terms = BTreeMap::from([(
                TheoryId("program".parse().unwrap()),
                BTreeMap::from([("test".parse().unwrap(), term)]),
            )]);
            assert!(matches!(
                codegen(&terms, GpuDialect::Cuda),
                Err(CodegenError::NoRuntimeRepresentation(actual)) if actual == ty
            ));
        }
    }
}

#[test]
fn unresolved_runtime_type_has_no_runtime_representation() {
    let ty = Tree::Leaf(0, ());
    assert!(matches!(
        runtime(&node("val", vec![ty.clone()])),
        Err(CodegenError::NoRuntimeRepresentation(actual)) if actual == ty
    ));
}

#[test]
fn meta_forwards_data_and_rejects_runtime_invention() {
    let templates = BTreeMap::new();
    let mut l = lowerer(&templates);
    let ty = scalar("u32");
    let input = value(ty.clone(), "input");
    let symbol = Tree::Leaf(99, ());
    let result = l
        .lower_operation("meta.test", vec![input.clone()], &[ty.clone(), symbol])
        .unwrap();
    assert_eq!(expr(&result[0]).unwrap(), "input");
    assert!(l.body.is_empty());
    assert!(
        l.lower_operation("meta.test", vec![input], &[ty.clone(), ty])
            .is_err()
    );
}

#[test]
fn unknown_erased_arrows_are_errors() {
    let templates = BTreeMap::new();
    let mut l = lowerer(&templates);
    for op in [
        "unknown.effect",
        "smolcat.apply.2.unknown.pack",
        "stdlib.numeric.unknown",
        "stdlib.gpu.scheduling.unknown",
    ] {
        assert!(l.lower_operation(op, vec![], &[proof()]).is_err(), "{op}");
    }
}

#[test]
fn writes_assertions_and_barriers_survive_proof_erasure() {
    let templates = BTreeMap::new();
    let mut l = lowerer(&templates);
    l.lower_operation(
        "stdlib.assert.assert_true",
        vec![value(scalar("bool"), "condition")],
        &[proof()],
    )
    .unwrap();
    assert!(
        l.lower_operation("stdlib.gpu.barriers.sync", vec![], &[proof()])
            .is_err()
    );
    l.place = Place::Device;
    let memory = node(
        "val",
        vec![node(
            "stdlib.gpu.memory.type.Global",
            vec![Tree::Leaf(0, ()), node("u32", vec![])],
        )],
    );
    l.lower_operation(
        "stdlib.gpu.memory.global_write",
        vec![
            value(memory, "buffer"),
            value(scalar("u32"), "index"),
            value(scalar("u32"), "item"),
            Value::erased(proof()),
        ],
        &[node("1", vec![])],
    )
    .unwrap();
    l.lower_operation("stdlib.gpu.barriers.sync", vec![], &[proof()])
        .unwrap();
    assert!(l.body.iter().any(
        |i| matches!(i,Instruction::Store {buffer,index,value} if buffer.name=="buffer" && index.name=="index" && value.name=="item")
    ));
    assert!(l.body.iter().any(|i| matches!(i, Instruction::Sync)));
    assert!(
        l.body
            .iter()
            .any(|i| matches!(i,Instruction::Assert {condition} if condition=="condition"))
    );
}

#[test]
fn fold_emits_loop_and_preserves_carried_state() {
    let templates = BTreeMap::new();
    let mut l = lowerer(&templates);
    let ty = scalar("u32");
    let domain = node("*", vec![ty.clone(), ty.clone()]);
    let callback = Value {
        ty: node("val", vec![node("->", vec![domain, ty.clone()])]),
        repr: Repr::Function("stdlib.numeric.+".parse().unwrap()),
    };
    let result = l
        .lower_operation(
            "core.fold.bounded",
            vec![
                value(ty.clone(), "end"),
                value(ty.clone(), "initial"),
                Value::erased(node("1", vec![])),
                callback,
                Value::erased(proof()),
            ],
            &[ty, proof()],
        )
        .unwrap();
    assert_eq!(result.len(), 2);
    let Instruction::For { body, .. } = l.body.last().unwrap() else {
        panic!("expected loop")
    };
    assert!(
        body.iter()
            .any(|i| matches!(i,Instruction::Let(_,s) if s.contains(" + ")))
    );
    assert!(matches!(body.last(), Some(Instruction::Assign(..))));
}

#[test]
fn launch_captures_values_and_emits_both_dialects() {
    let templates = BTreeMap::new();
    let mut l = lowerer(&templates);
    let grid_type = node(
        "val",
        vec![node(
            "stdlib.gpu.geometry.type.Grid",
            vec![Tree::Leaf(0, ()), Tree::Leaf(1, ())],
        )],
    );
    let shared_type = node(
        "val",
        vec![node(
            "stdlib.gpu.memory.type.SharedLayout",
            vec![node("list.Nil", vec![])],
        )],
    );
    let unit = node("1", vec![]);
    let flag = scalar("bool");
    let callback = Value {
        ty: node("val", vec![node("->", vec![flag.clone(), proof()])]),
        repr: Repr::Function("stdlib.assert.assert_true".parse().unwrap()),
    };
    let result = l
        .lower_operation(
            "unsafe.launch_shared",
            vec![
                value(grid_type, "host_grid"),
                value(shared_type, "host_layout"),
                value(flag, "host_flag"),
                callback,
            ],
            &[unit],
        )
        .unwrap();
    assert!(matches!(result[0].repr, Repr::Erased));
    assert_eq!(l.place, Place::Host);
    let kernel = l.modules.kernels.values().next().unwrap();
    assert_eq!(kernel.inputs.len(), 2);
    assert!(
        matches!(&kernel.body[0],Instruction::Assert {condition} if condition.starts_with("capture"))
    );
    let Instruction::Launch {
        shared_bytes,
        arguments,
        ..
    } = &l.body[0]
    else {
        panic!("launch was erased")
    };
    assert_eq!(shared_bytes, "host_layout.bytes");
    assert_eq!(arguments, &["host_grid", "host_flag"]);
    l.modules.functions.insert(
        "entry".into(),
        Function {
            symbol: "entry".into(),
            inputs: vec![],
            outputs: vec![],
            body: l.body,
        },
    );
    for dialect in [GpuDialect::Hip, GpuDialect::Cuda] {
        let source = render_runtime_module(&l.modules, dialect).unwrap().source;
        assert!(source.contains(
            "<<<host_grid.blocks,host_grid.threads,host_layout.bytes>>>(host_grid, host_flag)"
        ));
        assert!(source.contains(dialect.synchronize_fn()));
    }
}
