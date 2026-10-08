use super::lower_types::CType;
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
                (vec![Tree::Leaf(1, ()), ty.clone()], vec![])
            } else {
                (vec![Tree::Leaf(1, ())], vec![ty.clone()])
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
fn generic_witness_helpers_remain_available_without_runtime_exports() {
    use crate::pass::record_boundary_sizes::OperationWithBoundarySizes;
    use open_hypergraphs::lax::OpenHypergraph;

    let term = OpenHypergraph::singleton(
        OperationWithBoundarySizes {
            operation: "meta.test".parse().unwrap(),
            source_sizes: vec![1],
            target_sizes: vec![1],
        },
        vec![Tree::Leaf(0, ())],
        vec![proof()],
    );
    let terms = BTreeMap::from([(
        TheoryId("program".parse().unwrap()),
        BTreeMap::from([("test.helper".parse().unwrap(), term)]),
    )]);
    let modules = lower_to_ir(&terms).unwrap();
    assert!(modules.exports.is_empty());
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
        "stdlib.runtime.global_own_u64",
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
            value(scalar("u64"), "index"),
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
fn unsafe_global_read_loads_on_device_without_thread_or_permissions() {
    for kind in ["u32", "u64", "f32"] {
        let templates = BTreeMap::new();
        let mut l = lowerer(&templates);
        l.place = Place::Device;
        let buffer = node(
            "val",
            vec![node(
                "stdlib.gpu.memory.type.Global",
                vec![Tree::Leaf(0, ()), node(kind, vec![])],
            )],
        );
        let index = node(
            "val",
            vec![node("stdlib.gpu.memory.type.Ix", vec![Tree::Leaf(0, ())])],
        );
        let result = l
            .lower_operation(
                "stdlib.gpu.memory.global_read_unsafe",
                vec![
                    value(buffer.clone(), "buffer"),
                    value(index.clone(), "index"),
                ],
                &[scalar(kind)],
            )
            .unwrap();
        assert_eq!(result.len(), 1);
        assert!(
            matches!(&l.body[0], Instruction::Assert { condition } if condition == "index < buffer.count")
        );
        assert!(
            matches!(&l.body[1], Instruction::Load { result: loaded, buffer, index }
            if loaded.name == expr(&result[0]).unwrap()
                && buffer.ty == CType::Global(Box::new(runtime(&scalar(kind)).unwrap().unwrap()))
                && index.ty == CType::U64)
        );
        l.modules.kernels.insert(
            "read".into(),
            Function {
                symbol: "read".into(),
                inputs: vec![
                    Variable {
                        name: "buffer".into(),
                        ty: runtime(&buffer).unwrap().unwrap(),
                    },
                    Variable {
                        name: "index".into(),
                        ty: runtime(&index).unwrap().unwrap(),
                    },
                ],
                outputs: vec![],
                body: l.body,
            },
        );
        for dialect in [GpuDialect::Hip, GpuDialect::Cuda] {
            let source = render_runtime_module(&l.modules, dialect).unwrap().source;
            assert!(source.contains("catena_assert(index < buffer.count);"));
            assert!(source.contains("= buffer.data[index];"));
        }
    }
}

#[test]
fn unsafe_global_read_rejects_host_execution_and_missing_operands() {
    let templates = BTreeMap::new();
    let mut l = lowerer(&templates);
    let buffer = node(
        "val",
        vec![node(
            "stdlib.gpu.memory.type.Global",
            vec![Tree::Leaf(0, ()), node("u64", vec![])],
        )],
    );
    let args = vec![value(buffer, "buffer"), value(scalar("u64"), "index")];
    assert!(matches!(l.lower_operation(
        "stdlib.gpu.memory.global_read_unsafe", args.clone(), &[scalar("u64")],
    ), Err(CodegenError::Invalid { reason, .. }) if reason == "requires device execution"));
    assert!(l.body.is_empty());
    l.place = Place::Device;
    assert!(matches!(l.lower_operation(
        "stdlib.gpu.memory.global_read_unsafe", args[..1].to_vec(), &[scalar("u64")],
    ), Err(CodegenError::Invalid { reason, .. }) if reason == "invalid read operands"));
    assert!(l.body.is_empty());
}

#[test]
fn fold_emits_loop_and_preserves_carried_state() {
    for (op, kind) in [
        ("core.fold.bounded", "u32"),
        ("core.fold.trace", "u32"),
        ("core.fold.bounded_u64", "u64"),
        ("core.fold.trace_u64", "u64"),
    ] {
        let templates = BTreeMap::new();
        let mut l = lowerer(&templates);
        let ty = scalar(kind);
        let domain = node("*", vec![ty.clone(), ty.clone()]);
        let callback = Value {
            ty: node("val", vec![node("->", vec![domain, ty.clone()])]),
            repr: Repr::Function("stdlib.numeric.+".parse().unwrap()),
        };
        let result = l
            .lower_operation(
                op,
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
        let Instruction::For {
            body, index_type, ..
        } = l.body.last().unwrap()
        else {
            panic!("expected loop")
        };
        assert!(
            body.iter()
                .any(|i| matches!(i,Instruction::Let(_,s) if s.contains(" + ")))
        );
        assert!(matches!(body.last(), Some(Instruction::Assign(..))));
        assert_eq!(*index_type, runtime(&scalar(kind)).unwrap().unwrap());
    }
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

#[test]
fn new_numeric_primitives_lower_with_erased_evidence() {
    let templates = BTreeMap::new();
    let mut l = lowerer(&templates);
    for op in ["exp", "sqrt", "rsqrt"] {
        let result = l
            .lower_operation(
                &format!("stdlib.numeric.{op}"),
                vec![value(scalar("f32"), "x"), Value::erased(proof())],
                &[scalar("f32")],
            )
            .unwrap();
        assert_eq!(result.len(), 1);
    }
    l.lower_operation("stdlib.numeric.floating.f32", vec![], &[proof()])
        .unwrap();
    l.lower_operation(
        "stdlib.numeric.u32_to_f32",
        vec![value(scalar("u32"), "x")],
        &[scalar("f32")],
    )
    .unwrap();
    l.lower_operation(
        "stdlib.numeric.ceil_div_u64",
        vec![
            value(scalar("u64"), "n"),
            value(scalar("u64"), "d"),
            Value::erased(proof()),
        ],
        &[scalar("u64"), proof()],
    )
    .unwrap();
    assert!(
        l.body
            .iter()
            .any(|i| matches!(i, Instruction::Assert { condition } if condition == "d != 0"))
    );
}

#[test]
fn math_primitives_lower_and_reject_invalid_interfaces() {
    let templates = BTreeMap::new();
    for (op, inputs, output, expected) in [
        ("f32_bits", vec!["f32"], "u32", "catena_f32_bitcast_u32(x0)"),
        (
            "from_bits",
            vec!["u32"],
            "f32",
            "catena_u32_bitcast_f32(x0)",
        ),
        (
            "round_to_u32",
            vec!["f32"],
            "u32",
            "catena_round_to_u32(x0)",
        ),
        ("shift_right", vec!["u32", "u32"], "u32", "(x0 >> x1)"),
    ] {
        let name = format!("stdlib.math.{op}");
        let args = inputs
            .iter()
            .enumerate()
            .map(|(i, ty)| value(scalar(ty), &format!("x{i}")))
            .collect::<Vec<_>>();
        let mut l = lowerer(&templates);
        l.lower_operation(&name, args.clone(), &[scalar(output)])
            .unwrap();
        assert!(l.body.iter().any(|instruction| matches!(instruction,
            Instruction::Let(_, expression) if expression == expected)));
        if op == "shift_right" {
            assert!(l.body.iter().any(|instruction| matches!(instruction,
                Instruction::Assert { condition } if condition == "x1 < 32")));
        }
        assert!(l.lower_operation(&name, vec![], &[scalar(output)]).is_err());
        assert!(
            l.lower_operation(&name, args.clone(), &[scalar("bool")])
                .is_err()
        );
        let mut wrong = args;
        wrong[0] = value(scalar("u64"), "wrong");
        assert!(l.lower_operation(&name, wrong, &[scalar(output)]).is_err());
    }
    for dialect in [GpuDialect::Hip, GpuDialect::Cuda] {
        let source = prelude::render(dialect);
        assert!(source.contains("__host__ __device__ inline uint32_t catena_round_to_u32"));
        assert!(source.contains("__host__ __device__ inline uint32_t catena_f32_bitcast_u32"));
    }
}

#[test]
fn kernel_allocation_is_rejected() {
    let templates = BTreeMap::new();
    let mut l = lowerer(&templates);
    let bytes = scalar("u64");
    let mem = node("val", vec![node("mem", vec![node("cap.own", vec![])])]);
    let kernel = Value {
        ty: node("val", vec![node("->", vec![bytes.clone(), mem])]),
        repr: Repr::Function("stdlib.gpu.memory.allocate".parse().unwrap()),
    };
    let grid = node(
        "val",
        vec![node(
            "stdlib.gpu.geometry.type.Grid",
            vec![Tree::Leaf(0, ()), Tree::Leaf(1, ())],
        )],
    );
    let error = l
        .lower_operation(
            "unsafe.launch",
            vec![value(grid, "grid"), value(bytes, "bytes"), kernel],
            &[node("1", vec![])],
        )
        .unwrap_err();
    assert!(matches!(error, CodegenError::Invalid { op, reason }
        if op == "stdlib.gpu.memory.allocate" && reason == "allocation requires host execution"));
    assert!(l.modules.kernels.is_empty());
    assert!(!l.body.iter().any(|instruction| matches!(instruction,
        Instruction::Let(_, expression) if expression.contains("catena_allocate"))));
}

#[test]
fn memory_conversions_allocation_and_free_use_current_namespace() {
    let templates = BTreeMap::new();
    let mut l = lowerer(&templates);
    let mem = node("val", vec![node("mem", vec![node("cap.own", vec![])])]);
    let global = node(
        "val",
        vec![node(
            "stdlib.gpu.memory.type.Global",
            vec![Tree::Leaf(0, ()), node("u64", vec![])],
        )],
    );
    l.lower_operation(
        "stdlib.gpu.memory.allocate",
        vec![value(scalar("u64"), "bytes"), Value::erased(proof())],
        &[mem.clone()],
    )
    .unwrap();
    l.lower_operation(
        "stdlib.gpu.memory.global_own_u64",
        vec![value(mem.clone(), "memory"), value(scalar("u64"), "count")],
        &[global.clone(), proof()],
    )
    .unwrap();
    let outputs = [node(
        "*",
        vec![scalar("u64"), node("*", vec![global.clone(), proof()])],
    )];
    let cast = l
        .lower_operation(
            "stdlib.gpu.memory.mem_cast_own_u64",
            vec![value(mem.clone(), "memory")],
            &outputs,
        )
        .unwrap();
    assert_eq!(ops::runtime_args(&cast).len(), 2);
    l.lower_operation(
        "stdlib.gpu.memory.global_cast_equal_own",
        vec![
            value(global.clone(), "buffer"),
            value(scalar("u64"), "count"),
        ],
        &[global.clone(), proof()],
    )
    .unwrap();
    l.lower_operation(
        "stdlib.gpu.memory.global_u64_to_mem",
        vec![value(global.clone(), "buffer")],
        &[mem],
    )
    .unwrap();
    l.lower_operation(
        "stdlib.gpu.memory.free",
        vec![value(global, "buffer"), Value::erased(proof())],
        &[node("1", vec![])],
    )
    .unwrap();
    assert!(matches!(l.body.last(), Some(Instruction::Free { .. })));
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
        assert!(source.contains(dialect.device_alloc_fn()));
        assert!(source.contains(&format!("{}(buffer.data)", dialect.device_free_fn())));
        assert!(source.contains("uint64_t count"));
    }
}
