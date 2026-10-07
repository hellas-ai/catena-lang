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

#[test]
fn plain_memory_adapters_preserve_effects_and_check_placement() {
    let templates = BTreeMap::new();
    let mut l = lowerer(&templates);
    let own = node("val", vec![node("mem", vec![node("cap.own", vec![])])]);
    let reference = node("val", vec![node("mem", vec![node("cap.ref", vec![])])]);
    let memory = value(own.clone(), "memory");
    let index = value(scalar("u32"), "index");
    let f = value(scalar("f32"), "element");
    assert!(
        l.lower_operation(
            "stdlib.runtime.raw.read_f32",
            vec![memory.clone(), index.clone()],
            &[scalar("f32")]
        )
        .is_err()
    );
    let borrowed = l
        .lower_operation(
            "stdlib.runtime.raw.borrow",
            vec![memory.clone()],
            &[own.clone(), reference.clone()],
        )
        .unwrap();
    assert_eq!(expr(&borrowed[0]).unwrap(), "memory");
    assert_eq!(runtime(&borrowed[1].ty).unwrap(), Some(CType::MemRef));
    l.place = Place::Device;
    assert!(
        l.lower_operation(
            "stdlib.runtime.raw.alloc_f32",
            vec![index.clone()],
            &[own.clone()]
        )
        .is_err()
    );
    assert!(
        l.lower_operation("stdlib.runtime.raw.free", vec![memory.clone()], &[])
            .is_err()
    );
    assert!(
        l.lower_operation(
            "stdlib.runtime.raw.write_f32",
            vec![value(reference, "borrowed"), index.clone(), f.clone()],
            &[]
        )
        .is_err()
    );
    l.lower_operation(
        "stdlib.runtime.raw.write_f32",
        vec![memory.clone(), index, f],
        &[],
    )
    .unwrap();
    assert!(
        l.body
            .iter()
            .any(|i| matches!(i, Instruction::Store { .. }))
    );
    assert!(
        l.body
            .iter()
            .any(|i| matches!(i, Instruction::Assert { .. }))
    );
    l.place = Place::Host;
    l.lower_operation("stdlib.runtime.raw.free", vec![memory], &[])
        .unwrap();
    assert!(l.body.iter().any(
        |i| matches!(i,Instruction::Let(_,expression) if expression.starts_with("catena_free("))
    ));
}

#[test]
fn plain_numeric_adapters_reject_invalid_types_and_emit_checked_narrowing() {
    let templates = BTreeMap::new();
    let mut l = lowerer(&templates);
    assert!(
        l.lower_operation(
            "stdlib.numeric.raw.add",
            vec![value(scalar("u32"), "a"), value(scalar("f32"), "b")],
            &[scalar("f32")]
        )
        .is_err()
    );
    assert!(
        l.lower_operation(
            "stdlib.numeric.raw.select",
            vec![
                value(scalar("u32"), "c"),
                value(scalar("f32"), "a"),
                value(scalar("f32"), "b")
            ],
            &[scalar("f32")]
        )
        .is_err()
    );
    l.lower_operation(
        "stdlib.numeric.raw.to_u32",
        vec![value(scalar("u64"), "wide")],
        &[scalar("u32")],
    )
    .unwrap();
    assert!(
        l.body.iter().any(
            |i| matches!(i,Instruction::Assert{condition} if condition == "wide <= UINT32_MAX")
        )
    );
    for dialect in [GpuDialect::Hip, GpuDialect::Cuda] {
        let source = prelude::render(dialect);
        assert!(source.contains("nearbyintf(x)"));
        assert!(source.contains("mem.len % sizeof(T) == 0"));
        assert!(source.contains(dialect.device_alloc_fn()));
        assert!(source.contains(dialect.device_free_fn()));
    }
}

#[test]
fn reused_kernels_keep_per_launch_arguments_and_shared_memory() {
    let templates = BTreeMap::new();
    let mut l = lowerer(&templates);
    let grid_type = node(
        "val",
        vec![node(
            "stdlib.gpu.geometry.type.Grid",
            vec![node("1", vec![]), node("1", vec![])],
        )],
    );
    let layout_type = node(
        "val",
        vec![node(
            "stdlib.gpu.memory.type.SharedLayout",
            vec![node("list.Nil", vec![])],
        )],
    );
    let callback = Value {
        ty: node("val", vec![node("->", vec![scalar("bool"), proof()])]),
        repr: Repr::Function("stdlib.assert.assert_true".parse().unwrap()),
    };
    let unit = node("1", vec![]);
    for (grid, flag, shared) in [
        ("grid_a", "flag_a", None),
        ("grid_b", "flag_b", Some("layout_b")),
    ] {
        let mut args = vec![value(grid_type.clone(), grid)];
        if let Some(layout) = shared {
            args.push(value(layout_type.clone(), layout));
        }
        args.extend([value(scalar("bool"), flag), callback.clone()]);
        l.lower_operation(
            if shared.is_some() {
                "unsafe.launch_shared"
            } else {
                "unsafe.launch"
            },
            args,
            &[unit.clone()],
        )
        .unwrap();
    }
    assert_eq!(l.modules.kernels.len(), 1);
    let launches = l
        .body
        .iter()
        .filter_map(|instruction| match instruction {
            Instruction::Launch {
                kernel,
                arguments,
                shared_bytes,
                ..
            } => Some((kernel, arguments, shared_bytes)),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(launches.len(), 2);
    assert_eq!(launches[0].0, launches[1].0);
    assert_eq!(launches[0].1, &["grid_a", "flag_a"]);
    assert_eq!(launches[1].1, &["grid_b", "flag_b"]);
    assert_eq!(launches[0].2, "0");
    assert_eq!(launches[1].2, "layout_b.bytes");

    // A distinct Hex callback signature must not reuse a merely ABI-compatible kernel.
    let different = Value {
        ty: node(
            "val",
            vec![node(
                "->",
                vec![
                    scalar("bool"),
                    node("|-", vec![node("different.proposition", vec![])]),
                ],
            )],
        ),
        repr: callback.repr,
    };
    l.lower_operation(
        "unsafe.launch",
        vec![
            value(grid_type, "grid_c"),
            value(scalar("bool"), "flag_c"),
            different,
        ],
        &[unit.clone()],
    )
    .unwrap();
    assert_eq!(l.modules.kernels.len(), 2);
    l.place = Place::Device;
    assert!(l.lower_operation("unsafe.launch", vec![], &[unit]).is_err());
}

#[test]
fn memory_reference_tables_are_checked_and_host_only() {
    let templates = BTreeMap::new();
    let mut l = lowerer(&templates);
    let table_ty = node("val", vec![node("stdlib.runtime.type.MemRefs", vec![])]);
    let ref_ty = node("val", vec![node("mem", vec![node("cap.ref", vec![])])]);
    let own_ty = node("val", vec![node("mem", vec![node("cap.own", vec![])])]);
    assert!(runtime(&table_ty).unwrap().unwrap().abi().is_none());
    let empty = l
        .lower_operation(
            "stdlib.runtime.raw.mem_refs_empty",
            vec![],
            &[table_ty.clone()],
        )
        .unwrap()
        .remove(0);
    let table = l
        .lower_operation(
            "stdlib.runtime.raw.mem_refs_push",
            vec![empty, value(ref_ty.clone(), "buffer")],
            &[table_ty.clone()],
        )
        .unwrap()
        .remove(0);
    l.lower_operation(
        "stdlib.runtime.raw.mem_ref_at",
        vec![table.clone(), value(scalar("u32"), "index")],
        &[ref_ty.clone()],
    )
    .unwrap();
    assert!(l.body.iter().any(|i| matches!(i, Instruction::Assert {condition} if condition.contains("index <") && condition.contains(".size()"))));
    assert!(
        l.lower_operation(
            "stdlib.runtime.raw.mem_refs_push",
            vec![table.clone(), value(own_ty, "owned")],
            &[table_ty.clone()]
        )
        .is_err()
    );
    assert!(
        l.lower_operation(
            "stdlib.runtime.raw.mem_ref_at",
            vec![table.clone(), value(scalar("u64"), "index")],
            &[ref_ty.clone()]
        )
        .is_err()
    );
    assert!(
        l.lower_operation(
            "unsafe.launch_linear",
            vec![
                value(scalar("u32"), "count"),
                table.clone(),
                Value {
                    ty: node(
                        "val",
                        vec![node("->", vec![scalar("u32"), node("1", vec![])])]
                    ),
                    repr: Repr::Function("test".parse().unwrap())
                }
            ],
            &[node("1", vec![])]
        )
        .is_err()
    );
    l.place = Place::Device;
    assert!(
        l.lower_operation(
            "stdlib.runtime.raw.mem_ref_at",
            vec![table, value(scalar("u32"), "index")],
            &[ref_ty]
        )
        .is_err()
    );
    assert!(
        l.lower_operation("stdlib.runtime.raw.mem_refs_empty", vec![], &[table_ty])
            .is_err()
    );
}
