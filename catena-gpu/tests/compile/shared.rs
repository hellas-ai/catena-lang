use super::*;

const READ_WRITE_SYNC: &str = include_str!("shared/read_write_sync.hex");
const TILED_U64: &str = include_str!("../cases/matmul/tiled_u64.hex");
const WRITE_WITH_OWNERSHIP_FOR_ANOTHER_CELL: &str =
    include_str!("shared/write_rejects_ownership_for_another_cell.hex");
const WRITE_WITH_ASSIGNMENT_FOR_ANOTHER_CELL: &str =
    include_str!("shared/write_rejects_assignment_for_another_cell.hex");
const READ_WITH_OWNERSHIP_INSTEAD_OF_READ_PERMISSION: &str =
    include_str!("shared/read_rejects_ownership_instead_of_read_permission.hex");
const TAKE_WITH_WRONG_NAMED_SLOT: &str = include_str!("shared/take_rejects_wrong_named_slot.hex");

#[test]
fn shared_write_with_ownership_for_another_cell_is_rejected() {
    assert_shared_definition_is_rejected(
        WRITE_WITH_OWNERSHIP_FOR_ANOTHER_CELL,
        "shared-write-with-ownership-for-another-cell",
    );
}

#[test]
fn shared_write_with_assignment_for_another_cell_is_rejected() {
    assert_shared_definition_is_rejected(
        WRITE_WITH_ASSIGNMENT_FOR_ANOTHER_CELL,
        "shared-write-with-assignment-for-another-cell",
    );
}

#[test]
fn shared_read_with_ownership_instead_of_read_permission_is_rejected() {
    assert_shared_definition_is_rejected(
        READ_WITH_OWNERSHIP_INSTEAD_OF_READ_PERMISSION,
        "shared-read-with-ownership-instead-of-read-permission",
    );
}

#[test]
fn taking_a_shared_slot_under_a_different_name_is_rejected() {
    assert_shared_definition_is_rejected(
        TAKE_WITH_WRONG_NAMED_SLOT,
        "taking-tile-a-as-tile-b-is-rejected",
    );
}

#[test]
fn shared_memory_operations_and_barrier_sync_reach_gpu_codegen() {
    let raw = RawTheorySet::from_texts(stdlib::sources().chain([READ_WRITE_SYNC])).unwrap();
    let report = compile(raw).unwrap();
    let modules = report.gpu_modules.as_ref().unwrap();

    for dialect in [GpuDialect::Hip, GpuDialect::Cuda] {
        let generated = render_modules(modules, dialect).unwrap();
        assert!(generated.contains("catena_block_sync();"));
        assert!(generated.contains("x2[x3.first] = x4;"));
        assert!(generated.contains("= x2[x3.first];"));
        assert!(generated.contains("x0 * sizeof(float);"));
    }
}

#[test]
fn shared_layout_is_passed_to_tiled_kernel_launch() {
    let raw = RawTheorySet::from_texts(stdlib::sources().chain([TILED_U64])).unwrap();
    let report = compile(raw).unwrap();
    let modules = report.gpu_modules.as_ref().unwrap();

    for dialect in [GpuDialect::Hip, GpuDialect::Cuda] {
        let generated = render_modules(modules, dialect).unwrap();
        assert!(generated.contains("extern __shared__ unsigned char catena_shared[];"));
        assert!(generated.contains("uint64_t shared_layout"));
        assert!(generated.contains("block_index, block_dim, catena_shared, shared_layout };"));
        assert!(generated.contains("(shared_layout, kernel_argument_0"));
        assert!(generated.matches("* sizeof(uint64_t)").count() >= 2);
        let lines = generated.lines().collect::<Vec<_>>();
        let launch_index = lines
            .iter()
            .position(|line| line.contains("<<<dim3("))
            .expect("tiled fixture should emit a kernel launch");
        let launch = lines[launch_index];
        assert_eq!(launch.matches("), ").count(), 2);
        let shared_layout = launch
            .rsplit_once(", ")
            .and_then(|(_, suffix)| suffix.strip_suffix(">>>("))
            .expect("launch should use the shared layout as its third configuration value");
        assert!(
            lines[launch_index + 1]
                .trim_start()
                .starts_with(shared_layout)
        );
    }
}

#[test]
fn tiled_matmul_codegen_places_two_barriers_inside_each_traced_iteration() {
    let raw = RawTheorySet::from_texts(stdlib::sources().chain([TILED_U64])).unwrap();
    let report = compile(raw).unwrap();

    for dialect in [GpuDialect::Hip, GpuDialect::Cuda] {
        let generated = render_modules(report.gpu_modules.as_ref().unwrap(), dialect).unwrap();
        assert_eq!(generated.matches("catena_block_sync();").count(), 2);
        assert!(generated.contains("for (uint64_t fold_index_"));
        assert!(generated.contains(".shared +"));
        assert!(generated.contains(".in_block_index.first"));
    }
}

#[test]
fn tiled_matmul_takes_each_named_slot_without_assuming_equal_layout_halves() {
    let raw = RawTheorySet::from_texts(stdlib::sources().chain([TILED_U64])).unwrap();
    let report = compile(raw).unwrap();

    for dialect in [GpuDialect::Hip, GpuDialect::Cuda] {
        let generated = render_modules(report.gpu_modules.as_ref().unwrap(), dialect).unwrap();
        assert_eq!(generated.matches(".shared_layout -").count(), 2);
        assert!(!generated.contains("shared_layout / 2"));
    }
}

#[test]
fn tiled_matmul_asserts_shared_cell_ownership_before_entering_the_barrier_fold() {
    let raw = RawTheorySet::from_texts(stdlib::sources().chain([TILED_U64])).unwrap();
    let report = compile(raw).unwrap();

    for dialect in [GpuDialect::Hip, GpuDialect::Cuda] {
        let generated = render_modules(report.gpu_modules.as_ref().unwrap(), dialect).unwrap();
        assert!(generated.contains("{ CATENA_SCHEDULING_SHARED_OWN_EACH, 0,"));
        assert!(generated.contains("cell.first < scheduling.size && local == cell.first"));
        let ownership_decision = generated
            .find(" = (catena_scheduling_resolve(")
            .expect("shared scheduling should resolve ownership at runtime");
        let ownership_assertion = generated
            .find("if (!")
            .expect("shared ownership should be asserted");
        let fold = generated[ownership_decision..]
            .find("for (uint64_t fold_index_")
            .map(|offset| ownership_decision + offset)
            .expect("tiled kernel should contain its barrier fold");
        assert!(ownership_decision < fold);
        assert!(ownership_assertion < fold);
    }
}

fn assert_shared_definition_is_rejected(source: &'static str, expected_definition: &str) {
    let raw = RawTheorySet::from_texts(stdlib::sources().chain([source])).unwrap();
    let failure = compile(raw).expect_err("unguarded shared-memory access must not compile");

    let CompileError::Check(CheckError::Definition { definition, .. }) = &failure.cause else {
        panic!("expected a definition type error, got: {}", failure.cause);
    };
    assert_eq!(definition, expected_definition);
}
