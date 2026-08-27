use super::*;

const FILL_WITH_READABLE_SUM: &str = include_str!("../cases/launch/fill_with_readable_sum.hex");
const WRITE_BORROWED: &str = include_str!("references/write_borrowed.hex");

#[test]
fn borrowed_source_reads_need_no_schedule_or_grant() {
    let raw = RawTheorySet::from_texts(stdlib::sources().chain([FILL_WITH_READABLE_SUM])).unwrap();
    let report = compile(raw).unwrap();
    let generated = render_modules(report.gpu_modules.as_ref().unwrap(), GpuDialect::Hip).unwrap();

    assert!(generated.contains("catena_mem_ref_t"));
    assert!(generated.contains("const uint64_t *"));
    assert!(!generated.contains("CATENA_SCHEDULING_READ_ALL"));
}

#[test]
fn borrowed_global_memory_cannot_be_written() {
    let raw = RawTheorySet::from_texts(stdlib::sources().chain([WRITE_BORROWED])).unwrap();
    let failure = compile(raw).expect_err("gpu.global.write must reject cap.ref");

    let CompileError::Check(CheckError::Definition { definition, .. }) = &failure.cause else {
        panic!("expected a definition type error, got: {}", failure.cause);
    };
    assert_eq!(definition, "write-borrowed");
}
