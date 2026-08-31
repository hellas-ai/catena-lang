use super::*;

const READ_WRITE_SYNC: &str = include_str!("shared/read_write_sync.hex");

#[test]
fn shared_read_write_and_sync_reach_gpu_codegen() {
    let raw = RawTheorySet::from_texts(stdlib::sources().chain([READ_WRITE_SYNC])).unwrap();
    let report = compile(raw).unwrap();
    let modules = report.gpu_modules.as_ref().unwrap();

    for dialect in [GpuDialect::Hip, GpuDialect::Cuda] {
        let generated = render_modules(modules, dialect).unwrap();
        assert!(generated.contains("catena_block_sync();"));
        assert!(generated.contains("x2[x3.first] = x4;"));
        assert!(generated.contains("= x1[x2.first];"));
    }
}
