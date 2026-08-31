use super::*;

const READ_WRITE_SYNC: &str = include_str!("shared/read_write_sync.hex");
const TILED_U64: &str = include_str!("../cases/matmul/tiled_u64.hex");

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
        assert!(generated.contains("block_index, catena_shared, shared_layout };"));
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
