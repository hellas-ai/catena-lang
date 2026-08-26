use super::*;

const ADD: &str = include_str!("basic/add.hex");

#[test]
fn minimal_stdlib_elaborates_names_and_generates_gpu_modules() {
    let raw = RawTheorySet::from_texts(stdlib::sources().chain([ADD])).unwrap();
    let report = compile(raw).unwrap();

    let elaborated = report.elaborated.as_ref().unwrap().to_hexpr_text();
    assert!(elaborated.contains("name.add"));
    assert!(elaborated.contains("name.gpu.global.write"));
    assert!(elaborated.contains("name.gpu.launch"));
    let modules = report.gpu_modules.as_ref().unwrap();
    assert!(modules.values().any(|module| {
        module
            .source_name
            .as_ref()
            .is_some_and(|name| name.as_str() == "add")
    }));

    let generated = render_modules(modules, GpuDialect::Hip).unwrap();
    assert!(generated.contains(
        "extern \"C\" __host__ __device__ void program_add(uint64_t x0, uint64_t x1, uint64_t *out_0)"
    ));
    assert!(generated.contains(" = x0 + x1;"));
    assert!(generated.contains("catena_grid_t"));
    assert!(generated.contains("template<typename T"));
    assert!(generated.contains("void program_gpu_global_matrix_2d_row_major_read("));
    assert!(generated.contains("void program_gpu_global_matrix_2d_row_major_write("));
    let kernel = generated
        .find("void program_gpu_global_matrix_matmul_kernel(")
        .unwrap();
    assert!(generated[kernel.saturating_sub(100)..kernel].contains("template<"));
}
