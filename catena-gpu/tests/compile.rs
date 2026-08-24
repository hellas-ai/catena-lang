use catena_gpu::{
    codegen::{GpuDialect, gpu::render_modules},
    compile::compile,
    stdlib,
};
use metacat::theory::RawTheorySet;

mod cases;

#[test]
fn minimal_stdlib_elaborates_names_and_generates_gpu_modules() {
    let raw = RawTheorySet::from_texts(stdlib::sources().chain([cases::MINIMAL])).unwrap();
    let report = compile(raw).unwrap();

    let elaborated = report.elaborated.as_ref().unwrap().to_hexpr_text();
    assert!(elaborated.contains("name.add"));
    let modules = report.gpu_modules.as_ref().unwrap();
    assert!(modules.values().any(|module| {
        module
            .source_name
            .as_ref()
            .is_some_and(|name| name.as_str() == "add")
    }));

    let generated = render_modules(modules, GpuDialect::Hip).unwrap();
    assert!(generated.contains(
        "extern \"C\" __host__ void program_add(uint64_t x0, uint64_t x1, uint64_t *out_0)"
    ));
    assert!(generated.contains(" = x0 + x1;"));
}
