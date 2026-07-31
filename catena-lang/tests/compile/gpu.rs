use catena_lang::codegen::{
    GpuDialect,
    gpu::{render_module, render_modules},
};

const TILED_MATMUL_SOURCE: &str = include_str!("../cases/gpu/tiled_matmul.hex");
const NAIVE_MATMUL_SOURCE: &str = include_str!("../cases/matrix/matmul.hex");
const ORDINARY_MATERIALIZE_SOURCE: &str = include_str!("../cases/materialize/basic.hex");
const NESTED_GPU_MATERIALIZE_SOURCE: &str = r#"
(def program gpu-kernel-with-nested-gpu-materialize :
  ([n.] {
    ({[.n] u64} :)
    ({
      ({gpu.initial gpu.initial} gpu.state)
      ([.n] ix val)
    } *)
  })
  ->
  ([n.] (u64 val))
= ([len state-and-index.]
  ([.state-and-index] *.elim [state index.])
  ([.state] gpu.state.admit-discard)
  ([.len] gpu-materialize-with-ordinary-materialize [nested-output.])
  ([.nested-output] free)
  ([.index] ix.to-u64)
))

(def program invalid-gpu-materialize-nesting :
  ([grid n.] {
    ([.grid] gpu.launch_params val)
    ({[.n] u64} :)
  })
  ->
  ([grid n.] ({cap.own [.n] u64} buf val))
= ([params len.]
  ([.len] u64.copies2-param [
    materialize-len kernel-len n-param.
  ])
  ({
    [.params]
    [.materialize-len]
    ({
      [.n-param]
      [.kernel-len]
    } partial.gpu-kernel-with-nested-gpu-materialize.1)
  } gpu.materialize)
))
"#;

#[test]
fn tiled_matmul_converts_and_renders_cooperative_loads() -> anyhow::Result<()> {
    let report = super::compile_with_sources([TILED_MATMUL_SOURCE])?;
    let modules = report
        .gpu_modules
        .as_ref()
        .expect("successful compilation should produce GPU modules");

    assert!(modules.values().any(|module| {
        module
            .entry
            .assignments
            .iter()
            .any(|assignment| assignment.op.as_str() == "gpu.shared.row-major.cooperative-loadc")
    }));

    let source = render_modules(modules, GpuDialect::Hip)?;
    assert!(source.contains("(uint64_t)threadIdx.y"));
    assert!(source.contains("(uint64_t)threadIdx.x"));
    assert!(source.contains("__syncthreads();"));
    Ok(())
}

#[test]
fn ordinary_naive_matmul_uses_sequential_materialize_without_a_kernel() -> anyhow::Result<()> {
    let report = super::compile_with_sources([NAIVE_MATMUL_SOURCE])?;
    let modules = report
        .gpu_modules
        .as_ref()
        .expect("successful compilation should produce GPU modules");
    let ordinary = modules
        .values()
        .find(|module| {
            module
                .source_name
                .as_ref()
                .is_some_and(|name| name.as_str() == "matmul-2x2-via-mem")
        })
        .expect("ordinary naive matmul should have a generated module");

    assert!(
        ordinary
            .entry
            .assignments
            .iter()
            .any(|assignment| assignment.op.as_str() == "materializec")
    );
    assert!(
        ordinary
            .entry
            .assignments
            .iter()
            .all(|assignment| assignment.op.as_str() != "gpu.materialize")
    );

    let source = render_module(ordinary, GpuDialect::Cuda)?;
    assert!(source.contains("for (uint64_t"));
    assert!(!source.contains("__global__"));
    assert!(!source.contains("<<<"));
    assert!(!source.contains("DeviceSynchronize"));
    Ok(())
}

#[test]
fn gpu_naive_matmul_uses_gpu_materialize_and_a_kernel() -> anyhow::Result<()> {
    let report = super::compile_with_sources([NAIVE_MATMUL_SOURCE])?;
    let modules = report
        .gpu_modules
        .as_ref()
        .expect("successful compilation should produce GPU modules");

    assert!(modules.values().any(|module| {
        module
            .entry
            .assignments
            .iter()
            .any(|assignment| assignment.op.as_str() == "gpu.materialize")
    }));

    let source = render_modules(modules, GpuDialect::Cuda)?;
    assert!(source.contains("__global__ void materialize_program_gpu_naive_matmul_via_mem"));
    assert!(source.contains("materialize_program_gpu_naive_matmul_via_mem"));
    assert!(source.contains("<<<"));
    assert!(source.contains("cudaDeviceSynchronize"));
    Ok(())
}

#[test]
fn ordinary_materialize_producers_may_materialize() -> anyhow::Result<()> {
    let report = super::compile_with_sources([ORDINARY_MATERIALIZE_SOURCE])?;
    let modules = report
        .gpu_modules
        .as_ref()
        .expect("successful compilation should produce GPU modules");
    let ordinary = modules
        .values()
        .find(|module| {
            module
                .source_name
                .as_ref()
                .is_some_and(|name| name.as_str() == "nested-materialize-indexes-source")
        })
        .expect("nested ordinary materialize should have a generated module");
    let producer = modules
        .values()
        .find(|module| {
            module
                .source_name
                .as_ref()
                .is_some_and(|name| name.as_str() == "nested-materialize-index")
        })
        .expect("nested materialize producer should have a generated module");
    let source = format!(
        "{}\n{}",
        render_module(ordinary, GpuDialect::Cuda)?,
        render_module(producer, GpuDialect::Cuda)?
    );

    assert!(source.matches("for (uint64_t").count() >= 2);
    assert!(!source.contains("__global__"));
    Ok(())
}

#[test]
fn gpu_materialize_kernels_may_use_ordinary_materialize() -> anyhow::Result<()> {
    let report = super::compile_with_sources([ORDINARY_MATERIALIZE_SOURCE])?;
    let modules = report
        .gpu_modules
        .as_ref()
        .expect("successful compilation should produce GPU modules");
    let source = render_modules(modules, GpuDialect::Cuda)?;

    assert!(
        source.contains(
            "__global__ void materialize_program_gpu_materialize_with_ordinary_materialize"
        )
    );
    assert!(source.contains("#ifdef __CUDA_ARCH__"));
    assert!(source.contains("malloc("));
    Ok(())
}

#[test]
fn gpu_materialize_rejects_nested_gpu_materialize() {
    let error =
        super::compile_with_sources([ORDINARY_MATERIALIZE_SOURCE, NESTED_GPU_MATERIALIZE_SOURCE])
            .expect_err("gpu.materialize kernel nesting should be rejected");
    let message = format!("{error:#}");

    assert!(message.contains("gpu.materialize kernel"), "{message}");
    assert!(message.contains("another gpu.materialize"), "{message}");
    assert!(message.contains("Nested GPU kernel launches"), "{message}");
}
