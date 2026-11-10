//! End-to-end declaration, elaboration, closure conversion and GPU lowering.
use catena_lang::{
    codegen::CodegenKind,
    compile::compile,
    report::CompileReport,
    runtime::{GpuDialect, RuntimeModule},
    stdlib,
};
use metacat::theory::RawTheorySet;

fn build(source: &str, dialect: GpuDialect) -> anyhow::Result<RuntimeModule> {
    let mut sources = stdlib::core_sources().collect::<Vec<_>>();
    sources.push(include_str!("runtime/fixtures/bf16_numeric.hex"));
    sources.push(source);
    let raw = RawTheorySet::from_texts(sources)?;
    Ok(compile(
        &mut CompileReport::new(raw),
        CodegenKind::Default,
        dialect,
    )?)
}

#[test]
fn declarations_lower_through_a_real_kernel_launch() -> anyhow::Result<()> {
    let source = include_str!("runtime/fixtures/bf16.hex").replace(
        "([.x1 x1 x6] bf16x2.dot-f32 [x7.])",
        r#"([.x1 x1 x1 x1] mma.m16n8k16.bf16.a.pack [a.])
        ([.x1 x1] mma.m16n8k16.bf16.b.pack [b.])
        ([.x6 x6 x6 x6] mma.m16n8k16.f32.acc.pack [c.])
        ([.a b c] mma.m16n8k16.bf16-f32 [d.])
        ([.] unit.intro mma.m16n8k16.f32.acc.zero [z.])
        ([.a b z] mma.m16n8k16.bf16-f32 [e.])
        ([.] const.u32.0x00000003 [i.])
        ([.d i] mma.m16n8k16.f32.acc.get [x7.])
        ([.e i] mma.m16n8k16.f32.acc.get [w.])"#,
    );
    let cuda = build(&source, GpuDialect::Cuda)?;
    assert!(cuda.source.contains("catena_mma_a_pack("));
    assert!(cuda.source.contains("catena_mma_b_pack("));
    assert!(
        cuda.source
            .contains("catena_mma_f32_acc_t{{0.0f,0.0f,0.0f,0.0f}}")
    );
    assert_eq!(cuda.source.matches("= catena_mma_bf16_f32(").count(), 2);
    assert!(cuda.source.contains("catena_mma_acc_get("));
    assert!(!cuda.source.contains("CatenaWmmaEmulate"));
    let error = build(&source, GpuDialect::Hip).unwrap_err().to_string();
    assert!(error.contains("NVIDIA MMA requires CUDA"), "{error}");
    Ok(())
}

#[test]
fn host_mma_is_rejected_and_unused_declarations_emit_no_support() -> anyhow::Result<()> {
    for dialect in [GpuDialect::Cuda, GpuDialect::Hip] {
        let error = build(
            "(def program main : [] -> 1 = (unit.intro mma.m16n8k16.f32.acc.zero [z.] unit.intro))",
            dialect,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("requires device execution"), "{error}");
        let module = build(
            "(def program main : (f32 val) -> (f32 val) = ([a.] [.a]))",
            dialect,
        )?;
        assert!(!module.source.contains("catena_mma"));
    }
    Ok(())
}
