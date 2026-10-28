//! BF16 has typed memory/kernel values but keeps the public memory ABI.
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
fn shared_scalar_operations_keep_bf16_types() -> anyhow::Result<()> {
    let source = r#"(def program main : (f32 val) -> (f32 val) =
 ([a.] ([.a] f32.to-bf16 [b.]) ([.b b] stdlib.test.numeric.add [c.])
 ([.c b] stdlib.test.numeric.sub [d.]) ([.d b] stdlib.test.numeric.mul [e.])
 ([.e b] stdlib.test.numeric.div [f.])
 ([.f] stdlib.test.numeric.neg [h.])
 ([.h b] bf16x2.pack bf16x2.high bf16.to-f32)))"#;
    for dialect in [GpuDialect::Hip, GpuDialect::Cuda] {
        let module = build(source, dialect)?;
        assert!(module.source.contains("catena_bf16_mul("));
        assert!(module.source.contains("__hmul_rn(a,b)"));
        assert!(module.source.contains("uint16_t"));
        assert!(!module.source.contains("__builtin_amdgcn_fdot2"));
    }
    Ok(())
}
#[test]
fn typed_memory_and_device_dot_support_both_backends() -> anyhow::Result<()> {
    let source = include_str!("runtime/fixtures/bf16.hex");
    let module = build(source, GpuDialect::Hip)?;
    assert!(module.source.contains("__builtin_amdgcn_fdot2_f32_bf16"));
    assert!(module.source.contains("catena_global<catena_bf16x2_t>"));
    assert!(module.source.contains("catena_bf16_load_pair(&"));
    assert!(module.source.contains("catena_global<catena_bf16_t>"));
    assert!(module.source.contains("catena_mem_count<catena_bf16x2_t>"));
    assert!(module.source.contains("#error \"native packed BF16 dot"));
    let cuda = build(source, GpuDialect::Cuda)?;
    assert!(cuda.source.contains("#include <cuda_bf16.h>"));
    assert!(cuda.source.contains("__double2float_rn(__dadd_rn("));
    assert!(cuda.source.contains("catena_global<catena_bf16x2_t>"));
    Ok(())
}
#[test]
fn native_dot_and_transcendentals_cannot_execute_on_host() {
    for body in [
        "([.acc] f32.to-bf16 [b.]) ([.b b] bf16x2.pack [p.]) ([.p p acc] bf16x2.dot-f32)",
        "([.acc] f32.to-bf16 stdlib.test.numeric.exp bf16.to-f32)",
        "([.acc] f32.to-bf16 stdlib.test.numeric.sqrt bf16.to-f32)",
        "([.acc] f32.to-bf16 stdlib.test.numeric.rsqrt bf16.to-f32)",
    ] {
        let source = format!("(def program main : (f32 val) -> (f32 val) = ([acc.] {body}))");
        let error = build(&source, GpuDialect::Hip).unwrap_err().to_string();
        assert!(error.contains("requires device execution"), "{error}");
    }
}
#[test]
fn f32_modules_do_not_include_bf16_support() -> anyhow::Result<()> {
    let module = build(
        "(def program main : (f32 val) -> (f32 val) = ([a.] [.a]))",
        GpuDialect::Hip,
    )?;
    assert!(!module.source.contains("hip_bf16.h"));
    assert!(!module.source.contains("catena_bf16"));
    Ok(())
}

#[test]
fn removed_primitives_are_not_in_the_library() {
    let source = stdlib::core_sources().collect::<Vec<_>>().join("\n");
    for declaration in [
        "(arr u16 :",
        "(arr bf16.add ",
        "(arr bf16.fma ",
        "(arr bf16.select ",
        "(arr bf16.bitcast-u16 ",
        "(arr u16.bitcast-bf16 ",
    ] {
        assert!(
            !source.contains(declaration),
            "unexpected primitive {declaration}"
        );
    }
}
#[test]
fn generic_comparisons_min_max_and_zero_accept_bf16() -> anyhow::Result<()> {
    for operation in ["eq", "ne", "lt", "lte", "gt", "gte"] {
        let source = format!(
            "(def program main : {{(f32 val) (f32 val)}} -> (bool val) = ([a b.] ([.a] f32.to-bf16 [x.]) ([.b] f32.to-bf16 [y.]) ([.x y] stdlib.test.numeric.{operation})))"
        );
        build(&source, GpuDialect::Hip)?;
    }
    for operation in ["min", "max"] {
        let source = format!(
            "(def program main : {{(f32 val) (f32 val)}} -> (f32 val) = ([a b.] ([.a] f32.to-bf16 [x.]) ([.b] f32.to-bf16 [y.]) ([.x y] stdlib.test.numeric.{operation} bf16.to-f32)))"
        );
        build(&source, GpuDialect::Hip)?;
    }
    build(
        "(def program main : [] -> (f32 val) = (stdlib.numeric.numeric.bf16 stdlib.numeric.zero bf16.to-f32))",
        GpuDialect::Hip,
    )?;
    Ok(())
}

/// Exercise the emitted host arithmetic with the real HIP SDK, without a GPU.
#[test]
fn hip_sdk_scalar_rounding_and_special_values() -> anyhow::Result<()> {
    if std::env::var_os("CATENA_BF16_HOST_TESTS").is_none() {
        return Ok(());
    }
    let mut source = String::new();
    for op in ["add", "sub", "mul", "div", "min", "max"] {
        source += &format!(
            "(def program {op} : {{(f32 val) (f32 val)}} -> (f32 val) = ([a b.] ([.a] f32.to-bf16 [x.]) ([.b] f32.to-bf16 [y.]) ([.x y] stdlib.test.numeric.{op} bf16.to-f32)))\n"
        );
    }
    for op in ["eq", "ne", "lt", "lte", "gt", "gte"] {
        source += &format!(
            "(def program {op} : {{(f32 val) (f32 val)}} -> (bool val) = ([a b.] ([.a] f32.to-bf16 [x.]) ([.b] f32.to-bf16 [y.]) ([.x y] stdlib.test.numeric.{op})))\n"
        );
    }
    source += "(def program roundtrip : (f32 val) -> (f32 val) = (f32.to-bf16 bf16.to-f32))";
    let module = build(&source, GpuDialect::Hip)?;
    let symbol = |name| {
        module
            .functions
            .iter()
            .find(|f| f.source_name == name)
            .unwrap()
            .symbol
            .clone()
    };
    let mut cpp = module.source.clone();
    cpp += r#"
#include <cstring>
#include <limits>
#include <cassert>
uint32_t bits(float f) {uint32_t u;std::memcpy(&u,&f,4);return u;}
float word(uint32_t u) {float f;std::memcpy(&f,&u,4);return f;}
float rounded(float f) {uint32_t u=bits(f);if((u&0x7fffffff)>0x7f800000)u=(u&0xffff0000)|0x400000;else u=(u+0x7fff+((u>>16)&1))&0xffff0000;return word(u);}
void same(float a,float b) {assert((std::isnan(a)&&std::isnan(b)) || bits(a)==bits(b));}
int main() {
float values[]={0.0f,-0.0f,1.0f,-1.0f,1.0078125f,word(0x3f808000),word(0x3f818000),word(0x00010000),INFINITY,-INFINITY,NAN};
float result;
"#;
    cpp += &format!(
        "for(float a:values) {{ {}(a,&result);same(result,rounded(a)); }}\n",
        symbol("roundtrip")
    );
    cpp += "for(float a:values) for(float b:values) {float x=rounded(a),y=rounded(b);uint8_t compare;\n";
    for (op, expression) in [
        ("add", "x+y"),
        ("sub", "x-y"),
        ("mul", "x*y"),
        ("div", "x/y"),
        ("min", "x<y?x:y"),
        ("max", "x>y?x:y"),
    ] {
        cpp += &format!(
            "{}(a,b,&result);same(result,rounded({expression}));\n",
            symbol(op)
        );
    }
    for (op, expression) in [
        ("eq", "x==y"),
        ("ne", "x!=y"),
        ("lt", "x<y"),
        ("lte", "x<=y"),
        ("gt", "x>y"),
        ("gte", "x>=y"),
    ] {
        cpp += &format!(
            "{}(a,b,&compare);assert(bool(compare)==({expression}));\n",
            symbol(op)
        );
    }
    cpp += "} }\n";
    let dir = std::env::temp_dir().join(format!("catena-bf16-host-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let result = (|| -> anyhow::Result<()> {
        let file = dir.join("test.cpp");
        let exe = dir.join("test");
        std::fs::write(&file, cpp)?;
        let clang = std::env::var("CATENA_HIP_CLANG")
            .unwrap_or_else(|_| "/opt/rocm/llvm/bin/clang++".into());
        let output = std::process::Command::new(clang)
            .args([
                "-x",
                "hip",
                "--cuda-host-only",
                "--std=c++17",
                "-O2",
                "-ffp-contract=off",
                "-fno-fast-math",
            ])
            .arg(file)
            .arg("-o")
            .arg(&exe)
            .output()?;
        anyhow::ensure!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        anyhow::ensure!(
            std::process::Command::new(exe).status()?.success(),
            "BF16 numerical checks failed"
        );
        Ok(())
    })();
    std::fs::remove_dir_all(dir)?;
    result
}

#[test]
fn bf16_size_evidence_enables_its_runtime_representation() -> anyhow::Result<()> {
    let module = build(
        "(def program main : [] -> (u32 val) = ({unit.intro stdlib.numeric.numeric.bf16} stdlib.numeric.numeric_size :.forget))",
        GpuDialect::Hip,
    )?;
    assert!(module.source.contains("hip/hip_bf16.h"));
    assert!(module.source.contains("sizeof(catena_bf16_t)"));
    Ok(())
}

#[test]
fn wmma_device_fragments_lower_to_native_instruction() -> anyhow::Result<()> {
    let source = include_str!("runtime/fixtures/bf16.hex").replace(
        "([.x1 x1 x6] bf16x2.dot-f32 [x7.])",
        "({[.x1] [.x1] [.x1] [.x1] [.x1] [.x1] [.x1] [.x1]} wmma.bf16x16.pack [frag.]) (unit.intro wmma.f32x8.zero [zero.]) ({[.frag] [.frag] [.zero]} wmma.mma-bf16-f32 [acc.]) ({[.acc] (const.u32.0x00000000)} wmma.f32x8.get [x7.])",
    );
    for dialect in [GpuDialect::Hip, GpuDialect::Cuda] {
        let module = build(&source, dialect)?;
        assert!(
            module
                .source
                .contains("__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32")
        );
        assert!(module.source.contains("catena_wmma_mma("));
    }
    Ok(())
}

#[test]
fn wmma_is_rejected_in_host_code() {
    let source = "(def program main : (f32 val) -> (f32 val) = ([acc.] (unit.intro wmma.f32x8.zero [x.]) ({[.x] (const.u32.0x00000000)} wmma.f32x8.get)))";
    let error = build(source, GpuDialect::Hip).unwrap_err().to_string();
    assert!(error.contains("requires device execution"), "{error}");
}

#[test]
fn wmma_f32_fragment_constructor_lowers_in_device_code() -> anyhow::Result<()> {
    let source = include_str!("runtime/fixtures/bf16.hex").replace(
        "([.x1 x1 x6] bf16x2.dot-f32 [x7.])",
        "({[.x6] [.x6] [.x6] [.x6] [.x6] [.x6] [.x6] [.x6]} wmma.f32x8.pack [frag.]) ({[.frag] (const.u32.0x00000007)} wmma.f32x8.get [x7.])",
    );
    for dialect in [GpuDialect::Hip, GpuDialect::Cuda] {
        let module = build(&source, dialect)?;
        assert!(module.source.contains("catena_wmma_f32x8_t{{"));
    }
    Ok(())
}
