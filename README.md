# Catena: portable deterministic array programming

Catena is a **deterministic array programming language**:

- **Deterministic**: Programs produce bitwise identical results on all platforms
- **Secure**: It is safe to execute programs from third parties

<h3> ⚠️ NOTE: Catena is alpha quality software⚠️</h3>

If you find that these promises don't hold,
[open an issue!](https://github.com/hellas-ai/catena-lang/issues)

Catena is a key technical component enabling the
[Hellas Network](https://hellas.ai/), a
decentralised platform for trustless AI compute:

- **Determinism** enables **verifiability**: users can check code was faithfully executed
- **Secure** means that **arbitrary user programs** can be run by compute providers

# Usage

    cargo install catena-lang

Catena is intended to be called as a **library**.
here's how to run a program that returns a `u64` value:

```rust
use catena_lang::{
    codegen::CodegenKind,
    compile::compile,
    report::CompileReport,
    runtime::{GpuDialect, Runtime, Value},
    stdlib,
};
use metacat::theory::RawTheorySet;

fn main() -> anyhow::Result<()> {
    // programs in hexpr notation: https://github.com/hellas-ai/hexpr
    let source = r#"
        (def program identity : (u64 val) -> (u64 val) = [value])
    "#;

    let mut report = CompileReport::new(RawTheorySet::from_texts(
        stdlib::sources().chain([source]),
    )?);
    let module = compile(&mut report, CodegenKind::Default, GpuDialect::Hip)?;
    let mut runtime = Runtime::new(GpuDialect::Hip)?;
    let artifact = runtime.load(module)?;
    let [result] = artifact.exec("identity", [4_u64.into()])?;
    let Value::U64(result) = result else {
        anyhow::bail!("identity returned non-u64 value: {result:?}");
    };

    println!("Result: {result}");
    assert_eq!(result, 4);
    Ok(())
}
```

The default code generator and its standard library are used by default.
See [CLI usage](docs/cli.md) and [BF16 support and precision rules](BF16.md).

GPU runtime tests require an installed GPU toolkit and a compatible GPU:

```sh
cargo test --all-features --test runtime
CATENA_GPU_DIALECT=cuda cargo test --all-features --test runtime
```

The tests default to HIP; `CATENA_GPU_DIALECT` selects `hip` or `cuda`.
The HIP runtime loader checks the system library path, then `ROCM_PATH` and
`HIP_PATH` (both `lib` and `lib64`), then `/opt/rocm/lib` and `/opt/rocm/lib64`.
For a nonstandard ROCm installation, set `ROCM_PATH` to its toolkit root.
