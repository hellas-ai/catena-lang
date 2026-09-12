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
here's how to run a program that adds two `u64` values:

```rust
use catena_lang::{
    codegen::GpuDialect,
    runtime::{Runtime, Value},
    stdlib,
};

fn main() -> anyhow::Result<()> {
    // programs in hexpr notation: https://github.com/hellas-ai/hexpr
    let source = r#"
        (def program two-plus-two : [] -> (u64 val) = (
          ({u64.one u64.one} u64.add)
          {[two . two two]}
          u64.add
        ))
    "#;

    let mut runtime = Runtime::new(GpuDialect::Hip)?;
    let artifact = runtime.load_sources(stdlib::sources().chain([source]))?;
    let [result] = artifact.exec("two-plus-two", [])?;
    let Value::U64(result) = result else {
        anyhow::bail!("two-plus-two returned non-u64 value: {result:?}");
    };

    println!("2 + 2 = {result}");
    assert_eq!(result, 4);
    Ok(())
}
```

A more complete example is available as
[catena-lang/examples/runtime.rs](catena-lang/examples/runtime.rs):

```sh
cargo run -p catena-lang --example runtime
```

NOTE: by default this will run using the
[HIP](https://rocm.docs.amd.com/projects/HIP/en/latest/) backend.
With [Nix](https://nix.dev/), you can run the example with the required
dependencies as follows:

```sh
nix develop --command cargo run -p catena-lang --example runtime
```

### Resident GPU sessions

`safe_gpu::Session::with_backend(Backend::Auto, timeouts)` selects a usable CUDA
or HIP device inside an isolated worker process. `Backend::Cuda` and
`Backend::Hip` require a specific vendor. Existing `Session::new(GpuDialect)`
callers remain supported. The parent does not initialize a GPU context for the
resident API.

`safe_gpu::AssetOwner` uploads each verified content identity to VRAM once.
`Session::with_assets(&owner, timeouts)` shares those allocations across execution
workers through read-only GPU mappings. KV caches and scratch allocations stay
private. Replacing an execution worker preserves the owner's weights; source FDs
are used only for ingestion and can be closed after attachment. `owner.usage()`
reports weight allocation count, uploaded bytes, and rounded device bytes.

The resident API requires GPU virtual-memory sharing and HIP 7.15 or newer on
AMD. Older HIP releases can accept read-only access while installing writable
mappings and are rejected. Unsupported devices return an error. Budget VRAM for
the shared weights, allocation granularity, private state, and GPU contexts.

Run the same conformance tests in a matching GPU/toolchain environment:

```sh
CATENA_GPU_DIALECT=hip cargo test -p catena-lang --no-default-features --features runtime-tests --test runtime --test safe_gpu_assets --test safe_gpu_causal_lm
CATENA_GPU_DIALECT=cuda cargo test -p catena-lang --no-default-features --features runtime-tests --test runtime --test safe_gpu_assets --test safe_gpu_causal_lm
```

The resident tests also accept `CATENA_GPU_DIALECT=auto` (the default).
CUDA compilation requires an `nvcc` supporting `-arch=native` and targets the
worker-visible device. Set vendor visibility variables before creating a session.
