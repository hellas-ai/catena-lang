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

    let runtime = Runtime::from_sources(stdlib::sources().chain([source]), GpuDialect::Hip)?;
    let [result] = runtime.exec("two-plus-two", [])?;
    let Value::U64(result) = result else {
        anyhow::bail!("two-plus-two returned non-u64 value: {result:?}");
    };

    println!("2 + 2 = {result}");
    assert_eq!(result, 4);
    Ok(())
}
```

The same example is available as
[catena-lang/examples/readme.rs](catena-lang/examples/readme.rs):

```sh
cargo run -p catena-lang --example readme
```

NOTE: by default this will run using the
[HIP](https://rocm.docs.amd.com/projects/HIP/en/latest/) backend.
With [Nix](https://nix.dev/), you can run the example with the required
dependencies as follows:

```sh
nix develop --command cargo run -p catena-lang --example readme
```
