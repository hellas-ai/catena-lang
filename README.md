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
    codegen,
    compile::compile,
    runtime::{GpuDialect, Runtime, Value},
    stdlib,
};
use metacat::theory::RawTheorySet;

fn main() -> anyhow::Result<()> {
    // programs in hexpr notation: https://github.com/hellas-ai/hexpr
    let source = r#"
        (def program two-plus-two : [] -> (u64 val) = (
          ({u64.one u64.one} u64.add)
          {[two . two two]}
          u64.add
        ))
    "#;

    let report = compile(RawTheorySet::from_texts(
        stdlib::sources().chain([source]),
    )?)?;
    let module = codegen::runtime_module(
        report
            .gpu_modules
            .as_ref()
            .expect("successful compilation should contain generated modules"),
        GpuDialect::Hip,
    )?;
    let mut runtime = Runtime::new(GpuDialect::Hip)?;
    let artifact = runtime.load(module)?;
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

# Standard library bundles

The current library lives in `catena-lang/stdlib/default/`. Its `stdlib.json`
manifest is simply:

```json
{"name": "default"}
```

By default, a bundle loads every `.hex` file directly in its directory, sorted
by filename. Subdirectories and other file types are ignored. The default
library is embedded at build time using the same rule, so adding a `.hex` file
does not require maintaining a file list.

Put additional bundles in sibling folders, each with a `stdlib.json` containing
its name. Optional `extends` lists bundle dependencies; optional `files` selects
an explicit ordered file list instead of loading all `.hex` files.

```sh
# Use the embedded default bundle (also the behavior without selection flags)
cargo run -p catena-lang -- program.hex -o report --stdlib default

# Use a local bundle; its name must differ from embedded bundle names
cargo run -p catena-lang -- program.hex -o report --stdlib-dir ./my-stdlib

# Compile without library sources
cargo run -p catena-lang -- program.hex -o report --no-stdlib
```

`--stdlib` and `--stdlib-dir` can be repeated. Explicit selections replace the
implicit default; to extend it, use `"extends": ["default"]` in your manifest.
Local dependencies must also be supplied with `--stdlib-dir`. Dependencies are
loaded once, before their dependents; unknown names and cycles are errors.
