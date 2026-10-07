# Compiler CLI

The compiler CLI is provided by the `catena-lang` package. Run it from the
repository root:

```sh
cargo run -p catena-lang -- program.hex -o report
```

After `cargo install catena-lang`, the equivalent command is:

```sh
catena-lang program.hex -o report
```

The CLI compiles one or more input files and writes compiler reports and
generated code to the output directory. It does not execute the program. If
compilation fails, it writes the available partial report and exits with an
error. Input-loading errors occur before report generation.

Stage starts, completions, elapsed times, and closure-conversion progress are
printed to stderr. `timings.json` records each completed or failed stage with
`stage`, `elapsed_ms`, and `status`, including partial runs. Report generation
(including SVG rendering) is timed separately from compilation.

The separate `catena-cli` package currently provides only a `status` command.

## Options

| Argument or option | Behavior |
| --- | --- |
| `<PATHS>...` | One or more source files, loaded in the supplied order. |
| `-o`, `--output-dir <PATH>` | Required directory for compilation reports. |
| `--no-svg` | Skip SVG graph rendering. |
| `--dialect <NAME>` | Select `hip` (default) or `cuda`. |
| `--codegen <NAME>` | Select `default` (implicit) or `experimental`. |
| `--stdlib <NAME>` | Select a named bundle; repeat to select multiple bundles. |
| `--stdlib-dir <PATH>` | Load and select a local bundle containing `stdlib.json`; repeat for multiple directories. |
| `--no-stdlib` | Load only input files; conflicts with both stdlib selection options. |
| `-h`, `--help` | Show usage and options. |
| `-V`, `--version` | Show the version. |

## Code generator selection

```sh
cargo run -p catena-lang -- program.hex -o report --codegen default

cargo run -p catena-lang -- program.hex -o report \
  --stdlib-dir catena-lang/stdlib/experimental --codegen experimental
```

The accepted names are `default` and `experimental`; `gpu` is not an alias.
Codegen and stdlib selection are independent: selecting a codegen does not load
a library, and selecting a library does not change the codegen.

Both backends share the compiler passes through closure conversion and product
unpacking, then use separate lowering and rendering to emit one runtime module
for the selected dialect. The CLI writes its source to `gpu/hip.cpp` or
`gpu/cuda.cpp`. Unsupported arrows or runtime types are reported as errors.

## Standard library selection

Without selection flags, the CLI loads the embedded `default` bundle before
input files. Explicit selections replace that implicit choice:

```sh
# Explicitly select the embedded default bundle
cargo run -p catena-lang -- program.hex -o report --stdlib default

# Select a local bundle
cargo run -p catena-lang -- program.hex -o report --stdlib-dir ./my-stdlib

# Combine the embedded default and a local bundle
cargo run -p catena-lang -- program.hex -o report \
  --stdlib default --stdlib-dir ./my-stdlib

# Supply all language definitions through the input files
cargo run -p catena-lang -- definitions.hex program.hex -o report --no-stdlib
```

`default` is currently the only embedded bundle. Local bundle names must differ
from embedded names and from other registered local bundles.

Named selections are visited first in their flag order, then local bundles in
their flag order. Dependencies are loaded before their dependents, once per
bundle. Input files follow all library sources. Unknown bundle names,
dependency cycles, and duplicate definitions are errors.

## Bundle folders and manifests

The current library lives in
[`catena-lang/stdlib/default/`](../catena-lang/stdlib/default/). Its
[`stdlib.json`](../catena-lang/stdlib/default/stdlib.json) contains:

```json
{"name": "default"}
```

A bundle loads every `.hex` file directly in its directory, sorted by filename.
Subdirectories and other file types are ignored. The default library is
embedded at build time using the same rule, so adding a `.hex` file requires
no file-list update; rebuild the CLI to embed it.

Place additional bundles in sibling folders or outside the repository:

```text
my-stdlib/
  stdlib.json
  operations.hex
  arrays.hex
```

Only `name` is required. To extend the embedded library, use:

```json
{"name": "my-stdlib", "extends": ["default"]}
```

Omit `extends` for a standalone bundle. Local dependencies must also be supplied
with `--stdlib-dir`; all directories are registered before dependencies are
resolved, so their command-line order need not match dependency order.

An optional `files` array replaces automatic discovery with an explicit ordered
list, for example `"files": ["operations.hex", "arrays.hex"]`. Paths are relative
to the bundle directory and cannot contain `..`. An empty list loads no source
files from that bundle. Unknown manifest fields are rejected.

## Plain-value experimental execution

The experimental library also provides `execution.hex` for programs that use
plain `val` types instead of symbolic named values. These adapters share the
experimental IR and HIP/CUDA renderer:

- `core.fold.values` carries a state through an ascending u32-indexed fold.
- `unsafe.launch_linear` accepts a positive u32 element count and an index
  callback; it launches 256-thread blocks and guards the callback against
  padded threads. Launches synchronize before returning.
- `stdlib.runtime.raw.alloc_f32`, `free`, and `bytes` manage length-tagged
  buffers. Allocation and deallocation require host execution; borrowing
  returns both the owner and its reference.
- `stdlib.runtime.raw.read_f32`, `read_u64`, and `write_f32` require device
  execution. They check alignment, element counts, and bounds. Writes require
  owned memory; callers must preserve owners until all borrowed uses finish.
- `stdlib.numeric.raw.*` supplies plain-value arithmetic and conversions.
  Narrowing and shift counts are checked; rounded f32-to-u32 conversion uses
  `nearbyintf`, with a finite representable-result check.

These operations do not replace the proof-carrying geometry and memory API.
They provide an explicit runtime-checked path for programs such as the f32
SmolLM2 port. Temporary allocations must be freed after their final consumer;
returned owned buffers transfer responsibility to the runtime caller.
