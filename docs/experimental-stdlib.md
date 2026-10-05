# Experimental standard library

`catena-lang/stdlib/experimental` is a standalone library. It does not extend or
load `default`, and owns its foundational declarations.

| File               | Responsibility                                                                                                                |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| `foundation.hex`   | Value ascription, proofs, products, closures, scalar types, memory ownership types, Boolean operations, and exporter helpers. |
| `numeric.hex`      | Numeric operations and the symbolic result expressions used by their dependent signatures.                                    |
| `gpu/families.hex` | The exact proof-family pack/unpack signatures referenced by the exported launch definitions.                                  |

The manifest lists sources explicitly because the GPU modules are nested under
`gpu/`; automatic bundle discovery includes only files directly in the bundle
directory. Add new nested modules to this manifest too.

## Compile-time information

`meta.*` arrows expose metavariables used by partial applications and named
arrows. Their signatures can also forward runtime values or closures. Lowering
must preserve these forwarded values and retain the symbolic information until
its consumers have been processed.

`smolcat.callback.symbol` introduces a symbolic identity, while
`smolcat.value.named` attaches a symbolic identity to an existing runtime value.
`smolcat.family.Apply.2` describes a type family applied to two symbolic arguments.
The proof-family bridges preserve the entire proposition and its application
indices. None of these helpers should be treated as a general proof constructor.

The `type.const.u32.*` names occur in both theories: in `type` they are symbolic
literal identities; in `program` they materialize the corresponding integer
with that identity in its result ascription. Likewise, `type.stdlib.numeric.*`
constructs symbolic numeric expressions, while `stdlib.numeric.*` implements
runtime operations. Sharing a prefix does not imply identical lowering.

## Validation

```sh
cargo test -p catena-lang --test experimental_stdlib
```

This test loads the manifest into an empty bundle registry, checks for missing
declarations, elaborates partial applications and names, and type-checks the
whole library, including the GPU launch and matmul definitions. It does not
invoke the default codegen or establish experimental runtime support.

The separate `experimental_runtime` integration target executes hex fixtures
using only the experimental bundle and `CodegenKind::Experimental`:

```sh
CATENA_GPU_DIALECT=hip cargo test -p catena-lang --features runtime-tests --test experimental_runtime
# Use CATENA_GPU_DIALECT=cuda for CUDA.
```

It requires the selected GPU toolchain and device. Like the existing `runtime`
target, it is gated by `runtime-tests`. The existing structural stdlib test and
default-library runtime cases remain separate.

Runtime cases live under `catena-lang/tests/experimental_runtime/`: `support.rs`
loads the bundle and reads fixtures from disk, `matmul.rs` checks device results
against a CPU reference, and `fixtures/` contains the hex programs. The first
fixtures are `matmul_naive_u64.hex` and `matmul_tiled_u64.hex`, exported from the
Smolcat `matmul_simple_naive` and `matmul_simple_tiled` examples. Their
`Stdlib_simple` references map to the experimental bundle's `stdlib` namespace.
Six additional `program` arrow declarations were added to the hex fixtures:
three in `matmul_naive_u64.hex` and three in `matmul_tiled_u64.hex`. Each set
contains one `smolcat.apply.2.*.pack` arrow for input permissions, one
`smolcat.apply.2.*.unpack` arrow for output permissions, and one
`smolcat.apply.2.*.unpack` arrow for the barrier trace. Their exact proof-family
bridge signatures come from Smolcat's checking dependencies; its per-module
export omits these declarations. They are declared locally in the test fixtures
and were not added to the experimental stdlib itself. The tiled fixture's
generated `Barrier.id.280` name is mapped to the bundle's equivalent
`Barrier.id.182` constructor. These are export compatibility adjustments; the
matmul and launch bodies are preserved.

Each algorithm has separate `perfect_tiling` and `predicated_tiling` tests.
Perfect cases use rectangular matrices whose three dimensions are divisible by
the tile width. Predicated cases leave partial row, column, and inner tiles,
including a matrix smaller than one tile. Inputs are nonuniform, output memory
starts with a sentinel, every result is checked exactly, and input buffers are
checked for unintended writes. Compilation and runtime failures fail the tests;
they are not treated as successful runtime coverage.

## Closure conversion limitation

Experimental primitives such as `runtime.global_reads`, `unsafe.launch`, and
`stdlib.gpu.scheduling.permission_redistribution` return closures; some `meta.*`
arrows forward closures too. The generic `forget_closures` adapter currently
flattens returned closure domains into operation outputs, losing their role as
future callback arguments. Unlike definitions, primitives have no body to inline.
This remains unresolved. Do not bypass it by erasing schedulers: proof generation
must still run and report failure when it cannot establish the required proof.

## Follow-up questions for Smolcat

- [ ] Why does Smolcat generate the very long `smolcat.apply.2.*.pack` and
      `.unpack` arrow names now declared in `gpu/families.hex`? Understand why the
      full family structure is encoded in the name and whether that is necessary.
- [ ] Why are the `type.stdlib.numeric.*` operations missing from Smolcat's
      output? Determine whether their declarations are never generated or are
      generated but omitted during export. Perhaps referenced by `meta.stdlib.gpu.matmul.naive_matmul.helper.callback_metavars.b.0`?
- [ ] Why does Smolcat not export the foundational types and operations needed
      by the library? Determine how the export can include them so experimental
      remains standalone without manually restoring declarations.
