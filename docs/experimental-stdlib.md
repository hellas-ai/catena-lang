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
