# Catena Compiler Structure

The catena-lang compiler does the following:

- Elaboration
- Typechecking
- Validation (reject some interface types)
- Transformation passes
- Code generation

Here's each in more detail:

**Elaboration** adds additional data to the raw metacat theories:

- Reject any programs/libraries with defs/decls starting with `constant.*` or `name.*`
- Constant elaboration: Add constant declarations for each used `u64` or `u32` constant
- Name elaboration:     For each `f : A -> B`, add `name.f : I -> (A -> B)` (moral type)

**Typechecking** runs the metacat typechecker on the elaborated theory.

**Validation** ensures closure types do not appear on global boundaries

**Transformation passes** attempt to simplify the structure of each definition.
This includes:

- Closure conversion
- Closure elimination ("forgetting")

See later sections for more detail.

**Code generation** actually lowers to GPU code, including *monomorphising*
definitions where necessary.

## Transformation Passes

Catena has two main transformation passes:

1. Closure conversion (lowers primitives like `if` defined in terms of closures)
2. Closure forgetting removes remaining closures

The result of the transformation passes is a fully `=>`-free program.
