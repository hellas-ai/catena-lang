## On Control

Some notes from [Selinger2001].

### Callbacks are control

Categorically, `catena-lang` is a symmetric monoidal category where morphisms are dataflow extended with higher-order linear functions.

It has higher-order function objects:

- `A -> B`: first-class function pointers, no captured environment.
- `A => B`: closures, possibly with captured environment.

Closures are later lowered/converted into explicit environment-passing form: `A => B` becomes roughly `X * (X * A -> B)`. In the rest of this note, I am going to use `A -> B` to denote a function type considering capturing out of scope.

In the DSL, we often use callback/continuation passing style, that is, morphisms have input wires with function types and lowering generates programs with callbacks for these types that are invoked in the body.

Callbacks are a way to simulate control flow in dataflow with functions. The relevant example are conditionals, e.g.

```
if : (A => B) ● (A => B) ● Bool ● A -> B
```

Following [Selinger2001], we want to make this claim more precise. This should help us to figure out the connection between `catena-lang` and `catena-core`. In `catena-core` we model control flow explicitely with two interleaved theories of control and data. From the interleaved theories we generate a CFG with instruction blocks/gotos that is used to build structured programs. We believe that `catena-core` is just the other side of `catena-lang`, just like in [Selinger2001] control categories are continuation categories.

Here is a tentative plan.

1. `catena-lang` : `catena-core`(CFG) = continuation categories : control categories.
2. How can we build CFG in `catena-core` in a more principled way? Currently, we need a lot of hypergraph tailoring, but I feel it can be simplified.

### Selinger 101

Let's consider a simple program `entry` receives some input, makes a decision, and transfers control to one of several possible continuations, possibly passing a different kind of payload to each one. We can implement it in two different styles.

The direct/goto representation says:

`entry: Int -> 1#Int`

Given an integer, entry either exits with no payload or with an integer as payload. `#` is similar to coproduct, but it is premonoidal because it accounts for side effects.

The CPS/call representation says:

`entry: Int x (1 -> R) x (Int -> R) -> R` or

`entry: (1 -> R) x (Int -> R) -> (Int -> R)`

The semantics is the same, but this time `entry` doesn't forward values to the next programs, but invoked some callbacks. `R` is a special object denoting a result.

We can compare the two styles in the following table.

| Control                       | Callback/CPS                                  |
| ----------------------------- | --------------------------------------------- |
| `entry : Int -> 1 # Int`      | `entry : (1 -> R) x (Int -> R) -> (Int -> R)` |
| exits are outputs             | callbacks are inputs                          |
| `goto zero()`                 | call `zero()`                                 |
| `goto pos(x)`                 | call `pos(x)`                                 |
| linking happens after `entry` | handlers are passed into `entry`              |

#### How does it fit `catena-lang`?

In the lowering phase, `->` and `=>` are not ammitted as runtime values.

- primitive arrows are usually rendered inline;
- arrow definitions are rendered as functions, the use of an arrow definition becomes a call to the function;
- Function symbols from `name.*` are static operands, not runtime values. Codegen does not lower it as a runtime pointer. Instead it records it in a function symbol table.
- Ad hoc primitives: `bool.and`, `bool.ifc`, `eval`, `gpu.materialize` use the function symbol table.

Said that, `bool.ifc` lowering looks like it lives between the two worlds: "control" is passed on calling functions, but callbacks are erased. Something like where blocks are merged in a second pass.

```
block bool.ifc(flag, a, b)
  if flag
    goto 0 with a
  else
    goto 1 with b

block then-branch(a)
   call f(a)

block else-branch(b)
   call g(b)
```

### Conditionals

Control-style `select` is basically a conditional goto (i.e. what we call "transfer" in CFG).

```
select: Bool×A×B -> A#B
```

```
select(true, a, b);[f,g]
```

Where `[f,g]` is copairing (note: `f` and `g` outputs are merged). `select` looks like `if` in core stdlib, i.e.

```
 (def if : {(2 val) ([a] val)} -> ([a . a a] {val val} +) = (
    [b a]               # Value(2)     × Value(A)
    {2.elim [a]}        # Value(1 + 1) × Value(A)
    val.*.intro         # Value((1 + 1) × A)
    control.distr       # Value((1×A) + (1×A))
    control.val.+.elim  # Value(1×A) + Value(1×A)
    control.elim2       # Value(A) + Value(A)
  ))
```

CPS-style looks like `ifc`/`if`.

```
(A->R) x (B->R) -> (Bool×A×B->R)
```

### References

[Selinger2001] Selinger, Peter. "Control categories and duality: on the categorical semantics of the lambda-mu calculus." Mathematical structures in computer science 11.2 (2001): 207-260. [pdf](https://www.mathstat.dal.ca/~selinger/papers/control.pdf)
