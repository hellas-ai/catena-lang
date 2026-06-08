# Values and Predicates

Currently, catena models runtime values as anything tagged

    val(t)

where `t` must have some runtime representation.

But this is actually not quite enough for dependent types.
Suppose we have

    val(buf n t)

and we want to test if its length is greater than zero.
If we have

    len : val(buf n t) -> val(u64)

we can test the `val(u64)`, but we cannot *use* its "non-zeroness" as a proven
fact. To do this, we need a type-level name for the value.
Example:

    # val(n, u64)
    len : val(buf n t) -> (n : u64)

Here, `n` is a type-level name for the length of the input buffer.
In general, we can read

    x : t

as meaning:

- There is a runtime value of type t
- Its 'type level / symbolic name' is x

In this formulation, `val` means an *anonymous* (not type-level-named) value.
We could introduce an isomorphism with `:` as follows:

    # forget the name of 'x'
    forget_name : (x : t) -> val(t)

    # generate a fresh name for the anonymous value
    # NOTE: x' here and the x above are distinct by construction- we are *not*
    # really using textual representations of names here.
    fresh_name : val(t) -> (x' : t)

# Values, Buffers, and Predicates

Values are most useful for reasoning at compile time about buffer lengths.
Suppose we want to write `head-or-default`, which returns the first element of
a buffer, or a default if it has zero length.

    head-or-default : val(buf n t) ● val(t) -> val(t)

If a buffer has nonzero length, we can get a zero index value:

    ix.zero : |- n > 0 -> val(ix n)

This can be used to index the buffer - but how do we get a `|- n > 0`, i.e. a
proof that `n > 0`?

Obviously, if we have an *arbitrary* `n`, this fact may not be true.
So we have to test it with a branch!

In general, there are actually two approaches to 'gaining information' like this:

1. We branch, receiving `|- P` in the positive branch, and `|- ¬ P` in the other.
2. We *assert*, giving the whole program partial function semantics

Let's consider (2) to simplify things for now.
We can add a primitive like `assert-gt`:

    # *partial* assertion, which crashes the whole program unless `x > y`
    u64.assert-gt : (x : u64) ● (y : u64) -> (|- x > y)

The "branch" version of this is to provide the proof and its negation on two branches:

    u64.branch-gt : (x : u64) ● (y : u64) ● (|- x > y => r) ● (|- ¬ (x > y) => r) -> r

But notice: we still have to include a `branch-gt` primitive, rather than
decomposing into `gt` and `if`!

Can this be fixed?

# Linking runtime test values (bools) to proofs

Suppose we have a primitive

    u64.gt : (x : u64) ● (y : u64) -> (b : bool)

How can we combine this with `if`? For closures, it's something like this:

    if : (A => B) ● (A => B) ● Bool ● A -> B

But this fails to link the true/false branch information to each closure.
Let's do this in parts. First, modify `if` to take the predicate/its negation in each
branch:

    if : ( |- b = true ● A => B) ● (|- b = false ● A => B) ● (b : Bool) ● A -> B

Now, change the type of `u64.gt`:

    u64.gt : (x : u64) ● (y : u64) -> (b : bool) ● (|- b = true ⇒  x > y) ● (|- b = false ⇒  ¬ (x > y))

Then we can write something like:

    let (b, h_true, h_false) = u64.gt(x, y)

    # b_true : |- b = true
    # h_true : |- b = true => x > y
    # b_false : |- b = false
    # h_false : |- b = true => ¬ (x > y)
    if (\b_true a -> (modus_ponens(h_true, b_true)) ...)   # h_true b_true : |- x > y
       (\b_false a -> (modus_ponens(h_false, b_false)) ...)

# Lifting code to syntax

(TODO!)

This "lifting" feels as if it should be *automatic*: essentially, we are "lifting" our `u64.gt` primitive to the type level
by having it output an assertion we can use.
We might plausably have an exactly copy of `u64.gt` at the syntax level as follows:

    u64.gt : (x : u64) ● (y : u64) -> (b : bool) ● (|- b = true => u64.gt(x, y)) ● (|- b = false => ¬u64.gt(x, y))

This makes the "lifting" idea clearer.
(But u64.gt is not really a truth value... so what's going on here?)

Maybe we really want something like `[| u64.gt(x, y) |] = true`?

# Compositional assertions

As well as the branching approach, can we also make an "assert" which is compositional?
Idea:

    # partial: crashes unless b really is true at runtime
    assert : (b : bool) -> (|- b = true)

This basically lifts a boolean to type level, crashing if it is not true.
Use case: asserting that a `buf n t` has nonzero length:

- Have `buf n t`, but know nothing about `n`
- Want to get a zero index `Ix n`, but this only exists for `n > 0`
- Can run `u64.gt : (x : u64) ● (y : u64) -> (b : bool) ● (|- b = true => u64.gt(x, y)) ● (|- b = false => ¬u64.gt(x, y))`
- This links the runtime bool `b` with the type-level assertions that `x > y` and `¬ x > y`
- By adding an `assert` primitive, we could eliminate the bool, so that we have a `|- u64.gt(x, y)` value
- ... and the program will *crash* if this is not true

Using `assert` instead of explicit branching makes control flow a lot simpler here.
