# Conditionals: simulating control flow in dataflow with functions

(WIP)

After adding "finitary closed monoidal structure", we now want to be able to
add custom function eliminators.
For example, we would like to write `if` statements:

    if : (A => B) ● (A => B) ● Bool ● A -> B

One can regard this as the copairing function `copair_{f, g} : A + A -> A` for `A`,
but where 'branches' are *within* the theory itself instead of metatheoretic.

In fact, this can be factored into something simpler: a *selector* function:

    s_X : Bool ● X ● X -> X

One can then encode `if` by substituting `X = (A => B)`.

Note here a slight subtlety: `A => B` is a *closure* with an implicit set of
bound variables.
If we expand the two closures, we could get different bound variables,
which we'll denote `X` and `Y`.
Observe then what happens to `select`:

    select : Bool ● (X, X ● A -> B) ● (Y, Y ● A -> B) -> (???, ??? ● A -> B)

We want to *return* a closure, but it's not possible: the return type now
*depends* on the value of the bool!

Can we solve this with dependent types?
Only if we have a notion of 'if' statement in the *type* level too!

    select : (b : Bool) ● (f : (X, X ● A -> B)) ● (g : (Y, Y ● A -> B)) -> (b ? X : Y, b ? X : Y ● A -> B)

However, this is *still* problematic:

- This is specialised to function types- it doesn't arise from substitution of the `X` in `s_X`!
    - i.e., as a functor it is not well defined
    - conceptually we have a functor the other way: *hiding* closure envs quotients closures to `=>`
    - it's like an "unquotient" on types!
- Lowering `select` must be done as a special case during closure conversion!

Note also this makes our `if` become dependent too:

    if : (f : (X, X ● A -> B)) ● (g : (Y, Y ● A -> B)) ● Bool ● (b ? X : Y) ● A -> B

## A simpler approach: native `if`

Regard the "closure-typed" `if` as above

    if : (A => B) ● (A => B) ● Bool ● A -> B

Then closure conversion gives us

    if : (X, X ● A -> B) ● (Y, Y ● A -> B) ● Bool ● A -> B

Then, assuming monomorphisation, codegen for native if is something like this
(types omitted):

```c
void if(x, f, y, g, flag, a, *result) {
    if(flag) {
        *result = f(x, a)
    } else {
        *result = g(y, a)
    }
}
```
