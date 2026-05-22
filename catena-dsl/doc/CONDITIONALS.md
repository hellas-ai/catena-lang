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

## Implementing Select

So how should we implement

    s_X : Bool ● X ● X -> X

Naively, we can codegen this as something like

    def select(b, f, g):
        if b then f else g

However, this interacts badly with the *function

## Closure Conversion



When a definition is made having function boundaries, we would like to
transform it into one accepting a function *pointer* and a list of
explicit arguments - essentially an explicitly-encoded closure.

So for example, suppose 
