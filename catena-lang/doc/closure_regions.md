# Closure regions

## Graph and region data structures

The closure conversion pass turns closures into explicit functions and captured data that code generation can handle.

Region discovery and scheduling serve that extraction: they determine which operations belong in each function and which nested or captured closures must be converted first.

The starting graph is the output of [`forget_closures`](../src/pass/forget_closures.rs). Closure conversion first inlines named calls with closure-bearing interfaces, then discovers regions in that graph.

Forgetting removes closure combinators such as `defer`, `compose`, `tensor`, and `run` by rewiring the graph. Where a closure still needs conversion, a marker records its argument, result, and closure value:

```text
argument ──> body ──> result
    │                   │
    └────> !closure <───┘
               │
               v
         closure value
```

The marker has **two inputs** (argument and result) and **one output** (the closure value). It marks a boundary; it does not contain a nested graph.

```text
10 ───────────────────────┐
                          ▼
x ──────────────────────> add ──> result
│                                  │
└────────────> !closure <───────────┘
                   │
                   ▼
                   f ──> consume
```

In what follows, we use a simple pseudocode to represent the same graph:

```python
offset = 10
f = closure(x):
    return offset + x
consume(f)
```

Region discovery must recover the body of a closure from connectivity. In this example, it finds add on the path from `x` to `result`, and identifies `offset` as an external input.

The result is

```text
offset ──> environment ──────┐
                            ├──> converted consumer
          pointer(f_body) ──┘
```

## Closure Conversion

A dependency means one closure must be converted before another can be safely extracted. For example, if a closure captures another closure, it needs that captured closure’s final representation to build its own environment correctly. The scheduler chooses a closure with no unresolved dependencies, converts it, and then reassesses the updated graph.

- **Region**: the body operations between domain and codomain identified by closure marker, plus their captured inputs.
- **Dependencies**: other closures that must be converted before a region can be extracted.
- **Scheduler**: chooses a region whose dependencies have already been resolved.

```
while there are closure marker
   regions = discover regions
   dependencies = find dependencies in regions
   region = schedule next region
   extract region and replace it in the graph
```

## Dependency kinds

The [scheduler](../src/closure/schedule.rs) currently recognizes the first three dependency kinds below. The fourth is proposed for the callback adapter issue. **`A waits for B` means convert B first**, then rediscover regions.

**1. `NestedRegion`: another closure’s body is structurally inside this one.**

```python
outer = closure(x):
    inner = closure(y):
        return f(x, y)
    return apply(inner, x)
```

`outer waits for inner`. Extracting the outer body must wait until its nested closure has been replaced.

The implementation also recognizes this relationship when an internal wire touches an operation owned by another region.

**2. `CapturedClosure`: one closure captures another closure value.**

```python
producer = closure(x):
    return f(x)

consumer = closure(y):
    return apply(producer, y)
```

`consumer waits for producer`. The producer must become its `(environment, function_pointer)` representation before the consumer’s captured environment is constructed. These closures need not be structurally nested.

**3. `NestedContext`: an outer computation supplies static context to an inner closure.**

```python
outer = closure(x):
    n = type_parameter_of(x)
    inner = closure(y):
        return indexed_function[n](y)
    return apply(inner, x)
```

`outer waits for inner`. The scheduler can follow an otherwise unowned computation to the inner region’s **static context inputs**. Converting the inner closure allows subsequent discovery to include that context-computing path in the outer body.
