# Buffers, Materialization, and Views

This note separates three related concepts:

- a buffer stores values;
- materialization populates a buffer;
- a view describes how logical indices access stored values.

## Buffers

A buffer is a finite, linear sequence of values:

```text
buf(capability, length, element-type)
```

For example:

```text
buf(cap.own, 256, f32)
```

A buffer does not need to describe where its physical storage lives. It records
which values are available and permits flat indexing:

```text
Ix length × buf(cap.ref, length, T) -> T
```

Storage placement and lifetime can instead be determined by the operation that
creates or consumes the buffer.

## Materialization

Materialization evaluates a producer over a finite index space and stores the
results in a buffer:

```text
materialize:
    length × (Ix length => T)
      -> buf(cap.own, length, T)
```

Different materializers can use different execution strategies while returning
the same buffer abstraction.

For example:

```text
materialize
    evaluates a normal indexed producer

gpu.materialize
    evaluates one element per GPU invocation

gpu.cooperative-materialize
    lets the threads in a block populate one buffer together
```

The special operation carries the execution and storage rules. The resulting
buffer only represents the values that were stored.

## Views

A view is a closure from logical indices to values:

```text
LogicalIndex => T
```

Views hide the flat layout of a buffer. A row-major matrix view has the form:

```text
view : (Ix rows × Ix cols) => T

view(row, col) =
    buffer[row * cols + col]
```

The same buffer can be exposed through a different view without changing its
stored values.

## Tiled GPU example

A tiled matrix-multiplication iteration first constructs logical tile views:

```text
row-inner-tile : (Ix tile-size × Ix tile-size) => f32
inner-col-tile : (Ix tile-size × Ix tile-size) => f32
```

The block then materializes them cooperatively:

```text
row-inner-buffer =
    cooperative-materialize(row-inner-tile)

inner-col-buffer =
    cooperative-materialize(inner-col-tile)
```

After block synchronization, row-major views expose the buffers as matrices:

```text
row-inner-shared : (Ix tile-size × Ix tile-size) => f32
inner-col-shared : (Ix tile-size × Ix tile-size) => f32
```

The computation uses these views and does not need to know how the original
matrices were laid out.

Buffers produced cooperatively must remain within their block or kernel scope.
This restriction belongs to cooperative materialization and GPU validation; it
does not require a distinct shared-buffer type.
