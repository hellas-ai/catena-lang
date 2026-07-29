# GPU Tiled Matrix Multiplication

This note describes a tiled matrix-multiplication kernel in which each GPU
thread computes one element of `C`.

A tile is a rectangular region of a matrix.

## Execution layout

The GPU execution layout has two levels:

```text
grid         = 2D collection of thread blocks
thread block = 2D collection of threads
```

The launch dimensions are:

```text
grid  = (grid-cols, grid-rows)
block = (block-cols, block-rows)
```

Each thread is identified by:

```text
block-col  = blockIdx.x
block-row  = blockIdx.y
thread-col = threadIdx.x
thread-row = threadIdx.y
```

The grid divides `C` into tiles. One thread block computes one C tile:

```text
C divided between a 2 x 3 grid of thread blocks

                         grid-cols
                 0           1           2
             +-----------+-----------+-----------+
           0 | C tile    | C tile    | C tile    |
             |  (0, 0)   |  (0, 1)   |  (0, 2)   |
             +-----------+-----------+-----------+
grid-rows  1 | C tile    | C tile    | C tile    |
             |  (1, 0)   |  (1, 1)   |  (1, 2)   |
             +-----------+-----------+-----------+
```

The C tile has the same shape as the thread block:

```text
C tile shape = block-rows x block-cols
```

Each thread computes exactly one cell in that C tile:

```text
One C tile computed by a 2 x 3 thread block

                         thread-col
                 0           1           2
             +-----------+-----------+-----------+
           0 | thread    | thread    | thread    |
             | computes  | computes  | computes  |
             | one cell  | one cell  | one cell  |
             +-----------+-----------+-----------+
thread-row 1 | thread    | thread    | thread    |
             | computes  | computes  | computes  |
             | one cell  | one cell  | one cell  |
             +-----------+-----------+-----------+
```

Consequently:

```text
matrix-rows = grid-rows x block-rows
matrix-cols = grid-cols x block-cols
```

Each C tile cell corresponds to one thread. All threads in a block collectively
compute every cell of its C tile.

## Cell computed by one thread

For block `(block-row, block-col)` and thread
`(thread-row, thread-col)`, the output coordinates are:

```text
C-row = block-row x block-rows + thread-row
C-col = block-col x block-cols + thread-col
```

For example, with `16 x 16` thread blocks, block `(0, 1)` computes
`C[0..15, 16..31]`. Thread `(3, 5)` computes `C[3, 21]`:

```text
C

                       columns
                 0..15           16..31
             +-------------+---------------------+
rows  0..15 |             |          X          |
             |             |       C[3,21]       |
             +-------------+---------------------+
     16..31 |             |                     |
             +-------------+---------------------+
```

## A and B tiles

A and B tiles are matrix regions used to compute a C tile. They are data tiles,
not additional GPU blocks.

Matrix multiplication computes:

```text
C[row, col] = sum over k of A[row, k] * B[k, col]
```

The `k` dimension is the inner dimension shared by A and B. It is divided into
chunks of `tile-inner` elements:

```text
matrix-inner = k-tile-count x tile-inner
```

`q` identifies one of those chunks:

```text
q = 0 .. k-tile-count - 1

first-k = q x tile-inner
last-k  = first-k + tile-inner - 1
```

For C tile `(block-row, block-col)`, the three coordinates select the input
regions as follows:

```text
block-row selects the rows of A
block-col selects the columns of B
q         selects matching columns of A and rows of B
```

Inner-tile iteration `q` therefore uses:

```text
A-tile(block-row, q)
B-tile(q, block-col)
```

More explicitly:

```text
A tile:
  rows block-row * block-rows
       .. block-row * block-rows + block-rows - 1
  cols q * tile-inner
       .. q * tile-inner + tile-inner - 1

B tile:
  rows q * tile-inner
       .. q * tile-inner + tile-inner - 1
  cols block-col * block-cols
       .. block-col * block-cols + block-cols - 1
```

Their shapes are:

```text
A-tile :
  block-rows x tile-inner

B-tile :
  tile-inner x block-cols

C-tile :
  block-rows x block-cols
```

For a square tiled kernel:

```text
block-rows = block-cols = tile-inner = TILE
```

Suppose block `(1, 1)` computes C tile `(1, 1)`. Over all inner iterations, it
uses the selected row of A tiles:

```text
A divided into tiles

                         inner tile q
                 0           1           2
             +-----------+-----------+-----------+
           0 |           |           |           |
             +-----------+-----------+-----------+
block-row  1 | A(1, 0)   | A(1, 1)   | A(1, 2)   |
             +-----------+-----------+-----------+
           2 |           |           |           |
             +-----------+-----------+-----------+
```

and the selected column of B tiles:

```text
B divided into tiles

                         block-col
                 0           1           2
             +-----------+-----------+-----------+
           0 |           | B(0, 1)   |           |
             +-----------+-----------+-----------+
inner q    1 |           | B(1, 1)   |           |
             +-----------+-----------+-----------+
           2 |           | B(2, 1)   |           |
             +-----------+-----------+-----------+
```

Every pair `A(block-row, q)` and `B(q, block-col)` produces a partial result
for the same C tile. The block accumulates those partial results over `q`.

## What tiling changes

Tiling does not divide one C cell between multiple threads. Each thread owns one
C cell and computes its entire dot product.

Without tiling, the thread that owns `C[row,col]` reads the complete row of A
and complete column of B directly from global memory:

```text
C[row,col] =
    A[row,0] * B[0,col]
  + A[row,1] * B[1,col]
  + ...
  + A[row,matrix-inner-1] * B[matrix-inner-1,col]
```

With tiling, the same thread computes the same sum, but splits the `k` values
into groups:

```text
accumulator = 0

for each inner tile q:
    for each k inside tile q:
        accumulator += A[row,k] * B[k,col]

C[row,col] = accumulator
```

The thread keeps its accumulator across every `q` and writes C only after the
entire dot product is complete.

The purpose of tiling is data reuse. Threads in the same block first load one
A tile and one B tile into shared memory:

- threads computing different C columns reuse the same A values;
- threads computing different C rows reuse the same B values.

The division into `q` iterations is necessary because shared memory holds only
the selected tiles, not the complete matrices.

## What the kernel does

Each thread block allocates two flat shared-memory regions:

```text
A shared size = block-rows x tile-inner
B shared size = tile-inner x block-cols
```

For the square kernel, `block-rows`, `block-cols`, and `tile-inner` are all
`TILE`. For each inner tile `q`:

1. Every thread loads one value from the selected A tile into shared memory.
2. Every thread loads one value from the selected B tile into shared memory.
3. The threads use the A and B tiles in shared memory to compute their partial
   C cells.
4. Each thread adds that partial value to its accumulator.

For the square kernel, thread `(thread-row, thread-col)` loads:

```text
A[block-row * TILE + thread-row,
  q * TILE + thread-col]

B[q * TILE + thread-row,
  block-col * TILE + thread-col]
```

After the tiles have been loaded into shared memory:

```text
# A block synchronization is required here.
```

thread `(thread-row, thread-col)` computes:

```text
partial = 0

for k in 0 .. tile-inner:
    partial +=
        A-shared-tile(thread-row, k)
      * B-shared-tile(k, thread-col)
```

The shared tiles have these shapes:

```text
A-shared-tile: block-rows x tile-inner
B-shared-tile: tile-inner x block-cols
```

Before the next iteration overwrites shared memory:

```text
# A block synchronization is required here.
```

After every inner tile has been processed, each thread writes its accumulator
to its one C cell.

## Worked example from the output point of view

Consider these `4 x 4` matrices with `TILE = 2`:

```text
A                           B

+----+----+----+----+       +----+----+----+----+
|  1 |  2 |  3 |  4 |       |  1 |  0 |  1 |  0 |
+----+----+----+----+       +----+----+----+----+
|  5 |  6 |  7 |  8 |       |  0 |  1 |  0 |  1 |
+----+----+----+----+       +----+----+----+----+
|  9 | 10 | 11 | 12 |       |  1 |  1 |  1 |  1 |
+----+----+----+----+       +----+----+----+----+
| 13 | 14 | 15 | 16 |       |  2 |  0 |  2 |  0 |
+----+----+----+----+       +----+----+----+----+
```

The launch uses a `2 x 2` grid of thread blocks, each containing `2 x 2`
threads. The grid assigns one `2 x 2` output tile to each block:

```text
C output ownership

                         block-col
                  0                       1
          +-----------------------+-----------------------+
        0 | block (0,0)           | block (0,1)           |
          | C[0,0]  C[0,1]        | C[0,2]  C[0,3]        |
          | C[1,0]  C[1,1]        | C[1,2]  C[1,3]        |
          +-----------------------+-----------------------+
block-  1 | block (1,0)           | block (1,1)           |
row       | C[2,0]  C[2,1]        | C[2,2]  C[2,3]        |
          | C[3,0]  C[3,1]        | C[3,2]  C[3,3]        |
          +-----------------------+-----------------------+
```

Focus on block `(0,1)`. Its four threads own these four output cells:

```text
block (0,1)

                         thread-col
                     0                 1
              +-----------------+-----------------+
thread-row  0 | thread (0,0)    | thread (0,1)    |
              | computes C[0,2] | computes C[0,3] |
              +-----------------+-----------------+
            1 | thread (1,0)    | thread (1,1)    |
              | computes C[1,2] | computes C[1,3] |
              +-----------------+-----------------+
```

Every thread starts its accumulator at zero. Because the inner dimension has
length four and `TILE = 2`, the block performs two inner-tile iterations.

Here `q` selects a two-element chunk of the `k` dimension:

```text
q = 0 selects k = 0, 1
q = 1 selects k = 2, 3
```

Block `(0,1)` fixes:

```text
A rows = 0, 1
B cols = 2, 3
```

Then `q` selects matching A columns and B rows:

```text
A divided into 2 x 2 tiles

                       A columns
                   0,1             2,3
             +---------------+---------------+
A rows  0,1 | A(0,0), q = 0 | A(0,1), q = 1 |
             +---------------+---------------+
        2,3 |               |               |
             +---------------+---------------+


B divided into 2 x 2 tiles

                       B columns
                   0,1             2,3
             +---------------+---------------+
B rows  0,1 |               | B(0,1), q = 0 |
             +---------------+---------------+
        2,3 |               | B(1,1), q = 1 |
             +---------------+---------------+
```

Thus:

```text
q = 0:
  A rows 0,1 and cols 0,1
  B rows 0,1 and cols 2,3

q = 1:
  A rows 0,1 and cols 2,3
  B rows 2,3 and cols 2,3
```

At `q = 0`, the block uses:

```text
A tile                      B tile

+----+----+                 +----+----+
|  1 |  2 |                 |  1 |  0 |
+----+----+                 +----+----+
|  5 |  6 |                 |  0 |  1 |
+----+----+                 +----+----+
```

Each thread computes a partial value for its own C cell. Together those values
form this first partial C tile:

```text
partial C tile, q = 0

+----+----+
|  1 |  2 |
+----+----+
|  5 |  6 |
+----+----+
```

For example, thread `(0,0)` contributes:

```text
C[0,2] partial = 1 * 1 + 2 * 0 = 1
```

At `q = 1`, the same block and threads reuse shared memory for:

```text
A tile                      B tile

+----+----+                 +----+----+
|  3 |  4 |                 |  1 |  1 |
+----+----+                 +----+----+
|  7 |  8 |                 |  2 |  0 |
+----+----+                 +----+----+
```

Each thread continues with the same accumulator. Their second partial values
form this tile:

```text
partial C tile, q = 1

+----+----+
| 11 |  3 |
+----+----+
| 23 |  7 |
+----+----+
```

For thread `(0,0)`:

```text
C[0,2] partial = 3 * 1 + 4 * 2 = 11
```

Each thread adds its two partial values. Block `(0,1)` therefore writes:

```text
C tile written by block (0,1)

+----+----+       +----+----+       +----+----+
|  1 |  2 |       | 11 |  3 |       | 12 |  5 |
+----+----+   +   +----+----+   =   +----+----+
|  5 |  6 |       | 23 |  7 |       | 28 | 13 |
+----+----+       +----+----+       +----+----+
```

After all four blocks finish, the complete output is:

```text
C

+----+----+----+----+
| 12 |  5 | 12 |  5 |
+----+----+----+----+
| 28 | 13 | 28 | 13 |
+----+----+----+----+
| 44 | 21 | 44 | 21 |
+----+----+----+----+
| 60 | 29 | 60 | 29 |
+----+----+----+----+
```

From the output point of view:

- the grid assigns one C tile to each thread block;
- the thread block assigns one C cell to each thread;
- each thread keeps that cell's accumulator across all inner-tile iterations;
- the thread writes the cell only after its accumulator is complete.

## 32 x 32 layout example

For `32 x 32` matrices and `TILE = 16`:

```text
grid-rows   = 2
grid-cols   = 2
block-rows  = 16
block-cols  = 16
tile-inner  = 16
k-tile-count = 2
```

Block `(0, 1)`, thread `(3, 5)` computes `C[3,21]`.

At `q = 0`, it loads and uses these matrix regions:

```text
A

               columns
              0..15       16..31
          +-----------+-----------+
rows 0..15|   A tile  |           |
          +-----------+-----------+
    16..31|           |           |
          +-----------+-----------+

B

               columns
              0..15       16..31
          +-----------+-----------+
rows 0..15|           |   B tile  |
          +-----------+-----------+
    16..31|           |           |
          +-----------+-----------+
```

At `q = 1`, it loads and uses:

```text
A

               columns
              0..15       16..31
          +-----------+-----------+
rows 0..15|           |   A tile  |
          +-----------+-----------+
    16..31|           |           |
          +-----------+-----------+

B

               columns
              0..15       16..31
          +-----------+-----------+
rows 0..15|           |           |
          +-----------+-----------+
    16..31|           |   B tile  |
          +-----------+-----------+
```

Both iterations contribute to `C[3,21]`. The other threads in the same block
perform the same work for the other cells in `C[0..15,16..31]`.
