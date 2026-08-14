```text
Shape : Nat -> Type
Ix    : Shape d -> Type
size  : Shape d -> Nat
all   : (dims : Shape d) -> List (Ix dims)

Grid
    : (block-layout  : Shape d)
   -> (thread-layout : Shape d)
   -> Type

grid
    : (block-layout  : Shape d)
   -> (thread-layout : Shape d)
   -> Grid block-layout thread-layout

worker-layout
    : Grid block-layout thread-layout
   -> Shape d
worker-layout g = block-layout * thread-layout

Worker
    : (g : Grid block-layout thread-layout)
   -> Type

worker
    : (g : Grid block-layout thread-layout)
   -> Ix (worker-layout g)
   -> Worker g

worker-index
    : Worker g
   -> Ix (worker-layout g)

Buffer  : (dims : Shape d) -> Type -> Type
Pending : Grid block-layout thread-layout -> Type -> Type

record Scheduling
    (g             : Grid block-layout thread-layout)
    (output-layout : Shape o)
where
    writes
        : Worker g
       -> List (Ix output-layout)

    reads
        : Worker g
       -> List (Ix output-layout)

    one-writer
        : (i : Ix output-layout)
       -> count i
            (concat
                (map (\index -> writes (worker g index))
                     (all (worker-layout g))))
        = 1

    no-readers
        : (i : Ix output-layout)
       -> count i
            (concat
                (map (\index -> reads (worker g index))
                     (all (worker-layout g))))
        = 0

Writes
    : Worker g
   -> Buffer output-layout a
   -> Ix output-layout
   -> Type

Reads
    : Worker g
   -> Buffer output-layout a
   -> Ix output-layout
   -> Type

record Ownership
    (worker : Worker g)
    (output : Buffer output-layout a)
    (index  : Ix output-layout)
where
    write-access
        : Writes worker output index

    only-writer
        : (other : Worker g)
       -> Writes other output index
       -> other = worker

    no-reader
        : (other : Worker g)
       -> Not (Reads other output index)

ownership
    : (scheduling : Scheduling g output-layout)
   -> Member index (scheduling.writes worker)
   -> Ownership worker output index

store
   : (output : Buffer output-layout a)
   -> (index : Ix output-layout)
   -> a
   -> Ownership worker output index
   -> Unit

launch
   : (g          : Grid block-layout thread-layout)
   -> (output     : Buffer output-layout a)
   -> (scheduling : Scheduling g output-layout)
   -> (kernel     : (worker : Worker g)
                 -> (output : Buffer output-layout a)
                 -> (index : Ix output-layout)
                 -> Ownership worker output index
                 -> a)
   -> Pending g (Buffer output-layout a)

launch g output scheduling kernel =
    gpu-for g (\worker ->
        for index in scheduling.writes worker with member
            let owns = ownership scheduling member
            let value = kernel worker output index owns
            store output index value owns)

    pending g output

perfect
    : (g : Grid block-layout thread-layout)
   -> Scheduling g (worker-layout g)

perfect g =
    scheduling
        (\worker -> [worker-index worker])
        (\worker -> [])
        (\i -> all-contains-once (worker-layout g) i)
        (\i -> no-output-readers i)

predicated
    : (g             : Grid block-layout thread-layout)
   -> (output-layout : Shape d)
   -> output-layout <= worker-layout g
   -> Scheduling g output-layout

predicated g output-layout fits =
    scheduling
        writes
        (\worker -> [])
        (\i -> predicated-has-one-writer fits i)
        (\i -> no-output-readers i)
    where
        writes worker =
            if within output-layout (worker-index worker)
            then [narrow (worker-index worker)]
            else []

general
    : (g             : Grid block-layout thread-layout)
   -> (output-layout : Shape o)
   -> (writes : Worker g -> List (Ix output-layout))
   -> (reads  : Worker g -> List (Ix output-layout))
   -> ((i : Ix output-layout) -> exactly-one-writer writes i)
   -> ((i : Ix output-layout) -> no-reader reads i)
   -> Scheduling g output-layout

general g output-layout writes reads one-writer no-readers =
    scheduling writes reads one-writer no-readers

g : Grid (4, 4) (16, 16)
g = grid (4, 4) (16, 16)

perfect-output : Buffer (64, 64) f32
perfect-launch : Pending g (Buffer (64, 64) f32)
perfect-launch =
    launch g perfect-output
        (perfect g)
        (\worker output (row, col) owns ->
            a[row, col] + b[row, col])

predicated-output : Buffer (60, 50) f32
predicated-launch : Pending g (Buffer (60, 50) f32)
predicated-launch =
    launch g predicated-output
        (predicated g (60, 50) proof-60x50-fits-64x64)
        (\worker output (row, col) owns ->
            a[row, col] + b[row, col])

grid-stride
    : (g             : Grid block-layout thread-layout)
   -> (output-layout : Shape o)
   -> Scheduling g output-layout

grid-stride g output-layout =
    general
        g
        output-layout
        writes
        (\worker -> [])
        (\i -> grid-stride-has-one-writer g output-layout i)
        (\i -> no-output-readers i)
    where
        writes worker =
            map (unlinearize output-layout)
                [ rank (worker-index worker),
                  rank (worker-index worker) + size (worker-layout g),
                  rank (worker-index worker) + 2 * size (worker-layout g),
                  ...
                ]
                while offset < size output-layout

general-output : Buffer (100, 100) f32
general-launch : Pending g (Buffer (100, 100) f32)
general-launch =
    launch g general-output
        (grid-stride g (100, 100))
        (\worker output (row, col) owns ->
            a[row, col] + b[row, col])
```

```text
M N K Tile : Nat

GridX GridY BlockX BlockY : Nat

matmul-covers-output
    : (M, N) <= worker-layout (Grid (GridY, GridX) (BlockY, BlockX))

A : Buffer (M, K) f32
B : Buffer (K, N) f32
C : Buffer (M, N) f32

matmul-scheduling
    : Scheduling (Grid (GridY, GridX) (BlockY, BlockX)) (M, N)
matmul-scheduling =
    predicated
        (grid (GridY, GridX) (BlockY, BlockX))
        (M, N)
        matmul-covers-output

tiled-matmul-kernel
   : (worker : Worker (Grid (GridY, GridX) (BlockY, BlockX)))
   -> (C : Buffer (M, N) f32)
   -> (output-index : Ix (M, N))
   -> Ownership worker C output-index
   -> f32

tiled-matmul-kernel worker C (row, col) owns =
    let block              = block-context worker
    let (block-r, block-c) = block-index worker
    let (local-r, local-c) = thread-index worker
    let threads-per-block  = BlockY * BlockX
    let thread-rank        = local-r * BlockX + local-c
    let A-tile             = shared block (BlockY, Tile) f32
    let B-tile             = shared block (Tile, BlockX) f32
    let acc                = 0.0

    for k0 in 0 .. K step Tile
        for offset in thread-rank .. BlockY * Tile step threads-per-block
            let (tile-r, tile-k) = unlinearize (BlockY, Tile) offset
            let source-r         = block-r * BlockY + tile-r
            let source-k         = k0 + tile-k

            A-tile[tile-r, tile-k] =
                if source-r < M && source-k < K
                then A[source-r, source-k]
                else 0.0

        for offset in thread-rank .. Tile * BlockX step threads-per-block
            let (tile-k, tile-c) = unlinearize (Tile, BlockX) offset
            let source-k         = k0 + tile-k
            let source-c         = block-c * BlockX + tile-c

            B-tile[tile-k, tile-c] =
                if source-k < K && source-c < N
                then B[source-k, source-c]
                else 0.0

        sync block

        for k in 0 .. min Tile (K - k0)
            acc = acc
                + A-tile[local-r, k]
                * B-tile[k, local-c]

        sync block

    acc

pending-C
    : Pending (Grid (GridY, GridX) (BlockY, BlockX))
              (Buffer (M, N) f32)
pending-C =
    launch
        (grid (GridY, GridX) (BlockY, BlockX))
        C
        matmul-scheduling
        tiled-matmul-kernel
```
