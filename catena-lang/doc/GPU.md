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

Block
    : Grid block-layout thread-layout
   -> Type

block-context
    : Worker g
   -> Block g

BlockWorker
    : (block : Block g)
   -> Type

block-worker
    : (worker : Worker g)
   -> BlockWorker (block-context worker)

block-index
    : Block g
   -> Ix block-layout

block-worker-index
    : BlockWorker block
   -> Ix thread-layout

block-worker-index-is-injective
    : Injective (block-worker-index : BlockWorker block -> Ix thread-layout)

Buffer  : (dims : Shape d) -> Type -> Type
Pending : Grid block-layout thread-layout -> Type -> Type

SharedBuffer
    : (block : Block g)
   -> (dims : Shape d)
   -> Type
   -> Type

shared
    : (block : Block g)
   -> (dims : Shape d)
   -> (a : Type)
   -> SharedBuffer block dims a

SharedOwnership
    : (worker : BlockWorker block)
   -> (buffer : SharedBuffer block dims a)
   -> (index : Ix dims)
   -> Type

owns-shared-cell
    : (cell : BlockWorker block -> Ix dims)
   -> Injective cell
   -> (worker : BlockWorker block)
   -> (buffer : SharedBuffer block dims a)
   -> SharedOwnership worker buffer (cell worker)

write-shared
    : (buffer : SharedBuffer block dims a)
   -> (index : Ix dims)
   -> a
   -> SharedOwnership worker buffer index
   -> Unit

record Scheduling
    (g             : Grid block-layout thread-layout)
    (output-layout : Shape o)
where
    scheduled-index
        : Worker g
       -> Option (Ix output-layout)

    reads
        : Worker g
       -> List (Ix output-layout)

    one-writer
        : (i : Ix output-layout)
       -> count i
            (concat
                (map (\index -> option-to-list
                        (scheduled-index (worker g index)))
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

ScheduledOutput
    : (s      : Scheduling g output-layout)
   -> (output : Buffer output-layout a)
   -> (worker : Worker g)
   -> Type

some-output
    : (index : Ix output-layout)
   -> Ownership worker output index
   -> ScheduledOutput scheduling output worker

no-output
    : scheduling.scheduled-index worker = none
   -> ScheduledOutput scheduling output worker

scheduled-output
    : (s      : Scheduling g output-layout)
   -> (output : Buffer output-layout a)
   -> (worker : Worker g)
   -> ScheduledOutput s output worker

KernelResult
    : ScheduledOutput scheduling output worker
   -> Type
   -> Type

KernelResult (some-output index owns) a = a
KernelResult (no-output proof)        a = Unit

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
                 -> (scheduled : ScheduledOutput scheduling output worker)
                 -> KernelResult scheduled a)
   -> Pending g (Buffer output-layout a)

launch g output scheduling kernel =
    gpu-for g (\worker ->
        let scheduled = scheduled-output scheduling output worker
        let result    = kernel worker output scheduled

        case scheduled of
            some-output index owns -> store output index result owns
            no-output proof        -> ())

    pending g output

perfect
    : (g : Grid block-layout thread-layout)
   -> Scheduling g (worker-layout g)

perfect g =
    scheduling
        (\worker -> some (worker-index worker))
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
        scheduled-index
        (\worker -> [])
        (\i -> predicated-has-one-writer fits i)
        (\i -> no-output-readers i)
    where
        scheduled-index worker =
            if within output-layout (worker-index worker)
            then some (narrow (worker-index worker))
            else none

predicated-out-of-bounds
    : (worker : Worker g)
   -> predicated g output-layout fits
        .scheduled-index worker = none
   -> OutOfBounds (worker-index worker) output-layout

general
    : (g             : Grid block-layout thread-layout)
   -> (output-layout : Shape o)
   -> (scheduled-index : Worker g -> Option (Ix output-layout))
   -> (reads : Worker g -> List (Ix output-layout))
   -> ((i : Ix output-layout) -> exactly-one-writer scheduled-index i)
   -> ((i : Ix output-layout) -> no-reader reads i)
   -> Scheduling g output-layout

general g output-layout scheduled-index reads one-writer no-readers =
    scheduling scheduled-index reads one-writer no-readers

g : Grid (4, 4) (16, 16)
g = grid (4, 4) (16, 16)

perfect-output : Buffer (64, 64) f32
perfect-launch : Pending g (Buffer (64, 64) f32)
perfect-launch =
    launch g perfect-output (perfect g)
        (\worker output scheduled ->
            case scheduled of
                some-output (row, col) owns ->
                    a[row, col] + b[row, col]

                no-output impossible ->
                    absurd (perfect-always-has-output worker impossible))

predicated-output : Buffer (60, 50) f32
predicated-launch : Pending g (Buffer (60, 50) f32)
predicated-launch =
    launch g predicated-output
        (predicated g (60, 50) proof-60x50-fits-64x64)
        (\worker output scheduled ->
            case scheduled of
                some-output (row, col) owns ->
                    a[row, col] + b[row, col]

                no-output proof ->
                    ())

permuted : Scheduling g (64, 64)
permuted =
    general
        g
        (64, 64)
        (\worker -> some (swap (worker-index worker)))
        (\worker -> [])
        (\i -> swap-has-one-writer g i)
        (\i -> no-output-readers i)

general-output : Buffer (64, 64) f32
general-launch : Pending g (Buffer (64, 64) f32)
general-launch =
    launch g general-output permuted
        (\worker output scheduled ->
            case scheduled of
                some-output (row, col) owns ->
                    a[row, col] + b[row, col]

                no-output impossible ->
                    absurd (permuted-always-has-output worker impossible))
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
   -> (scheduled : ScheduledOutput matmul-scheduling C worker)
   -> KernelResult scheduled f32

tiled-matmul-kernel worker C scheduled =
    let block                           = block-context worker
    let worker-in-block                 = block-worker worker
    let (block-row, block-col)          = block-index block
    let worker-index-in-block           = block-worker-index worker-in-block
    let tile-cell                       =
        \block-worker ->
            cast-index
                (block-y-is-tile, block-x-is-tile)
                (block-worker-index block-worker)
    let tile-cell-is-injective          =
        cast-index-is-injective
            (block-y-is-tile, block-x-is-tile)
            block-worker-index-is-injective
    let tile-index : Ix (Tile, Tile)    = tile-cell worker-in-block
    let (tile-row, tile-col)            = tile-index
    let worker-row                      = block-row * BlockY
                                        + worker-index-in-block.row
    let worker-col                      = block-col * BlockX
                                        + worker-index-in-block.col
    let A-tile : SharedBuffer block (Tile, Tile) f32 =
        shared block (Tile, Tile) f32
    let B-tile : SharedBuffer block (Tile, Tile) f32 =
        shared block (Tile, Tile) f32
    let owns-A-tile-cell                =
        owns-shared-cell
            tile-cell tile-cell-is-injective worker-in-block A-tile
    let owns-B-tile-cell                =
        owns-shared-cell
            tile-cell tile-cell-is-injective worker-in-block B-tile
    let acc                             = 0.0

    for k-tile-start in 0 .. K step Tile
        let A-index : Option (Ix (M, K)) =
            checked-index (M, K) (worker-row, k-tile-start + tile-col)

        let B-index : Option (Ix (K, N)) =
            checked-index (K, N) (k-tile-start + tile-row, worker-col)

        write-shared A-tile tile-index
            (case A-index of
                some index -> A[index]
                none       -> 0.0)
            owns-A-tile-cell

        write-shared B-tile tile-index
            (case B-index of
                some index -> B[index]
                none       -> 0.0)
            owns-B-tile-cell

        sync block

        for k-in-tile in 0 .. min Tile (K - k-tile-start)
            acc = acc
                + A-tile[tile-row, k-in-tile]
                * B-tile[k-in-tile, tile-col]

        sync block

    case scheduled of
        some-output (output-row, output-col) owns ->
            acc

        no-output proof ->
            let outside = predicated-out-of-bounds worker proof
            ()

    where
        assume block-x-is-tile : BlockX = Tile
        assume block-y-is-tile : BlockY = Tile

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
