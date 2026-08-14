```text
Shape : Nat -> Type
Ix    : Shape d -> Type
size  : Shape d -> Nat
all   : (dims : Shape d) -> List (Ix dims)

Grid
    : (block-layout  : Shape d)
   -> (worker-layout : Shape d)
   -> Type

grid
    : (block-layout  : Shape d)
   -> (worker-layout : Shape d)
   -> Grid block-layout worker-layout

grid-worker-layout
    : Grid block-layout worker-layout
   -> Shape d
grid-worker-layout g = block-layout * worker-layout

Block : Grid block-layout worker-layout -> Type

OperationId : Type
SyncId      : Type

SharedMemory : Block g -> Type

available-shared-memory
    : (block : Block g)
   -> SharedMemory block

reserve
    : SharedMemory block
   -> SharedBuffer block dims a
   -> SharedMemory block

record State (block : Block g) where
    sync-trace    : List SyncId
    shared-memory : SharedMemory block
    shared        : SharedState block

SharedStatus
    = writable
    | writing
    | readable
    | reading

record SharedBufferState
    (block : Block g)
    (dims  : Shape d)
where
    allocated-at : OperationId
    generation   : Nat
    status       : SharedStatus
    writers      : Map (Ix dims) (OperationId, BlockWorker block)
    readers      : Map (Ix dims) (Set (OperationId, BlockWorker block))

SharedState block =
    Map BufferId (Sigma (Shape d) (\dims -> SharedBufferState block dims))

AfterAllocate
    : (state : State block)
   -> OperationId
   -> SharedBuffer block dims a
   -> State block

AfterWrite
    : (state : State block)
   -> OperationId
   -> BlockWorker block
   -> SharedBuffer block dims a
   -> Ix dims
   -> State block

AfterRead
    : (state : State block)
   -> OperationId
   -> BlockWorker block
   -> SharedBuffer block dims a
   -> Ix dims
   -> State block

AfterSync
    : State block
   -> SyncId
   -> State block

AfterAllocate state operation-id buffer =
    state {
        shared-memory = reserve state.shared-memory buffer,
        shared[buffer] = {
            allocated-at = operation-id,
            generation   = 0,
            status       = writable,
            writers      = {},
            readers      = {}
        }
    }

AfterWrite state operation-id worker buffer index =
    state {
        shared[buffer].status         = writing,
        shared[buffer].writers[index] =
            (operation-id, worker)
    }

AfterRead state operation-id worker buffer index =
    state {
        shared[buffer].status          = reading,
        shared[buffer].readers[index] +=
            (operation-id, worker)
    }

AfterSync state sync-id =
    collective-merge block sync-id state {
        sync-trace = state.sync-trace ++ [sync-id],
        shared     = infer-shared-state-at-barrier state.shared
    }

WorkerId : Grid block-layout worker-layout -> Type
WorkerId g = Ix (grid-worker-layout g)

block-of
    : (g : Grid block-layout worker-layout)
   -> WorkerId g
   -> Block g

record Worker
    (g     : Grid block-layout worker-layout)
    (block : Block g)
    (state : State block)
where
    grid-id     : WorkerId g
    block-proof : block-of g grid-id = block

worker-in-block
    : (id : WorkerId g)
   -> BlockWorker (block-of g id)

InitialState
    : (block : Block g)
   -> State block

InitialState block = {
    sync-trace    = [],
    shared-memory = available-shared-memory block,
    shared        = {}
}

worker
    : (g : Grid block-layout worker-layout)
   -> (id : WorkerId g)
   -> Worker g (block-of g id) (InitialState (block-of g id))

grid-worker-id
    : Worker g block state
   -> WorkerId g
grid-worker-id worker = worker.grid-id

worker-state
    : Worker g block state
   -> State block
worker-state worker = state

block-context
    : Worker g block state
   -> Block g
block-context worker = block

BlockWorker : (block : Block g) -> Type

block-worker
    : Worker g block state
   -> BlockWorker block

block-index
    : Block g
   -> Ix block-layout

block-worker-index
    : BlockWorker block
   -> Ix worker-layout

block-worker-index-is-injective
    : Injective (block-worker-index : BlockWorker block -> Ix worker-layout)

Buffer  : (dims : Shape d) -> Type -> Type
Pending : Grid block-layout worker-layout -> Type -> Type

SharedBuffer
    : (block : Block g)
   -> (dims : Shape d)
   -> Type
   -> Type

Proposition : Type
(|-) : Proposition -> Type

Defined
    : State block
   -> SharedBuffer block dims a
   -> Ix dims
   -> Proposition

SyncCompleted
    : State block
   -> SyncId
   -> Proposition

EveryWorkerWroteBefore
    : State block
   -> OperationId
   -> SyncId
   -> SharedBuffer block dims a
   -> (BlockWorker block -> Ix dims)
   -> Proposition

CoversEveryCell
    : (BlockWorker block -> Ix dims)
   -> Proposition

sync-completed
    : (worker : Worker g block state)
   -> (sync-id : SyncId)
   -> |- SyncCompleted state sync-id
sync-completed worker sync-id =
    derive (worker-state worker)

every-worker-wrote-before
    : (worker       : Worker g block state)
   -> (operation-id : OperationId)
   -> (sync-id      : SyncId)
   -> (buffer       : SharedBuffer block dims a)
   -> (cell         : BlockWorker block -> Ix dims)
   -> |- EveryWorkerWroteBefore
        state operation-id sync-id buffer cell
every-worker-wrote-before worker operation-id sync-id buffer cell =
    derive (worker-state worker)

defined-from-completed-writes
    : |- SyncCompleted state sync-id
   -> |- EveryWorkerWroteBefore
        state operation-id sync-id buffer cell
   -> |- CoversEveryCell cell
   -> (index : Ix dims)
   -> |- Defined state buffer index

defined-after-cooperative-load
    : (worker       : Worker g block state)
   -> (operation-id : OperationId)
   -> (sync-id      : SyncId)
   -> (buffer       : SharedBuffer block dims a)
   -> (cell         : BlockWorker block -> Ix dims)
   -> |- CoversEveryCell cell
   -> (index : Ix dims)
   -> |- Defined state buffer index
defined-after-cooperative-load
    worker operation-id sync-id buffer cell covers index =
    defined-from-completed-writes
        (sync-completed worker sync-id)
        (every-worker-wrote-before
            worker operation-id sync-id buffer cell)
        covers
        index

shared
    : (operation-id : OperationId)
   -> (worker : Worker g block state)
   -> (dims : Shape d)
   -> (a : Type)
   -> Sigma (SharedBuffer block dims a)
        (\buffer ->
            Worker g block (AfterAllocate state operation-id buffer))

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
    : (operation-id : OperationId)
   -> (worker : Worker g block state)
   -> (buffer : SharedBuffer block dims a)
   -> (index : Ix dims)
   -> a
   -> SharedOwnership (block-worker worker) buffer index
   -> Worker g block
        (AfterWrite state operation-id (block-worker worker) buffer index)

read-shared
    : (operation-id : OperationId)
   -> (worker : Worker g block state)
   -> (buffer : SharedBuffer block dims a)
   -> (index : Ix dims)
   -> |- Defined state buffer index
   -> (a,
       Worker g
           block
           (AfterRead state operation-id (block-worker worker) buffer index))

sync
    : (sync-id : SyncId)
   -> Worker g block state
   -> Worker g block (AfterSync state sync-id)

record Scheduling
    (g             : Grid block-layout worker-layout)
    (output-layout : Shape o)
where
    scheduled-index
        : WorkerId g
       -> Option (Ix output-layout)

    one-writer
        : (i : Ix output-layout)
       -> count i
            (concat
                (map (\id -> option-to-list (scheduled-index id))
                     (all (grid-worker-layout g))))
        = 1

    no-readers
        : (id : WorkerId g)
       -> (i : Ix output-layout)
       -> Not (Reads id i)

record Ownership
    (id     : WorkerId g)
    (output : Buffer output-layout a)
    (index  : Ix output-layout)
where
    write-access
        : Writes id output index

    only-writer
        : (other : WorkerId g)
       -> Writes other output index
       -> other = id

    no-reader
        : (other : WorkerId g)
       -> Not (Reads other index)

ScheduledOutput
    : (scheduling : Scheduling g output-layout)
   -> (output     : Buffer output-layout a)
   -> (worker     : Worker g block state)
   -> Type

has-output
    : (output-index : Ix output-layout)
   -> scheduling.scheduled-index (grid-worker-id worker) = some output-index
   -> Ownership (grid-worker-id worker) output output-index
   -> ScheduledOutput scheduling output worker

no-output
    : scheduling.scheduled-index (grid-worker-id worker) = none
   -> ScheduledOutput scheduling output worker

scheduled-output
    : (scheduling : Scheduling g output-layout)
   -> (output     : Buffer output-layout a)
   -> (worker     : Worker g block state)
   -> ScheduledOutput scheduling output worker

KernelResult
    : ScheduledOutput scheduling output worker
   -> Type
   -> Type

KernelResult (has-output index scheduled-here owns) a = a
KernelResult (no-output not-scheduled)               a = Unit

KernelExecution
    : (g : Grid block-layout worker-layout)
   -> Block g
   -> Type
   -> Type
KernelExecution g block result =
    Sigma (State block)
        (\final-state -> (result, Worker g block final-state))

store
    : (output : Buffer output-layout a)
   -> (index : Ix output-layout)
   -> a
   -> Ownership id output index
   -> Unit

launch
    : (g          : Grid block-layout worker-layout)
   -> (output     : Buffer output-layout a)
   -> (scheduling : Scheduling g output-layout)
   -> (kernel
        : {block : Block g}
       -> (worker : Worker g block (InitialState block))
       -> (output : Buffer output-layout a)
       -> (scheduled : ScheduledOutput scheduling output worker)
       -> KernelExecution g block (KernelResult scheduled a))
    -> Pending g (Buffer output-layout a)

launch g output scheduling kernel =
    gpu-for g (\id ->
        let worker-0 = worker g id
        assert-type worker-0 ==
            Worker g (block-of g id) (InitialState (block-of g id))
        let scheduled = scheduled-output scheduling output worker-0
        let (final-state, (result, final-worker)) =
            kernel worker-0 output scheduled

        case scheduled of
            has-output index scheduled-here owns ->
                store output index result owns

            no-output not-scheduled -> ())

    pending g output

perfect
    : (g : Grid block-layout worker-layout)
   -> Scheduling g (grid-worker-layout g)

perfect g =
    scheduling
        (\id -> some id)
        (\i -> all-contains-once (grid-worker-layout g) i)
        (\id i -> no-output-readers id i)

predicated
    : (g             : Grid block-layout worker-layout)
   -> (output-layout : Shape d)
   -> output-layout <= grid-worker-layout g
   -> Scheduling g output-layout

predicated g output-layout fits =
    scheduling
        scheduled-index
        (\i -> predicated-has-one-writer fits i)
        (\id i -> no-output-readers id i)
    where
        scheduled-index id =
            if within output-layout id
            then some (narrow id)
            else none

predicated-out-of-bounds
    : predicated g output-layout fits .scheduled-index id = none
   -> OutOfBounds id output-layout

general
    : (g             : Grid block-layout worker-layout)
   -> (output-layout : Shape o)
   -> (scheduled-index : WorkerId g -> Option (Ix output-layout))
   -> ((i : Ix output-layout) -> exactly-one-writer scheduled-index i)
   -> ((id : WorkerId g) -> (i : Ix output-layout) -> Not (Reads id i))
   -> Scheduling g output-layout

general g output-layout scheduled-index one-writer no-readers =
    scheduling scheduled-index one-writer no-readers
```

```text
M N K Tile : Nat

GridX GridY BlockX BlockY : Nat

matmul-covers-output
    : (M, N) <=
        grid-worker-layout (Grid (GridY, GridX) (BlockY, BlockX))

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
    : {block
        : Block (Grid (GridY, GridX) (BlockY, BlockX))}
   -> (worker-0
        : Worker
            (Grid (GridY, GridX) (BlockY, BlockX))
            block
            (InitialState block))
   -> (C : Buffer (M, N) f32)
   -> (scheduled
        : ScheduledOutput matmul-scheduling C worker-0)
   -> KernelExecution
        (Grid (GridY, GridX) (BlockY, BlockX))
        block
        (KernelResult scheduled f32)

tiled-matmul-kernel {block} worker-0 C scheduled =
    assert-type worker-0 == Worker _ block {
        sync-trace    = [],
        shared-memory = available-shared-memory block,
        shared        = {}
    }

    let (A-tile, worker-1) =
        shared @A-tile worker-0 (Tile, Tile) f32

    assert-type worker-1 == Worker _ block {
        sync-trace    = (worker-state worker-0).sync-trace,
        shared-memory = reserve (worker-state worker-0).shared-memory A-tile,
        shared        = (worker-state worker-0).shared + {
            A-tile -> {
                allocated-at = @A-tile,
                generation   = 0,
                status       = writable,
                writers      = {},
                readers      = {}
            }
        }
    }

    let (B-tile, worker-2) =
        shared @B-tile worker-1 (Tile, Tile) f32

    assert-type worker-2 == Worker _ block {
        sync-trace = (worker-state worker-0).sync-trace,
        shared-memory =
            reserve
                (reserve (worker-state worker-0).shared-memory A-tile)
                B-tile,
        shared = (worker-state worker-0).shared + {
            A-tile -> {
                allocated-at = @A-tile,
                generation   = 0,
                status       = writable,
                writers      = {},
                readers      = {}
            },
            B-tile -> {
                allocated-at = @B-tile,
                generation   = 0,
                status       = writable,
                writers      = {},
                readers      = {}
            }
        }
    }

    let worker-in-block = block-worker worker-2
    let (block-row, block-col) = block-index block
    let worker-index-in-block = block-worker-index worker-in-block

    let tile-cell =
        \block-worker ->
            cast-index
                (block-y-is-tile, block-x-is-tile)
                (block-worker-index block-worker)

    let tile-cell-is-injective =
        cast-index-is-injective
            (block-y-is-tile, block-x-is-tile)
            block-worker-index-is-injective

    let tile-cell-covers-every-cell
        : |- CoversEveryCell tile-cell
        = covers-from-equal-layouts
            block-y-is-tile
            block-x-is-tile
            tile-cell-is-injective

    let tile-index : Ix (Tile, Tile) = tile-cell worker-in-block
    let (tile-row, tile-col) = tile-index
    let worker-row = block-row * BlockY + worker-index-in-block.row
    let worker-col = block-col * BlockX + worker-index-in-block.col

    let owns-A-tile-cell =
        owns-shared-cell
            tile-cell tile-cell-is-injective worker-in-block A-tile

    let owns-B-tile-cell =
        owns-shared-cell
            tile-cell tile-cell-is-injective worker-in-block B-tile

    let acc = 0.0
    let worker = worker-2

    for k-tile-start in 0 .. K step Tile
        with (acc, worker)

        let A-index : Option (Ix (M, K)) =
            checked-index (M, K) (worker-row, k-tile-start + tile-col)

        let B-index : Option (Ix (K, N)) =
            checked-index (K, N) (k-tile-start + tile-row, worker-col)

        let worker-after-A-write =
            write-shared @load-A[k-tile-start]
                worker A-tile tile-index
                (case A-index of
                    some index -> A[index]
                    none       -> 0.0)
                owns-A-tile-cell

        assert-type worker-after-A-write ==
            Worker _ block
                (AfterWrite
                    (worker-state worker)
                    @load-A[k-tile-start]
                    (block-worker worker)
                    A-tile tile-index)

        let worker-after-B-write =
            write-shared @load-B[k-tile-start]
                worker-after-A-write B-tile tile-index
                (case B-index of
                    some index -> B[index]
                    none       -> 0.0)
                owns-B-tile-cell

        assert-type worker-after-B-write ==
            Worker _ block
                (AfterWrite
                    (worker-state worker-after-A-write)
                    @load-B[k-tile-start]
                    (block-worker worker-after-A-write)
                    B-tile tile-index)

        let worker-after-load-sync =
            sync @tiles-loaded[k-tile-start] worker-after-B-write

        assert-type worker-after-load-sync ==
            Worker _ block
                (AfterSync
                    (worker-state worker-after-B-write)
                    @tiles-loaded[k-tile-start])

        let tile-worker = worker-after-load-sync

        for k-in-tile in 0 .. min Tile (K - k-tile-start)
            with (acc, tile-worker)

            let A-read-index : Ix (Tile, Tile) =
                (tile-row, k-in-tile)

            let A-defined
                : |- Defined
                    (worker-state tile-worker)
                    A-tile
                    A-read-index
                = defined-after-cooperative-load
                    tile-worker
                    @load-A[k-tile-start]
                    @tiles-loaded[k-tile-start]
                    A-tile
                    tile-cell
                    tile-cell-covers-every-cell
                    A-read-index

            let (A-value, worker-after-A-read) =
                read-shared @read-A[k-tile-start, k-in-tile]
                    tile-worker A-tile A-read-index A-defined

            assert-type worker-after-A-read ==
                Worker _ block
                    (AfterRead
                        (worker-state tile-worker)
                        @read-A[k-tile-start, k-in-tile]
                        (block-worker tile-worker)
                        A-tile A-read-index)

            let B-read-index : Ix (Tile, Tile) =
                (k-in-tile, tile-col)

            let B-defined
                : |- Defined
                    (worker-state worker-after-A-read)
                    B-tile
                    B-read-index
                = defined-after-cooperative-load
                    worker-after-A-read
                    @load-B[k-tile-start]
                    @tiles-loaded[k-tile-start]
                    B-tile
                    tile-cell
                    tile-cell-covers-every-cell
                    B-read-index

            let (B-value, worker-after-B-read) =
                read-shared @read-B[k-tile-start, k-in-tile]
                    worker-after-A-read B-tile B-read-index B-defined

            assert-type worker-after-B-read ==
                Worker _ block
                    (AfterRead
                        (worker-state worker-after-A-read)
                        @read-B[k-tile-start, k-in-tile]
                        (block-worker worker-after-A-read)
                        B-tile B-read-index)

            acc = acc + A-value * B-value
            tile-worker = worker-after-B-read

        let worker-after-reads = tile-worker

        let worker-after-consumed-sync =
            sync
                @tiles-consumed[k-tile-start]
                worker-after-reads

        assert-type worker-after-consumed-sync ==
            Worker _ block
                (AfterSync
                    (worker-state worker-after-reads)
                    @tiles-consumed[k-tile-start])

        worker = worker-after-consumed-sync

    case scheduled of
        has-output (output-row, output-col) scheduled-here owns ->
            (worker-state worker, (acc, worker))

        no-output proof ->
            let outside = predicated-out-of-bounds proof
            (worker-state worker, ((), worker))

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
