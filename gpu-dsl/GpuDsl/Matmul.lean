import GpuDsl.Launch
import GpuDsl.Tiling

namespace GpuDsl

def predicatedConfig (rows cols tile : Nat) : LaunchConfig :=
  { grid := { x := ceilDiv cols tile, y := ceilDiv rows tile }
    block := { x := tile, y := tile } }

def perfectConfig (rows cols tile : Nat) : LaunchConfig :=
  { grid := { x := cols / tile, y := rows / tile }
    block := { x := tile, y := tile } }

/--
Assign each output cell to the thread at the corresponding tiled coordinate.
The bounds arguments are launch-side obligations. The resulting owner function
is the certificate that every cell has exactly one processing thread.
-/
def tiledScheduleCertificate
    (cfg : LaunchConfig)
    (rows cols tile : Nat)
    (tilePositive : 0 < tile)
    (blockX : Dim3.x (LaunchConfig.block cfg) = tile)
    (blockY : Dim3.y (LaunchConfig.block cfg) = tile)
    (gridCoversCols : cols ≤ Dim3.x (LaunchConfig.grid cfg) * tile)
    (gridCoversRows : rows ≤ Dim3.y (LaunchConfig.grid cfg) * tile)
    (gridZPositive : 0 < Dim3.z (LaunchConfig.grid cfg))
    (blockZPositive : 0 < Dim3.z (LaunchConfig.block cfg)) :
    ScheduleCertificate cfg (.d2 rows cols) where
  owner outputIndex :=
    let row := Fin.val (Index2.row outputIndex)
    let col := Fin.val (Index2.col outputIndex)
    let blockCol : Fin (Dim3.x (LaunchConfig.grid cfg)) :=
      ⟨col / tile, (Nat.div_lt_iff_lt_mul tilePositive).2
        (Nat.lt_of_lt_of_le (Fin.isLt (Index2.col outputIndex)) gridCoversCols)⟩
    let blockRow : Fin (Dim3.y (LaunchConfig.grid cfg)) :=
      ⟨row / tile, (Nat.div_lt_iff_lt_mul tilePositive).2
        (Nat.lt_of_lt_of_le (Fin.isLt (Index2.row outputIndex)) gridCoversRows)⟩
    let threadCol : Fin (Dim3.x (LaunchConfig.block cfg)) :=
      Fin.cast blockX.symm ⟨col % tile, Nat.mod_lt col tilePositive⟩
    let threadRow : Fin (Dim3.y (LaunchConfig.block cfg)) :=
      Fin.cast blockY.symm ⟨row % tile, Nat.mod_lt row tilePositive⟩
    { blockIdx := { x := blockCol, y := blockRow, z := ⟨0, gridZPositive⟩ }
      threadIdx := { x := threadCol, y := threadRow, z := ⟨0, blockZPositive⟩ } }

def tileRow (tilePositive : 0 < tile) (index : Fin (tile * tile)) : Fin tile :=
  ⟨Fin.val index / tile, (Nat.div_lt_iff_lt_mul tilePositive).2 (Fin.isLt index)⟩

def tileCol (tilePositive : 0 < tile) (index : Fin (tile * tile)) : Fin tile :=
  ⟨Fin.val index % tile, Nat.mod_lt _ tilePositive⟩

/-- Each shared cell names its unique physical writer; a thread may own many cells. -/
def tileWriteOwnership
    (blockX : Dim3.x (LaunchConfig.block cfg) = tile)
    (blockY : Dim3.y (LaunchConfig.block cfg) = tile)
    (blockZ : Dim3.z (LaunchConfig.block cfg) = 1)
    (tilePositive : 0 < tile) : WriteOwnership cfg (tile * tile) where
  owner index :=
    { x := Fin.cast blockX.symm (tileCol tilePositive index)
      y := Fin.cast blockY.symm (tileRow tilePositive index)
      z := Fin.cast blockZ.symm ⟨0, by decide⟩ }

def tiledMatmulStep (trace : SyncTrace) : SyncTrace :=
  (trace ++ [.tileLoaded]) ++ [.tileConsumed]

def foldValueFin (n : Nat) (initial : Value α)
    (body : Fin n → Value α → Value α) : Value α :=
  let rec go (i : Nat) (acc : Value α) : Value α :=
    if hi : i < n then go (i + 1) (body ⟨i, hi⟩ acc) else acc
  termination_by n - i
  go 0 initial

/-- The resources that must be available at the top of every tile iteration. -/
structure TileLoopState
    (sharedA : SharedBuffer block lengthA α)
    (sharedB : SharedBuffer block lengthB α)
    (trace : SyncTrace) where
  accumulator : Value α
  writableA : WritableAt sharedA trace
  writableB : WritableAt sharedB trace

/--
One tiled matmul kernel for both perfect and predicated launches.

`block.x = block.y = tile` is checked uniformly in the body instead of being a
constraint in the signature. All threads execute the shared-memory phases and
barriers, including threads with no scheduled output. Only returning an output
value is conditional on the assignment supplied and proved by `launch`.
-/
def tiledMatmulKernel
    {α : Type}
    [OfNat α 0]
    [Add α]
    [Mul α]
    (tile : Nat)
    (tilePositive : 0 < tile)
    (a : Matrix rows inner (Value α))
    (b : Matrix inner cols (Value α)) :
    Kernel (.d2 rows cols) α where
  shared := .buffer "tileA" (tile * tile) (.buffer "tileB" (tile * tile) .empty)
  trace := iterateTrace tiledMatmulStep (ceilDiv inner tile) []
  body := fun {cfg} {_State} {_certificate} thread scheduled =>
    KernelM.require
      (Dim3.x (LaunchConfig.block cfg) = tile ∧
        Dim3.y (LaunchConfig.block cfg) = tile ∧
        Dim3.z (LaunchConfig.block cfg) = 1)
      fun ⟨blockXEq, blockYEq, blockZEq⟩ =>
      let threadIndex := Thread.index thread
      let block := Thread.block thread
      let blockIndex := Block.index block
      let threadCol : Fin tile := Fin.cast blockXEq (Coord.x threadIndex)
      let threadRow : Fin tile := Fin.cast blockYEq (Coord.y threadIndex)
      let sharedA := SharedState.get block SharedRef.here
      let sharedB := SharedState.get block (SharedRef.there SharedRef.here)
      let ownership := tileWriteOwnership blockXEq blockYEq blockZEq tilePositive
      let initialState : TileLoopState sharedA sharedB [] :=
        { accumulator := Value.zero
          writableA := WritableAt.initial sharedA
          writableB := WritableAt.initial sharedB }

      KernelM.bind
        (KernelM.foldFinD (ceilDiv inner tile)
          (fun trace => TileLoopState sharedA sharedB trace) initialState
          fun _trace kTile state =>
          let valueA := fun index (_owned : Owns ownership thread index) =>
            let row := Fin.val (Coord.y blockIndex) * tile + Fin.val threadRow
            let kA := Fin.val kTile * tile + Fin.val threadCol
            if rowInBounds : row < rows then
              if kAInBounds : kA < inner then
                a ⟨⟨row, rowInBounds⟩, ⟨kA, kAInBounds⟩⟩
              else Value.zero
            else Value.zero
          let valueB := fun index (_owned : Owns ownership thread index) =>
            let kB := Fin.val kTile * tile + Fin.val threadRow
            let col := Fin.val (Coord.x blockIndex) * tile + Fin.val threadCol
            if kBInBounds : kB < inner then
              if colInBounds : col < cols then
                b ⟨⟨kB, kBInBounds⟩, ⟨col, colInBounds⟩⟩
              else Value.zero
            else Value.zero
          KernelM.bind
            (KernelM.initializeShared sharedA ownership
              (TileLoopState.writableA state) valueA) fun writesA =>
          KernelM.bind
            (KernelM.initializeShared sharedB ownership
              (TileLoopState.writableB state) valueB) fun writesB =>
          KernelM.bind (KernelM.syncBlock .tileLoaded) fun loaded =>
          let nextExpression :=
            foldValueFin tile (TileLoopState.accumulator state) fun k partialSum =>
            let indexA := Layout.offset Layout.rowMajor2D ⟨threadRow, k⟩
            let indexB := Layout.offset Layout.rowMajor2D ⟨k, threadCol⟩
            let definedA := SharedWritePhase.definedAfter writesA loaded indexA
            let definedB := SharedWritePhase.definedAfter writesB loaded indexB
            let av := SharedBuffer.read sharedA indexA definedA
            let bv := SharedBuffer.read sharedB indexB definedB
            Value.add partialSum (Value.mul av bv)
          KernelM.bind (KernelM.materialize nextExpression) fun nextAccumulator =>
          KernelM.bind (KernelM.finishSharedReads sharedA) fun readsA =>
          KernelM.bind (KernelM.finishSharedReads sharedB) fun readsB =>
          KernelM.bind (KernelM.syncBlock .tileConsumed) fun consumed =>
          KernelM.pure
            { accumulator := nextAccumulator
              writableA := SharedReadPhase.writableAfter readsA consumed
              writableB := SharedReadPhase.writableAfter readsB consumed }) fun result =>
        match scheduled with
        | ⟨some _outputIndex, _ownership⟩ =>
            KernelM.pure (TileLoopState.accumulator result)
        | ⟨none, _unassigned⟩ => KernelM.pure ()

end GpuDsl
