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

/-!
Build a statically bounded sequence of kernel statements.  This is a Lean
elaborator, not a new DSL primitive: the resulting term contains only `pure`,
`bind`, and the statements produced by `body`.
-/
def foldValueFinM
    (thread : Thread cfg BlockState State)
    (n : Nat) (initial : Value α)
    (body : Fin n → Value α → KernelM thread trace trace (Value α)) :
    KernelM thread trace trace (Value α) :=
  let rec go (i : Nat) (acc : Value α) :
      KernelM thread trace trace (Value α) :=
    if hi : i < n then
      KernelM.bind (body ⟨i, hi⟩ acc) fun next => go (i + 1) next
    else
      KernelM.pure acc
  termination_by n - i
  go 0 initial

/-- The resources that must be available at the top of every tile iteration. -/
structure TileLoopState
    (block : Block cfg BlockState) (trace : SyncTrace) (α : Type) where
  accumulator : Value α
  writable : SharedWritableAt block trace

/--
One tiled matmul kernel for both perfect and predicated launches.

`block.x = block.y = tile` is a launch requirement. All threads execute the
shared-memory phases and barriers, including threads with no scheduled output.
Only returning an output value is conditional on the assignment supplied and
proved by `launch`.
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
  requirements := fun cfg =>
    Dim3.x (LaunchConfig.block cfg) = tile ∧
      Dim3.y (LaunchConfig.block cfg) = tile ∧
      Dim3.z (LaunchConfig.block cfg) = 1
  trace := iterateTrace tiledMatmulStep (ceilDiv inner tile) []
  body := fun {_cfg} {_State} {_certificate}
      ⟨blockXEq, blockYEq, blockZEq⟩ thread scheduled =>
      let threadIndex := Thread.index thread
      let block := Thread.block thread
      let blockIndex := Block.index block
      let threadCol : Fin tile := Fin.cast blockXEq (Coord.x threadIndex)
      let threadRow : Fin tile := Fin.cast blockYEq (Coord.y threadIndex)
      let sharedA := SharedState.get block SharedRef.here
      let sharedB := SharedState.get block (SharedRef.there SharedRef.here)
      let ownership := tileWriteOwnership blockXEq blockYEq blockZEq tilePositive
      let initialState : TileLoopState block [] α :=
        { accumulator := Value.zero
          writable := SharedWritableAt.initial block }

      KernelM.bind
        (KernelM.foldFinD (ceilDiv inner tile)
          (fun trace => TileLoopState block trace α) initialState
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
              (TileLoopState.writable state) valueA) fun writesA =>
          KernelM.bind
            (KernelM.initializeShared sharedB ownership
              (TileLoopState.writable state) valueB) fun writesB =>
          KernelM.bind (KernelM.syncBlock .tileLoaded) fun loaded =>
          KernelM.bind
            (foldValueFinM thread tile (TileLoopState.accumulator state)
              fun k partialSum =>
              let indexA := Layout.offset Layout.rowMajor2D ⟨threadRow, k⟩
              let indexB := Layout.offset Layout.rowMajor2D ⟨k, threadCol⟩
              let definedA :=
                SharedWritePhase.definedAfterLoaded writesA loaded indexA
              let definedB :=
                SharedWritePhase.definedAfterLoaded writesB loaded indexB
              KernelM.bind (KernelM.readShared sharedA indexA definedA) fun av =>
              KernelM.bind (KernelM.readShared sharedB indexB definedB) fun bv =>
              KernelM.pure (Value.add partialSum (Value.mul av bv))) fun nextAccumulator =>
          KernelM.bind (KernelM.syncBlock .tileConsumed) fun consumed =>
          KernelM.pure
            { accumulator := nextAccumulator
              writable := BlockSync.writableAfterConsumed consumed }) fun result =>
        match scheduled with
        | ⟨some _outputIndex, _ownership⟩ =>
            KernelM.pure (TileLoopState.accumulator result)
        | ⟨none, _unassigned⟩ => KernelM.pure ()

end GpuDsl
