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

/--
One tiled matmul kernel for both perfect and predicated launches.

`block.x = block.y = tile` is a launch requirement. All threads execute the
shared-memory phases and barriers, including threads with no owned outputs.
The returned function supplies the accumulator for every output owned by the
current thread; edge threads may own no outputs.
-/
def tiledMatmulKernel
    {α : Type}
    [OfNat α 0]
    [Add α]
    [Mul α]
    (tile : Nat)
    (_tilePositive : 0 < tile)
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
      ⟨blockXEq, blockYEq, _blockZEq⟩ _launchFacts thread _work =>
      let threadIndex := Thread.index thread
      let block := Thread.block thread
      let blockIndex := Block.index block
      let threadCol : Fin tile := Fin.cast blockXEq (Coord.x threadIndex)
      let threadRow : Fin tile := Fin.cast blockYEq (Coord.y threadIndex)
      let sharedA := SharedState.get block "tileA"
      let sharedB := SharedState.get block "tileB"
      let tileIndex : Fin (tile * tile) :=
        Layout.offset Layout.rowMajor2D ⟨threadRow, threadCol⟩

      KernelM.bind
        (KernelM.foldFinD (ceilDiv inner tile)
          (fun _trace => Value α) Value.zero
          fun _trace kTile accumulator =>
          let valueA :=
            let row := Fin.val (Coord.y blockIndex) * tile + Fin.val threadRow
            let kA := Fin.val kTile * tile + Fin.val threadCol
            if rowInBounds : row < rows then
              if kAInBounds : kA < inner then
                a ⟨⟨row, rowInBounds⟩, ⟨kA, kAInBounds⟩⟩
              else Value.zero
            else Value.zero
          let valueB :=
            let kB := Fin.val kTile * tile + Fin.val threadRow
            let col := Fin.val (Coord.x blockIndex) * tile + Fin.val threadCol
            if kBInBounds : kB < inner then
              if colInBounds : col < cols then
                b ⟨⟨kB, kBInBounds⟩, ⟨col, colInBounds⟩⟩
              else Value.zero
            else Value.zero
          KernelM.bind
            (KernelM.writeShared sharedA tileIndex valueA) fun _ =>
          KernelM.bind
            (KernelM.writeShared sharedB tileIndex valueB) fun _ =>
          KernelM.bind (KernelM.syncBlock .tileLoaded) fun _ =>
          KernelM.bind
            (foldValueFinM thread tile accumulator
              fun k partialSum =>
              let indexA := Layout.offset Layout.rowMajor2D ⟨threadRow, k⟩
              let indexB := Layout.offset Layout.rowMajor2D ⟨k, threadCol⟩
              KernelM.bind (KernelM.readShared sharedA indexA) fun av =>
              KernelM.bind (KernelM.readShared sharedB indexB) fun bv =>
              KernelM.pure (Value.add partialSum (Value.mul av bv))) fun nextAccumulator =>
          KernelM.bind (KernelM.syncBlock .tileConsumed) fun _ =>
          KernelM.pure nextAccumulator) fun result =>
        KernelM.pure (fun _index _member => result)

end GpuDsl
