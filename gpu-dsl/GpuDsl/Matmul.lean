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

/--
One tiled matmul kernel for both perfect and predicated launches.

`block.x = block.y = tile` is checked uniformly in the body instead of being a
constraint in the signature. All threads execute the shared-memory phases and
barriers, including threads with no scheduled output. Only the final store is
conditional on the assignment supplied and proved by `launch`.
-/
def tiledMatmulKernel
    {α : Type}
    [OfNat α 0]
    [Add α]
    [Mul α]
    (tile : Nat)
    (a : Matrix rows inner (Value α))
    (b : Matrix inner cols (Value α)) :
    Kernel (.d2 rows cols) α :=
  fun {cfg} {_BlockState} {_State} {_certificate} thread scheduled =>
    KernelM.require
      (Dim3.x (LaunchConfig.block cfg) = tile ∧ Dim3.y (LaunchConfig.block cfg) = tile)
      fun blockShape => do
      let threadIndex := Thread.index thread
      let blockIndex := Block.index (Thread.block thread)
      let threadCol : Fin tile := Fin.cast blockShape.1 (Coord.x threadIndex)
      let threadRow : Fin tile := Fin.cast blockShape.2 (Coord.y threadIndex)
      let row := Fin.val (Coord.y blockIndex) * tile + Fin.val threadRow
      let col := Fin.val (Coord.x blockIndex) * tile + Fin.val threadCol
      let sharedA ← KernelM.allocShared "tileA" (tile * tile) α
      let sharedB ← KernelM.allocShared "tileB" (tile * tile) α
      let localIndex := Layout.offset Layout.rowMajor2D ⟨threadRow, threadCol⟩

      let result ← KernelM.foldFin (ceilDiv inner tile) Value.zero fun kTile acc => do
        let kA := Fin.val kTile * tile + Fin.val threadCol
        let kB := Fin.val kTile * tile + Fin.val threadRow

        let aValue ← if hr : row < rows then
          if hk : kA < inner then pure (a ⟨⟨row, hr⟩, ⟨kA, hk⟩⟩)
          else pure Value.zero
        else pure Value.zero
        let bValue ← if hk : kB < inner then
          if hc : col < cols then pure (b ⟨⟨kB, hk⟩, ⟨col, hc⟩⟩)
          else pure Value.zero
        else pure Value.zero

        KernelM.store sharedA localIndex aValue
        KernelM.store sharedB localIndex bValue
        KernelM.barrier

        let acc ← KernelM.foldFin tile acc fun k acc => do
          let av := Buffer.read sharedA (Layout.offset Layout.rowMajor2D ⟨threadRow, k⟩)
          let bv := Buffer.read sharedB (Layout.offset Layout.rowMajor2D ⟨k, threadCol⟩)
          pure (Value.add acc (Value.mul av bv))

        KernelM.barrier
        pure acc

      match scheduled with
      | ⟨some _outputIndex, _ownership⟩ => pure result
      | ⟨none, _unassigned⟩ => pure ()

end GpuDsl
