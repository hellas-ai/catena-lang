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

/-- Arbitrary global resources consumed by the tiled matmul kernel. -/
structure MatmulInputs (rows inner cols : Nat) (α : Type) where
  a : Buffer .global (rows * inner) α
  aLayout : Layout (.d2 rows inner) (rows * inner)
  b : Buffer .global (inner * cols) α
  bLayout : Layout (.d2 inner cols) (inner * cols)
  output : Buffer .global (rows * cols) α
  aOutputDistinct : a.allocation ≠ output.allocation
  bOutputDistinct : b.allocation ≠ output.allocation

/-- The exact A cells loaded by one physical thread across all tile rounds. -/
def matmulAReadRegion (tile : Nat)
    (inputs : MatmulInputs rows inner cols α)
    (threadId : ThreadId cfg) (index : Fin (rows * inner)) : Prop :=
  let row := threadId.blockIdx.y.val * tile + threadId.threadIdx.y.val
  ∃ kTile : Fin (ceilDiv inner tile),
    let col := kTile.val * tile + threadId.threadIdx.x.val
    ∃ rowInBounds : row < rows, ∃ colInBounds : col < inner,
      index = inputs.aLayout.offset
        ⟨⟨row, rowInBounds⟩, ⟨col, colInBounds⟩⟩

/-- The exact B cells loaded by one physical thread across all tile rounds. -/
def matmulBReadRegion (tile : Nat)
    (inputs : MatmulInputs rows inner cols α)
    (threadId : ThreadId cfg) (index : Fin (inner * cols)) : Prop :=
  let col := threadId.blockIdx.x.val * tile + threadId.threadIdx.x.val
  ∃ kTile : Fin (ceilDiv inner tile),
    let row := kTile.val * tile + threadId.threadIdx.y.val
    ∃ rowInBounds : row < inner, ∃ colInBounds : col < cols,
      index = inputs.bLayout.offset
        ⟨⟨row, rowInBounds⟩, ⟨col, colInBounds⟩⟩

private def tileThread
    (blockXEq : Dim3.x (LaunchConfig.block cfg) = tile)
    (blockYEq : Dim3.y (LaunchConfig.block cfg) = tile)
    (blockZEq : Dim3.z (LaunchConfig.block cfg) = 1)
    (row col : Fin tile) : Coord (LaunchConfig.block cfg) :=
  { x := Fin.cast blockXEq.symm col
    y := Fin.cast blockYEq.symm row
    z := Fin.cast blockZEq.symm ⟨0, by decide⟩ }

private def tiledOwnershipPlan
    (blockXEq : Dim3.x (LaunchConfig.block cfg) = tile)
    (blockYEq : Dim3.y (LaunchConfig.block cfg) = tile)
    (blockZEq : Dim3.z (LaunchConfig.block cfg) = 1) :
    OwnershipPlan cfg where
  Cell := Index (.d2 tile tile)
  owner := fun cell =>
    tileThread blockXEq blockYEq blockZEq cell.row cell.col

/-!
Build a statically bounded sequence of kernel statements.  This is a Lean
elaborator, not a new DSL primitive: the resulting term contains only `pure`,
`bind`, and the statements produced by `body`.
-/
def foldValueFinM
    (plan : OwnershipPlan cfg)
    (thread : Thread cfg BlockState State)
    (trace : SyncTrace)
    (n : Nat) (initial : Value α)
    (body : Fin n → Value α →
      KernelM plan α thread trace trace (Value α)) :
    KernelM plan α thread trace trace (Value α) :=
  let rec go (i : Nat) (acc : Value α) :
      KernelM plan α thread trace trace (Value α) :=
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
Global inputs and their per-thread permissions are supplied by `launch`.
Edge threads own no output cells but still participate in every barrier.
-/
def tiledMatmulKernel
    {α : Type}
    [OfNat α 0]
    [Add α]
    [Mul α]
    (tile : Nat)
    (tilePositive : 0 < tile) :
    Kernel (MatmulInputs rows inner cols α) α where
  shared := .buffer "tileA" (tile * tile) (.buffer "tileB" (tile * tile) .empty)
  requirements := fun cfg =>
    Dim3.x (LaunchConfig.block cfg) = tile ∧
      Dim3.y (LaunchConfig.block cfg) = tile ∧
      Dim3.z (LaunchConfig.block cfg) = 1 ∧
      cols ≤ Dim3.x (LaunchConfig.grid cfg) * tile ∧
      rows ≤ Dim3.y (LaunchConfig.grid cfg) * tile ∧
      0 < Dim3.z (LaunchConfig.grid cfg)
  globalAccess := fun cfg
      ⟨blockXEq, blockYEq, blockZEq, coversCols, coversRows, gridZPositive⟩
      inputs =>
    let certificate := tiledScheduleCertificate cfg rows cols tile tilePositive
      blockXEq blockYEq coversCols coversRows gridZPositive (by simp [blockZEq])
    .and (.reads inputs.a (matmulAReadRegion tile inputs))
      (.and (.reads inputs.b (matmulBReadRegion tile inputs))
        (.owns inputs.output Layout.rowMajor2D.offset
          Layout.rowMajor2D_injective certificate.owner))
  globalAccessRaceFree := fun cfg
      ⟨blockXEq, blockYEq, blockZEq, coversCols, coversRows, gridZPositive⟩
      inputs => by
    dsimp [GlobalAccessPlan.RaceFree]
    constructor
    · intro allocation offset first second firstWrites secondWrites
      simp only [GlobalAccessPlan.WritesAt, false_or] at firstWrites secondWrites
      rcases firstWrites with ⟨_firstAllocation, firstCell,
        firstOwner, firstOffset⟩
      rcases secondWrites with ⟨_secondAllocation, secondCell,
        secondOwner, secondOffset⟩
      have equalOffsets :
          Layout.rowMajor2D.offset firstCell =
            Layout.rowMajor2D.offset secondCell :=
        Fin.ext (firstOffset.trans secondOffset.symm)
      have equalCells := Layout.rowMajor2D_injective equalOffsets
      cases equalCells
      exact firstOwner.symm.trans secondOwner
    · intro allocation offset writer reader writerWrites readerReads
      simp only [GlobalAccessPlan.WritesAt, false_or] at writerWrites
      simp only [GlobalAccessPlan.ReadsAt, false_or, or_false] at readerReads
      rcases writerWrites with ⟨outputAllocation, _cell, _owner, _offset⟩
      rcases readerReads with readA | readB
      · exact inputs.aOutputDistinct (readA.1.trans outputAllocation.symm)
      · exact inputs.bOutputDistinct (readB.1.trans outputAllocation.symm)
  sharedOwner := fun _cfg
      ⟨blockXEq, blockYEq, blockZEq, _coversCols, _coversRows, _gridZPositive⟩ =>
    tiledOwnershipPlan blockXEq blockYEq blockZEq
  trace := iterateTrace tiledMatmulStep (ceilDiv inner tile) []
  body := fun {_cfg} {_State}
      ⟨blockXEq, blockYEq, blockZEq, coversCols, coversRows, gridZPositive⟩
      _launchFacts inputs thread ownership globalPermissions => by
      let readGlobalA := globalPermissions.1
      let readGlobalB := globalPermissions.2.1
      let ownedOutput := globalPermissions.2.2
      let threadIndex := Thread.index thread
      let block := Thread.block thread
      let blockIndex := Block.index block
      let threadCol : Fin tile := Fin.cast blockXEq (Coord.x threadIndex)
      let threadRow : Fin tile := Fin.cast blockYEq (Coord.y threadIndex)
      let sharedA := SharedState.get (length := tile * tile) block "tileA"
      let sharedB := SharedState.get (length := tile * tile) block "tileB"
      let cellAt := fun other : Coord _cfg.block =>
        (⟨Fin.cast blockYEq (Coord.y other),
          Fin.cast blockXEq (Coord.x other)⟩ : Index (.d2 tile tile))
      let indexAt := fun other => Layout.offset Layout.rowMajor2D (cellAt other)
      let tileCell : Index (.d2 tile tile) := cellAt threadIndex
      let tileIndex : Fin (tile * tile) := indexAt threadIndex
      let plan := tiledOwnershipPlan blockXEq blockYEq blockZEq
      let tileOwner := plan.owner
      have currentOwnsCell : tileOwner tileCell = threadIndex := by
        apply Coord.ext
        · simp [tileOwner, plan, tiledOwnershipPlan, tileCell, cellAt, tileThread,
            threadCol, threadIndex]
        · simp [tileOwner, plan, tiledOwnershipPlan, tileCell, cellAt, tileThread,
            threadRow, threadIndex]
        · apply Fin.ext
          have equalCasts := Subsingleton.elim
            (Fin.cast blockZEq (tileOwner tileCell).z)
            (Fin.cast blockZEq threadIndex.z)
          exact congrArg (fun z : Fin 1 => z.val) equalCasts
      let tileInvariant := fun (trace : SyncTrace) (other : Thread _cfg _ _State) =>
        Owns trace other sharedA (indexAt (Thread.index other)) ×
        Owns trace other sharedB (indexAt (Thread.index other))
      let publishWrites : SyncSpec (α := α) plan thread :=
        { barrier := .tileLoaded
          grants := .reads (.and (.one sharedA) (.one sharedB)) }
      let finishReads : SyncSpec (α := α) plan thread :=
        { barrier := .tileConsumed
          grants := .owns (.and
            (.one sharedA Layout.rowMajor2D.offset
              Layout.rowMajor2D_injective tileCell currentOwnsCell)
            (.one sharedB Layout.rowMajor2D.offset
              Layout.rowMajor2D_injective tileCell currentOwnsCell)) }

      let initialOwnsA : Owns [] thread sharedA tileIndex :=
        ownership.grantWrite sharedA Layout.rowMajor2D.offset
          Layout.rowMajor2D_injective tileCell currentOwnsCell
      let initialOwnsB : Owns [] thread sharedB tileIndex :=
        ownership.grantWrite sharedB Layout.rowMajor2D.offset
          Layout.rowMajor2D_injective tileCell currentOwnsCell
      let initialInvariant : tileInvariant [] thread :=
        ⟨initialOwnsA, initialOwnsB⟩

      exact KernelM.bind
        (KernelM.foldFinD (ceilDiv inner tile)
          tiledMatmulStep []
          (fun trace => tileInvariant trace thread)
          initialInvariant
          Value.zero
          fun currentTrace kTile partialSum invariant =>
          let loadA : KernelM plan α thread currentTrace currentTrace (Value α) :=
            let row := Fin.val (Coord.y blockIndex) * tile + Fin.val threadRow
            let kA := Fin.val kTile * tile + Fin.val threadCol
            if rowInBounds : row < rows then
              if kAInBounds : kA < inner then
                KernelM.readGlobal inputs.a
                  (Layout.offset inputs.aLayout
                    ⟨⟨row, rowInBounds⟩, ⟨kA, kAInBounds⟩⟩)
                  readGlobalA (by
                    exact ⟨kTile, rowInBounds, kAInBounds, rfl⟩)
              else KernelM.pure Value.zero
            else KernelM.pure Value.zero
          let loadB : KernelM plan α thread currentTrace currentTrace (Value α) :=
            let kB := Fin.val kTile * tile + Fin.val threadRow
            let col := Fin.val (Coord.x blockIndex) * tile + Fin.val threadCol
            if kBInBounds : kB < inner then
              if colInBounds : col < cols then
                KernelM.readGlobal inputs.b
                  (Layout.offset inputs.bLayout
                    ⟨⟨kB, kBInBounds⟩, ⟨col, colInBounds⟩⟩)
                  readGlobalB (by
                    exact ⟨kTile, kBInBounds, colInBounds, rfl⟩)
              else KernelM.pure Value.zero
            else KernelM.pure Value.zero
          let barrierFacts := fun factTrace other =>
            WroteSome α factTrace other sharedA (indexAt (Thread.index other)) ×
            WroteSome α factTrace other sharedB (indexAt (Thread.index other))
          KernelM.bind
            loadA fun valueA =>
          KernelM.bind
            loadB fun valueB =>
          KernelM.bind
            (KernelM.writeShared sharedA tileIndex valueA invariant.1) fun wroteA =>
          KernelM.bind
            (KernelM.writeShared sharedB tileIndex valueB invariant.2) fun wroteB =>
          let currentWrites : barrierFacts currentTrace thread :=
            (⟨valueA, wroteA⟩, ⟨valueB, wroteB⟩)
          KernelM.bind
            (KernelM.syncBlock block rfl publishWrites
              (facts := barrierFacts) currentWrites) fun ⟨allWrites, readShares⟩ =>
          let readA : ReadShare (currentTrace ++ [.tileLoaded]) thread sharedA := by
            simpa [publishWrites, BarrierGrants.Result, ReadGrants.Result] using
              readShares.1
          let readB : ReadShare (currentTrace ++ [.tileLoaded]) thread sharedB := by
            simpa [publishWrites, BarrierGrants.Result, ReadGrants.Result] using
              readShares.2
          KernelM.bind
            (foldValueFinM plan thread (currentTrace ++ [.tileLoaded])
              tile partialSum
              fun k sumWithinTile =>
              let indexA := Layout.offset Layout.rowMajor2D ⟨threadRow, k⟩
              let indexB := Layout.offset Layout.rowMajor2D ⟨k, threadCol⟩
              KernelM.bind (KernelM.readShared sharedA indexA (by
                let writerIndex := tileThread blockXEq blockYEq blockZEq threadRow k
                let writer := { thread with index := writerIndex }
                apply WroteSome.defined .tileLoaded (allWrites writer rfl).1
                simp [indexAt, cellAt, writer, writerIndex, indexA, tileThread])
                readA) fun av =>
              KernelM.bind (KernelM.readShared sharedB indexB (by
                let writerIndex := tileThread blockXEq blockYEq blockZEq k threadCol
                let writer := { thread with index := writerIndex }
                apply WroteSome.defined .tileLoaded (allWrites writer rfl).2
                simp [indexAt, cellAt, writer, writerIndex, indexB, tileThread])
                readB) fun bv =>
              KernelM.pure (Value.add sumWithinTile (Value.mul av bv)))
              fun sumAfterTile =>
          KernelM.bind
            (KernelM.syncBlock block rfl finishReads
              (facts := fun _factTrace _other => Unit) ()) fun ⟨_, nextOwnership⟩ =>
          let nextInvariant : tileInvariant (tiledMatmulStep currentTrace) thread := by
            simpa [tileInvariant, tiledMatmulStep, finishReads,
              BarrierGrants.Result, OwnGrants.Result] using nextOwnership
          KernelM.pure (sumAfterTile, nextInvariant)) fun result =>
        KernelM.forEach ownedOutput.items fun outputIndex member =>
          KernelM.bind
            (KernelM.writeGlobal inputs.output
              (Layout.offset Layout.rowMajor2D outputIndex) result
              (ownedOutput.owns outputIndex member)) fun _wroteOutput =>
          KernelM.pure ()

end GpuDsl
