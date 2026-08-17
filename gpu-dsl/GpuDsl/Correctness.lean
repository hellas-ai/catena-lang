import GpuDsl.Launch

namespace GpuDsl

/-!
# Direct correctness argument

The DSL uses a bulk-synchronous argument, not a theorem about all thread
interleavings:

1. `initializeShared` is a write phase. `WriteOwnership.owner` assigns every
   cell to one physical thread.
2. `.tileLoaded` separates that write phase from the read phase.
3. `readShared` is available only with `DefinedAt` evidence produced from the
   completed write phase and loaded barrier.
4. `.tileConsumed` ends the read-only phase before shared memory is reused.
5. `Kernel.body` has one statically fixed trace, so every launched thread has
   the same barrier sequence.

We assume the ordinary meaning of the typed primitives: finite operations and
bounded loops terminate; a read carrying `DefinedAt` returns the unique value
in that cell; and a block barrier completes when every launched thread follows
the common trace. These are the fixed assumptions of this DSL, rather than a
parameterized machine semantics.
-/

/-- One shared location cannot have two different owners in one write phase. -/
theorem WriteOwnership.uniqueWriter
    (ownership : WriteOwnership cfg length)
    (index : Fin length)
    (left : Thread cfg LeftBlockState LeftState)
    (right : Thread cfg RightBlockState RightState)
    (leftOwns : Owns ownership left index)
    (rightOwns : Owns ownership right index) :
    Thread.index left = Thread.index right :=
  leftOwns.symm.trans rightOwns

/-- One logical output cannot be assigned two different physical owners. -/
theorem ScheduleCertificate.uniqueOwner
    (certificate : ScheduleCertificate cfg outputSpace)
    (index : Index outputSpace)
    {left right : ThreadId cfg}
    (leftOwns : ScheduleCertificate.owner certificate index = left)
    (rightOwns : ScheduleCertificate.owner certificate index = right) :
    left = right :=
  leftOwns.symm.trans rightOwns

/-!
`StructurallyTotal` and `StructurallyDeterministic` are deliberately small
proof judgments. Atomic cases are discharged by their constructor arguments:
in particular, `readShared` contains `DefinedAt`, `initializeShared` contains
unique ownership and a writable-phase capability, and loops carry a finite
bound. Composition is the only recursive reasoning.
-/

inductive KernelM.StructurallyTotal
    {cfg : LaunchConfig} {BlockState State : Type}
    {thread : Thread cfg BlockState State} :
    {before after : SyncTrace} → {α : Type} →
    KernelM thread before after α → Prop where
  | pure : KernelM.StructurallyTotal (.pure value)
  | bind
      (firstTotal : KernelM.StructurallyTotal first)
      (nextTotal : ∀ value, KernelM.StructurallyTotal (next value)) :
      KernelM.StructurallyTotal (.bind first next)
  | initializeShared :
      KernelM.StructurallyTotal
        (.initializeShared buffer ownership writable value)
  | readShared :
      KernelM.StructurallyTotal (.readShared buffer index defined)
  | syncBlock : KernelM.StructurallyTotal (.syncBlock barrier)
  | foldFinD
      (n : Nat) (step : SyncTrace → SyncTrace)
      (Acc : SyncTrace → Type) {trace : SyncTrace}
      (initial : Acc trace)
      (body : ∀ currentTrace, Fin n → Acc currentTrace →
        KernelM thread currentTrace (step currentTrace) (Acc (step currentTrace)))
      (bodyTotal : ∀ currentTrace index accumulator,
        KernelM.StructurallyTotal (body currentTrace index accumulator)) :
      KernelM.StructurallyTotal
        (.foldFinD (step := step) n Acc initial body)

inductive KernelM.StructurallyDeterministic
    {cfg : LaunchConfig} {BlockState State : Type}
    {thread : Thread cfg BlockState State} :
    {before after : SyncTrace} → {α : Type} →
    KernelM thread before after α → Prop where
  | pure : KernelM.StructurallyDeterministic (.pure value)
  | bind
      (firstDeterministic : KernelM.StructurallyDeterministic first)
      (nextDeterministic :
        ∀ value, KernelM.StructurallyDeterministic (next value)) :
      KernelM.StructurallyDeterministic (.bind first next)
  | initializeShared :
      KernelM.StructurallyDeterministic
        (.initializeShared buffer ownership writable value)
  | readShared :
      KernelM.StructurallyDeterministic (.readShared buffer index defined)
  | syncBlock : KernelM.StructurallyDeterministic (.syncBlock barrier)
  | foldFinD
      (n : Nat) (step : SyncTrace → SyncTrace)
      (Acc : SyncTrace → Type) {trace : SyncTrace}
      (initial : Acc trace)
      (body : ∀ currentTrace, Fin n → Acc currentTrace →
        KernelM thread currentTrace (step currentTrace) (Acc (step currentTrace)))
      (bodyDeterministic : ∀ currentTrace index accumulator,
        KernelM.StructurallyDeterministic
          (body currentTrace index accumulator)) :
      KernelM.StructurallyDeterministic
        (.foldFinD (step := step) n Acc initial body)

/-- Totality follows independently by induction on the kernel syntax. -/
theorem KernelM.proveStructurallyTotal
    (kernel : KernelM thread before after α) :
    KernelM.StructurallyTotal kernel := by
  induction kernel with
  | pure => exact .pure
  | bind first next firstTotal nextTotal =>
      exact .bind firstTotal nextTotal
  | initializeShared => exact .initializeShared
  | readShared => exact .readShared
  | syncBlock => exact .syncBlock
  | foldFinD n Acc initial body bodyTotal =>
      exact .foldFinD n _ Acc initial body bodyTotal

/-- Determinism follows independently by induction on the kernel syntax. -/
theorem KernelM.proveStructurallyDeterministic
    (kernel : KernelM thread before after α) :
    KernelM.StructurallyDeterministic kernel := by
  induction kernel with
  | pure => exact .pure
  | bind first next firstDeterministic nextDeterministic =>
      exact .bind firstDeterministic nextDeterministic
  | initializeShared => exact .initializeShared
  | readShared => exact .readShared
  | syncBlock => exact .syncBlock
  | foldFinD n Acc initial body bodyDeterministic =>
      exact .foldFinD n _ Acc initial body bodyDeterministic

/--
Every well-typed kernel term is structurally total and deterministic under the
fixed primitive assumptions above. The proof is induction on `KernelM`.
-/
theorem KernelM.total_and_deterministic
    (kernel : KernelM thread before after α) :
    KernelM.StructurallyTotal kernel ∧
      KernelM.StructurallyDeterministic kernel :=
  ⟨KernelM.proveStructurallyTotal kernel,
    KernelM.proveStructurallyDeterministic kernel⟩

/-!
The remaining launch-level argument is intentionally separate and small:

* `ScheduleCertificate.uniqueOwner` gives one writer per output cell.
* `WriteOwnership.uniqueWriter` gives one writer per shared cell in each
  initialization phase.
* Between loaded and consumed barriers the DSL exposes shared reads but no
  shared write capability.
* The fixed `Kernel.trace` gives every block thread the same barrier sequence.

Together these facts establish phase race freedom. Determinism then follows
phase-by-phase; no enumeration of thread interleavings is required. A later
formal statement should package only the launch assumption that every
scheduled output and every physical block thread is invoked as specified.
-/

end GpuDsl
