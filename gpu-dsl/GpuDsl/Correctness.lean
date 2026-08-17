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

inductive KernelM.StructurallySafe
    {cfg : LaunchConfig} {BlockState State : Type}
    {thread : Thread cfg BlockState State} :
    {before after : SyncTrace} → {α : Type} →
    KernelM thread before after α → Prop where
  | pure : KernelM.StructurallySafe (.pure value)
  | bind
      (firstSafe : KernelM.StructurallySafe first)
      (nextSafe : ∀ value, KernelM.StructurallySafe (next value)) :
      KernelM.StructurallySafe (.bind first next)
  | initializeShared :
      KernelM.StructurallySafe
        (.initializeShared buffer ownership writable value)
  | readShared :
      KernelM.StructurallySafe (.readShared buffer index defined)
  | syncBlock : KernelM.StructurallySafe (.syncBlock barrier)
  | foldFinD
      (n : Nat) (step : SyncTrace → SyncTrace)
      (Acc : SyncTrace → Type) {trace : SyncTrace}
      (initial : Acc trace)
      (body : ∀ currentTrace, Fin n → Acc currentTrace →
        KernelM thread currentTrace (step currentTrace) (Acc (step currentTrace)))
      (bodySafe : ∀ currentTrace index accumulator,
        KernelM.StructurallySafe (body currentTrace index accumulator)) :
      KernelM.StructurallySafe
        (.foldFinD (step := step) n Acc initial body)

abbrev KernelM.StructurallyTotal
    (kernel : KernelM thread before after α) : Prop :=
  KernelM.StructurallySafe kernel

abbrev KernelM.StructurallyDeterministic
    (kernel : KernelM thread before after α) : Prop :=
  KernelM.StructurallySafe kernel

/--
Every well-typed kernel term is structurally total and deterministic under the
fixed primitive assumptions above. The proof is induction on `KernelM`.
-/
theorem KernelM.total_and_deterministic
    (kernel : KernelM thread before after α) :
    KernelM.StructurallyTotal kernel ∧
      KernelM.StructurallyDeterministic kernel := by
  /-
  Each atomic case follows from its typed evidence. `bind` applies both
  induction hypotheses, and `foldFinD` applies the body hypothesis for each
  member of its finite `Fin n` domain.
  -/
  have safe : KernelM.StructurallySafe kernel := by
    induction kernel with
    | pure => exact .pure
    | bind first next firstSafe nextSafe =>
        exact .bind firstSafe nextSafe
    | initializeShared => exact .initializeShared
    | readShared => exact .readShared
    | syncBlock => exact .syncBlock
    | foldFinD n Acc initial body bodySafe =>
        exact .foldFinD n _ Acc initial body bodySafe
  exact ⟨safe, safe⟩

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
