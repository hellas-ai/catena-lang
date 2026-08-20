import GpuDsl.Matrix
import GpuDsl.Shared

namespace GpuDsl

/--
A constructive scheduling certificate. Every logical cell has exactly one
owner because `owner` is a function. One thread may own zero, one, or many
cells.
-/
structure ScheduleCertificate (cfg : LaunchConfig) (space : Space) where
  owner : Index space → ThreadId cfg

/-- A closed description of which global resources a kernel may access. -/
inductive GlobalAccessPlan (cfg : LaunchConfig) : Type 1 where
  | none : GlobalAccessPlan cfg
  | reads {α : Type} {length : Nat}
      (buffer : Buffer .global length α)
      (region : ThreadId cfg → Fin length → Prop) : GlobalAccessPlan cfg
  | owns {α : Type} {length : Nat} {space : Space}
      (buffer : Buffer .global length α)
      (layout : Index space → Fin length)
      (layoutInjective : Injective layout)
      (owner : Index space → ThreadId cfg) : GlobalAccessPlan cfg
  | and (left right : GlobalAccessPlan cfg) : GlobalAccessPlan cfg

/-- Cells read by a physical thread according to a global access plan. -/
def GlobalAccessPlan.ReadsAt :
    GlobalAccessPlan cfg → ThreadId cfg → AllocationId → Nat → Prop
  | .none, _, _, _ => False
  | .reads buffer region, threadId, allocation, offset =>
      buffer.allocation = allocation ∧
        ∃ index, region threadId index ∧ index.val = offset
  | .owns _ _ _ _, _, _, _ => False
  | .and left right, threadId, allocation, offset =>
      left.ReadsAt threadId allocation offset ∨
        right.ReadsAt threadId allocation offset

/-- Cells written by a physical thread according to a global access plan. -/
def GlobalAccessPlan.WritesAt :
    GlobalAccessPlan cfg → ThreadId cfg → AllocationId → Nat → Prop
  | .none, _, _, _ => False
  | .reads _ _, _, _, _ => False
  | .owns buffer layout _ owner, threadId, allocation, offset =>
      buffer.allocation = allocation ∧
        ∃ cell, owner cell = threadId ∧ (layout cell).val = offset
  | .and left right, threadId, allocation, offset =>
      left.WritesAt threadId allocation offset ∨
        right.WritesAt threadId allocation offset

/--
No physical global cell has two different writers, or both a writer and a
reader. Accesses to different allocations or offsets remain independent.
-/
def GlobalAccessPlan.RaceFree (plan : GlobalAccessPlan cfg) : Prop :=
  (∀ allocation offset first second,
      plan.WritesAt first allocation offset →
      plan.WritesAt second allocation offset → first = second) ∧
  (∀ allocation offset writer reader,
      plan.WritesAt writer allocation offset →
      plan.ReadsAt reader allocation offset → False)

/--
The complete set of cells in one global buffer owned by a physical thread.
`launch` constructs this value from an `owns` access plan.
-/
structure OwnedGlobalCells {cfg : LaunchConfig} {α Cell : Type}
    {length : Nat} (threadId : ThreadId cfg)
    (buffer : Buffer .global length α)
    (layout : Cell → Fin length) (owner : Cell → ThreadId cfg) where
  private mk ::
  items : List Cell
  nodup : List.Nodup items
  complete : ∀ cell, List.Mem cell items ↔ owner cell = threadId
  owns : ∀ cell, List.Mem cell items →
    GlobalOwns threadId buffer (layout cell)

/-- Interpret a global access plan as the permissions received by one thread. -/
def GlobalAccessPlan.Permissions :
    GlobalAccessPlan cfg → ThreadId cfg → Type
  | .none, _ => Unit
  | .reads buffer region, threadId =>
      GlobalReadShare threadId buffer (region threadId)
  | .owns buffer layout _ owner, threadId =>
      OwnedGlobalCells threadId buffer layout owner
  | .and left right, threadId =>
      left.Permissions threadId × right.Permissions threadId

/--
Facts supplied by the implementation of `launch` to the kernel proof context.

They express the fairness guarantees expected of any correct GPU
implementation: every physical thread is invoked exactly once, and every
invocation executes the kernel's exact common synchronization trace.  The
functions record the implementation's observations; the two proofs state the
contract that those observations must satisfy.
-/
structure LaunchFacts (cfg : LaunchConfig) (trace : SyncTrace) where
  invocationCount : ThreadId cfg → Nat
  completedTrace : ThreadId cfg → SyncTrace
  everyThreadRunsExactlyOnce : ∀ threadId, invocationCount threadId = 1
  everyThreadHasExactTrace : ∀ threadId, completedTrace threadId = trace

/-- The checked ownership service supplied by `launch` to one thread. -/
structure LaunchOwnership {cfg : LaunchConfig} {BlockState State : Type}
    (plan : OwnershipPlan cfg) (α : Type)
    (thread : Thread cfg BlockState State) where
  grantWrite :
    ∀ {name : String} {n : Nat}
      (buffer : SharedBuffer (Thread.block thread) name n α)
      (layout : plan.Cell → Fin n),
      Injective layout →
      (cell : plan.Cell) →
      plan.owner cell = Thread.index thread →
      Owns [] thread buffer (layout cell)

/--
A kernel accepts an arbitrary input value. Its global access plan refers to the
actual resources inside that value, and `launch` supplies the corresponding
permissions for the current physical thread.
-/
structure Kernel (Inputs SharedScalar : Type) where
  /-- Shared buffers allocated once per block by `launch`. -/
  shared : SharedSpec
  /-- Static launch facts required by this kernel. -/
  requirements : LaunchConfig → Prop
  /-- Global reads and ownership allowed for these particular inputs. -/
  globalAccess : ∀ cfg, requirements cfg → Inputs → GlobalAccessPlan cfg
  /-- The global plan has no overlapping write/write or write/read regions. -/
  globalAccessRaceFree : ∀ cfg (proof : requirements cfg) (inputs : Inputs),
    (globalAccess cfg proof inputs).RaceFree
  /-- The cell-to-thread assignment realized by `launch`. -/
  sharedOwner : ∀ cfg, requirements cfg → OwnershipPlan cfg
  /-- The exact synchronization trace followed by every thread. -/
  trace : SyncTrace
  body :
    ∀ {cfg : LaunchConfig} {State : Type},
      (requirementsProof : requirements cfg) →
      LaunchFacts cfg trace →
      (inputs : Inputs) →
      (thread : Thread cfg (SharedState SharedScalar shared) State) →
      LaunchOwnership (sharedOwner cfg requirementsProof) SharedScalar thread →
      GlobalAccessPlan.Permissions
        (globalAccess cfg requirementsProof inputs) thread.id →
      KernelM (sharedOwner cfg requirementsProof) SharedScalar thread [] trace Unit

/-- An opaque launch token. Execution belongs to the eventual backend. -/
structure Launch where
  private token : Unit

/--
CUDA/HIP-shaped primitive. Its implementation creates block/thread state,
allocates `Kernel.shared` once per block, interprets `Kernel.globalAccess`, and
invokes `Kernel.body` exactly once per physical thread with permissions tied to
the actual `inputs`. No implementation is given here.
-/
axiom launch
    (cfg : LaunchConfig)
    (kernel : Kernel Inputs SharedScalar)
    (inputs : Inputs)
    (_requirements : kernel.requirements cfg) :
    Launch

end GpuDsl
