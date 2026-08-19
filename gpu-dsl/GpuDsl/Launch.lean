import GpuDsl.Matrix
import GpuDsl.Shared

namespace GpuDsl

/--
A constructive scheduling certificate. Every logical output index has exactly
one owner because `owner` is a function. One thread may own zero, one, or many
indices.
-/
structure ScheduleCertificate (cfg : LaunchConfig) (outputSpace : Space) where
  owner : Index outputSpace → ThreadId cfg

/--
The complete, duplicate-free list of logical outputs owned by one physical
thread. `launch` constructs one such value for every thread.
-/
structure OwnedOutputs
    (certificate : ScheduleCertificate cfg outputSpace)
    (threadId : ThreadId cfg) where
  items : List (Index outputSpace)
  nodup : List.Nodup items
  complete : ∀ index,
    List.Mem index items ↔ ScheduleCertificate.owner certificate index = threadId

/-- One result for every logical output in a thread's complete work list. -/
def OwnedOutputs.Result
    (work : OwnedOutputs certificate threadId) (α : Type) : Type :=
  ∀ index, List.Mem index (OwnedOutputs.items work) → Value α

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
A generic kernel depends only on its logical output space. Launch geometry,
block state, thread state, and the schedule certificate are supplied by the
`launch` implementation through the thread and its complete owned-output list.
-/
structure Kernel (outputSpace : Space) (α : Type) where
  /-- Shared buffers allocated once per block by `launch`. -/
  shared : SharedSpec
  /-- Static launch facts required by this kernel. -/
  requirements : LaunchConfig → Prop
  /-- The cell-to-thread assignment realized by `launch`. -/
  sharedOwner : ∀ cfg, requirements cfg → OwnershipPlan cfg
  /-- The exact synchronization trace followed by every thread. -/
  trace : SyncTrace
  body :
    ∀ {cfg : LaunchConfig} {State : Type}
      {certificate : ScheduleCertificate cfg outputSpace},
      (requirementsProof : requirements cfg) →
      LaunchFacts cfg trace →
      (thread : Thread cfg (SharedState α shared) State) →
      LaunchOwnership (sharedOwner cfg requirementsProof) α thread →
      (work : OwnedOutputs certificate (Thread.id thread)) →
      KernelM (sharedOwner cfg requirementsProof) α thread [] trace
        (OwnedOutputs.Result work α)

/-- An opaque launch token. Execution belongs to the eventual backend. -/
structure Launch (outputLength : Nat) (α : Type) where
  private token : Unit

/--
CUDA/HIP-shaped primitive. Its implementation creates block/thread state,
allocates `Kernel.shared` once per block, constructs each thread's ownership
grants and complete `OwnedOutputs`, and invokes `Kernel.body` exactly once per
physical thread. It supplies the corresponding `LaunchFacts` to the body and stores every returned
work-item result through `outputLayout`. No implementation is given here.
-/
axiom launch
    (cfg : LaunchConfig)
    (output : Buffer .global outputLength α)
    (outputLayout : Layout outputSpace outputLength)
    (certificate : ScheduleCertificate cfg outputSpace)
    (kernel : Kernel outputSpace α)
    (_requirements : kernel.requirements cfg) :
    Launch outputLength α

end GpuDsl
