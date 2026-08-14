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
One invocation assigned by `launch` to a thread. `none` certifies that this
thread owns no output; `some index` proves ownership of that work item. A
backend may invoke the kernel repeatedly for a thread that owns several cells.
-/
structure Scheduled
    (certificate : ScheduleCertificate cfg outputSpace)
    (threadId : ThreadId cfg) where
  output : Option (Index outputSpace)
  valid : match output with
    | some index => ScheduleCertificate.owner certificate index = threadId
    | none => ∀ index, ScheduleCertificate.owner certificate index ≠ threadId

/-- The result required from a kernel for one scheduled invocation. -/
def Scheduled.Result (scheduled : Scheduled certificate threadId) (α : Type) : Type :=
  match Scheduled.output scheduled with
  | some _ => Value α
  | none => Unit

/--
A generic kernel depends only on its logical output space. Launch geometry,
block state, thread state, and the schedule certificate are supplied by the
`launch` implementation through the thread and scheduled work item.
-/
structure Kernel (outputSpace : Space) (α : Type) where
  /-- Shared buffers allocated once per block by `launch`. -/
  shared : SharedSpec
  /-- Every thread must finish after this exact synchronization trace. -/
  trace : SyncTrace
  body :
    ∀ {cfg : LaunchConfig} {State : Type}
      {certificate : ScheduleCertificate cfg outputSpace},
      (thread : Thread cfg (SharedState α shared) State) →
      (scheduled : Scheduled certificate (Thread.id thread)) →
      KernelM thread [] trace (Scheduled.Result scheduled α)

/-- An opaque launch token. Execution belongs to the eventual backend. -/
structure Launch (outputLength : Nat) (α : Type) where
  private token : Unit

/--
CUDA/HIP-shaped primitive. Its implementation creates block/thread state,
allocates `Kernel.shared` once per block, enumerates work items from the
certificate, and invokes `Kernel.body`. No implementation is given here.
-/
axiom launch
    (cfg : LaunchConfig)
    (output : Buffer .global outputLength α)
    (outputLayout : Layout outputSpace outputLength)
    (certificate : ScheduleCertificate cfg outputSpace)
    (kernel : Kernel outputSpace α) :
    Launch outputLength α

end GpuDsl
