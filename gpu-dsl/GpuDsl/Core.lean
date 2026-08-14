namespace GpuDsl

def Injective (function : α → β) : Prop :=
  ∀ ⦃left right⦄, function left = function right → left = right

/-- CUDA/HIP's three launch axes. -/
inductive Axis where
  | x | y | z
  deriving Repr, DecidableEq

/-- A CUDA/HIP-style `dim3`. -/
structure Dim3 where
  x : Nat
  y : Nat := 1
  z : Nat := 1
  deriving Repr, DecidableEq

def Dim3.extent (d : Dim3) : Axis → Nat
  | .x => d.x
  | .y => d.y
  | .z => d.z

/-- Grid and thread-block geometry for one launch. -/
structure LaunchConfig where
  grid : Dim3
  block : Dim3
  deriving Repr, DecidableEq

/-- A bounded 3D coordinate in a grid or thread block. -/
structure Coord (bounds : Dim3) where
  x : Fin bounds.x
  y : Fin bounds.y
  z : Fin bounds.z

/-- The identity of a physical GPU thread within one launch. -/
structure ThreadId (cfg : LaunchConfig) where
  blockIdx : Coord (LaunchConfig.grid cfg)
  threadIdx : Coord (LaunchConfig.block cfg)

/-- A reference to the current block and its launch-created state. -/
structure Block (cfg : LaunchConfig) (State : Type) where
  index : Coord (LaunchConfig.grid cfg)
  state : State

/--
A reference to the running GPU thread. The launch implementation chooses and
creates both state types; the kernel merely receives their references.
-/
structure Thread (cfg : LaunchConfig) (BlockState State : Type) where
  block : Block cfg BlockState
  index : Coord (LaunchConfig.block cfg)
  state : State

def Thread.id (thread : Thread cfg BlockState State) : ThreadId cfg :=
  { blockIdx := Block.index (Thread.block thread)
    threadIdx := Thread.index thread }

inductive AddressSpace where
  | global
  | shared
  deriving Repr, DecidableEq

/--
A typed buffer handle. The length and address space are part of its type.

This prototype describes kernels; a later backend can replace `name` with a
real allocation while retaining the same safe indexing API.
-/
structure Buffer (space : AddressSpace) (length : Nat) (α : Type) where
  name : String
  deriving Repr

/-- A shared-memory handle tied to one particular block instance. -/
structure SharedBuffer
    (block : Block cfg BlockState) (length : Nat) (α : Type) where
  buffer : Buffer .shared length α

/-- Every shared-memory cell is assigned to exactly one physical block thread. -/
structure WriteOwnership (cfg : LaunchConfig) (length : Nat) where
  owner : Fin length → Coord (LaunchConfig.block cfg)

def Owns (ownership : WriteOwnership cfg length)
    (thread : Thread cfg BlockState State) (index : Fin length) : Prop :=
  WriteOwnership.owner ownership index = Thread.index thread

/-- Static names distinguish synchronization sites in a trace. -/
inductive BarrierId where
  | tileLoaded
  | tileConsumed
  deriving Repr, DecidableEq

abbrev SyncTrace := List BarrierId

def iterateTrace (step : SyncTrace → SyncTrace) : Nat → SyncTrace → SyncTrace
  | 0, trace => trace
  | n + 1, trace => iterateTrace step n (step trace)

/-- One physical thread wrote one particular shared cell at `trace`. -/
structure ThreadWroteAt
    (buffer : SharedBuffer block length α)
    (index : Fin length)
    (threadIndex : Coord (LaunchConfig.block cfg))
    (trace : SyncTrace) where
  private token : Unit

/--
Static evidence that every physical thread writes the shared cells assigned to
it at `trace`, immediately before a synchronization beginning at that trace.
-/
structure SharedWritePhase
    (buffer : SharedBuffer block length α)
    (ownership : WriteOwnership cfg length)
    (trace : SyncTrace) where
  private writesOwned : ∀ threadIndex index,
    WriteOwnership.owner ownership index = threadIndex →
    ThreadWroteAt buffer index threadIndex trace

/-- Ghost evidence that one physical thread has crossed a block barrier. -/
structure ThreadAfterSync
    (block : Block cfg BlockState)
    (threadIndex : Coord (LaunchConfig.block cfg))
    (before : SyncTrace)
    (barrier : BarrierId) where
  private token : Unit

/-- A block rendezvous: every physical thread has crossed this barrier. -/
structure BlockSync
    (block : Block cfg BlockState)
    (before : SyncTrace)
    (barrier : BarrierId) where
  private allAfter : ∀ threadIndex,
    ThreadAfterSync block threadIndex before barrier

/-- Ghost evidence that one particular buffer cell is defined at `trace`. -/
structure DefinedAt
    (buffer : SharedBuffer block length α)
    (index : Fin length)
    (trace : SyncTrace) where
  private token : Unit

/-- No thread can still be reading the previous contents at `trace`. -/
structure WritableAt
    (buffer : SharedBuffer block length α)
    (trace : SyncTrace) where
  private token : Unit

/-- Fresh launch-created shared memory is undefined and therefore writable. -/
def WritableAt.initial
    (buffer : SharedBuffer block length α) : WritableAt buffer [] :=
  ⟨()⟩

/-- Every physical thread has finished reading this buffer at `trace`. -/
structure SharedReadPhase
    (buffer : SharedBuffer block length α)
    (trace : SyncTrace) where
  private token : Unit

/-- A completed read phase followed by a block barrier permits reuse. -/
def SharedReadPhase.writableAfter
    {buffer : SharedBuffer block length α}
    (_reads : SharedReadPhase buffer trace)
    (_sync : BlockSync block trace barrier) :
    WritableAt buffer (trace ++ [barrier]) :=
  ⟨()⟩

/--
Every cell has an assigned thread; every thread crossed the barrier; and before
that barrier each thread wrote its assigned cells. Therefore this cell is
defined after the barrier. Matching `trace` parameters tie the writes to this
particular synchronization point.
-/
def SharedWritePhase.definedAfter
    {cfg : LaunchConfig} {BlockState α : Type}
    {block : Block cfg BlockState} {length : Nat}
    {buffer : SharedBuffer block length α}
    {ownership : WriteOwnership cfg length}
    {trace : SyncTrace} {barrier : BarrierId}
    (_writes : SharedWritePhase buffer ownership trace)
    (_sync : BlockSync block trace barrier)
    (index : Fin length) :
    DefinedAt (length := length) buffer index (trace ++ [barrier]) :=
  let writer := WriteOwnership.owner ownership index
  let _wroteBefore := SharedWritePhase.writesOwned _writes writer index rfl
  let _afterSync := BlockSync.allAfter _sync writer
  ⟨()⟩

/-- A pure staged value used by kernel statements. -/
inductive Value (α : Type) : Type where
  | literal (value : α) : Value α
  | load {space : AddressSpace} {length : Nat}
      (buffer : Buffer space length α) (index : Fin length) : Value α
  | binary (operation : α → α → α) (left right : Value α) : Value α

namespace Value

def zero [OfNat α 0] : Value α := .literal 0

def add [Add α] (left right : Value α) : Value α :=
  .binary (· + ·) left right

def mul [Mul α] (left right : Value α) : Value α :=
  .binary (· * ·) left right

end Value

/--
The statement language is indexed by the running `Thread`, whose type carries
the launch geometry and launch-created state types.

The higher-order constructors make this a compact typed operational IR. A
backend/interpreter supplies the values passed to the continuations.
-/
inductive KernelM {cfg : LaunchConfig} {BlockState State : Type}
    (thread : Thread cfg BlockState State) : SyncTrace → SyncTrace → Type → Type 1 where
  | pure {α : Type} (value : α) : KernelM thread trace trace α
  | bind {α β : Type}
      (first : KernelM thread before middle α)
      (next : α → KernelM thread middle after β) :
      KernelM thread before after β
  /-- Each physical thread writes every shared cell assigned to it. -/
  | initializeShared {n : Nat} {α : Type}
      (buffer : SharedBuffer (Thread.block thread) n α)
      (ownership : WriteOwnership cfg n)
      (writable : WritableAt buffer trace)
      (value : (index : Fin n) → Owns ownership thread index → Value α) :
      KernelM thread trace trace
        (SharedWritePhase buffer ownership trace)
  /-- Force a staged expression into a register before continuing. -/
  | materialize (value : Value α) : KernelM thread trace trace (Value α)
  /-- Mark the point after this thread's reads; uniform execution covers the block. -/
  | finishSharedReads (buffer : SharedBuffer (Thread.block thread) n α) :
      KernelM thread trace trace (SharedReadPhase buffer trace)
  | syncBlock (barrier : BarrierId) :
      KernelM thread trace (trace ++ [barrier])
        (BlockSync (Thread.block thread) trace barrier)
  /-- A uniform launch-time assumption to be discharged by a backend. -/
  | require (condition : Prop) [Decidable condition]
      (body : condition → KernelM thread before after α) :
      KernelM thread before after α
  /-- A statically bounded loop whose every iteration has the same trace transition. -/
  | foldFin (n : Nat) (initial : α)
      (body : ∀ trace, Fin n → α → KernelM thread trace (step trace) α) :
      KernelM thread trace (iterateTrace step n trace) α
  /-- A bounded loop whose loop-carried state depends on the current trace. -/
  | foldFinD (n : Nat) (Acc : SyncTrace → Type)
      (initial : Acc trace)
      (body : ∀ currentTrace, Fin n → Acc currentTrace →
        KernelM thread currentTrace (step currentTrace) (Acc (step currentTrace))) :
      KernelM thread trace (iterateTrace step n trace)
        (Acc (iterateTrace step n trace))

end GpuDsl
