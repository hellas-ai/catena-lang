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

@[ext] theorem Coord.ext {left right : Coord bounds}
    (x : left.x = right.x) (y : left.y = right.y) (z : left.z = right.z) :
    left = right := by
  cases left
  cases right
  simp_all

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

/-- Stable identity of one physical allocation. -/
structure AllocationId where
  name : String
  deriving Repr, DecidableEq

/--
A typed buffer handle. The length and address space are part of its type.

This prototype describes kernels; a later backend can replace `name` with a
real allocation while retaining the same safe indexing API.
-/
structure Buffer (space : AddressSpace) (length : Nat) (α : Type) where
  allocation : AllocationId
  deriving Repr

/-- A shared-memory handle tied to one particular block instance. -/
structure SharedBuffer
    (block : Block cfg BlockState) (name : String)
    (length : Nat) (α : Type) where
  buffer : Buffer .shared length α

/-- Static names distinguish synchronization sites in a trace. -/
inductive BarrierId where
  | tileLoaded
  | tileConsumed
  deriving Repr, DecidableEq

abbrev SyncTrace := List BarrierId

def iterateTrace (step : SyncTrace → SyncTrace) : Nat → SyncTrace → SyncTrace
  | 0, trace => trace
  | n + 1, trace => iterateTrace step n (step trace)

/-- A launch-time assignment of each logical shared cell to one block thread. -/
structure OwnershipPlan (cfg : LaunchConfig) where
  Cell : Type
  owner : Cell → Coord (LaunchConfig.block cfg)

/-- Exclusive access to one cell. -/
structure Owns {cfg : LaunchConfig} {BlockState State SharedScalar : Type}
    {name : String} {length : Nat} {block : Block cfg BlockState}
    (trace : SyncTrace)
    (thread : Thread cfg BlockState State)
    (buffer : SharedBuffer block name length SharedScalar)
    (index : Fin length) : Type where
  private intro ::

/-- Read-only access to one global buffer for one physical thread. -/
structure GlobalReadShare {cfg : LaunchConfig} {α : Type}
    {length : Nat} (thread : ThreadId cfg)
    (buffer : Buffer .global length α)
    (region : Fin length → Prop) : Type where
  private intro ::

/-- Exclusive access to one global-memory cell for one physical thread. -/
structure GlobalOwns {cfg : LaunchConfig} {α : Type}
    {length : Nat} (thread : ThreadId cfg)
    (buffer : Buffer .global length α) (index : Fin length) : Type where
  private intro ::

/-- A pure staged value used by kernel statements. -/
inductive Value (α : Type) : Type where
  | literal (value : α) : Value α
  | binary (operation : α → α → α) (left right : Value α) : Value α

namespace Value

def zero [OfNat α 0] : Value α := .literal 0

def add [Add α] (left right : Value α) : Value α :=
  .binary (· + ·) left right

def mul [Mul α] (left right : Value α) : Value α :=
  .binary (· * ·) left right

end Value

/-- Evidence produced only by a permission-checked global-memory write. -/
structure WroteGlobal {cfg : LaunchConfig} {α : Type}
    {length : Nat} (thread : ThreadId cfg)
    (buffer : Buffer .global length α) (index : Fin length)
    (value : Value α) : Type where
  private intro ::

/-- Evidence produced only by a shared-memory write statement. -/
structure Wrote {cfg : LaunchConfig} {BlockState State α : Type}
    {name : String} {length : Nat} {block : Block cfg BlockState}
    (trace : SyncTrace)
    (thread : Thread cfg BlockState State)
    (buffer : SharedBuffer block name length α) (index : Fin length)
    (value : Value α) : Type where
  private intro ::

/-- Constructive evidence that a thread wrote some value to a cell. -/
abbrev WroteSome (alpha : Type) {cfg : LaunchConfig} {BlockState State : Type}
    {name : String} {length : Nat} {block : Block cfg BlockState}
    (trace : SyncTrace)
    (thread : Thread cfg BlockState State)
    (buffer : SharedBuffer block name length alpha) (index : Fin length) : Type :=
  Σ value : Value alpha, Wrote trace thread buffer index value

/-- A cell is defined immediately after a tile-loaded barrier when some block
thread wrote it before that barrier. -/
structure Defined {cfg : LaunchConfig} {BlockState α : Type}
    {name : String} {length : Nat} {block : Block cfg BlockState}
    (trace : SyncTrace) (buffer : SharedBuffer block name length α)
    (index : Fin length) : Type where
  private intro ::

def Defined.ofWrote {cfg : LaunchConfig} {BlockState State α : Type}
    {name : String} {length : Nat} {block : Block cfg BlockState}
    {before : SyncTrace}
    {writer : Thread cfg BlockState State} {value : Value α}
    {buffer : SharedBuffer block name length α} {index : Fin length}
    (barrier : BarrierId) (_wrote : Wrote before writer buffer index value) :
    Defined (before ++ [barrier]) buffer index :=
  ⟨⟩

def WroteSome.defined {cfg : LaunchConfig} {BlockState State α : Type}
    {name : String} {length : Nat} {block : Block cfg BlockState}
    {before : SyncTrace}
    {writer : Thread cfg BlockState State}
    {buffer : SharedBuffer block name length α} {source target : Fin length}
    (barrier : BarrierId) (write : WroteSome α before writer buffer source)
    (equal : source = target) :
    Defined (before ++ [barrier]) buffer target := by
  cases equal
  exact Defined.ofWrote barrier write.2

/-- Shared, read-only access to a buffer. -/
structure ReadShare {cfg : LaunchConfig} {BlockState State α : Type}
    {name : String} {length : Nat} {block : Block cfg BlockState}
    (trace : SyncTrace) (thread : Thread cfg BlockState State)
    (buffer : SharedBuffer block name length α) : Type where
  private intro ::

/-- One or more shared read permissions granted by a barrier. -/
inductive ReadGrants {cfg : LaunchConfig} {BlockState State : Type}
    (α : Type) (thread : Thread cfg BlockState State) : Type where
  | one {name : String} {length : Nat}
      (buffer : SharedBuffer thread.block name length α) : ReadGrants α thread
  | and (left right : ReadGrants α thread) : ReadGrants α thread

def ReadGrants.Result (trace : SyncTrace) :
    ReadGrants α thread → Type
  | .one buffer => ReadShare trace thread buffer
  | .and left right => left.Result trace × right.Result trace

/-- One or more exclusive permissions, each checked against the launch plan. -/
inductive OwnGrants {cfg : LaunchConfig} {BlockState State : Type}
    (plan : OwnershipPlan cfg) (α : Type)
    (thread : Thread cfg BlockState State) : Type where
  | one {name : String} {length : Nat}
      (buffer : SharedBuffer thread.block name length α)
      (layout : plan.Cell → Fin length)
      (layoutInjective : Injective layout)
      (cell : plan.Cell)
      (isOwner : plan.owner cell = thread.index) : OwnGrants plan α thread
  | and (left right : OwnGrants plan α thread) : OwnGrants plan α thread

def OwnGrants.Result (trace : SyncTrace) :
    OwnGrants plan α thread → Type
  | .one buffer layout _ cell _ => Owns trace thread buffer (layout cell)
  | .and left right => left.Result trace × right.Result trace

/-- A barrier grants either shared reads or checked exclusive ownership. -/
inductive BarrierGrants {cfg : LaunchConfig} {BlockState State : Type}
    (plan : OwnershipPlan cfg) (α : Type)
    (thread : Thread cfg BlockState State) : Type where
  | reads (grants : ReadGrants α thread) : BarrierGrants plan α thread
  | owns (grants : OwnGrants plan α thread) : BarrierGrants plan α thread

def BarrierGrants.Result (trace : SyncTrace) :
    BarrierGrants plan α thread → Type
  | .reads grants => grants.Result trace
  | .owns grants => grants.Result trace

/-- The permission transition performed by one barrier. -/
structure SyncSpec {cfg : LaunchConfig} {BlockState State α : Type}
    (plan : OwnershipPlan cfg) (thread : Thread cfg BlockState State) where
  barrier : BarrierId
  grants : BarrierGrants plan α thread

/--
The statement language is indexed by the running `Thread`, whose type carries
the launch geometry and launch-created state types.

The higher-order constructors make this a compact typed operational IR. A
backend/interpreter supplies the values passed to the continuations.
-/
inductive KernelM {cfg : LaunchConfig} {BlockState State : Type}
    (plan : OwnershipPlan cfg) (SharedScalar : Type)
    (thread : Thread cfg BlockState State) :
    SyncTrace → SyncTrace → Type → Type 1 where
  | pure {α : Type} (value : α) : KernelM plan SharedScalar thread trace trace α
  | bind {α β : Type}
      (first : KernelM plan SharedScalar thread before middle α)
      (next : α → KernelM plan SharedScalar thread middle after β) :
      KernelM plan SharedScalar thread before after β
  /-- Read global memory using a launch-granted read permission. -/
  | readGlobal {α : Type} {n : Nat} {region : Fin n → Prop}
      (buffer : Buffer .global n α)
      (index : Fin n)
      (readShare : GlobalReadShare thread.id buffer region)
      (inRegion : region index) :
      KernelM plan SharedScalar thread trace trace (Value α)
  /-- Write one globally owned cell. -/
  | writeGlobal {α : Type} {n : Nat}
      (buffer : Buffer .global n α)
      (index : Fin n)
      (value : Value α)
      (owned : GlobalOwns thread.id buffer index) :
      KernelM plan SharedScalar thread trace trace
        (WroteGlobal thread.id buffer index value)
  /-- Apply a trace-preserving statement to every item in a finite list. -/
  | forEach {Item : Type}
      (items : List Item)
      (body : ∀ item, List.Mem item items →
        KernelM plan SharedScalar thread trace trace Unit) :
      KernelM plan SharedScalar thread trace trace Unit
  /-- Write one shared-memory cell from the current physical thread. -/
  | writeShared {name : String} {n : Nat}
      (buffer : SharedBuffer (Thread.block thread) name n SharedScalar)
      (index : Fin n)
      (value : Value SharedScalar)
      (owned : Owns trace thread buffer index) :
      KernelM plan SharedScalar thread trace trace
        (Wrote trace thread buffer index value)
  /-- Read one shared-memory cell from the current physical thread. -/
  | readShared {name : String} {n : Nat}
      (buffer : SharedBuffer (Thread.block thread) name n SharedScalar)
      (index : Fin n)
      (defined : Defined trace buffer index)
      (readShare : ReadShare trace thread buffer) :
      KernelM plan SharedScalar thread trace trace (Value SharedScalar)
  /-- A block barrier lifts a fact proved by the current thread to the same
  fact for every thread in the block. -/
  | syncBlock (block : Block cfg BlockState)
      (belongsToBlock : Thread.block thread = block)
      (spec : SyncSpec (α := SharedScalar) plan thread)
      {facts : SyncTrace → Thread cfg BlockState State → Type}
      (currentFact : facts trace thread) :
      KernelM plan SharedScalar thread trace (trace ++ [spec.barrier])
        ((∀ other, Thread.block other = block → facts trace other) ×
          spec.grants.Result (trace ++ [spec.barrier]))
  /-- A bounded loop with a trace-independent accumulator and a separate
  invariant. The initial value and every callback result have the same
  invariant shape, indexed by their current synchronization trace. -/
  | foldFinD (n : Nat) (step : SyncTrace → SyncTrace)
      (startTrace : SyncTrace) {α : Type} (Invariant : SyncTrace → Type)
      (initialInvariant : Invariant startTrace)
      (initial : α)
      (body : ∀ currentTrace, Fin n → α → Invariant currentTrace →
        KernelM plan SharedScalar thread currentTrace (step currentTrace)
          (α × Invariant (step currentTrace))) :
      KernelM plan SharedScalar thread startTrace (iterateTrace step n startTrace) α

end GpuDsl
