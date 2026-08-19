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

/-- A kernel-specific ownership relation over actual block-scoped buffers. -/
structure OwnershipRelation
    (cfg : LaunchConfig) (BlockState SharedScalar : Type) where
  owns : ∀ {name : String} {length : Nat} {block : Block cfg BlockState},
    SyncTrace → Coord (LaunchConfig.block cfg) →
      SharedBuffer block name length SharedScalar → Fin length → Prop

abbrev Owns {cfg : LaunchConfig} {BlockState State SharedScalar : Type}
    {name : String} {length : Nat}
    {relation : OwnershipRelation cfg BlockState SharedScalar} (trace : SyncTrace)
    (thread : Thread cfg BlockState State)
    (buffer : SharedBuffer (Thread.block thread) name length SharedScalar)
    (index : Fin length) : Prop :=
  relation.owns trace (Thread.index thread) buffer index

/-- A pure staged value used by kernel statements. -/
inductive Value (α : Type) : Type where
  | literal (value : α) : Value α
  | load {length : Nat}
      (buffer : Buffer .global length α) (index : Fin length) : Value α
  | binary (operation : α → α → α) (left right : Value α) : Value α

namespace Value

def zero [OfNat α 0] : Value α := .literal 0

def add [Add α] (left right : Value α) : Value α :=
  .binary (· + ·) left right

def mul [Mul α] (left right : Value α) : Value α :=
  .binary (· * ·) left right

end Value

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
    (_wrote : Wrote before writer buffer index value) :
    Defined (before ++ [.tileLoaded]) buffer index :=
  ⟨⟩

def WroteSome.defined {cfg : LaunchConfig} {BlockState State α : Type}
    {name : String} {length : Nat} {block : Block cfg BlockState}
    {before : SyncTrace}
    {writer : Thread cfg BlockState State}
    {buffer : SharedBuffer block name length α} {source target : Fin length}
    (write : WroteSome α before writer buffer source) (equal : source = target) :
    Defined (before ++ [.tileLoaded]) buffer target := by
  cases equal
  exact Defined.ofWrote write.2

/--
The statement language is indexed by the running `Thread`, whose type carries
the launch geometry and launch-created state types.

The higher-order constructors make this a compact typed operational IR. A
backend/interpreter supplies the values passed to the continuations.
-/
inductive KernelM {cfg : LaunchConfig} {BlockState State SharedScalar : Type}
    (ownershipRelation : OwnershipRelation cfg BlockState SharedScalar)
    (thread : Thread cfg BlockState State) :
    SyncTrace → SyncTrace → Type → Type 1 where
  | pure {α : Type} (value : α) : KernelM ownershipRelation thread trace trace α
  | bind {α β : Type}
      (first : KernelM ownershipRelation thread before middle α)
      (next : α → KernelM ownershipRelation thread middle after β) :
      KernelM ownershipRelation thread before after β
  /-- Write one shared-memory cell from the current physical thread. -/
  | writeShared {name : String} {n : Nat}
      (buffer : SharedBuffer (Thread.block thread) name n SharedScalar)
      (index : Fin n)
      (value : Value SharedScalar)
      (owned : Owns (relation := ownershipRelation) trace thread buffer index) :
      KernelM ownershipRelation thread trace trace
        (Wrote trace thread buffer index value)
  /-- Read one shared-memory cell from the current physical thread. -/
  | readShared {name : String} {n : Nat}
      (buffer : SharedBuffer (Thread.block thread) name n SharedScalar)
      (index : Fin n)
      (defined : Defined trace buffer index) :
      KernelM ownershipRelation thread trace trace (Value SharedScalar)
  /-- A block barrier lifts a fact proved by the current thread to the same
  fact for every thread in the block. -/
  | syncBlock (block : Block cfg BlockState)
      (belongsToBlock : Thread.block thread = block)
      (barrier : BarrierId)
      {facts : SyncTrace → Thread cfg BlockState State → Type}
      (currentFact : facts trace thread) :
      KernelM ownershipRelation thread trace (trace ++ [barrier])
        (∀ other, Thread.block other = block → facts trace other)
  /-- A bounded loop with an ordinary, trace-independent accumulator. -/
  | foldFinD (n : Nat) (step : SyncTrace → SyncTrace)
      (startTrace : SyncTrace) {α : Type}
      (initial : α)
      (body : ∀ currentTrace, Fin n → α →
        KernelM ownershipRelation thread currentTrace (step currentTrace) α) :
      KernelM ownershipRelation thread startTrace (iterateTrace step n startTrace) α

end GpuDsl
