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

/-- A thread's two-dimensional lane within its block. -/
structure Lane (rows cols : Nat) where
  row : Fin rows
  col : Fin cols

/-- A shared-memory handle tied to one particular block instance. -/
structure BlockBuffer
    (block : Block cfg BlockState) (length : Nat) (α : Type) where
  buffer : Buffer .shared length α

/--
Proof that the current thread is the unique writer of `index` in its block.
`blockZ = 1` rules out duplicate x/y lanes in the third dimension.
-/
structure ExclusiveWrite
    (thread : Thread cfg BlockState State) (index : Fin length) where
  blockZ : Dim3.z (LaunchConfig.block cfg) = 1
  owner : Lane (Dim3.y (LaunchConfig.block cfg)) (Dim3.x (LaunchConfig.block cfg)) →
    Fin length
  owns : owner ⟨Coord.y (Thread.index thread), Coord.x (Thread.index thread)⟩ = index
  unique : Injective owner

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
    (thread : Thread cfg BlockState State) : Type → Type 1 where
  | pure {α : Type} (value : α) : KernelM thread α
  | bind {α β : Type} (first : KernelM thread α)
      (next : α → KernelM thread β) : KernelM thread β
  | storeShared {n : Nat} {α : Type}
      (buffer : BlockBuffer (Thread.block thread) n α)
      (index : Fin n)
      (ownership : ExclusiveWrite thread index)
      (value : Value α) : KernelM thread Unit
  | barrier : KernelM thread Unit
  /-- A uniform launch-time assumption to be discharged by a backend. -/
  | require (condition : Prop) [Decidable condition]
      (body : condition → KernelM thread α) : KernelM thread α

instance : Monad (KernelM thread) where
  pure := KernelM.pure
  bind := KernelM.bind

/-- Build a statically bounded loop in the kernel IR. -/
def KernelM.foldFin (n : Nat) (initial : α)
    (body : Fin n → α → KernelM thread α) : KernelM thread α :=
  let rec go (i : Nat) (acc : α) : KernelM thread α :=
    if hi : i < n then
      body ⟨i, hi⟩ acc >>= go (i + 1)
    else
      pure acc
  termination_by n - i
  go 0 initial

end GpuDsl
