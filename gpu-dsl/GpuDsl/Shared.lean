import GpuDsl.Core

namespace GpuDsl

/--
A block-scoped shared-memory allocation plan. All slots use the kernel's scalar
type; each slot independently records its name and compile-time element count.
-/
inductive SharedSpec where
  | empty
  | buffer (name : String) (length : Nat) (rest : SharedSpec)
  deriving Repr, DecidableEq

/--
The handles allocated by `launch` for one block. Its shape must exactly match
the kernel's shared-memory specification.
-/
inductive SharedState (α : Type) : SharedSpec → Type where
  | empty : SharedState α .empty
  | buffer (handle : Buffer .shared length α) (rest : SharedState α spec) :
      SharedState α (.buffer name length spec)

/-- Compile-time evidence that a buffer of `length` occurs in `spec`. -/
inductive SharedRef (length : Nat) : SharedSpec → Type where
  | here : SharedRef length (.buffer name length rest)
  | there : SharedRef length rest → SharedRef length (.buffer name otherLength rest)

/--
Total lookup: the `SharedRef` proof makes requesting a missing or incorrectly
sized allocation unrepresentable.
-/
private def SharedState.lookup (state : SharedState α spec)
    (reference : SharedRef length spec) : Buffer .shared length α :=
  match state, reference with
  | .buffer handle _, .here => handle
  | .buffer _ rest, .there reference => SharedState.lookup rest reference

/-- Retrieve a handle that retains the identity of its owning block. -/
def SharedState.get (block : Block cfg (SharedState α spec))
    (reference : SharedRef length spec) : SharedBuffer block length α :=
  ⟨SharedState.lookup (Block.state block) reference⟩

/-- Read one shared cell with evidence that this exact index is defined now. -/
def SharedBuffer.read
    (buffer : SharedBuffer block length α)
    (index : Fin length)
    (_defined : DefinedAt (length := length) buffer index trace) : Value α :=
  Value.load (SharedBuffer.buffer buffer) index

end GpuDsl
