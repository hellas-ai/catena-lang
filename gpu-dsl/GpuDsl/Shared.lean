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

/--
Compile-time evidence that a buffer named `name` with `length` occurs in `spec`.
-/
inductive SharedRef (name : String) (length : Nat) : SharedSpec → Type where
  | here : SharedRef name length (.buffer name length rest)
  | there : SharedRef name length rest →
      SharedRef name length (.buffer otherName otherLength rest)

/-- Typeclass-searchable membership of a named allocation in a shared spec. -/
class HasShared (name : String) (length : Nat) (spec : SharedSpec) where
  reference : SharedRef name length spec

instance (priority := 1000) :
    HasShared name length (.buffer name length rest) where
  reference := .here

instance (priority := 900) [member : HasShared name length rest] :
    HasShared name length (.buffer otherName otherLength rest) where
  reference := .there member.reference

/--
Total lookup: the `SharedRef` proof makes requesting a missing or incorrectly
sized allocation unrepresentable.
-/
private def SharedState.lookup (state : SharedState α spec)
    (reference : SharedRef name length spec) : Buffer .shared length α :=
  match state, reference with
  | .buffer handle _, .here => handle
  | .buffer _ rest, .there reference => SharedState.lookup rest reference

/-- Retrieve a handle using explicit membership evidence. -/
private def SharedState.getRef (block : Block cfg (SharedState α spec))
    (reference : SharedRef name length spec) :
    SharedBuffer block name length α :=
  ⟨SharedState.lookup (Block.state block) reference⟩

/-- Retrieve a named handle; its `SharedRef` evidence is synthesized. -/
def SharedState.get (block : Block cfg (SharedState α spec))
    (name : String) [member : HasShared name length spec] :
    SharedBuffer block name length α :=
  SharedState.getRef block member.reference

end GpuDsl
