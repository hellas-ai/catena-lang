import GpuDsl.Core

namespace GpuDsl

universe u

/-- The logical domain of a closure. More dimensions can be added later. -/
inductive Space where
  | d1 (length : Nat)
  | d2 (rows cols : Nat)
  | d3 (depth rows cols : Nat)
  deriving Repr, DecidableEq

/-- Named coordinates are more readable than nested products in kernels. -/
structure Index2 (rows cols : Nat) where
  row : Fin rows
  col : Fin cols

structure Index3 (depth rows cols : Nat) where
  depth : Fin depth
  row : Fin rows
  col : Fin cols

/-- The dependent index type belonging to a logical space. -/
def Index : Space → Type
  | .d1 length => Fin length
  | .d2 rows cols => Index2 rows cols
  | .d3 depth rows cols => Index3 depth rows cols

/--
A logical tensor is only a closure from an index to a value. Mapping,
reindexing, and zipping tensors therefore create no allocation or copy.
-/
abbrev Tensor (space : Space) (α : Type u) := Index space → α

abbrev Vector (length : Nat) (α : Type u) := Tensor (.d1 length) α

abbrev Matrix (rows cols : Nat) (α : Type u) := Tensor (.d2 rows cols) α

abbrev Tensor3 (depth rows cols : Nat) (α : Type u) := Tensor (.d3 depth rows cols) α

/-- A layout maps logical indices onto a statically sized linear allocation. -/
structure Layout (space : Space) (length : Nat) where
  offset : Index space → Fin length

namespace Layout

def linear : Layout (.d1 length) length := ⟨id⟩

def rowMajor2D : Layout (.d2 rows cols) (rows * cols) where
  offset index := by
    refine ⟨Fin.val (Index2.row index) * cols + Fin.val (Index2.col index), ?_⟩
    calc
      Fin.val (Index2.row index) * cols + Fin.val (Index2.col index) <
          Fin.val (Index2.row index) * cols + cols :=
        Nat.add_lt_add_left (Fin.isLt (Index2.col index)) _
      _ = (Fin.val (Index2.row index) + 1) * cols := by
        simp [Nat.add_mul, Nat.add_comm]
      _ ≤ rows * cols :=
        Nat.mul_le_mul_right cols (Nat.succ_le_of_lt (Fin.isLt (Index2.row index)))

theorem rowMajor2D_injective : Injective (@rowMajor2D rows cols).offset := by
  intro left right equalOffsets
  cases cols with
  | zero => exact Fin.elim0 (Index2.col left)
  | succ cols =>
    have equalValues := congrArg Fin.val equalOffsets
    have equalCols : Fin.val (Index2.col left) = Fin.val (Index2.col right) := by
      have reduced := congrArg (fun value => value % (cols + 1)) equalValues
      simpa [rowMajor2D, Nat.add_mod, Nat.mul_mod, Nat.mod_eq_of_lt,
        Fin.isLt] using reduced
    have equalRows : Fin.val (Index2.row left) = Fin.val (Index2.row right) := by
      change Fin.val (Index2.row left) * (cols + 1) + Fin.val (Index2.col left) =
        Fin.val (Index2.row right) * (cols + 1) + Fin.val (Index2.col right) at equalValues
      rw [equalCols] at equalValues
      have equalProducts := Nat.add_right_cancel equalValues
      exact Nat.mul_right_cancel (Nat.zero_lt_succ cols) equalProducts
    cases left with
    | mk leftRow leftCol =>
      cases right with
      | mk rightRow rightCol =>
        have rowsEqual : leftRow = rightRow := Fin.ext equalRows
        have colsEqual : leftCol = rightCol := Fin.ext equalCols
        cases rowsEqual
        cases colsEqual
        rfl

def columnMajor2D : Layout (.d2 rows cols) (rows * cols) where
  offset index := by
    refine ⟨Fin.val (Index2.col index) * rows + Fin.val (Index2.row index), ?_⟩
    calc
      Fin.val (Index2.col index) * rows + Fin.val (Index2.row index) <
          Fin.val (Index2.col index) * rows + rows :=
        Nat.add_lt_add_left (Fin.isLt (Index2.row index)) _
      _ = (Fin.val (Index2.col index) + 1) * rows := by
        simp [Nat.add_mul, Nat.add_comm]
      _ ≤ cols * rows :=
        Nat.mul_le_mul_right rows (Nat.succ_le_of_lt (Fin.isLt (Index2.col index)))
      _ = rows * cols := Nat.mul_comm cols rows

/-- Conventional depth-major, then row-major, 3D layout. -/
def rowMajor3D : Layout (.d3 depth rows cols) (depth * (rows * cols)) where
  offset index := by
    let plane : Fin (rows * cols) :=
      Layout.offset rowMajor2D ⟨Index3.row index, Index3.col index⟩
    refine ⟨Fin.val (Index3.depth index) * (rows * cols) + Fin.val plane, ?_⟩
    calc
      Fin.val (Index3.depth index) * (rows * cols) + Fin.val plane <
          Fin.val (Index3.depth index) * (rows * cols) + (rows * cols) :=
        Nat.add_lt_add_left (Fin.isLt plane) _
      _ = (Fin.val (Index3.depth index) + 1) * (rows * cols) := by
        simp [Nat.add_mul, Nat.add_comm]
      _ ≤ depth * (rows * cols) :=
        Nat.mul_le_mul_right (rows * cols)
          (Nat.succ_le_of_lt (Fin.isLt (Index3.depth index)))

end Layout

namespace Tensor

/-- Reindex a closure without touching its underlying storage. -/
def reindex (source : Tensor sourceSpace α)
    (indexMap : Index targetSpace → Index sourceSpace) : Tensor targetSpace α :=
  fun index => source (indexMap index)

def map (f : α → β) (source : Tensor space α) : Tensor space β :=
  fun index => f (source index)

def zipWith (f : α → β → γ) (left : Tensor space α)
    (right : Tensor space β) : Tensor space γ :=
  fun index => f (left index) (right index)

/-- Turn a linear closure into a shaped closure using any layout. -/
def reshape (source : Vector length α) (layout : Layout space length) :
    Tensor space α :=
  fun index => source (Layout.offset layout index)

end Tensor

namespace Buffer

def named (name : String) : Buffer space length α := ⟨name⟩

end Buffer

namespace Matrix

def transpose (source : Matrix rows cols α) : Matrix cols rows α :=
  Tensor.reindex source fun index => ⟨Index2.col index, Index2.row index⟩

end Matrix

end GpuDsl
