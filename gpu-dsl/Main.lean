import GpuDsl

open GpuDsl

/-! Top-level, type-checked examples of the opaque `launch` primitive. -/

noncomputable def perfect64 : Launch :=
  let config := perfectConfig 64 64 16
  let aBuffer : Buffer .global (64 * 64) Float := Buffer.named "A"
  let bBuffer : Buffer .global (64 * 64) Float := Buffer.named "B"
  let out : Buffer .global (64 * 64) Float := Buffer.named "C"
  let inputs : MatmulInputs 64 64 64 Float :=
    { a := aBuffer
      aLayout := Layout.rowMajor2D
      b := bBuffer
      bLayout := Layout.columnMajor2D
      output := out
      aOutputDistinct := by decide
      bOutputDistinct := by decide }
  let kernel := tiledMatmulKernel (rows := 64) (inner := 64) (cols := 64)
    16 (by decide)
  launch config kernel inputs (by
    simp [kernel, tiledMatmulKernel, config, perfectConfig])

noncomputable def predicatedEdges : Launch :=
  let config := predicatedConfig 65 33 16
  let aBuffer : Buffer .global (65 * 70) Float := Buffer.named "A"
  let bBuffer : Buffer .global (70 * 33) Float := Buffer.named "B"
  let out : Buffer .global (65 * 33) Float := Buffer.named "C"
  let inputs : MatmulInputs 65 70 33 Float :=
    { a := aBuffer
      aLayout := Layout.rowMajor2D
      b := bBuffer
      bLayout := Layout.rowMajor2D
      output := out
      aOutputDistinct := by decide
      bOutputDistinct := by decide }
  let kernel := tiledMatmulKernel (rows := 65) (inner := 70) (cols := 33)
    16 (by decide)
  launch config kernel inputs (by
    simp [kernel, tiledMatmulKernel, config, predicatedConfig, ceilDiv])

def main : IO Unit := do
  IO.println "type-checked perfect and predicated tiled-matmul launches"
