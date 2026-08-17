import GpuDsl

open GpuDsl

/-! Top-level, type-checked examples of the opaque `launch` primitive. -/

noncomputable def perfect64 : Launch (64 * 64) Float :=
  let config := perfectConfig 64 64 16
  let aBuffer : Buffer .global (64 * 64) Float := Buffer.named "A"
  let bBuffer : Buffer .global (64 * 64) Float := Buffer.named "B"
  let out : Buffer .global (64 * 64) Float := Buffer.named "C"
  let a : Matrix 64 64 (Value Float) := Buffer.asRowMajorMatrix aBuffer
  let b : Matrix 64 64 (Value Float) := Buffer.asColumnMajorMatrix bBuffer
  let certificate := tiledScheduleCertificate config 64 64 16
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
  let kernel := tiledMatmulKernel 16 (by decide) a b
  launch config out Layout.rowMajor2D certificate kernel (by
    simp [kernel, tiledMatmulKernel, config, perfectConfig])

noncomputable def predicatedEdges : Launch (65 * 33) Float :=
  let config := predicatedConfig 65 33 16
  let aBuffer : Buffer .global (65 * 70) Float := Buffer.named "A"
  let bBuffer : Buffer .global (70 * 33) Float := Buffer.named "B"
  let out : Buffer .global (65 * 33) Float := Buffer.named "C"
  let a : Matrix 65 70 (Value Float) := Buffer.asRowMajorMatrix aBuffer
  let b : Matrix 70 33 (Value Float) := Buffer.asRowMajorMatrix bBuffer
  let certificate := tiledScheduleCertificate config 65 33 16
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
  let kernel := tiledMatmulKernel 16 (by decide) a b
  launch config out Layout.rowMajor2D certificate kernel (by
    simp [kernel, tiledMatmulKernel, config, predicatedConfig])

def main : IO Unit := do
  IO.println "type-checked perfect and predicated tiled-matmul launches"
