use catena_lang::runtime::Value;

use super::support::runtime_with_fixture;

// Rectangular matrices and multiple blocks/inner tiles distinguish all three
// dimensions. Predicated cases leave partial tiles in rows, columns and inner.
const PERFECT: &[(u32, u32, u32, u32)] = &[(4, 6, 8, 2), (8, 12, 4, 4)];
const PREDICATED: &[(u32, u32, u32, u32)] = &[(3, 5, 7, 2), (5, 7, 3, 4), (1, 1, 1, 4)];

#[test]
fn matmul_naive_u64_perfect_tiling() -> anyhow::Result<()> {
    check_matmul("matmul_naive_u64.hex", "matmul_naive_u64.matmul", PERFECT)
}

#[test]
fn matmul_naive_u64_predicated_tiling() -> anyhow::Result<()> {
    check_matmul(
        "matmul_naive_u64.hex",
        "matmul_naive_u64.matmul",
        PREDICATED,
    )
}

#[test]
fn matmul_tiled_u64_perfect_tiling() -> anyhow::Result<()> {
    check_matmul("matmul_tiled_u64.hex", "matmul_tiled_u64.matmul", PERFECT)
}

#[test]
fn matmul_tiled_u64_predicated_tiling() -> anyhow::Result<()> {
    check_matmul(
        "matmul_tiled_u64.hex",
        "matmul_tiled_u64.matmul",
        PREDICATED,
    )
}

fn check_matmul(fixture: &str, entry: &str, cases: &[(u32, u32, u32, u32)]) -> anyhow::Result<()> {
    let (runtime, artifact) = runtime_with_fixture(fixture)?;
    for &(rows, inner, columns, width) in cases {
        let (m, k, n) = (rows as usize, inner as usize, columns as usize);
        let a_values: Vec<u64> = (0..m * k).map(|i| ((i * 3 + 1) % 11) as u64).collect();
        let b_values: Vec<u64> = (0..k * n).map(|i| ((i * 5 + 2) % 13) as u64).collect();
        let expected: Vec<u64> = (0..m * n)
            .map(|i| {
                (0..k)
                    .map(|j| a_values[(i / n) * k + j] * b_values[j * n + i % n])
                    .sum()
            })
            .collect();

        let a = runtime.mem_u64(&a_values)?;
        let b = runtime.mem_u64(&b_values)?;
        // A nonzero sentinel exposes unwritten output elements.
        let c = runtime.mem_u64(&vec![u64::MAX; m * n])?;
        let [result] = artifact.exec(
            entry,
            [
                rows.into(),
                inner.into(),
                columns.into(),
                a.as_ref().into(),
                b.as_ref().into(),
                c.into(),
                width.into(),
            ],
        )?;
        let Value::MemOwn(result) = result else {
            anyhow::bail!("{entry} returned a non-owned-memory value: {result:?}");
        };
        assert_eq!(
            result.try_to_u64_vec()?,
            expected,
            "{entry}: rows={rows}, inner={inner}, columns={columns}, width={width}"
        );
        assert_eq!(a.try_to_u64_vec()?, a_values, "{entry} modified A");
        assert_eq!(b.try_to_u64_vec()?, b_values, "{entry} modified B");
    }
    Ok(())
}
