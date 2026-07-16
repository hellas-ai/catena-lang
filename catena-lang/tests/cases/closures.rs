use super::*;

const CONDITIONALS: &str = include_str!("closures/conditionals.hex");
const PROGRESSION: &str = include_str!("closures/progression.hex");
const SEQUENCE_STATS: &str = include_str!("closures/sequence_stats.hex");

/// A useful conditional with captured branches exercises `bool.if` closure
/// conversion for values both above and below the requested floor.
#[test]
fn max_with_floor_exec() -> anyhow::Result<()> {
    let runtime = runtime_with(CONDITIONALS)?;

    for (value, floor, expected) in [
        (3.5_f32, 2.0_f32, 3.5_f32),
        (-4.0, -1.5, -1.5),
        (2.25, 2.25, 2.25),
    ] {
        let [result] = runtime.exec("max-with-floor", [value.into(), floor.into()])?;
        let Value::F32(result) = result else {
            anyhow::bail!("max-with-floor returned non-f32 value: {result:?}");
        };
        assert_eq!(result, expected, "max-with-floor({value}, {floor})");
    }

    Ok(())
}

/// A product-valued conditional checks that the variadic boundary recorded for
/// `bool.ifc` becomes two independently returned runtime values.
#[test]
fn ordered_pair_exec() -> anyhow::Result<()> {
    let runtime = runtime_with(CONDITIONALS)?;

    for (lhs, rhs, expected) in [
        (3.5_f32, 2.0_f32, (2.0_f32, 3.5_f32)),
        (-4.0, -1.5, (-4.0, -1.5)),
        (2.25, 2.25, (2.25, 2.25)),
    ] {
        let [lower, upper] = runtime.exec("ordered-pair", [lhs.into(), rhs.into()])?;
        let Value::F32(lower) = lower else {
            anyhow::bail!("ordered-pair returned non-f32 lower value: {lower:?}");
        };
        let Value::F32(upper) = upper else {
            anyhow::bail!("ordered-pair returned non-f32 upper value: {upper:?}");
        };
        assert_eq!((lower, upper), expected, "ordered-pair({lhs}, {rhs})");
    }

    Ok(())
}

/// The indexed producer captures two runtime values while source-level
/// `reduce` supplies its context and lowers it to an explicit environment.
#[test]
fn sum_progression_exec() -> anyhow::Result<()> {
    let runtime = runtime_with(PROGRESSION)?;

    for (length, base, step, expected) in
        [(0_u64, 7_u64, 3_u64, 0_u64), (4, 2, 3, 26), (5, 1, 1, 15)]
    {
        let [result] =
            runtime.exec("sum-progression", [length.into(), base.into(), step.into()])?;
        let Value::U64(result) = result else {
            anyhow::bail!("sum-progression returned non-u64 value: {result:?}");
        };
        assert_eq!(
            result, expected,
            "sum-progression({length}, {base}, {step})"
        );
    }

    Ok(())
}

/// A captured indexed producer and product accumulator compute two useful
/// statistics in one pass, covering flattened environments and outputs.
#[test]
fn progression_moments_exec() -> anyhow::Result<()> {
    let runtime = runtime_with_sources([PROGRESSION, SEQUENCE_STATS])?;

    for (length, base, step, expected_sum, expected_squares) in [
        (0_u64, 7_u64, 3_u64, 0_u64, 0_u64),
        (4, 2, 3, 26, 214),
        (3, 1, 1, 6, 14),
    ] {
        let [sum, squares] = runtime.exec(
            "progression-moments",
            [length.into(), base.into(), step.into()],
        )?;
        let Value::U64(sum) = sum else {
            anyhow::bail!("progression-moments returned non-u64 sum: {sum:?}");
        };
        let Value::U64(squares) = squares else {
            anyhow::bail!("progression-moments returned non-u64 squares: {squares:?}");
        };
        assert_eq!(sum, expected_sum, "sum for ({length}, {base}, {step})");
        assert_eq!(
            squares, expected_squares,
            "sum of squares for ({length}, {base}, {step})"
        );
    }

    Ok(())
}
