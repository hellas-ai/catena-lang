use std::{fs::File, io::Write};

use catena_lang::safe_gpu::{
    GpuDialect, Session,
    causal_lm::{GenerationControl, GenerationError, GenerationTermination, ModelConfig},
    run_worker_if_requested,
};
use catena_lang::{
    safe_runtime::{SafeExecError, SafeRuntime},
    stdlib,
};

const SOURCE: &str = include_str!("fixtures/causal_lm.hex");

fn main() -> anyhow::Result<()> {
    if run_worker_if_requested()? {
        return Ok(());
    }

    let mut session = Session::new(GpuDialect::Hip)?;
    let program = session.prepare(SOURCE)?;
    let asset = session.attach([0x51; 32], weight_file(1)?)?;
    let weight = session.slice(&asset, 0, 8)?;
    let state_multipliers = [4];
    let asset_slices = [weight];
    let model = session.bind_causal_lm(
        &program,
        ModelConfig {
            entry_point: "causal-test",
            asset_slices: &asset_slices,
            state_byte_multipliers: &state_multipliers,
            vocabulary_size: 16,
            maximum_capacity: 16,
        },
    )?;
    anyhow::ensure!(model.vocabulary_size() == 16);
    anyhow::ensure!(model.maximum_capacity() == 16);

    // Each run starts with one zeroed, child-resident state allocation. The
    // second step consumes the replacement state without copying it to us.
    for _ in 0..2 {
        let result = model.generate_tokens(&[1, 2], 2, &[])?;
        anyhow::ensure!(result.generated_tokens == [6, 13]);
        anyhow::ensure!(result.termination == GenerationTermination::MaxNewTokens);
        anyhow::ensure!(result.stats.prompt_tokens == 2);
        anyhow::ensure!(result.stats.cached_tokens == 1);
    }

    let stopped = model.generate_tokens(&[1, 2], 2, &[6])?;
    anyhow::ensure!(stopped.generated_tokens.is_empty());
    anyhow::ensure!(stopped.termination == GenerationTermination::StopToken(6));

    let cancelled =
        model.generate_tokens_streaming(&[1, 2], 2, &[], |_| Ok(GenerationControl::Cancel))?;
    anyhow::ensure!(cancelled.generated_tokens == [6]);
    anyhow::ensure!(cancelled.termination == GenerationTermination::Cancelled);

    let callback_failure = model.generate_tokens_streaming(&[1, 2], 2, &[], |_| {
        anyhow::bail!("caller stopped decoding")
    });
    anyhow::ensure!(matches!(
        callback_failure,
        Err(GenerationError::Callback(_))
    ));
    // Callback failure dropped and released its generation; the model remains reusable.
    anyhow::ensure!(model.generate_tokens(&[1, 2], 1, &[])?.generated_tokens == [5]);

    // A device-side assertion must kill the worker. Returning a remote error
    // would incorrectly leave a poisoned GPU context reusable.
    let fault_model = session.bind_causal_lm(
        &program,
        ModelConfig {
            entry_point: "causal-test-fault",
            asset_slices: &asset_slices,
            state_byte_multipliers: &state_multipliers,
            vocabulary_size: 16,
            maximum_capacity: 16,
        },
    )?;
    anyhow::ensure!(matches!(
        fault_model.generate_tokens(&[1], 1, &[]),
        Err(GenerationError::Resident(
            catena_lang::safe_runtime::ResidentError::ChildTerminated { .. }
        ))
    ));
    anyhow::ensure!(matches!(
        fault_model.generate_tokens(&[1], 1, &[]),
        Err(GenerationError::Resident(
            catena_lang::safe_runtime::ResidentError::Unavailable { .. }
        ))
    ));
    verify_generic_fault_isolation()?;
    Ok(())
}

fn verify_generic_fault_isolation() -> anyhow::Result<()> {
    let runtime = SafeRuntime::new(GpuDialect::Hip)?;
    let artifact = runtime.load_sources(stdlib::sources().chain([SOURCE]))?;
    anyhow::ensure!(matches!(
        runtime.exec::<1, 1>(&artifact, "causal-test-fault-logits", [16_u64.into()]),
        Err(SafeExecError::ChildTerminated { .. })
    ));
    anyhow::ensure!(matches!(
        runtime.exec::<1, 1>(&artifact, "causal-test-fault-logits", [16_u64.into()]),
        Err(SafeExecError::Unavailable { .. })
    ));
    Ok(())
}

fn weight_file(first: u64) -> anyhow::Result<File> {
    let mut temporary = tempfile::NamedTempFile::new()?;
    temporary.write_all(&first.to_ne_bytes())?;
    temporary.as_file().set_len(4096)?;
    temporary.as_file().sync_all()?;
    let path = temporary.path().to_owned();
    let file = File::open(&path)?;
    temporary.close()?;
    anyhow::ensure!(!path.exists(), "temporary asset path still exists");
    Ok(file)
}
