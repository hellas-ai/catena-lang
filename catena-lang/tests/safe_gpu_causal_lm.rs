use std::{fs::File, io::Write, sync::mpsc, thread, time::Duration};

use catena_lang::safe_gpu::{
    Backend, GpuDialect, Session, SessionTimeouts,
    causal_lm::{
        GenerationControl, GenerationError, GenerationTermination, ModelConfig,
        minimum_generation_device_bytes,
    },
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

    let backend: Backend = std::env::var("CATENA_GPU_DIALECT")
        .unwrap_or_else(|_| "auto".into())
        .parse()
        .map_err(anyhow::Error::msg)?;
    let mut session = Session::with_backend(backend, SessionTimeouts::default())?;
    let dialect = session.dialect();
    eprintln!("resident model test backend: {dialect:?}");
    let program = session.prepare(SOURCE)?;
    let asset = session.attach([0x51; 32], weight_file(1)?)?;
    let weight = session.slice(&asset, 0, 8)?;
    let state_multipliers = [4];
    let asset_slices = [weight];
    let model_config = ModelConfig {
        entry_point: "causal-test",
        asset_slices: &asset_slices,
        state_byte_multipliers: &state_multipliers,
        vocabulary_size: 16,
        maximum_capacity: 16,
        // capacity 4 state (16) + conservative capacity-wide staging (32)
        // + generated next token/logits (72). The exact envelope also proves
        // generated allocation accounting resets before every later step.
        generation_device_allocation_budget_bytes: 120,
    };
    let model = session.bind_causal_lm(&program, model_config)?;
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

    // A destructor on another thread must return while this callback still
    // holds the generation lease. Its deferred ReleaseModel follows
    // ReleaseGeneration, and the next request proves framing stayed aligned.
    let callback_drop_model = session.bind_causal_lm(&program, model_config)?;
    let mut callback_drop_model = Some(callback_drop_model);
    let callback_drop = model.generate_tokens_streaming(&[1, 2], 1, &[], |_| {
        let callback_drop_model = callback_drop_model
            .take()
            .ok_or_else(|| anyhow::anyhow!("callback unexpectedly ran twice"))?;
        let (completed_sender, completed) = mpsc::sync_channel(1);
        let dropper = thread::spawn(move || {
            drop(callback_drop_model);
            let _ = completed_sender.send(());
        });
        completed
            .recv_timeout(Duration::from_secs(2))
            .map_err(|error| anyhow::anyhow!("cross-thread Model::drop blocked: {error}"))?;
        dropper
            .join()
            .map_err(|_| anyhow::anyhow!("cross-thread Model::drop panicked"))?;
        Ok(GenerationControl::Continue)
    })?;
    anyhow::ensure!(callback_drop.generated_tokens == [5]);
    anyhow::ensure!(callback_drop.termination == GenerationTermination::MaxNewTokens);
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
            generation_device_allocation_budget_bytes: 120,
        },
    )?;
    let fault = fault_model.generate_tokens(&[1], 1, &[]).unwrap_err();
    anyhow::ensure!(generation_killed_child(&fault));
    anyhow::ensure!(matches!(
        fault_model.generate_tokens(&[1], 1, &[]),
        Err(GenerationError::Resident(
            catena_lang::safe_runtime::ResidentError::Unavailable { .. }
        ))
    ));
    verify_known_insufficient_envelope_preserves_session(dialect)?;
    verify_generated_allocation_overflow_isolation(dialect)?;
    verify_generic_fault_isolation(dialect)?;
    Ok(())
}

fn verify_known_insufficient_envelope_preserves_session(dialect: GpuDialect) -> anyhow::Result<()> {
    let mut session = Session::new(dialect)?;
    let program = session.prepare(SOURCE)?;
    let asset = session.attach([0x52; 32], weight_file(1)?)?;
    let weight = session.slice(&asset, 0, 8)?;
    let state_multipliers = [4];
    let asset_slices = [weight];
    let minimum = minimum_generation_device_bytes(&state_multipliers, 2, 16)?;
    let undersized = session.bind_causal_lm(
        &program,
        ModelConfig {
            entry_point: "causal-test",
            asset_slices: &asset_slices,
            state_byte_multipliers: &state_multipliers,
            vocabulary_size: 16,
            maximum_capacity: 16,
            generation_device_allocation_budget_bytes: minimum - 1,
        },
    )?;
    let error = undersized.generate_tokens(&[1], 1, &[]).unwrap_err();
    anyhow::ensure!(matches!(
        error,
        GenerationError::InvalidRequest(ref message)
            if message.contains("requires at least 96 device bytes")
    ));
    anyhow::ensure!(!error.invalidates_session());

    let exact = session.bind_causal_lm(
        &program,
        ModelConfig {
            entry_point: "causal-test",
            asset_slices: &asset_slices,
            state_byte_multipliers: &state_multipliers,
            vocabulary_size: 16,
            maximum_capacity: 16,
            generation_device_allocation_budget_bytes: minimum,
        },
    )?;
    // The fixture returns head(prompt) + weight + capacity + position:
    // 1 + 1 + (one prompt + one generated token) + 0 = 4.
    anyhow::ensure!(exact.generate_tokens(&[1], 1, &[])?.generated_tokens == [4]);
    Ok(())
}

fn verify_generated_allocation_overflow_isolation(dialect: GpuDialect) -> anyhow::Result<()> {
    let runtime = SafeRuntime::new(dialect)?;
    let artifact = runtime.load_sources(stdlib::sources().chain([SOURCE]))?;
    let overflow = artifact
        .exec::<1, 1>("causal-test-logits", [u64::MAX.into()])
        .unwrap_err();
    let SafeExecError::ChildTerminated { stderr, .. } = overflow else {
        anyhow::bail!("allocation overflow did not terminate the child: {overflow:?}");
    };
    anyhow::ensure!(stderr.contains("byte count overflowed"));
    anyhow::ensure!(matches!(
        artifact.exec::<1, 1>("causal-test-logits", [1_u64.into()]),
        Err(SafeExecError::Unavailable { .. })
    ));
    Ok(())
}

fn verify_generic_fault_isolation(dialect: GpuDialect) -> anyhow::Result<()> {
    let runtime = SafeRuntime::new(dialect)?;
    let artifact = runtime.load_sources(stdlib::sources().chain([SOURCE]))?;
    anyhow::ensure!(matches!(
        artifact.exec::<1, 1>("causal-test-fault-logits", [16_u64.into()]),
        Err(SafeExecError::ChildTerminated { .. })
    ));
    anyhow::ensure!(matches!(
        artifact.exec::<1, 1>("causal-test-fault-logits", [16_u64.into()]),
        Err(SafeExecError::Unavailable { .. })
    ));
    Ok(())
}

fn generation_killed_child(error: &GenerationError) -> bool {
    match error {
        GenerationError::Resident(catena_lang::safe_runtime::ResidentError::ChildTerminated {
            ..
        }) => true,
        GenerationError::ReleaseAfterFailure { primary, .. } => generation_killed_child(primary),
        GenerationError::InvalidRequest(_)
        | GenerationError::Resident(_)
        | GenerationError::Callback(_) => false,
    }
}

fn weight_file(first: u64) -> anyhow::Result<File> {
    let mut temporary = tempfile::Builder::new()
        .prefix("causal-lm-weight-")
        .tempfile()?;
    temporary.write_all(&first.to_ne_bytes())?;
    temporary.as_file().set_len(4096)?;
    temporary.as_file().sync_all()?;
    let path = temporary.path().to_owned();
    let file = File::open(&path)?;
    temporary.close()?;
    anyhow::ensure!(!path.exists(), "temporary asset path still exists");
    Ok(file)
}
