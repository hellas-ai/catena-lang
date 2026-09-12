use std::{fs::File, io::Write};

use catena_lang::safe_gpu::{
    Asset, AssetError, AssetOwner, Backend, Session, SessionTimeouts,
    causal_lm::{GenerationControl, GenerationError, Model, ModelConfig},
    run_worker_if_requested,
};

fn main() -> anyhow::Result<()> {
    if run_worker_if_requested()? {
        return Ok(());
    }

    let functional_only = std::env::args()
        .skip(1)
        .any(|arg| arg == "--functional-only");
    if functional_only {
        eprintln!("functional-only selected: deliberate worker fault coverage is skipped");
    }
    let backend: Backend = std::env::var("CATENA_GPU_DIALECT")
        .unwrap_or_else(|_| "auto".into())
        .parse()
        .map_err(anyhow::Error::msg)?;
    let session = Session::with_backend(backend, SessionTimeouts::default())?;
    eprintln!("resident asset test backend: {:?}", session.dialect());
    let key = [0x42; 32];
    let asset = session.attach(key, unlinked_read_only_file(&[7; 4096])?)?;
    anyhow::ensure!(asset.byte_len() == 4096);
    anyhow::ensure!(asset == asset.clone());

    let slice = session.slice(&asset, 128, 1024)?;
    anyhow::ensure!(slice.offset() == 128);
    anyhow::ensure!(slice.byte_len() == 1024);
    anyhow::ensure!(session.slice(&asset, 4096, 0).is_ok());
    anyhow::ensure!(matches!(
        session.slice(&asset, 4090, 8),
        Err(AssetError::InvalidSlice { .. })
    ));
    anyhow::ensure!(matches!(
        session.slice(&asset, u64::MAX, 2),
        Err(AssetError::InvalidSlice { .. })
    ));

    // A stable opaque handle proves the resident mapping was reused rather
    // than registered again for the same content identity.
    let reused = session.attach(key, unlinked_read_only_file(&[7; 4096])?)?;
    anyhow::ensure!(reused == asset);

    let conflicting = session.attach(key, unlinked_read_only_file(&[7; 8192])?);
    anyhow::ensure!(matches!(conflicting, Err(AssetError::Remote(_))));

    let empty = session.attach([0x43; 32], unlinked_read_only_file(&[])?);
    anyhow::ensure!(matches!(empty, Err(AssetError::InvalidLength { .. })));
    let writable = tempfile::tempfile()?;
    writable.set_len(4096)?;
    anyhow::ensure!(matches!(
        session.attach([0x44; 32], writable),
        Err(AssetError::Remote(_))
    ));

    let other_session = Session::new(session.dialect())?;
    anyhow::ensure!(matches!(
        other_session.slice(&asset, 0, 1),
        Err(AssetError::WrongSession)
    ));
    let dialect = session.dialect();
    drop(other_session);
    drop(session);
    verify_remote_artifact_lifetime(dialect)?;
    verify_shared_owner(dialect, functional_only)?;
    Ok(())
}

fn unlinked_read_only_file(bytes: &[u8]) -> anyhow::Result<File> {
    let mut temporary = tempfile::NamedTempFile::new()?;
    temporary.write_all(bytes)?;
    temporary.as_file().sync_all()?;
    let path = temporary.path().to_owned();
    let file = File::open(&path)?;
    temporary.close()?;
    anyhow::ensure!(!path.exists(), "temporary asset path still exists");
    Ok(file)
}

fn verify_shared_owner(
    dialect: catena_lang::safe_gpu::GpuDialect,
    functional_only: bool,
) -> anyhow::Result<()> {
    let owner = AssetOwner::new(dialect)?;
    let mut source = tempfile::NamedTempFile::new()?;
    source.write_all(&1_u64.to_ne_bytes())?;
    source.as_file().set_len(4096)?;
    source.as_file().sync_all()?;
    let source_path = source.path().to_owned();
    let asset = owner.attach([0x61; 32], File::open(&source_path)?)?;
    source.close()?;
    anyhow::ensure!(!source_path.exists(), "source survived first owner upload");

    let initial_usage = owner.usage()?;
    anyhow::ensure!(initial_usage.allocation_count == 1);
    anyhow::ensure!(initial_usage.uploaded_bytes == 4096);
    anyhow::ensure!(initial_usage.device_bytes >= 4096);
    let mut identical = [0_u8; 4096];
    identical[..8].copy_from_slice(&1_u64.to_ne_bytes());
    let reattached = owner.attach([0x61; 32], unlinked_read_only_file(&identical)?)?;
    anyhow::ensure!(reattached == asset);
    anyhow::ensure!(
        owner.usage()? == initial_usage,
        "reattach uploaded the asset twice"
    );

    let mut first = Session::with_assets(&owner, SessionTimeouts::default())?;
    let mut second = Session::with_assets(&owner, SessionTimeouts::default())?;
    anyhow::ensure!(first.slice(&asset, 0, 8)? == second.slice(&asset, 0, 8)?);
    let first_model = bind_shared_model(&mut first, &asset, "shared-state-test")?;
    let second_model = bind_shared_model(&mut second, &asset, "shared-state-test")?;
    anyhow::ensure!(
        owner.usage()? == initial_usage,
        "session imports duplicated owner allocations"
    );

    // A pauses after writing state=2. B writes its independent state=5 and
    // completes [5, 11] before A resumes. A must still read its own state=2
    // and return 5; sharing mutable state would change that second token.
    // Releasing the final Program handle during A's active generation must
    // enqueue its release instead of trying to reacquire the generation lease.
    let mut unused_program =
        Some(first.prepare(
            "(def program unused-during-generation : (bool val) -> (bool val) = bool.not)",
        )?);
    let mut callbacks = 0;
    let interleaved = first_model.generate_tokens_streaming(&[1], 2, &[], |_| {
        callbacks += 1;
        if callbacks == 1 {
            drop(unused_program.take());
            anyhow::ensure!(
                second_model.generate_tokens(&[4], 2, &[])?.generated_tokens == [5, 11]
            );
        }
        Ok(GenerationControl::Continue)
    })?;
    anyhow::ensure!(callbacks == 2);
    anyhow::ensure!(unused_program.is_none());
    anyhow::ensure!(interleaved.generated_tokens == [2, 5]);
    anyhow::ensure!(
        owner.usage()? == initial_usage,
        "generation changed weight allocation usage"
    );
    anyhow::ensure!(second_model.generate_tokens(&[1], 2, &[])?.generated_tokens == [2, 5]);

    if !functional_only {
        let fault_model = bind_shared_model(&mut first, &asset, "causal-test-fault")?;
        let fault = fault_model.generate_tokens(&[1], 1, &[]).unwrap_err();
        anyhow::ensure!(
            generation_killed_child(&fault),
            "unexpected fault result: {fault:?}"
        );
        anyhow::ensure!(matches!(
            first_model.generate_tokens(&[1], 1, &[]),
            Err(GenerationError::Resident(
                catena_lang::safe_runtime::ResidentError::Unavailable { .. }
            ))
        ));
        anyhow::ensure!(
            owner.is_available(),
            "execution fault killed the asset owner"
        );
        anyhow::ensure!(second_model.generate_tokens(&[1], 2, &[])?.generated_tokens == [2, 5]);
        anyhow::ensure!(
            owner.usage()? == initial_usage,
            "worker fault changed owner allocations"
        );
        drop(fault_model);
    }
    drop(first_model);
    drop(first);

    // The replacement receives the original opaque asset. No file descriptor
    // or path is supplied again, and the surviving session still uses it too.
    let mut replacement = Session::with_assets(&owner, SessionTimeouts::default())?;
    anyhow::ensure!(replacement.slice(&asset, 0, 8)? == second.slice(&asset, 0, 8)?);
    let replacement_model = bind_shared_model(&mut replacement, &asset, "shared-state-test")?;
    anyhow::ensure!(
        replacement_model
            .generate_tokens(&[1], 2, &[])?
            .generated_tokens
            == [2, 5]
    );
    anyhow::ensure!(second_model.generate_tokens(&[4], 2, &[])?.generated_tokens == [5, 11]);
    let final_usage = owner.usage()?;
    anyhow::ensure!(
        final_usage == initial_usage,
        "replacement duplicated owner allocations"
    );
    eprintln!(
        "shared owner after reuse/replacement (fault_tested={}): allocations={}, uploaded_bytes={}, device_bytes={}",
        !functional_only,
        final_usage.allocation_count,
        final_usage.uploaded_bytes,
        final_usage.device_bytes,
    );
    Ok(())
}

fn bind_shared_model(
    session: &mut Session,
    asset: &Asset,
    entry_point: &str,
) -> anyhow::Result<Model> {
    let source = concat!(
        include_str!("fixtures/causal_lm.hex"),
        "\n",
        include_str!("fixtures/shared_asset_state.hex"),
    );
    let program = session.prepare(source)?;
    let slices = [session.slice(asset, 0, 8)?];
    let model = session.bind_causal_lm(
        &program,
        ModelConfig {
            entry_point,
            asset_slices: &slices,
            state_byte_multipliers: &[8],
            vocabulary_size: 16,
            maximum_capacity: 16,
            generation_device_allocation_budget_bytes: 4096,
        },
    )?;
    drop(program);
    Ok(model)
}

fn generation_killed_child(error: &GenerationError) -> bool {
    match error {
        GenerationError::Resident(catena_lang::safe_runtime::ResidentError::ChildTerminated {
            ..
        }) => true,
        GenerationError::ReleaseAfterFailure { primary, .. } => generation_killed_child(primary),
        _ => false,
    }
}

fn verify_remote_artifact_lifetime(
    dialect: catena_lang::safe_gpu::GpuDialect,
) -> anyhow::Result<()> {
    use catena_lang::{runtime::Value, safe_runtime::SafeRuntime, stdlib};

    let runtime = SafeRuntime::new(dialect)?;
    let artifact = runtime.load_sources(stdlib::sources().chain([
        "(def program lifetime-add-one : (u64 val) -> (u64 val) = ({_ u64.one} u64.add))",
    ]))?;
    let sibling = runtime.load_sources(
        stdlib::sources()
            .chain(["(def program lifetime-identity : (u64 val) -> (u64 val) = [value])"]),
    )?;
    drop(runtime);
    drop(sibling);

    anyhow::ensure!(
        artifact
            .entry_points()
            .iter()
            .any(|entry| entry.name() == "lifetime-add-one")
    );
    let [result] = artifact.exec("lifetime-add-one", [41_u64.into()])?;
    anyhow::ensure!(matches!(result, Value::U64(42)));
    let outputs = artifact.exec_values("lifetime-add-one", vec![6_u64.into()])?;
    anyhow::ensure!(matches!(outputs.as_slice(), [Value::U64(7)]));
    Ok(())
}
