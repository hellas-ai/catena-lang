use std::{collections::HashMap, sync::Arc};

use anyhow::{Context, Result, bail, ensure};

use super::{
    assets::AssetStore,
    protocol::{WireAssetSlice, WireModelBinding},
};
use crate::runtime::{Artifact, EntryPoint, ExecError, MemOwn, Runtime, Value, ValueKind};

pub(crate) const MAX_MODEL_STATES: usize = 8;
pub(crate) const MAX_MODEL_ASSET_SLICES: usize = 4096;
pub(crate) const MAX_MODEL_STATIC_BYTES: u64 = 1 << 40;
pub(crate) const MAX_MODEL_STATE_BYTES: u64 = 1 << 40;
pub(crate) const MAXIMUM_CAPACITY: u64 = 8_388_608;
pub(crate) const MAX_STOP_TOKEN_IDS: usize = 256;
pub(crate) const MAX_ENTRY_POINT_BYTES: usize = 1024;

const TOKEN_STAGING_ELEMENT_BYTES: u64 = std::mem::size_of::<u64>() as u64;
const LOGIT_ELEMENT_BYTES: u64 = std::mem::size_of::<f32>() as u64;
const NEXT_TOKEN_BYTES: u64 = std::mem::size_of::<u64>() as u64;

const MAX_RESIDENT_MODELS: usize = 256;
const MAX_RESIDENT_GENERATIONS: usize = 256;

/// Arithmetic failures while calculating the causal-LM device-memory floor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum GenerationDeviceEnvelopeError {
    #[error("generation state byte count overflowed")]
    StateBytesOverflow,
    #[error("generation state exceeds the {maximum}-byte limit")]
    StateBytesLimit { maximum: u64 },
    #[error("generation token-staging byte count overflowed")]
    TokenStagingBytesOverflow,
    #[error("generation logits byte count overflowed")]
    LogitsBytesOverflow,
    #[error("minimum generation device byte count overflowed")]
    TotalBytesOverflow,
}

/// Return the conservative device-memory floor for one causal-LM generation.
///
/// The floor covers capacity-scaled resident state, a worst-case token batch of
/// `capacity` u64 IDs, mandatory f32 logits, and one u64 next-token output.
/// Replacement state may reuse its resident input allocation. Generated
/// intermediates are deliberately not guessed here; the runtime continues to
/// charge every generated allocation cumulatively against the configured
/// envelope during each forward.
pub fn minimum_generation_device_bytes(
    state_byte_multipliers: &[u64],
    capacity: u64,
    vocabulary_size: u64,
) -> std::result::Result<u64, GenerationDeviceEnvelopeError> {
    let state_bytes = generation_state_bytes(state_byte_multipliers, capacity)?;
    let token_staging_bytes = capacity
        .checked_mul(TOKEN_STAGING_ELEMENT_BYTES)
        .ok_or(GenerationDeviceEnvelopeError::TokenStagingBytesOverflow)?;
    let logits_bytes = vocabulary_size
        .checked_mul(LOGIT_ELEMENT_BYTES)
        .ok_or(GenerationDeviceEnvelopeError::LogitsBytesOverflow)?;
    state_bytes
        .checked_add(token_staging_bytes)
        .and_then(|bytes| bytes.checked_add(logits_bytes))
        .and_then(|bytes| bytes.checked_add(NEXT_TOKEN_BYTES))
        .ok_or(GenerationDeviceEnvelopeError::TotalBytesOverflow)
}

fn generation_state_bytes(
    state_byte_multipliers: &[u64],
    capacity: u64,
) -> std::result::Result<u64, GenerationDeviceEnvelopeError> {
    state_byte_multipliers
        .iter()
        .try_fold(0_u64, |total, &multiplier| {
            let bytes = capacity
                .checked_mul(multiplier)
                .ok_or(GenerationDeviceEnvelopeError::StateBytesOverflow)?;
            total
                .checked_add(bytes)
                .ok_or(GenerationDeviceEnvelopeError::StateBytesOverflow)
                .and_then(|total| {
                    if total <= MAX_MODEL_STATE_BYTES {
                        Ok(total)
                    } else {
                        Err(GenerationDeviceEnvelopeError::StateBytesLimit {
                            maximum: MAX_MODEL_STATE_BYTES,
                        })
                    }
                })
        })
}

struct BoundModel {
    artifact: Arc<Artifact>,
    entry_point: String,
    assets: Vec<WireAssetSlice>,
    state_byte_multipliers: Vec<u64>,
    vocabulary_size: u64,
    maximum_capacity: u64,
    generation_device_allocation_budget_bytes: u64,
}

struct Generation {
    model: u64,
    states: Vec<MemOwn>,
    state_byte_len: u64,
    capacity: u64,
    position: u64,
}

pub(super) struct ResidentStore {
    models: HashMap<u64, BoundModel>,
    generations: HashMap<u64, Generation>,
    next_model: u64,
    next_generation: u64,
}

impl ResidentStore {
    pub(super) fn new() -> Self {
        Self {
            models: HashMap::new(),
            generations: HashMap::new(),
            next_model: 1,
            next_generation: 1,
        }
    }

    pub(super) fn bind_model(
        &mut self,
        artifact: Arc<Artifact>,
        assets: &AssetStore,
        binding: WireModelBinding,
    ) -> Result<u64> {
        ensure!(
            self.models.len() < MAX_RESIDENT_MODELS,
            "session already has the maximum of {MAX_RESIDENT_MODELS} resident models"
        );
        let entry_point = artifact
            .entry_points()
            .iter()
            .find(|entry| entry.name() == binding.entry_point)
            .with_context(|| format!("program has no entry point {:?}", binding.entry_point))?;
        validate_model_binding(entry_point, &binding)?;
        for slice in &binding.assets {
            let _ = assets.mem_ref(slice.asset, slice.offset, slice.byte_len)?;
        }

        let id = take_id(&mut self.next_model, "model")?;
        self.models.insert(
            id,
            BoundModel {
                artifact,
                entry_point: binding.entry_point,
                assets: binding.assets,
                state_byte_multipliers: binding.state_byte_multipliers,
                vocabulary_size: binding.vocabulary_size,
                maximum_capacity: binding.maximum_capacity,
                generation_device_allocation_budget_bytes: binding
                    .generation_device_allocation_budget_bytes,
            },
        );
        Ok(id)
    }

    pub(super) fn start_generation(
        &mut self,
        runtime: &Runtime,
        model: u64,
        capacity: u64,
    ) -> Result<u64> {
        ensure!(
            self.generations.len() < MAX_RESIDENT_GENERATIONS,
            "session already has the maximum of {MAX_RESIDENT_GENERATIONS} resident generations"
        );
        let model_spec = self
            .models
            .get(&model)
            .with_context(|| format!("unknown resident model {model}"))?;
        ensure!(capacity != 0, "generation capacity must be non-zero");
        ensure!(
            capacity <= model_spec.maximum_capacity,
            "generation capacity {capacity} exceeds model maximum {}",
            model_spec.maximum_capacity
        );

        // This pure validation is deliberately complete before the first GPU
        // allocation. A known-insufficient envelope is a structured request
        // rejection and leaves the resident worker healthy.
        let (state_byte_lens, state_byte_len) = checked_generation_layout(
            &model_spec.state_byte_multipliers,
            capacity,
            model_spec.vocabulary_size,
            model_spec.generation_device_allocation_budget_bytes,
        )?;

        let mut states = Vec::with_capacity(state_byte_lens.len());
        for byte_len in state_byte_lens {
            states.push(runtime.mem_zeroed_bytes(byte_len)?);
        }

        let id = take_id(&mut self.next_generation, "generation")?;
        self.generations.insert(
            id,
            Generation {
                model,
                states,
                state_byte_len,
                capacity,
                position: 0,
            },
        );
        Ok(id)
    }

    pub(super) fn step_generation(
        &mut self,
        runtime: &Runtime,
        assets: &AssetStore,
        generation: u64,
        tokens: Vec<u32>,
    ) -> Result<u32> {
        let mut generation_state = self
            .generations
            .remove(&generation)
            .with_context(|| format!("unknown or poisoned generation {generation}"))?;
        let model = self
            .models
            .get(&generation_state.model)
            .with_context(|| format!("unknown resident model {}", generation_state.model))?;
        let result = forward(runtime, assets, model, &mut generation_state, tokens);
        if result.is_ok() {
            self.generations.insert(generation, generation_state);
        }
        result
    }

    pub(super) fn release_generation(&mut self, generation: u64) {
        self.generations.remove(&generation);
    }

    pub(super) fn release_model(&mut self, model: u64) {
        self.models.remove(&model);
        self.generations
            .retain(|_, generation| generation.model != model);
    }
}

fn generation_state_layout(
    state_byte_multipliers: &[u64],
    capacity: u64,
) -> Result<(Vec<u64>, u64)> {
    let state_byte_lens = state_byte_multipliers
        .iter()
        .map(|&multiplier| {
            capacity
                .checked_mul(multiplier)
                .context("generation state byte count overflowed")
        })
        .collect::<Result<Vec<_>>>()?;
    let state_byte_len = generation_state_bytes(state_byte_multipliers, capacity)?;
    Ok((state_byte_lens, state_byte_len))
}

fn checked_generation_layout(
    state_byte_multipliers: &[u64],
    capacity: u64,
    vocabulary_size: u64,
    allocation_envelope: u64,
) -> Result<(Vec<u64>, u64)> {
    let minimum_device_bytes =
        minimum_generation_device_bytes(state_byte_multipliers, capacity, vocabulary_size)?;
    ensure!(
        minimum_device_bytes <= allocation_envelope,
        "causal-lm generation requires at least {minimum_device_bytes} device bytes at capacity {capacity}, exceeding its {allocation_envelope}-byte allocation envelope"
    );
    generation_state_layout(state_byte_multipliers, capacity)
}

fn remaining_generated_allocation_budget(
    state_byte_len: u64,
    token_count: u64,
    allocation_envelope: u64,
) -> Result<u64> {
    let token_staging_byte_len = token_count
        .checked_mul(TOKEN_STAGING_ELEMENT_BYTES)
        .context("token staging byte count overflowed")?;
    let resident_and_staging_bytes = state_byte_len
        .checked_add(token_staging_byte_len)
        .context("generation device byte count overflowed")?;
    ensure!(
        resident_and_staging_bytes <= allocation_envelope,
        "resident state and token staging require {resident_and_staging_bytes} device bytes, exceeding the generation's {allocation_envelope}-byte allocation envelope"
    );
    Ok(allocation_envelope - resident_and_staging_bytes)
}

fn validate_model_binding(entry_point: &EntryPoint, binding: &WireModelBinding) -> Result<()> {
    validate_model_spec(
        entry_point,
        &binding.entry_point,
        &binding.state_byte_multipliers,
        binding.assets.len(),
        binding.assets.iter().map(|slice| slice.byte_len),
        binding.vocabulary_size,
        binding.maximum_capacity,
    )
}

pub(crate) fn validate_model_spec(
    entry_point: &EntryPoint,
    entry_point_name: &str,
    state_byte_multipliers: &[u64],
    asset_count: usize,
    asset_byte_lens: impl IntoIterator<Item = u64>,
    vocabulary_size: u64,
    maximum_capacity: u64,
) -> Result<()> {
    ensure!(
        !entry_point_name.is_empty() && entry_point_name.len() <= MAX_ENTRY_POINT_BYTES,
        "entry-point name must contain 1..={MAX_ENTRY_POINT_BYTES} bytes"
    );
    ensure!(
        entry_point.name() == entry_point_name,
        "entry-point metadata does not match {entry_point_name:?}"
    );
    ensure!(
        state_byte_multipliers.len() <= MAX_MODEL_STATES,
        "model declares more than {MAX_MODEL_STATES} states"
    );
    ensure!(
        asset_count <= MAX_MODEL_ASSET_SLICES,
        "model declares more than {MAX_MODEL_ASSET_SLICES} asset slices"
    );
    ensure!(
        vocabulary_size != 0 && vocabulary_size <= u64::from(u32::MAX) + 1,
        "vocabulary size must fit the u32 token-ID space"
    );
    ensure!(
        maximum_capacity != 0 && maximum_capacity <= MAXIMUM_CAPACITY,
        "maximum capacity must be between 1 and {MAXIMUM_CAPACITY}"
    );

    let mut state_bytes = 0_u64;
    for (index, &multiplier) in state_byte_multipliers.iter().enumerate() {
        ensure!(
            multiplier != 0 && multiplier.is_multiple_of(4),
            "state multiplier {index} must be a non-zero multiple of four bytes"
        );
        let bytes = maximum_capacity
            .checked_mul(multiplier)
            .context("model state byte count overflowed")?;
        state_bytes = state_bytes
            .checked_add(bytes)
            .filter(|total| *total <= MAX_MODEL_STATE_BYTES)
            .context("model state exceeds the byte limit at maximum capacity")?;
    }
    asset_byte_lens
        .into_iter()
        .try_fold(0_u64, |total, byte_len| {
            total
                .checked_add(byte_len)
                .filter(|total| *total <= MAX_MODEL_STATIC_BYTES)
                .context("model asset slices exceed the aggregate byte limit")
        })?;

    validate_abi(entry_point, state_byte_multipliers.len(), asset_count)
}

fn validate_abi(entry_point: &EntryPoint, states: usize, assets: usize) -> Result<()> {
    let input_count = 3_usize
        .checked_add(states)
        .and_then(|count| count.checked_add(assets))
        .context("model input count overflowed")?;
    ensure!(
        entry_point.inputs().len() == input_count,
        "entry point {:?} has {} inputs, causal-lm requires {input_count}",
        entry_point.name(),
        entry_point.inputs().len()
    );
    let mut inputs = entry_point.inputs().iter().copied();
    ensure!(
        inputs.next() == Some(ValueKind::MemRef),
        "tokens input is not MemRef"
    );
    ensure!(
        inputs
            .by_ref()
            .take(states)
            .all(|kind| kind == ValueKind::MemOwn),
        "state inputs are not all MemOwn"
    );
    ensure!(
        inputs.next() == Some(ValueKind::U64),
        "capacity input is not u64"
    );
    ensure!(
        inputs.next() == Some(ValueKind::U64),
        "position input is not u64"
    );
    ensure!(
        inputs.all(|kind| kind == ValueKind::MemRef),
        "asset inputs are not all MemRef"
    );

    let output_count = states
        .checked_add(2)
        .context("model output count overflowed")?;
    ensure!(
        entry_point.outputs().len() == output_count,
        "entry point {:?} has {} outputs, causal-lm requires {output_count}",
        entry_point.name(),
        entry_point.outputs().len()
    );
    ensure!(
        entry_point
            .outputs()
            .iter()
            .all(|kind| *kind == ValueKind::MemOwn),
        "causal-lm outputs must be replacement states, logits, and next-token MemOwn buffers"
    );
    Ok(())
}

fn forward(
    runtime: &Runtime,
    assets: &AssetStore,
    model: &BoundModel,
    generation: &mut Generation,
    tokens: Vec<u32>,
) -> Result<u32> {
    ensure!(!tokens.is_empty(), "cannot invoke a model with no tokens");
    ensure!(
        tokens.len() <= MAXIMUM_CAPACITY as usize,
        "token batch exceeds the wire count limit"
    );
    ensure!(
        tokens
            .iter()
            .all(|&token| u64::from(token) < model.vocabulary_size),
        "input contains a token outside the declared vocabulary"
    );
    let token_count = u64::try_from(tokens.len()).context("token count exceeds u64")?;
    let generated_allocation_budget = remaining_generated_allocation_budget(
        generation.state_byte_len,
        token_count,
        model.generation_device_allocation_budget_bytes,
    )?;
    let end = generation
        .position
        .checked_add(token_count)
        .context("generation position overflowed")?;
    ensure!(
        end <= generation.capacity,
        "token range {}..{end} exceeds generation capacity {}",
        generation.position,
        generation.capacity
    );

    let token_staging = tokens.into_iter().map(u64::from).collect::<Vec<_>>();
    let token_memory = runtime.mem_u64(&token_staging)?;
    let states = std::mem::take(&mut generation.states);
    ensure!(
        states.len() == model.state_byte_multipliers.len(),
        "generation state is unavailable"
    );
    let input_count = 3 + states.len() + model.assets.len();
    let mut inputs = Vec::with_capacity(input_count);
    inputs.push(Value::MemRef(token_memory.as_ref()));
    inputs.extend(states.into_iter().map(Value::MemOwn));
    inputs.push(Value::U64(generation.capacity));
    inputs.push(Value::U64(generation.position));
    for slice in &model.assets {
        inputs.push(Value::MemRef(assets.mem_ref(
            slice.asset,
            slice.offset,
            slice.byte_len,
        )?));
    }

    let outputs = match model.artifact.exec_values_with_generated_allocation_budget(
        &model.entry_point,
        inputs,
        generated_allocation_budget,
    ) {
        Ok(outputs) => outputs,
        Err(ExecError::GpuSynchronization(error)) => {
            return Err(FatalGpuSynchronization(error).into());
        }
        Err(error) => {
            return Err(error)
                .with_context(|| format!("Catena entry point {:?} failed", model.entry_point));
        }
    };

    let mut outputs = outputs.into_iter();
    let mut replacement_states = Vec::with_capacity(model.state_byte_multipliers.len());
    for index in 0..model.state_byte_multipliers.len() {
        let Some(Value::MemOwn(state)) = outputs.next() else {
            bail!("replacement state {index} is not MemOwn");
        };
        let expected = generation
            .capacity
            .checked_mul(model.state_byte_multipliers[index])
            .context("replacement state byte count overflowed")?;
        ensure!(
            state.byte_len() == expected,
            "replacement state {index} is {} bytes, expected {expected}",
            state.byte_len()
        );
        replacement_states.push(state);
    }

    let Some(Value::MemOwn(logits)) = outputs.next() else {
        bail!("logits output is not MemOwn");
    };
    let logits_bytes = model
        .vocabulary_size
        .checked_mul(LOGIT_ELEMENT_BYTES)
        .context("logits byte count overflowed")?;
    ensure!(
        logits.byte_len() == logits_bytes,
        "logits output is {} bytes, expected {logits_bytes}",
        logits.byte_len()
    );

    let Some(Value::MemOwn(next_token)) = outputs.next() else {
        bail!("next-token output is not MemOwn");
    };
    ensure!(
        next_token.byte_len() == NEXT_TOKEN_BYTES,
        "next-token output is {} bytes, expected {NEXT_TOKEN_BYTES}",
        next_token.byte_len()
    );
    ensure!(
        outputs.next().is_none(),
        "entry point returned extra outputs"
    );
    let values = next_token
        .try_to_u64_vec()
        .context("failed to read next-token output")?;
    let [next_token] = values.as_slice() else {
        bail!("next-token output does not contain exactly one u64");
    };
    ensure!(
        *next_token < model.vocabulary_size,
        "program returned token {next_token} outside the declared vocabulary"
    );
    let next_token =
        u32::try_from(*next_token).context("program returned a token outside the u32 ID space")?;

    generation.states = replacement_states;
    generation.position = end;
    Ok(next_token)
}

fn take_id(next: &mut u64, kind: &str) -> Result<u64> {
    let id = *next;
    *next = next
        .checked_add(1)
        .with_context(|| format!("resident {kind} ID space exhausted"))?;
    Ok(id)
}

#[derive(Debug, thiserror::Error)]
#[error("GPU execution failed while synchronizing: {0}")]
struct FatalGpuSynchronization(String);

pub(super) fn gpu_synchronization(error: &anyhow::Error) -> Option<&str> {
    error
        .downcast_ref::<FatalGpuSynchronization>()
        .map(|error| error.0.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binding() -> WireModelBinding {
        WireModelBinding {
            artifact: 0,
            entry_point: "model".to_string(),
            assets: vec![WireAssetSlice {
                asset: 1,
                offset: 0,
                byte_len: 16,
            }],
            state_byte_multipliers: vec![4],
            vocabulary_size: 16,
            maximum_capacity: 8,
            generation_device_allocation_budget_bytes: 1024,
        }
    }

    fn entry(next_token: ValueKind) -> EntryPoint {
        EntryPoint::new(
            "model".to_string(),
            vec![
                ValueKind::MemRef,
                ValueKind::MemOwn,
                ValueKind::U64,
                ValueKind::U64,
                ValueKind::MemRef,
            ],
            vec![ValueKind::MemOwn, ValueKind::MemOwn, next_token],
        )
    }

    #[test]
    fn exact_owned_buffer_causal_lm_abi_is_accepted() {
        validate_model_binding(&entry(ValueKind::MemOwn), &binding()).unwrap();
    }

    #[test]
    fn scalar_next_token_shortcut_is_rejected() {
        let error = validate_model_binding(&entry(ValueKind::U64), &binding()).unwrap_err();
        assert!(error.to_string().contains("outputs must be"));
    }

    #[test]
    fn state_and_static_allocation_limits_are_checked() {
        let mut invalid_state = binding();
        invalid_state.state_byte_multipliers = vec![3];
        assert!(validate_model_binding(&entry(ValueKind::MemOwn), &invalid_state).is_err());

        let mut excessive_state = binding();
        excessive_state.maximum_capacity = MAXIMUM_CAPACITY;
        excessive_state.state_byte_multipliers =
            vec![MAX_MODEL_STATE_BYTES.checked_div(MAXIMUM_CAPACITY).unwrap() + 4];
        assert!(validate_model_binding(&entry(ValueKind::MemOwn), &excessive_state).is_err());

        let mut overflowing_state = binding();
        overflowing_state.maximum_capacity = MAXIMUM_CAPACITY;
        overflowing_state.state_byte_multipliers = vec![u64::MAX - 3];
        assert!(validate_model_binding(&entry(ValueKind::MemOwn), &overflowing_state).is_err());

        let mut excessive_static = binding();
        excessive_static.assets[0].byte_len = MAX_MODEL_STATIC_BYTES;
        excessive_static.assets.push(WireAssetSlice {
            asset: 1,
            offset: 0,
            byte_len: 1,
        });
        let entry = EntryPoint::new(
            "model".to_string(),
            vec![
                ValueKind::MemRef,
                ValueKind::MemOwn,
                ValueKind::U64,
                ValueKind::U64,
                ValueKind::MemRef,
                ValueKind::MemRef,
            ],
            vec![ValueKind::MemOwn; 3],
        );
        assert!(validate_model_binding(&entry, &excessive_static).is_err());
    }

    #[test]
    fn generation_envelope_is_checked_before_state_allocation() {
        let minimum = minimum_generation_device_bytes(&[4, 8], 10, 16).unwrap();
        assert_eq!(minimum, 272);
        assert_eq!(
            checked_generation_layout(&[4, 8], 10, 16, minimum).unwrap(),
            (vec![40, 80], 120)
        );
        let error = checked_generation_layout(&[4, 8], 10, 16, minimum - 1).unwrap_err();
        assert!(error.to_string().contains("allocation envelope"));
        assert!(generation_state_layout(&[u64::MAX], 2).is_err());
    }

    #[test]
    fn minimum_generation_device_bytes_reports_each_overflow_class() {
        assert_eq!(
            minimum_generation_device_bytes(&[u64::MAX], 2, 0),
            Err(GenerationDeviceEnvelopeError::StateBytesOverflow)
        );
        assert_eq!(
            minimum_generation_device_bytes(&[], u64::MAX, 0),
            Err(GenerationDeviceEnvelopeError::TokenStagingBytesOverflow)
        );
        assert_eq!(
            minimum_generation_device_bytes(&[], 0, u64::MAX),
            Err(GenerationDeviceEnvelopeError::LogitsBytesOverflow)
        );
        assert_eq!(
            minimum_generation_device_bytes(&[], u64::MAX / 8, 1),
            Err(GenerationDeviceEnvelopeError::TotalBytesOverflow)
        );
        assert_eq!(
            minimum_generation_device_bytes(&[MAX_MODEL_STATE_BYTES], 2, 0),
            Err(GenerationDeviceEnvelopeError::StateBytesLimit {
                maximum: MAX_MODEL_STATE_BYTES
            })
        );
    }

    #[test]
    fn generated_budget_reserves_resident_state_and_token_staging() {
        assert_eq!(
            remaining_generated_allocation_budget(80, 5, 200).unwrap(),
            80
        );
        let error = remaining_generated_allocation_budget(80, 5, 119).unwrap_err();
        assert!(error.to_string().contains("allocation envelope"));
        assert!(remaining_generated_allocation_budget(0, u64::MAX, u64::MAX).is_err());
    }

    #[test]
    fn teardown_removes_generation_state_even_after_model_release() {
        let mut store = ResidentStore::new();
        store.generations.insert(
            11,
            Generation {
                model: 7,
                states: Vec::new(),
                state_byte_len: 0,
                capacity: 8,
                position: 0,
            },
        );

        store.release_generation(11);
        assert!(!store.generations.contains_key(&11));
        store.release_generation(11);

        store.generations.insert(
            13,
            Generation {
                model: 7,
                states: Vec::new(),
                state_byte_len: 0,
                capacity: 8,
                position: 0,
            },
        );
        store.release_model(7);
        assert!(!store.models.contains_key(&7));
        assert!(!store.generations.contains_key(&13));
        store.release_model(7);
    }
}
