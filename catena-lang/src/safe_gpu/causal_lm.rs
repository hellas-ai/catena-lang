//! Frozen, token-native causal-language-model execution.
//!
//! A [`Model`] binds one prepared program, its exact entry point, ordered
//! read-only asset slices, and zeroed state layout once. Generation then sends
//! only token IDs to the isolated worker and receives one checked token ID per
//! step. Tokenization, text decoding, and chat templating remain outside this
//! trust boundary.

use std::{
    fmt,
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::Result as AnyResult;
use thiserror::Error;

use super::{AssetSlice, Program, Session};
use crate::{
    runtime::RuntimeId,
    safe_runtime::{
        ResidentAssetSlice, ResidentError, ResidentGeneration, ResidentModel, SafeRuntime,
        resident::{MAX_STOP_TOKEN_IDS, MAXIMUM_CAPACITY, validate_model_spec},
    },
};

/// Complete, named configuration for one frozen causal-LM ABI binding.
#[derive(Debug, Clone, Copy)]
pub struct ModelConfig<'a> {
    pub entry_point: &'a str,
    pub asset_slices: &'a [AssetSlice],
    /// Zeroed state bytes allocated per token of generation capacity.
    /// Every multiplier must be non-zero and divisible by four.
    pub state_byte_multipliers: &'a [u64],
    pub vocabulary_size: u64,
    pub maximum_capacity: u64,
}

/// A program or model description could not be bound to the causal-LM ABI.
#[derive(Debug, Error)]
pub enum BindError {
    #[error("program belongs to a different GPU session")]
    WrongProgramSession,
    #[error("asset slice {index} belongs to a different GPU session")]
    WrongAssetSession { index: usize },
    #[error("program has no entry point {0:?}")]
    UnknownEntryPoint(String),
    #[error("invalid causal-LM binding: {0}")]
    InvalidConfiguration(String),
    #[error(transparent)]
    Resident(#[from] ResidentError),
}

impl BindError {
    /// Whether the worker protocol can no longer be trusted after this error.
    pub fn invalidates_session(&self) -> bool {
        matches!(self, Self::Resident(error) if error.invalidates_session())
    }
}

/// The caller's decision after receiving one generated token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerationControl {
    Continue,
    Cancel,
}

/// Why token generation ended successfully.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerationTermination {
    /// This terminal token is not emitted or included in the result.
    StopToken(u32),
    MaxNewTokens,
    /// The callback accepted a token and then cancelled. The token is retained.
    Cancelled,
}

#[derive(Debug)]
pub struct TokenGenerationResult {
    pub generated_tokens: Vec<u32>,
    pub termination: GenerationTermination,
    pub stats: GenerationStats,
}

#[derive(Debug, Clone, Copy)]
pub struct GenerationStats {
    pub elapsed: Duration,
    pub prompt_eval_time: Duration,
    pub cached_eval_time: Duration,
    pub prompt_tokens: usize,
    pub cached_tokens: usize,
    pub generated_tokens: usize,
}

impl GenerationStats {
    pub fn prompt_tokens_per_second(self) -> f64 {
        rate(self.prompt_tokens, self.prompt_eval_time)
    }

    pub fn cached_tokens_per_second(self) -> f64 {
        rate(self.cached_tokens, self.cached_eval_time)
    }

    pub fn generated_tokens_per_second(self) -> f64 {
        rate(self.generated_tokens, self.elapsed)
    }
}

impl fmt::Display for GenerationStats {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            formatter,
            "prompt eval: {} token(s) in {:.2?} ({:.2} tok/s)",
            self.prompt_tokens,
            self.prompt_eval_time,
            self.prompt_tokens_per_second()
        )?;
        writeln!(
            formatter,
            "cached eval: {} token(s) in {:.2?} ({:.2} tok/s)",
            self.cached_tokens,
            self.cached_eval_time,
            self.cached_tokens_per_second()
        )?;
        write!(
            formatter,
            "generated {} token(s) in {:.2?} ({:.2} tok/s)",
            self.generated_tokens,
            self.elapsed,
            self.generated_tokens_per_second()
        )
    }
}

/// A generation request failed before or during resident execution.
#[derive(Debug, Error)]
pub enum GenerationError {
    #[error("invalid token-generation request: {0}")]
    InvalidRequest(String),
    #[error(transparent)]
    Resident(#[from] ResidentError),
    #[error("token callback failed: {0}")]
    Callback(#[source] anyhow::Error),
}

impl GenerationError {
    /// Whether the worker protocol can no longer be trusted after this error.
    pub fn invalidates_session(&self) -> bool {
        matches!(self, Self::Resident(error) if error.invalidates_session())
    }
}

/// Opaque child-resident causal-LM binding.
pub struct Model {
    runtime: Arc<SafeRuntime>,
    session: RuntimeId,
    resident: ResidentModel,
    vocabulary_size: u64,
    maximum_capacity: u64,
}

impl fmt::Debug for Model {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Model")
            .field("vocabulary_size", &self.vocabulary_size)
            .field("maximum_capacity", &self.maximum_capacity)
            .finish_non_exhaustive()
    }
}

impl Session {
    /// Validate and bind one exact causal-LM entry point in this worker.
    pub fn bind_causal_lm(
        &self,
        program: &Program,
        config: ModelConfig<'_>,
    ) -> Result<Model, BindError> {
        let session = self.runtime.id();
        if !program.belongs_to(session) {
            return Err(BindError::WrongProgramSession);
        }
        for (index, slice) in config.asset_slices.iter().enumerate() {
            if slice.asset.session != session {
                return Err(BindError::WrongAssetSession { index });
            }
        }
        let entry_point = program
            .entry_points()
            .iter()
            .find(|entry| entry.name() == config.entry_point)
            .ok_or_else(|| BindError::UnknownEntryPoint(config.entry_point.to_string()))?;
        validate_model_spec(
            entry_point,
            config.entry_point,
            config.state_byte_multipliers,
            config.asset_slices.len(),
            config.asset_slices.iter().map(|slice| slice.byte_len),
            config.vocabulary_size,
            config.maximum_capacity,
        )
        .map_err(|error| BindError::InvalidConfiguration(error.to_string()))?;

        let assets = config
            .asset_slices
            .iter()
            .map(|slice| ResidentAssetSlice {
                asset: slice.asset.resident,
                offset: slice.offset,
                byte_len: slice.byte_len,
            })
            .collect();
        let resident = self.runtime.bind_resident_model(
            program,
            config.entry_point.to_string(),
            assets,
            config.state_byte_multipliers.to_vec(),
            config.vocabulary_size,
            config.maximum_capacity,
        )?;
        Ok(Model {
            runtime: self.runtime.clone(),
            session,
            resident,
            vocabulary_size: config.vocabulary_size,
            maximum_capacity: config.maximum_capacity,
        })
    }
}

impl Model {
    pub fn vocabulary_size(&self) -> u64 {
        self.vocabulary_size
    }

    pub fn maximum_capacity(&self) -> u64 {
        self.maximum_capacity
    }

    /// Generate from pre-tokenized `u32` IDs and collect accepted tokens.
    pub fn generate_tokens(
        &self,
        prompt_tokens: &[u32],
        max_new_tokens: u32,
        stop_tokens: &[u32],
    ) -> Result<TokenGenerationResult, GenerationError> {
        self.generate_tokens_streaming(prompt_tokens, max_new_tokens, stop_tokens, |_| {
            Ok(GenerationControl::Continue)
        })
    }

    /// Generate token IDs and call `emit` after each accepted non-stop token.
    ///
    /// A stop token is checked before `emit` and is not collected. Cancellation
    /// retains the token already passed to `emit`. Callback errors remain
    /// errors, and every exit path releases the child-resident generation.
    pub fn generate_tokens_streaming(
        &self,
        prompt_tokens: &[u32],
        max_new_tokens: u32,
        stop_tokens: &[u32],
        emit: impl FnMut(u32) -> AnyResult<GenerationControl>,
    ) -> Result<TokenGenerationResult, GenerationError> {
        let Some(capacity) = validate_generation_request(
            prompt_tokens,
            stop_tokens,
            max_new_tokens,
            self.vocabulary_size,
            self.maximum_capacity,
        )?
        else {
            return Ok(empty_generation_result());
        };
        let handle = self
            .runtime
            .start_resident_generation(self.resident, capacity)?;
        let generation = ActiveGeneration {
            runtime: &self.runtime,
            handle,
        };
        generate_tokens_with(
            prompt_tokens,
            max_new_tokens as usize,
            |token| stop_tokens.contains(&token),
            |tokens| Ok(generation.forward(tokens)?),
            emit,
        )
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        debug_assert_eq!(self.session, self.runtime.id());
        self.runtime.release_resident_model(self.resident);
    }
}

struct ActiveGeneration<'a> {
    runtime: &'a SafeRuntime,
    handle: ResidentGeneration,
}

impl ActiveGeneration<'_> {
    fn forward(&self, tokens: &[u32]) -> Result<u32, ResidentError> {
        self.runtime
            .step_resident_generation(self.handle, tokens.to_vec())
    }
}

impl Drop for ActiveGeneration<'_> {
    fn drop(&mut self) {
        self.runtime.release_resident_generation(self.handle);
    }
}

fn validate_generation_request(
    prompt_tokens: &[u32],
    stop_tokens: &[u32],
    max_new_tokens: u32,
    vocabulary_size: u64,
    maximum_capacity: u64,
) -> Result<Option<u64>, GenerationError> {
    invalid(!prompt_tokens.is_empty(), "prompt contains no tokens")?;
    invalid(
        prompt_tokens.len() <= MAXIMUM_CAPACITY as usize,
        "prompt exceeds the token wire limit",
    )?;
    invalid(
        stop_tokens.len() <= MAX_STOP_TOKEN_IDS,
        "request contains too many stop tokens",
    )?;
    invalid(
        prompt_tokens
            .iter()
            .all(|&token| u64::from(token) < vocabulary_size),
        "prompt contains a token outside the declared vocabulary",
    )?;
    invalid(
        stop_tokens
            .iter()
            .all(|&token| u64::from(token) < vocabulary_size),
        "request contains a stop token outside the declared vocabulary",
    )?;
    if max_new_tokens == 0 {
        return Ok(None);
    }
    let prompt_len = u64::try_from(prompt_tokens.len())
        .map_err(|_| GenerationError::InvalidRequest("prompt token count exceeds u64".into()))?;
    let capacity = prompt_len
        .checked_add(u64::from(max_new_tokens))
        .ok_or_else(|| GenerationError::InvalidRequest("generation capacity overflowed".into()))?;
    invalid(
        capacity <= maximum_capacity,
        format!("requested capacity {capacity} exceeds model maximum {maximum_capacity}"),
    )?;
    Ok(Some(capacity))
}

fn generate_tokens_with(
    prompt_tokens: &[u32],
    max_new_tokens: usize,
    mut is_stop: impl FnMut(u32) -> bool,
    mut forward: impl FnMut(&[u32]) -> Result<u32, GenerationError>,
    mut emit: impl FnMut(u32) -> AnyResult<GenerationControl>,
) -> Result<TokenGenerationResult, GenerationError> {
    debug_assert!(!prompt_tokens.is_empty());
    debug_assert!(max_new_tokens > 0);

    let started = Instant::now();
    let prompt_started = Instant::now();
    let mut next_token = forward(prompt_tokens)?;
    let prompt_eval_time = prompt_started.elapsed();
    let mut cached_eval_time = Duration::ZERO;
    let mut cached_tokens = 0;
    let mut generated_tokens = Vec::new();

    for index in 0..max_new_tokens {
        if index != 0 {
            let previous = *generated_tokens
                .last()
                .expect("a later decode step must have a previous token");
            let cached_started = Instant::now();
            next_token = forward(std::slice::from_ref(&previous))?;
            cached_eval_time += cached_started.elapsed();
            cached_tokens += 1;
        }
        if is_stop(next_token) {
            return Ok(generation_result(
                started,
                prompt_eval_time,
                cached_eval_time,
                prompt_tokens.len(),
                cached_tokens,
                generated_tokens,
                GenerationTermination::StopToken(next_token),
            ));
        }
        generated_tokens.push(next_token);
        if emit(next_token).map_err(GenerationError::Callback)? == GenerationControl::Cancel {
            return Ok(generation_result(
                started,
                prompt_eval_time,
                cached_eval_time,
                prompt_tokens.len(),
                cached_tokens,
                generated_tokens,
                GenerationTermination::Cancelled,
            ));
        }
    }

    Ok(generation_result(
        started,
        prompt_eval_time,
        cached_eval_time,
        prompt_tokens.len(),
        cached_tokens,
        generated_tokens,
        GenerationTermination::MaxNewTokens,
    ))
}

fn generation_result(
    started: Instant,
    prompt_eval_time: Duration,
    cached_eval_time: Duration,
    prompt_tokens: usize,
    cached_tokens: usize,
    generated_tokens: Vec<u32>,
    termination: GenerationTermination,
) -> TokenGenerationResult {
    TokenGenerationResult {
        stats: GenerationStats {
            elapsed: started.elapsed(),
            prompt_eval_time,
            cached_eval_time,
            prompt_tokens,
            cached_tokens,
            generated_tokens: generated_tokens.len(),
        },
        generated_tokens,
        termination,
    }
}

fn empty_generation_result() -> TokenGenerationResult {
    TokenGenerationResult {
        generated_tokens: Vec::new(),
        termination: GenerationTermination::MaxNewTokens,
        stats: GenerationStats {
            elapsed: Duration::ZERO,
            prompt_eval_time: Duration::ZERO,
            cached_eval_time: Duration::ZERO,
            prompt_tokens: 0,
            cached_tokens: 0,
            generated_tokens: 0,
        },
    }
}

fn invalid(condition: bool, message: impl Into<String>) -> Result<(), GenerationError> {
    if condition {
        Ok(())
    } else {
        Err(GenerationError::InvalidRequest(message.into()))
    }
}

fn rate(tokens: usize, elapsed: Duration) -> f64 {
    if elapsed.is_zero() {
        f64::INFINITY
    } else {
        tokens as f64 / elapsed.as_secs_f64()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_preserves_wide_tokens_and_statistics() {
        let wide = u32::MAX - 1;
        let mut outputs = [wide, 12, 13].into_iter();
        let mut emitted = Vec::new();
        let result = generate_tokens_with(
            &[1, 2],
            3,
            |_| false,
            |_| Ok(outputs.next().unwrap()),
            |token| {
                emitted.push(token);
                Ok(GenerationControl::Continue)
            },
        )
        .unwrap();
        assert_eq!(result.generated_tokens, [wide, 12, 13]);
        assert_eq!(emitted, result.generated_tokens);
        assert_eq!(result.termination, GenerationTermination::MaxNewTokens);
        assert_eq!(result.stats.prompt_tokens, 2);
        assert_eq!(result.stats.cached_tokens, 2);
        assert_eq!(result.stats.generated_tokens, 3);
    }

    #[test]
    fn stop_token_is_not_emitted_or_collected() {
        let mut outputs = [7, 99].into_iter();
        let mut emitted = Vec::new();
        let result = generate_tokens_with(
            &[1],
            4,
            |token| token == 99,
            |_| Ok(outputs.next().unwrap()),
            |token| {
                emitted.push(token);
                Ok(GenerationControl::Continue)
            },
        )
        .unwrap();
        assert_eq!(result.generated_tokens, [7]);
        assert_eq!(emitted, [7]);
        assert_eq!(result.termination, GenerationTermination::StopToken(99));
    }

    #[test]
    fn cancellation_retains_the_emitted_token() {
        let mut outputs = [7, 8, 9].into_iter();
        let result = generate_tokens_with(
            &[1],
            3,
            |_| false,
            |_| Ok(outputs.next().unwrap()),
            |token| {
                Ok(if token == 8 {
                    GenerationControl::Cancel
                } else {
                    GenerationControl::Continue
                })
            },
        )
        .unwrap();
        assert_eq!(result.generated_tokens, [7, 8]);
        assert_eq!(result.termination, GenerationTermination::Cancelled);
    }

    #[test]
    fn request_validation_bounds_tokens_and_capacity() {
        assert!(validate_generation_request(&[], &[], 1, 10, 10).is_err());
        assert!(validate_generation_request(&[10], &[], 1, 10, 10).is_err());
        assert!(validate_generation_request(&[1], &[10], 1, 10, 10).is_err());
        assert!(validate_generation_request(&[1, 2], &[], 9, 10, 10).is_err());
        assert!(
            validate_generation_request(&[1], &[2; MAX_STOP_TOKEN_IDS + 1], 1, 10, 10).is_err()
        );
        assert_eq!(
            validate_generation_request(&[1], &[], 0, 10, 10).unwrap(),
            None
        );
    }

    #[test]
    fn typed_errors_preserve_only_healthy_sessions() {
        assert!(!BindError::UnknownEntryPoint("missing".into()).invalidates_session());
        assert!(
            BindError::Resident(ResidentError::Transport("closed".into())).invalidates_session()
        );
        assert!(!GenerationError::InvalidRequest("invalid".into()).invalidates_session());
        assert!(GenerationError::Resident(ResidentError::UnexpectedResponse).invalidates_session());
    }
}
