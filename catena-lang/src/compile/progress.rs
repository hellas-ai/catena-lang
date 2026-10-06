//! Pipeline timing and optional live progress, independent of the CLI.
use std::time::Instant;

use serde::Serialize;

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StageStatus {
    Completed,
    Failed,
}

#[derive(Clone, Debug, Serialize)]
pub struct StageTiming {
    pub stage: String,
    pub elapsed_ms: f64,
    pub status: StageStatus,
}

#[derive(Debug)]
pub enum ProgressEvent {
    Started {
        stage: &'static str,
    },
    Update {
        stage: &'static str,
        message: String,
        elapsed_ms: f64,
    },
    Finished(StageTiming),
}

/// Record a stage even when it returns an error. Detail updates share its clock.
pub(crate) fn timed<T, E>(
    timings: &mut Vec<StageTiming>,
    stage: &'static str,
    progress: &mut dyn FnMut(ProgressEvent),
    work: impl FnOnce(&mut dyn FnMut(&str)) -> Result<T, E>,
) -> Result<T, E> {
    let start = Instant::now();
    progress(ProgressEvent::Started { stage });
    let result = work(&mut |message| {
        progress(ProgressEvent::Update {
            stage,
            message: message.into(),
            elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
        });
    });
    let timing = StageTiming {
        stage: stage.into(),
        elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
        status: if result.is_ok() {
            StageStatus::Completed
        } else {
            StageStatus::Failed
        },
    };
    timings.push(timing.clone());
    progress(ProgressEvent::Finished(timing));
    result
}
