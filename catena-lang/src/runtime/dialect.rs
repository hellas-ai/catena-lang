use serde::{Deserialize, Serialize};

/// GPU platform targeted by a runtime module and runtime context.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuDialect {
    Hip,
    Cuda,
}
