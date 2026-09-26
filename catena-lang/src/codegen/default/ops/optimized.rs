use super::{
    gpu_support::StorageType,
    projection::{ProjectionBackend, ProjectionPlan},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::codegen) enum OptimizedOperation {
    CachedAttention { matrix_core: bool },
    CachedContext(StorageType),
    Projection(ProjectionPlan),
    GatedDeltaNet(StorageType),
}

impl OptimizedOperation {
    pub(in crate::codegen) fn from_operation(operation: &str) -> Option<Self> {
        if let Some(plan) = ProjectionPlan::from_operation(operation) {
            return Some(Self::Projection(plan));
        }
        match operation {
            "materializec.borrow-cached-attention-bf16" => {
                Some(Self::CachedAttention { matrix_core: false })
            }
            "materializec.borrow-cached-attention-bf16-wmma" => {
                Some(Self::CachedAttention { matrix_core: true })
            }
            "materializec.borrow-cached-context-f32" => Some(Self::CachedContext(StorageType::F32)),
            "materializec.borrow-cached-context-bf16" => {
                Some(Self::CachedContext(StorageType::Bf16))
            }
            "materializec.borrow-gated-delta-net-f32" => {
                Some(Self::GatedDeltaNet(StorageType::F32))
            }
            "materializec.borrow-gated-delta-net-bf16" => {
                Some(Self::GatedDeltaNet(StorageType::Bf16))
            }
            _ => None,
        }
    }

    pub(in crate::codegen) fn validates_materializec_producer(self) -> bool {
        match self {
            Self::Projection(plan) => plan.native_bf16,
            _ => true,
        }
    }

    pub(in crate::codegen) fn uses_blas(self) -> bool {
        matches!(
            self,
            Self::Projection(ProjectionPlan {
                backend: ProjectionBackend::Blas,
                ..
            })
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_non_projection_kernel_families() {
        assert_eq!(
            OptimizedOperation::from_operation("materializec.borrow-cached-attention-bf16-wmma"),
            Some(OptimizedOperation::CachedAttention { matrix_core: true })
        );
        assert_eq!(
            OptimizedOperation::from_operation("materializec.borrow-cached-context-bf16"),
            Some(OptimizedOperation::CachedContext(StorageType::Bf16))
        );
        assert_eq!(
            OptimizedOperation::from_operation("materializec.borrow-gated-delta-net-f32"),
            Some(OptimizedOperation::GatedDeltaNet(StorageType::F32))
        );
    }

    #[test]
    fn exposes_cross_cutting_codegen_properties() {
        let blas =
            OptimizedOperation::from_operation("materializec.borrow-native-bf16-gemm-blas-pair")
                .unwrap();
        assert!(blas.uses_blas());
        assert!(blas.validates_materializec_producer());

        let legacy =
            OptimizedOperation::from_operation("materializec.borrow-routed-bf16-gemv-pair")
                .unwrap();
        assert!(!legacy.uses_blas());
        assert!(!legacy.validates_materializec_producer());
        assert!(OptimizedOperation::from_operation("materializec.softmax-f32").is_none());
    }
}
