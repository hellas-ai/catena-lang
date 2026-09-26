use super::{gpu_support::StorageType, projection::ProjectionPlan};

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
            Self::Projection(plan) => plan.validates_materializec_producer(),
            _ => true,
        }
    }

    pub(in crate::codegen) fn uses_blas(self) -> bool {
        matches!(self, Self::Projection(plan) if plan.uses_blas())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Family {
        CachedAttention(bool),
        CachedContext(StorageType),
        Projection,
        GatedDeltaNet(StorageType),
    }

    fn family(operation: OptimizedOperation) -> Family {
        match operation {
            OptimizedOperation::CachedAttention { matrix_core } => {
                Family::CachedAttention(matrix_core)
            }
            OptimizedOperation::CachedContext(storage) => Family::CachedContext(storage),
            OptimizedOperation::Projection(_) => Family::Projection,
            OptimizedOperation::GatedDeltaNet(storage) => Family::GatedDeltaNet(storage),
        }
    }

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

    #[test]
    fn every_optimized_operation_has_one_stable_policy() {
        let cases = [
            (
                "materializec.borrow-cached-attention-bf16",
                Family::CachedAttention(false),
                true,
                false,
            ),
            (
                "materializec.borrow-cached-attention-bf16-wmma",
                Family::CachedAttention(true),
                true,
                false,
            ),
            (
                "materializec.borrow-cached-context-f32",
                Family::CachedContext(StorageType::F32),
                true,
                false,
            ),
            (
                "materializec.borrow-cached-context-bf16",
                Family::CachedContext(StorageType::Bf16),
                true,
                false,
            ),
            (
                "materializec.borrow-gated-delta-net-f32",
                Family::GatedDeltaNet(StorageType::F32),
                true,
                false,
            ),
            (
                "materializec.borrow-gated-delta-net-bf16",
                Family::GatedDeltaNet(StorageType::Bf16),
                true,
                false,
            ),
            ("materializec.bf16-gemv", Family::Projection, true, false),
            (
                "materializec.bf16-gemm-wmma",
                Family::Projection,
                true,
                false,
            ),
            (
                "materializec.bf16-gemm-blas",
                Family::Projection,
                true,
                true,
            ),
            (
                "materializec.borrow-native-bf16-gemv-pair",
                Family::Projection,
                true,
                false,
            ),
            (
                "materializec.borrow-native-bf16-gemm-wmma-pair",
                Family::Projection,
                true,
                false,
            ),
            (
                "materializec.borrow-native-bf16-gemm-blas-pair",
                Family::Projection,
                true,
                true,
            ),
            (
                "materializec.borrow-routed-bf16-gemv-pair",
                Family::Projection,
                false,
                false,
            ),
            (
                "materializec.borrow-routed-native-bf16-gemv-pair",
                Family::Projection,
                true,
                false,
            ),
            (
                "materializec.borrow-routed-native-bf16-gemm-wmma-pair",
                Family::Projection,
                true,
                false,
            ),
            (
                "materializec.borrow-routed-native-bf16-gemv-residual",
                Family::Projection,
                true,
                false,
            ),
            (
                "materializec.borrow-routed-native-bf16-gemm-wmma-residual",
                Family::Projection,
                true,
                false,
            ),
        ];
        let mut names = BTreeSet::new();
        for (name, expected_family, validates_producer, uses_blas) in cases {
            assert!(names.insert(name), "duplicate test case for {name}");
            let operation = OptimizedOperation::from_operation(name)
                .unwrap_or_else(|| panic!("{name} must be classified"));
            assert_eq!(family(operation), expected_family, "{name}");
            assert_eq!(
                operation.validates_materializec_producer(),
                validates_producer,
                "{name}"
            );
            assert_eq!(operation.uses_blas(), uses_blas, "{name}");
        }
        assert_eq!(names.len(), 17);
    }
}
