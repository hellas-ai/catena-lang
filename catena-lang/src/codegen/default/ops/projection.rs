#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::codegen) enum ProjectionBackend {
    Simt,
    Wmma,
    Blas,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::codegen) enum ProjectionTopology {
    Dense,
    Routed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::codegen) enum ProjectionOutputs {
    Single,
    Pair,
    Residual,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::codegen) struct ProjectionPlan {
    pub backend: ProjectionBackend,
    pub topology: ProjectionTopology,
    pub outputs: ProjectionOutputs,
    pub native_bf16: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::codegen) struct MmaTileSpec {
    pub rows: u32,
    pub columns: u32,
    pub reduction: u32,
    pub waves: u32,
    pub threads: u32,
    pub minimum_rows: u32,
}

impl MmaTileSpec {
    pub const DENSE: Self = Self {
        rows: 128,
        columns: 128,
        reduction: 16,
        waves: 8,
        threads: 256,
        minimum_rows: 64,
    };

    pub const fn column_fragments(self) -> u32 {
        self.columns / 16
    }
}

impl ProjectionPlan {
    pub(in crate::codegen) fn from_operation(operation: &str) -> Option<Self> {
        use ProjectionBackend::{Blas, Simt, Wmma};
        use ProjectionOutputs::{Pair, Residual, Single};
        use ProjectionTopology::{Dense, Routed};

        let plan = match operation {
            "materializec.bf16-gemv" => Self {
                backend: Simt,
                topology: Dense,
                outputs: Single,
                native_bf16: true,
            },
            "materializec.bf16-gemm-wmma" => Self {
                backend: Wmma,
                topology: Dense,
                outputs: Single,
                native_bf16: true,
            },
            "materializec.bf16-gemm-blas" => Self {
                backend: Blas,
                topology: Dense,
                outputs: Single,
                native_bf16: true,
            },
            "materializec.borrow-native-bf16-gemv-pair" => Self {
                backend: Simt,
                topology: Dense,
                outputs: Pair,
                native_bf16: true,
            },
            "materializec.borrow-native-bf16-gemm-wmma-pair" => Self {
                backend: Wmma,
                topology: Dense,
                outputs: Pair,
                native_bf16: true,
            },
            "materializec.borrow-native-bf16-gemm-blas-pair" => Self {
                backend: Blas,
                topology: Dense,
                outputs: Pair,
                native_bf16: true,
            },
            "materializec.borrow-routed-bf16-gemv-pair" => Self {
                backend: Simt,
                topology: Routed,
                outputs: Pair,
                native_bf16: false,
            },
            "materializec.borrow-routed-native-bf16-gemv-pair" => Self {
                backend: Simt,
                topology: Routed,
                outputs: Pair,
                native_bf16: true,
            },
            "materializec.borrow-routed-native-bf16-gemm-wmma-pair" => Self {
                backend: Wmma,
                topology: Routed,
                outputs: Pair,
                native_bf16: true,
            },
            "materializec.borrow-routed-native-bf16-gemv-residual" => Self {
                backend: Simt,
                topology: Routed,
                outputs: Residual,
                native_bf16: true,
            },
            "materializec.borrow-routed-native-bf16-gemm-wmma-residual" => Self {
                backend: Wmma,
                topology: Routed,
                outputs: Residual,
                native_bf16: true,
            },
            _ => return None,
        };
        Some(plan)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_names_map_to_orthogonal_execution_properties() {
        let routed_down = ProjectionPlan::from_operation(
            "materializec.borrow-routed-native-bf16-gemm-wmma-residual",
        )
        .unwrap();
        assert_eq!(routed_down.backend, ProjectionBackend::Wmma);
        assert_eq!(routed_down.topology, ProjectionTopology::Routed);
        assert_eq!(routed_down.outputs, ProjectionOutputs::Residual);
        assert!(routed_down.native_bf16);

        let dense_pair =
            ProjectionPlan::from_operation("materializec.borrow-native-bf16-gemm-blas-pair")
                .unwrap();
        assert_eq!(dense_pair.backend, ProjectionBackend::Blas);
        assert_eq!(dense_pair.topology, ProjectionTopology::Dense);
        assert_eq!(dense_pair.outputs, ProjectionOutputs::Pair);
    }

    #[test]
    fn non_projection_operations_are_not_classified() {
        assert!(ProjectionPlan::from_operation("materializec.softmax-f32").is_none());
    }

    #[test]
    fn dense_tile_preserves_the_tuned_launch_shape() {
        assert_eq!(MmaTileSpec::DENSE.rows, 128);
        assert_eq!(MmaTileSpec::DENSE.columns, 128);
        assert_eq!(MmaTileSpec::DENSE.reduction, 16);
        assert_eq!(MmaTileSpec::DENSE.waves, 8);
        assert_eq!(MmaTileSpec::DENSE.threads, 256);
        assert_eq!(MmaTileSpec::DENSE.minimum_rows, 64);
        assert_eq!(MmaTileSpec::DENSE.column_fragments(), 8);
    }
}
