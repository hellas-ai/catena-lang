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

    pub const ROUTED_PAIR: Self = Self {
        rows: 64,
        columns: 64,
        reduction: 16,
        waves: 4,
        threads: 128,
        minimum_rows: 4,
    };

    pub const ROUTED_DOWN: Self = Self {
        rows: 64,
        columns: 128,
        reduction: 16,
        waves: 4,
        threads: 128,
        minimum_rows: 4,
    };

    pub const fn column_fragments(self) -> u32 {
        self.columns / 16
    }
}

pub(in crate::codegen) fn render_routed_tile_template(
    out: &mut String,
    source: &str,
    kernel_name: &str,
    tile: MmaTileSpec,
) {
    let rendered = source
        .replace("__KERNEL__", kernel_name)
        .replace("__TILE_ROWS__", &tile.rows.to_string())
        .replace("__TILE_COLUMNS__", &tile.columns.to_string())
        .replace("__TILE_REDUCTION__", &tile.reduction.to_string())
        .replace("__TILE_WAVES__", &tile.waves.to_string())
        .replace("__TILE_FRAGMENTS__", &tile.column_fragments().to_string());
    out.push_str(&rendered);
}

pub(in crate::codegen) fn render_stable_route_grouping(out: &mut String, kernel_name: &str) {
    let source = r#"
__global__ void __KERNEL___group(
    const uint64_t *selected, uint64_t route_count, uint64_t expert_count,
    uint64_t *offsets, uint64_t *permutation) {
    uint64_t expert = (uint64_t)blockIdx.x;
    uint32_t lane = threadIdx.x;
    uint64_t offset = 0;
    for (uint64_t route_base = 0; route_base < route_count; route_base += 32) {
        uint64_t route = route_base + lane;
        unsigned long long lower = __ballot(
            route < route_count && selected[route] < expert);
        offset += (uint64_t)__popcll(lower);
    }
    if (lane == 0) {
        offsets[expert] = offset;
    }
    uint64_t written = 0;
    for (uint64_t route_base = 0; route_base < route_count; route_base += 32) {
        uint64_t route = route_base + lane;
        bool matches = route < route_count && selected[route] == expert;
        unsigned long long match_mask = __ballot(matches);
        unsigned long long preceding = lane == 0 ? 0 : ((1ull << lane) - 1);
        if (matches) {
            permutation[offset + written + (uint64_t)__popcll(match_mask & preceding)] = route;
        }
        written += (uint64_t)__popcll(match_mask);
    }
    if (lane == 0 && expert + 1 == expert_count) {
        offsets[expert_count] = offset + written;
        catena_assert(offsets[expert_count] == route_count);
    }
}

"#;
    out.push_str(&source.replace("__KERNEL__", kernel_name));
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

    #[test]
    fn routed_tiles_preserve_pair_and_down_shapes() {
        assert_eq!(MmaTileSpec::ROUTED_PAIR.rows, 64);
        assert_eq!(MmaTileSpec::ROUTED_PAIR.columns, 64);
        assert_eq!(MmaTileSpec::ROUTED_PAIR.column_fragments(), 4);
        assert_eq!(MmaTileSpec::ROUTED_DOWN.rows, 64);
        assert_eq!(MmaTileSpec::ROUTED_DOWN.columns, 128);
        assert_eq!(MmaTileSpec::ROUTED_DOWN.column_fragments(), 8);
        assert_eq!(MmaTileSpec::ROUTED_PAIR.threads, 128);
        assert_eq!(MmaTileSpec::ROUTED_DOWN.threads, 128);
    }

    #[test]
    fn routed_template_substitutes_compile_time_tile_values() {
        let mut source = String::new();
        render_routed_tile_template(
            &mut source,
            "__KERNEL__ __TILE_ROWS__ __TILE_COLUMNS__ __TILE_REDUCTION__ __TILE_WAVES__ __TILE_FRAGMENTS__",
            "routed",
            MmaTileSpec::ROUTED_DOWN,
        );
        assert_eq!(source, "routed 64 128 16 4 8");
    }

    #[test]
    fn stable_route_grouping_is_rendered_once_from_a_shared_template() {
        let mut source = String::new();
        render_stable_route_grouping(&mut source, "routes");
        assert!(source.contains("__global__ void routes_group("));
        assert!(source.contains("selected[route] < expert"));
        assert!(source.contains("selected[route] == expert"));
        assert!(source.contains("offsets[expert_count] == route_count"));
    }
}
