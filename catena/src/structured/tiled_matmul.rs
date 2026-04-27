use super::cuda::{
    render_cuda, CudaDecl, CudaKernelEnv, CudaLaunchConfig, CudaRenderMode, CudaStmt,
};
use super::ir::{EntryPoint, Param, Primitive, Program, Stmt};
use super::ramsey::ArrowSemantics;

#[derive(Debug, Clone, Copy)]
pub struct TiledMatmulSemantics;

impl ArrowSemantics for TiledMatmulSemantics {
    fn actions(&self, op: &str) -> Vec<Stmt> {
        if self.counted_loop(op).is_some() {
            return Vec::new();
        }
        if op.contains(".barrier.") {
            return vec![Stmt::Barrier];
        }
        vec![Stmt::Primitive(Primitive {
            name: op.to_string(),
            code: String::new(),
        })]
    }

    fn counted_loop(&self, op: &str) -> Option<(String, String)> {
        if op.contains("for-num-tiles.prelude") {
            Some(("p".to_string(), "num_tiles".to_string()))
        } else if op.contains("for-tile.prelude") {
            Some(("q".to_string(), "TILE".to_string()))
        } else {
            None
        }
    }
}

pub fn program(definition: &str, body: Vec<Stmt>) -> Program {
    let env = kernel_env();
    Program {
        name: sanitize_ident(definition),
        entry: EntryPoint {
            name: sanitize_ident(definition),
            params: env.params,
        },
        body,
    }
}

impl Program {
    pub fn render_c(&self) -> String {
        render_cuda(self, &kernel_env(), CudaRenderMode::Kernel, lower_primitive)
    }

    pub fn render_cuda_with_launch(&self) -> String {
        render_cuda(
            self,
            &kernel_env(),
            CudaRenderMode::KernelWithLaunch,
            lower_primitive,
        )
    }
}

fn kernel_env() -> CudaKernelEnv {
    CudaKernelEnv {
        tile_macro: "TILE".to_string(),
        params: vec![
            Param {
                ty: "const float*".to_string(),
                name: "A".to_string(),
            },
            Param {
                ty: "const float*".to_string(),
                name: "B".to_string(),
            },
            Param {
                ty: "float*".to_string(),
                name: "C".to_string(),
            },
            Param {
                ty: "int".to_string(),
                name: "M".to_string(),
            },
            Param {
                ty: "int".to_string(),
                name: "N".to_string(),
            },
            Param {
                ty: "int".to_string(),
                name: "K".to_string(),
            },
        ],
        shared: vec![
            CudaDecl {
                ty: "float".to_string(),
                name: "tile_A[TILE][TILE]".to_string(),
                init: None,
            },
            CudaDecl {
                ty: "float".to_string(),
                name: "tile_B[TILE][TILE]".to_string(),
                init: None,
            },
        ],
        prelude: vec![
            CudaStmt::Decl(CudaDecl {
                ty: "int".to_string(),
                name: "row".to_string(),
                init: Some("blockIdx.y * TILE + threadIdx.y".to_string()),
            }),
            CudaStmt::Decl(CudaDecl {
                ty: "int".to_string(),
                name: "col".to_string(),
                init: Some("blockIdx.x * TILE + threadIdx.x".to_string()),
            }),
            CudaStmt::Decl(CudaDecl {
                ty: "int".to_string(),
                name: "ty".to_string(),
                init: Some("threadIdx.y".to_string()),
            }),
            CudaStmt::Decl(CudaDecl {
                ty: "int".to_string(),
                name: "tx".to_string(),
                init: Some("threadIdx.x".to_string()),
            }),
            CudaStmt::Decl(CudaDecl {
                ty: "int".to_string(),
                name: "num_tiles".to_string(),
                init: Some("(K + TILE - 1) / TILE".to_string()),
            }),
        ],
        launch: Some(CudaLaunchConfig {
            block: "TILE, TILE".to_string(),
            grid: "(N + TILE - 1) / TILE, (M + TILE - 1) / TILE".to_string(),
        }),
    }
}

fn lower_primitive(primitive: &Primitive) -> Vec<CudaStmt> {
    match primitive.name.as_str() {
        "gpu.tiled-matmul.init-acc" => vec![CudaStmt::Decl(CudaDecl {
            ty: "float".to_string(),
            name: "acc".to_string(),
            init: Some("0.0f".to_string()),
        })],
        "gpu.tiled-matmul.collectively-load-shared-tile" => vec![
            CudaStmt::Decl(CudaDecl {
                ty: "int".to_string(),
                name: "a_col".to_string(),
                init: Some("p * TILE + tx".to_string()),
            }),
            CudaStmt::Decl(CudaDecl {
                ty: "int".to_string(),
                name: "b_row".to_string(),
                init: Some("p * TILE + ty".to_string()),
            }),
            CudaStmt::Assign {
                lhs: "tile_A[ty][tx]".to_string(),
                rhs: "(row < M && a_col < K) ? A[row * K + a_col] : 0.0f".to_string(),
            },
            CudaStmt::Assign {
                lhs: "tile_B[ty][tx]".to_string(),
                rhs: "(b_row < K && col < N) ? B[b_row * N + col] : 0.0f".to_string(),
            },
        ],
        "gpu.tiled-matmul.dot-by-thread" => vec![CudaStmt::AddAssign {
            lhs: "acc".to_string(),
            rhs: "tile_A[ty][q] * tile_B[q][tx]".to_string(),
        }],
        "gpu.tiled-matmul.store-output" => vec![CudaStmt::If {
            condition: "row < M && col < N".to_string(),
            body: vec![CudaStmt::Assign {
                lhs: "C[row * N + col]".to_string(),
                rhs: "acc".to_string(),
            }],
        }],
        other => vec![CudaStmt::Comment(format!("TODO: primitive {other};"))],
    }
}

fn sanitize_ident(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}
