use std::collections::HashMap;

use crate::{
    compile::cuda::{
        abi::{CudaAbiError, CudaLaunch},
        boundary::GridShape,
        shape::dimension_expr,
    },
    lang::Obj,
};

pub(super) fn launch_from_grid_contract(
    grid_shape: &GridShape,
    extent_cuda_names: &HashMap<usize, String>,
) -> Result<CudaLaunch, CudaAbiError> {
    let grid = resolve_grid_launch(grid_shape, extent_cuda_names)?;
    Ok(CudaLaunch {
        block_expr: grid.block.join(", "),
        grid_expr: grid.grid.join(", "),
    })
}

struct GridLaunch {
    grid: Vec<String>,
    block: Vec<String>,
}

fn resolve_grid_launch(
    shape: &GridShape,
    extent_cuda_names: &HashMap<usize, String>,
) -> Result<GridLaunch, CudaAbiError> {
    Ok(GridLaunch {
        grid: resolve_dimension_names(&shape.grid, extent_cuda_names)?,
        block: resolve_dimension_names(&shape.block, extent_cuda_names)?,
    })
}

fn resolve_dimension_names(
    dimensions: &[Obj],
    extent_cuda_names: &HashMap<usize, String>,
) -> Result<Vec<String>, CudaAbiError> {
    dimensions
        .iter()
        .map(|dimension| {
            dimension_expr(
                dimension,
                extent_cuda_names,
                CudaAbiError::MissingGridExtent,
                || CudaAbiError::InvalidGridShape,
            )
        })
        .collect()
}
