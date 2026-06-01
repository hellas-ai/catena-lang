use std::collections::{HashMap, HashSet};

use metacat::tree::Tree;

use crate::{compile::cuda::abi::CudaAbiError, lang::Obj};

#[derive(Debug, Clone)]
pub(super) struct DimensionExpr {
    pub(super) expr: String,
    pub(super) is_static: bool,
}

#[derive(Debug, Clone)]
pub(super) struct ShapeExpr {
    pub(super) dimensions: Vec<DimensionExpr>,
}

impl ShapeExpr {
    pub(super) fn product_expr(&self) -> String {
        self.dimensions
            .iter()
            .map(|dimension| dimension.expr.as_str())
            .collect::<Vec<_>>()
            .join(" * ")
    }

    pub(super) fn is_static(&self) -> bool {
        self.dimensions.iter().all(|dimension| dimension.is_static)
    }

    pub(super) fn cuda_array_suffix(&self) -> String {
        self.dimensions
            .iter()
            .map(|dimension| format!("[{}]", dimension.expr))
            .collect::<String>()
    }

    pub(super) fn rank(&self) -> usize {
        self.dimensions.len()
    }
}

pub(super) fn shape_expr(
    dimensions: &[&Obj],
    extent_names: &HashMap<usize, String>,
    static_extent_leaves: &HashSet<usize>,
    missing_extent: impl Fn(usize) -> CudaAbiError + Copy,
    invalid_shape: impl Fn() -> CudaAbiError + Copy,
) -> Result<ShapeExpr, CudaAbiError> {
    let dimensions = dimensions
        .iter()
        .map(|dimension| {
            dimension_expr_with_static(
                dimension,
                extent_names,
                static_extent_leaves,
                missing_extent,
                invalid_shape,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ShapeExpr { dimensions })
}

pub(super) fn dimension_expr(
    dimension: &Obj,
    extent_names: &HashMap<usize, String>,
    missing_extent: impl Fn(usize) -> CudaAbiError + Copy,
    invalid_shape: impl Fn() -> CudaAbiError + Copy,
) -> Result<String, CudaAbiError> {
    match dimension {
        Tree::Leaf(leaf, _) => extent_names
            .get(leaf)
            .cloned()
            .ok_or_else(|| missing_extent(*leaf)),
        Tree::Node(op, 0, children) if op.to_string() == "1" && children.is_empty() => {
            Ok("1".to_string())
        }
        Tree::Node(op, 0, children)
            if matches!(op.to_string().as_str(), "nat.mul" | "*") && children.len() == 2 =>
        {
            let lhs = dimension_expr(&children[0], extent_names, missing_extent, invalid_shape)?;
            let rhs = dimension_expr(&children[1], extent_names, missing_extent, invalid_shape)?;
            Ok(format!("({lhs} * {rhs})"))
        }
        Tree::Node(op, 0, children) if op.to_string() == "nat.ceil-div" && children.len() == 2 => {
            let numerator =
                dimension_expr(&children[0], extent_names, missing_extent, invalid_shape)?;
            let denominator =
                dimension_expr(&children[1], extent_names, missing_extent, invalid_shape)?;
            Ok(format!(
                "(({numerator} + {denominator} - 1) / {denominator})"
            ))
        }
        _ => {
            // TODO: allow literal dimension values. Today dimensions must be
            // backed by extent leaves and supported nat operations.
            Err(invalid_shape())
        }
    }
}

fn dimension_expr_with_static(
    dimension: &Obj,
    extent_names: &HashMap<usize, String>,
    static_extent_leaves: &HashSet<usize>,
    missing_extent: impl Fn(usize) -> CudaAbiError + Copy,
    invalid_shape: impl Fn() -> CudaAbiError + Copy,
) -> Result<DimensionExpr, CudaAbiError> {
    match dimension {
        Tree::Leaf(leaf, _) => Ok(DimensionExpr {
            expr: extent_names
                .get(leaf)
                .cloned()
                .ok_or_else(|| missing_extent(*leaf))?,
            is_static: static_extent_leaves.contains(leaf),
        }),
        Tree::Node(op, 0, children) if op.to_string() == "1" && children.is_empty() => {
            Ok(DimensionExpr {
                expr: "1".to_string(),
                is_static: true,
            })
        }
        Tree::Node(op, 0, children)
            if matches!(op.to_string().as_str(), "nat.mul" | "*") && children.len() == 2 =>
        {
            let lhs = dimension_expr_with_static(
                &children[0],
                extent_names,
                static_extent_leaves,
                missing_extent,
                invalid_shape,
            )?;
            let rhs = dimension_expr_with_static(
                &children[1],
                extent_names,
                static_extent_leaves,
                missing_extent,
                invalid_shape,
            )?;
            Ok(DimensionExpr {
                expr: format!("({} * {})", lhs.expr, rhs.expr),
                is_static: lhs.is_static && rhs.is_static,
            })
        }
        Tree::Node(op, 0, children) if op.to_string() == "nat.ceil-div" && children.len() == 2 => {
            let numerator = dimension_expr_with_static(
                &children[0],
                extent_names,
                static_extent_leaves,
                missing_extent,
                invalid_shape,
            )?;
            let denominator = dimension_expr_with_static(
                &children[1],
                extent_names,
                static_extent_leaves,
                missing_extent,
                invalid_shape,
            )?;
            Ok(DimensionExpr {
                expr: format!(
                    "(({} + {} - 1) / {})",
                    numerator.expr, denominator.expr, denominator.expr
                ),
                is_static: numerator.is_static && denominator.is_static,
            })
        }
        _ => {
            // TODO: allow literal dimension values and mark them static.
            Err(invalid_shape())
        }
    }
}

pub(super) fn rank_terms(dimension: &Obj) -> Option<Vec<Obj>> {
    let Tree::Node(rank, 0, children) = dimension else {
        return None;
    };
    let expected = match rank.to_string().as_str() {
        "1d" => 1,
        "2d" => 2,
        "3d" => 3,
        _ => return None,
    };
    if children.len() != expected {
        return None;
    };
    Some(children.clone())
}

pub(super) fn memory_dimensions(dimensions: &Obj) -> Option<Vec<&Obj>> {
    let Tree::Node(rank, 0, children) = dimensions else {
        return None;
    };
    let expected = match rank.to_string().as_str() {
        "1d" => 1,
        "2d" => 2,
        "3d" => 3,
        _ => return None,
    };
    if children.len() != expected {
        return None;
    }
    Some(children.iter().collect())
}

pub(super) fn extent_leaf(obj: &Obj) -> Option<usize> {
    let extent = unwrap_val(obj)?;
    let Tree::Node(extent, 0, children) = extent else {
        return None;
    };
    if extent.to_string() != "extent" {
        return None;
    }
    let [Tree::Leaf(leaf, _)] = children.as_slice() else {
        return None;
    };
    Some(*leaf)
}

pub(super) fn unwrap_val(obj: &Obj) -> Option<&Obj> {
    match obj {
        Tree::Node(wrapper, 0, children) if wrapper.to_string() == "val" => {
            let [inner] = children.as_slice() else {
                return None;
            };
            Some(inner)
        }
        _ => None,
    }
}
