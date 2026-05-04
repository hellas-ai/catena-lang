use hexpr::Operation;
use metacat::theory::{Theory, TheoryArrow};
use open_hypergraphs::category::Arrow;
use open_hypergraphs::lax::OpenHypergraph;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum LiftError {
    #[error("cannot lift {prefix}: target is not a user theory")]
    TargetIsNotUserTheory { prefix: &'static str },
    #[error("cannot lift {prefix}: missing object constructor `{object}` in syntax theory")]
    MissingObject {
        prefix: &'static str,
        object: String,
    },
    #[error(
        "cannot lift {prefix}: object constructor `{object}` has profile {source_arity} -> {target_arity}, expected {expected_source_arity} -> {expected_target_arity}"
    )]
    InvalidObjectProfile {
        prefix: &'static str,
        object: String,
        source_arity: usize,
        target_arity: usize,
        expected_source_arity: usize,
        expected_target_arity: usize,
    },
    #[error("cannot lift {prefix}: invalid lifted operation name `{name}`")]
    InvalidLiftedOperationName { prefix: &'static str, name: String },
}

pub fn lift_with_tensor(
    source: &Theory,
    target: &Theory,
    syntax: &Theory,
    prefix: &'static str,
    tensor: &str,
    unit: &str,
    excluded_prefixes: &[&str],
) -> Result<Theory, LiftError> {
    let tensor_key = require_object(syntax, prefix, tensor, 2, 1)?;
    let unit_key = require_object(syntax, prefix, unit, 0, 1)?;
    let mut theory = target.clone();
    let Theory::Theory { arrows, .. } = &mut theory else {
        return Err(LiftError::TargetIsNotUserTheory { prefix });
    };

    let mut operations = source_arrows(source, excluded_prefixes);
    operations.sort_by(|(left, _), (right, _)| left.cmp(right));

    for (name, arrow) in operations {
        let lifted_name = format!("{prefix}.{name}");
        let lifted_operation: Operation =
            lifted_name
                .parse()
                .map_err(|_| LiftError::InvalidLiftedOperationName {
                    prefix,
                    name: lifted_name.clone(),
                })?;
        let lifted_source =
            lift_object_map(&arrow.type_maps.0, syntax, &tensor_key, &unit_key, prefix)?;
        let lifted_target =
            lift_object_map(&arrow.type_maps.1, syntax, &tensor_key, &unit_key, prefix)?;

        let mut lifted_arrow = arrow.clone();
        lifted_arrow.name = lifted_operation.clone();
        lifted_arrow.type_maps = (lifted_source, lifted_target);
        lifted_arrow.definition = None;
        lifted_arrow.raw.name = lifted_operation.clone();
        lifted_arrow.raw.definition = None;
        arrows.insert(lifted_operation, lifted_arrow);
    }

    Ok(theory)
}

fn source_arrows(source: &Theory, excluded_prefixes: &[&str]) -> Vec<(Operation, TheoryArrow)> {
    match source {
        Theory::Nat => Vec::new(),
        Theory::Theory { arrows, .. } => arrows
            .iter()
            .filter(|(name, _)| {
                let name = name.to_string();
                !excluded_prefixes
                    .iter()
                    .any(|prefix| name.starts_with(&format!("{prefix}.")))
            })
            .map(|(name, arrow)| (name.clone(), arrow.clone()))
            .collect(),
    }
}

fn require_object(
    syntax: &Theory,
    prefix: &'static str,
    object: &str,
    expected_source_arity: usize,
    expected_target_arity: usize,
) -> Result<Operation, LiftError> {
    let operation: Operation = object.parse().map_err(|_| LiftError::MissingObject {
        prefix,
        object: object.to_string(),
    })?;
    let arrow = syntax
        .get_arrow(&operation)
        .ok_or_else(|| LiftError::MissingObject {
            prefix,
            object: object.to_string(),
        })?;
    let source_arity = arrow.type_maps.0.target().len();
    let target_arity = arrow.type_maps.1.target().len();
    if source_arity == expected_source_arity && target_arity == expected_target_arity {
        Ok(operation)
    } else {
        Err(LiftError::InvalidObjectProfile {
            prefix,
            object: object.to_string(),
            source_arity,
            target_arity,
            expected_source_arity,
            expected_target_arity,
        })
    }
}

fn lift_object_map(
    map: &OpenHypergraph<(), Operation>,
    syntax: &Theory,
    tensor_key: &Operation,
    unit_key: &Operation,
    prefix: &'static str,
) -> Result<OpenHypergraph<(), Operation>, LiftError> {
    for op in &map.hypergraph.edges {
        if syntax.get_arrow(op).is_none() {
            return Err(LiftError::MissingObject {
                prefix,
                object: op.to_string(),
            });
        }
    }

    let mut lifted = map.clone();

    match lifted.targets.len() {
        0 => {
            let unit_node = lifted.new_node(());
            lifted.new_edge(unit_key.clone(), (Vec::new(), vec![unit_node]));
            lifted.targets = vec![unit_node];
        }
        1 => {}
        _ => {
            let mut inputs = lifted.targets.clone();
            while inputs.len() > 1 {
                let left = inputs.remove(0);
                let right = inputs.remove(0);
                let product_node = lifted.new_node(());
                lifted.new_edge(tensor_key.clone(), (vec![left, right], vec![product_node]));
                inputs.insert(0, product_node);
            }
            lifted.targets = inputs;
        }
    }

    Ok(lifted)
}
