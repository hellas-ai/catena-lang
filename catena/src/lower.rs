use thiserror::Error;

use open_hypergraphs::strict::vec::FiniteFunction;
use std::collections::HashMap;

use hexpr::Operation;
use metacat::{check::check, ssa::SSAError, theory::Theory, tree::Tree};
use open_hypergraphs::lax::{OpenHypergraph, functor::Functor};

use crate::lang::{Arr, Obj};
use crate::pass::{
    discard_naturality::discard_naturality, erase::Erase, expand_eta::ExpandEta,
    forget_bound::ForgetBound, inline::Inline,
};

type Lowered = OpenHypergraph<Obj, Arr>;
type LowerPassFn = dyn Fn(&Lowered) -> Result<Lowered, LowerError>;
type LowerPass = (Pass, Box<LowerPassFn>);

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Pass {
    Check,
    Erase,
    ForgetBound,
    ExpandEta,
    DiscardNaturality,
}

#[derive(Error, Debug)]
pub enum LowerError {
    #[error("Invalid quotient: {0:?}")]
    InvalidQuotient(FiniteFunction),
    #[error("Unknown operation {0}")]
    UnknownOperation(String),
    #[error("Unknown definition {0}")]
    UnknownDefinition(String),
    #[error("Discard naturality pass failed: {0}")]
    DiscardNaturality(SSAError),
    #[error("Invalid definition name: {0}")]
    InvalidDefinition(String),
    #[error("Expected a user theory")]
    NotUserTheory,
    #[error("Typecheck failed: {0:?}")]
    TypecheckError(metacat::check::Error<Operation>),
}

pub fn lower(
    theory: &Theory,
    until: Pass,
    definition: &str,
) -> Result<OpenHypergraph<Tree<(), Operation>, Operation>, LowerError> {
    let key: Operation = definition
        .parse()
        .map_err(|_| LowerError::InvalidDefinition(definition.to_string()))?;

    let arrow = theory
        .get_arrow(&key)
        .ok_or_else(|| LowerError::UnknownDefinition(definition.to_string()))?;

    let mut current = declaration_term(theory, &key)?;
    let current = inline(theory, &mut current)?;
    let mut current = compute_types(theory, arrow, current)?;

    if until != Pass::Check {
        for (pass, apply) in lower_passes()? {
            current = apply(&current)?;
            current.quotient().map_err(LowerError::InvalidQuotient)?;
            if pass == until {
                break;
            }
        }
    }

    Ok(current)
}

fn lower_passes() -> Result<Vec<LowerPass>, LowerError> {
    let bound_key = parse_operation("bound")?;
    let value_key = parse_operation("value")?;
    let forget_bound = ForgetBound::new(bound_key, value_key);

    Ok(vec![
        (Pass::Erase, Box::new(|t| Ok(Erase.map_arrow(t)))),
        (
            Pass::ForgetBound,
            Box::new(move |t| Ok(forget_bound.map_arrow(t))),
        ),
        (Pass::ExpandEta, Box::new(|t| Ok(ExpandEta.map_arrow(t)))),
        (
            Pass::DiscardNaturality,
            Box::new(|t| discard_naturality(t.clone()).map_err(LowerError::DiscardNaturality)),
        ),
    ])
}

fn inline(
    theory: &Theory,
    t: &mut OpenHypergraph<(), Arr>,
) -> Result<OpenHypergraph<(), Arr>, LowerError> {
    let inline = {
        let names = ["f32.sum", "ones-2d", "id-matrix-2d"];
        let mut inline_defs = HashMap::new();
        for name in names {
            let op = parse_operation(name)?;
            let arrow = declaration_term(theory, &op)?;
            inline_defs.insert(op, arrow);
        }
        Inline {
            definitions: inline_defs,
        }
    };
    t.quotient().unwrap();
    Ok(inline.map_arrow(t))
}

fn declaration_term(
    theory: &Theory,
    key: &Operation,
) -> Result<OpenHypergraph<(), Arr>, LowerError> {
    theory
        .get_arrow(key)
        .and_then(|arrow| arrow.definition.clone())
        .ok_or_else(|| LowerError::UnknownDefinition(key.to_string()))
}

fn compute_types(
    theory: &Theory,
    arrow: &metacat::theory::TheoryArrow,
    mut term: OpenHypergraph<(), Arr>,
) -> Result<OpenHypergraph<Obj, Arr>, LowerError> {
    let result = check(
        theory,
        arrow.type_maps.0.clone(),
        arrow.type_maps.1.clone(),
        &mut term,
    )
    .map_err(LowerError::TypecheckError)?;
    Ok(term.with_nodes(|_| result).unwrap())
}

fn parse_operation(name: &str) -> Result<Operation, LowerError> {
    name.parse()
        .map_err(|_| LowerError::InvalidDefinition(name.to_string()))
}
