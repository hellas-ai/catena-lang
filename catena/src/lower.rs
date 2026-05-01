use thiserror::Error;

use open_hypergraphs::strict::vec::FiniteFunction;
use std::collections::HashMap;

use hexpr::{Operation, try_interpret};
use metacat::ssa::SSAError;
use metacat::{check::check, syntax::TheoryBundle, theory::OperationKey, tree::Tree};
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

/// An error during [`lower`]ing of a term
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
    #[error("Invalid hexpr: {0}")]
    InvalidHexpr(#[from] hexpr::interpret::Error<metacat::theory::Error>),
    #[error("Typecheck failed: {0:?}")]
    TypecheckError(metacat::check::Error<OperationKey>),
}

/// Lower a term by applying passes until the specified pass
/// TODO: add a post-processing hook on `lower` to transform any pass into readable strings - used
/// for lower command -> svg
pub fn lower(
    bundle: &TheoryBundle,
    until: Pass,
    definition: &str,
) -> Result<OpenHypergraph<Tree<(), OperationKey>, OperationKey>, LowerError> {
    let key: Operation = definition
        .parse()
        .map_err(|_| LowerError::InvalidDefinition(definition.to_string()))?;

    let declaration = bundle
        .definitions
        .get(&key)
        .ok_or_else(|| LowerError::UnknownDefinition(definition.to_string()))?;

    // Get term from declaration & key
    // NOTE: we *must* inline before typechecking: we need annotated nodes to be specialised to the
    // types applied to each definition.
    let mut current = declaration_term(bundle, &key)?;
    let current = inline(bundle, &mut current)?;

    // Check inlined
    let mut current = compute_types(bundle, declaration, current)?;

    // Run subsequent passes in order, stopping after the requested one
    if until != Pass::Check {
        for (pass, apply) in lower_passes(bundle)? {
            current = apply(&current)?;
            current.quotient().map_err(LowerError::InvalidQuotient)?;
            if pass == until {
                break;
            }
        }
    }

    Ok(current)
}

/// Construct the compiler lowering passes
fn lower_passes(bundle: &TheoryBundle) -> Result<Vec<LowerPass>, LowerError> {
    let bound_key = bundle.object_theory.get_operation_key("bound").unwrap();
    let value_key = bundle.object_theory.get_operation_key("value").unwrap();
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
    bundle: &TheoryBundle,
    t: &mut OpenHypergraph<(), Arr>,
) -> Result<OpenHypergraph<(), Arr>, LowerError> {
    let inline = {
        let names = ["f32.sum", "ones-2d", "id-matrix-2d"];
        let mut inline_defs = HashMap::new();
        for name in names {
            let op: Operation = name
                .parse()
                .map_err(|_| LowerError::InvalidDefinition(name.to_string()))?;
            let arrow = declaration_term(bundle, &op)?;
            let key = bundle
                .arrow_theory
                .get_operation_key(name)
                .ok_or(LowerError::UnknownOperation(name.to_string()))?;

            inline_defs.insert(key, arrow);
        }
        Inline {
            definitions: inline_defs,
        }
    };
    t.quotient().unwrap();
    Ok(inline.map_arrow(t))
}

fn declaration_term(
    bundle: &TheoryBundle,
    key: &Operation,
) -> Result<OpenHypergraph<(), Arr>, LowerError> {
    let hexpr = bundle
        .definitions
        .get(key)
        .and_then(|decl| decl.definition.clone())
        .ok_or_else(|| LowerError::UnknownDefinition(key.to_string()))?;

    Ok(forget_labels(try_interpret(&bundle.arrow_theory, &hexpr)?))
}

fn compute_types(
    bundle: &TheoryBundle,
    declaration: &metacat::syntax::Declaration,
    term: OpenHypergraph<(), Arr>,
) -> Result<OpenHypergraph<Obj, Arr>, LowerError> {
    let mut term = term;
    let source = forget_labels(try_interpret(
        &bundle.object_theory,
        &declaration.source_map,
    )?);
    let target = forget_labels(try_interpret(
        &bundle.object_theory,
        &declaration.target_map,
    )?);
    let result = check(&bundle.arrow_theory, source, target, &mut term)
        .map_err(LowerError::TypecheckError)?;
    Ok(term.with_nodes(|_| result).unwrap())
}

fn forget_labels<O, A>(f: OpenHypergraph<O, A>) -> OpenHypergraph<(), A> {
    f.map_nodes(|_| ())
}
