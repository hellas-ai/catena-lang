use hexpr::{Operation, try_interpret};
use metacat::theory::{
    RawTheorySet, Theory, TheoryId, TheorySet,
    ast::{RawTheory, RawTheoryArrow},
    transitive_dependency_subset,
};

use crate::elaborate::{ElaborateError, NAT_THEORY};

pub(crate) fn pre_elaboration_invariants(raw: &RawTheorySet) -> Result<(), ElaborateError> {
    for theory in raw.theories.values() {
        if theory.syntax_category.as_str() == NAT_THEORY {
            continue;
        }

        let syntax = interpreted_syntax(raw, theory)?;
        for arrow in theory.arrows.values() {
            validate_type_map_domains_match(&syntax, &theory.name, arrow)?;
        }
    }

    Ok(())
}

fn interpreted_syntax(raw: &RawTheorySet, theory: &RawTheory) -> Result<Theory, ElaborateError> {
    let syntax_theory_name = theory.syntax_category.clone();
    let raw_syntax_dependencies = transitive_dependency_subset([syntax_theory_name.clone()], raw)?;
    let syntax_dependencies = TheorySet::from_raw(raw_syntax_dependencies)?;
    syntax_dependencies
        .theories
        .get(&TheoryId(syntax_theory_name))
        .cloned()
        .ok_or_else(|| {
            ElaborateError::MissingInterpretedSyntaxTheory(theory.syntax_category.to_string())
        })
}

fn validate_type_map_domains_match(
    syntax: &Theory,
    theory_name: &Operation,
    raw: &RawTheoryArrow,
) -> Result<(), ElaborateError> {
    let interpreted_source =
        try_interpret(&syntax.local_signature(), &raw.type_maps.0).map_err(|error| {
            ElaborateError::NameSourceTypeMapInterpretation {
                theory: theory_name.to_string(),
                arrow: raw.name.to_string(),
                map: raw.type_maps.0.clone(),
                error,
            }
        })?;
    let interpreted_target =
        try_interpret(&syntax.local_signature(), &raw.type_maps.1).map_err(|error| {
            ElaborateError::NameTargetTypeMapInterpretation {
                theory: theory_name.to_string(),
                arrow: raw.name.to_string(),
                map: raw.type_maps.1.clone(),
                error,
            }
        })?;

    let source_domain = interpreted_source.sources.len();
    let target_domain = interpreted_target.sources.len();
    if source_domain == target_domain {
        return Ok(());
    }

    Err(ElaborateError::TypeMapDomainMismatch {
        theory: theory_name.to_string(),
        arrow: raw.name.to_string(),
        source_domain: source_domain.to_string(),
        target_domain: target_domain.to_string(),
    })
}
