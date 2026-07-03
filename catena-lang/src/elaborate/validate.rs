use hexpr::{Hexpr, Operation, try_interpret};
use metacat::theory::{
    RawTheorySet, Theory, TheoryId, TheorySet,
    ast::{RawTheory, RawTheoryArrow},
    transitive_dependency_subset,
};

use crate::{
    elaborate::{ElaborateError, NAT_THEORY},
    prefixes::{CONST_PREFIX, GENERATED_COPY_PREFIX, GENERATED_VARIABLE_PREFIX, NAME_PREFIX},
};

const RESERVED_OPERATION_PREFIXES: &[&str] = &[NAME_PREFIX, CONST_PREFIX, GENERATED_COPY_PREFIX];
const RESERVED_VARIABLE_PREFIXES: &[&str] = &[GENERATED_VARIABLE_PREFIX];

pub(crate) fn pre_elaboration_invariants(raw: &RawTheorySet) -> Result<(), ElaborateError> {
    check_reserved_operation_prefixes(raw)?;
    check_reserved_variable_prefixes(raw)?;
    check_type_map_domains(raw)?;
    Ok(())
}

fn check_type_map_domains(raw: &RawTheorySet) -> Result<(), ElaborateError> {
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

fn check_reserved_operation_prefixes(raw: &RawTheorySet) -> Result<(), ElaborateError> {
    for (theory_name, theory) in &raw.theories {
        for arrow_name in theory.arrows.keys() {
            if let Some(prefix) = RESERVED_OPERATION_PREFIXES
                .iter()
                .copied()
                .find(|prefix| arrow_name.as_str().starts_with(prefix))
            {
                return Err(ElaborateError::ReservedOperationPrefix {
                    theory: theory_name.to_string(),
                    arrow: arrow_name.to_string(),
                    prefix,
                });
            }
        }
    }
    Ok(())
}

fn check_reserved_variable_prefixes(raw: &RawTheorySet) -> Result<(), ElaborateError> {
    for (theory_name, theory) in &raw.theories {
        for (arrow_name, arrow) in &theory.arrows {
            for map in [&arrow.type_maps.0, &arrow.type_maps.1] {
                check_reserved_variables_in_hexpr(theory_name, arrow_name, map)?;
            }
            if let Some(definition) = &arrow.definition {
                check_reserved_variables_in_hexpr(theory_name, arrow_name, definition)?;
            }
        }
    }
    Ok(())
}

fn check_reserved_variables_in_hexpr(
    theory_name: &Operation,
    arrow_name: &Operation,
    expr: &Hexpr,
) -> Result<(), ElaborateError> {
    match expr {
        Hexpr::Composition(exprs) | Hexpr::Tensor(exprs) => {
            for expr in exprs {
                check_reserved_variables_in_hexpr(theory_name, arrow_name, expr)?;
            }
        }
        Hexpr::Frobenius { sources, targets } => {
            for variable in sources.iter().chain(targets) {
                let variable = variable.to_string();
                if let Some(prefix) = RESERVED_VARIABLE_PREFIXES
                    .iter()
                    .copied()
                    .find(|prefix| variable.starts_with(prefix))
                {
                    return Err(ElaborateError::ReservedVariablePrefix {
                        theory: theory_name.to_string(),
                        arrow: arrow_name.to_string(),
                        variable,
                        prefix,
                    });
                }
            }
        }
        Hexpr::Operation(_) => {}
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
