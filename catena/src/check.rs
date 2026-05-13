//! Typechecking and elaboration-by-interleaving of theories
use metacat::{
    check::check as metacat_check,
    theory::{RawTheorySet, Theory, TheorySet},
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CheckError {
    #[error(transparent)]
    Load(#[from] metacat::theory::LoadError),
    #[error("definition check failed in theory `{theory}`, definition `{definition}`: {error:?}")]
    Definition {
        theory: String,
        definition: String,
        error: metacat::check::Error<hexpr::Operation>,
    },
}

const SYNTAX_THEORY: &str = "syntax";
const NAT_THEORY: &str = "nat";

/// Interpret and typecheck an already-elaborated raw theory set.
pub fn check(elaborated: &RawTheorySet) -> Result<TheorySet, CheckError> {
    // Interpret all theories to get a TheorySet
    let interpreted = interpret_all(&elaborated)?;

    // Typecheck all definitions
    check_all(&interpreted)?;
    Ok(interpreted)
}

// Turn elaborated raw theories into a TheorySet.
// Should just be able to use "vanilla metacat" to do this.
fn interpret_all(elaborated: &RawTheorySet) -> Result<TheorySet, CheckError> {
    Ok(TheorySet::from_raw(elaborated.clone())?)
}

// For now, return yes/no for success/fail. Will return more deetail later.
fn check_all(elaborated: &TheorySet) -> Result<(), CheckError> {
    for (id, theory) in &elaborated.theories {
        if id.0.as_str() == NAT_THEORY || id.0.as_str() == SYNTAX_THEORY {
            continue;
        }
        check_definitions(theory, &id.to_string())?;
    }
    Ok(())
}

fn check_definitions(elaborated: &Theory, theory_name: &str) -> Result<(), CheckError> {
    let Theory::Theory { arrows, .. } = elaborated else {
        return Ok(());
    };

    for (name, arrow) in arrows {
        let Some(mut body) = arrow.definition.clone() else {
            continue;
        };
        metacat_check(
            elaborated,
            arrow.type_maps.0.clone(),
            arrow.type_maps.1.clone(),
            &mut body,
        )
        .map_err(|error| CheckError::Definition {
            theory: theory_name.to_string(),
            definition: name.to_string(),
            error,
        })?;
    }

    Ok(())
}
