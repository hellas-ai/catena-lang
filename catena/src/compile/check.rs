use hexpr::try_interpret;
use metacat::{check::check, syntax::TheoryBundle, theory::OperationKey};
use open_hypergraphs::lax::OpenHypergraph;
use thiserror::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CheckReport {
    pub definitions_checked: usize,
}

#[derive(Error, Debug)]
pub enum CheckError {
    #[error("invalid hexpr: {0}")]
    InvalidHexpr(#[from] hexpr::interpret::Error<metacat::theory::Error>),
    #[error("definition {definition} failed typecheck: {error:?}")]
    Typecheck {
        definition: String,
        error: metacat::check::Error<OperationKey>,
    },
}

pub fn check_bundle(bundle: &TheoryBundle) -> Result<CheckReport, CheckError> {
    let mut definitions: Vec<_> = bundle.definitions.iter().collect();
    definitions.sort_by_key(|(name, _)| name.to_string());

    let mut definitions_checked = 0;
    for (name, declaration) in definitions {
        let hexpr = declaration
            .definition
            .as_ref()
            .expect("definition entries always have a body");
        let mut term = forget_labels(try_interpret(&bundle.arrow_theory, hexpr)?);
        let source = forget_labels(try_interpret(
            &bundle.object_theory,
            &declaration.source_map,
        )?);
        let target = forget_labels(try_interpret(
            &bundle.object_theory,
            &declaration.target_map,
        )?);

        check(&bundle.arrow_theory, source, target, &mut term).map_err(|error| {
            CheckError::Typecheck {
                definition: name.to_string(),
                error,
            }
        })?;
        definitions_checked += 1;
    }

    Ok(CheckReport {
        definitions_checked,
    })
}

fn forget_labels<O, A>(f: OpenHypergraph<O, A>) -> OpenHypergraph<(), A> {
    f.map_nodes(|_| ())
}
