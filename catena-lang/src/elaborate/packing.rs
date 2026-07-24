use hexpr::{Hexpr, Variable};

use crate::{
    elaborate::ElaborateError,
    stdlib::constants::{PRODUCT_TYPE, UNIT_TYPE},
};

/// Pack a tensor boundary into the single object used by function types.
///
/// Zero objects become the unit object, one remains unchanged, and multiple
/// objects become a left-associated product.
pub(super) fn pack_object(
    object_count: usize,
    fresh_variable: &mut impl FnMut() -> Result<Variable, ElaborateError>,
) -> Result<Hexpr, ElaborateError> {
    match object_count {
        0 => operation(UNIT_TYPE),
        1 => Ok(identity(fresh_variable()?)),
        2 => operation(PRODUCT_TYPE),
        n => Ok(Hexpr::Composition(vec![
            Hexpr::Tensor(vec![
                pack_object(n - 1, fresh_variable)?,
                identity(fresh_variable()?),
            ]),
            operation(PRODUCT_TYPE)?,
        ])),
    }
}

fn identity(variable: Variable) -> Hexpr {
    Hexpr::Frobenius {
        sources: vec![variable.clone()],
        targets: vec![variable],
    }
}

fn operation(name: &str) -> Result<Hexpr, ElaborateError> {
    name.parse()
        .map(Hexpr::Operation)
        .map_err(|_| ElaborateError::InvalidGeneratedOperation(name.to_string()))
}
