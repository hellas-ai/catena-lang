use hexpr::{Hexpr, Operation};
use metacat::theory::{
    RawTheorySet,
    ast::{RawTheory, RawTheoryArrow},
};

use crate::{
    elaborate::ElaborateError,
    prefixes::{CONST_U32_PREFIX, CONST_U64_PREFIX},
};

#[derive(Debug, Clone, Copy)]
pub struct ConstantKind {
    prefix: &'static str,
    type_name: &'static str,
    hex_nibbles: usize,
}

pub const U64: ConstantKind = ConstantKind {
    prefix: CONST_U64_PREFIX,
    type_name: "u64",
    hex_nibbles: 16,
};

pub const U32: ConstantKind = ConstantKind {
    prefix: CONST_U32_PREFIX,
    type_name: "u32",
    hex_nibbles: 8,
};

/// For each operation `const.<type>.0x{c}` appearing in the source,
/// elaborate the theory with a constant `const.<type>.0x{c} : [] -> (<type> val)`.
pub fn elaborate(raw: &mut RawTheorySet, kind: ConstantKind) -> Result<(), ElaborateError> {
    let theory_names = raw.theories.keys().cloned().collect::<Vec<_>>();
    for theory_name in theory_names {
        let Some(theory) = raw.theories.get_mut(&theory_name) else {
            continue;
        };
        elaborate_theory(theory, kind)?;
    }
    Ok(())
}

fn elaborate_theory(theory: &mut RawTheory, kind: ConstantKind) -> Result<(), ElaborateError> {
    let constants = theory
        .arrows
        .values()
        .filter_map(|arrow| arrow.definition.as_ref())
        .flat_map(|definition| constants_in_hexpr(definition, kind))
        .collect::<Vec<_>>();

    for constant in constants {
        validate_constant(&constant, kind)?;
        theory
            .arrows
            .entry(constant.clone())
            .or_insert_with(|| const_arrow(constant, kind));
    }

    Ok(())
}

fn constants_in_hexpr(hexpr: &Hexpr, kind: ConstantKind) -> Vec<Operation> {
    let mut constants = Vec::new();
    collect_constants(hexpr, kind, &mut constants);
    constants
}

fn collect_constants(hexpr: &Hexpr, kind: ConstantKind, constants: &mut Vec<Operation>) {
    match hexpr {
        Hexpr::Composition(exprs) | Hexpr::Tensor(exprs) => {
            for expr in exprs {
                collect_constants(expr, kind, constants);
            }
        }
        Hexpr::Frobenius { .. } => {}
        Hexpr::Operation(op) if op.as_str().starts_with(kind.prefix) => {
            constants.push(op.clone());
        }
        Hexpr::Operation(_) => {}
    }
}

fn validate_constant(op: &Operation, kind: ConstantKind) -> Result<(), ElaborateError> {
    let literal =
        op.as_str()
            .strip_prefix(kind.prefix)
            .ok_or_else(|| ElaborateError::InvalidConstant {
                operation: op.to_string(),
                reason: format!("expected prefix `{}`", kind.prefix),
            })?;
    let Some(hex) = literal.strip_prefix("0x") else {
        return Err(ElaborateError::InvalidConstant {
            operation: op.to_string(),
            reason: "expected a hexadecimal literal beginning with `0x`".to_string(),
        });
    };
    let hex = hex.replace('_', "");
    if hex.len() != kind.hex_nibbles {
        return Err(ElaborateError::InvalidConstant {
            operation: op.to_string(),
            reason: format!("expected exactly {} hexadecimal nibbles", kind.hex_nibbles),
        });
    }
    if !hex.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(ElaborateError::InvalidConstant {
            operation: op.to_string(),
            reason: "literal contains a non-hexadecimal digit".to_string(),
        });
    }
    Ok(())
}

fn const_arrow(name: Operation, kind: ConstantKind) -> RawTheoryArrow {
    RawTheoryArrow {
        name,
        type_maps: (
            Hexpr::Frobenius {
                sources: Vec::new(),
                targets: Vec::new(),
            },
            Hexpr::Composition(vec![op(kind.type_name), op("val")]),
        ),
        definition: None,
    }
}

fn op(name: &str) -> Hexpr {
    Hexpr::Operation(name.parse().expect("generated operation should parse"))
}
