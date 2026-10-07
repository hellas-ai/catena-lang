//! Closure-free GPU instructions and generated function interfaces.
use std::collections::BTreeMap;

use super::lower_types::CType;
use crate::runtime::GeneratedFunction;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Variable {
    pub(super) name: String,
    pub(super) ty: CType,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Instruction {
    Let(Variable, String),
    Assign(String, String),
    /// Block-wide synchronization, including visibility of shared-memory writes.
    Sync,
    Assert {
        condition: String,
    },
    /// Buffer types distinguish global and shared address spaces.
    Load {
        result: Variable,
        buffer: Variable,
        index: Variable,
    },
    Store {
        buffer: Variable,
        index: Variable,
        value: Variable,
    },
    If {
        condition: String,
        yes: Vec<Instruction>,
        no: Vec<Instruction>,
    },
    Free {
        buffer: Variable,
    },
    For {
        index: String,
        index_type: CType,
        end: String,
        body: Vec<Instruction>,
    },
    Launch {
        kernel: String,
        grid: String,
        shared_bytes: String,
        arguments: Vec<String>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Function {
    pub(super) symbol: String,
    pub(super) inputs: Vec<Variable>,
    pub(super) outputs: Vec<Variable>,
    pub(super) body: Vec<Instruction>,
}

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub(super) struct Modules {
    pub(super) functions: BTreeMap<String, Function>,
    pub(super) kernels: BTreeMap<String, Function>,
    pub(super) exports: Vec<GeneratedFunction>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Place {
    Host,
    Device,
}
