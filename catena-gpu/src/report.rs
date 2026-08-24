use std::collections::BTreeMap;

use hexpr::Operation;
use metacat::theory::{RawTheorySet, TheoryId, TheorySet};

use crate::{
    check::{AnnotatedTerm, DefinitionTypes},
    codegen::GpuModuleMap,
};

pub type TheoryTermMap<A = Operation> = BTreeMap<TheoryId, BTreeMap<Operation, AnnotatedTerm<A>>>;

#[derive(Debug)]
pub struct CompileReport {
    pub raw_theories: RawTheorySet,
    pub elaborated: Option<RawTheorySet>,
    pub theory_set: Option<TheorySet>,
    pub definition_types: Option<DefinitionTypes>,
    pub gpu_modules: Option<GpuModuleMap>,
}

impl CompileReport {
    pub fn new(raw_theories: RawTheorySet) -> Self {
        Self {
            raw_theories,
            elaborated: None,
            theory_set: None,
            definition_types: None,
            gpu_modules: None,
        }
    }
}
