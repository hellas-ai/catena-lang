//! Closure conversion over graphs produced by `forget_closures`.
//!
//! The conversion is deliberately split into three stages: discover a delimited
//! control-flow region, turn that region into a definition, and replace the
//! original region with an explicit environment and function pointer.

use hexpr::Operation;
use metacat::theory::TheorySet;
use thiserror::Error;

use crate::{
    check::{CheckError, DefinitionTypes},
    pass::forget_closures::Region,
    report::TheoryTermMap,
};

/// Find regions by following closure domains to their codomains.
pub mod region;

/// Turn discovered regions into `closure.*` definitions and `name.closure.*` declarations.
pub mod definition;

/// Replace regions with explicit environments, function pointers, and context operations.
pub mod replace;

/// Complete output of closure conversion, including its debugging snapshots.
#[derive(Debug, Clone)]
pub struct Conversion {
    /// Regions discovered in the closure-forgotten input.
    pub regions: region::ClosureRegionMap,
    /// Theory after inserting the generated `closure.*` and `name.closure.*` arrows.
    pub definitions: TheorySet,
    /// Checked node labels for `definitions`, including generated closure bodies.
    pub definition_types: DefinitionTypes,
    /// Closure-forgotten definitions after replacing every region.
    pub replacements: TheoryTermMap,
    /// Theory containing the replaced definitions and generated context declarations.
    pub theory_set: TheorySet,
}

#[derive(Debug, Error)]
pub enum ConversionError {
    #[error(transparent)]
    FindRegions(#[from] region::FindRegionError),
    #[error(transparent)]
    DefineClosures(#[from] definition::DefineClosuresError),
    #[error(transparent)]
    CheckDefinitions(#[from] CheckError),
    #[error(transparent)]
    ReplaceClosures(#[from] replace::ReplaceClosuresError),
}

/// Closure-convert graphs produced by `forget_closures` as one compiler pass.
///
/// Region discovery, generated-arrow construction, validation, and replacement
/// remain separate implementation modules, but callers receive one coherent
/// result which preserves every useful intermediate representation.
pub fn run(
    theory_set: &TheorySet,
    forgotten: &TheoryTermMap<Region<Operation>>,
) -> Result<Conversion, ConversionError> {
    let regions = region::run(forgotten)?;
    let definitions = definition::run(theory_set, forgotten, &regions)?;
    let definition_types = crate::check::check(&definitions)?;
    let replacement = replace::run(&definitions, forgotten, &regions)?;

    Ok(Conversion {
        regions,
        definitions,
        definition_types,
        replacements: replacement.terms,
        theory_set: replacement.theory_set,
    })
}
