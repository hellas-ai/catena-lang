mod elaboration;
#[cfg(feature = "svg-reports")]
mod svg;

use std::{fs, io, path::Path};

use hexpr::Operation;
use metacat::{
    theory::{RawTheorySet, TheoryId, TheorySet},
    tree::Tree,
};
use std::collections::BTreeMap;

use crate::check::{AnnotatedTerm, PartialDefinitionTypes};
use crate::closure::Conversion;
use crate::compile::{ProgressEvent, StageTiming, timed};
use crate::pass::{
    forget_closures::ClosureForgotten, record_boundary_sizes::OperationWithBoundarySizes,
};

/// Generic storage for per-theory, per-definition graph results produced by compiler passes.
pub type TheoryTermMap<A = Operation> = BTreeMap<TheoryId, BTreeMap<Operation, AnnotatedTerm<A>>>;

#[derive(Clone, Copy, Debug)]
pub struct ReportOptions {
    #[cfg(feature = "svg-reports")]
    pub generate_svgs: bool,
}

impl Default for ReportOptions {
    fn default() -> Self {
        Self {
            #[cfg(feature = "svg-reports")]
            generate_svgs: true,
        }
    }
}

#[derive(Debug)]
pub struct CompileReport {
    pub timings: Vec<StageTiming>,
    pub raw_theories: RawTheorySet,
    pub elaborated: Option<RawTheorySet>,
    pub theory_set: Option<TheorySet>,
    pub definition_types: Option<BTreeMap<TheoryId, BTreeMap<Operation, Vec<Tree<(), Operation>>>>>,
    pub partial_definition_types: Option<PartialDefinitionTypes>,
    pub forgotten_closures: Option<TheoryTermMap<ClosureForgotten<Operation>>>,
    pub closure_conversion: Option<Conversion>,
    pub boundary_sizes: Option<TheoryTermMap<OperationWithBoundarySizes<Operation>>>,
    pub unpacked_products: Option<TheoryTermMap<OperationWithBoundarySizes<Operation>>>,
}

impl CompileReport {
    pub fn new(raw_theories: RawTheorySet) -> Self {
        Self {
            timings: Vec::new(),
            raw_theories,
            elaborated: None,
            theory_set: None,
            definition_types: None,
            partial_definition_types: None,
            forgotten_closures: None,
            closure_conversion: None,
            boundary_sizes: None,
            unpacked_products: None,
        }
    }
}

impl CompileReport {
    pub fn dump_graphs_to_dir(&self, dir: impl AsRef<Path>) -> io::Result<()> {
        self.dump_graphs_to_dir_with_options(dir, ReportOptions::default())
    }

    pub fn dump_graphs_to_dir_with_options(
        &self,
        dir: impl AsRef<Path>,
        options: ReportOptions,
    ) -> io::Result<()> {
        self.dump_graphs_to_dir_with_progress(dir, options, &mut |_| {})
    }

    /// Write compiler diagnostics, timing report generation separately from compilation.
    #[cfg_attr(not(feature = "svg-reports"), allow(unused_variables))]
    pub fn dump_graphs_to_dir_with_progress(
        &self,
        dir: impl AsRef<Path>,
        options: ReportOptions,
        progress: &mut dyn FnMut(ProgressEvent),
    ) -> io::Result<()> {
        let dir = dir.as_ref();
        fs::create_dir_all(dir)?;
        let mut timings = self.timings.clone();
        let write_timings = |timings: &[StageTiming]| -> io::Result<()> {
            let json = serde_json::to_vec_pretty(timings).map_err(io::Error::other)?;
            fs::write(dir.join("timings.json"), json)
        };
        // Preserve compilation timings even if writing graphs subsequently fails.
        write_timings(&timings)?;
        let result = timed(&mut timings, "Report generation", progress, |update| {
            update("Writing raw theories and elaboration report");
            fs::write(
                dir.join("raw_theories.hex"),
                self.raw_theories.to_hexpr_text(),
            )?;
            elaboration::dump_elaboration(self, dir)?;
            #[cfg(feature = "svg-reports")]
            if options.generate_svgs {
                update("Rendering SVG graphs");
                svg::dump_svgs(self, &dir.join("svgs"))?;
            }
            Ok(())
        });
        let timings_result = write_timings(&timings);
        result.and(timings_result)
    }
}
