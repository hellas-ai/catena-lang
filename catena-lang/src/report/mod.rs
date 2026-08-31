mod elaboration;
mod gpu;
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
use crate::codegen::GpuModuleMap;
use crate::pass::{
    forget_closures::ClosureForgotten, record_boundary_sizes::OperationWithBoundarySizes,
};

/// Generic storage for per-theory, per-definition graph results produced by compiler passes.
pub type TheoryTermMap<A = Operation> = BTreeMap<TheoryId, BTreeMap<Operation, AnnotatedTerm<A>>>;

#[derive(Clone, Copy, Debug)]
pub struct ReportOptions {
    pub generate_svgs: bool,
}

impl Default for ReportOptions {
    fn default() -> Self {
        Self {
            generate_svgs: true,
        }
    }
}

#[derive(Debug)]
pub struct CompileReport {
    pub raw_theories: RawTheorySet,
    pub elaborated: Option<RawTheorySet>,
    pub theory_set: Option<TheorySet>,
    pub definition_types: Option<BTreeMap<TheoryId, BTreeMap<Operation, Vec<Tree<(), Operation>>>>>,
    pub partial_definition_types: Option<PartialDefinitionTypes>,
    pub forgotten_closures: Option<TheoryTermMap<ClosureForgotten<Operation>>>,
    pub closure_conversion: Option<Conversion>,
    pub boundary_sizes: Option<TheoryTermMap<OperationWithBoundarySizes<Operation>>>,
    pub unpacked_products: Option<TheoryTermMap<OperationWithBoundarySizes<Operation>>>,
    pub gpu_modules: Option<GpuModuleMap>,
}

impl CompileReport {
    pub fn new(raw_theories: RawTheorySet) -> Self {
        Self {
            raw_theories,
            elaborated: None,
            theory_set: None,
            definition_types: None,
            partial_definition_types: None,
            forgotten_closures: None,
            closure_conversion: None,
            boundary_sizes: None,
            unpacked_products: None,
            gpu_modules: None,
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
        require_supported_options(options)?;
        let dir = dir.as_ref();
        fs::create_dir_all(dir)?;
        fs::write(
            dir.join("raw_theories.hex"),
            self.raw_theories.to_hexpr_text(),
        )?;
        elaboration::dump_elaboration(self, dir)?;
        if options.generate_svgs {
            #[cfg(feature = "svg-reports")]
            svg::dump_svgs(self, &dir.join("svgs"))?;
        }
        Ok(())
    }

    pub fn dump_to_dir(&self, dir: impl AsRef<Path>) -> io::Result<()> {
        self.dump_to_dir_with_options(dir, ReportOptions::default())
    }

    pub fn dump_to_dir_with_options(
        &self,
        dir: impl AsRef<Path>,
        options: ReportOptions,
    ) -> io::Result<()> {
        let dir = dir.as_ref();
        self.dump_graphs_to_dir_with_options(dir, options)?;
        gpu::dump_gpu(self, &dir.join("gpu"))?;
        Ok(())
    }
}

fn require_supported_options(options: ReportOptions) -> io::Result<()> {
    if options.generate_svgs && !cfg!(feature = "svg-reports") {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "SVG reports were requested, but catena-lang was built without the `svg-reports` feature; disable SVG generation or enable that feature",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_options_match_compiled_svg_support() {
        let requested = require_supported_options(ReportOptions {
            generate_svgs: true,
        });
        if cfg!(feature = "svg-reports") {
            requested.expect("default build must support requested SVG reports");
        } else {
            let error = requested.expect_err("SVG request must fail without renderer support");
            assert_eq!(error.kind(), io::ErrorKind::Unsupported);
            assert!(error.to_string().contains("`svg-reports` feature"));
        }
        require_supported_options(ReportOptions {
            generate_svgs: false,
        })
        .expect("non-SVG reports never require the renderer feature");
    }
}
