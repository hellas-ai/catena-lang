use std::path::PathBuf;

use metacat::theory::{RawTheorySet, TheorySet};
use thiserror::Error;

use crate::{
    check::{CheckError, check as check_elaborated_theory},
    compile::{
        CompileConfig, CompileGraph, CompileGraphError, analysis,
        cfg::{CfgOptions, render_program_cfg},
        check_render, compile_graph,
        cuda::CudaOptions,
        cuda::{CudaAbiError, render_cuda_source},
        graph_render,
        normalize::{NormalizeGraphError, normalize_graph},
        program::{
            ProgramCompileError, ProgramCompileOptions, compile_program_from_graph,
            compile_program_from_graph_with_options,
        },
        proof::{ProofCertificateError, ProofCertificates},
        structured::{StructuredCompileError, compile_structured_program},
    },
    elaborate::{ElaborateError, elaborate},
};

// High-level compiler pipeline:
//
// 1. Parse + elaborate source files into the data/control theory set used by
//    later stages. Elaboration is responsible for exposing cross-theory arrows.
// 2. Typecheck the elaborated theory. After this point the compiler works with
//    checked theory objects rather than raw syntax.
// 3. Build a CompileGraph for the requested entry.
//    - Interpret the checked definition as a hypergraph of operations and
//      wires.
//    - Annotate graph wires with checked type information and keep source
//      variable names for diagnostics/backend names.
//    - Preserve definition and cross-theory operation boundaries as child
//      regions so structured lowering can decide how to lower them.
//    - inline local graph definitions if in the allowed list
// 4. Normalize the graph for backend consumption. Normalization removes
//    typechecking-only structure, erases non-runtime operations, forgets
//    loopback markers, and quotients unified wires so later stages see the
//    executable runtime graph.
// 5. Verify any requested proof certificates against the normalized graph.
//    Proofs are checked after normalization because backend requirements are
//    stated over the graph shape the backend will actually consume.
// 6. Optionally emit analysis/debug reports for the normalized graph.
// 7. Compile the graph into a Program/CFG representation. This is where data
//    dependency scheduling and control CFG construction happen.
// 8. Structure the Program into structured IR with statements, blocks, and
//    control flow. CUDA lowering consumes this structured IR plus ABI/proof
//    metadata to render target source.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Emit {
    Cuda,
    CompileGraph,
    Cfg,
    Elaborated,
    Checked,
    StructuredIr,
    Analysis,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputFormat {
    Svg,
    Text,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompileRequest {
    pub paths: Vec<PathBuf>,
    pub emit: Emit,
    pub theory: Option<String>,
    pub entry: Option<String>,
    pub format: Option<OutputFormat>,
    pub cuda_options: CudaOptions,
    pub cfg_options: CfgOptions,
    pub proof_check: bool,
    pub proof_paths: Vec<PathBuf>,
}

#[derive(Debug, Error)]
pub enum CompilePipelineError {
    #[error("failed to parse source: {0}")]
    Parse(#[from] metacat::theory::ast::ParseRawError),
    #[error("failed to elaborate source: {0}")]
    Elaborate(#[from] ElaborateError),
    #[error("failed to typecheck source: {0}")]
    Check(#[from] CheckError),
    #[error("failed to build compile graph: {0}")]
    CompileGraph(#[from] CompileGraphError),
    #[error("failed to render compile graph: {0}")]
    RenderGraph(#[from] std::io::Error),
    #[error(transparent)]
    Normalize(#[from] NormalizeGraphError),
    #[error(transparent)]
    Program(#[from] ProgramCompileError),
    #[error(transparent)]
    Structured(#[from] StructuredCompileError),
    #[error(transparent)]
    CudaAbi(#[from] CudaAbiError),
    #[error("{argument} is required when emitting {emit:?}")]
    MissingArgument { argument: &'static str, emit: Emit },
    #[error("--format {format:?} is not supported when emitting {emit:?}")]
    UnsupportedFormat { emit: Emit, format: OutputFormat },
    #[error(transparent)]
    Proof(#[from] ProofCertificateError),
}

pub fn compile(request: CompileRequest) -> Result<Vec<u8>, CompilePipelineError> {
    let mut pipeline = CompilePipeline::new(request);
    pipeline.emit()
}

pub struct CompilePipeline {
    request: CompileRequest,
    elaborated: Option<RawTheorySet>,
    checked_elaborated_theory: Option<TheorySet>,
}

impl CompilePipeline {
    pub fn new(request: CompileRequest) -> Self {
        Self {
            request,
            elaborated: None,
            checked_elaborated_theory: None,
        }
    }

    pub fn emit(&mut self) -> Result<Vec<u8>, CompilePipelineError> {
        match self.request.emit {
            Emit::Elaborated => {
                self.require_format(OutputFormat::Text)?;
                Ok(self.elaborated()?.to_hexpr_text().into_bytes())
            }
            Emit::Checked => {
                self.require_format(OutputFormat::Text)?;
                Ok(check_render::summary(self.checked_elaborated_theory()?).into_bytes())
            }
            Emit::CompileGraph => {
                self.require_format(OutputFormat::Svg)?;
                self.proof_certificates()?;
                let compile_graph_request = self.compile_graph_request()?;
                let checked_elaborated_theory = self.checked_elaborated_theory()?;
                let graph = Self::compile_graph(checked_elaborated_theory, compile_graph_request)?;
                Ok(graph_render::nested_svg(&graph)?)
            }
            Emit::Analysis => {
                self.require_format(OutputFormat::Svg)?;
                let compile_graph_request = self.compile_graph_request()?;
                let checked_elaborated_theory = self.checked_elaborated_theory()?;
                let compile_graph =
                    Self::compile_graph(checked_elaborated_theory, compile_graph_request)?;
                let graph = normalize_graph(&compile_graph)?;
                Ok(analysis::render_analysis(&graph)?)
            }
            Emit::Cfg | Emit::Cuda | Emit::StructuredIr => {
                self.require_format(OutputFormat::Text)?;
                let proof_certificates = self.proof_certificates()?;
                let emit = self.request.emit;
                let cuda_options = self.request.cuda_options.clone();
                let compile_graph_request = self.compile_graph_request()?;
                let checked_elaborated_theory = self.checked_elaborated_theory()?;
                let compile_graph =
                    Self::compile_graph(checked_elaborated_theory, compile_graph_request)?;
                let graph = normalize_graph(&compile_graph)?;
                let proof_evidence = proof_certificates
                    .as_ref()
                    .map(|certificates| certificates.verify_graph_properties(&graph))
                    .transpose()?;
                if emit == Emit::Cfg {
                    let program = compile_program_from_graph_with_options(
                        &graph,
                        ProgramCompileOptions {
                            cfg: self.request.cfg_options,
                        },
                    )?;
                    return Ok(render_program_cfg(&program).into_bytes());
                }
                let program = compile_program_from_graph(&graph)?;
                let structured = compile_structured_program(&program)?;
                Ok(match emit {
                    Emit::Cuda => render_cuda_source(
                        checked_elaborated_theory,
                        &program,
                        &structured,
                        &cuda_options,
                        proof_evidence.as_ref(),
                    )?,
                    Emit::StructuredIr => structured.render_ir(),
                    _ => unreachable!("only structured-backed emits are handled here"),
                }
                .into_bytes())
            }
        }
    }

    pub fn elaborated(&mut self) -> Result<&RawTheorySet, CompilePipelineError> {
        if self.elaborated.is_none() {
            let raw = RawTheorySet::from_files(self.request.paths.clone())?;
            self.elaborated = Some(elaborate(raw)?);
        }
        Ok(self.elaborated.as_ref().expect("elaborated is initialized"))
    }

    pub fn checked_elaborated_theory(&mut self) -> Result<&TheorySet, CompilePipelineError> {
        if self.checked_elaborated_theory.is_none() {
            let elaborated = self.elaborated()?;
            self.checked_elaborated_theory = Some(check_elaborated_theory(elaborated)?);
        }
        Ok(self
            .checked_elaborated_theory
            .as_ref()
            .expect("checked elaborated theory is initialized"))
    }

    pub fn compile_graph_request(&self) -> Result<CompileGraphRequest, CompilePipelineError> {
        Ok(CompileGraphRequest {
            theory: self.required_input(PipelineInput::Theory)?,
            entry: self.required_input(PipelineInput::Entry)?,
        })
    }

    pub fn compile_graph(
        checked_elaborated_theory: &TheorySet,
        request: CompileGraphRequest,
    ) -> Result<CompileGraph, CompilePipelineError> {
        compile_graph_from_checked(checked_elaborated_theory, request)
    }

    fn required_input(&self, input: PipelineInput) -> Result<String, CompilePipelineError> {
        let value = match input {
            PipelineInput::Theory => self.request.theory.clone(),
            PipelineInput::Entry => self.request.entry.clone(),
        };
        value.ok_or(CompilePipelineError::MissingArgument {
            argument: input.name(),
            emit: self.request.emit,
        })
    }

    fn require_format(&self, expected: OutputFormat) -> Result<(), CompilePipelineError> {
        if let Some(format) = self.request.format
            && format != expected
        {
            return Err(CompilePipelineError::UnsupportedFormat {
                emit: self.request.emit,
                format,
            });
        }
        Ok(())
    }

    fn proof_certificates(&self) -> Result<Option<ProofCertificates>, CompilePipelineError> {
        if !self.request.proof_check {
            return Ok(None);
        }

        Ok(Some(ProofCertificates::from_files(
            &self.request.paths,
            &self.request.proof_paths,
        )?))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompileGraphRequest {
    pub theory: String,
    pub entry: String,
}

fn compile_graph_from_checked(
    checked_elaborated_theory: &TheorySet,
    request: CompileGraphRequest,
) -> Result<CompileGraph, CompilePipelineError> {
    Ok(compile_graph(
        checked_elaborated_theory,
        &CompileConfig::data_control(),
        &request.theory,
        &request.entry,
    )?)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PipelineInput {
    Theory,
    Entry,
}

impl PipelineInput {
    fn name(self) -> &'static str {
        match self {
            PipelineInput::Theory => "theory",
            PipelineInput::Entry => "entry",
        }
    }
}
