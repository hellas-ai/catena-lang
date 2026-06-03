use std::path::PathBuf;

use catena::compile::{
    CompilePipeline, CompileRequest, CudaOptions, Emit, OutputFormat, cfg::CfgOptions, compile,
};
use clap::{Parser, Subcommand, ValueEnum};

#[derive(Parser)]
#[command(name = "catena", version = env!("CARGO_PKG_VERSION"))]
#[command(about = "catena compiler")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Elaborate a multi-theory hex file by interleaving control/data theories
    Elaborate {
        #[arg(required = true)]
        paths: Vec<PathBuf>,
    },

    /// Elaborate and typecheck a multi-theory hex file
    Check {
        #[arg(required = true)]
        paths: Vec<PathBuf>,

        #[arg(long)]
        verbose: bool,
    },

    /// Run the Catena compile pipeline
    Compile {
        #[arg(required = true)]
        paths: Vec<PathBuf>,

        #[arg(long)]
        emit: EmitArg,

        #[arg(long)]
        theory: Option<String>,

        #[arg(long)]
        entry: Option<String>,

        #[arg(long, value_enum)]
        format: Option<OutputFormatArg>,

        /// Write output to a file instead of stdout
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Provide a compile-time CUDA size value, e.g. --cuda-static tile_rows=16.
        #[arg(long = "cuda-static", value_parser = parse_cuda_static)]
        cuda_static: Vec<(String, u64)>,

        /// Compile without requiring a matching .proof.hex certificate.
        #[arg(long = "no-proof")]
        no_proof: bool,

        /// Proof certificate file(s) to check before compiling.
        #[arg(long = "proof", num_args = 1..)]
        proof: Vec<PathBuf>,

        /// Keep monoidal-structure operations in CFG output for debugging.
        #[arg(long = "cfg-keep-monoidal-operations")]
        cfg_keep_monoidal_operations: bool,

        /// Keep control-flow-only operations in CFG output for debugging.
        #[arg(long = "cfg-keep-control-flow-operations")]
        cfg_keep_control_flow_operations: bool,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum EmitArg {
    Cuda,
    CompileGraph,
    Cfg,
    Elaborated,
    Checked,
    StructuredIr,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum OutputFormatArg {
    Svg,
    Text,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Elaborate { paths } => elaborate_command(paths),
        Command::Check { paths, verbose } => check_command(paths, verbose),
        Command::Compile {
            paths,
            emit,
            theory,
            entry,
            format,
            output,
            cuda_static,
            no_proof,
            proof,
            cfg_keep_monoidal_operations,
            cfg_keep_control_flow_operations,
        } => compile_command(
            paths,
            emit,
            theory,
            entry,
            format,
            output,
            cuda_static,
            no_proof,
            proof,
            cfg_keep_monoidal_operations,
            cfg_keep_control_flow_operations,
        ),
    }
}

fn check_command(paths: Vec<PathBuf>, verbose: bool) -> anyhow::Result<()> {
    let mut pipeline = CompilePipeline::new(CompileRequest {
        paths: paths.clone(),
        emit: Emit::Checked,
        theory: None,
        entry: None,
        format: None,
        cuda_options: CudaOptions::default(),
        cfg_options: CfgOptions::default(),
        proof_check: false,
        proof_paths: Vec::new(),
    });
    let theory_set = pipeline.checked_elaborated_theory()?;

    println!("OK: check passed");
    if paths.len() == 1 {
        println!("  file: {}", paths[0].display());
    } else {
        println!("  files: {}", paths.len());
    }
    if verbose {
        for (id, theory) in &theory_set.theories {
            if let metacat::theory::Theory::Theory { arrows, .. } = theory {
                let definitions = arrows
                    .values()
                    .filter(|arrow| arrow.definition.is_some())
                    .count();
                println!("  {}: {} definitions", id, definitions);
            }
        }
    }
    Ok(())
}

fn elaborate_command(paths: Vec<PathBuf>) -> anyhow::Result<()> {
    let generated = compile(CompileRequest {
        paths,
        emit: Emit::Elaborated,
        theory: None,
        entry: None,
        format: None,
        cuda_options: CudaOptions::default(),
        cfg_options: CfgOptions::default(),
        proof_check: false,
        proof_paths: Vec::new(),
    })?;
    write_output(None, &generated)
}

fn compile_command(
    paths: Vec<PathBuf>,
    emit: EmitArg,
    theory: Option<String>,
    entry: Option<String>,
    format: Option<OutputFormatArg>,
    output: Option<PathBuf>,
    cuda_static: Vec<(String, u64)>,
    no_proof: bool,
    proof: Vec<PathBuf>,
    cfg_keep_monoidal_operations: bool,
    cfg_keep_control_flow_operations: bool,
) -> anyhow::Result<()> {
    let mut static_values = std::collections::HashMap::new();
    for (name, value) in cuda_static {
        if static_values.insert(name.clone(), value).is_some() {
            anyhow::bail!("duplicate --cuda-static value for `{name}`");
        }
    }

    let generated = compile(CompileRequest {
        paths,
        emit: emit.into(),
        theory,
        entry,
        format: format.map(Into::into),
        cuda_options: CudaOptions { static_values },
        cfg_options: CfgOptions {
            keep_monoidal_operations: cfg_keep_monoidal_operations,
            keep_control_flow_operations: cfg_keep_control_flow_operations,
        },
        proof_check: !no_proof,
        proof_paths: proof,
    })?;

    write_output(output, &generated)
}

fn parse_cuda_static(value: &str) -> Result<(String, u64), String> {
    let (name, raw_value) = value
        .split_once('=')
        .ok_or_else(|| "expected NAME=VALUE".to_string())?;
    let name = name.trim();
    if name.is_empty() {
        return Err("static CUDA value name cannot be empty".to_string());
    }
    let parsed = raw_value
        .trim()
        .parse::<u64>()
        .map_err(|_| "static CUDA value must be an unsigned integer".to_string())?;
    if parsed == 0 {
        return Err("static CUDA value must be greater than zero".to_string());
    }
    Ok((name.to_string(), parsed))
}

fn write_output(output: Option<PathBuf>, generated: &[u8]) -> anyhow::Result<()> {
    match output {
        Some(output) => std::fs::write(output, generated)?,
        None => {
            use std::io::Write;
            std::io::stdout().write_all(generated)?;
        }
    }

    Ok(())
}

impl From<EmitArg> for Emit {
    fn from(value: EmitArg) -> Self {
        match value {
            EmitArg::Cuda => Emit::Cuda,
            EmitArg::CompileGraph => Emit::CompileGraph,
            EmitArg::Cfg => Emit::Cfg,
            EmitArg::Elaborated => Emit::Elaborated,
            EmitArg::Checked => Emit::Checked,
            EmitArg::StructuredIr => Emit::StructuredIr,
        }
    }
}

impl From<OutputFormatArg> for OutputFormat {
    fn from(value: OutputFormatArg) -> Self {
        match value {
            OutputFormatArg::Svg => OutputFormat::Svg,
            OutputFormatArg::Text => OutputFormat::Text,
        }
    }
}
