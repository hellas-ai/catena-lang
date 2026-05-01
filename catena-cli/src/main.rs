use catena::lower::{lower, Pass};
use catena::shallow::shallow_graph;

use clap::{Parser, Subcommand, ValueEnum};
use std::collections::HashSet;
use std::path::PathBuf;

use catena::backend::c::codegen::codegen;
use catena::compile::{check_bundle, check_compile_bundle, ArrowType, CompileCheckReport};
use catena::lang::Obj;
use catena::structured::structured_from_shallow;
use metacat::{syntax::TheoryBundle, theory::OperationKey};
use open_hypergraphs::category::Arrow;
use open_hypergraphs::lax::{NodeId, OpenHypergraph};

#[derive(Parser)]
#[command(name = "catena", version=env!("CARGO_PKG_VERSION"))]
#[command(about = "catena compiler")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Check a hex file with metacat
    Check {
        #[arg()]
        path: PathBuf,
    },

    /// Run codegen for a given pass
    Codegen {
        #[arg()]
        path: PathBuf,
        #[arg()]
        definition: String,
    },

    /// Run the new Catena compile pipeline
    Compile {
        #[command(subcommand)]
        command: CompileCommand,
    },

    /// Run compiler passes up to the given pass and output SVG
    Lower {
        #[arg()]
        pass: PassArg,
        #[arg()]
        path: PathBuf,
        #[arg()]
        definition: String,
    },

    /// Check one definition and output its graph without inlining called arrows
    ShallowGraph {
        /// Emit the shallow hypergraph as SVG instead of structured code
        #[arg(long)]
        svg: bool,

        /// Emit structured IR instead of C
        #[arg(long)]
        ir: bool,

        /// Select shallow output format
        #[arg(long, value_enum, default_value_t = ShallowOutput::Cuda)]
        output: ShallowOutput,

        /// CUDA tile size used by CUDA output modes
        #[arg(long, default_value_t = 16)]
        tile: usize,

        #[arg()]
        path: PathBuf,
        #[arg()]
        definition: String,
    },
}

#[derive(Subcommand)]
enum CompileCommand {
    /// Check data/control theories after Catena lift passes
    Check {
        #[arg(long)]
        data: PathBuf,

        #[arg(long)]
        control: PathBuf,

        #[arg(long)]
        verbose: bool,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum PassArg {
    Check,
    Erase,
    ForgetBound,
    ExpandEta,
    DiscardNaturality,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ShallowOutput {
    Ir,
    Cuda,
    CudaWithLaunch,
    Svg,
}

impl From<PassArg> for Pass {
    fn from(value: PassArg) -> Self {
        match value {
            PassArg::Check => Pass::Check,
            PassArg::Erase => Pass::Erase,
            PassArg::ForgetBound => Pass::ForgetBound,
            PassArg::ExpandEta => Pass::ExpandEta,
            PassArg::DiscardNaturality => Pass::DiscardNaturality,
        }
    }
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Check { path } => check_command(path),
        Command::Codegen { path, definition } => {
            let bundle = TheoryBundle::from_file(path)?;
            let lowered = lower(&bundle, Pass::DiscardNaturality, &definition)?;
            println!("{}", codegen(lowered, "out"));
            Ok(())
        }
        Command::Compile { command } => compile_command(command),
        Command::Lower {
            path,
            pass,
            definition,
        } => lower_command(TheoryBundle::from_file(path)?, pass.into(), &definition),
        Command::ShallowGraph {
            svg,
            ir,
            output,
            tile,
            path,
            definition,
        } => shallow_graph_command(
            TheoryBundle::from_file(path)?,
            &definition,
            if svg {
                ShallowOutput::Svg
            } else if ir {
                ShallowOutput::Ir
            } else {
                output
            },
            tile,
        ),
    }
}

fn compile_command(command: CompileCommand) -> anyhow::Result<()> {
    match command {
        CompileCommand::Check {
            data,
            control,
            verbose,
        } => compile_check_command(data, control, verbose),
    }
}

fn compile_check_command(data: PathBuf, control: PathBuf, verbose: bool) -> anyhow::Result<()> {
    let data_display = data.display().to_string();
    let control_display = control.display().to_string();
    let data_bundle = TheoryBundle::from_file(data)?;
    let control_bundle = TheoryBundle::from_file(control)?;
    let report = check_compile_bundle(&data_bundle, &control_bundle)?;

    print_compile_check_report(&data_display, &control_display, &report, verbose);
    Ok(())
}

fn check_command(path: PathBuf) -> anyhow::Result<()> {
    let path_display = path.display().to_string();
    let bundle = TheoryBundle::from_file(path)?;
    let report = check_bundle(&bundle)?;

    println!(
        "OK: checked {path_display} ({} definitions)",
        report.definitions_checked
    );
    Ok(())
}

fn print_compile_check_report(
    data: &str,
    control: &str,
    report: &CompileCheckReport,
    verbose: bool,
) {
    println!("OK: compile check passed");
    println!(
        "  data: {data} ({} definitions)",
        report.data.definitions_checked
    );
    println!("  control: {control}");
    println!(
        "  control + lifted data: {} definitions",
        report.control_with_data.definitions_checked
    );
    println!(
        "  data + lifted control: {} definitions",
        report.data_with_control.definitions_checked
    );
    println!(
        "  lifted data -> control: {} arrows",
        report.data_to_control.len()
    );
    println!(
        "  lifted control -> data: {} arrows",
        report.control_to_data.len()
    );

    if verbose {
        print_lift_report("data -> control", &report.data_to_control);
        print_lift_report("control -> data", &report.control_to_data);
    }
}

fn print_lift_report(label: &str, operations: &[ArrowType]) {
    println!("  {label}:");
    for arrow_type in operations {
        println!("    {}", render_arrow_declaration(arrow_type));
    }
}

fn render_arrow_declaration(arrow_type: &ArrowType) -> String {
    format!(
        "(arrow {} : {} -> {})",
        arrow_type.name,
        render_object_map(&arrow_type.source),
        render_object_map(&arrow_type.target)
    )
}

fn render_object_map(map: &OpenHypergraph<(), OperationKey>) -> String {
    let mut map = map.clone();
    let _ = map.quotient();
    let map = &map;
    let vars = source_vars(map);
    match map.target().len() {
        0 => "[]".to_string(),
        1 => render_target(map, map.targets[0], &vars),
        _ => {
            let targets = map
                .targets
                .iter()
                .map(|node| render_node(map, *node, &vars, &mut HashSet::new()))
                .collect::<Vec<_>>();
            render_spider(&vars, &targets)
        }
    }
}

fn render_target(map: &OpenHypergraph<(), OperationKey>, node: NodeId, vars: &[String]) -> String {
    render_node(map, node, vars, &mut HashSet::new())
}

fn render_edge(
    map: &OpenHypergraph<(), OperationKey>,
    edge_index: usize,
    vars: &[String],
    seen: &mut HashSet<NodeId>,
) -> String {
    let op = &map.hypergraph.edges[edge_index];
    let adjacency = &map.hypergraph.adjacency[edge_index];
    if adjacency.sources.is_empty() {
        return op.to_string();
    }

    let inputs = adjacency
        .sources
        .iter()
        .map(|node| render_node(map, *node, vars, seen))
        .collect::<Vec<_>>();
    format!("({} {op})", render_spider(vars, &inputs))
}

fn render_node(
    map: &OpenHypergraph<(), OperationKey>,
    node: NodeId,
    vars: &[String],
    seen: &mut HashSet<NodeId>,
) -> String {
    if let Some(var) = map
        .sources
        .iter()
        .position(|source| *source == node)
        .map(|index| vars[index].clone())
    {
        return var;
    }

    if !seen.insert(node) {
        return format!("n{}", node.0);
    }

    let rendered = producer_edge(map, node)
        .map(|edge_index| render_edge(map, edge_index, vars, seen))
        .or_else(|| {
            object_edge_at_node(map, node)
                .map(|edge_index| map.hypergraph.edges[edge_index].to_string())
        })
        .unwrap_or_else(|| format!("n{}", node.0));
    seen.remove(&node);
    rendered
}

fn render_spider(sources: &[String], targets: &[String]) -> String {
    if sources == targets {
        format!("[{}]", sources.join(" "))
    } else if targets.is_empty() {
        format!("[{} .]", sources.join(" "))
    } else {
        format!("[{} . {}]", sources.join(" "), targets.join(" "))
    }
}

fn source_vars(map: &OpenHypergraph<(), OperationKey>) -> Vec<String> {
    (0..map.source().len())
        .map(|index| format!("x{index}"))
        .collect()
}

fn producer_edge(map: &OpenHypergraph<(), OperationKey>, node: NodeId) -> Option<usize> {
    map.hypergraph
        .adjacency
        .iter()
        .position(|edge| edge.targets.contains(&node))
}

fn object_edge_at_node(map: &OpenHypergraph<(), OperationKey>, node: NodeId) -> Option<usize> {
    map.hypergraph
        .adjacency
        .iter()
        .position(|edge| edge.sources.is_empty() && edge.targets.contains(&node))
}

fn lower_command(bundle: TheoryBundle, until: Pass, definition: &str) -> anyhow::Result<()> {
    let current = lower(&bundle, until, definition)?;
    print_svg(&bundle, current)
}

fn shallow_graph_command(
    bundle: TheoryBundle,
    definition: &str,
    output: ShallowOutput,
    tile: usize,
) -> anyhow::Result<()> {
    let current = shallow_graph(&bundle, definition)?;
    if matches!(output, ShallowOutput::Cuda | ShallowOutput::CudaWithLaunch) && tile == 0 {
        anyhow::bail!("--tile must be greater than zero");
    }
    match output {
        ShallowOutput::Svg => print_svg(&bundle, current),
        ShallowOutput::Ir => {
            let program = structured_from_shallow(&current, definition)?;
            print!("{}", program.render_ir());
            Ok(())
        }
        ShallowOutput::Cuda => {
            let program = structured_from_shallow(&current, definition)?;
            print!("{}", program.render_c_with_tile(tile)?);
            Ok(())
        }
        ShallowOutput::CudaWithLaunch => {
            let program = structured_from_shallow(&current, definition)?;
            print!("{}", program.render_cuda_with_launch_with_tile(tile)?);
            Ok(())
        }
    }
}

fn print_svg(
    bundle: &TheoryBundle,
    current: open_hypergraphs::lax::OpenHypergraph<Obj, OperationKey>,
) -> anyhow::Result<()> {
    // Pretty-print
    let coarity =
        |op: &OperationKey| -> usize { bundle.object_theory.type_maps(op).1.targets.len() };

    let labels: Vec<String> = current
        .hypergraph
        .nodes
        .iter()
        .map(|n| n.pretty(Some(&coarity)))
        .collect();

    use open_hypergraphs_dot::{svg::to_svg_with, Options};
    use std::io::Write;

    let opts = Options::default().display();
    std::io::stdout().write_all(&to_svg_with(
        &current
            .with_nodes(|_| labels)
            .expect("labels length mismatch"),
        &opts,
    )?)?;

    Ok(())
}
