use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "catena", version = env!("CARGO_PKG_VERSION"))]
#[command(about = "catena development CLI")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Report the current catena-core implementation status.
    Status,
}

fn main() -> anyhow::Result<()> {
    match Cli::parse().command {
        Command::Status => {
            let status = catena_core::STATUS;
            println!("{} scaffolded: {}", status.name, !status.implemented);
            Ok(())
        }
    }
}
