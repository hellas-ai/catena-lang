use catena_gpu::{check::check, elaborate::elaborate};
use metacat::theory::{RawTheorySet, TheorySet};

const LINEAR_SCHEDULING: &str = include_str!("cases/linear_scheduling.hex");

#[test]
fn perfect_tiling_linear_scheduling_typechecks() -> anyhow::Result<()> {
    let parameters = LaunchParameters {
        buffer_size: 8,
        grid_x: 2,
        block_x: 4,
    };
    assert_eq!(parameters.launched_threads(), parameters.buffer_size);
    check_launch_case(parameters)
}

#[test]
fn predicated_tiling_linear_scheduling_typechecks() -> anyhow::Result<()> {
    let parameters = LaunchParameters {
        buffer_size: 7,
        grid_x: 2,
        block_x: 4,
    };
    assert!(parameters.launched_threads() > parameters.buffer_size);
    check_launch_case(parameters)
}

struct LaunchParameters {
    buffer_size: u64,
    grid_x: u64,
    block_x: u64,
}

impl LaunchParameters {
    fn launched_threads(&self) -> u64 {
        self.grid_x
            .checked_mul(self.block_x)
            .expect("test launch dimensions should not overflow")
    }
}

fn check_launch_case(parameters: LaunchParameters) -> anyhow::Result<()> {
    assert!(parameters.buffer_size <= parameters.launched_threads());
    check_with_sources([LINEAR_SCHEDULING])
}

fn check_with_sources(sources: impl IntoIterator<Item = &'static str>) -> anyhow::Result<()> {
    let raw = RawTheorySet::from_texts(catena_gpu::stdlib::sources().chain(sources))?;
    let elaborated = elaborate(raw)?;
    let theories = TheorySet::from_raw(elaborated)?;
    check(&theories)?;
    Ok(())
}
