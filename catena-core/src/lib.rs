//! Minimal scaffold for `catena-core`.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Status {
    pub name: &'static str,
    pub implemented: bool,
}

pub const STATUS: Status = Status {
    name: "catena-core",
    implemented: false,
};
