//! Bit-exact BF16 WMMA simulation and the shared native/simulated wrapper.
//! Finite steps use bounded 32-bit arithmetic and explicit device RNE conversion.

pub(super) fn render() -> String {
    [include_str!("arithmetic.h"), include_str!("fragments.h")].concat()
}

#[cfg(test)]
mod tests;
