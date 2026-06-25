pub mod forget_closures;
pub mod record_boundary_sizes;
pub mod unpack_products;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum PassError {
    #[error("pass `{pass}` produced an unquotientable term for `{theory}.{definition}`")]
    Quotient {
        pass: &'static str,
        theory: String,
        definition: String,
    },
}
