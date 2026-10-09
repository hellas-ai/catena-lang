//! Values carried while lowering closure-converted graphs.
use hexpr::Operation;
use metacat::tree::Tree;
pub(super) type Obj = Tree<(), Operation>;
#[derive(Clone, Debug)]
pub(super) struct Value {
    pub(super) ty: Obj,
    pub(super) repr: Repr,
}
#[derive(Clone, Debug)]
pub(super) enum Repr {
    Erased,
    Runtime(String),
    Product(Vec<Value>),
    /// A function reference produced by closure conversion; captures are separate operands.
    Function(Operation),
}
impl Value {
    pub(super) fn erased(ty: Obj) -> Self {
        Self {
            ty,
            repr: Repr::Erased,
        }
    }
    pub(super) fn runtime_values(&self, out: &mut Vec<Value>) {
        match &self.repr {
            Repr::Runtime(_) => out.push(self.clone()),
            Repr::Product(values) => {
                for value in values {
                    value.runtime_values(out);
                }
            }
            _ => {}
        }
    }
    pub(super) fn flatten(self, out: &mut Vec<Value>) {
        match self.repr {
            Repr::Product(fields) => {
                for field in fields {
                    field.flatten(out);
                }
            }
            _ if super::lower_types::children(&self.ty, "1").is_some() => {}
            _ => out.push(self),
        }
    }
}

/// Compiler-side callback binding. Function references stay statically known;
/// consumers inline their bodies into branches, loops, or generated kernels.
pub(super) struct Callback<'a> {
    pub(super) function: &'a Value,
    pub(super) captures: Vec<Value>,
    pub(super) parameters: Vec<Obj>,
    pub(super) outputs: Vec<Obj>,
}

impl<'a> Callback<'a> {
    pub(super) fn new(
        function: &'a Value,
        environment: &Value,
    ) -> Result<Self, super::CodegenError> {
        if !matches!(&function.repr, Repr::Function(_)) {
            return Err(super::invalid(
                "callback",
                "expected a statically known function reference",
            ));
        }
        let (domain, codomain) = super::function_parts(&function.ty)?;
        let mut parameters = crate::pass::unpack_products::flatten_object(domain);
        let mut captures = Vec::new();
        environment.clone().flatten(&mut captures);
        if captures.len() > parameters.len() {
            return Err(super::invalid(
                "callback",
                "environment exceeds function domain",
            ));
        }
        let parameters = parameters.split_off(captures.len());
        Ok(Self {
            function,
            captures,
            parameters,
            outputs: crate::pass::unpack_products::flatten_object(codomain),
        })
    }
}
