//! Values carried while lowering closure-converted graphs.
use hexpr::Operation;
use metacat::tree::Tree;
pub type Obj = Tree<(), Operation>;
#[derive(Clone, Debug)]
pub struct Value {
    pub ty: Obj,
    pub repr: Repr,
}
#[derive(Clone, Debug)]
pub enum Repr {
    Erased,
    Runtime(String),
    Product(Vec<Value>),
    /// A function reference produced by closure conversion; captures are separate operands.
    Function(Operation),
}
impl Value {
    pub fn erased(ty: Obj) -> Self {
        Self {
            ty,
            repr: Repr::Erased,
        }
    }
    pub fn runtime_values(&self, out: &mut Vec<Value>) {
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
    pub fn flatten(self, out: &mut Vec<Value>) {
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
