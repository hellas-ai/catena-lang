use hexpr::Operation;
use metacat::tree::Tree;

pub type Obj = Tree<(), Operation>;
pub type Arr = Operation;

pub fn is_value(o: &Obj) -> bool {
    // true iff the root of the tree is "value" (the type of runtime values)
    match o {
        Tree::Node(key, _, _) => key.to_string() == "value",
        _ => false,
    }
}
