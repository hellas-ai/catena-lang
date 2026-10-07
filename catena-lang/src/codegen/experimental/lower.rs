//! Walk converted definitions, binding generic types at direct call sites.
use super::{
    CodegenError, Lowerer, Template, invalid,
    ir::*,
    lower_types::{children, runtime},
    ops, pack, product,
    values::*,
};
use hexpr::Operation;
use metacat::ssa::ssa;
use metacat::tree::Tree;
use std::collections::{BTreeMap, BTreeSet};

fn bind(
    pattern: &Obj,
    actual: &Obj,
    bindings: &mut BTreeMap<usize, Obj>,
) -> Result<(), CodegenError> {
    match (pattern, actual) {
        (Tree::Leaf(i, ()), _) => {
            if let Some(previous) = bindings.get(i) {
                if previous != actual {
                    return Err(CodegenError::Type(format!(
                        "inconsistent call type arguments: {previous:?} versus {actual:?}"
                    )));
                }
            } else {
                bindings.insert(*i, actual.clone());
            }
        }
        (Tree::Node(a, ai, ac), Tree::Node(b, bi, bc))
            if a == b && ai == bi && ac.len() == bc.len() =>
        {
            for (p, v) in ac.iter().zip(bc) {
                bind(p, v, bindings)?;
            }
        }
        (Tree::Empty, Tree::Empty) => {}
        _ => {
            return Err(CodegenError::Type(format!(
                "call type mismatch: {pattern:?} versus {actual:?}"
            )));
        }
    }
    Ok(())
}

fn substitute(ty: &Obj, bindings: &BTreeMap<usize, Obj>) -> Obj {
    match ty {
        Tree::Leaf(i, ()) => bindings.get(i).cloned().unwrap_or_else(|| ty.clone()),
        Tree::Node(op, port, children) => Tree::Node(
            op.clone(),
            *port,
            children.iter().map(|v| substitute(v, bindings)).collect(),
        ),
        Tree::Empty => Tree::Empty,
    }
}

impl<'a> Lowerer<'a> {
    pub(super) fn new(templates: &'a BTreeMap<Operation, Template>) -> Self {
        fn type_bound(ty: &Obj) -> usize {
            match ty {
                Tree::Leaf(id, ()) => id + 1,
                Tree::Node(_, _, children) => children.iter().map(type_bound).max().unwrap_or(0),
                Tree::Empty => 0,
            }
        }
        let next_type = templates
            .values()
            .flat_map(|t| &t.term.hypergraph.nodes)
            .map(type_bound)
            .max()
            .unwrap_or(0);
        Self {
            templates,
            modules: Modules::default(),
            body: vec![],
            place: Place::Host,
            next: 0,
            next_type,
            stack: vec![],
            kernel_cache: vec![],
        }
    }
    pub(super) fn fresh(&mut self, prefix: &str) -> String {
        let n = self.next;
        self.next += 1;
        format!("{prefix}{n}")
    }
    pub(super) fn emit(
        &mut self,
        ty: &Obj,
        expression: impl Into<String>,
    ) -> Result<Value, CodegenError> {
        let ctype = runtime(ty)?.ok_or_else(|| {
            CodegenError::Type(format!("runtime expression has erased type {ty:?}"))
        })?;
        let name = self.fresh("v");
        self.body.push(Instruction::Let(
            Variable {
                name: name.clone(),
                ty: ctype,
            },
            expression.into(),
        ));
        Ok(Value {
            ty: ty.clone(),
            repr: Repr::Runtime(name),
        })
    }
    pub(super) fn input(
        &mut self,
        ty: &Obj,
        parameters: &mut Vec<Variable>,
    ) -> Result<Value, CodegenError> {
        if let Some(fields) = children(ty, "*") {
            return Ok(product(
                ty.clone(),
                fields
                    .iter()
                    .map(|t| self.input(t, parameters))
                    .collect::<Result<_, _>>()?,
            ));
        }
        if let Some(ctype) = runtime(ty)? {
            let name = self.fresh("arg");
            parameters.push(Variable {
                name: name.clone(),
                ty: ctype,
            });
            Ok(Value {
                ty: ty.clone(),
                repr: Repr::Runtime(name),
            })
        } else {
            self.erased(ty)
        }
    }
    pub(super) fn erased(&self, ty: &Obj) -> Result<Value, CodegenError> {
        if let Some(fields) = children(ty, "*") {
            return Ok(product(
                ty.clone(),
                fields
                    .iter()
                    .map(|t| self.erased(t))
                    .collect::<Result<_, _>>()?,
            ));
        }
        if runtime(ty)?.is_some() {
            return Err(CodegenError::Type(format!(
                "cannot erase runtime output {ty:?}"
            )));
        }
        Ok(Value::erased(ty.clone()))
    }
    pub(super) fn call(
        &mut self,
        op: &Operation,
        args: Vec<Value>,
        outputs: &[Obj],
    ) -> Result<Vec<Value>, CodegenError> {
        if let Some(template) = self.templates.get(op) {
            if self.stack.contains(op) || self.stack.len() > 128 {
                return Err(CodegenError::Recursive(op.to_string()));
            }
            let term = template.term.clone();
            if term.sources.len() != args.len() || term.targets.len() != outputs.len() {
                return Err(invalid(op.as_str(), "call arity mismatch"));
            }
            let mut bindings = BTreeMap::new();
            for (node, value) in term.sources.iter().zip(&args) {
                bind(&term.hypergraph.nodes[node.0], &value.ty, &mut bindings)?;
            }
            for (node, ty) in term.targets.iter().zip(outputs) {
                bind(&term.hypergraph.nodes[node.0], ty, &mut bindings)?;
            }
            // Freshen otherwise-local symbolic identities across calls.
            fn leaves(ty: &Obj, out: &mut BTreeSet<usize>) {
                match ty {
                    Tree::Leaf(i, ()) => {
                        out.insert(*i);
                    }
                    Tree::Node(_, _, args) => {
                        for a in args {
                            leaves(a, out);
                        }
                    }
                    _ => {}
                }
            }
            let mut ids = BTreeSet::new();
            for ty in &term.hypergraph.nodes {
                leaves(ty, &mut ids);
            }
            for i in ids {
                bindings.entry(i).or_insert_with(|| {
                    let id = self.next_type;
                    self.next_type += 1;
                    Tree::Leaf(id, ())
                });
            }
            let labels = term
                .hypergraph
                .nodes
                .iter()
                .map(|ty| substitute(ty, &bindings))
                .collect::<Vec<_>>();
            let mut values = BTreeMap::new();
            for (n, value) in term.sources.iter().zip(args) {
                values.insert(n.0, value);
            }
            self.stack.push(op.clone());
            for assignment in ssa(term.clone().to_strict())? {
                let inputs = assignment
                    .sources
                    .iter()
                    .map(|(n, _)| {
                        match values.get(&n.0) {
                            Some(value) => Ok(value.clone()),
                            // Context erasure can leave producer-free type witnesses.
                            None => self.erased(&labels[n.0]).map_err(|_| {
                                invalid(op.as_str(), format!("missing runtime SSA input {}", n.0))
                            }),
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let types = assignment
                    .targets
                    .iter()
                    .map(|(n, _)| labels[n.0].clone())
                    .collect::<Vec<_>>();
                let results = if self.templates.contains_key(&assignment.op.operation) {
                    self.call(&assignment.op.operation, inputs, &types)?
                } else {
                    // Preserve pre-flattening port groups for primitive callback environments.
                    let mut inputs = inputs.into_iter();
                    let groups = assignment
                        .op
                        .source_sizes
                        .iter()
                        .map(|n| pack(inputs.by_ref().take(*n).collect()))
                        .collect();
                    self.lower_operation(assignment.op.operation.as_str(), groups, &types)?
                };
                if results.len() != assignment.targets.len() {
                    return Err(invalid(
                        assignment.op.operation.as_str(),
                        "lowering produced wrong output count",
                    ));
                }
                for ((n, _), mut value) in assignment.targets.iter().zip(results) {
                    value.ty = labels[n.0].clone();
                    values.insert(n.0, value);
                }
            }
            self.stack.pop();
            term.targets
                .iter()
                .map(|n| {
                    match values.get(&n.0) {
                        Some(value) => Ok(value.clone()),
                        // Erased context projections may also be function results.
                        None => self
                            .erased(&labels[n.0])
                            .map_err(|_| invalid(op.as_str(), "missing runtime function result")),
                    }
                })
                .collect()
        } else {
            self.lower_operation(op.as_str(), args, outputs)
        }
    }
    pub(super) fn lower_operation(
        &mut self,
        op: &str,
        args: Vec<Value>,
        outputs: &[Obj],
    ) -> Result<Vec<Value>, CodegenError> {
        if let Some(name) = op.strip_prefix("name.") {
            let [ty] = outputs else {
                return Err(invalid(op, "function reference must have one output"));
            };
            super::function_parts(ty)?;
            return Ok(vec![Value {
                ty: ty.clone(),
                repr: Repr::Function(
                    name.parse()
                        .map_err(|_| invalid(op, "invalid function name"))?,
                ),
            }]);
        }
        if op == "eval" {
            let (function, arguments) = args
                .split_last()
                .ok_or_else(|| invalid(op, "missing function reference"))?;
            let mut flat = Vec::new();
            for argument in arguments {
                argument.clone().flatten(&mut flat);
            }
            return self.call_function(function, flat, outputs);
        }
        if op.starts_with("meta.") {
            return super::meta::lower(self, op, &args, outputs);
        }
        if op.starts_with("smolcat.") {
            return super::smolcat_compat::lower(self, op, &args, outputs);
        }
        match op {
            "core.if" | "core.if_guarded" | "core.fold.bounded" | "core.fold.trace"
            | "core.fold.values" => ops::control::lower(self, op, args, outputs),
            "unsafe.launch_linear" => ops::launch::linear(self, op, &args, outputs),
            "unsafe.launch" | "unsafe.launch_shared" => {
                ops::launch::lower(self, op, &args, outputs)
            }
            ":.forget" | ":.ty" | ":.param" => ops::structure::lower(self, op, args, outputs),
            _ => ops::lower(self, op, args, outputs),
        }
    }
    pub(super) fn call_function(
        &mut self,
        function: &Value,
        args: Vec<Value>,
        outputs: &[Obj],
    ) -> Result<Vec<Value>, CodegenError> {
        let Repr::Function(target) = &function.repr else {
            return Err(invalid(
                "eval",
                "expected a statically known function reference",
            ));
        };
        self.call(target, args, outputs)
    }
    /// Inline a converted callback with its explicit environment and runtime operands.
    pub(super) fn callback(
        &mut self,
        function: &Value,
        environment: &Value,
        inputs: Vec<Value>,
    ) -> Result<Vec<Value>, CodegenError> {
        let (domain, codomain) = super::function_parts(&function.ty)?;
        let types = crate::pass::unpack_products::flatten_object(domain);
        let outputs = crate::pass::unpack_products::flatten_object(codomain);
        let mut args = Vec::new();
        environment.clone().flatten(&mut args);
        let mut data = inputs.into_iter();
        let remaining = types
            .get(args.len()..)
            .ok_or_else(|| invalid("callback", "environment exceeds function domain"))?;
        for ty in remaining {
            args.push(self.argument(ty, &mut data)?);
        }
        if data.next().is_some() {
            return Err(invalid("callback", "too many runtime operands"));
        }
        self.call_function(function, args, &outputs)
    }
    /// Rebuild a callback argument, filling only its erased fields implicitly.
    pub(super) fn argument(
        &self,
        ty: &Obj,
        values: &mut impl Iterator<Item = Value>,
    ) -> Result<Value, CodegenError> {
        if let Some(fields) = children(ty, "*") {
            return Ok(product(
                ty.clone(),
                fields
                    .iter()
                    .map(|t| self.argument(t, values))
                    .collect::<Result<_, _>>()?,
            ));
        }
        if let Some(expected) = runtime(ty)? {
            let mut value = values
                .next()
                .ok_or_else(|| CodegenError::Type("not enough callback arguments".into()))?;
            if runtime(&value.ty)? != Some(expected) {
                return Err(CodegenError::Type(
                    "callback runtime argument type mismatch".into(),
                ));
            }
            value.ty = ty.clone();
            Ok(value)
        } else {
            self.erased(ty)
        }
    }
}
