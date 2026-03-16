// C should be the *Runtime* type
use catena::backend::c::{Runtime, Value};

fn main() -> anyhow::Result<()> {
    // Create a C runtime. This lets us load and run catena code safely inside
    // a 'sandbox' child process.
    // Compiles, typechecks, and lowers the contents of 'stdlib.hex' to init the runtime.
    let runtime = Runtime::new(&std::fs::read_to_string("stdlib.hex")?)?;

    // Look up a function by name and execute it.
    // Uses const generics to return fixed size array, returning error if the
    // constant size is different to the dynamically-inspected number of return
    // values of 'materialize-range'.
    let [result] = runtime.exec("sum-range-f32", [Value::Extent(10)])?;
    println!("result: {result:?}");

    Ok(())
}
