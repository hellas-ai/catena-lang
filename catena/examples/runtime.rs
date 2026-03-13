// C should be the *Runtime* type
use catena::backend::c::{Runtime, Value};

fn main() -> anyhow::Result<()> {
    // Create a C runtime. This lets us load and run catena code safely inside
    // a 'sandbox' child process.
    let runtime = Runtime::new();

    // Compile, typecheck, and lower the specified def-arrow blocks in program_source
    let source = std::fs::read_to_string("stdlib.hex")?;
    runtime.compile(&source)?;

    // Create a runtime 'extent' value reference
    let n = runtime.value(Value::Extent(10));

    // Look up a function by name and execute it.
    // Uses const generics to return fixed size array, returning error if the
    // constant size is different to the dynamically-inspected number of return
    // values of 'materialize-range'.
    let [_result] = runtime.exec("materialize-range", [n])?;

    // Check we got a memref back (expected)
    /* later
    match result {
        // Cast
        Value::Memref(bytes) => cast_and_print_as_usize_array(bytes),
        _ => panic!("oh no!"),
    }
    */
    Ok(())
}
