use dylibso_observe_sdk::adapter::{
    otel_formatter::Value, otelstdout::OtelStdoutAdapter, AdapterMetadata::OpenTelemetry, Attribute,
};
use wasmtime::Val;

#[tokio::main]
pub async fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let data = std::fs::read(&args[0])?;
    let left: i32 = args[1].parse()?;
    let right: i32 = args[2].parse()?;
    let config = wasmtime::Config::new();

    // Create instance
    let engine = wasmtime::Engine::new(&config)?;
    let module = wasmtime::Module::new(&engine, &data)?;

    let adapter = OtelStdoutAdapter::create();

    // Setup WASI
    let wasi_ctx = wasmtime_wasi::WasiCtxBuilder::new()
        .inherit_env()?
        .inherit_stdio()
        .args(&args.clone())?
        .build();

    let mut store = wasmtime::Store::new(&engine, wasi_ctx);
    let mut linker = wasmtime::Linker::new(&engine);
    wasmtime_wasi::add_to_linker(&mut linker, |wasi| wasi)?;

    // Provide the observability functions to the `Linker` to be made available
    // to the instrumented guest code. These are safe to add and are a no-op
    // if guest code is uninstrumented.
    let trace_ctx = adapter.start(&mut linker, &data, Default::default())?;

    let instance = linker.instantiate(&mut store, &module)?;

    // get the function and run it, the events pop into the queue
    // as the function is running

    let f = instance
        .get_func(&mut store, "add")
        .expect("function exists");

    let mut out = [Val::I32(0)];
    f.call(&mut store, &[Val::I32(left), Val::I32(right)], &mut out)?;
    // check if the value from the wasm function is "too big", and if so add some metadata to the trace
    if let Some(solution) = out[0].i32() {
        if solution > 10 {
            trace_ctx
                .set_metadata(OpenTelemetry(vec![Attribute {
                    key: "problem-too-hard".into(),
                    value: Value {
                        string_value: None,
                        int_value: Some(solution as i64),
                    },
                }]))
                .await
        }
    }

    trace_ctx.shutdown().await;

    Ok(())
}
