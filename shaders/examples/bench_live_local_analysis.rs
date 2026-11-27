//! Synthetic native public Naga optimizer benchmark; no GPU execution or game inputs.
use playgta6_shaders::ir_opt as production;
use std::time::Instant;
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let kind = args.get(1).map(String::as_str).unwrap_or("live");
    let n: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(1024);
    let reps: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(3);
    assert!(matches!(
        kind,
        "live" | "aggregate" | "forwarded" | "no-local" | "sparse-live" | "dead-store"
    ));
    assert!(n > 0 && n <= 65536 && reps > 0 && reps <= 100);
    let mut src = String::from(
        "@group(0) @binding(0) var<storage,read_write> out: array<u32>;\n@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id:vec3<u32>) {\n",
    );
    if kind == "sparse-live" || kind == "dead-store" {
        if kind == "sparse-live" {
            src.push_str("var a:array<u32,4>; a[id.x&3u]=id.y;\n");
        } else {
            src.push_str("var unused:u32=123u;\n");
        }
        src.push_str("let v0=id.x;\n");
        for i in 1..=n {
            src.push_str(&format!("let v{i}=v{}*1664525u+id.y;\n", i - 1));
        }
        if kind == "sparse-live" {
            src.push_str("out[0]=a[id.x&3u];\n");
        } else {
            src.push_str("out[0]=id.x;\n");
        }
    } else if kind == "no-local" {
        src.push_str("let v0=id.x;\n");
        for i in 1..=n {
            src.push_str(&format!("let v{i}=v{}*1664525u+id.y;\n", i - 1));
        }
        src.push_str(&format!("out[0]=v{n};\n"));
    } else if kind == "aggregate" {
        src.push_str("var a:array<u32,4>; a[0]=id.x; a[1]=id.y; a[2]=id.z; a[3]=id.x^id.z;\n");
        for i in 0..n {
            src.push_str(&format!(
                "a[(id.x+{i}u)&3u]=a[(id.y+{i}u)&3u]*1664525u+id.z;\n"
            ));
        }
        src.push_str("out[0]=a[id.z&3u];\n");
    } else {
        src.push_str("var acc:u32=id.x;\n");
        for i in 0..n {
            if kind == "live" {
                src.push_str(&format!("if(((id.x^{i}u)&1u)!=0u){{acc=acc*1664525u+id.y;}}else{{acc=acc*1013904223u+id.z;}}\n"));
            } else {
                src.push_str("acc=acc*1664525u+id.y;\n");
            }
        }
        src.push_str("out[0]=acc;\n");
    }
    src.push_str("}\n");
    let module = naga::front::wgsl::parse_str(&src).unwrap();
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator.validate(&module).unwrap();
    let before_exprs = module.entry_points[0].function.expressions.len();
    let before_locals = module.entry_points[0].function.local_variables.len();
    assert!(kind != "no-local" || before_locals == 0);
    let mut warmup = module.clone();
    production::optimize_module(&mut warmup);
    validator.validate(&warmup).unwrap();
    for rep in 0..reps {
        let mut m = module.clone();
        let start = Instant::now();
        let stats = production::optimize_module(&mut m);
        let ms = start.elapsed().as_secs_f64() * 1000.;
        let info = validator.validate(&m).unwrap();
        let text =
            naga::back::wgsl::write_string(&m, &info, naga::back::wgsl::WriterFlags::empty())
                .unwrap();
        if rep == 0 {
            if let Some(path) = args.get(4) {
                std::fs::write(path, &text).unwrap();
            }
        }
        let mut hash = 0xcbf29ce484222325u64;
        for byte in text.bytes() {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
        }
        println!(
            "{{\"kind\":\"{kind}\",\"operations\":{n},\"sample\":{rep},\"source_bytes\":{},\"input_expressions\":{before_exprs},\"input_locals\":{before_locals},\"output_expressions\":{},\"output_locals\":{},\"wall_ms\":{ms},\"output_fnv64\":\"{hash:016x}\",\"stats\":{:?}}}",
            src.len(),
            m.entry_points[0].function.expressions.len(),
            m.entry_points[0].function.local_variables.len(),
            format!("{stats:?}")
        );
    }
}
