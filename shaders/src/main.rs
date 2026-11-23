use std::env;
use std::process::ExitCode;

const ABI_VERSION: u32 = 1;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        eprintln!("playgta6-shaders (ABI v{ABI_VERSION})");
        eprintln!("Usage: playgta6-shaders --spirv <dir> --out <local-data/shaders>");
        eprintln!();
        eprintln!("Target pipeline: RDNA2 -> SPIR-V (--mesh-compute, --wave64-emulate) -> WGSL (naga).");
        return ExitCode::SUCCESS;
    }

    eprintln!("SPIR-V -> WGSL lowering (naga) is not wired yet. Received args: {args:?}");
    ExitCode::FAILURE
}
