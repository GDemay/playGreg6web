use std::env;
use std::process::ExitCode;

const ABI_VERSION: u32 = 1;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        eprintln!("playgta6-recomp (ABI v{ABI_VERSION})");
        eprintln!("Usage: playgta6-recomp <input-pe> --out <work-dir>");
        eprintln!();
        eprintln!("Pipeline target: Xbox Series x86-64 PE -> LLVM IR -> Wasm64 (memory64).");
        return ExitCode::SUCCESS;
    }

    eprintln!("PE lifting pipeline is not implemented yet. Received args: {args:?}");
    ExitCode::FAILURE
}
