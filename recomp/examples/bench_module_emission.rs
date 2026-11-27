//! Synthetic public module-emitter timing; no lifting/LLVM compilation is timed.
use playgta6_recomp::{
    code_image::{CodeImage, CodeSection},
    discover::discover,
    lift::{ShardDispatch, emit_module, emit_module_with_dispatch, lift_function},
};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    collections::HashSet,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
    time::Instant,
};
struct Tracked;
static ACTIVE: AtomicBool = AtomicBool::new(false);
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
fn added(n: usize) {
    let live = LIVE.fetch_add(n, Ordering::Relaxed) + n;
    PEAK.fetch_max(live, Ordering::Relaxed);
}
unsafe impl GlobalAlloc for Tracked {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        if !p.is_null() && ACTIVE.load(Ordering::Relaxed) {
            added(l.size());
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        if ACTIVE.load(Ordering::Relaxed) {
            LIVE.fetch_sub(l.size(), Ordering::Relaxed);
        }
        unsafe { System.dealloc(p, l) };
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        let q = unsafe { System.realloc(p, l, n) };
        if !q.is_null() && ACTIVE.load(Ordering::Relaxed) {
            if n >= l.size() {
                added(n - l.size());
            } else {
                LIVE.fetch_sub(l.size() - n, Ordering::Relaxed);
            }
        }
        q
    }
}
#[global_allocator]
static ALLOCATOR: Tracked = Tracked;
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let kind = args.get(1).map(String::as_str).unwrap_or("helper");
    let mib: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(64);
    let bytes: &[u8] = if kind.starts_with("helper") || kind == "crlf" {
        &[0x48, 0xf7, 0xf1, 0xc3]
    } else {
        &[0xc3]
    };
    let image = CodeImage::new(
        0x1000,
        vec![CodeSection::new(0x1000, bytes, true)],
        vec![0x1000],
    );
    let function = discover(&image).functions.remove(&0x1000).unwrap();
    let mut lifted = lift_function(&function).unwrap();
    let newline = if kind == "crlf" { "\r\n" } else { "\n" };
    let attrs = if kind.starts_with("raw") { "" } else { " #0" };
    let function_count = if kind.ends_with("many") {
        1536
    } else if kind.contains("skew") {
        32
    } else {
        1
    };
    let mut ir = format!("define void @synthetic(ptr %state){attrs} {{{newline}");
    if kind.contains("dispatch") {
        ir.push_str(&format!(
            "  %v = load i64, ptr %state, !tbaa !9002{newline}"
        ));
    }
    let line = format!("  ; deterministic synthetic lifted body 0123456789abcdef{newline}");
    while ir.len()
        < mib * 1024 * 1024
            / if kind.contains("skew") {
                1
            } else {
                function_count
            }
    {
        ir.push_str(&line);
    }
    ir.push_str(&format!("  ret void{newline}}}{newline}"));
    lifted.ir.clear();
    let fixtures: Vec<_> = (0..function_count)
        .map(|i| {
            let mut f = lifted.clone();
            f.entry += i as u64 * 16;
            f.symbol = format!(
                "synthetic{i}{}",
                if kind.contains("long") {
                    "x".repeat(512)
                } else {
                    String::new()
                }
            );
            f.ir = if kind.contains("skew") && i > 0 {
                format!("define void @synthetic{i}(ptr %state) #0 {{\n  ret void\n}}\n")
            } else {
                ir.replacen("@synthetic(", &format!("@{}(", f.symbol), 1)
            };
            f
        })
        .collect();
    let external_entries: HashSet<_> = fixtures
        .iter()
        .enumerate()
        .filter(|(i, _)| i % 3 == 0)
        .map(|(_, f)| f.entry)
        .collect();
    let dispatcher_symbol = if kind.contains("long") {
        "dispatch_".repeat(1024)
    } else {
        "dispatch".to_owned()
    };
    let dispatch = kind.contains("dispatch").then_some(ShardDispatch {
        dispatcher_symbol: &dispatcher_symbol,
        external_entries: &external_entries,
    });
    LIVE.store(0, Ordering::Relaxed);
    PEAK.store(0, Ordering::Relaxed);
    ACTIVE.store(true, Ordering::Relaxed);
    let start = Instant::now();
    let output = match dispatch {
        Some(d) => emit_module_with_dispatch(&fixtures, Some(d)),
        None => emit_module(&fixtures),
    }
    .unwrap();
    let wall_ms = start.elapsed().as_secs_f64() * 1000.;
    ACTIVE.store(false, Ordering::Relaxed);
    let peak = PEAK.load(Ordering::Relaxed);
    let mut hash = 0xcbf29ce484222325u64;
    for byte in output.bytes() {
        hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
    }
    if let Some(path) = args.get(3) {
        std::fs::write(path, &output).unwrap();
    }
    println!(
        "{{\"kind\":\"{kind}\",\"fixture_mib\":{mib},\"wall_ms\":{wall_ms},\"output_bytes\":{},\"output_capacity_bytes\":{},\"peak_live_requested_bytes\":{peak},\"output_fnv64\":\"{hash:016x}\"}}",
        output.len(),
        output.capacity()
    );
}
