//! R19 measurements on real lifted code (tool-gated, local only): lifts a sample of the functions
//! of an executable under every guest memory model, writes one module per shard and model,
//! compiles each to a wasm64 object and reports IR and object sizes.
//!
//! Enabled only when `PLAYGTA6_R19_EXE` (the PE file, a read-only input) and `PLAYGTA6_R19_OUT`
//! (an output directory, by convention under the gitignored `work/`) are set. Optional:
//! `PLAYGTA6_R19_STRIDE` (lift every Nth discovered function, default 100),
//! `PLAYGTA6_R19_SHARD` (functions per module, default 250), `PLAYGTA6_R19_MODELS`
//! (comma-separated model names, default all), `PLAYGTA6_R19_NO_OBJ` (IR only).
//! Nothing derived from the input goes anywhere but the output directory; the printed summary is
//! counts and sizes only.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::{env, fs};

use super::ir::random_program;
use crate::cfg;
use crate::discover::discover;
use crate::lift::compile::{compile_to_wasm64_object, find_llvm_tool};
use crate::lift::memory::{Window, WindowCheck};
use crate::lift::{
    GuestBase, MemoryModel, emit_module, lift_function_with, lift_module, lift_module_with,
};
use crate::pe::Pe64;
use crate::pipeline::{Seeded, seed};

/// The CPU window as the browser would place it: above the 16 GiB wasm64 memory, sized for the
/// larger console profile (24 GiB; Series X uses 16 GiB of it).
pub(super) const REAL_WINDOW_BASE: u64 = 0x4_0000_0000;
pub(super) const REAL_WINDOW_SIZE: u64 = 0x6_0000_0000;
/// Page-table slots covering guest VAs [0, 64 GiB).
pub(super) const REAL_PAGES: u64 = 1 << 20;

fn window(check: WindowCheck, stack_flat: bool, contiguous_runs: bool) -> MemoryModel {
    MemoryModel::Windowed(Window {
        flat: GuestBase::Constant(0),
        base: REAL_WINDOW_BASE,
        size: REAL_WINDOW_SIZE,
        check,
        stack_flat,
        contiguous_runs,
    })
}

/// The models measured, by name.
pub(super) fn models() -> Vec<(&'static str, MemoryModel)> {
    vec![
        ("identity", GuestBase::Constant(0).into()),
        ("symbol", GuestBase::Symbol.into()),
        ("window", window(WindowCheck::Branch, false, false)),
        ("window-stack", window(WindowCheck::Branch, true, false)),
        ("window-runs", window(WindowCheck::Branch, true, true)),
        (
            "window-outline-stack",
            window(WindowCheck::Outline, true, false),
        ),
        ("window-select", window(WindowCheck::Select, false, false)),
        (
            "page",
            MemoryModel::PageTable {
                pages: REAL_PAGES,
                stack: None,
                contiguous_runs: false,
            },
        ),
        (
            "page-runs",
            MemoryModel::PageTable {
                pages: REAL_PAGES,
                stack: Some(GuestBase::Constant(0)),
                contiguous_runs: true,
            },
        ),
    ]
}

/// Symbols only the table models emit.
const TABLE_MODEL_SYMBOLS: [&str; 7] = [
    "__playgta6_window_table",
    "__playgta6_page_table",
    "__playgta6_guest_read",
    "__playgta6_guest_scratch",
    "__playgta6_guest_write",
    "__playgta6_guest_memmove",
    "__playgta6_guest_memset",
];

#[test]
fn default_model_output_is_the_base_translation() {
    // The default is MemoryModel::Base(GuestBase::Symbol), emitted exactly as before R19 (the
    // byte-for-byte comparison against the pre-R19 lifter on 3,562 real functions is recorded in
    // the R19 report); nothing of the table models leaks into it.
    let functions = random_program(150);
    let default = lift_module(functions.values()).expect("lifts").ir;
    let explicit = lift_module_with(functions.values(), GuestBase::Symbol)
        .expect("lifts")
        .ir;
    assert_eq!(default, explicit);
    assert_eq!(MemoryModel::default(), MemoryModel::Base(GuestBase::Symbol));
    for symbol in TABLE_MODEL_SYMBOLS {
        assert!(!default.contains(symbol), "{symbol} in default output");
    }
}

#[test]
fn table_entries_are_ordinary_loads_and_the_base_stays_invariant() {
    let functions = random_program(60);
    for (name, model) in models().into_iter().skip(2) {
        let ir = lift_module_with(functions.values(), model)
            .expect("lifts")
            .ir;
        let mut entry_loads = 0;
        for line in ir.lines() {
            // An entry load reads a pointer computed from the table (`getelementptr ... @table`).
            if line.contains("load i64, ptr %") && line.contains("align 8") {
                assert!(
                    !line.contains("invariant"),
                    "{name}: table entry load {line}"
                );
                entry_loads += 1;
            }
            if line.contains("@__playgta6_guest_base,") {
                assert!(line.contains("!invariant.load"), "{name}: {line}");
            }
        }
        assert!(entry_loads > 0, "{name}: no table loads");
        assert!(
            ir.contains("@__playgta6_window_table") || ir.contains("@__playgta6_page_table"),
            "{name}"
        );
    }
}

#[test]
fn contiguous_runs_store_marks_its_last_slot_only_when_it_crosses() {
    // `mov [rcx], rax` under window-runs: one hot dirty check for the store's slot (with its
    // atomic release store behind `icmp eq i8 ..., 0`); the dirty check for the last byte's slot
    // sits in its own block behind `icmp ne` (taken only by a store that leaves its slot), not on
    // the hot path.
    use iced_x86::code_asm::*;
    let f = super::asm_function(0x1000, |a| {
        a.mov(qword_ptr(rcx), rax)?;
        a.ret()
    });
    let ir = lift_module_with([&f], window(WindowCheck::Branch, true, true))
        .expect("lifts")
        .ir;
    // Labels of the blocks holding a dirty check and a release store.
    let mut label = "";
    let mut checked = Vec::new();
    let mut stored = Vec::new();
    for line in ir.lines() {
        if let Some(l) = line.strip_suffix(':') {
            label = l;
        } else if line.contains("load atomic i8, ptr ") && line.contains(" monotonic, align 1") {
            checked.push(label);
        } else if line.contains("store atomic i8 1, ptr ") && line.contains(" release, align 1") {
            stored.push(label);
        }
    }
    assert_eq!(checked.len(), 2, "{ir}");
    assert_eq!(stored.len(), 2, "{ir}");
    assert_ne!(
        checked[0], checked[1],
        "both dirty checks on the hot path:\n{ir}"
    );
    let cold = checked[1];
    assert!(cold.starts_with("gm"), "{ir}");
    // The cold mark block is entered only by an `icmp ne` of the two slot indices.
    let branch = ir
        .lines()
        .find(|l| l.contains(&format!("label %{cold},")))
        .expect("branch to the cold mark");
    let cond = branch
        .split_whitespace()
        .nth(2)
        .expect("condition")
        .trim_end_matches(',');
    assert!(
        ir.lines()
            .any(|l| l.trim_start().starts_with(&format!("{cond} = icmp ne i64"))),
        "{ir}"
    );
    // Both atomic release stores sit in cold `gms*` blocks entered by `icmp eq i8 ..., 0`.
    for set_block in stored {
        assert!(set_block.starts_with("gms"), "{ir}");
        let set_br = ir
            .lines()
            .find(|l| l.contains(&format!("label %{set_block},")))
            .expect("branch to mark-set block");
        assert!(
            set_br.contains("!prof !{!\"branch_weights\", i32 1, i32 2000}"),
            "{ir}"
        );
        let set_cond = set_br
            .split_whitespace()
            .nth(2)
            .expect("condition")
            .trim_end_matches(',');
        assert!(
            ir.lines()
                .any(|l| l.trim_start().starts_with(&format!("{set_cond} = icmp eq i8"))
                    && l.ends_with(", 0")),
            "{ir}"
        );
    }
}

#[test]
fn window_dirty_marks_guard_atomic_release_store_with_monotonic_check() {
    // R57: every window model guards `store atomic i8 1 ... release` behind a `load atomic i8 ...
    // monotonic` + `icmp eq i8 ..., 0` cold branch so already-dirty slots avoid `LOCK XCHG`.
    use iced_x86::code_asm::*;
    let f = super::asm_function(0x1000, |a| {
        a.mov(qword_ptr(rcx), rax)?;
        a.lock().xadd(qword_ptr(rdx), rax)?;
        a.ret()
    });
    for name in ["window", "window-outline", "window-runs"] {
        let model = MemoryModel::from_name(name).expect("model");
        let ir = lift_module_with([&f], model).expect("lifts").ir;
        let loads = ir.matches("load atomic i8, ptr ").count();
        let stores = ir.matches("store atomic i8 1, ptr ").count();
        assert!(stores > 0, "{name}: expected dirty marks:\n{ir}");
        assert_eq!(
            loads, stores,
            "{name}: every dirty release store must have a matching monotonic load guard:\n{ir}"
        );
    }
    let select_ir = lift_module_with([&f], window(WindowCheck::Select, false, false))
        .expect("lifts")
        .ir;
    let loads = select_ir.matches("load atomic i8, ptr ").count();
    let stores = select_ir.matches("store atomic i8 1, ptr ").count();
    assert!(stores > 0, "window-select:\n{select_ir}");
    assert_eq!(loads, stores, "window-select:\n{select_ir}");
}

#[test]
fn constant_addresses_outside_window_elide_window_checks_and_atomics() {
    use iced_x86::{Code, Instruction, MemoryOperand, Register};
    // RIP-relative globals in the executable image (0x1_4AA3_6C18..0x1_4AA3_6C80), strictly below
    // GUEST_WINDOW_BASE (0x4_0000_0000), plus an address above GUEST_WINDOW_BASE + SIZE.
    let g_below = 0x1_4AA3_6C18u64;
    let g_below2 = 0x1_4AA3_6C20u64;
    let g_vec = 0x1_4AA3_6C40u64;
    let g_vec2 = 0x1_4AA3_6C60u64;
    let bytes = super::encode(
        0x1_4000_1000,
        &[
            Instruction::with2(
                Code::Mov_r64_rm64,
                Register::RAX,
                MemoryOperand::with_base_displ(Register::RIP, g_below as i64),
            )
            .expect("mov load"),
            Instruction::with2(
                Code::Mov_rm64_r64,
                MemoryOperand::with_base_displ(Register::RIP, g_below2 as i64),
                Register::RAX,
            )
            .expect("mov store"),
            Instruction::with2(
                Code::VEX_Vmovdqu_ymm_ymmm256,
                Register::YMM0,
                MemoryOperand::with_base_displ(Register::RIP, g_vec as i64),
            )
            .expect("vmovdqu load"),
            Instruction::with2(
                Code::VEX_Vmovdqu_ymmm256_ymm,
                MemoryOperand::with_base_displ(Register::RIP, g_vec2 as i64),
                Register::YMM0,
            )
            .expect("vmovdqu store"),
            Instruction::with(Code::Retnq),
        ],
    );
    let f = super::Program::new()
        .bytes(0x1_4000_1000, &bytes)
        .function(0x1_4000_1000);

    for (name, model) in [
        ("window", MemoryModel::from_name("window").expect("m")),
        (
            "window-runs",
            MemoryModel::from_name("window-runs").expect("m"),
        ),
        (
            "window-outline",
            MemoryModel::from_name("window-outline").expect("m"),
        ),
        ("window-select", window(WindowCheck::Select, true, false)),
    ] {
        let module = lift_module_with([&f], model).expect("lifts");
        let ir = &module.ir;
        assert!(
            !ir.contains("__playgta6_window_table"),
            "{name}: constant out-of-window addresses must not reference window_table:\n{ir}"
        );
        assert!(
            !ir.contains("__playgta6_window_dirty"),
            "{name}: constant out-of-window stores must not reference window_dirty:\n{ir}"
        );
        assert!(
            !ir.contains("icmp ult"),
            "{name}: constant out-of-window addresses must not emit range checks:\n{ir}"
        );
        assert!(
            !module.requires_shared_memory(),
            "{name}: non-atomic out-of-window stores must not pull in +atomics:\n{ir}"
        );
    }

    // Locked atomic to a constant out-of-window global also skips window_table and window_dirty.
    let mut lock_xadd = Instruction::with2(
        Code::Xadd_rm64_r64,
        MemoryOperand::with_base_displ(Register::RIP, g_below as i64),
        Register::RAX,
    )
    .expect("xadd");
    lock_xadd.set_has_lock_prefix(true);
    let atomic_bytes = super::encode(
        0x1_4000_2000,
        &[lock_xadd, Instruction::with(Code::Retnq)],
    );
    let atomic_f = super::Program::new()
        .bytes(0x1_4000_2000, &atomic_bytes)
        .function(0x1_4000_2000);
    let atomic_ir = lift_module_with(
        [&atomic_f],
        MemoryModel::from_name("window-runs").expect("m"),
    )
    .expect("lifts")
    .ir;
    assert!(
        !atomic_ir.contains("__playgta6_window_table")
            && !atomic_ir.contains("__playgta6_window_dirty"),
        "atomic on out-of-window global must not touch window tables:\n{atomic_ir}"
    );
    assert!(atomic_ir.contains("atomicrmw add ptr"), "{atomic_ir}");
}

#[test]
fn constant_addresses_inside_window_fold_slot_and_straddle_checks() {
    use iced_x86::{Code, Instruction, MemoryOperand, Register};
    // Code near the window so RIP-relative addresses land inside [0x4_0000_0000, 0xA_0000_0000):
    // aligned slot 1 (+0x10008), and straddling slot 1|2 (+0x1FFFC, 8 bytes).
    let base = REAL_WINDOW_BASE;
    let aligned = base + 0x1_0008;
    let straddle = base + 0x1_FFFC;
    let code_va = base - 0x1000;
    let bytes = super::encode(
        code_va,
        &[
            Instruction::with2(
                Code::Mov_r64_rm64,
                Register::RAX,
                MemoryOperand::with_base_displ(Register::RIP, aligned as i64),
            )
            .expect("aligned load"),
            Instruction::with2(
                Code::Mov_rm64_r64,
                MemoryOperand::with_base_displ(Register::RIP, aligned as i64),
                Register::RAX,
            )
            .expect("aligned store"),
            Instruction::with2(
                Code::Mov_rm64_r64,
                MemoryOperand::with_base_displ(Register::RIP, straddle as i64),
                Register::RAX,
            )
            .expect("straddle store"),
            Instruction::with(Code::Retnq),
        ],
    );
    let f = super::Program::new().bytes(code_va, &bytes).function(code_va);

    // window-runs: no dynamic range branch (`icmp ult`), direct constant slot GEPs, 1 dirty mark
    // for aligned store + 2 static dirty marks (slots 1 and 2) for straddling store (no `icmp ne`).
    let runs_ir = lift_module_with([&f], window(WindowCheck::Branch, true, true))
        .expect("lifts")
        .ir;
    assert!(
        !runs_ir.contains("icmp ult"),
        "constant in-window must elide range check:\n{runs_ir}"
    );
    assert!(
        !runs_ir.contains("icmp ne"),
        "constant in-window store must elide runtime crosses check:\n{runs_ir}"
    );
    assert!(
        runs_ir.contains("getelementptr i64, ptr @__playgta6_window_table, i64 1"),
        "{runs_ir}"
    );
    assert!(
        runs_ir.contains("getelementptr i8, ptr @__playgta6_window_dirty, i64 2"),
        "straddling store marks slot 2 statically:\n{runs_ir}"
    );
    assert_eq!(runs_ir.matches("store atomic i8 1, ptr ").count(), 3, "{runs_ir}");

    // window (!contiguous_runs): straddling constant store goes directly to `@__playgta6_slow_store8`.
    let win_ir = lift_module_with([&f], window(WindowCheck::Branch, true, false))
        .expect("lifts")
        .ir;
    assert!(!win_ir.contains("icmp ult"), "{win_ir}");
    assert!(!win_ir.contains("icmp ule"), "{win_ir}");
    assert!(
        win_ir.contains("call void @__playgta6_slow_store8("),
        "{win_ir}"
    );

    // Exact window boundary and u64::MAX wraparound checks:
    //  - W - 1 (0x3_FFFF_FFFF), W + S (0xA_0000_0000), and u64::MAX - 7 (0xFFFF_FFFF_FFFF_FFF8)
    //    must all stay flat (outside window).
    //  - W (0x4_0000_0000) and W + S - 8 (0x9_FFFF_FFF8) must fold to window_table slots 0 and
    //    (S >> 16) - 1 with zero range checks.
    for (label, va, expect_window, expect_slot) in [
        ("below_w", REAL_WINDOW_BASE - 1, false, 0u64),
        ("at_w", REAL_WINDOW_BASE, true, 0u64),
        (
            "at_end_minus_8",
            REAL_WINDOW_BASE + REAL_WINDOW_SIZE - 8,
            true,
            (REAL_WINDOW_SIZE >> 16) - 1,
        ),
        ("at_end", REAL_WINDOW_BASE + REAL_WINDOW_SIZE, false, 0u64),
        ("near_u64_max", u64::MAX - 7, false, 0u64),
    ] {
        let code_at = va.wrapping_sub(0x100);
        let b = super::encode(
            code_at,
            &[
                Instruction::with2(
                    Code::Mov_r64_rm64,
                    Register::RAX,
                    MemoryOperand::with_base_displ(Register::RIP, va as i64),
                )
                .expect("mov"),
                Instruction::with(Code::Retnq),
            ],
        );
        let func = super::Program::new().bytes(code_at, &b).function(code_at);
        let ir = lift_module_with([&func], window(WindowCheck::Branch, true, true))
            .expect("lifts")
            .ir;
        assert!(!ir.contains("icmp ult"), "{label}: no dynamic range check:\n{ir}");
        if expect_window {
            assert!(
                ir.contains(&format!(
                    "getelementptr i64, ptr @__playgta6_window_table, i64 {expect_slot}"
                )),
                "{label}: expected slot {expect_slot}:\n{ir}"
            );
        } else {
            assert!(
                !ir.contains("__playgta6_window_table"),
                "{label}: outside window must stay flat:\n{ir}"
            );
        }
    }
}

#[test]
fn avx256_stack_frame_and_burst_optimizations_reduce_window_checks() {
    use iced_x86::code_asm::*;
    // 1. 256-bit AVX stack spill/reload and frame-pointer `[rbp - 0x10]` skip window checks.
    let stack_f = super::asm_function(0x1000, |a| {
        a.push(rbp)?;
        a.mov(rbp, rsp)?;
        a.sub(rsp, 0x60)?;
        a.vmovdqu(ymmword_ptr(rsp + 0x20), ymm0)?;
        a.vmovdqu(ymm1, ymmword_ptr(rsp + 0x20))?;
        a.mov(qword_ptr(rbp - 0x10), rax)?;
        a.add(rax, qword_ptr(rbp - 0x10))?;
        a.add(rsp, 0x60)?;
        a.pop(rbp)?;
        a.ret()
    });
    let stack_ir = lift_module_with([&stack_f], window(WindowCheck::Branch, true, true))
        .expect("lifts")
        .ir;
    assert!(
        !stack_ir.contains("__playgta6_window_table")
            && !stack_ir.contains("__playgta6_window_dirty"),
        "AVX256 stack spills and frame-pointer accesses must stay flat:\n{stack_ir}"
    );

    // 2. Single 256-bit AVX load + store uses one 32-byte `<4 x i64>` access each (not two 16-byte).
    let avx_f = super::asm_function(0x2000, |a| {
        a.vmovdqu(ymm0, ymmword_ptr(rcx))?;
        a.vmovdqu(ymmword_ptr(rdx), ymm0)?;
        a.ret()
    });
    let avx_ir = lift_module_with([&avx_f], window(WindowCheck::Branch, true, true))
        .expect("lifts")
        .ir;
    assert_eq!(avx_ir.matches("load <4 x i64>, ptr ").count(), 1, "{avx_ir}");
    assert_eq!(
        avx_ir.matches("store <4 x i64> ").count(),
        2,
        "one flat store + one window store (32 bytes each):\n{avx_ir}"
    );
    assert_eq!(
        avx_ir.matches("@__playgta6_window_table").count(),
        3,
        "1 decl + 1 load lookup + 1 store lookup:\n{avx_ir}"
    );

    // 3. Burst of 4 AVX loads from same base `[rcx + 0/32/64/96]` emits only ONE window table load.
    let burst_f = super::asm_function(0x3000, |a| {
        a.vmovdqu(ymm0, ymmword_ptr(rcx))?;
        a.vmovdqu(ymm1, ymmword_ptr(rcx + 32))?;
        a.vmovdqu(ymm2, ymmword_ptr(rcx + 64))?;
        a.vmovdqu(ymm3, ymmword_ptr(rcx + 96))?;
        a.vpaddq(ymm0, ymm0, ymm1)?;
        a.vpaddq(ymm2, ymm2, ymm3)?;
        a.vpaddq(ymm0, ymm0, ymm2)?;
        a.ret()
    });
    let burst_ir = lift_module_with([&burst_f], window(WindowCheck::Branch, true, true))
        .expect("lifts")
        .ir;
    assert_eq!(
        burst_ir
            .matches("getelementptr i64, ptr @__playgta6_window_table")
            .count(),
        1,
        "4 coalesced AVX loads must share one window_table lookup:\n{burst_ir}"
    );

    // 4. Read-modify-write `add [rcx], rax` under window-runs reuses the load's table lookup.
    let rmw_f = super::asm_function(0x4000, |a| {
        a.add(qword_ptr(rcx), rax)?;
        a.ret()
    });
    let rmw_ir = lift_module_with([&rmw_f], window(WindowCheck::Branch, true, true))
        .expect("lifts")
        .ir;
    assert_eq!(
        rmw_ir
            .matches("getelementptr i64, ptr @__playgta6_window_table")
            .count(),
        1,
        "RMW under window-runs must reuse the load's window_table lookup:\n{rmw_ir}"
    );
}

/// FNV-1a 64 of a byte string (stable fingerprint of emitted IR).
fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

#[test]
fn r19_measure_real_code() {
    let (Some(exe), Some(out)) = (
        env::var_os("PLAYGTA6_R19_EXE"),
        env::var_os("PLAYGTA6_R19_OUT"),
    ) else {
        println!("SKIP r19_measure_real_code: PLAYGTA6_R19_EXE / PLAYGTA6_R19_OUT not set");
        return;
    };
    let out = PathBuf::from(out);
    let stride: usize = env::var("PLAYGTA6_R19_STRIDE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100);
    let shard: usize = env::var("PLAYGTA6_R19_SHARD")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(250);
    let wanted: Option<Vec<String>> = env::var("PLAYGTA6_R19_MODELS")
        .ok()
        .map(|v| v.split(',').map(str::to_string).collect());
    let emit_obj = env::var_os("PLAYGTA6_R19_NO_OBJ").is_none();
    let tool = if emit_obj { find_llvm_tool() } else { None };

    let bytes = fs::read(&exe).expect("read executable");
    let pe = Pe64::parse(&bytes).expect("parse PE");
    let Seeded { image, .. } = seed(&pe, &bytes);
    let analysis = discover(&image);
    let sample: Vec<&cfg::Function> = analysis.functions.values().step_by(stride).collect();
    let instructions: usize = sample
        .iter()
        .flat_map(|f| f.blocks.values())
        .map(|b| b.instructions.len())
        .sum();
    println!(
        "r19: {} functions discovered, {} sampled (every {stride}th), {instructions} instructions",
        analysis.functions.len(),
        sample.len()
    );

    let mut report = String::new();
    let mut totals: BTreeMap<&str, (usize, u64, u64, u64)> = BTreeMap::new();
    for (name, model) in models() {
        if wanted
            .as_ref()
            .is_some_and(|w| !w.iter().any(|m| m == name))
        {
            continue;
        }
        let dir = out.join(name);
        fs::create_dir_all(&dir).expect("create output dir");
        let (mut ir_bytes, mut obj_bytes, mut hash, mut lifted) = (0u64, 0u64, 0u64, 0usize);
        for (i, chunk) in sample.chunks(shard).enumerate() {
            let mut functions = Vec::new();
            for f in chunk {
                if let Ok(l) = lift_function_with(f, model) {
                    functions.push(l);
                }
            }
            lifted += functions.len();
            let ir = emit_module(&functions).expect("emit module");
            ir_bytes += ir.len() as u64;
            hash = hash.rotate_left(5) ^ fnv(ir.as_bytes());
            let ll = dir.join(format!("shard-{i:04}.ll"));
            fs::write(&ll, &ir).expect("write .ll");
            if let Some(tool) = &tool {
                let obj = ll.with_extension("o");
                compile_to_wasm64_object(tool, &ll, &obj).expect("compile shard");
                obj_bytes += fs::metadata(&obj).map(|m| m.len()).unwrap_or(0);
            }
        }
        let _ = writeln!(
            report,
            "{name}\tfunctions={lifted}\tir_bytes={ir_bytes}\tobj_bytes={obj_bytes}\tir_hash={hash:016x}"
        );
        totals.insert(name, (lifted, ir_bytes, obj_bytes, hash));
    }
    fs::write(out.join("sizes.tsv"), &report).expect("write report");
    print!("{report}");
}
