//! Static census of the instructions the lifter cannot translate.
//!
//! Every discovered function is lifted on its own; an instruction that turns into an
//! `__playgta6_unimplemented` trap is counted by mnemonic and grouped by the CPU feature
//! (extension) iced-x86 attributes to it. Mnemonics that the lifter lowers to a runtime helper
//! call (`cpuid`, `rdtsc`, `rdtscp`, `xgetbv`) are counted separately for every decoded instance.
//! Each row also says how many of its functions are reachable from the boot roots (PE entry
//! point, exports and TLS callbacks) over direct calls and tail calls. Indirect calls are not
//! followed, so the boot numbers are a lower bound.
//!
//! The report holds mnemonics, feature names and counts only: no addresses and no bytes.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

use iced_x86::Mnemonic;

use crate::cfg::{Edge, Function};
use crate::lift::{FlagSummaries, GuestBase, lift_function_in};

/// Mnemonics lowered to a runtime helper call instead of inline code.
const HELPER_MNEMONICS: &[Mnemonic] = &[
    Mnemonic::Cpuid,
    Mnemonic::Rdtsc,
    Mnemonic::Rdtscp,
    Mnemonic::Xgetbv,
];

/// How the lifter handles an instruction counted in the census.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Kind {
    /// Replaced by an unimplemented trap.
    Unsupported,
    /// Lowered to a runtime helper call.
    Helper,
}

/// One mnemonic of the census.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Row {
    /// CPU feature group (`AVX2`, `BMI1`, `FPU`, ...).
    pub feature: String,
    pub kind: Kind,
    /// Instruction instances.
    pub instructions: usize,
    /// Functions containing at least one.
    pub functions: usize,
    /// Boot-reachable functions containing at least one.
    pub boot_functions: usize,
    /// Instances inside boot-reachable functions.
    pub boot_instructions: usize,
    /// Reasons the lifter gave (unsupported rows only).
    pub reasons: BTreeMap<String, usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Census {
    pub functions: usize,
    pub boot_functions: usize,
    pub instructions: usize,
    /// Functions whose lifting failed outright (not instruction-level).
    pub lift_errors: usize,
    pub rows: BTreeMap<String, Row>,
}

/// Functions reachable from `roots` over direct calls and tail calls.
pub fn reachable(functions: &BTreeMap<u64, Function>, roots: &[u64]) -> BTreeSet<u64> {
    let mut seen = BTreeSet::new();
    let mut work: Vec<u64> = roots
        .iter()
        .copied()
        .filter(|r| functions.contains_key(r))
        .collect();
    while let Some(entry) = work.pop() {
        if !seen.insert(entry) {
            continue;
        }
        let Some(func) = functions.get(&entry) else {
            continue;
        };
        let tail = func
            .blocks
            .values()
            .flat_map(|b| b.successors.iter())
            .filter_map(|e| match e {
                Edge::TailCall(t) => Some(*t),
                _ => None,
            });
        for target in func.callees.iter().copied().chain(tail) {
            if functions.contains_key(&target) && !seen.contains(&target) {
                work.push(target);
            }
        }
    }
    seen
}

fn feature_of(ins: &iced_x86::Instruction) -> String {
    ins.cpuid_features()
        .first()
        .map_or_else(|| "BASE".to_string(), |f| format!("{f:?}").to_uppercase())
}

fn mnemonic_name(m: Mnemonic) -> String {
    format!("{m:?}").to_lowercase()
}

type PerFunction = BTreeMap<String, (Kind, String, usize, BTreeMap<String, usize>)>;

/// Runs the census over `functions` with `boot` as the set of boot-reachable entries.
pub fn census(functions: &BTreeMap<u64, Function>, boot: &BTreeSet<u64>) -> Census {
    let summaries = FlagSummaries::compute(functions.values());
    let mut out = Census {
        functions: functions.len(),
        boot_functions: boot.len(),
        ..Census::default()
    };
    for (entry, func) in functions {
        let is_boot = boot.contains(entry);
        let mut per_fn: PerFunction = BTreeMap::new();
        for ins in func.blocks.values().flat_map(|b| b.instructions.iter()) {
            out.instructions += 1;
            if HELPER_MNEMONICS.contains(&ins.mnemonic()) {
                let slot = per_fn
                    .entry(mnemonic_name(ins.mnemonic()))
                    .or_insert_with(|| (Kind::Helper, feature_of(ins), 0, BTreeMap::new()));
                slot.2 += 1;
            }
        }
        match lift_function_in(func, GuestBase::default(), &summaries) {
            Ok(lifted) => {
                for u in &lifted.unsupported {
                    let feature = func
                        .blocks
                        .values()
                        .flat_map(|b| b.instructions.iter())
                        .find(|i| i.ip() == u.address)
                        .map_or_else(|| "BASE".to_string(), feature_of);
                    let slot = per_fn
                        .entry(u.mnemonic.clone())
                        .or_insert_with(|| (Kind::Unsupported, feature, 0, BTreeMap::new()));
                    slot.2 += 1;
                    *slot.3.entry(u.reason.clone()).or_default() += 1;
                }
            }
            Err(_) => out.lift_errors += 1,
        }
        for (mnemonic, (kind, feature, count, reasons)) in per_fn {
            let row = out.rows.entry(mnemonic).or_insert_with(|| Row {
                feature,
                kind,
                instructions: 0,
                functions: 0,
                boot_functions: 0,
                boot_instructions: 0,
                reasons: BTreeMap::new(),
            });
            row.instructions += count;
            row.functions += 1;
            if is_boot {
                row.boot_functions += 1;
                row.boot_instructions += count;
            }
            for (r, n) in reasons {
                *row.reasons.entry(r).or_default() += n;
            }
        }
    }
    out
}

/// Renders the census grouped by feature, biggest groups first.
pub fn render(c: &Census) -> String {
    let mut s = String::new();
    let _ = writeln!(
        s,
        "Opcode census: {} functions ({} reachable from boot), {} instructions, {} lift errors",
        c.functions, c.boot_functions, c.instructions, c.lift_errors
    );
    for kind in [Kind::Unsupported, Kind::Helper] {
        let rows: Vec<(&String, &Row)> = c.rows.iter().filter(|(_, r)| r.kind == kind).collect();
        let total: usize = rows.iter().map(|(_, r)| r.instructions).sum();
        let boot: usize = rows.iter().map(|(_, r)| r.boot_instructions).sum();
        let label = match kind {
            Kind::Unsupported => "Unsupported (trap stub)",
            Kind::Helper => "Lowered to runtime helper",
        };
        let _ = writeln!(
            s,
            "\n{label}: {total} instructions ({boot} in boot-reachable functions), {} mnemonics",
            rows.len()
        );
        let mut groups: BTreeMap<&str, Vec<(&String, &Row)>> = BTreeMap::new();
        for r in rows {
            groups.entry(r.1.feature.as_str()).or_default().push(r);
        }
        let mut groups: Vec<_> = groups.into_iter().collect();
        groups.sort_by_key(|(f, g)| {
            (
                std::cmp::Reverse(g.iter().map(|r| r.1.instructions).sum::<usize>()),
                *f,
            )
        });
        for (feature, mut g) in groups {
            g.sort_by_key(|(m, r)| (std::cmp::Reverse(r.instructions), *m));
            let sum: usize = g.iter().map(|r| r.1.instructions).sum();
            let _ = writeln!(s, "  [{feature}] {sum}");
            for (m, r) in g {
                let _ = writeln!(
                    s,
                    "    {m:<14} {:>7} insns {:>6} fns | boot {:>7} insns {:>6} fns",
                    r.instructions, r.functions, r.boot_instructions, r.boot_functions
                );
                if let Some((reason, n)) = r.reasons.iter().max_by_key(|(_, n)| **n) {
                    let _ = writeln!(s, "      top reason ({n}): {reason}");
                }
            }
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::code_image::{CodeImage, CodeSection};
    use crate::discover::discover;
    use iced_x86::code_asm::*;

    const BASE: u64 = 0x1_4000_1000;

    #[test]
    fn counts_unsupported_and_helper_instructions_by_boot_reachability() {
        let mut a = CodeAssembler::new(64).unwrap();
        let mut leaf = a.create_label();
        let mut orphan = a.create_label();
        // boot function at BASE: cpuid, call leaf.
        a.cpuid().unwrap();
        a.call(leaf).unwrap();
        a.ret().unwrap();
        for _ in 0..8 {
            a.int3().unwrap();
        }
        // leaf at BASE+0x10: port I/O is not lifted.
        a.set_label(&mut leaf).unwrap();
        a.in_(eax, dx).unwrap();
        a.ret().unwrap();
        for _ in 0..14 {
            a.int3().unwrap();
        }
        // orphan at BASE+0x20: not reachable from boot.
        a.set_label(&mut orphan).unwrap();
        a.in_(eax, dx).unwrap();
        a.in_(eax, dx).unwrap();
        a.ret().unwrap();
        let code = a.assemble(BASE).unwrap();
        let leaf_va = BASE + 0x10;
        let orphan_va = BASE + 0x20;
        let image = CodeImage::new(
            BASE,
            vec![CodeSection::new(BASE, &code, true)],
            vec![BASE, leaf_va, orphan_va],
        );
        let analysis = discover(&image);
        let boot = reachable(&analysis.functions, &[BASE]);
        assert!(boot.contains(&BASE) && boot.contains(&leaf_va));
        assert!(!boot.contains(&orphan_va));
        let c = census(&analysis.functions, &boot);
        let io = &c.rows["in"];
        assert_eq!(io.kind, Kind::Unsupported);
        assert_eq!((io.instructions, io.functions), (2, 2));
        assert_eq!((io.boot_instructions, io.boot_functions), (1, 1));
        let cpuid = &c.rows["cpuid"];
        assert_eq!(cpuid.kind, Kind::Helper);
        assert_eq!((cpuid.instructions, cpuid.boot_functions), (1, 1));
        let text = render(&c);
        assert!(
            text.contains("Unsupported (trap stub): 2 instructions (1 in boot"),
            "{text}"
        );
    }
}
