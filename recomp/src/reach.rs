//! Register values that reach a block over every intra-function path (for jump-table recovery).
//!
//! [`crate::discover`] slices an indirect `jmp` together with its chain of sole predecessors. When
//! that chain stops at a merge (a loop header, or a block with several incoming edges), the table
//! base or the index may be defined before the merge, on several paths. [`reaching_values`] runs
//! a forward dataflow over every block that can reach the merge block and reports, for each
//! requested 64-bit register, what holds on *all* incoming paths:
//!
//! * a single value (`[v, v]`), e.g. a RIP-relative `lea` of the table, possibly carried around a
//!   loop through register copies, or
//! * an unsigned interval `[lo, hi]` of the full 64-bit register, from constants, masks, shifts,
//!   zero extension and the `cmp` / `test` + conditional-branch guards on the edges, or
//! * nothing (the register is unknown on at least one path).
//!
//! The lattice per register is unvisited < interval < unknown; the join is the interval hull.
//! Each block's input is the join over its incoming edges, each refined by the guard of the edge.
//! Function entries start with every register unknown. A block whose input changes more than
//! [`WIDEN_AFTER`] times widens every interval it keeps changing to unknown, so loops terminate.
//!
//! Soundness: every transfer either computes an over-approximation of the register's value or
//! yields unknown; anything not modelled (memory loads into 64-bit registers, partial 8/16-bit
//! writes, other arithmetic, `pop`, calls for Win64-volatile registers) yields unknown. Guards
//! refine only the compared register, only for unsigned conditions or when the interval is
//! already non-negative, and only when no instruction between the compare and the branch writes
//! that register. Blocks that cannot reach the merge never contribute (the analysis only sees the
//! backward-reachable subgraph), so a register reused for something else there does not matter.
//!
//! Cost: one deterministic backward BFS to collect at most [`MAX_REACH_BLOCKS`] blocks within
//! [`MAX_REACH_DEPTH`] hops, then a bounded deterministic worklist over them; every block's input
//! can change at most `WIDEN_AFTER + 1` times per register before it is unknown, so the work is
//! `O(blocks * instructions per block * (WIDEN_AFTER + 1) * 16)` per queried merge (capped by
//! [`MAX_WORKLIST_STEPS`]).

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use iced_x86::{
    Decoder, DecoderOptions, FlowControl, Instruction, InstructionInfoFactory, Mnemonic, OpAccess,
    OpKind, Register,
};

use crate::cfg::{BasicBlock, Edge, Terminator};

/// Blocks of the backward-reachable subgraph examined for one merge (beyond: no result).
pub(crate) const MAX_REACH_BLOCKS: usize = 4096;

/// Maximum backward BFS hop depth from the target block before bailing out.
pub(crate) const MAX_REACH_DEPTH: usize = MAX_REACH_BLOCKS;

/// Input changes of one block after which still-changing intervals widen to unknown.
const WIDEN_AFTER: u32 = 8;

/// Hard upper bound on forward worklist pops per query (defensive livelock/cycle guard).
/// Each block in a monotone `GPR_COUNT`-register lattice (tracking both 64-bit and 32-bit
/// intervals) with widening after `WIDEN_AFTER` updates can change at most
/// `WIDEN_AFTER + 2 * GPR_COUNT` times after its initial visit.
pub(crate) const MAX_WORKLIST_STEPS: usize =
    MAX_REACH_BLOCKS * (WIDEN_AFTER as usize + 2 * GPR_COUNT + 2);

/// What is known about a full 64-bit register on every path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Known {
    /// No path seen yet (only during the analysis).
    Unvisited,
    /// Inclusive unsigned interval of the full 64-bit value.
    Range { lo: u64, hi: u64 },
    /// Not known on at least one path.
    Unknown,
}

impl Known {
    fn join(self, other: Known) -> Known {
        match (self, other) {
            (Known::Unvisited, x) | (x, Known::Unvisited) => x,
            (Known::Unknown, _) | (_, Known::Unknown) => Known::Unknown,
            (Known::Range { lo: a, hi: b }, Known::Range { lo: c, hi: d }) => Known::Range {
                lo: a.min(c),
                hi: b.max(d),
            },
        }
    }

    fn exact(v: u64) -> Known {
        Known::Range { lo: v, hi: v }
    }

    fn within_u32(self) -> bool {
        matches!(self, Known::Range { hi, .. } if hi <= u64::from(u32::MAX))
    }
}

const GPR_COUNT: usize = 16;

/// Copy-alias relation for one register slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Alias {
    /// No path seen yet (only during the analysis).
    Unvisited,
    /// Zero-extended 32-bit copy (`reg == zext32(src)`).
    Copy32(u8),
    /// Full 64-bit copy (`reg == src`).
    Copy64(u8),
    /// No register copy known.
    None,
}

/// Register file state: per-GPR full 64-bit unsigned intervals, low 32-bit unsigned intervals,
/// and register copy aliases (index = `Register::number()` of RAX..R15).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct State {
    ranges: [Known; GPR_COUNT],
    u32_ranges: [Known; GPR_COUNT],
    aliases: [Alias; GPR_COUNT],
}

/// Low 32-bit interval implied by a full 64-bit [`Known`] value.
fn low32_of(k: Known) -> Known {
    match k {
        Known::Unvisited => Known::Unvisited,
        Known::Unknown => Known::Unknown,
        Known::Range { lo, hi } if (lo >> 32) == (hi >> 32) => Known::Range {
            lo: lo & mask(32),
            hi: hi & mask(32),
        },
        Known::Range { .. } => Known::Unknown,
    }
}

fn is_high_byte(reg: Register) -> bool {
    matches!(
        reg,
        Register::AH | Register::BH | Register::CH | Register::DH
    )
}

impl State {
    fn unvisited() -> Self {
        Self {
            ranges: [Known::Unvisited; GPR_COUNT],
            u32_ranges: [Known::Unvisited; GPR_COUNT],
            aliases: [Alias::Unvisited; GPR_COUNT],
        }
    }
    fn unknown() -> Self {
        Self {
            ranges: [Known::Unknown; GPR_COUNT],
            u32_ranges: [Known::Unknown; GPR_COUNT],
            aliases: [Alias::None; GPR_COUNT],
        }
    }
    fn get(&self, reg: Register) -> Known {
        slot(reg).map_or(Known::Unknown, |i| self.ranges[i])
    }
    fn get_u32(&self, reg: Register) -> Known {
        slot(reg).map_or(Known::Unknown, |i| self.u32_ranges[i])
    }
    fn set(&mut self, reg: Register, value: Known, u32_value: Known) {
        if let Some(i) = slot(reg) {
            self.ranges[i] = value;
            self.u32_ranges[i] = u32_value;
        }
    }
    fn join(&self, other: &State) -> State {
        let mut out = *self;
        for (o, x) in out.ranges.iter_mut().zip(other.ranges.iter()) {
            *o = o.join(*x);
        }
        for (o, x) in out.u32_ranges.iter_mut().zip(other.u32_ranges.iter()) {
            *o = o.join(*x);
        }
        for (i, o) in out.aliases.iter_mut().enumerate() {
            *o = match (self.aliases[i], other.aliases[i]) {
                (Alias::Unvisited, y) | (y, Alias::Unvisited) => y,
                (a, b) if a == b => a,
                // Copies of one source on one path, of each other on the other: `i = zext32(a)`
                // and `b = zext32(a)` on the first path make `i = zext32(b)` there too (and
                // symmetrically), and likewise for full 64-bit copies.
                (Alias::Copy32(a), Alias::Copy32(b))
                    if self.aliases[usize::from(b)] == Alias::Copy32(a) =>
                {
                    Alias::Copy32(b)
                }
                (Alias::Copy32(a), Alias::Copy32(b))
                    if other.aliases[usize::from(a)] == Alias::Copy32(b) =>
                {
                    Alias::Copy32(a)
                }
                (Alias::Copy64(a), Alias::Copy64(b))
                    if self.aliases[usize::from(b)] == Alias::Copy64(a) =>
                {
                    Alias::Copy64(b)
                }
                (Alias::Copy64(a), Alias::Copy64(b))
                    if other.aliases[usize::from(a)] == Alias::Copy64(b) =>
                {
                    Alias::Copy64(a)
                }
                _ => Alias::None,
            };
        }
        out
    }
}

fn slot(reg: Register) -> Option<usize> {
    let full = reg.full_register();
    full.is_gpr64().then(|| full.number())
}

fn is_win64_volatile(reg: Register) -> bool {
    matches!(
        reg.full_register(),
        Register::RAX
            | Register::RCX
            | Register::RDX
            | Register::R8
            | Register::R9
            | Register::R10
            | Register::R11
    )
}

fn mask(bits: u32) -> u64 {
    if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}

/// Deterministic backward-reachable subgraph collected by [`collect_backward_subgraph_bounded`].
struct BackwardSubgraph {
    /// Every block reachable backward from `target`, strictly sorted by VA.
    blocks: BTreeSet<u64>,
    /// Forward-analysis seed blocks (function entries and predecessor-less blocks), sorted by VA.
    roots: BTreeSet<u64>,
}

/// Deterministic backward BFS from `target` over intra-function predecessors, stopping at
/// `is_entry` blocks. Guarantees sorted `BTreeSet` membership, deduplicates and sorts predecessor
/// lists so results are invariant to caller ordering, skips self-loops and cycles cleanly, and
/// returns `None` if `target` or any backward-reachable predecessor is missing from `blocks` or if
/// `max_blocks` or `max_depth` would be exceeded.
fn collect_backward_subgraph_bounded(
    blocks: &BTreeMap<u64, BasicBlock>,
    preds: &dyn Fn(u64) -> Vec<u64>,
    is_entry: &dyn Fn(u64) -> bool,
    target: u64,
    max_blocks: usize,
    max_depth: usize,
) -> Option<BackwardSubgraph> {
    if max_blocks == 0 {
        return None;
    }
    blocks.get(&target)?;

    let mut sub: BTreeSet<u64> = BTreeSet::new();
    let mut roots: BTreeSet<u64> = BTreeSet::new();
    let mut queue: VecDeque<(u64, usize)> = VecDeque::new();
    sub.insert(target);
    queue.push_back((target, 0));

    while let Some((b, depth)) = queue.pop_front() {
        if is_entry(b) {
            roots.insert(b);
            continue;
        }
        let mut pred_list = preds(b);
        if pred_list.is_empty() {
            // Blocks whose predecessors are all outside the function's known code (no entry above
            // them) can still start a path: treat a block without predecessors as unknown on entry.
            roots.insert(b);
            continue;
        }
        pred_list.sort_unstable();
        pred_list.dedup();

        for p in pred_list {
            // Self-loops (`p == b`) and back-edges to already-discovered blocks are skipped without
            // growing the queue or advancing depth.
            if sub.contains(&p) {
                continue;
            }
            blocks.get(&p)?;
            let next_depth = depth.checked_add(1)?;
            if next_depth > max_depth || sub.len() >= max_blocks {
                return None;
            }
            sub.insert(p);
            queue.push_back((p, next_depth));
        }
    }

    Some(BackwardSubgraph { blocks: sub, roots })
}

/// Combined reaching facts on entry to a block: full 64-bit intervals, low 32-bit intervals,
/// and register copies (`(dst, src, is_64bit)`).
pub(crate) struct ReachingFacts {
    pub values: Vec<(Register, Known)>,
    pub u32_values: Vec<(Register, Known)>,
    pub copies: Vec<(Register, Register, bool)>,
}

/// Compute full 64-bit intervals, low 32-bit intervals, and register copy aliases on entry to
/// `target` in one forward dataflow pass.
pub(crate) fn reaching_facts(
    blocks: &BTreeMap<u64, BasicBlock>,
    preds: &dyn Fn(u64) -> Vec<u64>,
    is_entry: &dyn Fn(u64) -> bool,
    target: u64,
    regs: &[Register],
) -> Option<ReachingFacts> {
    let input = reaching_state_bounded(
        blocks,
        preds,
        is_entry,
        target,
        MAX_REACH_BLOCKS,
        MAX_REACH_DEPTH,
        MAX_WORKLIST_STEPS,
    )?;
    let normalize = |k: Known| if k == Known::Unvisited { Known::Unknown } else { k };
    let values = regs.iter().map(|&r| (r, normalize(input.get(r)))).collect();
    let u32_values = regs
        .iter()
        .map(|&r| (r, normalize(input.get_u32(r))))
        .collect();
    let copies = regs
        .iter()
        .filter_map(|&r| match input.aliases[slot(r)?] {
            Alias::Copy32(a) if usize::from(a) < GPR_COUNT => {
                Some((r.full_register(), Register::RAX + u32::from(a), false))
            }
            Alias::Copy64(a) if usize::from(a) < GPR_COUNT => {
                Some((r.full_register(), Register::RAX + u32::from(a), true))
            }
            _ => None,
        })
        .collect();
    Some(ReachingFacts {
        values,
        u32_values,
        copies,
    })
}

/// Values of `regs` on entry to `target`, over every intra-function path that reaches it.
/// `preds(b)` lists the blocks with an intra-function edge into `b`; `is_entry(b)` marks function
/// entries (where every register is unknown). `None` when the reachable subgraph exceeds
/// [`MAX_REACH_BLOCKS`] / [`MAX_REACH_DEPTH`], the worklist exceeds [`MAX_WORKLIST_STEPS`], or
/// `target` (or a backward-reachable predecessor) is unknown.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn reaching_values(
    blocks: &BTreeMap<u64, BasicBlock>,
    preds: &dyn Fn(u64) -> Vec<u64>,
    is_entry: &dyn Fn(u64) -> bool,
    target: u64,
    regs: &[Register],
) -> Option<Vec<(Register, Known)>> {
    reaching_values_bounded(
        blocks,
        preds,
        is_entry,
        target,
        regs,
        MAX_REACH_BLOCKS,
        MAX_REACH_DEPTH,
        MAX_WORKLIST_STEPS,
    )
}

#[cfg_attr(not(test), allow(dead_code))]
#[allow(clippy::too_many_arguments)]
fn reaching_values_bounded(
    blocks: &BTreeMap<u64, BasicBlock>,
    preds: &dyn Fn(u64) -> Vec<u64>,
    is_entry: &dyn Fn(u64) -> bool,
    target: u64,
    regs: &[Register],
    max_blocks: usize,
    max_depth: usize,
    max_worklist_steps: usize,
) -> Option<Vec<(Register, Known)>> {
    let input = reaching_state_bounded(
        blocks,
        preds,
        is_entry,
        target,
        max_blocks,
        max_depth,
        max_worklist_steps,
    )?;
    Some(
        regs.iter()
            .map(|&r| {
                let k = input.get(r);
                (
                    r,
                    if k == Known::Unvisited {
                        Known::Unknown
                    } else {
                        k
                    },
                )
            })
            .collect(),
    )
}

/// Registers of `regs` that are a zero-extended 32-bit copy (`mov dst32, src32`) of another
/// register on every path into `target`, as `(dst, src)` full registers.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn reaching_copies(
    blocks: &BTreeMap<u64, BasicBlock>,
    preds: &dyn Fn(u64) -> Vec<u64>,
    is_entry: &dyn Fn(u64) -> bool,
    target: u64,
    regs: &[Register],
) -> Option<Vec<(Register, Register)>> {
    let input = reaching_state_bounded(
        blocks,
        preds,
        is_entry,
        target,
        MAX_REACH_BLOCKS,
        MAX_REACH_DEPTH,
        MAX_WORKLIST_STEPS,
    )?;
    Some(
        regs.iter()
            .filter_map(|&r| match input.aliases[slot(r)?] {
                Alias::Copy32(a) if usize::from(a) < GPR_COUNT => {
                    Some((r.full_register(), Register::RAX + u32::from(a)))
                }
                _ => None,
            })
            .collect(),
    )
}

/// Most distinct constants tracked per register by [`reaching_constant_sets`].
pub(crate) const MAX_CONSTANT_SET: usize = 4;

/// What [`reaching_constant_sets`] knows about one register: the exact constants it holds on the
/// incoming paths (one of them on each path), or nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ConstSet {
    Unvisited,
    /// Sorted, distinct, at most [`MAX_CONSTANT_SET`] values.
    Set(Vec<u64>),
    Unknown,
}

impl ConstSet {
    fn join(&self, other: &ConstSet) -> ConstSet {
        match (self, other) {
            (ConstSet::Unvisited, x) | (x, ConstSet::Unvisited) => x.clone(),
            (ConstSet::Set(a), ConstSet::Set(b)) => {
                let union: BTreeSet<u64> = a.iter().chain(b.iter()).copied().collect();
                if union.len() > MAX_CONSTANT_SET {
                    ConstSet::Unknown
                } else {
                    ConstSet::Set(union.into_iter().collect())
                }
            }
            _ => ConstSet::Unknown,
        }
    }
}

/// Effect of `instr` on the constant sets: only definitions by a constant (`lea r64, [rip+X]`,
/// `mov r, imm`, `xor r32, r32`) and full 64-bit register copies keep a set; every other write
/// of a register (and a call, for the Win64-volatile registers) makes it unknown.
fn transfer_constants(
    info: &mut InstructionInfoFactory,
    instr: &Instruction,
    state: &mut [ConstSet],
) {
    if matches!(
        instr.flow_control(),
        FlowControl::Call | FlowControl::IndirectCall
    ) {
        for (i, set) in state.iter_mut().enumerate() {
            if is_win64_volatile(Register::RAX + i as u32) {
                *set = ConstSet::Unknown;
            }
        }
        return;
    }
    let mut written: Vec<usize> = Vec::new();
    for u in info.info(instr).used_registers() {
        let full = u.register().full_register();
        if full.is_gpr64()
            && matches!(
                u.access(),
                OpAccess::Write
                    | OpAccess::CondWrite
                    | OpAccess::ReadWrite
                    | OpAccess::ReadCondWrite
            )
            && let Some(i) = slot(full)
            && !written.contains(&i)
        {
            written.push(i);
        }
    }
    if written.is_empty() {
        return;
    }
    let dst = (instr.op_count() >= 1 && instr.op0_kind() == OpKind::Register)
        .then(|| instr.op0_register());
    let value: Option<(usize, ConstSet)> = dst.and_then(|dst| {
        let d = slot(dst)?;
        let d64 = dst.is_gpr64();
        let d32 = dst.is_gpr32();
        let imm = (instr.op_count() == 2)
            .then(|| instr.try_immediate(1).ok())
            .flatten();
        let src = (instr.op_count() == 2 && instr.op1_kind() == OpKind::Register)
            .then(|| instr.op1_register());
        let set = match instr.mnemonic() {
            Mnemonic::Lea if d64 && instr.is_ip_rel_memory_operand() => {
                ConstSet::Set(vec![instr.ip_rel_memory_address()])
            }
            Mnemonic::Mov if (d64 || d32) && imm.is_some() => {
                ConstSet::Set(vec![imm? & mask(if d64 { 64 } else { 32 })])
            }
            Mnemonic::Xor if d32 && src == Some(dst) => ConstSet::Set(vec![0]),
            Mnemonic::Mov if d64 && src.is_some_and(|s| s.is_gpr64()) => state[slot(src?)?].clone(),
            _ => return None,
        };
        Some((d, set))
    });
    for i in written {
        state[i] = match &value {
            Some((d, set)) if *d == i => set.clone(),
            _ => ConstSet::Unknown,
        };
    }
}

/// Registers of `regs` that hold one of at most [`MAX_CONSTANT_SET`] exact constants on entry to
/// `target`: each path into `target` defines the register by one of the returned constants
/// (sorted, at least two; a single constant is [`reaching_values`]' business). The typical case is
/// a dispatcher whose table base is loaded by `lea` on every path, from two different tables.
/// Only constant definitions and 64-bit register copies are followed (no arithmetic, no guards),
/// so the set is exact: no path brings another value. `None` as for [`reaching_values`].
pub(crate) fn reaching_constant_sets(
    blocks: &BTreeMap<u64, BasicBlock>,
    preds: &dyn Fn(u64) -> Vec<u64>,
    is_entry: &dyn Fn(u64) -> bool,
    target: u64,
    regs: &[Register],
) -> Option<Vec<(Register, Vec<u64>)>> {
    let BackwardSubgraph { blocks: sub, roots } = collect_backward_subgraph_bounded(
        blocks,
        preds,
        is_entry,
        target,
        MAX_REACH_BLOCKS,
        MAX_REACH_DEPTH,
    )?;
    let unvisited = || vec![ConstSet::Unvisited; GPR_COUNT];
    let mut info = InstructionInfoFactory::new();
    let mut inputs: BTreeMap<u64, Vec<ConstSet>> = sub.iter().map(|&b| (b, unvisited())).collect();
    let mut queue: VecDeque<u64> = VecDeque::new();
    let mut queued: BTreeSet<u64> = BTreeSet::new();
    for b in roots {
        inputs.insert(b, vec![ConstSet::Unknown; GPR_COUNT]);
        if queued.insert(b) {
            queue.push_back(b);
        }
    }
    // Every input only grows (at most `MAX_CONSTANT_SET + 1` changes per register), so the
    // worklist terminates; the step bound is a defensive guard as in `reaching_state_bounded`.
    let mut steps: usize = 0;
    while let Some(b) = queue.pop_front() {
        queued.remove(&b);
        if steps >= MAX_WORKLIST_STEPS {
            return None;
        }
        steps += 1;
        let Some(block) = blocks.get(&b) else {
            continue;
        };
        let mut state = inputs.get(&b).cloned().unwrap_or_else(unvisited);
        for instr in &block.instructions {
            transfer_constants(&mut info, instr, &mut state);
        }
        let mut next_blocks: Vec<u64> = block
            .successors
            .iter()
            .filter_map(|edge| edge.intra_target())
            .collect();
        next_blocks.sort_unstable();
        next_blocks.dedup();
        for next in next_blocks {
            if !sub.contains(&next) || is_entry(next) {
                continue;
            }
            let old = inputs.get(&next).cloned().unwrap_or_else(unvisited);
            let new: Vec<ConstSet> = old.iter().zip(&state).map(|(a, b)| a.join(b)).collect();
            if new == old {
                continue;
            }
            inputs.insert(next, new);
            if queued.insert(next) {
                queue.push_back(next);
            }
        }
    }
    let input = inputs.get(&target)?;
    Some(
        regs.iter()
            .filter_map(|&r| match &input[slot(r)?] {
                ConstSet::Set(values) if values.len() >= 2 => {
                    Some((r.full_register(), values.clone()))
                }
                _ => None,
            })
            .collect(),
    )
}

fn reaching_state_bounded(
    blocks: &BTreeMap<u64, BasicBlock>,
    preds: &dyn Fn(u64) -> Vec<u64>,
    is_entry: &dyn Fn(u64) -> bool,
    target: u64,
    max_blocks: usize,
    max_depth: usize,
    max_worklist_steps: usize,
) -> Option<State> {
    let BackwardSubgraph { blocks: sub, roots } =
        collect_backward_subgraph_bounded(blocks, preds, is_entry, target, max_blocks, max_depth)?;

    let mut info = InstructionInfoFactory::new();
    let mut inputs: BTreeMap<u64, State> = sub.iter().map(|&b| (b, State::unvisited())).collect();
    let mut changes: BTreeMap<u64, u32> = BTreeMap::new();
    let mut queue: VecDeque<u64> = VecDeque::new();
    let mut queued: BTreeSet<u64> = BTreeSet::new();

    // Seed all root blocks in strictly ascending VA order (deterministic worklist start).
    for b in roots {
        inputs.insert(b, State::unknown());
        if queued.insert(b) {
            queue.push_back(b);
        }
    }

    let mut steps: usize = 0;
    while let Some(b) = queue.pop_front() {
        queued.remove(&b);
        if steps >= max_worklist_steps {
            return None;
        }
        steps += 1;
        let Some(block) = blocks.get(&b) else {
            continue;
        };
        let mut state = inputs.get(&b).copied().unwrap_or_else(State::unvisited);
        for instr in &block.instructions {
            transfer(&mut info, instr, &mut state);
        }
        // Sort and deduplicate outgoing intra-function edges so traversal and widening order
        // never depend on how a caller populated `block.successors`.
        let mut intra_edges: Vec<Edge> = block
            .successors
            .iter()
            .copied()
            .filter(|edge| edge.intra_target().is_some())
            .collect();
        intra_edges.sort_unstable();
        intra_edges.dedup();

        for edge in intra_edges {
            let Some(next) = edge.intra_target() else {
                continue;
            };
            if !sub.contains(&next) || is_entry(next) {
                continue;
            }
            let mut out = state;
            if block.terminator == Terminator::Branch {
                refine_edge(&mut info, block, edge, &mut out);
            }
            let old = inputs.get(&next).copied().unwrap_or_else(State::unvisited);
            let mut new = old.join(&out);
            if new == old {
                continue;
            }
            let n = changes.entry(next).or_insert(0);
            *n += 1;
            if *n > WIDEN_AFTER {
                for (i, k) in new.ranges.iter_mut().enumerate() {
                    if *k != old.ranges[i] {
                        *k = Known::Unknown;
                    }
                }
                for (i, k) in new.u32_ranges.iter_mut().enumerate() {
                    if *k != old.u32_ranges[i] {
                        *k = Known::Unknown;
                    }
                }
                for (i, a) in new.aliases.iter_mut().enumerate() {
                    if *a != old.aliases[i] {
                        *a = Alias::None;
                    }
                }
            }
            inputs.insert(next, new);
            if queued.insert(next) {
                queue.push_back(next);
            }
        }
    }

    Some(inputs.get(&target).copied().unwrap_or_else(State::unvisited))
}

fn writes(info: &mut InstructionInfoFactory, instr: &Instruction, reg: Register) -> bool {
    let reg = reg.full_register();
    info.info(instr).used_registers().iter().any(|u| {
        u.register().full_register() == reg
            && matches!(
                u.access(),
                OpAccess::Write
                    | OpAccess::CondWrite
                    | OpAccess::ReadWrite
                    | OpAccess::ReadCondWrite
            )
    })
}

/// Register `i` is overwritten: it is no longer a copy of anything, and the registers that were
/// copies of it (`j = zext32(i)` or `j = i`) and survive the write (`keeps(j)`) stay copies of
/// each other within their own width class. The first survivor takes the place of `i` as their
/// source; the rest lose their relation.
fn unlink(state: &mut State, i: usize, keeps: &dyn Fn(usize) -> bool) {
    state.aliases[i] = Alias::None;
    let mut rep32: Option<u8> = None;
    let mut rep64: Option<u8> = None;
    for j in 0..GPR_COUNT {
        if j == i {
            continue;
        }
        match state.aliases[j] {
            Alias::Copy32(src) if usize::from(src) == i => {
                if !keeps(j) {
                    state.aliases[j] = Alias::None;
                } else if let Some(r) = rep32 {
                    state.aliases[j] = Alias::Copy32(r);
                } else {
                    rep32 = Some(j as u8);
                    state.aliases[j] = Alias::None;
                }
            }
            Alias::Copy64(src) if usize::from(src) == i => {
                if !keeps(j) {
                    state.aliases[j] = Alias::None;
                } else if let Some(r) = rep64 {
                    state.aliases[j] = Alias::Copy64(r);
                } else {
                    rep64 = Some(j as u8);
                    state.aliases[j] = Alias::None;
                }
            }
            _ => {}
        }
    }
}

/// Detect whether `instr` makes its destination a 32-bit zero-extended copy or 64-bit full copy
/// of another GPR (`mov r32, r32`, `mov r64, r64`, or `lea r32/r64, [src64 + 0]`).
fn copy_relation(instr: &Instruction) -> Option<(Register, Register, bool)> {
    if instr.op_count() != 2 || instr.op0_kind() != OpKind::Register {
        return None;
    }
    let dst = instr.op0_register();
    if !(dst.is_gpr32() || dst.is_gpr64()) {
        return None;
    }
    let is_64 = dst.is_gpr64();
    let src = match instr.mnemonic() {
        Mnemonic::Mov if instr.op1_kind() == OpKind::Register => {
            let s = instr.op1_register();
            if (is_64 && s.is_gpr64()) || (!is_64 && s.is_gpr32()) {
                s
            } else {
                return None;
            }
        }
        Mnemonic::Lea
            if instr.op1_kind() == OpKind::Memory
                && !instr.is_ip_rel_memory_operand()
                && !matches!(instr.memory_segment(), Register::FS | Register::GS)
                && instr.memory_base().is_gpr64()
                && instr.memory_index() == Register::None
                && instr.memory_displacement64() == 0 =>
        {
            instr.memory_base()
        }
        _ => return None,
    };
    let dst_full = dst.full_register();
    let src_full = src.full_register();
    (dst_full != src_full).then_some((dst_full, src_full, is_64))
}

fn transfer(info: &mut InstructionInfoFactory, instr: &Instruction, state: &mut State) {
    if matches!(
        instr.flow_control(),
        FlowControl::Call | FlowControl::IndirectCall
    ) {
        for i in 0..GPR_COUNT {
            if is_win64_volatile(Register::RAX + i as u32) {
                state.ranges[i] = Known::Unknown;
                state.u32_ranges[i] = Known::Unknown;
            }
        }
        for i in 0..GPR_COUNT {
            if is_win64_volatile(Register::RAX + i as u32) {
                unlink(state, i, &|j| !is_win64_volatile(Register::RAX + j as u32));
            }
        }
        return;
    }
    let written: Vec<Register> = {
        let used = info.info(instr).used_registers();
        let mut out = Vec::new();
        for u in used {
            let full = u.register().full_register();
            if full.is_gpr64()
                && matches!(
                    u.access(),
                    OpAccess::Write
                        | OpAccess::CondWrite
                        | OpAccess::ReadWrite
                        | OpAccess::ReadCondWrite
                )
                && !out.contains(&full)
            {
                out.push(full);
            }
        }
        out
    };
    if written.is_empty() {
        return;
    }
    let copy = copy_relation(instr);
    let written_slots: Vec<usize> = written.iter().filter_map(|r| slot(*r)).collect();
    for &i in &written_slots {
        unlink(state, i, &|j| !written_slots.contains(&j));
    }
    if let Some((dst, src, is_64)) = copy
        && let (Some(d), Some(sr)) = (slot(dst), slot(src))
    {
        state.aliases[d] = if is_64 {
            Alias::Copy64(sr as u8)
        } else {
            Alias::Copy32(sr as u8)
        };
    }
    // The one register an exactly modelled instruction writes (operand 0).
    let modelled = model(instr, state);
    for reg in written {
        let (full_v, u32_v) = match modelled {
            Some((dst, fv, uv)) if dst == reg => (fv, uv),
            _ => (Known::Unknown, Known::Unknown),
        };
        state.set(reg, full_v, u32_v);
    }
}

/// Shift `base` by signed `delta` within `[0, mask(width)]`.
fn shift_range(base: Known, delta: i128, width: u32) -> Known {
    let fallback = if width == 32 {
        Known::Range {
            lo: 0,
            hi: mask(32),
        }
    } else {
        Known::Unknown
    };
    let Known::Range { lo, hi } = base else {
        return fallback;
    };
    let nlo = i128::from(lo) + delta;
    let nhi = i128::from(hi) + delta;
    if nlo < 0 || nhi > i128::from(mask(width)) {
        fallback
    } else {
        Known::Range {
            lo: nlo as u64,
            hi: nhi as u64,
        }
    }
}

/// Exact effect of `instr` on its destination register (operand 0), if modelled:
/// `(full_reg, full_64_known, u32_known)`.
fn model(instr: &Instruction, state: &State) -> Option<(Register, Known, Known)> {
    if instr.mnemonic() == Mnemonic::Cdqe {
        let u32_v = state.get_u32(Register::RAX);
        let full_v = match u32_v {
            Known::Range { lo, hi } if hi <= 0x7fff_ffff => Known::Range { lo, hi },
            _ => Known::Unknown,
        };
        return Some((Register::RAX, full_v, u32_v));
    }
    if instr.op_count() < 1 || instr.op0_kind() != OpKind::Register {
        return None;
    }
    let dst = instr.op0_register();
    let full = dst.full_register();
    if !full.is_gpr64() {
        return None;
    }
    let d64 = dst.is_gpr64();
    let d32 = dst.is_gpr32();
    if !(d64 || d32) {
        // 8/16-bit partial writes keep the other bits: not modelled.
        return None;
    }
    let width = if d64 { 64 } else { 32 };
    let src_reg = (instr.op_count() == 2 && instr.op1_kind() == OpKind::Register)
        .then(|| instr.op1_register());
    let imm = (instr.op_count() == 2)
        .then(|| instr.try_immediate(1).ok())
        .flatten();
    let cur = state.get(full);
    let cur_u32 = state.get_u32(full);
    let (full_v, u32_v) = match instr.mnemonic() {
        Mnemonic::Lea if d64 && instr.is_ip_rel_memory_operand() => {
            let k = Known::exact(instr.ip_rel_memory_address());
            (k, low32_of(k))
        }
        Mnemonic::Lea
            if instr.op_count() == 2
                && instr.op1_kind() == OpKind::Memory
                && !instr.is_ip_rel_memory_operand()
                && !matches!(instr.memory_segment(), Register::FS | Register::GS)
                && instr.memory_base().is_gpr64()
                && instr.memory_index() == Register::None =>
        {
            let src = instr.memory_base();
            let disp = instr.memory_displacement64() as i64;
            if d32 {
                let r = shift_range(state.get_u32(src), i128::from(disp as i32), 32);
                (r, r)
            } else {
                let fv = shift_range(state.get(src), i128::from(disp), 64);
                let uv = match state.get_u32(src) {
                    Known::Range { .. } => {
                        let s = shift_range(state.get_u32(src), i128::from(disp as i32), 32);
                        if s == (Known::Range { lo: 0, hi: mask(32) }) {
                            low32_of(fv)
                        } else {
                            s
                        }
                    }
                    _ => low32_of(fv),
                };
                (fv, uv)
            }
        }
        Mnemonic::Mov if imm.is_some() => {
            let k = Known::exact(imm? & mask(width));
            (k, low32_of(k))
        }
        Mnemonic::Xor if src_reg == Some(dst) => (Known::exact(0), Known::exact(0)),
        Mnemonic::Mov if src_reg.is_some_and(|s| s.is_gpr64()) && d64 => {
            let s = src_reg?;
            (state.get(s), state.get_u32(s))
        }
        Mnemonic::Mov if src_reg.is_some_and(|s| s.is_gpr32()) && d32 => {
            let s = state.get_u32(src_reg?);
            let r = if s.within_u32() {
                s
            } else {
                Known::Range {
                    lo: 0,
                    hi: mask(32),
                }
            };
            (r, r)
        }
        Mnemonic::Mov if instr.op_count() == 2 && instr.op1_kind() == OpKind::Memory && d32 => {
            let r = Known::Range {
                lo: 0,
                hi: mask(32),
            };
            (r, r)
        }
        Mnemonic::Movzx if instr.op_count() == 2 => {
            let bits = match instr.op1_kind() {
                OpKind::Register => {
                    let s = instr.op1_register();
                    if is_high_byte(s) {
                        8
                    } else {
                        s.size() as u32 * 8
                    }
                }
                OpKind::Memory => instr.memory_size().size() as u32 * 8,
                _ => return None,
            };
            let m = mask(bits);
            let r = match (instr.op1_kind(), src_reg.map(|s| state.get_u32(s))) {
                (OpKind::Register, Some(Known::Range { lo, hi }))
                    if hi <= m && !src_reg.is_some_and(is_high_byte) =>
                {
                    Known::Range { lo, hi }
                }
                _ => Known::Range { lo: 0, hi: m },
            };
            (r, r)
        }
        Mnemonic::Movsxd if d64 && src_reg.is_some_and(|s| s.is_gpr32()) => {
            let s = state.get_u32(src_reg?);
            match s {
                Known::Range { lo, hi } if hi <= 0x7fff_ffff => {
                    let r = Known::Range { lo, hi };
                    (r, r)
                }
                Known::Range { lo, hi } => (Known::Unknown, Known::Range { lo, hi }),
                _ => (Known::Unknown, Known::Unknown),
            }
        }
        Mnemonic::And if imm.is_some() => {
            let raw = imm?;
            if d32 {
                let m = raw & mask(32);
                let r = match cur_u32 {
                    Known::Range { hi, .. } => Known::Range {
                        lo: 0,
                        hi: hi.min(m),
                    },
                    _ => Known::Range { lo: 0, hi: m },
                };
                (r, r)
            } else {
                let top = if (raw as i64) < 0 { None } else { Some(raw) };
                let fv = match (top, cur) {
                    (Some(m), Known::Range { hi, .. }) => Known::Range {
                        lo: 0,
                        hi: hi.min(m),
                    },
                    (Some(m), _) => Known::Range { lo: 0, hi: m },
                    (None, _) => Known::Unknown,
                };
                let m32 = raw & mask(32);
                let uv = match cur_u32 {
                    Known::Range { hi, .. } => Known::Range {
                        lo: 0,
                        hi: hi.min(m32),
                    },
                    _ => Known::Range { lo: 0, hi: m32 },
                };
                (fv, uv)
            }
        }
        Mnemonic::Shr if imm.is_some() => {
            if d32 {
                let s = (imm? & 31) as u32;
                let r = match cur_u32 {
                    Known::Range { lo, hi } => Known::Range {
                        lo: lo >> s,
                        hi: hi >> s,
                    },
                    _ => Known::Range {
                        lo: 0,
                        hi: mask(32) >> s,
                    },
                };
                (r, r)
            } else {
                let s = (imm? & 63) as u32;
                let fv = match cur {
                    Known::Range { lo, hi } => Known::Range {
                        lo: lo >> s,
                        hi: hi >> s,
                    },
                    _ => Known::Range {
                        lo: 0,
                        hi: mask(64) >> s,
                    },
                };
                (fv, low32_of(fv))
            }
        }
        Mnemonic::Add | Mnemonic::Sub | Mnemonic::Inc | Mnemonic::Dec => {
            let delta: i128 = match instr.mnemonic() {
                Mnemonic::Inc => 1,
                Mnemonic::Dec => -1,
                _ => {
                    let v = imm?;
                    let signed = if width == 64 {
                        i128::from(v as i64)
                    } else {
                        i128::from(v as u32 as i32)
                    };
                    if instr.mnemonic() == Mnemonic::Sub {
                        -signed
                    } else {
                        signed
                    }
                }
            };
            if d32 {
                let r = shift_range(cur_u32, delta, 32);
                (r, r)
            } else {
                let fv = shift_range(cur, delta, 64);
                let uv = match cur_u32 {
                    Known::Range { .. }
                        if (i128::from(i32::MIN)..=i128::from(i32::MAX)).contains(&delta) =>
                    {
                        let s = shift_range(cur_u32, delta, 32);
                        if s == (Known::Range { lo: 0, hi: mask(32) }) {
                            low32_of(fv)
                        } else {
                            s
                        }
                    }
                    _ => low32_of(fv),
                };
                (fv, uv)
            }
        }
        // Any other unconditional write to a 32-bit register zero-extends: bounded by 2^32.
        _ if d32 => {
            let r = Known::Range {
                lo: 0,
                hi: mask(32),
            };
            (r, r)
        }
        _ => return None,
    };
    Some((full, full_v, u32_v))
}

/// Extract the `[nlo, nhi]` range implied for the low `bits` bits of `reg` by `instr` + `jcc`
/// on the `taken` side, given its current range `[lo, hi]`.
fn edge_fact(
    instr: &Instruction,
    jcc: Mnemonic,
    taken: bool,
    reg: Register,
    bits: u32,
    lo: u64,
    hi: u64,
) -> Option<(u64, u64)> {
    let m = mask(bits);
    match instr.mnemonic() {
        Mnemonic::Cmp => {
            let raw = instr.try_immediate(1).ok()?;
            let imm = raw & m;
            unsigned_fact_bits(jcc, taken, imm, lo, hi, bits)
        }
        Mnemonic::Test
            if instr.op_count() == 2
                && instr.op1_kind() == OpKind::Register
                && instr.op1_register() == reg =>
        {
            match (jcc, taken) {
                (Mnemonic::Je, true) | (Mnemonic::Jne, false) => Some((0, 0)),
                (Mnemonic::Je, false) | (Mnemonic::Jne, true) if lo == 0 && hi > 0 => Some((1, hi)),
                (Mnemonic::Js, false)
                | (Mnemonic::Jns, true)
                | (Mnemonic::Jl, false)
                | (Mnemonic::Jge, true) => Some((0, mask(bits.saturating_sub(1)))),
                _ => None,
            }
        }
        Mnemonic::Test if instr.op_count() == 2 && instr.try_immediate(1).is_ok() => {
            let imm_u = instr.try_immediate(1).ok()? & m;
            match (jcc, taken) {
                (Mnemonic::Je, true) | (Mnemonic::Jne, false) => Some((0, (!imm_u) & m)),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Refine `state` for the edge `edge` out of `block` (which ends in a conditional branch).
fn refine_edge(
    info: &mut InstructionInfoFactory,
    block: &BasicBlock,
    edge: Edge,
    state: &mut State,
) {
    let Some(jcc) = block.instructions.last() else {
        return;
    };
    let taken = match edge {
        Edge::Taken(_) => true,
        Edge::Fallthrough(_) => false,
        _ => return,
    };
    // Taken and fallthrough to the same block: no refinement.
    if block
        .successors
        .iter()
        .filter(|e| e.intra_target() == edge.intra_target())
        .count()
        > 1
    {
        return;
    }
    let n = block.instructions.len();
    for i in (0..n.saturating_sub(1)).rev() {
        let instr = &block.instructions[i];
        if matches!(
            instr.flow_control(),
            FlowControl::Call | FlowControl::IndirectCall
        ) {
            return;
        }
        if instr.rflags_modified() == 0 {
            continue;
        }
        if instr.op_count() < 1 || instr.op0_kind() != OpKind::Register {
            return;
        }
        let reg = instr.op0_register();
        if !reg.is_gpr()
            || is_high_byte(reg)
            || block.instructions[i + 1..n - 1]
                .iter()
                .any(|k| writes(info, k, reg))
        {
            return;
        }
        let bits = reg.size() as u32 * 8;
        if !matches!(bits, 8 | 16 | 32 | 64) {
            return;
        }
        let full = reg.full_register();
        let Some(idx) = slot(full) else {
            return;
        };
        let cur64 = state.ranges[idx];
        let cur32 = state.u32_ranges[idx];
        match bits {
            64 => {
                let (lo, hi) = match cur64 {
                    Known::Range { lo, hi } => (lo, hi),
                    _ => (0, u64::MAX),
                };
                if let Some((nlo, nhi)) = edge_fact(instr, jcc.mnemonic(), taken, reg, 64, lo, hi) {
                    let nlo = nlo.max(lo);
                    let nhi = nhi.min(hi);
                    if nlo <= nhi {
                        let new64 = Known::Range { lo: nlo, hi: nhi };
                        state.ranges[idx] = new64;
                        if let Known::Range { lo: l32, hi: h32 } = low32_of(new64) {
                            let (ulo, uhi) = match cur32 {
                                Known::Range { lo, hi } => (lo, hi),
                                _ => (0, mask(32)),
                            };
                            let nlo32 = l32.max(ulo);
                            let nhi32 = h32.min(uhi);
                            if nlo32 <= nhi32 {
                                state.u32_ranges[idx] = Known::Range {
                                    lo: nlo32,
                                    hi: nhi32,
                                };
                            }
                        }
                    }
                }
            }
            32 => {
                let (ulo, uhi) = match cur32 {
                    Known::Range { lo, hi } => (lo, hi),
                    _ => (0, mask(32)),
                };
                if let Some((nlo, nhi)) = edge_fact(instr, jcc.mnemonic(), taken, reg, 32, ulo, uhi)
                {
                    let nlo = nlo.max(ulo);
                    let nhi = nhi.min(uhi).min(mask(32));
                    if nlo <= nhi {
                        state.u32_ranges[idx] = Known::Range { lo: nlo, hi: nhi };
                    }
                }
                if cur64.within_u32() {
                    let Known::Range { lo, hi } = cur64 else {
                        return;
                    };
                    if let Some((nlo, nhi)) =
                        edge_fact(instr, jcc.mnemonic(), taken, reg, 32, lo, hi)
                    {
                        let nlo = nlo.max(lo);
                        let nhi = nhi.min(hi).min(mask(32));
                        if nlo <= nhi {
                            state.ranges[idx] = Known::Range { lo: nlo, hi: nhi };
                            state.u32_ranges[idx] = Known::Range { lo: nlo, hi: nhi };
                        }
                    }
                }
            }
            8 | 16 => {
                let m = mask(bits);
                if let Known::Range { lo: ulo, hi: uhi } = cur32
                    && uhi <= m
                    && let Some((nlo, nhi)) =
                        edge_fact(instr, jcc.mnemonic(), taken, reg, bits, ulo, uhi)
                {
                    let nlo = nlo.max(ulo);
                    let nhi = nhi.min(uhi).min(m);
                    if nlo <= nhi {
                        state.u32_ranges[idx] = Known::Range { lo: nlo, hi: nhi };
                    }
                }
                if let Known::Range { lo, hi } = cur64
                    && hi <= m
                    && let Some((nlo, nhi)) =
                        edge_fact(instr, jcc.mnemonic(), taken, reg, bits, lo, hi)
                {
                    let nlo = nlo.max(lo);
                    let nhi = nhi.min(hi).min(m);
                    if nlo <= nhi {
                        state.ranges[idx] = Known::Range { lo: nlo, hi: nhi };
                        state.u32_ranges[idx] = Known::Range { lo: nlo, hi: nhi };
                    }
                }
            }
            _ => {}
        }
        return;
    }
}

/// `[lo, hi]` implied for the compared register by `cmp reg, imm` + `jcc` on the `taken` side,
/// given its current range `[cur_lo, cur_hi]` (signed conditions only when that is non-negative
/// below `2^(bits-1)`).
#[cfg_attr(not(test), allow(dead_code))]
fn unsigned_fact(
    jcc: Mnemonic,
    taken: bool,
    imm: u64,
    cur_lo: u64,
    cur_hi: u64,
) -> Option<(u64, u64)> {
    unsigned_fact_bits(jcc, taken, imm, cur_lo, cur_hi, 32)
}

fn unsigned_fact_bits(
    jcc: Mnemonic,
    taken: bool,
    imm: u64,
    cur_lo: u64,
    cur_hi: u64,
    bits: u32,
) -> Option<(u64, u64)> {
    let sign_max = mask(bits.saturating_sub(1));
    let nonneg = cur_hi <= sign_max && imm <= sign_max;
    let (op, holds) = match jcc {
        Mnemonic::Ja => ("gt", taken),
        Mnemonic::Jae => ("ge", taken),
        Mnemonic::Jb => ("ge", !taken),
        Mnemonic::Jbe => ("gt", !taken),
        Mnemonic::Jg if nonneg => ("gt", taken),
        Mnemonic::Jge if nonneg => ("ge", taken),
        Mnemonic::Jl if nonneg => ("ge", !taken),
        Mnemonic::Jle if nonneg => ("gt", !taken),
        Mnemonic::Je if taken => return Some((imm, imm)),
        Mnemonic::Jne if !taken => return Some((imm, imm)),
        // `reg != imm` excludes one value: it tightens the range only at an end of it (e.g.
        // `and r10d, 0xf; cmp r10, 0xf; je default` leaves `[0, 14]` on the fallthrough).
        Mnemonic::Je | Mnemonic::Jne => {
            return if cur_lo >= cur_hi {
                None
            } else if imm == cur_hi {
                Some((cur_lo, cur_hi - 1))
            } else if imm == cur_lo {
                Some((cur_lo + 1, cur_hi))
            } else {
                None
            };
        }
        _ => return None,
    };
    // holds: reg > imm (gt) / reg >= imm (ge); otherwise the negation.
    match (op, holds) {
        ("gt", true) => imm.checked_add(1).map(|l| (l, u64::MAX)),
        ("gt", false) => Some((0, imm)),
        ("ge", true) => Some((imm, u64::MAX)),
        ("ge", false) => imm.checked_sub(1).map(|h| (0, h)),
        _ => None,
    }
}

/// Synthetic `cmp reg, hi; ja` (plus `cmp reg, lo; jb` when `lo > 0`) at `width` bits (32 or 64).
fn synthesize_cmp_bounds(reg: Register, lo: u64, hi: u64, is_64: bool) -> Vec<(Instruction, bool)> {
    let n = reg.full_register().number() as u8;
    let cmp = |imm: u64| -> Option<Instruction> {
        let mut bytes = Vec::with_capacity(7);
        if is_64 {
            bytes.push(0x48 | (n >> 3));
        } else if n >= 8 {
            bytes.push(0x41);
        }
        bytes.push(0x81);
        bytes.push(0xF8 | (n & 7));
        bytes.extend_from_slice(&(imm as u32).to_le_bytes());
        decode(&bytes, 0)
    };
    let jcc = |opcode: u8| decode(&[0x0F, opcode, 0, 0, 0, 0], 0);
    let mut out = Vec::new();
    match (cmp(hi), jcc(0x87)) {
        (Some(c), Some(j)) => {
            out.push((c, false));
            out.push((j, true));
        }
        _ => return Vec::new(),
    }
    if lo > 0
        && let (Some(c), Some(j)) = (cmp(lo), jcc(0x82))
    {
        out.push((c, false));
        out.push((j, true));
    }
    out
}

/// Synthetic instructions that state full 64-bit `known` for `reg` in a form the jump-table
/// slicer reads: `lea reg, [rip+v]` for a single in-image address, else `cmp r64, hi; ja` when
/// `hi <= i32::MAX` (bounding both 64-bit and 32-bit reads) or `cmp r32, hi; ja` when
/// `hi <= u32::MAX`, plus the matching lower-bound `cmp reg, lo; jb` when `lo > 0`.
///
/// Returns the instructions with whether each is a not-taken conditional branch; empty when
/// nothing useful can be stated.
pub(crate) fn synthesize(
    reg: Register,
    known: Known,
    image_contains: &dyn Fn(u64) -> bool,
) -> Vec<(Instruction, bool)> {
    let Known::Range { lo, hi } = known else {
        return Vec::new();
    };
    let n = reg.full_register().number() as u8;
    if lo == hi && image_contains(lo) {
        let bytes = [
            0x48 | ((n >> 3) << 2),
            0x8D,
            ((n & 7) << 3) | 0x05,
            0,
            0,
            0,
            0,
        ];
        return decode(&bytes, lo.wrapping_sub(7))
            .filter(|i| i.op0_register() == reg.full_register() && i.ip_rel_memory_address() == lo)
            .map(|i| vec![(i, false)])
            .unwrap_or_default();
    }
    if hi > u64::from(u32::MAX) {
        return Vec::new();
    }
    // `cmp r64, imm32` sign-extends `imm32`, so it encodes positive unsigned bounds up to i32::MAX
    // and bounds raw 64-bit index reads as well as 32-bit extensions. Larger values up to u32::MAX
    // fall back to 32-bit `cmp r32, imm32`.
    let is_64 = hi <= i32::MAX as u64;
    synthesize_cmp_bounds(reg, lo, hi, is_64)
}

/// Synthetic 32-bit `cmp r32, hi; ja` (plus `cmp r32, lo; jb` when `lo > 0`) for a register whose
/// low 32 bits are bounded even when its upper 32 bits are unknown. Safely bounds `movsxd`/`mov r32`
/// reads without falsely bounding raw 64-bit index reads.
pub(crate) fn synthesize_u32(reg: Register, known: Known) -> Vec<(Instruction, bool)> {
    let Known::Range { lo, hi } = known else {
        return Vec::new();
    };
    if hi > u64::from(u32::MAX) || (lo == 0 && hi == u64::from(u32::MAX)) {
        return Vec::new();
    }
    synthesize_cmp_bounds(reg, lo, hi, false)
}

/// Synthetic `mov dst32, src32` stating that `dst` is a zero-extended copy of `src`.
pub(crate) fn synthesize_copy(dst: Register, src: Register) -> Option<(Instruction, bool)> {
    let d = dst.full_register().number() as u8;
    let s = src.full_register().number() as u8;
    let rex = 0x40 | ((s >> 3) << 2) | (d >> 3);
    let bytes = [rex, 0x89, 0xC0 | ((s & 7) << 3) | (d & 7)];
    decode(&bytes, 0)
        .filter(|i| {
            i.op0_register().is_gpr32()
                && i.op0_register().full_register() == dst.full_register()
                && i.op1_register().full_register() == src.full_register()
        })
        .map(|i| (i, false))
}

/// Synthetic `mov dst64, src64` stating that `dst` is a full 64-bit copy of `src`.
pub(crate) fn synthesize_copy64(dst: Register, src: Register) -> Option<(Instruction, bool)> {
    let d = dst.full_register().number() as u8;
    let s = src.full_register().number() as u8;
    let rex = 0x48 | ((s >> 3) << 2) | (d >> 3);
    let bytes = [rex, 0x89, 0xC0 | ((s & 7) << 3) | (d & 7)];
    decode(&bytes, 0)
        .filter(|i| {
            i.op0_register().is_gpr64()
                && i.op0_register().full_register() == dst.full_register()
                && i.op1_register().full_register() == src.full_register()
        })
        .map(|i| (i, false))
}

fn decode(bytes: &[u8], ip: u64) -> Option<Instruction> {
    let mut decoder = Decoder::with_ip(64, bytes, ip, DecoderOptions::NONE);
    let instr = decoder.decode();
    (!instr.is_invalid() && instr.len() == bytes.len()).then_some(instr)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode_seq(ip: u64, bytes: &[u8]) -> Vec<Instruction> {
        let mut decoder = Decoder::with_ip(64, bytes, ip, DecoderOptions::NONE);
        let mut out = Vec::new();
        while decoder.can_decode() {
            let instr = decoder.decode();
            assert!(!instr.is_invalid(), "invalid test instruction at {ip:#x}");
            out.push(instr);
        }
        out
    }

    /// Encode `lea reg64, [rip + disp32]` targeting `target_va` from `ip` (7 bytes).
    fn encode_lea_rip(reg: Register, ip: u64, target_va: u64) -> [u8; 7] {
        let n = reg.full_register().number() as u8;
        let delta = (target_va as i128) - (ip.wrapping_add(7) as i128);
        let disp = i32::try_from(delta).expect("RIP-relative displacement must fit in i32");
        let [d0, d1, d2, d3] = disp.to_le_bytes();
        [
            0x48 | ((n >> 3) << 2),
            0x8D,
            ((n & 7) << 3) | 0x05,
            d0,
            d1,
            d2,
            d3,
        ]
    }

    fn make_block(
        start: u64,
        bytes: &[u8],
        terminator: Terminator,
        successors: Vec<Edge>,
    ) -> BasicBlock {
        let instructions = decode_seq(start, bytes);
        let end = instructions
            .last()
            .map_or(start, Instruction::next_ip);
        BasicBlock {
            start,
            end,
            instructions,
            terminator,
            successors,
            calls: Vec::new(),
            diagnostics: Vec::new(),
        }
    }

    fn empty_blocks(addrs: &[u64]) -> BTreeMap<u64, BasicBlock> {
        addrs.iter().map(|&a| (a, BasicBlock::new(a))).collect()
    }

    #[test]
    fn not_equal_edge_trims_an_end_of_the_range() {
        // `cmp r, 15; je` fallthrough / `jne` taken on [0, 15]: [0, 14].
        assert_eq!(unsigned_fact(Mnemonic::Je, false, 15, 0, 15), Some((0, 14)));
        assert_eq!(unsigned_fact(Mnemonic::Jne, true, 15, 0, 15), Some((0, 14)));
        // Excluding the low end: [1, 15].
        assert_eq!(unsigned_fact(Mnemonic::Je, false, 0, 0, 15), Some((1, 15)));
        assert_eq!(unsigned_fact(Mnemonic::Jne, true, 0, 0, 15), Some((1, 15)));
        // An interior value or one outside the range excludes nothing expressible.
        assert_eq!(unsigned_fact(Mnemonic::Je, false, 7, 0, 15), None);
        assert_eq!(unsigned_fact(Mnemonic::Je, false, 16, 0, 15), None);
        // A single-value range is never emptied.
        assert_eq!(unsigned_fact(Mnemonic::Je, false, 3, 3, 3), None);
        // The equal sides are unchanged.
        assert_eq!(unsigned_fact(Mnemonic::Je, true, 15, 0, 15), Some((15, 15)));
        assert_eq!(
            unsigned_fact(Mnemonic::Jne, false, 15, 0, 15),
            Some((15, 15))
        );
    }

    #[test]
    fn backward_subgraph_is_sorted_and_invariant_to_pred_order_and_duplicates() {
        // Diamond + back-edge + unrelated dead block 0x9000:
        //   0x1000 (entry) -> 0x2000, 0x3000 -> 0x4000 (target)
        //   0x4000 -> 0x2000 (cycle), 0x4000 -> 0x4000 (self-loop)
        let blocks = empty_blocks(&[0x1000, 0x2000, 0x3000, 0x4000, 0x9000]);
        let is_entry = |b: u64| b == 0x1000;

        let preds_fwd = |b: u64| match b {
            0x2000 => vec![0x1000, 0x4000],
            0x3000 => vec![0x1000],
            0x4000 => vec![0x2000, 0x3000, 0x4000],
            _ => Vec::new(),
        };
        let preds_rev_dup = |b: u64| match b {
            0x2000 => vec![0x4000, 0x1000, 0x4000, 0x1000],
            0x3000 => vec![0x1000, 0x1000],
            0x4000 => vec![0x4000, 0x3000, 0x2000, 0x3000, 0x4000],
            _ => Vec::new(),
        };

        let a = collect_backward_subgraph_bounded(
            &blocks,
            &preds_fwd,
            &is_entry,
            0x4000,
            MAX_REACH_BLOCKS,
            MAX_REACH_DEPTH,
        )
        .expect("forward subgraph");
        let b = collect_backward_subgraph_bounded(
            &blocks,
            &preds_rev_dup,
            &is_entry,
            0x4000,
            MAX_REACH_BLOCKS,
            MAX_REACH_DEPTH,
        )
        .expect("reversed+duplicated subgraph");

        let expected: BTreeSet<u64> = [0x1000, 0x2000, 0x3000, 0x4000].into_iter().collect();
        assert_eq!(a.blocks, expected);
        assert_eq!(b.blocks, expected);
        assert_eq!(a.roots, BTreeSet::from([0x1000]));
        assert_eq!(b.roots, BTreeSet::from([0x1000]));

        // Confirm strictly ascending deterministic order.
        let ordered: Vec<u64> = a.blocks.iter().copied().collect();
        assert_eq!(ordered, vec![0x1000, 0x2000, 0x3000, 0x4000]);
    }

    #[test]
    fn backward_subgraph_handles_self_loop_and_mutual_cycles_at_tight_bounds() {
        // Isolated self-loop on target: must succeed even at max_blocks=1, max_depth=0.
        let self_blocks = empty_blocks(&[0x2000]);
        let self_preds = |b: u64| if b == 0x2000 { vec![0x2000, 0x2000] } else { Vec::new() };
        let sub = collect_backward_subgraph_bounded(
            &self_blocks,
            &self_preds,
            &|_| false,
            0x2000,
            1,
            0,
        )
        .expect("self-loop must not exceed tight block/depth bounds");
        assert_eq!(sub.blocks, BTreeSet::from([0x2000]));
        assert!(sub.roots.is_empty());

        // Mutual 2-cycle 0x2000 <-> 0x3000: must succeed at max_blocks=2, max_depth=1.
        let cycle_blocks = empty_blocks(&[0x2000, 0x3000]);
        let cycle_preds = |b: u64| match b {
            0x2000 => vec![0x3000, 0x2000],
            0x3000 => vec![0x2000, 0x3000],
            _ => Vec::new(),
        };
        let sub = collect_backward_subgraph_bounded(
            &cycle_blocks,
            &cycle_preds,
            &|_| false,
            0x2000,
            2,
            1,
        )
        .expect("2-cycle must not loop or falsely trigger bounds");
        assert_eq!(sub.blocks, BTreeSet::from([0x2000, 0x3000]));
        assert!(sub.roots.is_empty());
    }

    #[test]
    fn backward_subgraph_enforces_block_depth_and_missing_block_guards() {
        // Linear chain: 0x1000 -> 0x1010 -> 0x1020 -> 0x1030 -> 0x1040 (depth 4, 5 blocks).
        let chain = [0x1000, 0x1010, 0x1020, 0x1030, 0x1040];
        let blocks = empty_blocks(&chain);
        let preds = |b: u64| match b {
            0x1010 => vec![0x1000],
            0x1020 => vec![0x1010],
            0x1030 => vec![0x1020],
            0x1040 => vec![0x1030],
            _ => Vec::new(),
        };
        let is_entry = |b: u64| b == 0x1000;

        // Exact depth & block limits succeed.
        let ok = collect_backward_subgraph_bounded(&blocks, &preds, &is_entry, 0x1040, 5, 4)
            .expect("exact limits should succeed");
        assert_eq!(ok.blocks, chain.into_iter().collect());

        // Depth limit 3 rejects the 4-hop chain.
        assert!(
            collect_backward_subgraph_bounded(&blocks, &preds, &is_entry, 0x1040, 5, 3).is_none()
        );
        // Block limit 4 rejects the 5-block chain.
        assert!(
            collect_backward_subgraph_bounded(&blocks, &preds, &is_entry, 0x1040, 4, 10).is_none()
        );
        // Zero block budget rejects immediately.
        assert!(
            collect_backward_subgraph_bounded(&blocks, &preds, &is_entry, 0x1040, 0, 10).is_none()
        );
        // Missing target block returns None.
        assert!(
            collect_backward_subgraph_bounded(&blocks, &preds, &is_entry, 0xDEAD, 10, 10).is_none()
        );
        // Missing predecessor block returns None.
        let broken_preds = |b: u64| if b == 0x1040 { vec![0xBEEF] } else { Vec::new() };
        assert!(
            collect_backward_subgraph_bounded(&blocks, &broken_preds, &is_entry, 0x1040, 10, 10)
                .is_none()
        );
    }

    #[test]
    fn reaching_values_self_loop_preserves_invariant_and_widens_counter() {
        // 0x14000_1000 (entry): lea rsi, [rip -> 0x14000_2000]; xor ecx, ecx; -> Jump(0x14000_2000)
        // 0x14000_2000 (loop):  add rcx, 1; -> Jump(0x14000_2000) (self-loop!)
        let entry_va = 0x14000_1000u64;
        let loop_va = 0x14000_2000u64;
        let table_va = 0x14000_8000u64;
        let mut entry_bytes = Vec::new();
        entry_bytes.extend_from_slice(&encode_lea_rip(Register::RSI, entry_va, table_va));
        entry_bytes.extend_from_slice(&[0x31, 0xC9]); // xor ecx, ecx

        let mut blocks = BTreeMap::new();
        blocks.insert(
            entry_va,
            make_block(entry_va, &entry_bytes, Terminator::Jump, vec![Edge::Jump(loop_va)]),
        );
        blocks.insert(
            loop_va,
            make_block(
                loop_va,
                &[0x48, 0x83, 0xC1, 0x01], // add rcx, 1
                Terminator::Jump,
                vec![Edge::Jump(loop_va)],
            ),
        );

        let preds = |b: u64| match b {
            x if x == loop_va => vec![loop_va, entry_va],
            _ => Vec::new(),
        };
        let result = reaching_values(
            &blocks,
            &preds,
            &|b| b == entry_va,
            loop_va,
            &[Register::RSI, Register::RCX],
        )
        .expect("self-loop must converge via widening");

        assert_eq!(
            result,
            vec![
                (Register::RSI, Known::exact(table_va)),
                (Register::RCX, Known::Unknown),
            ]
        );
    }

    #[test]
    fn reaching_values_mutual_cycle_preserves_invariant_and_bounds_constants() {
        // 0x14000_1000 (entry): lea rbx, [rip -> 0x14000_9000]; mov eax, 1 -> Jump(0x14000_2000)
        // 0x14000_2000 (merge): nop -> Jump(0x14000_3000), Jump(0x14000_4000)
        // 0x14000_3000:         mov eax, 3 -> Jump(0x14000_2000)
        // 0x14000_4000:         mov eax, 5 -> Jump(0x14000_2000)
        let entry_va = 0x14000_1000u64;
        let merge_va = 0x14000_2000u64;
        let left_va = 0x14000_3000u64;
        let right_va = 0x14000_4000u64;
        let table_va = 0x14000_9000u64;
        let mut entry_bytes = Vec::new();
        entry_bytes.extend_from_slice(&encode_lea_rip(Register::RBX, entry_va, table_va));
        entry_bytes.extend_from_slice(&[0xB8, 0x01, 0x00, 0x00, 0x00]); // mov eax, 1

        let mut blocks = BTreeMap::new();
        blocks.insert(
            entry_va,
            make_block(entry_va, &entry_bytes, Terminator::Jump, vec![Edge::Jump(merge_va)]),
        );
        blocks.insert(
            merge_va,
            make_block(
                merge_va,
                &[0x90], // nop
                Terminator::Jump,
                vec![Edge::Jump(right_va), Edge::Jump(left_va)],
            ),
        );
        blocks.insert(
            left_va,
            make_block(
                left_va,
                &[0xB8, 0x03, 0x00, 0x00, 0x00], // mov eax, 3
                Terminator::Jump,
                vec![Edge::Jump(merge_va)],
            ),
        );
        blocks.insert(
            right_va,
            make_block(
                right_va,
                &[0xB8, 0x05, 0x00, 0x00, 0x00], // mov eax, 5
                Terminator::Jump,
                vec![Edge::Jump(merge_va)],
            ),
        );

        let preds = |b: u64| match b {
            x if x == merge_va => vec![right_va, entry_va, left_va],
            x if x == left_va || x == right_va => vec![merge_va],
            _ => Vec::new(),
        };
        let result = reaching_values(
            &blocks,
            &preds,
            &|b| b == entry_va,
            merge_va,
            &[Register::RBX, Register::RAX],
        )
        .expect("mutual cycle must converge");

        assert_eq!(
            result,
            vec![
                (Register::RBX, Known::exact(table_va)),
                (Register::RAX, Known::Range { lo: 1, hi: 5 }),
            ]
        );
    }

    #[test]
    fn reaching_values_isolated_cycle_without_entry_yields_unknown() {
        // Pure 2-cycle 0x2000 <-> 0x3000 with no entry and no pred-less block.
        let mut blocks = BTreeMap::new();
        blocks.insert(
            0x2000,
            make_block(
                0x2000,
                &[0xB8, 0x07, 0x00, 0x00, 0x00], // mov eax, 7
                Terminator::Jump,
                vec![Edge::Jump(0x3000)],
            ),
        );
        blocks.insert(
            0x3000,
            make_block(0x3000, &[0x90], Terminator::Jump, vec![Edge::Jump(0x2000)]),
        );
        let preds = |b: u64| match b {
            0x2000 => vec![0x3000],
            0x3000 => vec![0x2000],
            _ => Vec::new(),
        };
        let result = reaching_values(
            &blocks,
            &preds,
            &|_| false,
            0x2000,
            &[Register::RAX, Register::RBX],
        )
        .expect("isolated cycle must terminate cleanly");

        assert_eq!(
            result,
            vec![
                (Register::RAX, Known::Unknown),
                (Register::RBX, Known::Unknown),
            ]
        );
    }

    #[test]
    fn reaching_values_is_deterministic_across_pred_and_successor_permutations() {
        // Multi-entry diamond feeding a loop merge: test that reversing/duplicating both
        // predecessor lists and block successor lists produces identical results.
        let build_blocks = |reverse_edges: bool| {
            let mut blocks = BTreeMap::new();
            let mut e1_edges = vec![Edge::Jump(0x3000), Edge::Jump(0x4000)];
            let mut e2_edges = vec![Edge::Jump(0x3000), Edge::Jump(0x4000)];
            if reverse_edges {
                e1_edges.reverse();
                e1_edges.push(Edge::Jump(0x3000));
                e2_edges.reverse();
                e2_edges.push(Edge::Jump(0x4000));
            }
            blocks.insert(
                0x1000,
                make_block(
                    0x1000,
                    &[0xB8, 0x02, 0x00, 0x00, 0x00], // mov eax, 2
                    Terminator::Jump,
                    e1_edges,
                ),
            );
            blocks.insert(
                0x2000,
                make_block(
                    0x2000,
                    &[0xB8, 0x06, 0x00, 0x00, 0x00], // mov eax, 6
                    Terminator::Jump,
                    e2_edges,
                ),
            );
            blocks.insert(
                0x3000,
                make_block(
                    0x3000,
                    &[0x48, 0x83, 0xC0, 0x01], // add rax, 1
                    Terminator::Jump,
                    vec![Edge::Jump(0x5000)],
                ),
            );
            blocks.insert(
                0x4000,
                make_block(
                    0x4000,
                    &[0x48, 0x83, 0xE8, 0x01], // sub rax, 1
                    Terminator::Jump,
                    vec![Edge::Jump(0x5000)],
                ),
            );
            blocks.insert(
                0x5000,
                make_block(0x5000, &[0x90], Terminator::Return, vec![Edge::Return]),
            );
            blocks
        };

        let preds_a = |b: u64| match b {
            0x3000 | 0x4000 => vec![0x1000, 0x2000],
            0x5000 => vec![0x3000, 0x4000],
            _ => Vec::new(),
        };
        let preds_b = |b: u64| match b {
            0x3000 | 0x4000 => vec![0x2000, 0x1000, 0x2000],
            0x5000 => vec![0x4000, 0x3000, 0x4000],
            _ => Vec::new(),
        };
        let is_entry = |b: u64| b == 0x1000 || b == 0x2000;

        let r1 = reaching_values(&build_blocks(false), &preds_a, &is_entry, 0x5000, &[Register::RAX])
            .expect("canonical run");
        let r2 = reaching_values(&build_blocks(true), &preds_b, &is_entry, 0x5000, &[Register::RAX])
            .expect("permuted run");

        assert_eq!(r1, r2);
        // Entry paths yield 2 and 6; 0x3000 adds 1 -> [3, 7], 0x4000 subs 1 -> [1, 5]; hull is [1, 7].
        assert_eq!(r1, vec![(Register::RAX, Known::Range { lo: 1, hi: 7 })]);
    }

    #[test]
    fn reaching_values_worklist_step_guard_bails_out_cleanly() {
        // Widening self-loop requires > 3 worklist pops to reach fixpoint.
        let mut blocks = BTreeMap::new();
        blocks.insert(
            0x1000,
            make_block(
                0x1000,
                &[0x31, 0xC9], // xor ecx, ecx
                Terminator::Jump,
                vec![Edge::Jump(0x2000)],
            ),
        );
        blocks.insert(
            0x2000,
            make_block(
                0x2000,
                &[0x48, 0x83, 0xC1, 0x01], // add rcx, 1
                Terminator::Jump,
                vec![Edge::Jump(0x2000)],
            ),
        );
        let preds = |b: u64| match b {
            0x2000 => vec![0x1000, 0x2000],
            _ => Vec::new(),
        };
        let is_entry = |b: u64| b == 0x1000;

        assert!(
            reaching_values_bounded(
                &blocks,
                &preds,
                &is_entry,
                0x2000,
                &[Register::RCX],
                MAX_REACH_BLOCKS,
                MAX_REACH_DEPTH,
                3,
            )
            .is_none()
        );
        assert_eq!(
            reaching_values_bounded(
                &blocks,
                &preds,
                &is_entry,
                0x2000,
                &[Register::RCX],
                MAX_REACH_BLOCKS,
                MAX_REACH_DEPTH,
                MAX_WORKLIST_STEPS,
            ),
            Some(vec![(Register::RCX, Known::Unknown)])
        );
    }

    #[test]
    fn diamond_u32_param_guard_bounds_movsxd_without_falsely_bounding_raw_64bit() {
        use crate::code_image::{CodeImage, CodeSection};
        use crate::discover::discover;

        // Build a diamond CFG where live-in `ecx` is guarded BEFORE the diamond:
        //   0x00 (entry):   cmp ecx, 2; ja default(0x45); test edx, edx; je right(0x14)
        //   0x0a (left):    nop; jmp merge(0x16)
        //   0x14 (right):   nop (fallthrough to merge)
        //   0x16 (merge):   movsxd rax, ecx; lea rdx, [rip -> table]; movsxd rcx, [rdx + rax*4];
        //                   add rcx, rdx; jmp rcx
        //   0x2e..0x45:     cases 0, 1, 2
        //   0x45:           default: xor eax, eax; ret
        const BASE: u64 = 0x1_4000_0000;
        const TEXT: u64 = BASE + 0x1000;
        const DATA: u64 = BASE + 0x2000;

        // Helper to assemble a diamond with either `movsxd rax, ecx` + `[rdx + rax*4]`
        // or raw 64-bit `[rdx + rcx*4]`.
        let build = |use_movsxd: bool| -> crate::discover::Analysis {
            let mut code = Vec::new();
            // 0x00: cmp ecx, 2 (3B)
            code.extend_from_slice(&[0x83, 0xf9, 0x02]);
            // 0x03: ja default (rel8 placeholder at index 4)
            code.extend_from_slice(&[0x77, 0x00]);
            // 0x05: test edx, edx (2B)
            code.extend_from_slice(&[0x85, 0xd2]);
            // 0x07: je right (rel8 placeholder at index 8)
            code.extend_from_slice(&[0x74, 0x00]);
            // 0x09 (left): nop; jmp merge (rel8 placeholder at index 11)
            code.extend_from_slice(&[0x90, 0xeb, 0x00]);
            let right_off = code.len();
            // right: nop (falls through to merge)
            code.push(0x90);
            let merge_off = code.len();
            if use_movsxd {
                // movsxd rax, ecx
                code.extend_from_slice(&[0x48, 0x63, 0xc1]);
            }
            let lea_ip = TEXT + code.len() as u64;
            code.extend_from_slice(&encode_lea_rip(Register::RDX, lea_ip, DATA));
            if use_movsxd {
                // movsxd rcx, dword ptr [rdx + rax*4]
                code.extend_from_slice(&[0x48, 0x63, 0x0c, 0x82]);
            } else {
                // movsxd rcx, dword ptr [rdx + rcx*4] (raw 64-bit rcx!)
                code.extend_from_slice(&[0x48, 0x63, 0x0c, 0x8a]);
            }
            // add rcx, rdx; jmp rcx
            code.extend_from_slice(&[0x48, 0x01, 0xd1, 0xff, 0xe1]);

            let c0 = code.len();
            code.extend_from_slice(&[0xb8, 0x00, 0x00, 0x00, 0x00, 0xc3]);
            let c1 = code.len();
            code.extend_from_slice(&[0xb8, 0x01, 0x00, 0x00, 0x00, 0xc3]);
            let c2 = code.len();
            code.extend_from_slice(&[0xb8, 0x02, 0x00, 0x00, 0x00, 0xc3]);
            let def = code.len();
            code.extend_from_slice(&[0x31, 0xc0, 0xc3]);

            code[4] = (def - 5) as u8;
            code[8] = (right_off - 9) as u8;
            code[11] = (merge_off - 12) as u8;

            let mut data = Vec::new();
            for off in [c0, c1, c2] {
                let delta = ((TEXT + off as u64) as i64 - DATA as i64) as i32;
                data.extend_from_slice(&delta.to_le_bytes());
            }

            let image = CodeImage::new(
                BASE,
                vec![
                    CodeSection::new(TEXT, &code, true),
                    CodeSection::new(DATA, &data, false),
                ],
                vec![TEXT],
            );
            discover(&image)
        };

        // With `movsxd rax, ecx`, the 32-bit diamond guard resolves the switch with a proven bound!
        let resolved = build(true);
        assert_eq!(resolved.jump_tables.len(), 1);
        let jt = resolved.jump_tables.values().next().unwrap();
        assert!(!jt.inferred);
        assert_eq!(jt.targets.len(), 3);

        // With raw 64-bit `[rdx + rcx*4]`, the 32-bit guard on live-in `ecx` must NOT falsely
        // bound 64-bit `rcx` (upper 32 bits are unknown)!
        let raw64 = build(false);
        assert!(
            raw64.jump_tables.is_empty(),
            "32-bit guard on live-in ecx must not bound raw 64-bit rcx: {:?}",
            raw64.jump_tables
        );
    }

    #[test]
    fn lea_offset_test_mask_narrow_cmp_and_64bit_synthetic_cmp() {
        // 1) `test ecx, ~7; jne default` + `lea eax, [rcx - 2]` after `cmp ecx, 2; jb default`:
        //    entry (0x1000): test ecx, 0xffff_fff8; jne 0x9000 -> fallthrough to 0x1010
        //    0x1010:         cmp ecx, 2; jb 0x9000 -> fallthrough to 0x1020
        //    0x1020:         lea eax, [rcx - 2] -> jump to 0x1030
        //    0x1030:         nop
        let mut blocks = BTreeMap::new();
        blocks.insert(
            0x1000,
            make_block(
                0x1000,
                &[0xf7, 0xc1, 0xf8, 0xff, 0xff, 0xff, 0x75, 0x00], // test ecx, ~7; jne
                Terminator::Branch,
                vec![Edge::Taken(0x9000), Edge::Fallthrough(0x1010)],
            ),
        );
        blocks.insert(
            0x1010,
            make_block(
                0x1010,
                &[0x83, 0xf9, 0x02, 0x72, 0x00], // cmp ecx, 2; jb
                Terminator::Branch,
                vec![Edge::Taken(0x9000), Edge::Fallthrough(0x1020)],
            ),
        );
        blocks.insert(
            0x1020,
            make_block(
                0x1020,
                &[0x8d, 0x41, 0xfe], // lea eax, [rcx - 2]
                Terminator::Jump,
                vec![Edge::Jump(0x1030)],
            ),
        );
        blocks.insert(
            0x1030,
            make_block(0x1030, &[0x90], Terminator::Return, vec![Edge::Return]),
        );

        let preds = |b: u64| match b {
            0x1010 => vec![0x1000],
            0x1020 => vec![0x1010],
            0x1030 => vec![0x1020],
            _ => Vec::new(),
        };
        let facts = reaching_facts(
            &blocks,
            &preds,
            &|b| b == 0x1000,
            0x1030,
            &[Register::RCX, Register::RAX],
        )
        .expect("facts computed");

        // Live-in RCX has unknown upper 32 bits, but low 32 bits are proven in [2, 7].
        assert_eq!(facts.values[0], (Register::RCX, Known::Unknown));
        assert_eq!(
            facts.u32_values[0],
            (Register::RCX, Known::Range { lo: 2, hi: 7 })
        );
        // `lea eax, [rcx - 2]` zero-extends into RAX and shifts [2, 7] -> [0, 5] in full 64-bit!
        assert_eq!(
            facts.values[1],
            (Register::RAX, Known::Range { lo: 0, hi: 5 })
        );
        assert_eq!(
            facts.u32_values[1],
            (Register::RAX, Known::Range { lo: 0, hi: 5 })
        );

        // 2) Full 64-bit `synthesize` emits 64-bit `cmp r64, hi; ja` when hi <= i32::MAX,
        //    whereas `synthesize_u32` emits 32-bit `cmp r32, hi; ja`.
        let synth64 = synthesize(Register::R9, Known::Range { lo: 1, hi: 5 }, &|_| false);
        assert_eq!(synth64.len(), 4);
        assert!(synth64[0].0.op0_register().is_gpr64());
        assert_eq!(synth64[0].0.op0_register(), Register::R9);
        assert_eq!(synth64[0].0.try_immediate(1).ok(), Some(5));
        assert!(synth64[2].0.op0_register().is_gpr64());
        assert_eq!(synth64[2].0.try_immediate(1).ok(), Some(1));

        let synth32 = synthesize_u32(Register::RCX, Known::Range { lo: 2, hi: 7 });
        assert_eq!(synth32.len(), 4);
        assert!(synth32[0].0.op0_register().is_gpr32());
        assert_eq!(synth32[0].0.op0_register(), Register::ECX);
        assert_eq!(synth32[0].0.try_immediate(1).ok(), Some(7));

        // 3) 8-bit `cmp al, 4; ja default` refines RAX when already zero-extended to a byte.
        let mut narrow_blocks = BTreeMap::new();
        narrow_blocks.insert(
            0x2000,
            make_block(
                0x2000,
                &[0x0f, 0xb6, 0xc1, 0x3c, 0x04, 0x77, 0x00], // movzx eax, cl; cmp al, 4; ja
                Terminator::Branch,
                vec![Edge::Taken(0x9000), Edge::Fallthrough(0x2010)],
            ),
        );
        narrow_blocks.insert(
            0x2010,
            make_block(0x2010, &[0x90], Terminator::Return, vec![Edge::Return]),
        );
        let narrow_res = reaching_values(
            &narrow_blocks,
            &|b| if b == 0x2010 { vec![0x2000] } else { vec![] },
            &|b| b == 0x2000,
            0x2010,
            &[Register::RAX],
        )
        .expect("narrow cmp");
        assert_eq!(
            narrow_res,
            vec![(Register::RAX, Known::Range { lo: 0, hi: 4 })]
        );

        // 4) 64-bit copy tracking (`mov r9, rcx` and `lea r10, [r9]`) + 64-bit synthetic cmp
        //    bounding a direct 64-bit table load after a diamond merge.
        let mut copy_blocks = BTreeMap::new();
        copy_blocks.insert(
            0x3000,
            make_block(
                0x3000,
                &[0x49, 0x89, 0xc9, 0x4d, 0x8d, 0x11], // mov r9, rcx; lea r10, [r9]
                Terminator::Jump,
                vec![Edge::Jump(0x3010)],
            ),
        );
        copy_blocks.insert(
            0x3010,
            make_block(0x3010, &[0x90], Terminator::Return, vec![Edge::Return]),
        );
        let copy_facts = reaching_facts(
            &copy_blocks,
            &|b| if b == 0x3010 { vec![0x3000] } else { vec![] },
            &|b| b == 0x3000,
            0x3010,
            &[Register::R9, Register::R10],
        )
        .expect("copy facts");
        assert_eq!(
            copy_facts.copies,
            vec![
                (Register::R9, Register::RCX, true),
                (Register::R10, Register::R9, true),
            ]
        );
        let c64 = synthesize_copy64(Register::R9, Register::RCX).expect("mov r9, rcx");
        assert!(c64.0.op0_register().is_gpr64());
        assert_eq!(c64.0.op0_register(), Register::R9);
        assert_eq!(c64.0.op1_register(), Register::RCX);

        // `reaching_copies` reports only zero-extended 32-bit copies (`mov r32, r32` /
        // `lea r32, [src + 0]`), not 64-bit copies (`mov r64, r64`).
        let mut copy32_blocks = BTreeMap::new();
        copy32_blocks.insert(
            0x3100,
            make_block(
                0x3100,
                &[0x41, 0x89, 0xc9, 0x44, 0x8d, 0x11, 0x49, 0x89, 0xcb], // mov r9d, ecx; lea r10d, [rcx]; mov r11, rcx
                Terminator::Jump,
                vec![Edge::Jump(0x3110)],
            ),
        );
        copy32_blocks.insert(
            0x3110,
            make_block(0x3110, &[0x90], Terminator::Return, vec![Edge::Return]),
        );
        let zext_copies = reaching_copies(
            &copy32_blocks,
            &|b| if b == 0x3110 { vec![0x3100] } else { vec![] },
            &|b| b == 0x3100,
            0x3110,
            &[Register::R9, Register::R10, Register::R11],
        )
        .expect("32-bit copies");
        assert_eq!(
            zext_copies,
            vec![
                (Register::R9, Register::RCX),
                (Register::R10, Register::RCX),
            ]
        );
    }

    #[test]
    fn full_64bit_reaching_bound_resolves_raw_64bit_switch_across_diamond() {
        use crate::code_image::{CodeImage, CodeSection};
        use crate::discover::discover;

        // Entry guards full 64-bit `rcx` (`cmp rcx, 2; ja default`), branches on `edx` into a
        // diamond, and the merge block indexes `[rdx + rcx*4]` directly using 64-bit `rcx`
        // without `movsxd`. Our 64-bit synthetic `cmp rcx, 2; ja` must resolve it!
        const BASE: u64 = 0x1_4000_0000;
        const TEXT: u64 = BASE + 0x1000;
        const DATA: u64 = BASE + 0x2000;

        let mut code = Vec::new();
        // 0x00: cmp rcx, 2 (4B: 48 83 f9 02)
        code.extend_from_slice(&[0x48, 0x83, 0xf9, 0x02]);
        // 0x04: ja default (rel8 placeholder at index 5)
        code.extend_from_slice(&[0x77, 0x00]);
        // 0x06: test edx, edx (2B)
        code.extend_from_slice(&[0x85, 0xd2]);
        // 0x08: je right (rel8 placeholder at index 9)
        code.extend_from_slice(&[0x74, 0x00]);
        // left: nop; jmp merge (rel8 placeholder at index 12)
        code.extend_from_slice(&[0x90, 0xeb, 0x00]);
        let right_off = code.len();
        code.push(0x90);
        let merge_off = code.len();
        let lea_ip = TEXT + code.len() as u64;
        code.extend_from_slice(&encode_lea_rip(Register::RDX, lea_ip, DATA));
        // movsxd rax, dword ptr [rdx + rcx*4]; add rax, rdx; jmp rax
        code.extend_from_slice(&[0x48, 0x63, 0x04, 0x8a, 0x48, 0x01, 0xd0, 0xff, 0xe0]);
        let c0 = code.len();
        code.extend_from_slice(&[0xb8, 0x00, 0x00, 0x00, 0x00, 0xc3]);
        let c1 = code.len();
        code.extend_from_slice(&[0xb8, 0x01, 0x00, 0x00, 0x00, 0xc3]);
        let c2 = code.len();
        code.extend_from_slice(&[0xb8, 0x02, 0x00, 0x00, 0x00, 0xc3]);
        let def = code.len();
        code.extend_from_slice(&[0x31, 0xc0, 0xc3]);

        code[5] = (def - 6) as u8;
        code[9] = (right_off - 10) as u8;
        code[12] = (merge_off - 13) as u8;

        let mut data = Vec::new();
        for off in [c0, c1, c2] {
            let delta = ((TEXT + off as u64) as i64 - DATA as i64) as i32;
            data.extend_from_slice(&delta.to_le_bytes());
        }
        let image = CodeImage::new(
            BASE,
            vec![
                CodeSection::new(TEXT, &code, true),
                CodeSection::new(DATA, &data, false),
            ],
            vec![TEXT],
        );
        let analysis = discover(&image);
        assert_eq!(analysis.jump_tables.len(), 1);
        let jt = analysis.jump_tables.values().next().unwrap();
        assert!(!jt.inferred);
        assert_eq!(jt.targets.len(), 3);
    }
}
