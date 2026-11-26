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
/// Each block in a monotone `GPR_COUNT`-register lattice with widening after `WIDEN_AFTER`
/// updates can change at most `WIDEN_AFTER + GPR_COUNT` times after its initial visit.
pub(crate) const MAX_WORKLIST_STEPS: usize =
    MAX_REACH_BLOCKS * (WIDEN_AFTER as usize + GPR_COUNT + 2);

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

/// Register file state: one [`Known`] per 64-bit GPR (index = `Register::number()` of RAX..R15).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct State([Known; GPR_COUNT], [u8; GPR_COUNT]);

/// [`State`] alias slot: no register copy known.
const NO_ALIAS: u8 = 0xff;
/// [`State`] alias slot: no path seen yet (only during the analysis).
const ALIAS_UNVISITED: u8 = 0xfe;

impl State {
    fn unvisited() -> Self {
        Self([Known::Unvisited; GPR_COUNT], [ALIAS_UNVISITED; GPR_COUNT])
    }
    fn unknown() -> Self {
        Self([Known::Unknown; GPR_COUNT], [NO_ALIAS; GPR_COUNT])
    }
    fn get(&self, reg: Register) -> Known {
        slot(reg).map_or(Known::Unknown, |i| self.0[i])
    }
    fn set(&mut self, reg: Register, value: Known) {
        if let Some(i) = slot(reg) {
            self.0[i] = value;
        }
    }
    fn join(&self, other: &State) -> State {
        let mut out = *self;
        for (o, x) in out.0.iter_mut().zip(other.0.iter()) {
            *o = o.join(*x);
        }
        for (i, o) in out.1.iter_mut().enumerate() {
            *o = match (self.1[i], other.1[i]) {
                (ALIAS_UNVISITED, y) | (y, ALIAS_UNVISITED) => y,
                (a, b) if a == b => a,
                // Copies of one source on one path, of each other on the other: `i = a` and
                // `b = a` on the first path make `i` a copy of `b` there too (and symmetrically).
                (a, b) if b < GPR_COUNT as u8 && self.1[usize::from(b)] == a => b,
                (a, b) if a < GPR_COUNT as u8 && other.1[usize::from(a)] == b => a,
                _ => NO_ALIAS,
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

/// Values of `regs` on entry to `target`, over every intra-function path that reaches it.
/// `preds(b)` lists the blocks with an intra-function edge into `b`; `is_entry(b)` marks function
/// entries (where every register is unknown). `None` when the reachable subgraph exceeds
/// [`MAX_REACH_BLOCKS`] / [`MAX_REACH_DEPTH`], the worklist exceeds [`MAX_WORKLIST_STEPS`], or
/// `target` (or a backward-reachable predecessor) is unknown.
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
            .filter_map(|&r| {
                let a = input.1[slot(r)?];
                (a < GPR_COUNT as u8).then(|| (r.full_register(), Register::RAX + u32::from(a)))
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
                for (i, k) in new.0.iter_mut().enumerate() {
                    if *k != old.0[i] {
                        *k = Known::Unknown;
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

/// Apply `instr` to `state`.
/// Register `i` is overwritten: it is no longer a copy of anything, and the registers that were
/// copies of it (`j = zext(i32)`) and survive the write (`keeps(j)`) stay copies of each other.
/// The first of them takes the place of `i` as their source; the rest lose their relation.
fn unlink(state: &mut State, i: usize, keeps: &dyn Fn(usize) -> bool) {
    state.1[i] = NO_ALIAS;
    let mut rep: Option<u8> = None;
    for j in 0..GPR_COUNT {
        if j == i || usize::from(state.1[j]) != i {
            continue;
        }
        if !keeps(j) {
            state.1[j] = NO_ALIAS;
            continue;
        }
        match rep {
            None => {
                rep = Some(j as u8);
                state.1[j] = NO_ALIAS;
            }
            Some(r) => state.1[j] = r,
        }
    }
}

fn transfer(info: &mut InstructionInfoFactory, instr: &Instruction, state: &mut State) {
    if matches!(
        instr.flow_control(),
        FlowControl::Call | FlowControl::IndirectCall
    ) {
        for (i, known) in state.0.iter_mut().enumerate() {
            if is_win64_volatile(Register::RAX + i as u32) {
                *known = Known::Unknown;
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
    // A `mov r32, r32` between distinct registers leaves the destination a zero-extended copy of
    // the source; any other write ends the copy relations of the written register.
    let copy = (instr.mnemonic() == Mnemonic::Mov
        && instr.op_count() == 2
        && instr.op0_kind() == OpKind::Register
        && instr.op1_kind() == OpKind::Register
        && instr.op0_register().is_gpr32()
        && instr.op1_register().is_gpr32()
        && instr.op0_register().full_register() != instr.op1_register().full_register())
    .then(|| {
        (
            instr.op0_register().full_register(),
            instr.op1_register().full_register(),
        )
    });
    let written_slots: Vec<usize> = written.iter().filter_map(|r| slot(*r)).collect();
    for &i in &written_slots {
        unlink(state, i, &|j| !written_slots.contains(&j));
    }
    if let Some((dst, src)) = copy
        && let (Some(d), Some(sr)) = (slot(dst), slot(src))
    {
        state.1[d] = sr as u8;
    }
    // The one register an exactly modelled instruction writes (operand 0).
    let modelled = model(instr, state);
    for reg in written {
        let value = match modelled {
            Some((dst, v)) if dst == reg => v,
            _ => Known::Unknown,
        };
        state.set(reg, value);
    }
}

/// Exact effect of `instr` on its destination register (operand 0), if modelled.
fn model(instr: &Instruction, state: &State) -> Option<(Register, Known)> {
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
    let known = match instr.mnemonic() {
        Mnemonic::Lea if d64 && instr.is_ip_rel_memory_operand() => {
            Known::exact(instr.ip_rel_memory_address())
        }
        Mnemonic::Mov if imm.is_some() => Known::exact(imm? & mask(width)),
        Mnemonic::Xor if src_reg == Some(dst) => Known::exact(0),
        Mnemonic::Mov if src_reg.is_some_and(|s| s.is_gpr64()) && d64 => state.get(src_reg?),
        Mnemonic::Mov if src_reg.is_some_and(|s| s.is_gpr32()) && d32 => {
            let s = state.get(src_reg?);
            if s.within_u32() {
                s
            } else {
                Known::Range {
                    lo: 0,
                    hi: mask(32),
                }
            }
        }
        Mnemonic::Mov if instr.op_count() == 2 && instr.op1_kind() == OpKind::Memory && d32 => {
            Known::Range {
                lo: 0,
                hi: mask(32),
            }
        }
        Mnemonic::Movzx if instr.op_count() == 2 => {
            let bits = match instr.op1_kind() {
                OpKind::Register => {
                    let s = instr.op1_register();
                    if matches!(s, Register::AH | Register::BH | Register::CH | Register::DH) {
                        8
                    } else {
                        s.size() as u32 * 8
                    }
                }
                OpKind::Memory => instr.memory_size().size() as u32 * 8,
                _ => return None,
            };
            let m = mask(bits);
            match (instr.op1_kind(), src_reg.map(|s| state.get(s))) {
                (OpKind::Register, Some(Known::Range { lo, hi }))
                    if hi <= m
                        && !matches!(
                            src_reg,
                            Some(Register::AH | Register::BH | Register::CH | Register::DH)
                        ) =>
                {
                    Known::Range { lo, hi }
                }
                _ => Known::Range { lo: 0, hi: m },
            }
        }
        Mnemonic::Movsxd if d64 && src_reg.is_some_and(|s| s.is_gpr32()) => {
            match state.get(src_reg?) {
                Known::Range { lo, hi } if hi <= 0x7fff_ffff => Known::Range { lo, hi },
                _ => Known::Unknown,
            }
        }
        Mnemonic::And if imm.is_some() => {
            let m = imm? & mask(width);
            let top = if width == 64 && (m as i64) < 0 {
                None
            } else {
                Some(m)
            };
            match (top, cur) {
                (Some(m), Known::Range { hi, .. }) => Known::Range {
                    lo: 0,
                    hi: hi.min(m),
                },
                (Some(m), _) => Known::Range { lo: 0, hi: m },
                (None, _) => Known::Unknown,
            }
        }
        Mnemonic::Shr if imm.is_some() => {
            let s = (imm? & 63) as u32;
            let base = if d32 {
                if cur.within_u32() {
                    cur
                } else {
                    Known::Range {
                        lo: 0,
                        hi: mask(32),
                    }
                }
            } else {
                cur
            };
            match base {
                Known::Range { lo, hi } => Known::Range {
                    lo: lo >> s,
                    hi: hi >> s,
                },
                _ => Known::Range {
                    lo: 0,
                    hi: mask(width) >> s,
                },
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
            let base = if d32 && !cur.within_u32() {
                Known::Unknown
            } else {
                cur
            };
            match base {
                Known::Range { lo, hi } => {
                    let nlo = i128::from(lo) + delta;
                    let nhi = i128::from(hi) + delta;
                    if nlo < 0 || nhi > i128::from(mask(width)) {
                        if d32 {
                            Known::Range {
                                lo: 0,
                                hi: mask(32),
                            }
                        } else {
                            Known::Unknown
                        }
                    } else {
                        Known::Range {
                            lo: nlo as u64,
                            hi: nhi as u64,
                        }
                    }
                }
                _ if d32 => Known::Range {
                    lo: 0,
                    hi: mask(32),
                },
                _ => Known::Unknown,
            }
        }
        // Any other unconditional write to a 32-bit register zero-extends: bounded by 2^32.
        _ if d32 => Known::Range {
            lo: 0,
            hi: mask(32),
        },
        _ => return None,
    };
    Some((full, known))
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
        if instr.rflags_modified() == 0 {
            continue;
        }
        if instr.op_count() < 1 || instr.op0_kind() != OpKind::Register {
            return;
        }
        let reg = instr.op0_register();
        if !(reg.is_gpr32() || reg.is_gpr64())
            || block.instructions[i + 1..n - 1]
                .iter()
                .any(|k| writes(info, k, reg))
        {
            return;
        }
        let bits = reg.size() as u32 * 8;
        let full = reg.full_register();
        let cur = state.get(full);
        // A 32-bit compare bounds the full register only when its upper half is known zero.
        if bits == 32 && !cur.within_u32() {
            return;
        }
        let (lo, hi) = match cur {
            Known::Range { lo, hi } => (lo, hi),
            _ => (0, mask(bits)),
        };
        let fact = match instr.mnemonic() {
            Mnemonic::Cmp => {
                let Ok(raw) = instr.try_immediate(1) else {
                    return;
                };
                let imm = raw & mask(bits);
                unsigned_fact(jcc.mnemonic(), taken, imm, lo, hi)
            }
            Mnemonic::Test
                if instr.op_count() == 2
                    && instr.op1_kind() == OpKind::Register
                    && instr.op1_register() == reg =>
            {
                match (jcc.mnemonic(), taken) {
                    (Mnemonic::Je, true) | (Mnemonic::Jne, false) => Some((0, 0)),
                    _ => None,
                }
            }
            _ => None,
        };
        if let Some((nlo, nhi)) = fact {
            let nlo = nlo.max(lo);
            let nhi = nhi.min(hi);
            if nlo <= nhi {
                state.set(full, Known::Range { lo: nlo, hi: nhi });
            }
        }
        return;
    }
}

/// `[lo, hi]` implied for the compared register by `cmp reg, imm` + `jcc` on the `taken` side,
/// given its current range `[cur_lo, cur_hi]` (signed conditions only when that is non-negative
/// below 2^31).
fn unsigned_fact(
    jcc: Mnemonic,
    taken: bool,
    imm: u64,
    cur_lo: u64,
    cur_hi: u64,
) -> Option<(u64, u64)> {
    let nonneg = cur_hi <= 0x7fff_ffff && imm <= 0x7fff_ffff;
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

/// Synthetic instructions that state `known` for `reg` in a form the jump-table slicer reads:
/// `lea reg, [rip+v]` for a single in-image address, else `cmp reg, hi` with a not-taken `ja`,
/// plus `cmp reg, lo` with a not-taken `jb` when `lo > 0`.
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
    // `cmp r32, imm32` compares the low half; valid because `hi < 2^32` bounds the whole register.
    let cmp = |imm: u64| -> Option<Instruction> {
        let mut bytes = Vec::with_capacity(7);
        if n >= 8 {
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
}
