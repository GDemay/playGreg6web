//! RDNA2 (GFX10.3) control-flow graph construction and structured branch lifting.
//!
//! Track 4 / S29: RDNA2 scalar branches (`s_branch`, `s_cbranch_scc0/scc1`,
//! `s_cbranch_vccz/vccnz`, `s_cbranch_execz/execnz`) are the single largest
//! category of unsupported ops in real console shader packs. WGSL has no
//! `goto`, so the linear decoded instruction stream must be lifted into
//! structured control flow (`if` / `loop` / `break` / `continue` / `return`).
//!
//! This module is deliberately self-contained (it only reads the decoder's
//! [`Rdna2Instruction`] stream) so it can be wired into the WGSL lowering
//! (`rdna2_to_wgsl.rs`) independently:
//!
//! 1. [`ControlFlowGraph::from_instructions`] maps byte offsets to instruction
//!    indices, resolves PC-relative SOPP branch targets
//!    (`offset + 4 + sext(simm16) * 4`, as already computed by the decoder),
//!    and splits the stream into maximal basic blocks. Invalid / out-of-range
//!    targets never panic: they are recorded as [`CfgDiagnostic`]s and routed
//!    to a synthetic, instruction-less exit block (program end).
//! 2. [`structure_cfg`] recognizes clean structured shapes (sequences,
//!    `if`-skip, `if`/`else` diamonds, natural loops with `break`/`continue`)
//!    and falls back to a bounded `loop { switch (pc) { ... } }` dispatcher
//!    ([`StructuredNode::Dispatch`]) for irreducible or otherwise complex
//!    regions, so *every* CFG lifts to valid, terminating WGSL.
//! 3. [`emit_wgsl_control_flow`] prints the structured tree as WGSL statements,
//!    delegating per-block straight-line code to a caller-supplied closure.
//!
//! Exec-mask writes (`s_and_saveexec_b32/b64`, `s_andn2_wrexec_b32/b64`, ...)
//! are ordinary data-flow instructions (SOP1) that never end a basic block;
//! [`classify_exec_mask_op`] identifies them for the lowering so divergent
//! `s_cbranch_execz` regions can be tied back to their mask setup.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::ops::Range;

use crate::rdna2::{
    Rdna2AtomicDomain, Rdna2Detail, Rdna2DsOpKind, Rdna2ExportTarget, Rdna2Instruction,
};

/// Decoded RDNA2 instruction as produced by [`crate::rdna2::decode_program`].
pub type DecodedInstruction = Rdna2Instruction;

/// Index of a [`BasicBlock`] inside [`ControlFlowGraph::blocks`].
pub type BlockId = usize;

/// Default iteration bound for [`StructuredNode::Dispatch`] fallbacks.
pub const DEFAULT_DISPATCH_MAX_ITERATIONS: u32 = 65_536;

/// Iteration bound emitted as a guard around every structured WGSL `loop`,
/// so a mis-lifted or data-dependent infinite loop can never hang the GPU.
pub const STRUCTURED_LOOP_MAX_ITERATIONS: u32 = 65_536;

/// `_pg6_pc` sentinel meaning "leave the dispatcher and continue after it".
pub const DISPATCH_EXIT_PC: u32 = u32::MAX;

/// Maximum structured nesting depth before falling back to `Dispatch`
/// (bounds recursion in structuring, emission, and downstream Naga/WGSL
/// parsing/validation even on constrained debug stacks).
const MAX_STRUCTURE_DEPTH: usize = 32;

/// Hard upper bound on structuring region visits so pathological large CFGs
/// degrade to bounded `Dispatch` in deterministic linear-ish time.
const MAX_STRUCTURE_BUDGET: usize = 262_144;

// SOPP opcodes (GFX10.3).
const SOPP_ENDPGM: u8 = 0x01;
const SOPP_BRANCH: u8 = 0x02;
const SOPP_CBRANCH_SCC0: u8 = 0x04;
const SOPP_CBRANCH_SCC1: u8 = 0x05;
const SOPP_CBRANCH_VCCZ: u8 = 0x06;
const SOPP_CBRANCH_VCCNZ: u8 = 0x07;
const SOPP_CBRANCH_EXECZ: u8 = 0x08;
const SOPP_CBRANCH_EXECNZ: u8 = 0x09;
const SOPP_CBRANCH_CDBG_FIRST: u8 = 0x17;
const SOPP_CBRANCH_CDBG_LAST: u8 = 0x1A;
const SOPP_ENDPGM_SAVED: u8 = 0x1B;
const SOPP_ENDPGM_ORDERED_PS_DONE: u8 = 0x1E;
const SOPP_CODE_END: u8 = 0x1F;

// ---------------------------------------------------------------------------
// Branch / terminator model
// ---------------------------------------------------------------------------

/// Condition under which an RDNA2 SOPP branch is taken.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BranchKind {
    /// `s_branch`: always taken.
    Unconditional,
    /// `s_cbranch_scc0`: taken when `SCC == 0`.
    Scc0,
    /// `s_cbranch_scc1`: taken when `SCC == 1`.
    Scc1,
    /// `s_cbranch_vccz`: taken when `VCC == 0`.
    Vccz,
    /// `s_cbranch_vccnz`: taken when `VCC != 0`.
    Vccnz,
    /// `s_cbranch_execz`: taken when `EXEC == 0` (all lanes inactive).
    Execz,
    /// `s_cbranch_execnz`: taken when `EXEC != 0`.
    Execnz,
}

impl BranchKind {
    /// Map a SOPP opcode to its branch condition (`None` for non-branches).
    pub const fn from_sopp_op(op: u8) -> Option<Self> {
        match op {
            SOPP_BRANCH => Some(Self::Unconditional),
            SOPP_CBRANCH_SCC0 => Some(Self::Scc0),
            SOPP_CBRANCH_SCC1 => Some(Self::Scc1),
            SOPP_CBRANCH_VCCZ => Some(Self::Vccz),
            SOPP_CBRANCH_VCCNZ => Some(Self::Vccnz),
            SOPP_CBRANCH_EXECZ => Some(Self::Execz),
            SOPP_CBRANCH_EXECNZ => Some(Self::Execnz),
            _ => None,
        }
    }

    /// Logical negation of a conditional kind (`None` for `Unconditional`).
    pub const fn negate(self) -> Option<Self> {
        match self {
            Self::Unconditional => None,
            Self::Scc0 => Some(Self::Scc1),
            Self::Scc1 => Some(Self::Scc0),
            Self::Vccz => Some(Self::Vccnz),
            Self::Vccnz => Some(Self::Vccz),
            Self::Execz => Some(Self::Execnz),
            Self::Execnz => Some(Self::Execz),
        }
    }

    pub const fn is_conditional(self) -> bool {
        !matches!(self, Self::Unconditional)
    }

    /// RDNA2 assembler mnemonic of the branch.
    pub const fn mnemonic(self) -> &'static str {
        match self {
            Self::Unconditional => "s_branch",
            Self::Scc0 => "s_cbranch_scc0",
            Self::Scc1 => "s_cbranch_scc1",
            Self::Vccz => "s_cbranch_vccz",
            Self::Vccnz => "s_cbranch_vccnz",
            Self::Execz => "s_cbranch_execz",
            Self::Execnz => "s_cbranch_execnz",
        }
    }
}

/// WGSL boolean expression for "branch of `kind` is taken".
///
/// Assumes the lowering keeps `scc: bool`, `vcc: u32` and `exec_mask: u32`
/// function-scope variables (wave32 view; wave64 lowers its low half).
pub const fn cond_expr(kind: BranchKind) -> &'static str {
    match kind {
        BranchKind::Unconditional => "true",
        BranchKind::Scc0 => "!scc",
        BranchKind::Scc1 => "scc",
        BranchKind::Vccz => "vcc == 0u",
        BranchKind::Vccnz => "vcc != 0u",
        BranchKind::Execz => "exec_mask == 0u",
        BranchKind::Execnz => "exec_mask != 0u",
    }
}

/// How control leaves a basic block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Terminator {
    /// No control instruction: execution continues into the next block.
    Fallthrough(BlockId),
    /// `s_branch` to a known block.
    Branch(BlockId),
    /// `s_cbranch_*`: `target` when the condition holds, else `fallthrough`.
    CondBranch {
        kind: BranchKind,
        target: BlockId,
        fallthrough: BlockId,
    },
    /// Program end (`s_endpgm*`, indirect `s_setpc_b64`, invalid unconditional
    /// branch, or falling off the end of the decoded stream).
    End,
}

impl Terminator {
    /// Successor blocks in (target, fallthrough) order.
    pub fn successors(&self) -> Vec<BlockId> {
        match *self {
            Self::Fallthrough(t) | Self::Branch(t) => vec![t],
            Self::CondBranch {
                target,
                fallthrough,
                ..
            } => {
                if target == fallthrough {
                    vec![target]
                } else {
                    vec![target, fallthrough]
                }
            }
            Self::End => Vec::new(),
        }
    }
}

/// A maximal straight-line run of instructions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BasicBlock {
    pub id: BlockId,
    /// Byte offset of the first instruction (end-of-program offset for the
    /// synthetic exit block).
    pub start_offset: usize,
    /// Instruction indices (into the slice given to
    /// [`ControlFlowGraph::from_instructions`]) covered by this block.
    pub instr_range: Range<usize>,
    pub terminator: Terminator,
    /// Index of the control instruction (branch / `s_endpgm` / `s_setpc_b64`)
    /// that ends this block, if any. It is always the last instruction.
    pub control_instr: Option<usize>,
    /// `true` for the instruction-less block that invalid branch targets and
    /// end-of-stream fallthroughs are routed to.
    pub synthetic_exit: bool,
}

impl BasicBlock {
    /// Instructions to lower as straight-line code: `instr_range` minus the
    /// trailing control instruction (which [`emit_wgsl_control_flow`] emits).
    pub fn body_range(&self) -> Range<usize> {
        match self.control_instr {
            Some(ci) if ci + 1 == self.instr_range.end && ci >= self.instr_range.start => {
                self.instr_range.start..ci
            }
            _ => self.instr_range.clone(),
        }
    }

    pub fn len(&self) -> usize {
        self.instr_range.len()
    }

    pub fn is_empty(&self) -> bool {
        self.instr_range.is_empty()
    }
}

/// Non-fatal issue found while building the CFG. None of these abort lifting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CfgDiagnostic {
    /// Branch target lies outside the decoded instruction stream.
    BranchTargetOutOfRange {
        instr_index: usize,
        offset: usize,
        target: i64,
    },
    /// Branch target lies inside the stream but not on an instruction boundary
    /// (e.g. the literal dword of a 64-bit instruction).
    BranchTargetMisaligned {
        instr_index: usize,
        offset: usize,
        target: i64,
    },
    /// Conditional branch or plain instruction falls off the end of the stream.
    FallthroughPastEnd { instr_index: usize, offset: usize },
    /// Indirect jump (`s_setpc_b64`): treated as program end.
    IndirectBranch {
        instr_index: usize,
        offset: usize,
        mnemonic: String,
    },
    /// Branch-like instruction not lifted (calls, debug branches, subvector
    /// loops): treated as a non-branching instruction.
    UnsupportedBranch {
        instr_index: usize,
        offset: usize,
        mnemonic: String,
    },
}

/// Per-instruction control classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Control {
    None,
    End,
    Jump {
        kind: BranchKind,
        target: Option<usize>,
    },
}

/// Control-flow graph over a decoded RDNA2 instruction stream.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ControlFlowGraph {
    /// Blocks in address order; block 0 is the entry. A synthetic exit block,
    /// when present, is last.
    pub blocks: Vec<BasicBlock>,
    /// Synthetic program-end block targeted by invalid branches.
    pub exit_block: Option<BlockId>,
    pub diagnostics: Vec<CfgDiagnostic>,
    offset_to_index: BTreeMap<usize, usize>,
    instr_to_block: Vec<BlockId>,
}

impl ControlFlowGraph {
    /// Build the CFG. Never panics, whatever the instruction stream contains.
    pub fn from_instructions(instrs: &[DecodedInstruction]) -> ControlFlowGraph {
        let mut cfg = ControlFlowGraph::default();
        let n = instrs.len();
        if n == 0 {
            return cfg;
        }

        for (idx, inst) in instrs.iter().enumerate() {
            cfg.offset_to_index.entry(inst.offset).or_insert(idx);
        }
        let first_offset = instrs.iter().map(|i| i.offset).min().unwrap_or(0);
        let end_offset = instrs
            .iter()
            .map(|i| i.offset.saturating_add(i.size.max(1)))
            .max()
            .unwrap_or(0);

        // 1. Classify every instruction.
        let mut controls = Vec::with_capacity(n);
        for (idx, inst) in instrs.iter().enumerate() {
            let control = match &inst.detail {
                Rdna2Detail::Sopp {
                    op, branch_target, ..
                } => match *op {
                    SOPP_ENDPGM
                    | SOPP_ENDPGM_SAVED
                    | SOPP_ENDPGM_ORDERED_PS_DONE
                    | SOPP_CODE_END => Control::End,
                    SOPP_CBRANCH_CDBG_FIRST..=SOPP_CBRANCH_CDBG_LAST => {
                        cfg.diagnostics.push(CfgDiagnostic::UnsupportedBranch {
                            instr_index: idx,
                            offset: inst.offset,
                            mnemonic: inst.mnemonic.clone(),
                        });
                        Control::None
                    }
                    other => match BranchKind::from_sopp_op(other) {
                        Some(kind) => {
                            let target = branch_target.and_then(|t| {
                                cfg.resolve_target(idx, inst.offset, t, first_offset, end_offset)
                            });
                            Control::Jump { kind, target }
                        }
                        None => Control::None,
                    },
                },
                Rdna2Detail::Sopk {
                    branch_target: Some(_),
                    ..
                } => {
                    cfg.diagnostics.push(CfgDiagnostic::UnsupportedBranch {
                        instr_index: idx,
                        offset: inst.offset,
                        mnemonic: inst.mnemonic.clone(),
                    });
                    Control::None
                }
                Rdna2Detail::Sop1 { .. } if inst.mnemonic == "s_setpc_b64" => {
                    cfg.diagnostics.push(CfgDiagnostic::IndirectBranch {
                        instr_index: idx,
                        offset: inst.offset,
                        mnemonic: inst.mnemonic.clone(),
                    });
                    Control::End
                }
                Rdna2Detail::Sop1 { .. } if inst.mnemonic == "s_swappc_b64" => {
                    cfg.diagnostics.push(CfgDiagnostic::UnsupportedBranch {
                        instr_index: idx,
                        offset: inst.offset,
                        mnemonic: inst.mnemonic.clone(),
                    });
                    Control::None
                }
                _ => Control::None,
            };
            controls.push(control);
        }

        // 2. Leaders: entry, valid branch targets, instruction after any control op.
        let mut leader = vec![false; n];
        leader[0] = true;
        for (idx, control) in controls.iter().enumerate() {
            match *control {
                Control::None => {}
                Control::End => {
                    if idx + 1 < n {
                        leader[idx + 1] = true;
                    }
                }
                Control::Jump { target, .. } => {
                    if let Some(t) = target {
                        leader[t] = true;
                    }
                    if idx + 1 < n {
                        leader[idx + 1] = true;
                    }
                }
            }
        }

        // 3. Block boundaries.
        let starts: Vec<usize> = (0..n).filter(|&i| leader[i]).collect();
        let real_blocks = starts.len();
        let exit_id = real_blocks;
        let mut instr_to_block = vec![0usize; n];
        for (bid, &start) in starts.iter().enumerate() {
            let end = starts.get(bid + 1).copied().unwrap_or(n);
            for slot in &mut instr_to_block[start..end] {
                *slot = bid;
            }
        }

        // 4. Terminators.
        let mut needs_exit = false;
        let mut blocks = Vec::with_capacity(real_blocks + 1);
        for (bid, &start) in starts.iter().enumerate() {
            let end = starts.get(bid + 1).copied().unwrap_or(n);
            let last = end - 1;
            let next_block = if bid + 1 < real_blocks {
                bid + 1
            } else {
                needs_exit = true;
                exit_id
            };
            let falls_off = bid + 1 >= real_blocks;
            let (terminator, control_instr) = match controls[last] {
                Control::None => {
                    if falls_off {
                        cfg.diagnostics.push(CfgDiagnostic::FallthroughPastEnd {
                            instr_index: last,
                            offset: instrs[last].offset,
                        });
                    }
                    (Terminator::Fallthrough(next_block), None)
                }
                Control::End => (Terminator::End, Some(last)),
                Control::Jump {
                    kind: BranchKind::Unconditional,
                    target,
                } => match target {
                    Some(t) => (Terminator::Branch(instr_to_block[t]), Some(last)),
                    None => (Terminator::End, Some(last)),
                },
                Control::Jump { kind, target } => {
                    if falls_off {
                        cfg.diagnostics.push(CfgDiagnostic::FallthroughPastEnd {
                            instr_index: last,
                            offset: instrs[last].offset,
                        });
                    }
                    let target = match target {
                        Some(t) => instr_to_block[t],
                        None => {
                            needs_exit = true;
                            exit_id
                        }
                    };
                    (
                        Terminator::CondBranch {
                            kind,
                            target,
                            fallthrough: next_block,
                        },
                        Some(last),
                    )
                }
            };
            blocks.push(BasicBlock {
                id: bid,
                start_offset: instrs[start].offset,
                instr_range: start..end,
                terminator,
                control_instr,
                synthetic_exit: false,
            });
        }
        // A block that merely falls into the exit has no successor worth keeping.
        for block in &mut blocks {
            if block.terminator == Terminator::Fallthrough(exit_id) {
                block.terminator = Terminator::End;
            }
        }
        needs_exit = needs_exit
            && blocks
                .iter()
                .any(|b| b.terminator.successors().contains(&exit_id));
        if needs_exit {
            blocks.push(BasicBlock {
                id: exit_id,
                start_offset: end_offset,
                instr_range: n..n,
                terminator: Terminator::End,
                control_instr: None,
                synthetic_exit: true,
            });
            cfg.exit_block = Some(exit_id);
        }

        cfg.blocks = blocks;
        cfg.instr_to_block = instr_to_block;
        cfg
    }

    fn resolve_target(
        &mut self,
        instr_index: usize,
        offset: usize,
        target: i64,
        first_offset: usize,
        end_offset: usize,
    ) -> Option<usize> {
        let in_range =
            usize::try_from(target).is_ok_and(|t| t >= first_offset && t < end_offset);
        let found = usize::try_from(target)
            .ok()
            .and_then(|t| self.offset_to_index.get(&t).copied());
        match found {
            Some(idx) => Some(idx),
            None => {
                self.diagnostics.push(if in_range {
                    CfgDiagnostic::BranchTargetMisaligned {
                        instr_index,
                        offset,
                        target,
                    }
                } else {
                    CfgDiagnostic::BranchTargetOutOfRange {
                        instr_index,
                        offset,
                        target,
                    }
                });
                None
            }
        }
    }

    /// Entry block (always block 0 for a non-empty CFG).
    pub fn entry(&self) -> Option<BlockId> {
        if self.blocks.is_empty() {
            None
        } else {
            Some(0)
        }
    }

    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Instruction index starting exactly at byte `offset`.
    pub fn instruction_index_at(&self, offset: i64) -> Option<usize> {
        usize::try_from(offset)
            .ok()
            .and_then(|o| self.offset_to_index.get(&o).copied())
    }

    /// Block containing instruction `instr_index`.
    pub fn block_of_instruction(&self, instr_index: usize) -> Option<BlockId> {
        self.instr_to_block.get(instr_index).copied()
    }

    /// Block starting exactly at byte `offset`.
    pub fn block_at_offset(&self, offset: i64) -> Option<BlockId> {
        let idx = self.instruction_index_at(offset)?;
        let bid = self.block_of_instruction(idx)?;
        (self.blocks.get(bid)?.instr_range.start == idx).then_some(bid)
    }

    /// Predecessor lists indexed by block id.
    pub fn predecessors(&self) -> Vec<Vec<BlockId>> {
        let mut preds = vec![Vec::new(); self.blocks.len()];
        for block in &self.blocks {
            for succ in block.terminator.successors() {
                if let Some(list) = preds.get_mut(succ) {
                    list.push(block.id);
                }
            }
        }
        preds
    }

    /// Blocks reachable from the entry, indexed by block id.
    pub fn reachable(&self) -> Vec<bool> {
        let mut seen = vec![false; self.blocks.len()];
        let mut stack: Vec<BlockId> = self.entry().into_iter().collect();
        while let Some(b) = stack.pop() {
            match seen.get_mut(b) {
                Some(s) if !*s => *s = true,
                _ => continue,
            }
            stack.extend(self.blocks[b].terminator.successors());
        }
        seen
    }

    /// Compute immediate dominators (`doms[b]`) over all reachable blocks via iterative
    /// Cooper-Harvey-Kennedy dataflow over reverse postorder.
    ///
    /// `doms[entry] == Some(entry)`; unreachable blocks map to `None`.
    pub fn dominators(&self) -> Vec<Option<BlockId>> {
        let n = self.blocks.len();
        let mut doms = vec![None; n];
        let Some(entry) = self.entry() else {
            return doms;
        };

        // Iterative DFS to compute postorder, then reverse for RPO.
        let mut visited = vec![false; n];
        let mut postorder = Vec::with_capacity(n);
        let mut stack: Vec<(BlockId, usize)> = vec![(entry, 0)];
        visited[entry] = true;
        while let Some((b, next_idx)) = stack.pop() {
            let succs = self.blocks[b].terminator.successors();
            if next_idx < succs.len() {
                stack.push((b, next_idx + 1));
                let s = succs[next_idx];
                if s < n && !visited[s] {
                    visited[s] = true;
                    stack.push((s, 0));
                }
            } else {
                postorder.push(b);
            }
        }
        postorder.reverse();
        let rpo = postorder;

        let mut rpo_num = vec![usize::MAX; n];
        for (pos, &b) in rpo.iter().enumerate() {
            rpo_num[b] = pos;
        }

        let preds = self.predecessors();
        doms[entry] = Some(entry);

        let intersect = |mut b1: BlockId, mut b2: BlockId, doms: &[Option<BlockId>]| -> BlockId {
            let mut guard = 0usize;
            let limit = n.saturating_mul(2).saturating_add(4);
            while b1 != b2 && guard < limit {
                guard += 1;
                while rpo_num[b1] > rpo_num[b2] {
                    match doms[b1] {
                        Some(d) if d != b1 => b1 = d,
                        _ => return entry,
                    }
                }
                while rpo_num[b2] > rpo_num[b1] {
                    match doms[b2] {
                        Some(d) if d != b2 => b2 = d,
                        _ => return entry,
                    }
                }
            }
            b1
        };

        let mut changed = true;
        let max_passes = n.saturating_add(4);
        let mut passes = 0usize;
        while changed && passes < max_passes {
            changed = false;
            passes += 1;
            for &b in &rpo {
                if b == entry {
                    continue;
                }
                let mut new_idom: Option<BlockId> = None;
                for &p in &preds[b] {
                    if p < n && doms[p].is_some() {
                        new_idom = Some(match new_idom {
                            None => p,
                            Some(cur) => intersect(p, cur, &doms),
                        });
                    }
                }
                if doms[b] != new_idom && new_idom.is_some() {
                    doms[b] = new_idom;
                    changed = true;
                }
            }
        }
        doms
    }

    /// Returns `true` iff block `a` dominates block `b` under a precomputed [`Self::dominators`]
    /// table. Unreachable blocks are not dominated by anything.
    pub fn dominates(&self, doms: &[Option<BlockId>], a: BlockId, b: BlockId) -> bool {
        if a >= self.blocks.len() || b >= self.blocks.len() || doms.get(b).copied().flatten().is_none()
        {
            return false;
        }
        let mut cur = b;
        for _ in 0..=self.blocks.len() {
            if cur == a {
                return true;
            }
            match doms.get(cur).copied().flatten() {
                Some(parent) if parent != cur => cur = parent,
                _ => return false,
            }
        }
        false
    }
}

// ---------------------------------------------------------------------------
// Exec-mask instruction classification
// ---------------------------------------------------------------------------

/// SOP1 exec-mask manipulation ops that set up divergent RDNA2 regions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExecMaskOp {
    AndSaveExec,
    OrSaveExec,
    XorSaveExec,
    AndN2SaveExec,
    OrN2SaveExec,
    NandSaveExec,
    NorSaveExec,
    XnorSaveExec,
    AndN1SaveExec,
    OrN1SaveExec,
    AndN1WrExec,
    AndN2WrExec,
}

/// Exec-mask write classification of a decoded instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExecMaskWrite {
    pub op: ExecMaskOp,
    /// `_b64` (wave64) form.
    pub wave64: bool,
}

/// Identify `s_*_saveexec_b32/b64` and `s_andn{1,2}_wrexec_b32/b64`.
///
/// These never terminate a basic block; they only rewrite `EXEC` (and save
/// the old mask), typically right before an `s_cbranch_execz` skip.
pub fn classify_exec_mask_op(inst: &DecodedInstruction) -> Option<ExecMaskWrite> {
    if !matches!(inst.detail, Rdna2Detail::Sop1 { .. }) {
        return None;
    }
    let (stem, wave64) = if let Some(s) = inst.mnemonic.strip_suffix("_b64") {
        (s, true)
    } else if let Some(s) = inst.mnemonic.strip_suffix("_b32") {
        (s, false)
    } else {
        return None;
    };
    let op = match stem {
        "s_and_saveexec" => ExecMaskOp::AndSaveExec,
        "s_or_saveexec" => ExecMaskOp::OrSaveExec,
        "s_xor_saveexec" => ExecMaskOp::XorSaveExec,
        "s_andn2_saveexec" => ExecMaskOp::AndN2SaveExec,
        "s_orn2_saveexec" => ExecMaskOp::OrN2SaveExec,
        "s_nand_saveexec" => ExecMaskOp::NandSaveExec,
        "s_nor_saveexec" => ExecMaskOp::NorSaveExec,
        "s_xnor_saveexec" => ExecMaskOp::XnorSaveExec,
        "s_andn1_saveexec" => ExecMaskOp::AndN1SaveExec,
        "s_orn1_saveexec" => ExecMaskOp::OrN1SaveExec,
        "s_andn1_wrexec" => ExecMaskOp::AndN1WrExec,
        "s_andn2_wrexec" => ExecMaskOp::AndN2WrExec,
        _ => return None,
    };
    Some(ExecMaskWrite { op, wave64 })
}

// ---------------------------------------------------------------------------
// Structured control-flow tree
// ---------------------------------------------------------------------------

/// Structured control-flow tree lifted from a [`ControlFlowGraph`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StructuredNode {
    /// Straight-line body of a basic block.
    Block(BlockId),
    /// Execute children in order.
    Seq(Vec<StructuredNode>),
    /// Two-way branch on the condition evaluated at the end of `cond_block`:
    /// `then_branch` runs when `cond` holds (note `cond` may be the negation
    /// of the block's own branch kind), `else_branch` otherwise.
    If {
        cond_block: BlockId,
        cond: BranchKind,
        then_branch: Box<StructuredNode>,
        else_branch: Option<Box<StructuredNode>>,
    },
    /// Repeats `body`; leave via [`StructuredNode::Break`] / `Return`.
    Loop { body: Box<StructuredNode> },
    /// Exit the innermost enclosing `Loop`.
    Break,
    /// Restart the innermost enclosing `Loop`.
    Continue,
    /// Program end.
    Return,
    /// Universal fallback for irreducible / multi-exit regions: a bounded
    /// program-counter dispatcher over `blocks`, starting at `entry`. Edges to
    /// blocks outside `blocks` leave the dispatcher (to the code after it), and
    /// edges to the synthetic exit block return.
    Dispatch {
        entry: BlockId,
        blocks: Vec<BlockId>,
        max_iterations: u32,
    },
}

impl StructuredNode {
    /// Number of `Dispatch` fallbacks in the tree (iterative, stack-safe).
    pub fn dispatch_count(&self) -> usize {
        let mut count = 0usize;
        let mut stack = vec![self];
        while let Some(node) = stack.pop() {
            match node {
                Self::Dispatch { .. } => count += 1,
                Self::Seq(nodes) => stack.extend(nodes.iter()),
                Self::If {
                    then_branch,
                    else_branch,
                    ..
                } => {
                    stack.push(then_branch);
                    if let Some(e) = else_branch {
                        stack.push(e);
                    }
                }
                Self::Loop { body } => stack.push(body),
                Self::Block(_) | Self::Break | Self::Continue | Self::Return => {}
            }
        }
        count
    }

    /// `true` when no `Dispatch` fallback was needed.
    pub fn is_fully_structured(&self) -> bool {
        self.dispatch_count() == 0
    }

    fn escapes(&self) -> bool {
        self.escapes_bounded(0)
    }

    fn escapes_bounded(&self, depth: usize) -> bool {
        if depth > MAX_STRUCTURE_DEPTH * 4 {
            return false;
        }
        match self {
            Self::Break | Self::Continue | Self::Return => true,
            Self::Seq(nodes) => nodes
                .last()
                .is_some_and(|n| n.escapes_bounded(depth + 1)),
            Self::If {
                then_branch,
                else_branch: Some(else_branch),
                ..
            } => {
                then_branch.escapes_bounded(depth + 1)
                    && else_branch.escapes_bounded(depth + 1)
            }
            _ => false,
        }
    }
}

fn seq(mut nodes: Vec<StructuredNode>) -> StructuredNode {
    if nodes.len() == 1 {
        nodes.pop().unwrap_or(StructuredNode::Seq(Vec::new()))
    } else {
        StructuredNode::Seq(nodes)
    }
}

/// Lift a CFG into a structured tree. Always succeeds: unstructurable regions
/// become [`StructuredNode::Dispatch`] fallbacks.
pub fn structure_cfg(cfg: &ControlFlowGraph) -> StructuredNode {
    let Some(mut s) = Structurer::new(cfg) else {
        return StructuredNode::Seq(Vec::new());
    };
    let n = s.order.len();
    match s.region(0, n, None, None, false, 0) {
        Ok(nodes) => seq(nodes),
        Err(_) => s.dispatch(0, n),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Fail {
    /// This shape cannot be structured here (a caller may fall back).
    Shape,
    /// Global work budget exhausted: abort to a whole-program dispatch.
    Budget,
}

/// Edge target in structuring position space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tgt {
    /// The synthetic exit block (program end).
    Exit,
    /// Position in `Structurer::order`.
    Pos(usize),
}

#[derive(Debug, Clone, Copy)]
struct LoopCtx {
    header: usize,
    exit: Option<usize>,
}

struct Structurer<'a> {
    cfg: &'a ControlFlowGraph,
    /// Reachable, non-synthetic blocks in address order.
    order: Vec<BlockId>,
    pos_of: Vec<Option<usize>>,
    /// Predecessor positions, per position.
    preds: Vec<Vec<usize>>,
    budget: usize,
}

impl<'a> Structurer<'a> {
    fn new(cfg: &'a ControlFlowGraph) -> Option<Self> {
        let reachable = cfg.reachable();
        let order: Vec<BlockId> = cfg
            .blocks
            .iter()
            .enumerate()
            .filter(|&(idx, b)| {
                b.id == idx && reachable.get(idx).copied().unwrap_or(false) && !b.synthetic_exit
            })
            .map(|(idx, _)| idx)
            .collect();
        if order.is_empty() {
            return None;
        }
        let mut pos_of = vec![None; cfg.blocks.len()];
        for (p, &b) in order.iter().enumerate() {
            pos_of[b] = Some(p);
        }
        let mut preds = vec![Vec::new(); order.len()];
        for (p, &b) in order.iter().enumerate() {
            for succ in cfg.blocks[b].terminator.successors() {
                if let Some(Some(sp)) = pos_of.get(succ) {
                    preds[*sp].push(p);
                }
            }
        }
        let budget = order
            .len()
            .saturating_mul(32)
            .saturating_add(256)
            .min(MAX_STRUCTURE_BUDGET);
        Some(Self {
            cfg,
            order,
            pos_of,
            preds,
            budget,
        })
    }

    fn single_entry(&self, lo: usize, hi: usize) -> bool {
        (lo + 1..hi).all(|p| self.preds[p].iter().all(|&q| q >= lo && q < hi))
    }

    fn tgt(&self, block: BlockId) -> Result<Tgt, Fail> {
        if Some(block) == self.cfg.exit_block {
            return Ok(Tgt::Exit);
        }
        match self.pos_of.get(block) {
            Some(Some(p)) => Ok(Tgt::Pos(*p)),
            _ => Err(Fail::Shape),
        }
    }

    fn succ_tgts(&self, pos: usize) -> Vec<Tgt> {
        self.cfg.blocks[self.order[pos]]
            .terminator
            .successors()
            .into_iter()
            .filter_map(|b| self.tgt(b).ok())
            .collect()
    }

    /// Resolve an edge leaving the node that ends at position `q` of region
    /// `[.., hi)` (whose successor is `follow`): `Ok(None)` = falls into the
    /// next node, `Ok(Some(escape))` = needs an explicit statement.
    fn resolve(
        &self,
        x: Tgt,
        q: usize,
        hi: usize,
        follow: Option<usize>,
        lp: Option<LoopCtx>,
    ) -> Result<Option<StructuredNode>, Fail> {
        let p = match x {
            Tgt::Exit => return Ok(Some(StructuredNode::Return)),
            Tgt::Pos(p) => p,
        };
        if q < hi && p == q {
            return Ok(None);
        }
        if q >= hi && Some(p) == follow {
            return Ok(None);
        }
        if let Some(l) = lp {
            if p == l.header {
                return Ok(Some(StructuredNode::Continue));
            }
            if Some(p) == l.exit {
                return Ok(Some(StructuredNode::Break));
            }
        }
        Err(Fail::Shape)
    }

    /// Structure positions `[lo, hi)`, entered at `lo`, leaving to `follow`.
    fn region(
        &mut self,
        lo: usize,
        hi: usize,
        follow: Option<usize>,
        lp: Option<LoopCtx>,
        header_at_lo: bool,
        depth: usize,
    ) -> Result<Vec<StructuredNode>, Fail> {
        if self.budget == 0 {
            return Err(Fail::Budget);
        }
        self.budget -= 1;
        if depth > MAX_STRUCTURE_DEPTH || lo >= hi || hi > self.order.len() {
            return Err(Fail::Shape);
        }
        // Single entry: interior positions may only be reached from inside.
        if !self.single_entry(lo, hi) {
            return Err(Fail::Shape);
        }

        let mut nodes = Vec::new();
        // (start position, `nodes.len()` before that node) for backtracking.
        let mut starts: Vec<(usize, usize)> = Vec::new();
        let mut i = lo;
        while i < hi {
            starts.push((i, nodes.len()));
            match self.node_at(i, hi, follow, lp, header_at_lo && i == lo, depth) {
                Ok((mut ns, end)) => {
                    if end <= i || end > hi {
                        return Err(Fail::Shape);
                    }
                    nodes.append(&mut ns);
                    i = end;
                }
                Err(Fail::Shape) => {
                    // Dispatch the smallest suffix `[j, hi)` (j = a node start)
                    // that is single-entry and whose edges stay inside / leave to
                    // `follow` / end the program. When `header_at_lo && j == lo`,
                    // let the enclosing loop handler emit `Dispatch` directly
                    // without a redundant `loop { dispatch; break; }` wrapper.
                    let Some(&(j, len)) = starts.iter().rev().find(|&&(j, _)| {
                        !(header_at_lo && j == lo)
                            && self.single_entry(j, hi)
                            && self.dispatchable(j, hi, follow)
                    }) else {
                        return Err(Fail::Shape);
                    };
                    nodes.truncate(len);
                    nodes.push(self.dispatch(j, hi));
                    i = hi;
                }
                Err(e) => return Err(e),
            }
        }
        Ok(nodes)
    }

    /// Structure the node starting at position `i`; returns its statements and
    /// the position right after it.
    fn node_at(
        &mut self,
        i: usize,
        hi: usize,
        follow: Option<usize>,
        lp: Option<LoopCtx>,
        is_loop_header: bool,
        depth: usize,
    ) -> Result<(Vec<StructuredNode>, usize), Fail> {
        // Natural loop headed at `i`: back edges from `[i, hi)`.
        if !is_loop_header && let Some(&latch) = self.preds[i].iter().filter(|&&p| p >= i).max() {
            if latch >= hi {
                return Err(Fail::Shape);
            }
            let end = latch + 1;
            let nf = if end < hi { Some(end) } else { follow };
            let ctx = LoopCtx {
                header: i,
                exit: nf,
            };
            match self.region(i, end, nf, Some(ctx), true, depth + 1) {
                Ok(mut body) => {
                    if !body.last().is_some_and(StructuredNode::escapes) {
                        body.push(StructuredNode::Break);
                    }
                    let node = StructuredNode::Loop {
                        body: Box::new(seq(body)),
                    };
                    return Ok((vec![node], end));
                }
                Err(Fail::Shape) if self.single_entry(i, end) && self.dispatchable(i, end, nf) => {
                    return Ok((vec![self.dispatch(i, end)], end));
                }
                Err(e) => return Err(e),
            }
        }

        let bid = self.order[i];
        let terminator = self.cfg.blocks[bid].terminator;
        let q = i + 1;
        let mut out = vec![StructuredNode::Block(bid)];

        let (kind, target, fallthrough) = match terminator {
            Terminator::End => {
                out.push(StructuredNode::Return);
                return Ok((out, q));
            }
            Terminator::Fallthrough(t) | Terminator::Branch(t) => {
                let tg = self.tgt(t)?;
                if let Some(esc) = self.resolve(tg, q, hi, follow, lp)? {
                    out.push(esc);
                }
                return Ok((out, q));
            }
            Terminator::CondBranch {
                kind,
                target,
                fallthrough,
            } => (kind, target, fallthrough),
        };

        let tg = self.tgt(target)?;
        let fg = self.tgt(fallthrough)?;
        if tg == fg || !kind.is_conditional() {
            if let Some(esc) = self.resolve(tg, q, hi, follow, lp)? {
                out.push(esc);
            }
            return Ok((out, q));
        }
        let neg = kind.negate().ok_or(Fail::Shape)?;

        // Both edges resolve locally (escapes / fall into next node).
        let rt = self.resolve(tg, q, hi, follow, lp);
        let rf = self.resolve(fg, q, hi, follow, lp);
        if let (Ok(et), Ok(ef)) = (&rt, &rf) {
            let node = match (et.clone(), ef.clone()) {
                (Some(et), None) => Some(StructuredNode::If {
                    cond_block: bid,
                    cond: kind,
                    then_branch: Box::new(et),
                    else_branch: None,
                }),
                (None, Some(ef)) => Some(StructuredNode::If {
                    cond_block: bid,
                    cond: neg,
                    then_branch: Box::new(ef),
                    else_branch: None,
                }),
                (Some(et), Some(ef)) => Some(StructuredNode::If {
                    cond_block: bid,
                    cond: kind,
                    then_branch: Box::new(et),
                    else_branch: Some(Box::new(ef)),
                }),
                (None, None) => None,
            };
            out.extend(node);
            return Ok((out, q));
        }

        // Forward branch over a structured region: if-skip or if/else diamond.
        let tp = match (tg, fg) {
            (Tgt::Pos(tp), Tgt::Pos(fp)) if fp == q && tp > q => tp,
            _ => return Err(Fail::Shape),
        };
        let (end, nf) = if tp < hi {
            (tp, Some(tp))
        } else if Some(tp) == follow {
            (hi, follow)
        } else {
            return Err(Fail::Shape);
        };

        // Detect an `if/else` diamond upfront when the last block of `[q, tp)`
        // unconditionally jumps past `tp` to a merge target that is not an
        // enclosing loop exit. Skipping the doomed `if-skip` probe eliminates
        // O(2^D) exponential backtracking on deeply nested diamonds.
        let diamond_merge = if tp < hi {
            let then_last = self.order[tp - 1];
            match self.cfg.blocks[then_last].terminator {
                Terminator::Branch(merge) => match self.tgt(merge).ok() {
                    Some(Tgt::Pos(mp)) if mp > tp && mp < hi => Some((mp, Some(mp))),
                    Some(Tgt::Pos(mp)) if mp > tp && Some(mp) == follow => Some((hi, follow)),
                    _ => None,
                },
                _ => None,
            }
        } else {
            None
        };
        let skip_impossible = diamond_merge
            .is_some_and(|(_, nf2)| nf2 != lp.and_then(|l| l.exit));

        // if (!taken) { fallthrough region }
        if !skip_impossible {
            let skip = self.region(q, end, nf, lp, false, depth + 1);
            match skip {
                Ok(then_nodes) => {
                    out.push(StructuredNode::If {
                        cond_block: bid,
                        cond: neg,
                        then_branch: Box::new(seq(then_nodes)),
                        else_branch: None,
                    });
                    return Ok((out, end));
                }
                Err(Fail::Budget) => return Err(Fail::Budget),
                Err(Fail::Shape) => {}
            }
        }

        // Diamond: then-region `[q, tp)` ends with `s_branch merge`.
        let Some((else_end, nf2)) = diamond_merge else {
            return Err(Fail::Shape);
        };
        let then_nodes = self.region(q, tp, nf2, lp, false, depth + 1)?;
        let else_nodes = self.region(tp, else_end, nf2, lp, false, depth + 1)?;
        out.push(StructuredNode::If {
            cond_block: bid,
            cond: neg,
            then_branch: Box::new(seq(then_nodes)),
            else_branch: Some(Box::new(seq(else_nodes))),
        });
        Ok((out, else_end))
    }

    /// `[i, hi)` can become a dispatcher iff every edge stays inside, leaves
    /// to `follow`, or ends the program.
    fn dispatchable(&self, i: usize, hi: usize, follow: Option<usize>) -> bool {
        (i..hi).all(|p| {
            self.succ_tgts(p).into_iter().all(|t| match t {
                Tgt::Exit => true,
                Tgt::Pos(x) => (i..hi).contains(&x) || Some(x) == follow,
            })
        })
    }

    fn dispatch(&self, i: usize, hi: usize) -> StructuredNode {
        let blocks = self.order[i..hi].to_vec();
        StructuredNode::Dispatch {
            entry: blocks.first().copied().unwrap_or(0),
            blocks,
            max_iterations: DEFAULT_DISPATCH_MAX_ITERATIONS,
        }
    }
}

// ---------------------------------------------------------------------------
// WGSL emission
// ---------------------------------------------------------------------------

/// Emit `node` as WGSL function-body statements (indented one level).
///
/// `emit_block` returns the straight-line WGSL for a block's
/// [`BasicBlock::body_range`]; branch conditions read the `scc`, `vcc` and
/// `exec_mask` variables (see [`cond_expr`]). Every emitted `loop` is
/// iteration-bounded, so the output always terminates.
pub fn emit_wgsl_control_flow<F>(
    node: &StructuredNode,
    cfg: &ControlFlowGraph,
    emit_block: F,
) -> String
where
    F: FnMut(&BasicBlock) -> String,
{
    emit_wgsl_control_flow_indented(node, cfg, 1, emit_block)
}

/// [`emit_wgsl_control_flow`] with an explicit base indentation level.
pub fn emit_wgsl_control_flow_indented<F>(
    node: &StructuredNode,
    cfg: &ControlFlowGraph,
    indent: usize,
    emit_block: F,
) -> String
where
    F: FnMut(&BasicBlock) -> String,
{
    let mut emitter = Emitter {
        cfg,
        emit_block,
        out: String::new(),
        loop_counter: 0,
    };
    emitter.node(node, indent, 0);
    emitter.out
}

struct Emitter<'a, F> {
    cfg: &'a ControlFlowGraph,
    emit_block: F,
    out: String,
    loop_counter: usize,
}

impl<F> Emitter<'_, F>
where
    F: FnMut(&BasicBlock) -> String,
{
    fn line(&mut self, indent: usize, text: &str) {
        for _ in 0..indent {
            self.out.push_str("    ");
        }
        self.out.push_str(text);
        self.out.push('\n');
    }

    fn block_body(&mut self, id: BlockId, indent: usize) {
        let Some(block) = self.cfg.blocks.get(id) else {
            self.line(indent, &format!("// bb{id}: <invalid block>"));
            return;
        };
        self.line(indent, &format!("// bb{id} @ 0x{:x}", block.start_offset));
        let text = (self.emit_block)(block);
        for l in text.lines() {
            let l = l.trim_end();
            if !l.is_empty() {
                self.line(indent, l);
            }
        }
    }

    fn node(&mut self, node: &StructuredNode, indent: usize, depth: usize) {
        if depth > MAX_STRUCTURE_DEPTH * 4 {
            self.line(indent, "return; // nesting limit");
            return;
        }
        match node {
            StructuredNode::Block(id) => self.block_body(*id, indent),
            StructuredNode::Seq(nodes) => {
                for n in nodes {
                    self.node(n, indent, depth + 1);
                }
            }
            StructuredNode::If {
                cond,
                then_branch,
                else_branch,
                ..
            } => {
                self.line(indent, &format!("if ({}) {{", cond_expr(*cond)));
                self.node(then_branch, indent + 1, depth + 1);
                if let Some(else_branch) = else_branch {
                    self.line(indent, "} else {");
                    self.node(else_branch, indent + 1, depth + 1);
                }
                self.line(indent, "}");
            }
            StructuredNode::Loop { body } => {
                let guard = format!("_pg6_loop{}", self.loop_counter);
                self.loop_counter += 1;
                self.line(indent, &format!("var {guard}: u32 = 0u;"));
                self.line(indent, "loop {");
                self.line(
                    indent + 1,
                    &format!("if ({guard} >= {STRUCTURED_LOOP_MAX_ITERATIONS}u) {{ break; }}"),
                );
                self.line(indent + 1, &format!("{guard} = {guard} + 1u;"));
                self.node(body, indent + 1, depth + 1);
                self.line(indent, "}");
            }
            StructuredNode::Break => self.line(indent, "break;"),
            StructuredNode::Continue => self.line(indent, "continue;"),
            StructuredNode::Return => self.line(indent, "return;"),
            StructuredNode::Dispatch {
                entry,
                blocks,
                max_iterations,
            } => self.dispatch(*entry, blocks, *max_iterations, indent),
        }
    }

    fn transfer(&self, target: BlockId, members: &BTreeSet<BlockId>) -> String {
        if Some(target) == self.cfg.exit_block {
            "return;".to_string()
        } else if members.contains(&target) {
            match u32::try_from(target) {
                Ok(v) if v != DISPATCH_EXIT_PC => format!("_pg6_pc = {v}u;"),
                _ => format!("_pg6_pc = {DISPATCH_EXIT_PC}u;"),
            }
        } else {
            format!("_pg6_pc = {DISPATCH_EXIT_PC}u;")
        }
    }

    fn dispatch(&mut self, entry: BlockId, blocks: &[BlockId], max_iterations: u32, indent: usize) {
        let members: BTreeSet<BlockId> = blocks
            .iter()
            .copied()
            .filter(|&b| {
                self.cfg.blocks.get(b).is_some_and(|blk| !blk.synthetic_exit)
                    && u32::try_from(b).is_ok_and(|v| v != DISPATCH_EXIT_PC)
            })
            .collect();
        let entry_pc = if members.contains(&entry) {
            u32::try_from(entry).unwrap_or(DISPATCH_EXIT_PC)
        } else {
            DISPATCH_EXIT_PC
        };
        self.line(indent, "{");
        let i1 = indent + 1;
        self.line(i1, &format!("var _pg6_pc: u32 = {entry_pc}u;"));
        self.line(i1, "var _pg6_iter: u32 = 0u;");
        self.line(i1, "loop {");
        let i2 = i1 + 1;
        self.line(
            i2,
            &format!("if (_pg6_iter >= {max_iterations}u) {{ break; }}"),
        );
        self.line(
            i2,
            &format!("if (_pg6_pc == {DISPATCH_EXIT_PC}u) {{ break; }}"),
        );
        self.line(i2, "_pg6_iter = _pg6_iter + 1u;");
        self.line(i2, "switch (_pg6_pc) {");
        let i3 = i2 + 1;
        for &id in &members {
            let Some(block) = self.cfg.blocks.get(id) else {
                continue;
            };
            let terminator = block.terminator;
            self.line(i3, &format!("case {id}u: {{"));
            self.block_body(id, i3 + 1);
            match terminator {
                Terminator::End => self.line(i3 + 1, "return;"),
                Terminator::Fallthrough(t) | Terminator::Branch(t) => {
                    let s = self.transfer(t, &members);
                    self.line(i3 + 1, &s);
                }
                Terminator::CondBranch {
                    kind,
                    target,
                    fallthrough,
                } => {
                    let taken = self.transfer(target, &members);
                    if target == fallthrough || !kind.is_conditional() {
                        self.line(i3 + 1, &taken);
                    } else {
                        let not_taken = self.transfer(fallthrough, &members);
                        self.line(i3 + 1, &format!("if ({}) {{", cond_expr(kind)));
                        self.line(i3 + 2, &taken);
                        self.line(i3 + 1, "} else {");
                        self.line(i3 + 2, &not_taken);
                        self.line(i3 + 1, "}");
                    }
                }
            }
            self.line(i3, "}");
        }
        self.line(i3, "default: {");
        self.line(i3 + 1, &format!("_pg6_pc = {DISPATCH_EXIT_PC}u;"));
        self.line(i3, "}");
        self.line(i2, "}");
        self.line(i1, "}");
        self.line(indent, "}");
    }
}

/// Human-readable one-line summary (for tooling / telemetry).
pub fn summarize(cfg: &ControlFlowGraph, node: &StructuredNode) -> String {
    let mut out = String::new();
    let _ = write!(
        out,
        "blocks={} diagnostics={} dispatch={} structured={}",
        cfg.blocks.len(),
        cfg.diagnostics.len(),
        node.dispatch_count(),
        node.is_fully_structured()
    );
    out
}

// ---------------------------------------------------------------------------
// 3D Compute, Mesh-Shader, Atomic & Barrier CFG Analysis (S69)
// ---------------------------------------------------------------------------

/// Per-basic-block summary of synchronization, LDS, atomic, wave, and mesh export operations.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BlockSyncSummary {
    pub block_id: BlockId,
    /// Instruction indices of `s_barrier` inside this block.
    pub barrier_instrs: Vec<usize>,
    /// Instruction indices of `s_waitcnt*` inside this block.
    pub waitcnt_instrs: Vec<usize>,
    pub lds_read_count: usize,
    pub lds_write_count: usize,
    /// Count of cross-lane DS wave ops (`ds_swizzle_b32`, `ds_permute_b32`, `ds_bpermute_b32`).
    pub lds_wave_count: usize,
    pub buffer_atomic_count: usize,
    pub image_atomic_count: usize,
    pub global_atomic_count: usize,
    pub lds_atomic_count: usize,
    pub wave_op_count: usize,
    /// Instruction indices of `s_sendmsg(GS_ALLOC_REQ)` in this block.
    pub gs_alloc_req_instrs: Vec<usize>,
    /// Instruction indices of `exp prim` in this block.
    pub prim_export_instrs: Vec<usize>,
    /// Instruction indices of `exp pos*` in this block.
    pub pos_export_instrs: Vec<usize>,
    /// Instruction indices of `exp param*` in this block.
    pub param_export_instrs: Vec<usize>,
    /// `true` if this block contains an `s_*_saveexec_*` or `s_andn*_wrexec_*` instruction.
    pub has_exec_mask_write: bool,
}

impl BlockSyncSummary {
    pub const fn total_atomics(&self) -> usize {
        self.buffer_atomic_count
            + self.image_atomic_count
            + self.global_atomic_count
            + self.lds_atomic_count
    }

    pub fn has_barrier(&self) -> bool {
        !self.barrier_instrs.is_empty()
    }

    pub fn has_any_mesh_export(&self) -> bool {
        !self.prim_export_instrs.is_empty()
            || !self.pos_export_instrs.is_empty()
            || !self.param_export_instrs.is_empty()
    }
}

/// Control-flow guard pattern enclosing an `s_sendmsg(GS_ALLOC_REQ)` meshlet allocation point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MeshAllocGuardKind {
    /// Executed unconditionally (entry block or straight-line path without branching).
    Unconditional,
    /// Guarded by a single-lane / EXEC / VCC mask pattern (`s_and_saveexec` + `s_cbranch_execz`,
    /// `v_cmpx_*`, or `s_cbranch_vccz`).
    SingleLaneExecGuard,
    /// Guarded by a wave-uniform scalar branch (`s_cbranch_scc0` / `s_cbranch_scc1`).
    WaveUniformScalarGuard,
    /// Inside another conditional or multi-predecessor control-flow region.
    OtherConditional,
}

/// Analysis record for one `s_sendmsg(GS_ALLOC_REQ)` allocation instruction in a mesh shader.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeshAllocPoint {
    pub block_id: BlockId,
    pub instr_index: usize,
    pub offset: usize,
    pub guard_kind: MeshAllocGuardKind,
    /// `true` iff this allocation (or its immediate single-lane/wave-uniform if-skip guard header)
    /// dominates every mesh export (`exp prim` / `exp pos*` / `exp param*`) in the CFG.
    pub dominates_all_exports: bool,
    pub inside_loop: bool,
}

/// Analysis record for one `s_barrier` workgroup synchronization point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BarrierRegion {
    pub barrier_block: BlockId,
    pub barrier_instr_index: usize,
    pub inside_loop: bool,
    /// `true` when no enclosing divergent `EXEC`/`VCC` branch or un-restored exec-mask narrowing
    /// can cause a subset of lanes to skip this `s_barrier`.
    pub in_uniform_control_flow: bool,
}

/// Structured CFG analysis summary for 3D compute, culling, lighting, shadow, and mesh shaders.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ComputeMeshCfgAnalysis {
    pub blocks: Vec<BlockSyncSummary>,
    pub is_mesh_shader: bool,
    pub mesh_alloc_points: Vec<MeshAllocPoint>,
    /// `true` when there are no mesh exports or every mesh export is dominated by at least one
    /// `GS_ALLOC_REQ` allocation point.
    pub all_exports_dominated_by_alloc: bool,
    pub unallocated_export_blocks: Vec<BlockId>,
    pub barrier_regions: Vec<BarrierRegion>,
    pub atomic_blocks: Vec<BlockId>,
    pub total_buffer_atomics: usize,
    pub total_image_atomics: usize,
    pub total_global_atomics: usize,
    pub total_lds_atomics: usize,
    pub total_barriers: usize,
    pub total_waitcnts: usize,
    pub total_lds_ops: usize,
    pub total_wave_ops: usize,
    pub has_non_uniform_barrier: bool,
}

fn find_natural_loop_blocks(cfg: &ControlFlowGraph, doms: &[Option<BlockId>]) -> Vec<bool> {
    let n = cfg.blocks.len();
    let mut in_loop = vec![false; n];
    let preds = cfg.predecessors();

    for block in &cfg.blocks {
        let u = block.id;
        if doms.get(u).copied().flatten().is_none() {
            continue;
        }
        for v in block.terminator.successors() {
            if v < n && cfg.dominates(doms, v, u) {
                // Natural loop for back-edge u -> v.
                in_loop[v] = true;
                if !in_loop[u] {
                    in_loop[u] = true;
                }
                let mut worklist = vec![u];
                let mut seen = vec![false; n];
                seen[v] = true;
                seen[u] = true;
                while let Some(cur) = worklist.pop() {
                    if cur == v {
                        continue;
                    }
                    for &p in &preds[cur] {
                        if p < n && !seen[p] && doms[p].is_some() {
                            seen[p] = true;
                            in_loop[p] = true;
                            worklist.push(p);
                        }
                    }
                }
            }
        }
    }
    in_loop
}

/// Returns `true` if there is any path from `start` to a `Terminator::End` / exit block that does
/// NOT pass through `avoid_block`.
fn can_reach_end_avoiding(cfg: &ControlFlowGraph, start: BlockId, avoid_block: BlockId) -> bool {
    let n = cfg.blocks.len();
    if start >= n || start == avoid_block {
        return false;
    }
    let mut seen = vec![false; n];
    let mut stack = vec![start];
    seen[start] = true;
    while let Some(b) = stack.pop() {
        let succs = cfg.blocks[b].terminator.successors();
        if succs.is_empty() || Some(b) == cfg.exit_block {
            return true;
        }
        for s in succs {
            if s < n && s != avoid_block && !seen[s] {
                seen[s] = true;
                stack.push(s);
            }
        }
    }
    false
}

/// Analyze a CFG and its decoded instruction stream for 3D compute / mesh-shader metadata:
/// `s_sendmsg(GS_ALLOC_REQ)` guard classification and domination over mesh exports
/// (`exp prim`/`exp pos*`/`exp param*`), LDS/buffer/image/global atomic regions, and
/// `s_barrier`/`s_waitcnt` synchronization safety.
pub fn analyze_compute_mesh_cfg(
    cfg: &ControlFlowGraph,
    instrs: &[DecodedInstruction],
) -> ComputeMeshCfgAnalysis {
    let mut analysis = ComputeMeshCfgAnalysis::default();
    if cfg.blocks.is_empty() {
        return analysis;
    }

    let doms = cfg.dominators();
    let preds = cfg.predecessors();
    let in_loop = find_natural_loop_blocks(cfg, &doms);

    // 1. Gather per-block instruction metrics.
    let mut blocks = Vec::with_capacity(cfg.blocks.len());
    let mut all_export_sites: Vec<(BlockId, usize)> = Vec::new();
    let mut raw_alloc_sites: Vec<(BlockId, usize, usize)> = Vec::new();

    for block in &cfg.blocks {
        let mut summary = BlockSyncSummary {
            block_id: block.id,
            ..BlockSyncSummary::default()
        };
        if !block.synthetic_exit {
            let start = block.instr_range.start.min(instrs.len());
            let end = block.instr_range.end.min(instrs.len());
            for (idx, inst) in instrs[start..end].iter().enumerate() {
                let abs_idx = start + idx;
                if inst.is_barrier() {
                    summary.barrier_instrs.push(abs_idx);
                }
                if inst.is_waitcnt() {
                    summary.waitcnt_instrs.push(abs_idx);
                }
                if classify_exec_mask_op(inst).is_some() {
                    summary.has_exec_mask_write = true;
                }
                if let Some(ds) = inst.ds_op_info() {
                    match ds.kind {
                        Rdna2DsOpKind::Read => summary.lds_read_count += 1,
                        Rdna2DsOpKind::Write => summary.lds_write_count += 1,
                        Rdna2DsOpKind::Swizzle
                        | Rdna2DsOpKind::Permute
                        | Rdna2DsOpKind::BPermute => summary.lds_wave_count += 1,
                        Rdna2DsOpKind::Atomic | Rdna2DsOpKind::Other => {}
                    }
                }
                if let Some(atomic) = inst.atomic_info() {
                    match atomic.domain {
                        Rdna2AtomicDomain::Buffer => summary.buffer_atomic_count += 1,
                        Rdna2AtomicDomain::Image => summary.image_atomic_count += 1,
                        Rdna2AtomicDomain::Flat
                        | Rdna2AtomicDomain::Global
                        | Rdna2AtomicDomain::Scratch => summary.global_atomic_count += 1,
                        Rdna2AtomicDomain::Lds => summary.lds_atomic_count += 1,
                    }
                }
                if inst.wave_op_info().is_some() {
                    summary.wave_op_count += 1;
                }
                if let Some(msg) = inst.sendmsg_info()
                    && msg.is_gs_alloc_req
                {
                    summary.gs_alloc_req_instrs.push(abs_idx);
                    raw_alloc_sites.push((block.id, abs_idx, inst.offset));
                }
                if let Rdna2Detail::Exp { target, .. } = inst.detail {
                    match target {
                        Rdna2ExportTarget::Prim => {
                            summary.prim_export_instrs.push(abs_idx);
                            all_export_sites.push((block.id, abs_idx));
                        }
                        Rdna2ExportTarget::Pos(_) => {
                            summary.pos_export_instrs.push(abs_idx);
                            all_export_sites.push((block.id, abs_idx));
                        }
                        Rdna2ExportTarget::Param(_) => {
                            summary.param_export_instrs.push(abs_idx);
                            all_export_sites.push((block.id, abs_idx));
                        }
                        _ => {}
                    }
                }
            }
        }

        analysis.total_buffer_atomics += summary.buffer_atomic_count;
        analysis.total_image_atomics += summary.image_atomic_count;
        analysis.total_global_atomics += summary.global_atomic_count;
        analysis.total_lds_atomics += summary.lds_atomic_count;
        analysis.total_barriers += summary.barrier_instrs.len();
        analysis.total_waitcnts += summary.waitcnt_instrs.len();
        analysis.total_lds_ops +=
            summary.lds_read_count + summary.lds_write_count + summary.lds_wave_count + summary.lds_atomic_count;
        analysis.total_wave_ops += summary.wave_op_count;
        if summary.total_atomics() > 0 {
            analysis.atomic_blocks.push(block.id);
        }
        blocks.push(summary);
    }

    let has_prim_exports = blocks.iter().any(|b| !b.prim_export_instrs.is_empty());
    analysis.is_mesh_shader = !raw_alloc_sites.is_empty() || has_prim_exports;

    // 2. Analyze GS_ALLOC_REQ guard patterns & domination over mesh exports.
    let mut covered_export_sites: BTreeSet<(BlockId, usize)> = BTreeSet::new();
    let entry = cfg.entry().unwrap_or(0);

    for (alloc_block, alloc_idx, alloc_offset) in raw_alloc_sites {
        // Check for inline exec-mask narrowing before GS_ALLOC_REQ in the same block.
        let b_start = cfg.blocks[alloc_block].instr_range.start.min(instrs.len());
        let inline_exec_guard = instrs[b_start..alloc_idx.min(instrs.len())]
            .iter()
            .any(|i| classify_exec_mask_op(i).is_some() || i.mnemonic.starts_with("v_cmpx_"));

        let block_preds = &preds[alloc_block];
        let mut guard_header: Option<BlockId> = None;
        let guard_kind = if inline_exec_guard {
            MeshAllocGuardKind::SingleLaneExecGuard
        } else if alloc_block == entry {
            MeshAllocGuardKind::Unconditional
        } else if block_preds.len() == 1 {
            let p = block_preds[0];
            match cfg.blocks[p].terminator {
                Terminator::CondBranch { kind, .. } => {
                    guard_header = Some(p);
                    match kind {
                        BranchKind::Execz
                        | BranchKind::Execnz
                        | BranchKind::Vccz
                        | BranchKind::Vccnz => MeshAllocGuardKind::SingleLaneExecGuard,
                        BranchKind::Scc0 | BranchKind::Scc1 => {
                            MeshAllocGuardKind::WaveUniformScalarGuard
                        }
                        BranchKind::Unconditional => MeshAllocGuardKind::Unconditional,
                    }
                }
                Terminator::Fallthrough(_) | Terminator::Branch(_) => {
                    // Walk dominator chain: unconditional if no conditional dominates us.
                    let mut has_cond_dom = false;
                    let mut cur = p;
                    for _ in 0..cfg.blocks.len() {
                        if matches!(cfg.blocks[cur].terminator, Terminator::CondBranch { .. }) {
                            has_cond_dom = true;
                            break;
                        }
                        match doms[cur] {
                            Some(parent) if parent != cur => cur = parent,
                            _ => break,
                        }
                    }
                    if has_cond_dom {
                        MeshAllocGuardKind::OtherConditional
                    } else {
                        MeshAllocGuardKind::Unconditional
                    }
                }
                Terminator::End => MeshAllocGuardKind::OtherConditional,
            }
        } else {
            MeshAllocGuardKind::OtherConditional
        };

        // Effective domination anchor: for the canonical AMD NGG/mesh single-lane/wave-0 if-skip
        // guard (`guard_hdr` skips `alloc_block` straight into `merge`, while `alloc_block` flows
        // directly to `merge`), `guard_hdr` represents the wave-wide allocation point for blocks
        // strictly after `guard_hdr`.
        let dom_anchor = match (guard_kind, guard_header) {
            (
                MeshAllocGuardKind::SingleLaneExecGuard
                | MeshAllocGuardKind::WaveUniformScalarGuard,
                Some(hdr),
            ) => {
                if let Terminator::CondBranch {
                    target,
                    fallthrough,
                    ..
                } = cfg.blocks[hdr].terminator
                {
                    let other_arm = if target == alloc_block {
                        fallthrough
                    } else {
                        target
                    };
                    let alloc_succs = cfg.blocks[alloc_block].terminator.successors();
                    if alloc_succs.len() == 1 && alloc_succs[0] == other_arm {
                        hdr
                    } else {
                        alloc_block
                    }
                } else {
                    alloc_block
                }
            }
            _ => alloc_block,
        };

        let mut dominates_all = true;
        for &(exp_block, exp_idx) in &all_export_sites {
            let dominated = if exp_block == alloc_block {
                alloc_idx < exp_idx
            } else if dom_anchor != alloc_block && exp_block == dom_anchor {
                // Export occurs inside the guard header BEFORE GS_ALLOC_REQ!
                false
            } else {
                cfg.dominates(&doms, dom_anchor, exp_block)
            };
            if dominated {
                covered_export_sites.insert((exp_block, exp_idx));
            } else {
                dominates_all = false;
            }
        }

        analysis.mesh_alloc_points.push(MeshAllocPoint {
            block_id: alloc_block,
            instr_index: alloc_idx,
            offset: alloc_offset,
            guard_kind,
            dominates_all_exports: dominates_all,
            inside_loop: in_loop[alloc_block],
        });
    }

    let mut unallocated: BTreeSet<BlockId> = BTreeSet::new();
    for &(exp_block, exp_idx) in &all_export_sites {
        if !covered_export_sites.contains(&(exp_block, exp_idx)) {
            unallocated.insert(exp_block);
        }
    }
    analysis.unallocated_export_blocks = unallocated.into_iter().collect();
    analysis.all_exports_dominated_by_alloc =
        all_export_sites.is_empty() || analysis.unallocated_export_blocks.is_empty();

    // 3. Analyze s_barrier uniformity & loop nesting.
    for b_summary in &blocks {
        let bid = b_summary.block_id;
        for &bar_idx in &b_summary.barrier_instrs {
            let mut uniform = true;

            // Check if any dominating divergent EXEC/VCC conditional branch can bypass this barrier.
            for d in &cfg.blocks {
                if d.id == bid {
                    continue;
                }
                if let Terminator::CondBranch { kind, .. } = d.terminator
                    && matches!(
                        kind,
                        BranchKind::Execz
                            | BranchKind::Execnz
                            | BranchKind::Vccz
                            | BranchKind::Vccnz
                    )
                    && cfg.dominates(&doms, d.id, bid)
                    && can_reach_end_avoiding(cfg, d.id, bid)
                {
                    uniform = false;
                    break;
                }
            }

            // Also check for an un-restored inline exec-mask narrowing earlier in the same block.
            if uniform {
                let start = cfg.blocks[bid].instr_range.start.min(instrs.len());
                let mut narrowed = false;
                for inst in &instrs[start..bar_idx.min(instrs.len())] {
                    if classify_exec_mask_op(inst).is_some()
                        || inst.mnemonic.starts_with("v_cmpx_")
                    {
                        narrowed = true;
                    } else if matches!(
                        inst.detail,
                        Rdna2Detail::Sop1 { sdst: 126, .. } | Rdna2Detail::Sop2 { sdst: 126, .. }
                    ) {
                        // Direct restore to `exec` / `exec_lo`.
                        narrowed = false;
                    }
                }
                if narrowed {
                    uniform = false;
                }
            }

            if !uniform {
                analysis.has_non_uniform_barrier = true;
            }
            analysis.barrier_regions.push(BarrierRegion {
                barrier_block: bid,
                barrier_instr_index: bar_idx,
                inside_loop: in_loop[bid],
                in_uniform_control_flow: uniform,
            });
        }
    }

    analysis.blocks = blocks;
    analysis
}

/// Deterministic sanitized one-line summary of a [`ComputeMeshCfgAnalysis`].
pub fn summarize_compute_mesh_cfg(analysis: &ComputeMeshCfgAnalysis) -> String {
    format!(
        "mesh={} allocs={} exports_dom={} barriers={} non_uniform_barriers={} waitcnts={} atomics(buf={},img={},glob={},lds={}) lds_ops={} wave_ops={}",
        analysis.is_mesh_shader,
        analysis.mesh_alloc_points.len(),
        analysis.all_exports_dominated_by_alloc,
        analysis.total_barriers,
        analysis.has_non_uniform_barrier,
        analysis.total_waitcnts,
        analysis.total_buffer_atomics,
        analysis.total_image_atomics,
        analysis.total_global_atomics,
        analysis.total_lds_atomics,
        analysis.total_lds_ops,
        analysis.total_wave_ops,
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rdna2::{Rdna2Encoding, decode_program};

    // --- Synthetic RDNA2 encoders -------------------------------------------------

    fn enc_sopp(op: u8, simm16: u16) -> u32 {
        (0b1_0111_1111_u32 << 23) | (u32::from(op & 0x7F) << 16) | u32::from(simm16)
    }

    fn enc_sop1(op: u8, sdst: u8, ssrc0: u8) -> u32 {
        (0b1_0111_1101_u32 << 23)
            | (u32::from(sdst & 0x7F) << 16)
            | (u32::from(op) << 8)
            | u32::from(ssrc0)
    }

    /// Program builder over single-dword instructions (index == offset / 4).
    #[derive(Clone, Copy)]
    enum Op {
        Nop,
        End,
        Br(BranchKind, i64),
        Word(u32),
    }

    fn sopp_op(kind: BranchKind) -> u8 {
        match kind {
            BranchKind::Unconditional => SOPP_BRANCH,
            BranchKind::Scc0 => SOPP_CBRANCH_SCC0,
            BranchKind::Scc1 => SOPP_CBRANCH_SCC1,
            BranchKind::Vccz => SOPP_CBRANCH_VCCZ,
            BranchKind::Vccnz => SOPP_CBRANCH_VCCNZ,
            BranchKind::Execz => SOPP_CBRANCH_EXECZ,
            BranchKind::Execnz => SOPP_CBRANCH_EXECNZ,
        }
    }

    fn assemble(ops: &[Op]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for (idx, op) in ops.iter().enumerate() {
            let w = match *op {
                Op::Nop => enc_sopp(0x00, 0),
                Op::End => enc_sopp(SOPP_ENDPGM, 0),
                // Target instruction index -> simm16 = target - (idx + 1).
                Op::Br(kind, target) => {
                    let rel = target - (idx as i64 + 1);
                    enc_sopp(sopp_op(kind), (rel as i16) as u16)
                }
                Op::Word(w) => w,
            };
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        bytes
    }

    fn build(ops: &[Op]) -> (Vec<DecodedInstruction>, ControlFlowGraph) {
        let instrs = decode_program(&assemble(ops));
        assert_eq!(instrs.len(), ops.len());
        let cfg = ControlFlowGraph::from_instructions(&instrs);
        (instrs, cfg)
    }

    // --- Semantic reference executors ---------------------------------------------

    fn oracle(seed: u64, block: BlockId, visit: u32) -> bool {
        let mut x = seed ^ ((block as u64) << 32) ^ u64::from(visit).wrapping_mul(0x9E37_79B9);
        x ^= x >> 33;
        x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
        x ^= x >> 33;
        x = x.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
        x ^= x >> 33;
        x & 1 == 1
    }

    /// Execute the raw CFG; returns visited (non-synthetic) block trace.
    fn run_cfg(cfg: &ControlFlowGraph, seed: u64, max_steps: usize) -> Vec<BlockId> {
        let mut trace = Vec::new();
        let mut visits = vec![0u32; cfg.blocks.len()];
        let Some(mut cur) = cfg.entry() else {
            return trace;
        };
        while trace.len() < max_steps {
            if Some(cur) == cfg.exit_block {
                break;
            }
            trace.push(cur);
            visits[cur] += 1;
            cur = match cfg.blocks[cur].terminator {
                Terminator::End => break,
                Terminator::Fallthrough(t) | Terminator::Branch(t) => t,
                Terminator::CondBranch {
                    target,
                    fallthrough,
                    ..
                } => {
                    if oracle(seed, cur, visits[cur]) {
                        target
                    } else {
                        fallthrough
                    }
                }
            };
        }
        trace
    }

    #[derive(Debug, PartialEq, Eq)]
    enum Flow {
        Normal,
        Break,
        Continue,
        Return,
        Halt,
    }

    struct Exec<'a> {
        cfg: &'a ControlFlowGraph,
        seed: u64,
        max_steps: usize,
        trace: Vec<BlockId>,
        visits: Vec<u32>,
    }

    impl Exec<'_> {
        fn visit(&mut self, id: BlockId) -> bool {
            if self.trace.len() >= self.max_steps {
                return false;
            }
            self.trace.push(id);
            self.visits[id] += 1;
            true
        }

        fn taken(&self, id: BlockId) -> bool {
            oracle(self.seed, id, self.visits[id])
        }

        fn run(&mut self, node: &StructuredNode) -> Flow {
            match node {
                StructuredNode::Block(id) => {
                    if self.visit(*id) {
                        Flow::Normal
                    } else {
                        Flow::Halt
                    }
                }
                StructuredNode::Seq(nodes) => {
                    for n in nodes {
                        let f = self.run(n);
                        if f != Flow::Normal {
                            return f;
                        }
                    }
                    Flow::Normal
                }
                StructuredNode::If {
                    cond_block,
                    cond,
                    then_branch,
                    else_branch,
                } => {
                    let Terminator::CondBranch { kind, .. } =
                        self.cfg.blocks[*cond_block].terminator
                    else {
                        panic!("If on non-conditional block {cond_block}");
                    };
                    let taken = self.taken(*cond_block);
                    let holds = if *cond == kind {
                        taken
                    } else {
                        assert_eq!(Some(*cond), kind.negate());
                        !taken
                    };
                    if holds {
                        self.run(then_branch)
                    } else if let Some(e) = else_branch {
                        self.run(e)
                    } else {
                        Flow::Normal
                    }
                }
                StructuredNode::Loop { body } => loop {
                    match self.run(body) {
                        Flow::Normal | Flow::Continue => continue,
                        Flow::Break => return Flow::Normal,
                        f => return f,
                    }
                },
                StructuredNode::Break => Flow::Break,
                StructuredNode::Continue => Flow::Continue,
                StructuredNode::Return => Flow::Return,
                StructuredNode::Dispatch { entry, blocks, .. } => {
                    let members: BTreeSet<BlockId> = blocks.iter().copied().collect();
                    let mut pc = *entry;
                    loop {
                        if !members.contains(&pc) {
                            return if Some(pc) == self.cfg.exit_block {
                                Flow::Return
                            } else {
                                Flow::Normal
                            };
                        }
                        if !self.visit(pc) {
                            return Flow::Halt;
                        }
                        pc = match self.cfg.blocks[pc].terminator {
                            Terminator::End => return Flow::Return,
                            Terminator::Fallthrough(t) | Terminator::Branch(t) => t,
                            Terminator::CondBranch {
                                target,
                                fallthrough,
                                ..
                            } => {
                                if self.taken(pc) {
                                    target
                                } else {
                                    fallthrough
                                }
                            }
                        };
                    }
                }
            }
        }
    }

    fn run_structured(
        cfg: &ControlFlowGraph,
        node: &StructuredNode,
        seed: u64,
        max_steps: usize,
    ) -> Vec<BlockId> {
        let mut exec = Exec {
            cfg,
            seed,
            max_steps,
            trace: Vec::new(),
            visits: vec![0; cfg.blocks.len()],
        };
        let flow = exec.run(node);
        assert_ne!(flow, Flow::Break, "break escaped the outermost loop");
        assert_ne!(flow, Flow::Continue, "continue escaped the outermost loop");
        exec.trace
    }

    /// The structured tree must replay exactly the same block trace as the CFG.
    fn assert_equivalent(cfg: &ControlFlowGraph, node: &StructuredNode) {
        for seed in 0..48u64 {
            let expected = run_cfg(cfg, seed, 96);
            let actual = run_structured(cfg, node, seed, 96);
            assert_eq!(
                expected, actual,
                "trace mismatch for seed {seed}\ncfg: {cfg:#?}\nnode: {node:#?}"
            );
        }
    }

    // --- Naga validation ---------------------------------------------------------

    fn test_block_body(block: &BasicBlock) -> String {
        format!(
            "acc = acc * 31u + {id}u;\nscc = (acc & 1u) == 0u;\nvcc = acc & 6u;\nexec_mask = exec_mask ^ (acc << 1u);",
            id = block.id
        )
    }

    fn wgsl_function(cfg: &ControlFlowGraph, node: &StructuredNode) -> String {
        let body = emit_wgsl_control_flow(node, cfg, test_block_body);
        format!(
            "var<private> sink: u32;\n\nfn lifted() {{\n    var scc: bool = false;\n    var vcc: u32 = 0u;\n    var exec_mask: u32 = 0xffffffffu;\n    var acc: u32 = 0u;\n{body}    sink = acc;\n}}\n\n@compute @workgroup_size(1)\nfn main() {{\n    lifted();\n}}\n"
        )
    }

    fn validate_wgsl(src: &str) {
        let module = match naga::front::wgsl::parse_str(src) {
            Ok(m) => m,
            Err(e) => panic!("WGSL parse error: {}\n{src}", e.emit_to_string(src)),
        };
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        if let Err(e) = validator.validate(&module) {
            panic!("WGSL validation error: {e:?}\n{src}");
        }
    }

    fn lift_and_check(cfg: &ControlFlowGraph) -> StructuredNode {
        let node = structure_cfg(cfg);
        assert_equivalent(cfg, &node);
        validate_wgsl(&wgsl_function(cfg, &node));
        node
    }

    use BranchKind::*;
    use Op::{Br, End, Nop};

    // --- CFG construction --------------------------------------------------------

    #[test]
    fn empty_stream_yields_empty_cfg() {
        let cfg = ControlFlowGraph::from_instructions(&[]);
        assert!(cfg.is_empty());
        assert_eq!(cfg.entry(), None);
        let node = structure_cfg(&cfg);
        assert_eq!(node, StructuredNode::Seq(Vec::new()));
        validate_wgsl(&wgsl_function(&cfg, &node));
    }

    #[test]
    fn linear_shader_is_single_block() {
        let (_, cfg) = build(&[Nop, Nop, Nop, End]);
        assert_eq!(cfg.blocks.len(), 1);
        assert_eq!(cfg.blocks[0].instr_range, 0..4);
        assert_eq!(cfg.blocks[0].terminator, Terminator::End);
        assert_eq!(cfg.blocks[0].control_instr, Some(3));
        assert_eq!(cfg.blocks[0].body_range(), 0..3);
        assert!(cfg.diagnostics.is_empty());
        assert_eq!(cfg.exit_block, None);
        let node = lift_and_check(&cfg);
        assert_eq!(
            node,
            StructuredNode::Seq(vec![StructuredNode::Block(0), StructuredNode::Return])
        );
    }

    #[test]
    fn branch_target_matches_decoder_formula() {
        // s_cbranch_scc0 at offset 4 with simm16 = +1 -> target 4 + 4 + 4 = 12.
        let (instrs, cfg) = build(&[Nop, Br(Scc0, 3), Nop, Nop, End]);
        let Rdna2Detail::Sopp {
            simm16,
            branch_target,
            ..
        } = instrs[1].detail
        else {
            panic!("expected SOPP");
        };
        assert_eq!(instrs[1].encoding, Rdna2Encoding::Sopp);
        assert_eq!(instrs[1].mnemonic, "s_cbranch_scc0");
        let expected = instrs[1].offset as i64 + 4 + i64::from(simm16 as i16) * 4;
        assert_eq!(branch_target, Some(expected));
        assert_eq!(expected, 12);
        assert_eq!(cfg.instruction_index_at(expected), Some(3));
        assert_eq!(cfg.block_at_offset(expected), Some(2));
        assert_eq!(cfg.block_at_offset(8), Some(1));
        assert_eq!(cfg.block_at_offset(13), None);
        assert_eq!(cfg.block_at_offset(-4), None);
    }

    // --- Forward branches ----------------------------------------------------------

    #[test]
    fn forward_scc0_if_skip() {
        // 0: s_cmp ; 1: s_cbranch_scc0 -> 3 ; 2: then ; 3: merge ; 4: s_endpgm
        let (_, cfg) = build(&[Nop, Br(Scc0, 3), Nop, Nop, End]);
        assert_eq!(cfg.blocks.len(), 3);
        assert_eq!(cfg.blocks[0].instr_range, 0..2);
        assert_eq!(
            cfg.blocks[0].terminator,
            Terminator::CondBranch {
                kind: Scc0,
                target: 2,
                fallthrough: 1
            }
        );
        assert_eq!(cfg.blocks[0].body_range(), 0..1);
        assert_eq!(cfg.blocks[1].terminator, Terminator::Fallthrough(2));
        assert_eq!(cfg.blocks[2].terminator, Terminator::End);

        let node = lift_and_check(&cfg);
        assert_eq!(
            node,
            StructuredNode::Seq(vec![
                StructuredNode::Block(0),
                StructuredNode::If {
                    cond_block: 0,
                    cond: Scc1,
                    then_branch: Box::new(StructuredNode::Block(1)),
                    else_branch: None,
                },
                StructuredNode::Block(2),
                StructuredNode::Return,
            ])
        );
        let wgsl = emit_wgsl_control_flow(&node, &cfg, |_| String::new());
        assert!(wgsl.contains("if (scc) {"), "{wgsl}");
    }

    #[test]
    fn execz_skip_after_and_saveexec() {
        // s_and_saveexec_b32 s0, vcc_lo ; s_cbranch_execz -> 4 ; body ; body ;
        // s_or_b32 exec restore (nop stand-in) ; s_endpgm
        let saveexec = enc_sop1(0x3C, 0, 106);
        let (instrs, cfg) = build(&[Op::Word(saveexec), Br(Execz, 4), Nop, Nop, Nop, End]);
        assert_eq!(instrs[0].mnemonic, "s_and_saveexec_b32");
        assert_eq!(
            classify_exec_mask_op(&instrs[0]),
            Some(ExecMaskWrite {
                op: ExecMaskOp::AndSaveExec,
                wave64: false
            })
        );
        assert_eq!(classify_exec_mask_op(&instrs[1]), None);
        // The saveexec does not split the block.
        assert_eq!(cfg.blocks[0].instr_range, 0..2);

        let node = lift_and_check(&cfg);
        let StructuredNode::Seq(items) = &node else {
            panic!("expected Seq: {node:?}");
        };
        assert!(matches!(
            items[1],
            StructuredNode::If {
                cond: Execnz,
                else_branch: None,
                ..
            }
        ));
        let wgsl = emit_wgsl_control_flow(&node, &cfg, |_| String::new());
        assert!(wgsl.contains("if (exec_mask != 0u) {"), "{wgsl}");
    }

    #[test]
    fn exec_mask_ops_classify_b32_and_b64() {
        let cases = [
            (0x24u8, ExecMaskOp::AndSaveExec, true),
            (0x3C, ExecMaskOp::AndSaveExec, false),
            (0x3A, ExecMaskOp::AndN2WrExec, true),
            (0x47, ExecMaskOp::AndN2WrExec, false),
            (0x3E, ExecMaskOp::XorSaveExec, false),
        ];
        for (op, expected, wave64) in cases {
            let instrs = decode_program(&enc_sop1(op, 0, 106).to_le_bytes());
            assert_eq!(
                classify_exec_mask_op(&instrs[0]),
                Some(ExecMaskWrite {
                    op: expected,
                    wave64
                }),
                "{}",
                instrs[0].mnemonic
            );
        }
        // s_mov_b32 is not an exec-mask op.
        let instrs = decode_program(&enc_sop1(0x03, 0, 106).to_le_bytes());
        assert_eq!(classify_exec_mask_op(&instrs[0]), None);
    }

    #[test]
    fn if_else_diamond_with_branch_merge() {
        // 0: s_cbranch_scc1 -> 3 ; 1: then ; 2: s_branch -> 4 ; 3: else ; 4: merge ; 5: end
        let (_, cfg) = build(&[Br(Scc1, 3), Nop, Br(Unconditional, 4), Nop, Nop, End]);
        assert_eq!(cfg.blocks.len(), 4);
        assert_eq!(cfg.blocks[1].terminator, Terminator::Branch(3));
        assert_eq!(cfg.blocks[1].body_range(), 1..2);

        let node = lift_and_check(&cfg);
        assert_eq!(
            node,
            StructuredNode::Seq(vec![
                StructuredNode::Block(0),
                StructuredNode::If {
                    cond_block: 0,
                    cond: Scc0,
                    then_branch: Box::new(StructuredNode::Block(1)),
                    else_branch: Some(Box::new(StructuredNode::Block(2))),
                },
                StructuredNode::Block(3),
                StructuredNode::Return,
            ])
        );
        let wgsl = emit_wgsl_control_flow(&node, &cfg, |_| String::new());
        assert!(
            wgsl.contains("if (!scc) {") && wgsl.contains("} else {"),
            "{wgsl}"
        );
    }

    #[test]
    fn nested_if_else_sharing_outer_merge() {
        // if (a) { if (b) { X } else { Y } } else { Z } with X/Y jumping to the outer merge.
        let ops = [
            Br(Scc0, 6),          // 0: outer cond -> Z
            Br(Vccz, 4),          // 1: inner cond -> Y
            Nop,                  // 2: X
            Br(Unconditional, 7), // 3: X -> merge
            Nop,                  // 4: Y
            Br(Unconditional, 7), // 5: Y -> merge
            Nop,                  // 6: Z (falls into merge)
            Nop,                  // 7: merge
            End,                  // 8
        ];
        let (_, cfg) = build(&ops);
        let node = lift_and_check(&cfg);
        assert!(node.is_fully_structured(), "{node:#?}");
    }

    #[test]
    fn if_with_early_endpgm_in_then() {
        let (_, cfg) = build(&[Br(Execz, 3), Nop, End, Nop, End]);
        let node = lift_and_check(&cfg);
        assert!(node.is_fully_structured(), "{node:#?}");
    }

    // --- Loops ------------------------------------------------------------------------

    #[test]
    fn backward_scc1_do_while_loop() {
        // 0: pre ; 1: body ; 2: s_cbranch_scc1 -> 1 ; 3: s_endpgm
        let (_, cfg) = build(&[Nop, Nop, Br(Scc1, 1), End]);
        assert_eq!(cfg.blocks.len(), 3);
        assert_eq!(
            cfg.blocks[1].terminator,
            Terminator::CondBranch {
                kind: Scc1,
                target: 1,
                fallthrough: 2
            }
        );
        let node = lift_and_check(&cfg);
        assert_eq!(
            node,
            StructuredNode::Seq(vec![
                StructuredNode::Block(0),
                StructuredNode::Loop {
                    body: Box::new(StructuredNode::Seq(vec![
                        StructuredNode::Block(1),
                        StructuredNode::If {
                            cond_block: 1,
                            cond: Scc1,
                            then_branch: Box::new(StructuredNode::Continue),
                            else_branch: None,
                        },
                        StructuredNode::Break,
                    ])),
                },
                StructuredNode::Block(2),
                StructuredNode::Return,
            ])
        );
        let wgsl = emit_wgsl_control_flow(&node, &cfg, |_| String::new());
        assert!(
            wgsl.contains("loop {") && wgsl.contains("continue;"),
            "{wgsl}"
        );
        assert!(
            wgsl.contains(&format!("{STRUCTURED_LOOP_MAX_ITERATIONS}u")),
            "{wgsl}"
        );
    }

    #[test]
    fn while_loop_with_s_branch_back_edge() {
        // 0: pre ; 1: s_cbranch_scc0 -> 4 (exit) ; 2: body ; 3: s_branch -> 1 ; 4: end
        let (_, cfg) = build(&[Nop, Br(Scc0, 4), Nop, Br(Unconditional, 1), End]);
        let node = lift_and_check(&cfg);
        assert_eq!(
            node,
            StructuredNode::Seq(vec![
                StructuredNode::Block(0),
                StructuredNode::Loop {
                    body: Box::new(StructuredNode::Seq(vec![
                        StructuredNode::Block(1),
                        StructuredNode::If {
                            cond_block: 1,
                            cond: Scc0,
                            then_branch: Box::new(StructuredNode::Break),
                            else_branch: None,
                        },
                        StructuredNode::Block(2),
                        StructuredNode::Continue,
                    ])),
                },
                StructuredNode::Block(3),
                StructuredNode::Return,
            ])
        );
    }

    #[test]
    fn self_loop_and_nested_loops() {
        // Self loop on block at index 1.
        let (_, cfg) = build(&[Nop, Br(Vccnz, 1), End]);
        let node = lift_and_check(&cfg);
        assert!(node.is_fully_structured());

        // Outer loop 1..=5 containing inner loop 2..=3 and an execz skip.
        let ops = [
            Nop,          // 0
            Nop,          // 1 outer header
            Nop,          // 2 inner header
            Br(Scc1, 2),  // 3 inner latch
            Br(Execz, 6), // 4 execz skip over 5
            Nop,          // 5
            Br(Vccnz, 1), // 6 outer latch
            End,          // 7
        ];
        let (_, cfg) = build(&ops);
        let node = lift_and_check(&cfg);
        assert!(node.is_fully_structured(), "{node:#?}");
    }

    #[test]
    fn loop_with_mid_body_break_and_continue() {
        let ops = [
            Nop,          // 0
            Nop,          // 1 header
            Br(Execz, 6), // 2 break to exit
            Br(Scc1, 1),  // 3 continue
            Nop,          // 4
            Br(Vccnz, 1), // 5 latch
            End,          // 6 exit
        ];
        let (_, cfg) = build(&ops);
        let node = lift_and_check(&cfg);
        assert!(node.is_fully_structured(), "{node:#?}");
    }

    // --- Irreducible fallback -------------------------------------------------------

    #[test]
    fn irreducible_multi_entry_loop_falls_back_to_dispatch() {
        // Entry jumps into the middle of a cycle A <-> B: two loop entries.
        let ops = [
            Br(Scc0, 3),  // 0: -> B or A
            Nop,          // 1: A
            Nop,          // 2: A
            Nop,          // 3: B
            Br(Vccnz, 1), // 4: B -> A (back edge)
            End,          // 5
        ];
        let (_, cfg) = build(&ops);
        let node = lift_and_check(&cfg);
        assert_eq!(node.dispatch_count(), 1, "{node:#?}");
        let StructuredNode::Dispatch {
            entry,
            blocks,
            max_iterations,
        } = &node
        else {
            panic!("expected whole-program dispatch: {node:#?}");
        };
        assert_eq!(*entry, 0);
        assert_eq!(blocks, &vec![0, 1, 2, 3]);
        assert_eq!(*max_iterations, DEFAULT_DISPATCH_MAX_ITERATIONS);

        let wgsl = emit_wgsl_control_flow(&node, &cfg, |_| String::new());
        assert!(wgsl.contains("var _pg6_pc: u32 = 0u;"), "{wgsl}");
        assert!(wgsl.contains("var _pg6_iter: u32 = 0u;"), "{wgsl}");
        assert!(wgsl.contains("switch (_pg6_pc) {"), "{wgsl}");
        assert!(
            wgsl.contains("if (_pg6_iter >= 65536u) { break; }"),
            "{wgsl}"
        );
        assert!(wgsl.contains("default: {"), "{wgsl}");
    }

    #[test]
    fn irreducible_region_after_structured_prefix_dispatches_locally() {
        let ops = [
            Br(Scc0, 2),   // 0: if-skip
            Nop,           // 1
            Br(Vccz, 5),   // 2: irreducible region starts here -> B or A
            Nop,           // 3: A
            Nop,           // 4: A
            Nop,           // 5: B
            Br(Execnz, 3), // 6: B -> A
            End,           // 7
        ];
        let (_, cfg) = build(&ops);
        let node = lift_and_check(&cfg);
        let StructuredNode::Seq(items) = &node else {
            panic!("expected Seq: {node:#?}");
        };
        assert!(matches!(items[0], StructuredNode::Block(0)));
        assert!(matches!(items[1], StructuredNode::If { .. }));
        assert!(matches!(
            items.last(),
            Some(StructuredNode::Dispatch { .. })
        ));
        assert_eq!(node.dispatch_count(), 1);
    }

    #[test]
    fn irreducible_region_inside_then_branch_dispatches_and_exits_to_merge() {
        let ops = [
            Br(Scc0, 7),   // 0: outer if-skip over 1..7
            Br(Vccz, 4),   // 1: -> B or A
            Nop,           // 2: A
            Nop,           // 3: A
            Nop,           // 4: B
            Br(Execnz, 2), // 5: B -> A
            Nop,           // 6: fall to merge
            Nop,           // 7: merge
            End,           // 8
        ];
        let (_, cfg) = build(&ops);
        let node = lift_and_check(&cfg);
        assert_eq!(node.dispatch_count(), 1, "{node:#?}");
        // The merge block stays outside the dispatcher.
        let StructuredNode::Seq(items) = &node else {
            panic!("expected Seq: {node:#?}");
        };
        assert!(matches!(items[1], StructuredNode::If { .. }));
        let wgsl = emit_wgsl_control_flow(&node, &cfg, |_| String::new());
        assert!(
            wgsl.contains(&format!("_pg6_pc = {DISPATCH_EXIT_PC}u;")),
            "{wgsl}"
        );
    }

    // --- Robustness ---------------------------------------------------------------

    #[test]
    fn out_of_bounds_branch_targets_do_not_panic() {
        let ops = [
            Nop,
            Br(Unconditional, 100), // 1: way past the end
            Br(Scc0, -50),          // 2: negative target
            Br(Vccz, 40),           // 3: past the end
            Nop,                    // 4
            Br(Execz, 5),           // 5: self loop as last instruction
        ];
        let (_, cfg) = build(&ops);
        assert!(cfg.diagnostics.iter().any(|d| matches!(
            d,
            CfgDiagnostic::BranchTargetOutOfRange { instr_index: 1, .. }
        )));
        assert!(cfg.diagnostics.iter().any(|d| matches!(
            d,
            CfgDiagnostic::BranchTargetOutOfRange {
                instr_index: 2,
                target,
                ..
            } if *target < 0
        )));
        assert!(
            cfg.diagnostics
                .iter()
                .any(|d| matches!(d, CfgDiagnostic::FallthroughPastEnd { instr_index: 5, .. }))
        );
        // Invalid unconditional branch ends the program.
        assert_eq!(cfg.blocks[0].terminator, Terminator::End);
        // Invalid conditional targets route to the synthetic exit.
        let exit = cfg.exit_block.expect("exit block");
        assert!(cfg.blocks[exit].synthetic_exit);
        assert!(cfg.blocks[exit].is_empty());
        lift_and_check(&cfg);
    }

    #[test]
    fn misaligned_branch_target_is_diagnosed() {
        // 0: s_mov_b32 s0, 0x12345678 (8 bytes: offsets 0..8)
        // 8: s_cbranch_scc1 -> offset 4 (inside the literal)
        // 12: s_endpgm
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&enc_sop1(0x03, 0, 0xFF).to_le_bytes());
        bytes.extend_from_slice(&0x1234_5678u32.to_le_bytes());
        bytes.extend_from_slice(&enc_sopp(SOPP_CBRANCH_SCC1, (-2i16) as u16).to_le_bytes());
        bytes.extend_from_slice(&enc_sopp(SOPP_ENDPGM, 0).to_le_bytes());
        let instrs = decode_program(&bytes);
        assert_eq!(instrs.len(), 3);
        assert_eq!(instrs[0].size, 8);
        let cfg = ControlFlowGraph::from_instructions(&instrs);
        assert!(cfg.diagnostics.iter().any(|d| matches!(
            d,
            CfgDiagnostic::BranchTargetMisaligned {
                instr_index: 1,
                target: 4,
                ..
            }
        )));
        assert_eq!(cfg.instruction_index_at(8), Some(1));
        assert_eq!(cfg.instruction_index_at(4), None);
        lift_and_check(&cfg);
    }

    #[test]
    fn missing_endpgm_and_unreachable_code() {
        let (_, cfg) = build(&[Nop, Br(Unconditional, 3), Nop, Nop]);
        assert!(
            cfg.diagnostics
                .iter()
                .any(|d| matches!(d, CfgDiagnostic::FallthroughPastEnd { .. }))
        );
        let reachable = cfg.reachable();
        assert!(!reachable[1], "block after s_branch is unreachable");
        lift_and_check(&cfg);
    }

    fn xorshift(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        x
    }

    #[test]
    fn random_cfgs_lift_equivalently_and_validate() {
        const KINDS: [BranchKind; 7] = [Unconditional, Scc0, Scc1, Vccz, Vccnz, Execz, Execnz];
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut structured = 0usize;
        let mut dispatched = 0usize;
        for _ in 0..400 {
            let n = 2 + (xorshift(&mut state) % 22) as usize;
            let mut ops = Vec::with_capacity(n);
            for _ in 0..n {
                let r = xorshift(&mut state) % 100;
                let op = if r < 40 {
                    Nop
                } else if r < 46 {
                    End
                } else {
                    let kind = KINDS[(xorshift(&mut state) % 7) as usize];
                    let target = if xorshift(&mut state).is_multiple_of(16) {
                        n as i64 + 3
                    } else {
                        (xorshift(&mut state) % n as u64) as i64
                    };
                    Br(kind, target)
                };
                ops.push(op);
            }
            if !xorshift(&mut state).is_multiple_of(5) {
                ops.push(End);
            }
            let (_, cfg) = build(&ops);
            let node = lift_and_check(&cfg);
            if node.is_fully_structured() {
                structured += 1;
            } else {
                dispatched += 1;
            }
        }
        assert!(
            structured > 0 && dispatched > 0,
            "{structured} / {dispatched}"
        );
    }

    #[test]
    fn arbitrary_bytes_never_panic_and_emit_valid_wgsl() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        for round in 0..60 {
            let words = 4 + (xorshift(&mut state) % 48) as usize;
            let mut bytes = Vec::with_capacity(words * 4);
            for _ in 0..words {
                // Bias toward SOPP words so branches are frequent.
                let w = if xorshift(&mut state).is_multiple_of(3) {
                    let op = [0x01u8, 0x02, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09]
                        [(xorshift(&mut state) % 8) as usize];
                    let rel = (xorshift(&mut state) % 24) as i16 - 12;
                    enc_sopp(op, rel as u16)
                } else {
                    xorshift(&mut state) as u32
                };
                bytes.extend_from_slice(&w.to_le_bytes());
            }
            if round % 2 == 0 {
                bytes.push(0xAB); // trailing garbage / truncation
            }
            let instrs = decode_program(&bytes);
            let cfg = ControlFlowGraph::from_instructions(&instrs);
            let node = lift_and_check(&cfg);
            let _ = summarize(&cfg, &node);
        }
    }

    #[test]
    fn dispatch_is_bounded_and_scoped_per_region() {
        // Two independent irreducible regions inside separate if-skips, so two
        // dispatchers share one function scope without name clashes.
        let ops = [
            Br(Scc0, 6),   // 0
            Br(Vccz, 3),   // 1 -> B or A
            Nop,           // 2 A
            Nop,           // 3 B
            Br(Execnz, 2), // 4 B -> A
            Nop,           // 5
            Br(Scc1, 12),  // 6
            Br(Vccnz, 9),  // 7 -> D or C
            Nop,           // 8 C
            Nop,           // 9 D
            Br(Execz, 8),  // 10 D -> C
            Nop,           // 11
            End,           // 12
        ];
        let (_, cfg) = build(&ops);
        let node = lift_and_check(&cfg);
        assert_eq!(node.dispatch_count(), 2, "{node:#?}");
    }

    // --- S49: Nested loops, diamonds-in-loops, early exits, truncation, determinism ---

    #[test]
    fn nested_three_level_loops_with_early_break_and_continue() {
        // 3 nested loops with early break and continue at every nesting level:
        // outer: 1..=14 (exit 15)
        //   middle: 3..=12 (exit 13)
        //     inner: 5..=9 (exit 10)
        let ops = [
            Nop,           // 0: pre-loop
            Nop,           // 1: outer header
            Br(Scc0, 15),  // 2: outer early break
            Nop,           // 3: middle header
            Br(Vccz, 13),  // 4: middle early break
            Nop,           // 5: inner header
            Br(Execz, 10), // 6: inner early break
            Br(Scc1, 5),   // 7: inner early continue
            Nop,           // 8: inner body
            Br(Vccnz, 5),  // 9: inner latch -> 5
            Nop,           // 10: middle post-inner
            Br(Execnz, 3), // 11: middle early continue
            Br(Scc1, 3),   // 12: middle latch -> 3
            Br(Vccz, 1),   // 13: outer early continue
            Br(Execnz, 1), // 14: outer latch -> 1
            End,           // 15: outer exit
        ];
        let (_, cfg) = build(&ops);
        let node = lift_and_check(&cfg);
        assert!(node.is_fully_structured(), "{node:#?}");
        let wgsl = emit_wgsl_control_flow(&node, &cfg, |_| String::new());
        assert!(wgsl.contains("_pg6_loop0"), "{wgsl}");
        assert!(wgsl.contains("_pg6_loop1"), "{wgsl}");
        assert!(wgsl.contains("_pg6_loop2"), "{wgsl}");
    }

    #[test]
    fn nested_loop_multilevel_break_dispatches_locally() {
        // Structured prefix `if`, then an outer loop whose inner loop breaks
        // directly to the outer exit (no WGSL labeled break), then a structured
        // post-loop `if`. Only the outer loop should fall back to `Dispatch`.
        let ops = [
            Br(Scc0, 2),   // 0: prefix if-skip over 1
            Nop,           // 1: prefix body
            Nop,           // 2: outer header
            Nop,           // 3: inner header
            Br(Execz, 7),  // 4: multi-level break directly to outer exit 7
            Br(Scc1, 3),   // 5: inner latch -> 3
            Br(Vccnz, 2),  // 6: outer latch -> 2
            Br(Execnz, 9), // 7: post-loop if-skip over 8
            Nop,           // 8: post-loop body
            End,           // 9: end
        ];
        let (_, cfg) = build(&ops);
        let node = lift_and_check(&cfg);
        assert_eq!(node.dispatch_count(), 1, "{node:#?}");
        let StructuredNode::Seq(items) = &node else {
            panic!("expected top-level Seq: {node:#?}");
        };
        assert!(matches!(items[0], StructuredNode::Block(0)));
        assert!(matches!(items[1], StructuredNode::If { .. }));
        assert!(matches!(items[2], StructuredNode::Dispatch { .. }));
        // Post-loop `if` remains cleanly structured outside the dispatcher.
        assert!(items[3..].iter().any(|n| matches!(n, StructuredNode::If { .. })));
    }

    #[test]
    fn diamond_inside_loop_with_early_break_and_continue() {
        // Natural loop containing an `if/else` diamond where `then` has an
        // early `break` and `else` has an early `continue`, both merging into
        // the loop latch.
        let ops = [
            Nop,                  // 0: pre
            Nop,                  // 1: loop header
            Br(Scc0, 6),          // 2: if (scc1) then=3..6 else=6..9
            Br(Execz, 10),        // 3: then early break -> exit
            Nop,                  // 4: then body
            Br(Unconditional, 9), // 5: then -> merge/latch
            Br(Vccz, 1),          // 6: else early continue -> header
            Nop,                  // 7: else body
            Nop,                  // 8: else fallthrough -> merge/latch
            Br(Vccnz, 1),         // 9: loop latch -> header
            End,                  // 10: loop exit
        ];
        let (_, cfg) = build(&ops);
        let node = lift_and_check(&cfg);
        assert!(node.is_fully_structured(), "{node:#?}");
        let wgsl = emit_wgsl_control_flow(&node, &cfg, |_| String::new());
        assert!(
            wgsl.contains("loop {") && wgsl.contains("} else {") && wgsl.contains("break;") && wgsl.contains("continue;"),
            "{wgsl}"
        );
    }

    #[test]
    fn deeply_nested_diamonds_avoid_exponential_backtracking() {
        // 20 nested `if/else` diamonds where each `then` branch contains the
        // next inner diamond and ends with `s_branch outer_merge`. Without
        // upfront diamond detection this triggers O(2^20) backtracking and
        // exhausts the structuring budget.
        let depth = 20usize;
        // Layout:
        // - depth headers at 0..depth: Br(Scc0, else_idx[d])
        // - innermost then body at depth, followed by Br(Unconditional, merge_idx[depth-1])
        // - for each d from depth-1 down to 0:
        //   inner_merge_branch (if d < depth-1), else_body, merge_nop
        // Simpler symmetric construction: build recursively in index order.
        fn emit_nested_diamond(ops: &mut Vec<Op>, depth: usize) {
            if depth == 0 {
                ops.push(Nop);
                return;
            }
            let cond_idx = ops.len();
            ops.push(Nop); // placeholder for Br(Scc0, else_start)
            emit_nested_diamond(ops, depth - 1);
            let jump_idx = ops.len();
            ops.push(Nop); // placeholder for Br(Unconditional, merge_start)
            let else_start = ops.len();
            ops.push(Nop); // else body
            let merge_start = ops.len();
            ops.push(Nop); // merge block
            ops[cond_idx] = Br(Scc0, else_start as i64);
            ops[jump_idx] = Br(Unconditional, merge_start as i64);
        }

        let mut ops = Vec::new();
        emit_nested_diamond(&mut ops, depth);
        ops.push(End);

        let (_, cfg) = build(&ops);
        let node = lift_and_check(&cfg);
        assert!(
            node.is_fully_structured(),
            "20-deep nested diamonds should structure cleanly without budget exhaustion: {node:#?}"
        );
    }

    #[test]
    fn deeply_nested_loops_hit_depth_guard_without_stack_overflow() {
        // Exceed MAX_STRUCTURE_DEPTH with nested loops: must never stack
        // overflow, must remain semantically equivalent, and must emit valid
        // Naga WGSL by falling back to bounded `Dispatch` at the depth limit.
        let levels = MAX_STRUCTURE_DEPTH + 16;
        let mut ops = Vec::with_capacity(levels * 2 + 2);
        ops.push(Nop); // 0: entry
        for _ in 0..levels {
            ops.push(Nop); // header for level k is at index 1 + k
        }
        for k in (0..levels).rev() {
            let header = (1 + k) as i64;
            ops.push(Br(Scc1, header));
        }
        ops.push(End);

        let (_, cfg) = build(&ops);
        let node = structure_cfg(&cfg);
        assert!(
            node.dispatch_count() >= 1,
            "nesting beyond MAX_STRUCTURE_DEPTH must fall back to Dispatch"
        );
        assert_equivalent(&cfg, &node);
        validate_wgsl(&wgsl_function(&cfg, &node));
    }

    #[test]
    fn fallthrough_and_dead_backedge_after_endpgm() {
        // Part A: all s_endpgm variants followed by dead padding containing a
        // backward jump into block 0. The unreachable backedge must NOT turn
        // block 0 into a loop.
        for end_op in [
            SOPP_ENDPGM,
            SOPP_ENDPGM_SAVED,
            SOPP_ENDPGM_ORDERED_PS_DONE,
            SOPP_CODE_END,
        ] {
            let ops = [
                Nop,                    // 0: live entry
                Op::Word(enc_sopp(end_op, 0)), // 1: program end
                Nop,                    // 2: dead trailing instruction
                Br(Unconditional, 0),   // 3: dead backedge to 0
                Nop,                    // 4: dead fallthrough past end
            ];
            let (_, cfg) = build(&ops);
            let reachable = cfg.reachable();
            assert!(reachable[0]);
            assert!(!reachable[1], "post-endpgm block must be unreachable");
            let node = lift_and_check(&cfg);
            assert!(node.is_fully_structured());
            assert_eq!(
                node,
                StructuredNode::Seq(vec![StructuredNode::Block(0), StructuredNode::Return])
            );
        }

        // Part B: early s_endpgm inside a loop and inside an if-then, followed
        // by reachable code that itself falls off the end without a second
        // s_endpgm.
        let ops = [
            Nop,          // 0: loop header
            Br(Scc0, 4),  // 1: skip early return
            Nop,          // 2: early-exit prep
            End,          // 3: early s_endpgm inside loop
            Br(Vccnz, 0), // 4: reachable after s_endpgm; backedge to 0
            Nop,          // 5: loop exit, falls off end without s_endpgm
        ];
        let (_, cfg) = build(&ops);
        assert!(
            cfg.diagnostics
                .iter()
                .any(|d| matches!(d, CfgDiagnostic::FallthroughPastEnd { instr_index: 5, .. }))
        );
        let node = lift_and_check(&cfg);
        assert!(node.is_fully_structured(), "{node:#?}");
    }

    #[test]
    fn empty_truncated_and_extreme_offset_slices_never_panic() {
        // Empty and sub-dword truncated slices.
        for bytes in [&[][..], &[0x00], &[0x01, 0x02], &[0x01, 0x02, 0x03]] {
            let instrs = decode_program(bytes);
            let cfg = ControlFlowGraph::from_instructions(&instrs);
            let node = lift_and_check(&cfg);
            assert!(node.is_fully_structured());
        }

        // Truncated 64-bit SOP1 literal instruction (missing 2nd dword).
        let truncated_lit = enc_sop1(0x03, 0, 0xFF).to_le_bytes();
        let instrs = decode_program(&truncated_lit);
        let cfg = ControlFlowGraph::from_instructions(&instrs);
        lift_and_check(&cfg);

        // Single-instruction streams (including self-loops and out-of-range jumps).
        for single in [
            [Nop],
            [End],
            [Br(Unconditional, 0)],
            [Br(Scc1, 0)],
            [Br(Scc0, 99)],
            [Br(Unconditional, -10)],
        ] {
            let (_, cfg) = build(&single);
            lift_and_check(&cfg);
        }

        // Extreme high byte offsets near usize::MAX must never wrap or panic.
        let mut high = decode_program(&assemble(&[Br(Scc0, 2), End]));
        high[0].offset = usize::MAX - 8;
        high[1].offset = usize::MAX - 4;
        if let Rdna2Detail::Sopp {
            ref mut branch_target,
            ..
        } = high[0].detail
        {
            *branch_target = Some(-16);
        }
        let cfg = ControlFlowGraph::from_instructions(&high);
        assert!(cfg.diagnostics.iter().any(|d| matches!(
            d,
            CfgDiagnostic::BranchTargetOutOfRange {
                instr_index: 0,
                target: -16,
                ..
            }
        )));
        lift_and_check(&cfg);
    }

    #[test]
    fn irreducible_two_entry_loops_and_in_loop_dispatch() {
        // Case A: A 2-entry irreducible cycle nested inside a loop's `if` arm.
        // The enclosing outer loop stays cleanly structured while only the
        // irreducible arm dispatches locally.
        let ops = [
            Nop,           // 0: outer loop header
            Br(Scc0, 8),   // 1: if (scc1) enter irreducible arm 2..8
            Br(Vccz, 5),   // 2: -> entry B (5) or entry A (3)
            Nop,           // 3: A
            Nop,           // 4: A -> B
            Nop,           // 5: B
            Br(Execnz, 3), // 6: B -> A (2-entry cycle)
            Nop,           // 7: exit irreducible arm to merge 8
            Br(Vccnz, 0),  // 8: merge & outer loop latch -> 0
            End,           // 9: outer loop exit
        ];
        let (_, cfg) = build(&ops);
        let node = lift_and_check(&cfg);
        assert_eq!(node.dispatch_count(), 1, "{node:#?}");
        let StructuredNode::Seq(items) = &node else {
            panic!("expected Seq: {node:#?}");
        };
        assert!(
            matches!(items[0], StructuredNode::Loop { .. }),
            "enclosing natural loop should remain structured: {node:#?}"
        );

        // Case B: Figure-8 interlocking 2-entry cycle with early s_endpgm exit
        // and a post-cycle merge block.
        let fig8 = [
            Br(Scc0, 4),   // 0: entry -> left (1) or right (4)
            Nop,           // 1: left
            Br(Execz, 7),  // 2: left early exit
            Br(Vccnz, 4),  // 3: left -> right
            Nop,           // 4: right
            Br(Scc1, 1),   // 5: right -> left (backedge)
            Br(Execnz, 8), // 6: right -> merge
            End,           // 7: early s_endpgm
            Nop,           // 8: merge
            End,           // 9: end
        ];
        let (_, cfg2) = build(&fig8);
        let node2 = lift_and_check(&cfg2);
        assert!(node2.dispatch_count() >= 1, "{node2:#?}");
    }

    #[test]
    fn deterministic_cfg_structuring_and_wgsl_output() {
        let ops = [
            Nop,                  // 0
            Br(Scc0, 6),          // 1: if-else diamond
            Nop,                  // 2
            Br(Vccnz, 2),         // 3: inner loop inside then
            Br(Unconditional, 8), // 4: then -> merge
            Nop,                  // 5: unreachable dead block
            Br(Execz, 8),         // 6: else
            Nop,                  // 7
            Br(Scc1, 11),         // 8: irreducible 2-entry suffix
            Nop,                  // 9: A
            Nop,                  // 10: A
            Nop,                  // 11: B
            Br(Vccnz, 9),         // 12: B -> A
            End,                  // 13
        ];
        let (instrs, baseline_cfg) = build(&ops);
        let baseline_node = lift_and_check(&baseline_cfg);
        let baseline_wgsl = emit_wgsl_control_flow(&baseline_node, &baseline_cfg, test_block_body);

        for _ in 0..8 {
            let cfg = ControlFlowGraph::from_instructions(&instrs);
            let node = structure_cfg(&cfg);
            let wgsl = emit_wgsl_control_flow(&node, &cfg, test_block_body);
            assert_eq!(cfg, baseline_cfg);
            assert_eq!(node, baseline_node);
            assert_eq!(wgsl, baseline_wgsl);
        }

        // Manually permuted / duplicated / out-of-range block lists in a
        // `Dispatch` node must still emit identically sorted, valid WGSL cases.
        let mut forward = vec![0, 1, 2];
        let mut scrambled = vec![2, 999, DISPATCH_EXIT_PC as usize, 0, 1, 2, 0];
        let (_, simple_cfg) = build(&[Br(Scc0, 2), Nop, End]);
        let n1 = StructuredNode::Dispatch {
            entry: 0,
            blocks: std::mem::take(&mut forward),
            max_iterations: 64,
        };
        let n2 = StructuredNode::Dispatch {
            entry: 0,
            blocks: std::mem::take(&mut scrambled),
            max_iterations: 64,
        };
        let w1 = wgsl_function(&simple_cfg, &n1);
        let w2 = wgsl_function(&simple_cfg, &n2);
        assert_eq!(w1, w2);
        validate_wgsl(&w1);
    }
}
