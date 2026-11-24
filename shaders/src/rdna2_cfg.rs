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

use crate::rdna2::{Rdna2Detail, Rdna2Instruction};

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
/// (bounds recursion in both structuring and emission).
const MAX_STRUCTURE_DEPTH: usize = 160;

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
        let in_range = target >= first_offset as i64 && target < end_offset as i64;
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
    /// Number of `Dispatch` fallbacks in the tree.
    pub fn dispatch_count(&self) -> usize {
        match self {
            Self::Dispatch { .. } => 1,
            Self::Seq(nodes) => nodes.iter().map(Self::dispatch_count).sum(),
            Self::If {
                then_branch,
                else_branch,
                ..
            } => {
                then_branch.dispatch_count()
                    + else_branch.as_ref().map_or(0, |e| e.dispatch_count())
            }
            Self::Loop { body } => body.dispatch_count(),
            Self::Block(_) | Self::Break | Self::Continue | Self::Return => 0,
        }
    }

    /// `true` when no `Dispatch` fallback was needed.
    pub fn is_fully_structured(&self) -> bool {
        self.dispatch_count() == 0
    }

    fn escapes(&self) -> bool {
        match self {
            Self::Break | Self::Continue | Self::Return => true,
            Self::Seq(nodes) => nodes.last().is_some_and(Self::escapes),
            Self::If {
                then_branch,
                else_branch: Some(else_branch),
                ..
            } => then_branch.escapes() && else_branch.escapes(),
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
            .filter(|b| reachable[b.id] && !b.synthetic_exit)
            .map(|b| b.id)
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
        let budget = order.len().saturating_mul(32).saturating_add(256);
        Some(Self {
            cfg,
            order,
            pos_of,
            preds,
            budget,
        })
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
        for p in lo + 1..hi {
            if self.preds[p].iter().any(|&q| q < lo || q >= hi) {
                return Err(Fail::Shape);
            }
        }

        let mut nodes = Vec::new();
        // (start position, `nodes.len()` before that node) for backtracking.
        let mut starts: Vec<(usize, usize)> = Vec::new();
        let mut i = lo;
        while i < hi {
            starts.push((i, nodes.len()));
            match self.node_at(i, hi, follow, lp, header_at_lo && i == lo, depth) {
                Ok((mut ns, end)) => {
                    nodes.append(&mut ns);
                    i = end;
                }
                Err(Fail::Shape) if lp.is_none() => {
                    // Dispatch the smallest suffix `[j, hi)` (j = a node start)
                    // whose edges stay inside / leave to `follow` / end the program.
                    // Earlier nodes only fall into `j` or return, so `[j, hi)`
                    // is single-entry.
                    let Some(&(j, len)) = starts
                        .iter()
                        .rev()
                        .find(|&&(j, _)| self.dispatchable(j, hi, follow))
                    else {
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
            let mut body = self.region(i, end, nf, Some(ctx), true, depth + 1)?;
            if !body.last().is_some_and(StructuredNode::escapes) {
                body.push(StructuredNode::Break);
            }
            let node = StructuredNode::Loop {
                body: Box::new(seq(body)),
            };
            return Ok((vec![node], end));
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

        // if (!taken) { fallthrough region }
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

        // Diamond: then-region `[q, tp)` ends with `s_branch merge`.
        if tp >= hi {
            return Err(Fail::Shape);
        }
        let then_last = self.order[tp - 1];
        let Terminator::Branch(merge) = self.cfg.blocks[then_last].terminator else {
            return Err(Fail::Shape);
        };
        let (else_end, nf2) = match self.tgt(merge)? {
            Tgt::Pos(mp) if mp > tp && mp < hi => (mp, Some(mp)),
            Tgt::Pos(mp) if Some(mp) == follow => (hi, follow),
            _ => return Err(Fail::Shape),
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
            format!("_pg6_pc = {target}u;")
        } else {
            format!("_pg6_pc = {DISPATCH_EXIT_PC}u;")
        }
    }

    fn dispatch(&mut self, entry: BlockId, blocks: &[BlockId], max_iterations: u32, indent: usize) {
        let members: BTreeSet<BlockId> = blocks.iter().copied().collect();
        let entry_pc = if members.contains(&entry) {
            entry.to_string()
        } else {
            DISPATCH_EXIT_PC.to_string()
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
                    let not_taken = self.transfer(fallthrough, &members);
                    self.line(i3 + 1, &format!("if ({}) {{", cond_expr(kind)));
                    self.line(i3 + 2, &taken);
                    self.line(i3 + 1, "} else {");
                    self.line(i3 + 2, &not_taken);
                    self.line(i3 + 1, "}");
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
}
