//! Per-function control-flow graph produced by [`crate::discover`] and consumed by the lifter.
//!
//! Invariants:
//! - Every [`Edge::Fallthrough`], [`Edge::Taken`], [`Edge::Jump`] and [`Edge::JumpTable`] target
//!   of a block has a block with that start address in the same [`Function`].
//! - [`Edge::TailCall`] targets are entries of other discovered functions.
//! - A block with no instructions marks an address that could not be decoded; its
//!   [`BasicBlock::diagnostics`] say why and its terminator is [`Terminator::Invalid`].
//! - Every [`Function::resume_targets`] address is the start of a block of the function.

use std::collections::{BTreeMap, BTreeSet, VecDeque, btree_map::Entry};

use iced_x86::Instruction;

/// Why a basic block ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Terminator {
    /// No control-flow instruction: the next instruction starts another block.
    Fallthrough,
    /// Conditional branch (`jcc`, `loop*`, `jrcxz`, `xbegin`).
    Branch,
    /// Direct unconditional `jmp`.
    Jump,
    /// Indirect `jmp reg` / `jmp [mem]`.
    IndirectJump,
    /// `ret` (and other return-class instructions).
    Return,
    /// Call to a target listed in [`crate::code_image::CodeImage::no_return`].
    NoReturnCall,
    /// `int3`, `int1`, `int 0x29` (`__fastfail`), `ud0`/`ud1`/`ud2`, `hlt`.
    Trap,
    /// Decoding stopped on an invalid, truncated or unmapped instruction (see diagnostics). The
    /// faulting instruction is not part of the block.
    Invalid,
}

impl Terminator {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fallthrough => "fallthrough",
            Self::Branch => "branch",
            Self::Jump => "jump",
            Self::IndirectJump => "indirect_jump",
            Self::Return => "return",
            Self::NoReturnCall => "no_return_call",
            Self::Trap => "trap",
            Self::Invalid => "invalid",
        }
    }
}

/// Typed successor edge of a basic block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Edge {
    /// Sequential flow into the next block (also the not-taken side of a conditional branch).
    Fallthrough(u64),
    /// Taken side of a conditional branch.
    Taken(u64),
    /// Direct unconditional jump inside the function.
    Jump(u64),
    /// Control leaves to the entry of another function without pushing a return address
    /// (`jmp`, `jcc` or fallthrough into a known function entry). The block's terminator tells
    /// which form it was.
    TailCall(u64),
    /// Unresolved indirect jump. `slot` is the VA read by a RIP-relative `jmp [rip+x]` (import
    /// thunk / IAT entry); `None` for register or computed memory operands (e.g. jump tables
    /// that could not be resolved).
    IndirectJump { slot: Option<u64> },
    /// Return to the caller.
    Return,
    /// One distinct case target of a resolved switch jump table (see [`JumpTable`]); a block
    /// ending in a resolved indirect jump has one such edge per distinct target, ascending.
    JumpTable(u64),
}

impl Edge {
    /// Target VA for direct edges (including tail calls).
    pub fn target(&self) -> Option<u64> {
        match *self {
            Self::Fallthrough(t)
            | Self::Taken(t)
            | Self::Jump(t)
            | Self::TailCall(t)
            | Self::JumpTable(t) => Some(t),
            Self::IndirectJump { .. } | Self::Return => None,
        }
    }

    /// Target VA for edges that stay inside the function.
    pub fn intra_target(&self) -> Option<u64> {
        match *self {
            Self::Fallthrough(t) | Self::Taken(t) | Self::Jump(t) | Self::JumpTable(t) => Some(t),
            _ => None,
        }
    }

    pub fn kind_str(&self) -> &'static str {
        match self {
            Self::Fallthrough(_) => "fallthrough",
            Self::Taken(_) => "taken",
            Self::Jump(_) => "jump",
            Self::TailCall(_) => "tail_call",
            Self::IndirectJump { .. } => "indirect_jump",
            Self::Return => "return",
            Self::JumpTable(_) => "jump_table",
        }
    }

    /// Total, deterministic sort key used to canonicalize and deduplicate successor edge lists.
    ///
    /// Orders `Taken` and `TailCall` ahead of `Fallthrough` so conditional branches preserve the
    /// `[Taken, Fallthrough]`, `[TailCall, Fallthrough]` and `[Taken, TailCall]` contracts while
    /// sorting all target-carrying edges ascending by target VA.
    pub fn canonical_sort_key(&self) -> (u8, u64, u8) {
        match *self {
            Self::Taken(t) => (0, t, 0),
            Self::TailCall(t) => (1, t, 0),
            Self::Fallthrough(t) => (2, t, 0),
            Self::Jump(t) => (3, t, 0),
            Self::JumpTable(t) => (4, t, 0),
            Self::IndirectJump { slot } => (5, slot.unwrap_or(0), u8::from(slot.is_some())),
            Self::Return => (6, 0, 0),
        }
    }
}

/// Callee of a `call` instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CallTarget {
    /// `call rel32`.
    Direct(u64),
    /// `call reg` / `call [mem]`. `slot` is the VA read by a RIP-relative `call [rip+x]`.
    Indirect { slot: Option<u64> },
}

/// A `call` instruction inside a block. Calls do not end blocks unless the target is known not to
/// return.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CallSite {
    /// VA of the `call` instruction.
    pub site: u64,
    pub target: CallTarget,
}

/// Byte index table of a two-level switch: the switch value selects a byte that indexes the
/// jump table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct JumpTableIndex {
    /// VA of the `movzx` that reads the index table.
    pub load_site: u64,
    /// VA of the first index byte.
    pub table: u64,
    /// Number of index bytes reachable under the proven bound.
    pub len: u64,
}

/// Why an indirect jump without an import slot stayed on run-time dispatch (see
/// [`crate::discover::Analysis::unresolved_jumps`]). Read from the jump's slice; descriptive only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum UnresolvedJump {
    /// `jmp [reg + disp]`: a function pointer read from memory (vtable / callback slot tail call).
    MemoryOperand,
    /// `jmp reg` with `reg` a 64-bit pointer loaded from memory (no table shape).
    LoadedPointer,
    /// A table-shaped read (`[base + index*4]` or `[base + index*8]`) whose base is not a proven
    /// constant on every path.
    TableBaseUnknown,
    /// A table with a proven base whose index has no proven bound, and no extent could be
    /// inferred (no `.pdata` owner, or fewer than two leading entries inside the function).
    TableUnbounded,
    /// The jump register is not written in the slice (set before a merge or passed in): a tail
    /// call through a register, or a value the slice cannot see.
    RegisterLiveIn,
    /// Any other computation.
    Computed,
}

impl UnresolvedJump {
    /// Every kind, in declaration order.
    pub const ALL: [UnresolvedJump; 6] = [
        Self::MemoryOperand,
        Self::LoadedPointer,
        Self::TableBaseUnknown,
        Self::TableUnbounded,
        Self::RegisterLiveIn,
        Self::Computed,
    ];

    /// Stable snake_case name (reports and manifests).
    pub fn name(self) -> &'static str {
        match self {
            Self::MemoryOperand => "memory_operand",
            Self::LoadedPointer => "loaded_pointer",
            Self::TableBaseUnknown => "table_base_unknown",
            Self::TableUnbounded => "table_unbounded",
            Self::RegisterLiveIn => "register_live_in",
            Self::Computed => "computed",
        }
    }
}

/// A switch dispatch resolved from its jump table (MSVC x64 shape: 32-bit entries holding RVAs
/// relative to the image base).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct JumpTable {
    /// VA of the indirect `jmp`.
    pub site: u64,
    /// VA of the instruction that reads the table entry.
    pub load_site: u64,
    /// VA of the first entry.
    pub table: u64,
    /// Case target of every entry, in table order (`targets.len()` entries).
    pub targets: Vec<u64>,
    /// Set for the two-level variant.
    pub index: Option<JumpTableIndex>,
    /// No bound on the index was proven: `targets` is the table's leading run of entries that
    /// land inside the jumping function (`.pdata` range). The lifted `switch` on the computed
    /// target sends every other value through the runtime dispatcher, as for an unresolved jump.
    pub inferred: bool,
}

impl JumpTable {
    /// Distinct case targets sorted in ascending VA order.
    pub fn unique_targets(&self) -> Vec<u64> {
        let unique: BTreeSet<u64> = self.targets.iter().copied().collect();
        unique.into_iter().collect()
    }
}

/// Category of a discovery problem.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DiagnosticKind {
    /// The bytes at `address` do not decode to a valid instruction.
    InvalidInstruction,
    /// The instruction at `address` runs past the end of its section's bytes.
    TruncatedInstruction,
    /// `address` is not backed by executable section bytes.
    NotExecutable,
    /// `address` is not inside any section.
    Unmapped,
    /// Decoding at `site` overlaps a differently aligned instruction stream at `address`.
    OverlappingInstruction,
}

impl DiagnosticKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::InvalidInstruction => "invalid_instruction",
            Self::TruncatedInstruction => "truncated_instruction",
            Self::NotExecutable => "not_executable",
            Self::Unmapped => "unmapped",
            Self::OverlappingInstruction => "overlapping_instruction",
        }
    }
}

/// A non-fatal problem found while decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Diagnostic {
    /// VA of the responsible instruction or block start.
    pub site: u64,
    /// The offending address (equal to `site` for decode failures).
    pub address: u64,
    pub kind: DiagnosticKind,
}

/// Straight-line run of instructions with a single entry at `start`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BasicBlock {
    /// VA of the first instruction.
    pub start: u64,
    /// VA one past the last instruction (`== start` for an empty block).
    pub end: u64,
    /// Instructions in address order; the last one is the terminator unless the block falls
    /// through or ends in [`Terminator::Invalid`].
    pub instructions: Vec<Instruction>,
    pub terminator: Terminator,
    /// Successors: `[Taken, Fallthrough]` for branches, one [`Edge::JumpTable`] per distinct
    /// target for a resolved jump table, one edge otherwise, none for traps, non-returning calls
    /// and invalid blocks.
    pub successors: Vec<Edge>,
    /// Calls inside this block, in address order.
    pub calls: Vec<CallSite>,
    /// Problems found in this block, in address order.
    pub diagnostics: Vec<Diagnostic>,
}

impl BasicBlock {
    pub fn new(start: u64) -> Self {
        Self {
            start,
            end: start,
            instructions: Vec::new(),
            terminator: Terminator::Fallthrough,
            successors: Vec::new(),
            calls: Vec::new(),
            diagnostics: Vec::new(),
        }
    }

    /// Whether `va` is inside `[start, end)`.
    pub fn contains(&self, va: u64) -> bool {
        va >= self.start && va < self.end
    }

    /// Canonically sort and deduplicate `successors`, `calls`, and `diagnostics` in place.
    pub fn canonicalize(&mut self) {
        self.successors.sort_by_key(Edge::canonical_sort_key);
        self.successors.dedup();
        self.calls.sort();
        self.calls.dedup();
        self.diagnostics.sort();
        self.diagnostics.dedup();
    }

    /// Return a canonically sorted, deduplicated copy of `self.successors`.
    pub fn canonical_successors(&self) -> Vec<Edge> {
        let mut edges = self.successors.clone();
        edges.sort_by_key(Edge::canonical_sort_key);
        edges.dedup();
        edges
    }

    /// Distinct intra-function successor block VAs, sorted ascending.
    pub fn intra_successors(&self) -> Vec<u64> {
        let unique: BTreeSet<u64> = self
            .successors
            .iter()
            .filter_map(Edge::intra_target)
            .collect();
        unique.into_iter().collect()
    }
}

/// Structural integrity problem detected by [`Function::validate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CfgValidationIssue {
    /// `Function::entry` is not present in `Function::blocks`.
    MissingEntry { entry: u64 },
    /// A block map key does not match its `BasicBlock::start`.
    MismatchedBlockStart { key: u64, start: u64 },
    /// A block has `end < start`.
    InvertedSpan { block: u64, start: u64, end: u64 },
    /// An intra-function edge points to a VA with no block in `Function::blocks`.
    MissingTarget { from: u64, target: u64 },
    /// A `resume_targets` address has no block in `Function::blocks`.
    MissingResumeTarget { target: u64 },
}

/// A discovered function: its entry plus every block reachable through intra-function edges.
/// Blocks reachable from several functions (shared tails) are copied into each of them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Function {
    pub entry: u64,
    /// Keyed by block start VA.
    pub blocks: BTreeMap<u64, BasicBlock>,
    /// Entries of functions reached through direct calls.
    pub callees: BTreeSet<u64>,
    /// Block starts where execution can resume after an exception unwind or `longjmp`
    /// (`RtlRestoreContext` continuing at an arbitrary `rip` inside this function): C++ catch
    /// continuations, SEH `__except` bodies, the return sites of `setjmp`-family calls and
    /// address-taken code inside the function (see [`crate::eh`] and [`crate::discover`]).
    pub resume_targets: BTreeSet<u64>,
}

impl Function {
    /// Construct an empty function rooted at `entry`.
    pub fn new(entry: u64) -> Self {
        Self {
            entry,
            blocks: BTreeMap::new(),
            callees: BTreeSet::new(),
            resume_targets: BTreeSet::new(),
        }
    }

    /// Deterministically construct a canonicalized function from an arbitrary (possibly permuted
    /// or duplicated) sequence of blocks.
    pub fn from_blocks(entry: u64, blocks: impl IntoIterator<Item = BasicBlock>) -> Self {
        let mut func = Self::new(entry);
        for mut block in blocks {
            block.canonicalize();
            match func.blocks.entry(block.start) {
                Entry::Vacant(slot) => {
                    slot.insert(block);
                }
                Entry::Occupied(mut slot) => {
                    let existing = slot.get_mut();
                    if existing.instructions.is_empty() && !block.instructions.is_empty() {
                        existing.instructions = block.instructions;
                        existing.end = block.end;
                        existing.terminator = block.terminator;
                    } else if existing.instructions.is_empty() && block.end > existing.end {
                        existing.end = block.end;
                    }
                    existing.successors.extend(block.successors);
                    existing.calls.extend(block.calls);
                    existing.diagnostics.extend(block.diagnostics);
                    existing.canonicalize();
                }
            }
        }
        func.canonicalize();
        for block in func.blocks.values() {
            for call in &block.calls {
                if let CallTarget::Direct(target) = call.target {
                    func.callees.insert(target);
                }
            }
        }
        func
    }

    /// Canonically sort and deduplicate every block's successor, call, and diagnostic lists.
    pub fn canonicalize(&mut self) {
        for block in self.blocks.values_mut() {
            block.canonicalize();
        }
    }

    pub fn entry_block(&self) -> Option<&BasicBlock> {
        self.blocks.get(&self.entry)
    }

    pub fn instruction_count(&self) -> usize {
        self.blocks
            .values()
            .fold(0usize, |acc, b| acc.saturating_add(b.instructions.len()))
    }

    /// All calls in address order.
    pub fn calls(&self) -> impl Iterator<Item = &CallSite> {
        self.blocks.values().flat_map(|b| b.calls.iter())
    }

    /// All diagnostics in block order.
    pub fn diagnostics(&self) -> impl Iterator<Item = &Diagnostic> {
        self.blocks.values().flat_map(|b| b.diagnostics.iter())
    }

    /// All tail-call targets in block order.
    pub fn tail_calls(&self) -> impl Iterator<Item = u64> + '_ {
        self.blocks
            .values()
            .flat_map(|b| b.successors.iter())
            .filter_map(|e| match e {
                Edge::TailCall(t) => Some(*t),
                _ => None,
            })
    }

    /// Distinct tail-call targets sorted in ascending VA order.
    pub fn canonical_tail_calls(&self) -> Vec<u64> {
        let unique: BTreeSet<u64> = self.tail_calls().collect();
        unique.into_iter().collect()
    }

    /// Canonically sorted, deduplicated intra-function successor VAs of `va` that exist in
    /// `self.blocks`. Returns an empty `Vec` if `va` is missing or has no in-graph successors.
    pub fn intra_successors(&self, va: u64) -> Vec<u64> {
        let Some(block) = self.blocks.get(&va) else {
            return Vec::new();
        };
        let unique: BTreeSet<u64> = block
            .successors
            .iter()
            .filter_map(Edge::intra_target)
            .filter(|target| self.blocks.contains_key(target))
            .collect();
        unique.into_iter().collect()
    }

    /// Deterministic map from every block VA to its canonically sorted, deduplicated in-graph
    /// intra-function successor VAs.
    pub fn successors(&self) -> BTreeMap<u64, Vec<u64>> {
        self.blocks
            .keys()
            .map(|&va| (va, self.intra_successors(va)))
            .collect()
    }

    /// Deterministic map from every block VA to its canonically sorted, deduplicated predecessor
    /// VAs. Only edges targeting blocks present in `self.blocks` are recorded.
    pub fn predecessors(&self) -> BTreeMap<u64, Vec<u64>> {
        let mut preds: BTreeMap<u64, BTreeSet<u64>> = self
            .blocks
            .keys()
            .map(|&va| (va, BTreeSet::new()))
            .collect();

        for (&src, block) in &self.blocks {
            for edge in &block.successors {
                if let Some(dst) = edge.intra_target()
                    && let Some(dst_preds) = preds.get_mut(&dst)
                {
                    dst_preds.insert(src);
                }
            }
        }

        preds
            .into_iter()
            .map(|(va, set)| (va, set.into_iter().collect()))
            .collect()
    }

    /// Canonically sorted, deduplicated predecessor VAs of `va` (empty if `va` is not in
    /// `self.blocks`).
    pub fn predecessors_of(&self, va: u64) -> Vec<u64> {
        if !self.blocks.contains_key(&va) {
            return Vec::new();
        }
        let mut preds: BTreeSet<u64> = BTreeSet::new();
        for (&src, block) in &self.blocks {
            if block
                .successors
                .iter()
                .any(|edge| edge.intra_target() == Some(va))
            {
                preds.insert(src);
            }
        }
        preds.into_iter().collect()
    }

    /// Deterministic, cycle-safe iterative BFS starting at `start` over existing intra-function
    /// edges (ascending successor order). Returns empty if `start` is absent from `self.blocks`.
    pub fn bfs_from(&self, start: u64) -> Vec<u64> {
        self.reachable_from_roots([start])
    }

    /// Deterministic, cycle-safe iterative BFS starting from `roots` (sorted, deduplicated, and
    /// filtered to existing blocks in `self.blocks`).
    pub fn reachable_from_roots(&self, roots: impl IntoIterator<Item = u64>) -> Vec<u64> {
        if self.blocks.is_empty() {
            return Vec::new();
        }
        let valid_roots: BTreeSet<u64> = roots
            .into_iter()
            .filter(|va| self.blocks.contains_key(va))
            .collect();
        if valid_roots.is_empty() {
            return Vec::new();
        }

        let max_nodes = self.blocks.len();
        let mut visited: BTreeSet<u64> = BTreeSet::new();
        let mut queue: VecDeque<u64> = VecDeque::with_capacity(valid_roots.len());
        let mut order: Vec<u64> = Vec::with_capacity(max_nodes);

        for root in valid_roots {
            if visited.insert(root) {
                queue.push_back(root);
            }
        }

        while let Some(va) = queue.pop_front() {
            order.push(va);
            if order.len() >= max_nodes {
                break;
            }
            for target in self.intra_successors(va) {
                if visited.insert(target) {
                    queue.push_back(target);
                }
            }
        }

        order
    }

    /// Blocks reachable from `self.entry` in deterministic BFS order.
    pub fn reachable_blocks(&self) -> Vec<u64> {
        self.bfs_from(self.entry)
    }

    /// Deterministic, cycle-safe iterative DFS postorder starting from `start` over existing
    /// intra-function edges (ascending successor order). Returns empty if `start` is absent.
    pub fn postorder_from(&self, start: u64) -> Vec<u64> {
        if !self.blocks.contains_key(&start) {
            return Vec::new();
        }
        let mut visited: BTreeSet<u64> = BTreeSet::new();
        let mut order: Vec<u64> = Vec::with_capacity(self.blocks.len());
        let mut stack: Vec<(u64, Vec<u64>, usize)> = Vec::new();

        visited.insert(start);
        stack.push((start, self.intra_successors(start), 0));

        while let Some((va, succs, next_idx)) = stack.last_mut() {
            if *next_idx < succs.len() {
                let target = succs[*next_idx];
                *next_idx += 1;
                if visited.insert(target) {
                    let target_succs = self.intra_successors(target);
                    stack.push((target, target_succs, 0));
                }
            } else {
                let finished = *va;
                stack.pop();
                order.push(finished);
            }
        }

        order
    }

    /// Reverse postorder of blocks reachable from `self.entry` (empty if entry is absent).
    pub fn reverse_postorder(&self) -> Vec<u64> {
        let mut order = self.postorder_from(self.entry);
        order.reverse();
        order
    }

    /// Deterministic, iterative DFS back-edge detection across the entire function (starting at
    /// `self.entry` when present, then any remaining blocks in ascending VA order).
    ///
    /// Returns canonically sorted, deduplicated `(src, dst)` pairs, covering self-loops, mutual
    /// cycles, and unreachable cycles without recursion or infinite loops.
    pub fn back_edges(&self) -> Vec<(u64, u64)> {
        if self.blocks.is_empty() {
            return Vec::new();
        }

        #[derive(Clone, Copy, PartialEq, Eq)]
        enum Color {
            Gray,
            Black,
        }

        let mut color: BTreeMap<u64, Color> = BTreeMap::new();
        let mut back: BTreeSet<(u64, u64)> = BTreeSet::new();
        let mut roots: Vec<u64> = Vec::with_capacity(self.blocks.len());
        if self.blocks.contains_key(&self.entry) {
            roots.push(self.entry);
        }
        for &va in self.blocks.keys() {
            if va != self.entry {
                roots.push(va);
            }
        }

        for root in roots {
            if color.contains_key(&root) {
                continue;
            }
            color.insert(root, Color::Gray);
            let mut stack: Vec<(u64, Vec<u64>, usize)> =
                vec![(root, self.intra_successors(root), 0)];
            while let Some((va, succs, next_idx)) = stack.last_mut() {
                if *next_idx < succs.len() {
                    let src = *va;
                    let dst = succs[*next_idx];
                    *next_idx += 1;
                    match color.get(&dst).copied() {
                        None => {
                            color.insert(dst, Color::Gray);
                            stack.push((dst, self.intra_successors(dst), 0));
                        }
                        Some(Color::Gray) => {
                            back.insert((src, dst));
                        }
                        Some(Color::Black) => {}
                    }
                } else {
                    let finished = *va;
                    stack.pop();
                    color.insert(finished, Color::Black);
                }
            }
        }

        back.into_iter().collect()
    }

    /// Whether the intra-function CFG contains any cycle (including self-loops or unreachable
    /// cycles).
    pub fn has_cycle(&self) -> bool {
        !self.back_edges().is_empty()
    }

    /// Deterministic map from each block VA with dangling intra-function edges to its sorted,
    /// deduplicated list of missing target VAs.
    pub fn missing_targets(&self) -> BTreeMap<u64, Vec<u64>> {
        let mut missing: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
        for (&src, block) in &self.blocks {
            let dangling: BTreeSet<u64> = block
                .successors
                .iter()
                .filter_map(Edge::intra_target)
                .filter(|target| !self.blocks.contains_key(target))
                .collect();
            if !dangling.is_empty() {
                missing.insert(src, dangling.into_iter().collect());
            }
        }
        missing
    }

    /// Check CFG structural invariants without panicking, returning issues in canonical order.
    pub fn validate(&self) -> Vec<CfgValidationIssue> {
        let mut issues: BTreeSet<CfgValidationIssue> = BTreeSet::new();

        if !self.blocks.contains_key(&self.entry) {
            issues.insert(CfgValidationIssue::MissingEntry { entry: self.entry });
        }

        for (&key, block) in &self.blocks {
            if key != block.start {
                issues.insert(CfgValidationIssue::MismatchedBlockStart {
                    key,
                    start: block.start,
                });
            }
            if block.end < block.start {
                issues.insert(CfgValidationIssue::InvertedSpan {
                    block: key,
                    start: block.start,
                    end: block.end,
                });
            }
            for edge in &block.successors {
                if let Some(target) = edge.intra_target()
                    && !self.blocks.contains_key(&target)
                {
                    issues.insert(CfgValidationIssue::MissingTarget { from: key, target });
                }
            }
        }

        for &target in &self.resume_targets {
            if !self.blocks.contains_key(&target) {
                issues.insert(CfgValidationIssue::MissingResumeTarget { target });
            }
        }

        issues.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_block(start: u64, end: u64, terminator: Terminator, successors: Vec<Edge>) -> BasicBlock {
        BasicBlock {
            start,
            end,
            instructions: Vec::new(),
            terminator,
            successors,
            calls: Vec::new(),
            diagnostics: Vec::new(),
        }
    }

    #[test]
    fn empty_graph_operations_are_safe_and_deterministic() {
        let mut func = Function::new(0x1000);
        func.canonicalize();

        assert!(func.entry_block().is_none());
        assert_eq!(func.instruction_count(), 0);
        assert_eq!(func.calls().count(), 0);
        assert_eq!(func.diagnostics().count(), 0);
        assert_eq!(func.tail_calls().count(), 0);
        assert!(func.canonical_tail_calls().is_empty());
        assert!(func.intra_successors(0x1000).is_empty());
        assert!(func.successors().is_empty());
        assert!(func.predecessors().is_empty());
        assert!(func.predecessors_of(0x1000).is_empty());
        assert!(func.bfs_from(0x1000).is_empty());
        assert!(func.reachable_blocks().is_empty());
        assert!(func.reachable_from_roots([0x2000, 0x1000, 0x1000]).is_empty());
        assert!(func.postorder_from(0x1000).is_empty());
        assert!(func.reverse_postorder().is_empty());
        assert!(func.back_edges().is_empty());
        assert!(!func.has_cycle());
        assert!(func.missing_targets().is_empty());
        assert_eq!(
            func.validate(),
            vec![CfgValidationIssue::MissingEntry { entry: 0x1000 }]
        );
    }

    #[test]
    fn self_loop_dedup_traversal_and_cycle_detection() {
        let loop_block = make_block(
            0x1000,
            0x1008,
            Terminator::Branch,
            vec![
                Edge::Fallthrough(0x1000),
                Edge::Taken(0x1000),
                Edge::Taken(0x1000),
            ],
        );
        let func = Function::from_blocks(0x1000, [loop_block]);

        let block = func.entry_block().expect("entry block");
        assert_eq!(
            block.successors,
            vec![Edge::Taken(0x1000), Edge::Fallthrough(0x1000)]
        );
        assert_eq!(func.intra_successors(0x1000), vec![0x1000]);
        assert_eq!(func.predecessors_of(0x1000), vec![0x1000]);
        assert_eq!(func.predecessors(), BTreeMap::from([(0x1000, vec![0x1000])]));
        assert_eq!(func.reachable_blocks(), vec![0x1000]);
        assert_eq!(func.postorder_from(0x1000), vec![0x1000]);
        assert_eq!(func.reverse_postorder(), vec![0x1000]);
        assert_eq!(func.back_edges(), vec![(0x1000, 0x1000)]);
        assert!(func.has_cycle());
        assert!(func.validate().is_empty());
    }

    #[test]
    fn mutual_cycles_reachable_and_unreachable_do_not_loop() {
        let blocks = vec![
            make_block(0x1000, 0x1004, Terminator::Jump, vec![Edge::Jump(0x1010)]),
            make_block(
                0x1010,
                0x1018,
                Terminator::Branch,
                vec![Edge::Fallthrough(0x1020), Edge::Taken(0x1000)],
            ),
            make_block(0x1020, 0x1021, Terminator::Return, vec![Edge::Return]),
            // Unreachable mutual cycle: 0x2000 <-> 0x2010
            make_block(0x2000, 0x2004, Terminator::Jump, vec![Edge::Jump(0x2010)]),
            make_block(0x2010, 0x2014, Terminator::Jump, vec![Edge::Jump(0x2000)]),
        ];
        let func = Function::from_blocks(0x1000, blocks);

        assert_eq!(func.reachable_blocks(), vec![0x1000, 0x1010, 0x1020]);
        assert_eq!(
            func.reachable_from_roots([0x2010, 0x1000]),
            vec![0x1000, 0x2010, 0x1010, 0x2000, 0x1020]
        );
        assert_eq!(func.postorder_from(0x1000), vec![0x1020, 0x1010, 0x1000]);
        assert_eq!(func.reverse_postorder(), vec![0x1000, 0x1010, 0x1020]);
        assert_eq!(
            func.predecessors(),
            BTreeMap::from([
                (0x1000, vec![0x1010]),
                (0x1010, vec![0x1000]),
                (0x1020, vec![0x1010]),
                (0x2000, vec![0x2010]),
                (0x2010, vec![0x2000]),
            ])
        );
        assert_eq!(func.back_edges(), vec![(0x1010, 0x1000), (0x2010, 0x2000)]);
        assert!(func.has_cycle());
        assert!(func.validate().is_empty());
    }

    #[test]
    fn missing_targets_and_missing_entry_are_guarded() {
        let mut func = Function::from_blocks(
            0x9000,
            vec![
                make_block(
                    0x1000,
                    0x1008,
                    Terminator::Branch,
                    vec![
                        Edge::Taken(0xDEAD),
                        Edge::Fallthrough(0x1010),
                        Edge::Jump(0xBEEF),
                        Edge::Jump(0xDEAD),
                    ],
                ),
                make_block(
                    0x1010,
                    0x1014,
                    Terminator::Jump,
                    vec![Edge::Jump(0xFEED), Edge::TailCall(0x5000)],
                ),
            ],
        );
        func.resume_targets.insert(0x1010);
        func.resume_targets.insert(0xCAFE);

        // Traversals from missing entry or over missing targets never panic or invent nodes.
        assert!(func.reachable_blocks().is_empty());
        assert!(func.reverse_postorder().is_empty());
        assert_eq!(func.bfs_from(0x1000), vec![0x1000, 0x1010]);
        assert_eq!(func.postorder_from(0x1000), vec![0x1010, 0x1000]);
        assert_eq!(func.intra_successors(0x1000), vec![0x1010]);
        assert!(func.intra_successors(0x1010).is_empty());
        assert!(func.intra_successors(0xDEAD).is_empty());
        assert_eq!(
            func.predecessors(),
            BTreeMap::from([(0x1000, vec![]), (0x1010, vec![0x1000])])
        );
        assert!(func.predecessors_of(0xDEAD).is_empty());
        assert!(!func.has_cycle());

        assert_eq!(
            func.missing_targets(),
            BTreeMap::from([
                (0x1000, vec![0xBEEF, 0xDEAD]),
                (0x1010, vec![0xFEED]),
            ])
        );
        assert_eq!(
            func.validate(),
            vec![
                CfgValidationIssue::MissingEntry { entry: 0x9000 },
                CfgValidationIssue::MissingTarget {
                    from: 0x1000,
                    target: 0xBEEF,
                },
                CfgValidationIssue::MissingTarget {
                    from: 0x1000,
                    target: 0xDEAD,
                },
                CfgValidationIssue::MissingTarget {
                    from: 0x1010,
                    target: 0xFEED,
                },
                CfgValidationIssue::MissingResumeTarget { target: 0xCAFE },
            ]
        );
    }

    #[test]
    fn permuted_and_duplicated_edges_and_blocks_produce_identical_canonical_cfg() {
        let diag_a = Diagnostic {
            site: 0x1002,
            address: 0x1002,
            kind: DiagnosticKind::InvalidInstruction,
        };
        let diag_b = Diagnostic {
            site: 0x1006,
            address: 0x1006,
            kind: DiagnosticKind::OverlappingInstruction,
        };
        let call_a = CallSite {
            site: 0x1001,
            target: CallTarget::Direct(0x4000),
        };
        let call_b = CallSite {
            site: 0x1004,
            target: CallTarget::Indirect { slot: Some(0x8000) },
        };

        let mut b1_a = make_block(
            0x1000,
            0x1010,
            Terminator::IndirectJump,
            vec![
                Edge::JumpTable(0x1030),
                Edge::JumpTable(0x1010),
                Edge::JumpTable(0x1020),
                Edge::JumpTable(0x1010),
                Edge::JumpTable(0x1030),
            ],
        );
        b1_a.calls = vec![call_b, call_a, call_b];
        b1_a.diagnostics = vec![diag_b, diag_a, diag_a];

        let b2_a = make_block(
            0x1010,
            0x1018,
            Terminator::Branch,
            vec![
                Edge::Fallthrough(0x1030),
                Edge::Taken(0x1020),
                Edge::Fallthrough(0x1030),
            ],
        );
        let b3_a = make_block(
            0x1020,
            0x1024,
            Terminator::Jump,
            vec![Edge::TailCall(0x6000), Edge::TailCall(0x5000), Edge::TailCall(0x6000)],
        );
        let b4_a = make_block(0x1030, 0x1031, Terminator::Return, vec![Edge::Return, Edge::Return]);

        // Second construction: reversed block order, duplicate block entry, permuted edge/call/diag order.
        let mut b1_b_part1 = make_block(
            0x1000,
            0x1010,
            Terminator::IndirectJump,
            vec![Edge::JumpTable(0x1020), Edge::JumpTable(0x1030)],
        );
        b1_b_part1.calls = vec![call_a];
        b1_b_part1.diagnostics = vec![diag_a];

        let mut b1_b_part2 = make_block(
            0x1000,
            0x1000,
            Terminator::IndirectJump,
            vec![Edge::JumpTable(0x1010), Edge::JumpTable(0x1020)],
        );
        b1_b_part2.calls = vec![call_b];
        b1_b_part2.diagnostics = vec![diag_b];

        let b2_b = make_block(
            0x1010,
            0x1018,
            Terminator::Branch,
            vec![Edge::Taken(0x1020), Edge::Fallthrough(0x1030)],
        );
        let b3_b = make_block(
            0x1020,
            0x1024,
            Terminator::Jump,
            vec![Edge::TailCall(0x5000), Edge::TailCall(0x6000)],
        );
        let b4_b = make_block(0x1030, 0x1031, Terminator::Return, vec![Edge::Return]);

        let func_a = Function::from_blocks(0x1000, [b1_a, b2_a, b3_a, b4_a]);
        let func_b = Function::from_blocks(0x1000, [b4_b, b3_b, b1_b_part2, b2_b, b1_b_part1]);

        assert_eq!(func_a, func_b);
        assert_eq!(func_a.successors(), func_b.successors());
        assert_eq!(func_a.predecessors(), func_b.predecessors());
        assert_eq!(func_a.reachable_blocks(), func_b.reachable_blocks());
        assert_eq!(func_a.reverse_postorder(), func_b.reverse_postorder());
        assert_eq!(func_a.back_edges(), func_b.back_edges());
        assert_eq!(func_a.canonical_tail_calls(), vec![0x5000, 0x6000]);
        assert_eq!(func_a.callees, BTreeSet::from([0x4000]));
        assert_eq!(
            func_a.blocks[&0x1000].successors,
            vec![
                Edge::JumpTable(0x1010),
                Edge::JumpTable(0x1020),
                Edge::JumpTable(0x1030),
            ]
        );
        assert_eq!(func_a.blocks[&0x1000].calls, vec![call_a, call_b]);
        assert_eq!(func_a.blocks[&0x1000].diagnostics, vec![diag_a, diag_b]);
    }

    #[test]
    fn branch_canonical_order_preserves_taken_and_tail_call_before_fallthrough() {
        let mut taken_fallthrough = make_block(
            0x1000,
            0x1004,
            Terminator::Branch,
            vec![Edge::Fallthrough(0x1004), Edge::Taken(0x1020)],
        );
        taken_fallthrough.canonicalize();
        assert_eq!(
            taken_fallthrough.successors,
            vec![Edge::Taken(0x1020), Edge::Fallthrough(0x1004)]
        );

        let mut cond_tail_call = make_block(
            0x1000,
            0x1004,
            Terminator::Branch,
            vec![Edge::Fallthrough(0x1004), Edge::TailCall(0x2000)],
        );
        cond_tail_call.canonicalize();
        assert_eq!(
            cond_tail_call.successors,
            vec![Edge::TailCall(0x2000), Edge::Fallthrough(0x1004)]
        );

        let mut taken_tail_call = make_block(
            0x1000,
            0x1004,
            Terminator::Branch,
            vec![Edge::TailCall(0x2000), Edge::Taken(0x1020)],
        );
        taken_tail_call.canonicalize();
        assert_eq!(
            taken_tail_call.successors,
            vec![Edge::Taken(0x1020), Edge::TailCall(0x2000)]
        );
    }

    #[test]
    fn dag_diamond_has_no_cycles_and_valid_rpo_and_predecessors() {
        let blocks = vec![
            make_block(
                0x1000,
                0x1008,
                Terminator::Branch,
                vec![Edge::Taken(0x1020), Edge::Fallthrough(0x1010)],
            ),
            make_block(0x1010, 0x1014, Terminator::Jump, vec![Edge::Jump(0x1030)]),
            make_block(0x1020, 0x1024, Terminator::Jump, vec![Edge::Jump(0x1030)]),
            make_block(0x1030, 0x1031, Terminator::Return, vec![Edge::Return]),
        ];
        let func = Function::from_blocks(0x1000, blocks);

        assert!(!func.has_cycle());
        assert!(func.back_edges().is_empty());
        assert_eq!(func.reachable_blocks(), vec![0x1000, 0x1010, 0x1020, 0x1030]);
        assert_eq!(func.predecessors_of(0x1030), vec![0x1010, 0x1020]);

        let rpo = func.reverse_postorder();
        assert_eq!(rpo.len(), 4);
        assert_eq!(rpo[0], 0x1000);
        assert_eq!(rpo[3], 0x1030);
    }

    #[test]
    fn deep_cyclic_chain_traverses_iteratively_without_stack_overflow() {
        const COUNT: u64 = 5_000;
        let mut blocks = Vec::with_capacity(COUNT as usize);
        for i in 0..COUNT {
            let va = 0x1000 + i * 4;
            let next = if i + 1 < COUNT {
                0x1000 + (i + 1) * 4
            } else {
                0x1000
            };
            blocks.push(make_block(va, va + 4, Terminator::Jump, vec![Edge::Jump(next)]));
        }
        let func = Function::from_blocks(0x1000, blocks);

        let bfs = func.reachable_blocks();
        assert_eq!(bfs.len(), COUNT as usize);
        assert_eq!(bfs.first().copied(), Some(0x1000));
        assert_eq!(bfs.last().copied(), Some(0x1000 + (COUNT - 1) * 4));

        let rpo = func.reverse_postorder();
        assert_eq!(rpo, bfs);
        assert_eq!(func.back_edges(), vec![(0x1000 + (COUNT - 1) * 4, 0x1000)]);
        assert!(func.has_cycle());
    }

    #[test]
    fn jump_table_unique_targets_and_block_span_validation() {
        let jt = JumpTable {
            site: 0x1000,
            load_site: 0x0FF8,
            table: 0x3000,
            targets: vec![0x1030, 0x1010, 0x1030, 0x1020, 0x1010],
            index: None,
            inferred: false,
        };
        assert_eq!(jt.unique_targets(), vec![0x1010, 0x1020, 0x1030]);

        let inverted = make_block(0x2000, 0x1FF0, Terminator::Invalid, Vec::new());
        assert!(!inverted.contains(0x1FF8));
        assert!(!inverted.contains(0x2000));

        let mut func = Function::new(0x1000);
        func.blocks.insert(0x1000, inverted);
        assert_eq!(
            func.validate(),
            vec![
                CfgValidationIssue::MismatchedBlockStart {
                    key: 0x1000,
                    start: 0x2000,
                },
                CfgValidationIssue::InvertedSpan {
                    block: 0x1000,
                    start: 0x2000,
                    end: 0x1FF0,
                },
            ]
        );
    }
}
