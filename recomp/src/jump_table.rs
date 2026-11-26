//! MSVC x64 switch jump-table recovery.
//!
//! [`crate::discover`] hands over the instruction trace that ends in an unresolved indirect `jmp`:
//! the jump's block preceded by its chain of sole predecessors, with the direction taken through
//! every conditional branch between them. A backward slice over that trace matches the dispatch
//! shapes MSVC (and Clang-cl) emit for x64 `switch` statements:
//!
//! 1. **Image-base-relative / anchor-relative 32-bit offset tables** (`mov` unsigned or `movsxd`
//!    signed, `add` or `lea`, with base loaded in the same block or a predecessor, across register
//!    copies, `lea rTab, [rBase + table_rva]`, or Win64 non-volatile registers across calls):
//!    ```text
//!        cmp    ecx, N                              ; guard, usually in a predecessor block
//!        ja     default                             ; (or jae/jb/jbe/jg/jge/jl/jle, or `and` mask)
//!        movsxd rax, ecx                            ; optional zero/sign extension
//!        lea    rdx, [rip+__ImageBase]              ; or any section/anchor address
//!      [ movzx  eax, byte/word [rdx+rax*s+idx_rva] ]; optional two-level byte/word index table
//!        mov    ecx, dword ptr [rdx+rax*4+table_rva]; or `movsxd rcx, dword ptr [...]`
//!        add    rcx, rdx                            ; or `lea rcx, [rdx+rcx]`
//!        jmp    rcx
//!    ```
//! 2. **Table-relative 32-bit signed offset tables:**
//!    ```text
//!        lea    rdx, [rip+table_va]
//!        movsxd rax, dword ptr [rdx+rcx*4]
//!        add    rax, rdx
//!        jmp    rax
//!    ```
//! 3. **Absolute 64-bit pointer tables:**
//!    ```text
//!        lea    rdx, [rip+table_va]
//!        mov    rax, qword ptr [rdx+rcx*8] ; jmp rax
//!        ; or: jmp qword ptr [rdx+rcx*8]
//!    ```
//!
//! Register order, the choice of registers and interleaved unrelated instructions do not matter;
//! what is matched is the data flow. The number of entries read comes only from a proven bound:
//! a compare/test guard on the path with no clobber in between, a bitmask (`and`) or logical right
//! shift (`shr`), a zero-extension of a byte/word, or, for the second level, the largest entry of
//! a bounded byte/word index table. Every target must be decodable code owned by the same `.pdata`
//! function as the jump (or, for code no `.pdata` range covers, by none). Anything else leaves the
//! jump unresolved; nothing is guessed.

use iced_x86::{
    Decoder, DecoderOptions, FlowControl, Instruction, InstructionInfoFactory, Mnemonic, OpAccess,
    OpKind, Register,
};

use crate::cfg::{JumpTable, JumpTableIndex, UnresolvedJump};
use crate::code_image::CodeImage;

/// Largest number of jump-table (or index-table) entries read for one dispatch.
pub const MAX_JUMP_TABLE_ENTRIES: u64 = 1 << 16;

/// Bound on nested slice steps (register copies, extensions, address chains) followed for one value.
const MAX_SLICE_DEPTH: u32 = 16;

/// Bound on total recursive slice visits per resolution/classification (guards against exponential
/// fan-out across `add`/`lea` DAGs).
const MAX_SLICE_STEPS: usize = 4096;

/// Bound on alias hops followed when matching a branch guard against an aliased register.
const MAX_ALIAS_DEPTH: u32 = 4;

/// Maximum number of distinct `!=` guard exclusions tracked per index bound.
const MAX_EXCLUDED_VALUES: usize = 8;

/// Return the unique case targets of `table` in strictly ascending, deterministic order.
///
/// [`JumpTable::targets`] preserves table-slot order (including duplicate entries) so slot-level
/// relocations remain aligned with table offsets; callers building CFG edges or canonical target
/// sets consume this sorted, deduplicated view.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn sorted_unique_targets(table: &JumpTable) -> Vec<u64> {
    let mut targets = table.targets.clone();
    targets.sort_unstable();
    targets.dedup();
    targets
}

/// Resolve an IP-relative memory operand address with checked 32-bit signed displacement arithmetic
/// so 64-bit address-space wraparound is rejected.
fn checked_rip_target(instr: &Instruction) -> Option<u64> {
    if !instr.is_ip_rel_memory_operand() {
        return None;
    }
    let next_ip = instr.next_ip();
    let target = instr.ip_rel_memory_address();
    let rel = target.wrapping_sub(next_ip) as i64;
    if !(i64::from(i32::MIN)..=i64::from(i32::MAX)).contains(&rel) {
        return None;
    }
    (next_ip.checked_add_signed(rel)? == target).then_some(target)
}

/// Add a sign-extended x86-64 memory displacement (`disp` as `i64`) to `base` without wrapping
/// across `0` or `u64::MAX`.
fn checked_disp_add(base: u64, disp: u64) -> Option<u64> {
    base.checked_add_signed(disp as i64)
}

/// One instruction of the trace. `taken` is set on a conditional branch that ends a predecessor
/// block: whether the path to the jump took it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Step<'b> {
    pub instr: &'b Instruction,
    pub taken: Option<bool>,
}

/// How jump-table entries encode target virtual addresses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EntryEncoding {
    /// 32-bit unsigned offset added to `base` (e.g. `__ImageBase + u32`).
    Unsigned32 { base: u64 },
    /// 32-bit signed offset added to `base` (e.g. `table_va + i32` or `anchor_va + i32`).
    Signed32 { base: u64 },
    /// 64-bit absolute target virtual address stored directly in the table.
    Absolute64,
}

impl EntryEncoding {
    fn entry_size(self) -> u64 {
        match self {
            Self::Unsigned32 { .. } | Self::Signed32 { .. } => 4,
            Self::Absolute64 => 8,
        }
    }

    fn decode(self, chunk: &[u8]) -> Option<u64> {
        match self {
            Self::Unsigned32 { base } => {
                let bytes: [u8; 4] = chunk.try_into().ok()?;
                let raw = u32::from_le_bytes(bytes);
                base.checked_add(u64::from(raw))
            }
            Self::Signed32 { base } => {
                let bytes: [u8; 4] = chunk.try_into().ok()?;
                let raw = i32::from_le_bytes(bytes);
                base.checked_add_signed(i64::from(raw))
            }
            Self::Absolute64 => {
                let bytes: [u8; 8] = chunk.try_into().ok()?;
                Some(u64::from_le_bytes(bytes))
            }
        }
    }
}

/// Symbolic value of a 64-bit register at a trace position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Value {
    /// Known address constant (e.g. `__ImageBase`, table VA, or section/anchor VA).
    Const(u64),
    /// Pre-computed pointer to a table element (`lea rPtr, [base + index*scale + disp]`).
    TablePtr {
        table: u64,
        index: Register,
        index_at: usize,
        index_width: u32,
        scale: u32,
    },
    /// A 32-bit table entry loaded into a register.
    Entry { load: Load, signed: bool },
    /// Resolved dispatch target address.
    Target { load: Load, encoding: EntryEncoding },
}

/// `[base + index*scale + disp]` table read at trace position `at`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Load {
    /// Trace position of the memory-read instruction (`load_site`).
    at: usize,
    /// Base VA of the table (index 0).
    table: u64,
    /// Full 64-bit register holding the unscaled table index.
    index: Register,
    /// Trace position where `index` is consumed (usually `at`, or an earlier `shl`/`lea`).
    index_at: usize,
    /// Bit width of `index` read at `index_at` (usually 64; 32 for 32-bit pre-scaling).
    index_width: u32,
}

/// Proven inclusive unsigned range `[min, max]` of an index.
#[derive(Debug, Clone, Copy)]
struct Bound {
    min: u64,
    max: u64,
    index: Option<JumpTableIndex>,
}

/// Constraint proven on a register by a conditional branch.
#[derive(Debug, Clone, Copy)]
enum GuardFact {
    /// Unsigned `reg <= max`.
    UnsignedUpper(u64),
    /// Unsigned `reg >= min`.
    UnsignedLower(u64),
    /// Signed `reg <= max`.
    SignedUpper(i64),
    /// Signed `reg >= min`.
    SignedLower(i64),
    /// Exact equality `reg == val`.
    Equal(u64),
    /// Exclusion `reg != val`.
    NotEqual(u64),
}

#[derive(Debug, Clone, Copy)]
struct RawGuard {
    reg: Register,
    bits: u32,
    fact: GuardFact,
}

/// Resolve the jump table dispatched by the indirect `jmp` that ends `trace`.
#[cfg(test)]
pub(crate) fn resolve(image: &CodeImage<'_>, trace: &[Step<'_>]) -> Option<JumpTable> {
    resolve_in(image, trace, None)
}

/// [`resolve`] for a jump whose function extent is known: `leaf` is `[entry, next entry)` of the
/// discovered function containing the jump. It is used only for a jump outside every `.pdata`
/// range (a leaf function), to bound an inferred table ([`infer_extent`]).
pub(crate) fn resolve_in(
    image: &CodeImage<'_>,
    trace: &[Step<'_>],
    leaf: Option<(u64, u64)>,
) -> Option<JumpTable> {
    let at = trace.len().checked_sub(1)?;
    let jmp = trace.get(at)?.instr;
    if jmp.is_invalid() || jmp.flow_control() != FlowControl::IndirectBranch {
        return None;
    }
    let mut slicer = Slicer::new(image, trace);
    let (load, encoding) = match jmp.op0_kind() {
        OpKind::Register if jmp.op0_register().is_gpr64() => {
            match slicer.value(jmp.op0_register(), at, 0)? {
                Value::Target { load, encoding } => (load, encoding),
                _ => return None,
            }
        }
        OpKind::Memory if !jmp.is_ip_rel_memory_operand() && jmp.memory_size().size() == 8 => {
            let load = slicer.table_load(at, 0, 8, 0)?;
            (load, EntryEncoding::Absolute64)
        }
        _ => return None,
    };
    let load_site = trace.get(load.at)?.instr.ip();

    let Some(bound) = slicer.bound(load.index, load.index_at, load.index_width, 0) else {
        return infer_extent(
            image,
            jmp,
            load_site,
            load.table,
            encoding,
            leaf,
            MAX_INFERRED_ENTRIES,
        );
    };
    if bound.min > bound.max {
        return None;
    }
    let count = bound.max.checked_sub(bound.min)?.checked_add(1)?;
    if count > MAX_JUMP_TABLE_ENTRIES {
        return None;
    }
    let count_usize = usize::try_from(count).ok()?;
    let entry_size = encoding.entry_size();
    let table_start = load.table.checked_add(bound.min.checked_mul(entry_size)?)?;
    let byte_len = count.checked_mul(entry_size)?;
    table_start.checked_add(byte_len)?;
    // The only bound is the width of a zero-extended byte/word index (e.g. its mask or guard lies
    // before a loop the slice does not cross): a table that ends before that range does is read
    // as far as its entries stay inside the function, as for a table without a bound. A bound
    // narrowed by a guard or a definition still rejects such a table.
    let width_only = slicer.width_only && bound.min == 0 && matches!(bound.max, 0xff | 0xffff);
    let infer_within_width = || {
        width_only
            .then(|| {
                infer_extent(
                    image,
                    jmp,
                    load_site,
                    load.table,
                    encoding,
                    leaf,
                    count_usize,
                )
            })
            .flatten()
    };
    let Some(bytes) = read(image, table_start, byte_len) else {
        return infer_within_width();
    };
    let owner = image.function_table.function_at(jmp.ip());
    let mut targets = Vec::with_capacity(count_usize);
    for chunk in bytes.chunks_exact(entry_size as usize) {
        let Some(target) = encoding.decode(chunk) else {
            return infer_within_width();
        };
        if image.code_bytes(target).is_err() || image.function_table.function_at(target) != owner {
            return infer_within_width();
        }
        targets.push(target);
    }
    if targets.len() != count_usize {
        return infer_within_width();
    }
    Some(JumpTable {
        site: jmp.ip(),
        load_site,
        table: table_start,
        targets,
        index: bound.index,
        inferred: false,
    })
}

/// Why the indirect jump ending `trace` did not resolve (call only after [`resolve`] failed).
pub(crate) fn classify(image: &CodeImage<'_>, trace: &[Step<'_>]) -> UnresolvedJump {
    let Some(at) = trace.len().checked_sub(1) else {
        return UnresolvedJump::Computed;
    };
    let jmp = trace[at].instr;
    if jmp.is_invalid() {
        return UnresolvedJump::Computed;
    }
    if jmp.op0_kind() == OpKind::Memory {
        let scaled = jmp.memory_index() != Register::None && jmp.memory_index_scale() == 8;
        return if scaled {
            UnresolvedJump::TableBaseUnknown
        } else {
            UnresolvedJump::MemoryOperand
        };
    }
    if jmp.op0_kind() != OpKind::Register {
        return UnresolvedJump::Computed;
    }
    let mut slicer = Slicer::new(image, trace);
    if let Some(Value::Target { .. }) = slicer.value(jmp.op0_register(), at, 0) {
        return UnresolvedJump::TableUnbounded;
    }
    // Follow register copies to the instruction that computes the target.
    let mut reg = jmp.op0_register();
    let mut before = at;
    for _ in 0..MAX_SLICE_DEPTH {
        let Some(def) = slicer.def(reg, before) else {
            return UnresolvedJump::RegisterLiveIn;
        };
        let Some(instr) = trace.get(def).map(|s| s.instr) else {
            return UnresolvedJump::Computed;
        };
        let src_reg = (instr.op_count() == 2 && instr.op1_kind() == OpKind::Register)
            .then(|| instr.op1_register());
        match instr.mnemonic() {
            Mnemonic::Mov if src_reg.is_some_and(|r| r.is_gpr64()) => {
                reg = src_reg.unwrap_or(reg);
                before = def;
                continue;
            }
            Mnemonic::Mov | Mnemonic::Movsxd
                if instr.op_count() == 2 && instr.op1_kind() == OpKind::Memory =>
            {
                return if is_table_read(instr) {
                    UnresolvedJump::TableBaseUnknown
                } else if instr.op0_register().is_gpr64() && instr.memory_size().size() == 8 {
                    UnresolvedJump::LoadedPointer
                } else {
                    UnresolvedJump::Computed
                };
            }
            Mnemonic::Add | Mnemonic::Lea => {
                // `add target, base` / `lea target, [a + b]`: table-shaped when an operand was
                // read from a table.
                let operands: Vec<Register> = if instr.mnemonic() == Mnemonic::Add {
                    std::iter::once(instr.op0_register())
                        .chain(src_reg)
                        .collect()
                } else {
                    [instr.memory_base(), instr.memory_index()]
                        .into_iter()
                        .filter(|r| *r != Register::None)
                        .collect()
                };
                let table_shaped = operands.iter().any(|&r| {
                    slicer
                        .def(r, def)
                        .and_then(|d| trace.get(d))
                        .is_some_and(|s| is_table_read(s.instr))
                });
                return if table_shaped {
                    UnresolvedJump::TableBaseUnknown
                } else {
                    UnresolvedJump::Computed
                };
            }
            _ => return UnresolvedJump::Computed,
        }
    }
    UnresolvedJump::Computed
}

/// `mov r32 / movsxd r64 / mov r64, [base + index*scale + disp]` with a scaled index (4 or 8).
fn is_table_read(instr: &Instruction) -> bool {
    matches!(instr.mnemonic(), Mnemonic::Mov | Mnemonic::Movsxd)
        && instr.op_count() == 2
        && instr.op1_kind() == OpKind::Memory
        && instr.memory_index() != Register::None
        && matches!(instr.memory_index_scale(), 4 | 8)
}

/// Fewest leading entries an inferred table must have.
const MIN_INFERRED_ENTRIES: usize = 2;
/// Most entries read for an inferred table.
const MAX_INFERRED_ENTRIES: usize = 4096;
/// Largest `.pdata` gap accepted as the extent of a leaf function's inferred table.
const MAX_LEAF_GAP: u64 = 0x1000;

/// A matched table load (`[base + index*scale]` on a known table base, the dispatch shape of a
/// compiled switch) whose index bound the slice cannot prove, e.g. a remainder computed by a loop
/// or a merge with several incoming paths. The table is read from index 0 while entries land
/// inside the `.pdata` function that contains the jump, each on an instruction start
/// ([`Boundaries`]); the first entry that does not ends it.
/// Over-reading only adds case blocks inside the function, under-reading leaves a value to the
/// runtime dispatcher (see [`JumpTable::inferred`]). A jump outside every `.pdata` range is in a
/// leaf function (no unwind data): its cases are taken from the jump to the next discovered
/// function entry (`leaf` = `[entry, next entry)`), inside the gap between the neighbouring
/// `.pdata` ranges (bounded on both sides, at most [`MAX_LEAF_GAP`] bytes). Not done without a
/// `.pdata` owner or both bounds.
fn infer_extent(
    image: &CodeImage<'_>,
    jmp: &Instruction,
    load_site: u64,
    table: u64,
    encoding: EntryEncoding,
    leaf: Option<(u64, u64)>,
    limit: usize,
) -> Option<JumpTable> {
    if limit < MIN_INFERRED_ENTRIES {
        return None;
    }
    let owner = image.function_table.function_at(jmp.ip());
    let leaf_gap = match owner {
        Some(_) => None,
        None => {
            let (gap_start, gap_end) = image
                .function_table
                .gap_at(jmp.ip())
                .filter(|(start, end)| end.checked_sub(*start).is_some_and(|g| g <= MAX_LEAF_GAP))?;
            let (entry, next) = leaf?;
            let (start, end) = (entry.max(gap_start), next.min(gap_end));
            (start <= jmp.ip() && jmp.ip() < end).then_some(())?;
            let next_ip = jmp.next_ip();
            // Reject IP wraparound and empty post-dispatch spans.
            (next_ip > jmp.ip() && next_ip < end).then_some(())?;
            // Cases after the dispatch only: an over-read entry that lands before it (e.g. inside
            // another switch's dispatch sequence) would split blocks of proven code.
            Some((next_ip, end))
        }
    };
    let inside = |target: u64| match leaf_gap {
        Some((start, end)) => (start..end).contains(&target),
        None => image.function_table.function_at(target) == owner,
    };
    let entry_size = encoding.entry_size();
    let cap = limit
        .min(MAX_INFERRED_ENTRIES)
        .min(usize::try_from(MAX_JUMP_TABLE_ENTRIES).unwrap_or(usize::MAX));
    let mut boundaries = Boundaries::default();
    let mut targets = Vec::new();
    let mut va = table;
    while targets.len() < cap {
        let Some(chunk) = read(image, va, entry_size) else {
            break;
        };
        let Some(target) = encoding.decode(chunk) else {
            break;
        };
        // An entry inside an instruction is not a case: it is past the table's end (e.g. the
        // next table, whose entries are relative to another base).
        let sweep_from = leaf_gap.map(|(start, _)| start);
        if image.code_bytes(target).is_err()
            || !inside(target)
            || !boundaries.on_boundary(image, target, sweep_from)
        {
            break;
        }
        targets.push(target);
        let Some(next_va) = va.checked_add(entry_size) else {
            break;
        };
        va = next_va;
    }
    (targets.len() >= MIN_INFERRED_ENTRIES).then(|| JumpTable {
        site: jmp.ip(),
        load_site,
        table,
        targets,
        index: None,
        inferred: true,
    })
}

/// Longest linear sweep (bytes) [`Boundaries`] decodes to place one target.
const MAX_SWEEP_BYTES: u64 = 1 << 20;

/// Instruction starts of a linear sweep over code, used to reject table entries that land inside
/// an instruction. A `.pdata` chunk holds only code, so decoding it from its start yields its
/// real instruction boundaries; a leaf function's span is swept from the given start (the
/// instruction after the dispatch). A target the sweep cannot place (no chunk, too far, invalid
/// bytes before it) counts as not on a boundary: the entry is not taken, never guessed.
#[derive(Default)]
struct Boundaries {
    /// `(sweep start, instruction starts)` of the sweeps done so far.
    sweeps: Vec<(u64, Vec<u64>)>,
}

impl Boundaries {
    fn on_boundary(&mut self, image: &CodeImage<'_>, target: u64, start: Option<u64>) -> bool {
        let Some(start) = start.or_else(|| image.function_table.range_at(target).map(|r| r.start))
        else {
            return false;
        };
        if target < start || target - start > MAX_SWEEP_BYTES {
            return false;
        }
        let idx = match self.sweeps.iter().position(|(s, _)| *s == start) {
            Some(i) => i,
            None => {
                self.sweeps.push((start, Vec::new()));
                self.sweeps.len() - 1
            }
        };
        let starts = &mut self.sweeps[idx].1;
        if starts.last().is_none_or(|&last| last < target) {
            // Extend the sweep up to `target`.
            let from = starts.last().copied().map_or(Some(start), |last| {
                // Resume after the last decoded instruction.
                image.code_bytes(last).ok().and_then(|b| {
                    let mut d = Decoder::with_ip(64, b, last, DecoderOptions::NONE);
                    let i = d.decode();
                    (!i.is_invalid()).then(|| i.next_ip())
                })
            });
            let Some(from) = from else {
                return false;
            };
            if from <= target {
                let Ok(bytes) = image.code_bytes(from) else {
                    return false;
                };
                let len = usize::try_from(target - from + 1).unwrap_or(usize::MAX);
                let bytes = &bytes[..bytes.len().min(len.saturating_add(15))];
                let mut decoder = Decoder::with_ip(64, bytes, from, DecoderOptions::NONE);
                let mut instr = Instruction::default();
                while decoder.can_decode() {
                    decoder.decode_out(&mut instr);
                    if instr.is_invalid() {
                        break;
                    }
                    starts.push(instr.ip());
                    if instr.ip() >= target {
                        break;
                    }
                }
            }
        }
        starts.binary_search(&target).is_ok()
    }
}

/// `len` initialised bytes at `va` from a single section, rejecting zero-length reads and VA
/// wraparound past `u64::MAX`.
fn read<'a>(image: &CodeImage<'a>, va: u64, len: u64) -> Option<&'a [u8]> {
    if len == 0 {
        return None;
    }
    va.checked_add(len)?;
    let section = image.section_at(va)?;
    let start = usize::try_from(va.checked_sub(section.va)?).ok()?;
    let end = start.checked_add(usize::try_from(len).ok()?)?;
    section.data.get(start..end)
}

fn mask(bits: u32) -> u64 {
    if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}

fn sign_extend(val: u64, bits: u32) -> i64 {
    if bits == 0 {
        return 0;
    }
    let shift = 64u32.saturating_sub(bits);
    ((val << shift) as i64) >> shift
}

fn min_signed(bits: u32) -> i64 {
    if bits == 0 {
        return 0;
    }
    i64::MIN >> 64u32.saturating_sub(bits)
}

fn max_signed(bits: u32) -> i64 {
    if bits == 0 {
        return 0;
    }
    i64::MAX >> 64u32.saturating_sub(bits)
}

/// `AH`/`CH`/`DH`/`BH` alias bits 8..16 of their full register.
fn is_high_byte(reg: Register) -> bool {
    matches!(
        reg,
        Register::AH | Register::CH | Register::DH | Register::BH
    )
}

/// Windows x64 ABI non-volatile (callee-saved) general-purpose registers.
fn is_win64_nonvolatile(reg: Register) -> bool {
    matches!(
        reg.full_register(),
        Register::RBX
            | Register::RBP
            | Register::RDI
            | Register::RSI
            | Register::RSP
            | Register::R12
            | Register::R13
            | Register::R14
            | Register::R15
    )
}

struct Slicer<'t, 'b, 'i, 'a> {
    image: &'i CodeImage<'a>,
    trace: &'t [Step<'b>],
    info: InstructionInfoFactory,
    /// Set when a bound came only from the width of a zero-extended byte/word (no guard or
    /// definition narrowed it): the full range `[0, 0xff]` or `[0, 0xffff]`.
    width_only: bool,
    /// Remaining recursive slice steps before aborting to bound worst-case work.
    steps_remaining: usize,
}

impl<'t, 'b, 'i, 'a> Slicer<'t, 'b, 'i, 'a> {
    fn new(image: &'i CodeImage<'a>, trace: &'t [Step<'b>]) -> Self {
        Self {
            image,
            trace,
            info: InstructionInfoFactory::new(),
            width_only: false,
            steps_remaining: MAX_SLICE_STEPS,
        }
    }

    fn consume_step(&mut self) -> bool {
        if let Some(next) = self.steps_remaining.checked_sub(1) {
            self.steps_remaining = next;
            true
        } else {
            false
        }
    }
}

impl Slicer<'_, '_, '_, '_> {
    /// Whether `instr` may write any part of the full register `reg`. Calls clobber volatile
    /// Win64 registers and preserve non-volatile callee-saved registers.
    fn writes(&mut self, instr: &Instruction, reg: Register) -> bool {
        let reg = reg.full_register();
        if matches!(
            instr.flow_control(),
            FlowControl::Call | FlowControl::IndirectCall
        ) {
            return !is_win64_nonvolatile(reg);
        }
        self.info.info(instr).used_registers().iter().any(|used| {
            used.register().full_register() == reg
                && matches!(
                    used.access(),
                    OpAccess::Write
                        | OpAccess::CondWrite
                        | OpAccess::ReadWrite
                        | OpAccess::ReadCondWrite
                )
        })
    }

    /// Position of the last write to `reg` before `before`.
    fn def(&mut self, reg: Register, before: usize) -> Option<usize> {
        let reg = reg.full_register();
        let trace = self.trace;
        (0..before)
            .rev()
            .find(|&i| self.writes(trace[i].instr, reg))
    }

    /// Whether bits 32..63 of the full 64-bit register `reg` are guaranteed zero just before
    /// `before` (e.g., produced by a 32-bit GPR write, `movzx`, or `and r64, imm32_u`).
    fn is_zero_extended_32(&mut self, reg: Register, before: usize, depth: u32) -> bool {
        if depth > MAX_ALIAS_DEPTH || !self.consume_step() {
            return false;
        }
        let reg = reg.full_register();
        let Some(at) = self.def(reg, before) else {
            return false;
        };
        let instr = self.trace[at].instr;
        if matches!(
            instr.flow_control(),
            FlowControl::Call | FlowControl::IndirectCall
        ) {
            return false;
        }
        if instr.mnemonic() == Mnemonic::Movzx
            && instr.op_count() == 2
            && instr.op0_kind() == OpKind::Register
            && (instr.op0_register().is_gpr32() || instr.op0_register().is_gpr64())
        {
            return true;
        }
        if instr.mnemonic() == Mnemonic::And
            && instr.op_count() == 2
            && instr.op0_kind() == OpKind::Register
            && instr.op0_register().is_gpr64()
            && let Ok(imm) = instr.try_immediate(1)
            && imm <= u64::from(u32::MAX)
        {
            return true;
        }
        if instr.mnemonic() == Mnemonic::Mov
            && instr.op_count() == 2
            && instr.op0_kind() == OpKind::Register
            && instr.op1_kind() == OpKind::Register
            && instr.op0_register().is_gpr64()
            && instr.op1_register().is_gpr64()
        {
            return self.is_zero_extended_32(instr.op1_register(), at, depth + 1);
        }
        // In x86-64, every unconditional write to a 32-bit GPR zeroes bits 32..63.
        // Note: `iced_x86::InstructionInfo::used_registers()` widens 32-bit GPR writes to the
        // 64-bit register in 64-bit mode, so inspect operand 0 directly alongside `op0_access()`.
        let info = self.info.info(instr);
        instr.op_count() >= 1
            && instr.op0_kind() == OpKind::Register
            && instr.op0_register().is_gpr32()
            && instr.op0_register().full_register() == reg
            && matches!(info.op0_access(), OpAccess::Write | OpAccess::ReadWrite)
    }

    /// Whether every bit of the full register `reg` from bit `bits` up is zero just before
    /// `before`: its last write is a `movzx` from at most `bits` bits, an `and` with a mask
    /// below `1 << bits`, or a `shr` by an immediate of a value zero above `bits + shift`, into a
    /// 32- or 64-bit register.
    fn is_zero_above(&mut self, reg: Register, before: usize, bits: u32) -> bool {
        self.is_zero_above_at(reg, before, bits, 0)
    }

    fn is_zero_above_at(&mut self, reg: Register, before: usize, bits: u32, depth: u32) -> bool {
        if bits == 0 || bits > 64 || depth > MAX_ALIAS_DEPTH || !self.consume_step() {
            return false;
        }
        let Some(at) = self.def(reg, before) else {
            return false;
        };
        let instr = self.trace[at].instr;
        if instr.op_count() != 2
            || instr.op0_kind() != OpKind::Register
            || !(instr.op0_register().is_gpr32() || instr.op0_register().is_gpr64())
        {
            return false;
        }
        match instr.mnemonic() {
            Mnemonic::Movzx => {
                let src_bits = match instr.op1_kind() {
                    OpKind::Register if !is_high_byte(instr.op1_register()) => {
                        instr.op1_register().size() as u32 * 8
                    }
                    OpKind::Memory => instr.memory_size().size() as u32 * 8,
                    _ => return false,
                };
                src_bits > 0 && src_bits <= bits
            }
            Mnemonic::And => instr
                .try_immediate(1)
                .is_ok_and(|imm| imm & mask(instr.op0_register().size() as u32 * 8) <= mask(bits)),
            // `movzx ecx, cx; shr ecx, 4; cmp cx, 4; ja default`: a logical right shift by an
            // immediate leaves the top `shift` bits of the (zero-extended) destination zero, and
            // moves the source's zero bits down by `shift`.
            Mnemonic::Shr if instr.op1_kind() == OpKind::Immediate8 => {
                let dst_bits = instr.op0_register().size() as u32 * 8;
                let shift = (instr.immediate8() as u32) & (dst_bits - 1);
                if dst_bits.saturating_sub(shift) <= bits {
                    return true;
                }
                shift > 0 && self.is_zero_above_at(reg, at, bits + shift, depth + 1)
            }
            _ => false,
        }
    }

    /// Symbolic value of the full register `reg` just before position `before`.
    fn value(&mut self, reg: Register, before: usize, depth: u32) -> Option<Value> {
        if depth > MAX_SLICE_DEPTH || !self.consume_step() {
            return None;
        }
        let reg = reg.full_register();
        let at = self.def(reg, before)?;
        let instr = self.trace[at].instr;
        if instr.mnemonic() == Mnemonic::Cdqe {
            return match self.value(Register::RAX, at, depth + 1)? {
                Value::Entry { load, .. } => Some(Value::Entry { load, signed: true }),
                _ => None,
            };
        }
        if instr.op_count() != 2 || instr.op0_kind() != OpKind::Register {
            return None;
        }
        let dst = instr.op0_register();
        match (instr.mnemonic(), instr.op1_kind()) {
            (Mnemonic::Lea, OpKind::Memory)
                if dst.is_gpr64() && instr.is_ip_rel_memory_operand() =>
            {
                let va = checked_rip_target(instr)?;
                self.image.contains(va).then_some(Value::Const(va))
            }
            (Mnemonic::Lea, OpKind::Memory)
                if dst.is_gpr64() && !instr.is_ip_rel_memory_operand() =>
            {
                self.lea_value(at, depth)
            }
            (Mnemonic::Mov, OpKind::Immediate64 | OpKind::Immediate32to64) if dst.is_gpr64() => {
                let va = instr.try_immediate(1).ok()?;
                self.image.contains(va).then_some(Value::Const(va))
            }
            (Mnemonic::Mov, OpKind::Register)
                if dst.is_gpr64() && instr.op1_register().is_gpr64() =>
            {
                self.value(instr.op1_register(), at, depth + 1)
            }
            (Mnemonic::Mov, OpKind::Register)
                if dst.is_gpr32() && instr.op1_register().is_gpr32() =>
            {
                match self.value(instr.op1_register(), at, depth + 1)? {
                    Value::Entry { load, .. } => Some(Value::Entry {
                        load,
                        signed: false,
                    }),
                    _ => None,
                }
            }
            (Mnemonic::Movsxd, OpKind::Register)
                if dst.is_gpr64() && instr.op1_register().is_gpr32() =>
            {
                match self.value(instr.op1_register(), at, depth + 1)? {
                    Value::Entry { load, .. } => Some(Value::Entry { load, signed: true }),
                    _ => None,
                }
            }
            (Mnemonic::Mov, OpKind::Memory) if dst.is_gpr32() => {
                let load = self.table_load(at, 1, 4, depth)?;
                Some(Value::Entry {
                    load,
                    signed: false,
                })
            }
            (Mnemonic::Movsxd, OpKind::Memory) if dst.is_gpr64() => {
                let load = self.table_load(at, 1, 4, depth)?;
                Some(Value::Entry { load, signed: true })
            }
            (Mnemonic::Mov, OpKind::Memory) if dst.is_gpr64() => {
                let load = self.table_load(at, 1, 8, depth)?;
                Some(Value::Target {
                    load,
                    encoding: EntryEncoding::Absolute64,
                })
            }
            (Mnemonic::Add, OpKind::Register)
                if dst.is_gpr64() && instr.op1_register().is_gpr64() =>
            {
                self.add_reg_value(dst, instr.op1_register(), at, 0, depth)
            }
            (Mnemonic::Add | Mnemonic::Sub, _)
                if dst.is_gpr64() && instr.try_immediate(1).is_ok() =>
            {
                let imm = instr.try_immediate(1).ok()? as i64;
                let delta = if instr.mnemonic() == Mnemonic::Sub {
                    imm.checked_neg()?
                } else {
                    imm
                };
                let val = self.value(dst, at, depth + 1)?;
                Self::offset_value(val, delta)
            }
            _ => None,
        }
    }

    fn offset_value(val: Value, delta: i64) -> Option<Value> {
        match val {
            Value::Const(va) => va.checked_add_signed(delta).map(Value::Const),
            Value::TablePtr {
                table,
                index,
                index_at,
                index_width,
                scale,
            } => Some(Value::TablePtr {
                table: table.checked_add_signed(delta)?,
                index,
                index_at,
                index_width,
                scale,
            }),
            Value::Target {
                load,
                encoding: EntryEncoding::Unsigned32 { base },
            } => Some(Value::Target {
                load,
                encoding: EntryEncoding::Unsigned32 {
                    base: base.checked_add_signed(delta)?,
                },
            }),
            Value::Target {
                load,
                encoding: EntryEncoding::Signed32 { base },
            } => Some(Value::Target {
                load,
                encoding: EntryEncoding::Signed32 {
                    base: base.checked_add_signed(delta)?,
                },
            }),
            _ => None,
        }
    }

    /// Combine two registers added together (`add a, b` or `lea dst, [a + b + disp]`).
    fn add_reg_value(
        &mut self,
        a: Register,
        b: Register,
        at: usize,
        disp: u64,
        depth: u32,
    ) -> Option<Value> {
        let lhs = self.value(a, at, depth + 1);
        let rhs = self.value(b, at, depth + 1);
        match (lhs, rhs) {
            (Some(Value::Const(base)), Some(Value::Entry { load, signed }))
            | (Some(Value::Entry { load, signed }), Some(Value::Const(base))) => {
                let adj_base = checked_disp_add(base, disp)?;
                let encoding = if signed {
                    EntryEncoding::Signed32 { base: adj_base }
                } else {
                    EntryEncoding::Unsigned32 { base: adj_base }
                };
                Some(Value::Target { load, encoding })
            }
            (Some(Value::Const(base)), _) => self.ptr_from_base_and_idx(base, b, at, disp, depth),
            (_, Some(Value::Const(base))) => self.ptr_from_base_and_idx(base, a, at, disp, depth),
            _ => None,
        }
    }

    fn ptr_from_base_and_idx(
        &mut self,
        base: u64,
        idx: Register,
        at: usize,
        disp: u64,
        depth: u32,
    ) -> Option<Value> {
        let table = checked_disp_add(base, disp)?;
        if let Some((orig_idx, orig_at, orig_width, scale)) = self.scaled_index(idx, at, depth + 1)
        {
            Some(Value::TablePtr {
                table,
                index: orig_idx,
                index_at: orig_at,
                index_width: orig_width,
                scale,
            })
        } else {
            Some(Value::TablePtr {
                table,
                index: idx.full_register(),
                index_at: at,
                index_width: 64,
                scale: 1,
            })
        }
    }

    /// Evaluate a non-RIP `lea dst64, [base + index*scale + disp]` at `at`.
    fn lea_value(&mut self, at: usize, depth: u32) -> Option<Value> {
        let instr = self.trace[at].instr;
        let segment = instr.memory_segment();
        if segment == Register::FS || segment == Register::GS {
            return None;
        }
        let base = instr.memory_base();
        let index = instr.memory_index();
        let scale = instr.memory_index_scale();
        let disp = instr.memory_displacement64();

        if base.is_gpr64() && index == Register::None {
            let v = self.value(base, at, depth + 1)?;
            return Self::offset_value(v, disp as i64);
        }
        if base.is_gpr64() && index.is_gpr64() && scale == 1 {
            return self.add_reg_value(base, index, at, disp, depth);
        }
        if base.is_gpr64()
            && index.is_gpr64()
            && scale > 1
            && let Some(Value::Const(base_va)) = self.value(base, at, depth + 1)
        {
            return Some(Value::TablePtr {
                table: checked_disp_add(base_va, disp)?,
                index: index.full_register(),
                index_at: at,
                index_width: 64,
                scale,
            });
        }
        if base == Register::None && index.is_gpr64() && scale > 1 && self.image.contains(disp) {
            return Some(Value::TablePtr {
                table: disp,
                index: index.full_register(),
                index_at: at,
                index_width: 64,
                scale,
            });
        }
        None
    }

    /// If `reg` before `before` holds `orig_idx * scale` (with `scale` in `{2, 4, 8}`), return
    /// `(orig_idx, orig_at, orig_width, scale)`.
    fn scaled_index(
        &mut self,
        reg: Register,
        before: usize,
        depth: u32,
    ) -> Option<(Register, usize, u32, u32)> {
        if depth > MAX_SLICE_DEPTH || !self.consume_step() {
            return None;
        }
        let reg = reg.full_register();
        let at = self.def(reg, before)?;
        let instr = self.trace[at].instr;
        if instr.op_count() != 2 || instr.op0_kind() != OpKind::Register {
            return None;
        }
        let dst = instr.op0_register();
        if !(dst.is_gpr32() || dst.is_gpr64()) {
            return None;
        }
        let width = if dst.is_gpr32() { 32 } else { 64 };
        match (instr.mnemonic(), instr.op1_kind()) {
            (Mnemonic::Mov, OpKind::Register)
                if dst.is_gpr64() && instr.op1_register().is_gpr64() =>
            {
                self.scaled_index(instr.op1_register(), at, depth + 1)
            }
            (Mnemonic::Shl, _) if instr.try_immediate(1).is_ok() => {
                let shift = instr.try_immediate(1).ok()?;
                if matches!(shift, 1..=3) {
                    Some((dst.full_register(), at, width, 1u32 << shift))
                } else {
                    None
                }
            }
            (Mnemonic::Add, OpKind::Register) if dst == instr.op1_register() => {
                if let Some((orig, orig_at, orig_w, s)) = self.scaled_index(dst, at, depth + 1)
                    && s <= 4
                {
                    Some((orig, orig_at, orig_w.min(width), s * 2))
                } else {
                    Some((dst.full_register(), at, width, 2))
                }
            }
            (Mnemonic::Lea, OpKind::Memory)
                if !instr.is_ip_rel_memory_operand()
                    && instr.memory_base() == Register::None
                    && instr.memory_index().is_gpr64()
                    && instr.memory_displacement64() == 0
                    && matches!(instr.memory_index_scale(), 2 | 4 | 8) =>
            {
                Some((
                    instr.memory_index().full_register(),
                    at,
                    width,
                    instr.memory_index_scale(),
                ))
            }
            _ => None,
        }
    }

    /// Match an `element`-byte table read in operand `mem_op` of the instruction at `at`.
    fn table_load(&mut self, at: usize, mem_op: u32, element: u32, depth: u32) -> Option<Load> {
        if depth > MAX_SLICE_DEPTH || !self.consume_step() {
            return None;
        }
        let instr = self.trace.get(at)?.instr;
        if instr.op_kind(mem_op) != OpKind::Memory
            || instr.is_ip_rel_memory_operand()
            || instr.memory_size().size() != element as usize
        {
            return None;
        }
        let segment = instr.memory_segment();
        if segment == Register::FS || segment == Register::GS {
            return None;
        }
        let base = instr.memory_base();
        let index = instr.memory_index();
        let scale = instr.memory_index_scale();
        let disp = instr.memory_displacement64();

        // Standard `[rBase + rIdx*element + disp]` (and swapped `[rIdx + rBase + disp]` for scale 1).
        if base.is_gpr64() && index.is_gpr64() && scale == element {
            if let Some(Value::Const(base_va)) = self.value(base, at, depth + 1)
                && let Some(table) = checked_disp_add(base_va, disp)
            {
                return Some(Load {
                    at,
                    table,
                    index: index.full_register(),
                    index_at: at,
                    index_width: 64,
                });
            }
            if element == 1
                && let Some(Value::Const(base_va)) = self.value(index, at, depth + 1)
                && let Some(table) = checked_disp_add(base_va, disp)
            {
                return Some(Load {
                    at,
                    table,
                    index: base.full_register(),
                    index_at: at,
                    index_width: 64,
                });
            }
        }

        // Pre-scaled index `[rBase + rScaled*scale + disp]` where `0 < scale < element`.
        if base.is_gpr64()
            && index.is_gpr64()
            && scale > 0
            && scale < element
            && element.is_multiple_of(scale)
        {
            let need = element / scale;
            let pairs: &[(Register, Register)] = if scale == 1 {
                &[(base, index), (index, base)]
            } else {
                &[(base, index)]
            };
            for &(b, idx) in pairs {
                if let Some(Value::Const(base_va)) = self.value(b, at, depth + 1)
                    && let Some(table) = checked_disp_add(base_va, disp)
                    && let Some((orig_idx, orig_at, orig_width, s)) =
                        self.scaled_index(idx, at, depth + 1)
                    && s == need
                {
                    return Some(Load {
                        at,
                        table,
                        index: orig_idx,
                        index_at: orig_at,
                        index_width: orig_width,
                    });
                }
            }
        }

        // Pre-computed element pointer `[rPtr + disp]` (e.g. from `lea rPtr, [rBase + rIdx*4 + ...]`).
        if base.is_gpr64()
            && index == Register::None
            && let Some(Value::TablePtr {
                table: ptr_base,
                index: orig_idx,
                index_at,
                index_width,
                scale: ptr_scale,
            }) = self.value(base, at, depth + 1)
            && ptr_scale == element
            && let Some(table) = checked_disp_add(ptr_base, disp)
        {
            return Some(Load {
                at,
                table,
                index: orig_idx,
                index_at,
                index_width,
            });
        }

        // Base-less absolute table `[rIdx*element + table_va]` (low-address image).
        if base == Register::None
            && index.is_gpr64()
            && scale == element
            && self.image.contains(disp)
        {
            return Some(Load {
                at,
                table: disp,
                index: index.full_register(),
                index_at: at,
                index_width: 64,
            });
        }

        None
    }

    /// Proven inclusive unsigned range `[min, max]` of the low `width` bits of the full register
    /// `reg` just before position `before` (`width == 64`: of the whole register).
    fn bound(&mut self, reg: Register, before: usize, width: u32, depth: u32) -> Option<Bound> {
        if depth > MAX_SLICE_DEPTH || width == 0 || width > 64 || !self.consume_step() {
            return None;
        }
        let reg = reg.full_register();
        let wmask = mask(width);
        let mut unsigned_upper: Option<u64> = None;
        let mut unsigned_lower: u64 = 0;
        let mut signed_upper: Option<i64> = None;
        let mut signed_lower: Option<i64> = None;
        let mut excluded: Vec<u64> = Vec::new();
        let mut def_at: Option<usize> = None;

        for i in (0..before).rev() {
            let step = self.trace[i];
            if self.writes(step.instr, reg) {
                def_at = Some(i);
                break;
            }
            if let Some(taken) = step.taken
                && let Some(raw) = self.guard_at(i, taken)
                && let Some(fact) = self.adapt_guard(reg, width, i, raw)
            {
                match fact {
                    GuardFact::UnsignedUpper(u) => {
                        let u = u.min(wmask);
                        unsigned_upper = Some(unsigned_upper.map_or(u, |cur| cur.min(u)));
                    }
                    GuardFact::UnsignedLower(l) => {
                        unsigned_lower = unsigned_lower.max(l);
                    }
                    GuardFact::SignedUpper(su) => {
                        signed_upper = Some(signed_upper.map_or(su, |cur| cur.min(su)));
                    }
                    GuardFact::SignedLower(sl) => {
                        signed_lower = Some(signed_lower.map_or(sl, |cur| cur.max(sl)));
                    }
                    GuardFact::Equal(v) => {
                        let v = v.min(wmask);
                        unsigned_lower = unsigned_lower.max(v);
                        unsigned_upper = Some(unsigned_upper.map_or(v, |cur| cur.min(v)));
                    }
                    GuardFact::NotEqual(v) => {
                        if !excluded.contains(&v) && excluded.len() < MAX_EXCLUDED_VALUES {
                            excluded.push(v);
                        }
                    }
                }
            }
        }

        let def_b = def_at.and_then(|at| self.def_bound(at, width, depth));

        let mut max_opt = unsigned_upper;
        if let Some(su) = signed_upper {
            if su < 0 {
                return None;
            }
            if let Some(sl) = signed_lower
                && sl < 0
            {
                return None;
            }
            let u = (su as u64).min(wmask);
            max_opt = Some(max_opt.map_or(u, |cur| cur.min(u)));
        }

        let mut min = unsigned_lower;
        if let Some(sl) = signed_lower
            && sl > 0
        {
            min = min.max(sl as u64);
        }

        let (mut min, mut max, index) = match (def_b, max_opt) {
            (Some(db), Some(gmax)) => (db.min.max(min), db.max.min(gmax), db.index),
            (Some(db), None) => (db.min.max(min), db.max, db.index),
            (None, Some(gmax)) => (min, gmax, None),
            (None, None) => return None,
        };

        if min > max {
            return None;
        }
        excluded.sort_unstable();
        while excluded.binary_search(&max).is_ok() {
            if max == min {
                return None;
            }
            max = max.checked_sub(1)?;
        }
        while excluded.binary_search(&min).is_ok() {
            if min == max {
                return None;
            }
            min = min.checked_add(1)?;
        }
        if min > max {
            return None;
        }
        Some(Bound { min, max, index })
    }

    /// Shift a proven range `[min, max]` by signed `delta`, failing if it would wrap `eff_bits`
    /// or `width`.
    fn shift_bound(deeper: Bound, delta: i128, eff_bits: u32, width: u32) -> Option<Bound> {
        if deeper.index.is_some() || eff_bits == 0 || width == 0 {
            return None;
        }
        let new_min = i128::from(deeper.min).checked_add(delta)?;
        let new_max = i128::from(deeper.max).checked_add(delta)?;
        let max_allowed = i128::from(mask(eff_bits.min(width)));
        if new_min < 0 || new_min > new_max || new_max > max_allowed {
            return None;
        }
        Some(Bound {
            min: u64::try_from(new_min).ok()?,
            max: u64::try_from(new_max).ok()?,
            index: None,
        })
    }

    /// Bound of the register written by the instruction at `at`.
    fn def_bound(&mut self, at: usize, width: u32, depth: u32) -> Option<Bound> {
        let instr = self.trace.get(at)?.instr;
        let dst = instr.op0_register();
        let src = instr.op1_register();
        let src_is_reg = instr.op_count() == 2 && instr.op1_kind() == OpKind::Register;
        let sign_extended = |bound: Bound| {
            // Sign extension preserves an unsigned bound only below 2^31.
            (width <= 32 || bound.max < 0x8000_0000).then_some(bound)
        };
        match instr.mnemonic() {
            Mnemonic::Mov if src_is_reg && dst.is_gpr32() && src.is_gpr32() => {
                self.bound(src, at, 32, depth + 1)
            }
            Mnemonic::Mov if src_is_reg && dst.is_gpr64() && src.is_gpr64() => {
                self.bound(src, at, width, depth + 1)
            }
            Mnemonic::Mov
                if instr.op_count() == 2
                    && instr.op0_kind() == OpKind::Register
                    && instr.op1_kind() == OpKind::Memory
                    && width <= 8
                    && dst.is_gpr8()
                    && !is_high_byte(dst) =>
            {
                let load = self.table_load(at, 1, 1, depth)?;
                self.index_table_from_load(load, 1, depth)
            }
            // Index reloaded from a guarded memory slot (`mov eax, [rsp+0x30]`).
            Mnemonic::Mov
                if instr.op_count() == 2
                    && instr.op0_kind() == OpKind::Register
                    && instr.op1_kind() == OpKind::Memory
                    && ((dst.is_gpr32() && instr.memory_size().size() == 4)
                        || (dst.is_gpr64() && instr.memory_size().size() == 8)) =>
            {
                self.slot_bound(at, width)
            }
            Mnemonic::Mov
                if instr.op_count() == 2
                    && instr.op0_kind() == OpKind::Register
                    && dst.is_gpr()
                    && instr.try_immediate(1).is_ok()
                    && (dst.is_gpr32()
                        || dst.is_gpr64()
                        || (!is_high_byte(dst) && width <= dst.size() as u32 * 8)) =>
            {
                let dst_bits = dst.size() as u32 * 8;
                let v = (instr.try_immediate(1).ok()? & mask(dst_bits)) & mask(width);
                Some(Bound {
                    min: v,
                    max: v,
                    index: None,
                })
            }
            Mnemonic::Xor
                if src_is_reg
                    && dst == src
                    && dst.is_gpr()
                    && (dst.is_gpr32()
                        || dst.is_gpr64()
                        || (!is_high_byte(dst) && width <= dst.size() as u32 * 8)) =>
            {
                Some(Bound {
                    min: 0,
                    max: 0,
                    index: None,
                })
            }
            Mnemonic::Movsxd if src_is_reg && dst.is_gpr64() && src.is_gpr32() => {
                self.bound(src, at, 32, depth + 1).and_then(sign_extended)
            }
            Mnemonic::Cdqe => self
                .bound(Register::RAX, at, 32, depth + 1)
                .and_then(sign_extended),
            Mnemonic::Movzx
                if src_is_reg
                    && (src.is_gpr8() || src.is_gpr16())
                    && (dst.is_gpr32() || dst.is_gpr64() || width <= dst.size() as u32 * 8) =>
            {
                let bits = src.size() as u32 * 8;
                let deeper = if is_high_byte(src) {
                    None
                } else {
                    self.bound(src, at, bits, depth + 1)
                };
                match deeper {
                    Some(b) => Some(Bound {
                        min: b.min.min(mask(bits)),
                        max: b.max.min(mask(bits)),
                        index: b.index,
                    }),
                    None => {
                        self.width_only = true;
                        Some(Bound {
                            min: 0,
                            max: mask(bits),
                            index: None,
                        })
                    }
                }
            }
            Mnemonic::Movzx
                if instr.op1_kind() == OpKind::Memory
                    && matches!(instr.memory_size().size(), 1 | 2)
                    && (dst.is_gpr32() || dst.is_gpr64()) =>
            {
                let elem = instr.memory_size().size() as u32;
                if let Some(load) = self.table_load(at, 1, elem, depth) {
                    self.index_table_from_load(load, elem, depth)
                } else if elem == 1 {
                    self.width_only = true;
                    Some(Bound {
                        min: 0,
                        max: 0xff,
                        index: None,
                    })
                } else {
                    None
                }
            }
            // Power-of-2 or arbitrary bitmask bound (`and ecx, 7`, `and rcx, 0xf`, etc.).
            Mnemonic::And
                if instr.op_count() == 2
                    && instr.op0_kind() == OpKind::Register
                    && dst.is_gpr()
                    && instr.try_immediate(1).is_ok() =>
            {
                let dst_bits = dst.size() as u32 * 8;
                if !(dst.is_gpr32() || dst.is_gpr64() || (!is_high_byte(dst) && width <= dst_bits))
                {
                    return None;
                }
                let imm_mask = instr.try_immediate(1).ok()? & mask(dst_bits);
                let deeper = self.bound(dst, at, dst_bits, depth + 1);
                let max = deeper
                    .map_or(imm_mask, |b| b.max.min(imm_mask))
                    .min(mask(width));
                Some(Bound {
                    min: 0,
                    max,
                    index: None,
                })
            }
            // Value-preserving `and r, r` / `or r, r` (often used to test sign/zero).
            Mnemonic::And | Mnemonic::Or
                if src_is_reg && dst == src && (dst.is_gpr32() || dst.is_gpr64()) =>
            {
                let eff_bits = if dst.is_gpr32() { 32 } else { width };
                self.bound(dst, at, eff_bits, depth + 1)
            }
            // Logical right shift (`shr eax, 28` -> `0..=15`).
            Mnemonic::Shr
                if instr.op_count() == 2
                    && instr.op0_kind() == OpKind::Register
                    && dst.is_gpr()
                    && instr.try_immediate(1).is_ok() =>
            {
                let dst_bits = dst.size() as u32 * 8;
                if !(dst.is_gpr32() || dst.is_gpr64() || (!is_high_byte(dst) && width <= dst_bits))
                {
                    return None;
                }
                let shift =
                    (instr.try_immediate(1).ok()? & if dst_bits == 64 { 63 } else { 31 }) as u32;
                let deeper = self.bound(dst, at, dst_bits, depth + 1);
                let wmask = mask(width);
                let (min, max) = match deeper {
                    Some(b) => {
                        let lo = b.min >> shift;
                        let hi = b.max >> shift;
                        if hi <= wmask {
                            (lo, hi)
                        } else {
                            (0, wmask)
                        }
                    }
                    None => (0, (mask(dst_bits) >> shift).min(wmask)),
                };
                Some(Bound {
                    min,
                    max,
                    index: None,
                })
            }
            // Case-offset shift: `sub r, imm` / `add r, imm` / `dec r` / `inc r`.
            Mnemonic::Sub | Mnemonic::Add
                if instr.op_count() == 2
                    && instr.op0_kind() == OpKind::Register
                    && instr.try_immediate(1).is_ok()
                    && (dst.is_gpr32() || dst.is_gpr64()) =>
            {
                let eff_bits = if dst.is_gpr32() { 32 } else { width };
                let imm = instr.try_immediate(1).ok()?;
                let delta = if instr.mnemonic() == Mnemonic::Sub {
                    -i128::from(imm & mask(eff_bits))
                } else {
                    i128::from(sign_extend(imm, eff_bits))
                };
                let deeper = self.bound(dst, at, eff_bits, depth + 1)?;
                Self::shift_bound(deeper, delta, eff_bits, width)
            }
            Mnemonic::Dec | Mnemonic::Inc
                if instr.op_count() == 1
                    && instr.op0_kind() == OpKind::Register
                    && (dst.is_gpr32() || dst.is_gpr64()) =>
            {
                let eff_bits = if dst.is_gpr32() { 32 } else { width };
                let delta = if instr.mnemonic() == Mnemonic::Dec {
                    -1
                } else {
                    1
                };
                let deeper = self.bound(dst, at, eff_bits, depth + 1)?;
                Self::shift_bound(deeper, delta, eff_bits, width)
            }
            // `lea dst, [src + disp]` (copy or case-offset shift, e.g. `lea eax, [rcx - min_case]`).
            Mnemonic::Lea
                if instr.op_count() == 2
                    && instr.op0_kind() == OpKind::Register
                    && instr.op1_kind() == OpKind::Memory
                    && !instr.is_ip_rel_memory_operand()
                    && (dst.is_gpr32() || dst.is_gpr64())
                    && instr.memory_base().is_gpr64()
                    && instr.memory_index() == Register::None =>
            {
                let eff_bits = if dst.is_gpr32() { 32 } else { width };
                let src_reg = instr.memory_base();
                let disp = instr.memory_displacement64() as i64;
                if disp == 0 {
                    self.bound(src_reg, at, eff_bits, depth + 1)
                } else {
                    let deeper = self.bound(src_reg, at, eff_bits, depth + 1)?;
                    Self::shift_bound(deeper, i128::from(disp), eff_bits, width)
                }
            }
            _ => None,
        }
    }

    /// Two-level dispatch: `movzx r, byte/word [base + index*element + index_rva]` from `load`.
    fn index_table_from_load(&mut self, load: Load, element: u32, depth: u32) -> Option<Bound> {
        if depth > MAX_SLICE_DEPTH || !self.consume_step() {
            return None;
        }
        let load_site = self.trace.get(load.at)?.instr.ip();
        let outer = self.bound(load.index, load.index_at, load.index_width, depth + 1)?;
        if outer.index.is_some() || outer.min > outer.max {
            return None;
        }
        let count = outer.max.checked_sub(outer.min)?.checked_add(1)?;
        if count > MAX_JUMP_TABLE_ENTRIES {
            return None;
        }
        let elem = u64::from(element);
        let table_start = load.table.checked_add(outer.min.checked_mul(elem)?)?;
        let byte_len = count.checked_mul(elem)?;
        table_start.checked_add(byte_len)?;
        let bytes = read(self.image, table_start, byte_len)?;
        let max = match element {
            1 => u64::from(bytes.iter().copied().max()?),
            2 => bytes
                .chunks_exact(2)
                .map(|c| u64::from(u16::from_le_bytes([c[0], c[1]])))
                .max()?,
            _ => return None,
        };
        if max >= MAX_JUMP_TABLE_ENTRIES {
            return None;
        }
        Some(Bound {
            min: 0,
            max,
            index: Some(JumpTableIndex {
                load_site,
                table: table_start,
                len: byte_len,
            }),
        })
    }

    /// Constraint implied by the conditional branch at `at` (taken or not, per `taken`) from the
    /// last flag-modifying instruction before it.
    fn guard_at(&mut self, at: usize, taken: bool) -> Option<RawGuard> {
        let jcc = self.trace.get(at)?.instr.mnemonic();
        for i in (0..at).rev() {
            let instr = self.trace[i].instr;
            if matches!(
                instr.flow_control(),
                FlowControl::Call | FlowControl::IndirectCall
            ) {
                return None;
            }
            if instr.rflags_modified() == 0 {
                continue;
            }
            match instr.mnemonic() {
                Mnemonic::Cmp => {
                    if instr.op0_kind() != OpKind::Register {
                        return None;
                    }
                    let cmp_reg = instr.op0_register();
                    let bits = cmp_reg.size() as u32 * 8;
                    if !cmp_reg.is_gpr()
                        || bits == 0
                        || bits > 64
                        || is_high_byte(cmp_reg)
                        || (i + 1..at).any(|k| self.writes(self.trace[k].instr, cmp_reg))
                    {
                        return None;
                    }
                    let imm = instr.try_immediate(1).ok()?;
                    let fact = Self::cmp_imm_fact(jcc, taken, imm, bits)?;
                    return Some(RawGuard {
                        reg: cmp_reg,
                        bits,
                        fact,
                    });
                }
                Mnemonic::Test | Mnemonic::And | Mnemonic::Or => {
                    if instr.op0_kind() != OpKind::Register {
                        return None;
                    }
                    let r = instr.op0_register();
                    let bits = r.size() as u32 * 8;
                    if !r.is_gpr()
                        || bits == 0
                        || bits > 64
                        || is_high_byte(r)
                        || (i + 1..at).any(|k| self.writes(self.trace[k].instr, r))
                    {
                        return None;
                    }
                    if instr.op1_kind() == OpKind::Register && instr.op1_register() == r {
                        let fact = match (jcc, taken) {
                            (Mnemonic::Js, false)
                            | (Mnemonic::Jns, true)
                            | (Mnemonic::Jl, false)
                            | (Mnemonic::Jge, true) => GuardFact::SignedLower(0),
                            (Mnemonic::Jle, false) | (Mnemonic::Jg, true) => {
                                GuardFact::SignedLower(1)
                            }
                            (Mnemonic::Je, true) | (Mnemonic::Jne, false) => GuardFact::Equal(0),
                            (Mnemonic::Je, false) | (Mnemonic::Jne, true) => GuardFact::NotEqual(0),
                            _ => return None,
                        };
                        return Some(RawGuard { reg: r, bits, fact });
                    }
                    if instr.mnemonic() == Mnemonic::Test
                        && let Ok(imm) = instr.try_immediate(1)
                    {
                        let imm_u = imm & mask(bits);
                        if matches!((jcc, taken), (Mnemonic::Je, true) | (Mnemonic::Jne, false)) {
                            return Some(RawGuard {
                                reg: r,
                                bits,
                                fact: GuardFact::UnsignedUpper((!imm_u) & mask(bits)),
                            });
                        }
                    }
                    return None;
                }
                _ => return None,
            }
        }
        None
    }

    /// Whether `instr` may write memory (explicit operands, pushes, string stores, calls).
    fn writes_memory(&mut self, instr: &Instruction) -> bool {
        if matches!(
            instr.flow_control(),
            FlowControl::Call | FlowControl::IndirectCall
        ) {
            return true;
        }
        self.info.info(instr).used_memory().iter().any(|m| {
            matches!(
                m.access(),
                OpAccess::Write
                    | OpAccess::CondWrite
                    | OpAccess::ReadWrite
                    | OpAccess::ReadCondWrite
            )
        })
    }

    /// Whether the memory operands of `a` (operand `a_op`) and `b` (operand `b_op`) name the same
    /// non-indexed location of the same size, e.g. the stack slot `[rsp+0x30]` in
    /// `cmp dword [rsp+0x30], 11` and `mov eax, [rsp+0x30]`.
    fn same_slot(a: &Instruction, a_op: u32, b: &Instruction, b_op: u32) -> bool {
        a.op_kind(a_op) == OpKind::Memory
            && b.op_kind(b_op) == OpKind::Memory
            && a.memory_index() == Register::None
            && b.memory_index() == Register::None
            && a.memory_segment() == b.memory_segment()
            && !matches!(a.memory_segment(), Register::FS | Register::GS)
            && a.memory_size().size() == b.memory_size().size()
            && if a.is_ip_rel_memory_operand() || b.is_ip_rel_memory_operand() {
                matches!(
                    (checked_rip_target(a), checked_rip_target(b)),
                    (Some(ta), Some(tb)) if ta == tb
                )
            } else {
                a.memory_base().is_gpr64()
                    && a.memory_base() == b.memory_base()
                    && a.memory_displacement64() == b.memory_displacement64()
            }
    }

    /// Bound of `mov r, [slot]` at `at` from `cmp [slot], imm` + `jcc` guards on the path before
    /// it, accepted only while nothing on the path between guard and load can write memory or
    /// change the slot's base register (`cmp dword [rsp+0x30], 11; ja default; ...;
    /// mov eax, [rsp+0x30]`).
    fn slot_bound(&mut self, at: usize, width: u32) -> Option<Bound> {
        if width == 0 || width > 64 {
            return None;
        }
        let load = self.trace.get(at)?.instr;
        let base = load.memory_base();
        let bits = load.memory_size().size() as u32 * 8;
        if bits == 0 || bits > 64 {
            return None;
        }
        let mut upper: Option<u64> = None;
        let mut lower: u64 = 0;
        for i in (0..at).rev() {
            let instr = self.trace[i].instr;
            if self.writes_memory(instr) || (base.is_gpr64() && self.writes(instr, base)) {
                break;
            }
            if instr.mnemonic() == Mnemonic::Cmp
                && Self::same_slot(instr, 0, load, 1)
                && let Ok(imm) = instr.try_immediate(1)
            {
                // The flags of this `cmp` reach every later `jcc` up to the next flag write.
                for k in i + 1..at {
                    let step = self.trace[k];
                    if let Some(taken) = step.taken
                        && let Some(fact) =
                            Self::cmp_imm_fact(step.instr.mnemonic(), taken, imm, bits)
                    {
                        match fact {
                            GuardFact::UnsignedUpper(u) => {
                                upper = Some(upper.map_or(u, |cur| cur.min(u)));
                            }
                            GuardFact::UnsignedLower(l) => lower = lower.max(l),
                            GuardFact::Equal(v) => {
                                lower = lower.max(v);
                                upper = Some(upper.map_or(v, |cur| cur.min(v)));
                            }
                            _ => {}
                        }
                    }
                    if step.instr.rflags_modified() != 0 {
                        break;
                    }
                }
            }
        }
        let upper = upper?;
        if upper > mask(width) {
            return Some(Bound {
                min: 0,
                max: mask(width),
                index: None,
            });
        }
        let max = upper;
        (lower <= max).then_some(Bound {
            min: lower,
            max,
            index: None,
        })
    }

    /// Fact proven about the left operand of `cmp x, imm` (`bits` wide) when the conditional
    /// branch `jcc` is taken (`taken`) or falls through.
    fn cmp_imm_fact(jcc: Mnemonic, taken: bool, imm: u64, bits: u32) -> Option<GuardFact> {
        if bits == 0 || bits > 64 {
            return None;
        }
        let imm_u = imm & mask(bits);
        let imm_s = sign_extend(imm, bits);
        let fact = match (jcc, taken) {
            (Mnemonic::Ja, false) | (Mnemonic::Jbe, true) => GuardFact::UnsignedUpper(imm_u),
            (Mnemonic::Jae, false) | (Mnemonic::Jb, true) => {
                GuardFact::UnsignedUpper(imm_u.checked_sub(1)?)
            }
            (Mnemonic::Jb, false) | (Mnemonic::Jae, true) => GuardFact::UnsignedLower(imm_u),
            (Mnemonic::Jbe, false) | (Mnemonic::Ja, true) => {
                if imm_u < mask(bits) {
                    GuardFact::UnsignedLower(imm_u.checked_add(1)?)
                } else {
                    return None;
                }
            }
            (Mnemonic::Jg, false) | (Mnemonic::Jle, true) => GuardFact::SignedUpper(imm_s),
            (Mnemonic::Jge, false) | (Mnemonic::Jl, true) => {
                if imm_s > min_signed(bits) {
                    GuardFact::SignedUpper(imm_s.checked_sub(1)?)
                } else {
                    return None;
                }
            }
            (Mnemonic::Jl, false) | (Mnemonic::Jge, true) => GuardFact::SignedLower(imm_s),
            (Mnemonic::Jle, false) | (Mnemonic::Jg, true) => {
                if imm_s < max_signed(bits) {
                    GuardFact::SignedLower(imm_s.checked_add(1)?)
                } else {
                    return None;
                }
            }
            (Mnemonic::Je, true) | (Mnemonic::Jne, false) => GuardFact::Equal(imm_u),
            (Mnemonic::Je, false) | (Mnemonic::Jne, true) => GuardFact::NotEqual(imm_u),
            (Mnemonic::Js, false) | (Mnemonic::Jns, true) if imm_u == 0 => {
                GuardFact::SignedLower(0)
            }
            _ => return None,
        };
        Some(fact)
    }

    /// Adapt `raw` (proven at `branch_at`) to `(reg, width)`, following register copies and
    /// extensions performed before `branch_at` when `reg` and `raw.reg` differ (e.g.
    /// `movsxd rax, ecx; cmp ecx, N; ja default`).
    fn adapt_guard(
        &mut self,
        reg: Register,
        width: u32,
        branch_at: usize,
        raw: RawGuard,
    ) -> Option<GuardFact> {
        let mut cur_reg = reg.full_register();
        let mut cur_width = width;
        let mut cur_pos = branch_at;
        let mut sign_ext_32 = false;
        let target_full = raw.reg.full_register();

        for _ in 0..=MAX_ALIAS_DEPTH {
            if !self.consume_step() {
                return None;
            }
            if cur_reg == target_full
                && raw.bits < 32
                && raw.bits < cur_width
                && self.is_zero_above(cur_reg, cur_pos, raw.bits)
            {
                // `movzx eax, dl; cmp al, 0x28; ja default`: the bits above the compared byte are
                // zero, so an unsigned fact about the byte holds for the whole register (a signed
                // one does not: a "negative" byte is a large zero-extended value).
                return match raw.fact {
                    fact @ (GuardFact::UnsignedUpper(_)
                    | GuardFact::UnsignedLower(_)
                    | GuardFact::Equal(_)
                    | GuardFact::NotEqual(_)) => Some(fact),
                    _ => None,
                };
            }
            if cur_reg == target_full {
                let width_ok = raw.bits >= cur_width
                    || (raw.bits == 32
                        && cur_width == 64
                        && self.is_zero_extended_32(cur_reg, cur_pos, 0));
                if width_ok {
                    if sign_ext_32 {
                        return match raw.fact {
                            GuardFact::UnsignedUpper(u) if u < 0x8000_0000 => {
                                Some(GuardFact::UnsignedUpper(u))
                            }
                            GuardFact::UnsignedLower(l) if l < 0x8000_0000 => {
                                Some(GuardFact::UnsignedLower(l))
                            }
                            GuardFact::Equal(v) if v < 0x8000_0000 => Some(GuardFact::Equal(v)),
                            GuardFact::NotEqual(v) if v < 0x8000_0000 => {
                                Some(GuardFact::NotEqual(v))
                            }
                            GuardFact::SignedUpper(su) => Some(GuardFact::SignedUpper(su)),
                            GuardFact::SignedLower(sl) => Some(GuardFact::SignedLower(sl)),
                            _ => None,
                        };
                    }
                    return Some(raw.fact);
                }
            }

            let def_at = self.def(cur_reg, cur_pos)?;
            let instr = self.trace[def_at].instr;
            let (next_reg, next_width, needs_sext) = match instr.mnemonic() {
                Mnemonic::Mov
                    if instr.op_count() == 2
                        && instr.op0_kind() == OpKind::Register
                        && instr.op1_kind() == OpKind::Register =>
                {
                    let (dst, src) = (instr.op0_register(), instr.op1_register());
                    if dst.is_gpr64() && src.is_gpr64() {
                        (src.full_register(), cur_width, false)
                    } else if dst.is_gpr32() && src.is_gpr32() {
                        (src.full_register(), cur_width.min(32), false)
                    } else {
                        return None;
                    }
                }
                Mnemonic::Movsxd
                    if instr.op_count() == 2
                        && instr.op0_kind() == OpKind::Register
                        && instr.op1_kind() == OpKind::Register
                        && instr.op0_register().is_gpr64()
                        && instr.op1_register().is_gpr32() =>
                {
                    (
                        instr.op1_register().full_register(),
                        cur_width.min(32),
                        cur_width > 32,
                    )
                }
                Mnemonic::Cdqe if cur_reg == Register::RAX => {
                    (Register::RAX, cur_width.min(32), cur_width > 32)
                }
                Mnemonic::Movzx
                    if instr.op_count() == 2
                        && instr.op0_kind() == OpKind::Register
                        && instr.op1_kind() == OpKind::Register
                        && (instr.op0_register().is_gpr32() || instr.op0_register().is_gpr64())
                        && !is_high_byte(instr.op1_register()) =>
                {
                    let src = instr.op1_register();
                    (
                        src.full_register(),
                        cur_width.min(src.size() as u32 * 8),
                        false,
                    )
                }
                Mnemonic::Lea
                    if instr.op_count() == 2
                        && instr.op0_kind() == OpKind::Register
                        && instr.op1_kind() == OpKind::Memory
                        && !instr.is_ip_rel_memory_operand()
                        && (instr.op0_register().is_gpr32() || instr.op0_register().is_gpr64())
                        && instr.memory_base().is_gpr64()
                        && instr.memory_index() == Register::None
                        && instr.memory_displacement64() == 0 =>
                {
                    let next_w = if instr.op0_register().is_gpr32() {
                        cur_width.min(32)
                    } else {
                        cur_width
                    };
                    (instr.memory_base().full_register(), next_w, false)
                }
                _ => return None,
            };
            if (def_at + 1..branch_at).any(|k| self.writes(self.trace[k].instr, next_reg)) {
                return None;
            }
            cur_reg = next_reg;
            cur_width = next_width;
            cur_pos = def_at;
            sign_ext_32 |= needs_sext;
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use crate::cfg::{Edge, Function, JumpTable, JumpTableIndex, Terminator, UnresolvedJump};
    use crate::code_image::{CodeImage, CodeSection, Region};
    use crate::discover::{Analysis, discover};
    use crate::pdata::{FunctionRange, FunctionTable};
    use crate::reloc::{RelocKind, Relocation};

    const BASE: u64 = 0x1_4000_0000;
    const TEXT_RVA: u32 = 0x1000;
    const DATA_RVA: u32 = 0x2000;
    const TEXT: u64 = BASE + TEXT_RVA as u64;
    const DATA: u64 = BASE + DATA_RVA as u64;
    /// Case blocks of [`switch_code`] (offsets); the last one is also the default.
    const CASES: [u32; 4] = [0x1b, 0x21, 0x27, 0x2d];

    /// `lea rdx, [rip+x]` at `rva` addressing `__ImageBase`.
    fn lea_rdx_image_base(rva: u32) -> Vec<u8> {
        let disp = -i64::from(rva + 7) as i32;
        [&[0x48, 0x8d, 0x15][..], &disp.to_le_bytes()].concat()
    }

    /// MSVC-style `switch (ecx)` with three cases and a default at `text_rva`, reading a 32-bit
    /// RVA table at `table_rva`.
    fn switch_code(text_rva: u32, table_rva: u32) -> Vec<u8> {
        [
            &[0x83, 0xf9, 0x03][..],           // 0: cmp ecx, 3
            &[0x77, 0x28],                     // 3: ja 0x2d
            &[0x48, 0x63, 0xc1],               // 5: movsxd rax, ecx
            &lea_rdx_image_base(text_rva + 8), // 8: lea rdx, [__ImageBase]
            &[0x8b, 0x8c, 0x82],
            &table_rva.to_le_bytes(), // f: mov ecx, [rdx+rax*4+table_rva]
            &[0x48, 0x01, 0xd1],      // 16: add rcx, rdx
            &[0xff, 0xe1],            // 19: jmp rcx
            &[0xb8, 0x00, 0x00, 0x00, 0x00, 0xc3], // 1b: mov eax, 0; ret
            &[0xb8, 0x01, 0x00, 0x00, 0x00, 0xc3], // 21: mov eax, 1; ret
            &[0xb8, 0x02, 0x00, 0x00, 0x00, 0xc3], // 27: mov eax, 2; ret
            &[0x31, 0xc0, 0xc3],      // 2d: default: xor eax, eax; ret
        ]
        .concat()
    }

    fn rva_table(rvas: impl IntoIterator<Item = u32>) -> Vec<u8> {
        rvas.into_iter().flat_map(u32::to_le_bytes).collect()
    }

    fn case_table(text_rva: u32) -> Vec<u8> {
        rva_table(CASES.map(|c| text_rva + c))
    }

    fn analyze(code: &[u8], data: &[u8], entries: &[u32], ranges: &[(u32, u32, u32)]) -> Analysis {
        let mut image = CodeImage::new(
            BASE,
            vec![
                CodeSection::new(TEXT, code, true),
                CodeSection::new(DATA, data, false),
            ],
            entries.iter().map(|&e| TEXT + u64::from(e)).collect(),
        );
        image.function_table = FunctionTable::new(
            ranges
                .iter()
                .map(|&(start, end, function)| FunctionRange {
                    start: TEXT + u64::from(start),
                    end: TEXT + u64::from(end),
                    function: TEXT + u64::from(function),
                })
                .collect(),
        );
        discover(&image)
    }

    /// Successors of the block of `function` that ends in the jump at `jmp`.
    fn jump_successors(function: &Function, jmp: u64) -> &[Edge] {
        let block = function
            .blocks
            .values()
            .find(|b| b.instructions.last().map(|i| i.ip()) == Some(jmp))
            .expect("block ending in the jump");
        assert_eq!(block.terminator, Terminator::IndirectJump);
        &block.successors
    }

    /// No bound was proven for the jump at `jmp`: it is unresolved or only inferred.
    fn assert_not_proven(analysis: &Analysis, jmp: u32) {
        if let Some(table) = analysis.jump_tables.get(&(TEXT + u64::from(jmp))) {
            assert!(table.inferred, "bound must not be proven: {table:#x?}");
        }
    }

    fn assert_unresolved(analysis: &Analysis, jmp: u32) {
        assert!(analysis.jump_tables.is_empty());
        assert_eq!(
            jump_successors(&analysis.functions[&TEXT], TEXT + u64::from(jmp)),
            [Edge::IndirectJump { slot: None }]
        );
    }

    #[test]
    fn resolves_rva_table_switch() {
        let code = switch_code(TEXT_RVA, DATA_RVA);
        let data = case_table(TEXT_RVA);
        let targets = CASES.map(|c| TEXT + u64::from(c));
        for ranges in [&[][..], &[(0, 0x30, 0)]] {
            let analysis = analyze(&code, &data, &[0], ranges);
            assert_eq!(
                analysis.jump_tables.values().collect::<Vec<_>>(),
                [&JumpTable {
                    site: TEXT + 0x19,
                    load_site: TEXT + 0xf,
                    table: DATA,
                    targets: targets.to_vec(),
                    index: None,
                    inferred: false,
                }]
            );
            assert_eq!(analysis.functions.len(), 1);
            let f = &analysis.functions[&TEXT];
            assert_eq!(
                jump_successors(f, TEXT + 0x19),
                targets.map(Edge::JumpTable)
            );
            assert_eq!(
                f.blocks.keys().map(|va| va - TEXT).collect::<Vec<_>>(),
                [0x0, 0x5, 0x1b, 0x21, 0x27, 0x2d]
            );
            let table_relocs: Vec<_> = analysis
                .relocations
                .entries()
                .iter()
                .filter(|r| matches!(r.kind, RelocKind::JumpTable | RelocKind::JumpTableEntry))
                .map(|r| (r.site, r.kind, r.target, r.region))
                .collect();
            assert_eq!(
                table_relocs,
                [
                    (TEXT + 0xf, RelocKind::JumpTable, DATA, Region::Data),
                    (DATA, RelocKind::JumpTableEntry, targets[0], Region::Code),
                    (
                        DATA + 4,
                        RelocKind::JumpTableEntry,
                        targets[1],
                        Region::Code
                    ),
                    (
                        DATA + 8,
                        RelocKind::JumpTableEntry,
                        targets[2],
                        Region::Code
                    ),
                    (
                        DATA + 12,
                        RelocKind::JumpTableEntry,
                        targets[3],
                        Region::Code
                    ),
                ]
            );
        }
    }

    #[test]
    fn resolves_two_level_byte_index_switch() {
        let code = [
            &[0x83, 0xf9, 0x07][..],           // 0: cmp ecx, 7
            &[0x77, 0x2a],                     // 3: ja 0x2f
            &[0x48, 0x63, 0xc1],               // 5: movsxd rax, ecx
            &lea_rdx_image_base(TEXT_RVA + 8), // 8: lea rdx, [__ImageBase]
            &[0x0f, 0xb6, 0x84, 0x02],
            &(DATA_RVA + 0x20).to_le_bytes(), // f: movzx eax, byte [rdx+rax+index_rva]
            &[0x8b, 0x8c, 0x82],
            &DATA_RVA.to_le_bytes(), // 17: mov ecx, [rdx+rax*4+table_rva]
            &[0x48, 0x01, 0xd1],     // 1e: add rcx, rdx
            &[0xff, 0xe1],           // 21: jmp rcx
            &[0xb8, 0x0a, 0x00, 0x00, 0x00, 0xc3], // 23: mov eax, 10; ret
            &[0xb8, 0x0b, 0x00, 0x00, 0x00, 0xc3], // 29: mov eax, 11; ret
            &[0x31, 0xc0, 0xc3],     // 2f: default
        ]
        .concat();
        let mut data = rva_table([0x1023, 0x1029, 0x102f]);
        data.resize(0x20, 0xcc); // bytes past the proven 3 entries are never read
        data.extend_from_slice(&[0, 1, 0, 2, 2, 1, 0, 2]);
        data.extend_from_slice(&[0xff; 8]); // index bytes past the bound are never read

        let analysis = analyze(&code, &data, &[0], &[(0, 0x32, 0)]);
        let targets = [0x23, 0x29, 0x2f].map(|c| TEXT + c);
        assert_eq!(
            analysis.jump_tables[&(TEXT + 0x21)],
            JumpTable {
                site: TEXT + 0x21,
                load_site: TEXT + 0x17,
                table: DATA,
                targets: targets.to_vec(),
                index: Some(JumpTableIndex {
                    load_site: TEXT + 0xf,
                    table: DATA + 0x20,
                    len: 8,
                }),
                inferred: false,
            }
        );
        assert_eq!(
            jump_successors(&analysis.functions[&TEXT], TEXT + 0x21),
            targets.map(Edge::JumpTable)
        );
        assert_eq!(
            analysis.relocations.at(TEXT + 0xf),
            &[Relocation {
                site: TEXT + 0xf,
                kind: RelocKind::JumpTableIndex,
                target: DATA + 0x20,
                region: Region::Data,
            }]
        );
    }

    #[test]
    fn rejects_targets_outside_the_function() {
        let code = switch_code(TEXT_RVA, DATA_RVA);
        let data = case_table(TEXT_RVA);
        // The default case belongs to another .pdata function.
        let analysis = analyze(&code, &data, &[0], &[(0, 0x2d, 0), (0x2d, 0x30, 0x2d)]);
        assert_unresolved(&analysis, 0x19);

        // An entry pointing into a data section.
        let mut data = case_table(TEXT_RVA);
        data[4..8].copy_from_slice(&(DATA_RVA + 0x10).to_le_bytes());
        data.resize(0x20, 0);
        assert_unresolved(&analyze(&code, &data, &[0], &[]), 0x19);

        // Table past the end of its section's bytes.
        assert_unresolved(&analyze(&code, &data[..12], &[0], &[]), 0x19);
    }

    #[test]
    fn unprovable_bounds_stay_unresolved() {
        let data = case_table(TEXT_RVA);
        let base_code = switch_code(TEXT_RVA, DATA_RVA);

        // No guard at all.
        let mut code = base_code.clone();
        code[0..5].copy_from_slice(&[0x0f, 0x1f, 0x44, 0x00, 0x00]); // 5-byte nop
        assert_unresolved(&analyze(&code, &data, &[0], &[]), 0x19);

        // The branch bounds the wrong side (`jb default`).
        let mut code = base_code.clone();
        code[3] = 0x72;
        assert_unresolved(&analyze(&code, &data, &[0], &[]), 0x19);

        // 32-bit compare, but the full 64-bit register indexes the table (upper half unknown).
        let mut code = base_code.clone();
        code[5..8].copy_from_slice(&[0x0f, 0x1f, 0x00]); // 3-byte nop instead of movsxd
        code[0x11] = 0x8a; // mov ecx, [rdx+rcx*4+table]
        assert_unresolved(&analyze(&code, &data, &[0], &[]), 0x19);

        // The index comes from a register the guard did not check.
        let mut code = base_code.clone();
        code[5..8].copy_from_slice(&[0x48, 0x63, 0xc2]); // movsxd rax, edx
        assert_unresolved(&analyze(&code, &data, &[0], &[]), 0x19);

        // The table base is not __ImageBase.
        let mut code = base_code;
        code[0x0b..0x0f].copy_from_slice(&0i32.to_le_bytes()); // lea rdx, [rip]
        assert_unresolved(&analyze(&code, &data, &[0], &[]), 0x19);
    }

    #[test]
    fn resolution_is_demoted_when_a_case_jumps_back_into_the_dispatch() {
        // Case 1 decrements the index and re-enters the dispatch without a guard, so the bound
        // proven in the first round no longer holds once the case code is decoded.
        let mut code = switch_code(TEXT_RVA, DATA_RVA);
        code[0x21..0x27].copy_from_slice(&[0xff, 0xc9, 0xeb, 0xe0, 0x90, 0x90]); // dec ecx; jmp 0x5
        let analysis = analyze(&code, &case_table(TEXT_RVA), &[0], &[]);
        assert_unresolved(&analysis, 0x19);
        assert!(
            analysis
                .relocations
                .entries()
                .iter()
                .all(|r| !matches!(r.kind, RelocKind::JumpTable | RelocKind::JumpTableEntry))
        );
    }

    #[test]
    fn nested_switch_found_through_a_case_and_order_independent() {
        // The outer switch's case 0 is a second switch placed after the outer one; it is only
        // reachable through the outer table.
        let outer = switch_code(TEXT_RVA, DATA_RVA);
        let inner = switch_code(TEXT_RVA + 0x30, DATA_RVA + 0x10);
        let code = [outer, inner].concat();
        let mut data = rva_table([0x1030, 0x1021, 0x1027, 0x102d]);
        data.extend(case_table(TEXT_RVA + 0x30));

        let forward = analyze(&code, &data, &[0], &[(0, 0x60, 0)]);
        assert_eq!(
            forward.jump_tables.keys().copied().collect::<Vec<_>>(),
            [TEXT + 0x19, TEXT + 0x49]
        );
        assert_eq!(
            forward.jump_tables[&(TEXT + 0x49)].targets,
            CASES.map(|c| TEXT + 0x30 + u64::from(c))
        );
        let again = analyze(&code, &data, &[0], &[(0, 0x60, 0)]);
        assert_eq!(forward, again);
        // Seeding the inner switch directly resolves the same tables.
        let with_inner_entry = analyze(&code, &data, &[0x30, 0], &[(0, 0x60, 0)]);
        assert_eq!(forward.jump_tables, with_inner_entry.jump_tables);
        assert_eq!(forward.relocations, with_inner_entry.relocations);
    }

    #[test]
    fn pdata_only_function_with_switch_in_synthetic_pe() {
        use crate::pdata::ExceptionDirectory;
        use crate::pdata::tests::{pe_with_pdata, put, unwind_bytes};
        use crate::pe::Pe64;
        use crate::summary::render_summary_json;

        let mut file = pe_with_pdata(&[(0x1100, 0x1130, 0x2700)]);
        put(&mut file, 0x2700, &unwind_bytes(0, &[]));
        put(&mut file, 0x1100, &switch_code(0x1100, 0x2600));
        put(&mut file, 0x2600, &case_table(0x1100));
        let pe = Pe64::parse(&file).expect("synthetic PE parses");
        let mut image = CodeImage::from_pe(&pe, &file);
        assert!(discover(&image).function(BASE + 0x1100).is_none());

        let pdata = ExceptionDirectory::from_pe(&pe, &file);
        assert!(pdata.diagnostics.is_empty());
        pdata.seed(&mut image);
        let analysis = discover(&image);
        let f = analysis
            .function(BASE + 0x1100)
            .expect("found through .pdata only");
        assert_eq!(
            jump_successors(f, BASE + 0x1119),
            CASES.map(|c| Edge::JumpTable(BASE + 0x1100 + u64::from(c)))
        );
        let json = render_summary_json(&image, &analysis);
        assert!(json.contains("\"pdata\": {\"functions\": 1, \"ranges\": 1},"));
        assert!(json.contains("\"jump_tables\": {\"resolved\": 1, \"unresolved\": 0},"));
        assert!(json.contains("\"jump_table_entry\": 4"));
    }

    /// Synthetic PE with `.pdata`: four switch functions (0x1100, 0x1140, 0x1180, 0x11c0) and a
    /// caller at 0x1080 that calls two of them and tail-jumps into a third.
    fn multi_switch_image(file: &[u8]) -> CodeImage<'_> {
        use crate::pdata::ExceptionDirectory;
        use crate::pe::Pe64;

        let pe = Pe64::parse(file).expect("synthetic PE parses");
        let mut image = CodeImage::from_pe(&pe, file);
        ExceptionDirectory::from_pe(&pe, file).seed(&mut image);
        image
    }

    fn multi_switch_file() -> Vec<u8> {
        use crate::pdata::tests::{pe_with_pdata, put, unwind_bytes};

        const SWITCHES: [u32; 4] = [0x1100, 0x1140, 0x1180, 0x11c0];
        let mut records = vec![(0x1080, 0x1090, 0x2700)];
        records.extend(SWITCHES.map(|rva| (rva, rva + 0x30, 0x2700)));
        let mut file = pe_with_pdata(&records);
        put(&mut file, 0x2700, &unwind_bytes(0, &[]));
        for (i, rva) in SWITCHES.into_iter().enumerate() {
            let table = 0x2600 + 0x10 * i as u32;
            put(&mut file, rva, &switch_code(rva, table));
            put(&mut file, table, &case_table(rva));
        }
        let rel = |from: u32, to: u32| (to.wrapping_sub(from + 5)).to_le_bytes();
        let caller = [
            &[0xe8][..],
            &rel(0x1080, 0x1100), // 0: call 0x1100
            &[0xe8],
            &rel(0x1085, 0x1140), // 5: call 0x1140
            &[0xe9],
            &rel(0x108a, 0x1180), // a: jmp 0x1180 (tail call)
        ]
        .concat();
        put(&mut file, 0x1080, &caller);
        file
    }

    #[test]
    fn parallel_discovery_matches_serial_on_synthetic_pe() {
        use crate::discover::discover_with_threads;

        let file = multi_switch_file();
        let image = multi_switch_image(&file);
        let serial = discover_with_threads(&image, 1);
        assert_eq!(serial.jump_tables.len(), 4);
        assert!(serial.function(BASE + 0x1080).is_some());
        let parallel = discover_with_threads(&image, 4);
        assert_eq!(serial, parallel);
        assert_eq!(
            serial.functions.keys().collect::<Vec<_>>(),
            parallel.functions.keys().collect::<Vec<_>>()
        );
    }

    #[test]
    fn discovery_is_deterministic_across_thread_counts() {
        use crate::discover::discover_with_threads;

        let file = multi_switch_file();
        let image = multi_switch_image(&file);
        let reference = discover_with_threads(&image, 1);
        for threads in [0, 2, 3, 5, 8, 64] {
            for _ in 0..3 {
                assert_eq!(
                    reference,
                    discover_with_threads(&image, threads),
                    "{threads} threads"
                );
            }
        }
    }

    // ---- Extended MSVC x64 jump-table pattern tests ----

    use iced_x86::code_asm::*;
    use iced_x86::{Code, Decoder, DecoderOptions, Instruction, MemoryOperand, Register};

    /// Emit `lea dst, [rip + (va - next_ip)]` in `CodeAssembler`.
    fn lea_rip(a: &mut CodeAssembler, dst: Register, va: u64) -> Result<(), iced_x86::IcedError> {
        let instr = Instruction::with2(
            Code::Lea_r64_m,
            dst,
            MemoryOperand::with_base_displ(Register::RIP, va as i64),
        )?;
        a.add_instruction(instr)
    }

    /// Assemble `build` at `TEXT` and return the encoded bytes.
    fn assemble_at_text(
        build: impl FnOnce(&mut CodeAssembler) -> Result<(), iced_x86::IcedError>,
    ) -> Vec<u8> {
        let mut a = CodeAssembler::new(64).expect("64-bit assembler");
        build(&mut a).expect("assemble");
        a.assemble(TEXT).expect("encode at TEXT")
    }

    /// Find the VA of the first indirect `jmp` in `code` (mapped at `TEXT`).
    fn find_indirect_jmp(code: &[u8]) -> u64 {
        let mut dec = Decoder::with_ip(64, code, TEXT, DecoderOptions::NONE);
        while dec.can_decode() {
            let instr = dec.decode();
            if instr.flow_control() == iced_x86::FlowControl::IndirectBranch {
                return instr.ip();
            }
        }
        panic!("no indirect jmp found");
    }

    /// Append 4 standard 6-byte case stubs (`mov eax, 0..=2; ret` and `xor eax, eax; ret`) to `a`
    /// and return their offsets after assembly (caller extracts exact IPs via `find_case_ips`).
    fn emit_four_cases(
        a: &mut CodeAssembler,
        c0: &mut CodeLabel,
        c1: &mut CodeLabel,
        c2: &mut CodeLabel,
        def: &mut CodeLabel,
    ) -> Result<(), iced_x86::IcedError> {
        a.set_label(c0)?;
        a.mov(eax, 0x10)?;
        a.ret()?;
        a.set_label(c1)?;
        a.mov(eax, 0x11)?;
        a.ret()?;
        a.set_label(c2)?;
        a.mov(eax, 0x12)?;
        a.ret()?;
        a.set_label(def)?;
        a.xor(eax, eax)?;
        a.ret()?;
        Ok(())
    }

    /// Locate the 4 case VAs emitted by `emit_four_cases` (trailing `mov eax, 0x10/0x11/0x12` and
    /// `xor eax, eax`).
    fn find_four_cases(code: &[u8]) -> [u64; 4] {
        let mut dec = Decoder::with_ip(64, code, TEXT, DecoderOptions::NONE);
        let mut found = [0u64; 4];
        while dec.can_decode() {
            let instr = dec.decode();
            if instr.mnemonic() == iced_x86::Mnemonic::Mov
                && instr.op0_register() == iced_x86::Register::EAX
                && let Ok(imm) = instr.try_immediate(1)
            {
                match imm {
                    0x10 => found[0] = instr.ip(),
                    0x11 => found[1] = instr.ip(),
                    0x12 => found[2] = instr.ip(),
                    _ => {}
                }
            } else if instr.mnemonic() == iced_x86::Mnemonic::Xor
                && instr.op0_register() == iced_x86::Register::EAX
                && instr.op1_register() == iced_x86::Register::EAX
            {
                found[3] = instr.ip();
            }
        }
        assert!(
            found.iter().all(|&va| va != 0),
            "missing case stub: {found:#x?}"
        );
        found
    }

    #[test]
    fn resolves_rva_table_with_predecessor_lea_copy_and_movsxd() {
        // `lea r12, [__ImageBase]` in predecessor block, copied to `rdx`, loaded via `movsxd`,
        // added back to `r12`.
        let code = assemble_at_text(|a| {
            let (mut c0, mut c1, mut c2, mut def) = (
                a.create_label(),
                a.create_label(),
                a.create_label(),
                a.create_label(),
            );
            lea_rip(a, Register::R12, BASE)?;
            a.cmp(ecx, 3)?;
            a.ja(def)?;
            a.mov(rdx, r12)?;
            a.movsxd(rax, ecx)?;
            a.movsxd(rcx, dword_ptr(rdx + rax * 4 + DATA_RVA as i32))?;
            a.add(rcx, r12)?;
            a.jmp(rcx)?;
            emit_four_cases(a, &mut c0, &mut c1, &mut c2, &mut def)
        });
        let cases = find_four_cases(&code);
        let jmp_va = find_indirect_jmp(&code);
        let data = rva_table(cases.map(|va| (va - BASE) as u32));
        let analysis = analyze(&code, &data, &[0], &[(0, code.len() as u32, 0)]);

        let jt = analysis
            .jump_tables
            .get(&jmp_va)
            .expect("predecessor lea + copy + movsxd resolved");
        assert_eq!(jt.table, DATA);
        assert_eq!(jt.targets, cases);
        assert_eq!(
            jump_successors(&analysis.functions[&TEXT], jmp_va),
            cases.map(Edge::JumpTable)
        );
    }

    /// Switch whose index is bounded in a stack slot and reloaded from it:
    /// `cmp dword [rsp+0x30], 2; ja default; <between>; mov eax, [rsp+0x30]; ...; jmp rcx`.
    fn stack_slot_switch(
        between: impl FnOnce(&mut CodeAssembler) -> Result<(), iced_x86::IcedError>,
    ) -> (Vec<u8>, [u64; 4], u64) {
        let code = assemble_at_text(|a| {
            let (mut c0, mut c1, mut c2, mut def) = (
                a.create_label(),
                a.create_label(),
                a.create_label(),
                a.create_label(),
            );
            a.mov(dword_ptr(rsp + 0x30), ecx)?;
            a.cmp(dword_ptr(rsp + 0x30), 2)?;
            a.ja(def)?;
            between(a)?;
            a.mov(eax, dword_ptr(rsp + 0x30))?;
            lea_rip(a, Register::RDX, BASE)?;
            a.movsxd(rcx, dword_ptr(rdx + rax * 4 + DATA_RVA as i32))?;
            a.add(rcx, rdx)?;
            a.jmp(rcx)?;
            emit_four_cases(a, &mut c0, &mut c1, &mut c2, &mut def)
        });
        let cases = find_four_cases(&code);
        let jmp_va = find_indirect_jmp(&code);
        (code, cases, jmp_va)
    }

    #[test]
    fn resolves_switch_with_index_bounded_in_a_stack_slot() {
        let (code, cases, jmp_va) = stack_slot_switch(|a| a.test(r8d, r8d));
        let data = rva_table(cases[..3].iter().map(|va| (va - BASE) as u32));
        let analysis = analyze(&code, &data, &[0], &[(0, code.len() as u32, 0)]);
        let jt = analysis
            .jump_tables
            .get(&jmp_va)
            .expect("stack-slot bound resolved");
        assert_eq!(jt.table, DATA);
        assert_eq!(jt.targets, cases[..3]);
    }

    #[test]
    fn stack_slot_bound_is_dropped_after_a_store_or_base_change() {
        let stores: [fn(&mut CodeAssembler) -> Result<(), iced_x86::IcedError>; 3] = [
            |a| a.mov(dword_ptr(rsp + 0x30), r8d),
            |a| a.push(rbx),
            |a| a.call(0x1_4000_9000u64),
        ];
        for store in stores {
            let (code, cases, jmp_va) = stack_slot_switch(store);
            let data = rva_table(cases[..3].iter().map(|va| (va - BASE) as u32));
            let analysis = analyze(&code, &data, &[0], &[(0, code.len() as u32, 0)]);
            assert_not_proven(&analysis, (jmp_va - TEXT) as u32);
        }
    }

    /// `movzx eax, dl; cmp al, 2; <jcc> default; ...; jmp rcx` over a three-entry table.
    fn byte_guard_switch(signed: bool) -> (Vec<u8>, [u64; 4], u64) {
        let code = assemble_at_text(|a| {
            let (mut c0, mut c1, mut c2, mut def) = (
                a.create_label(),
                a.create_label(),
                a.create_label(),
                a.create_label(),
            );
            a.movzx(eax, dl)?;
            a.cmp(al, 2)?;
            if signed {
                a.jg(def)?;
            } else {
                a.ja(def)?;
            }
            lea_rip(a, Register::RDX, BASE)?;
            a.movsxd(rcx, dword_ptr(rdx + rax * 4 + DATA_RVA as i32))?;
            a.add(rcx, rdx)?;
            a.jmp(rcx)?;
            emit_four_cases(a, &mut c0, &mut c1, &mut c2, &mut def)
        });
        let cases = find_four_cases(&code);
        let jmp_va = find_indirect_jmp(&code);
        (code, cases, jmp_va)
    }

    #[test]
    fn byte_guard_bounds_a_zero_extended_index() {
        let (code, cases, jmp_va) = byte_guard_switch(false);
        let mut data = rva_table(cases[..3].iter().map(|va| (va - BASE) as u32));
        data.extend_from_slice(&rva_table([DATA_RVA; 4])); // past the bound: never read
        let analysis = analyze(&code, &data, &[0], &[]);
        let table = &analysis.jump_tables[&jmp_va];
        assert!(!table.inferred);
        assert_eq!(table.targets, cases[..3]);

        // A signed byte guard does not bound the zero-extended value (0x80..0xff pass `jg`).
        let (code, cases, jmp_va) = byte_guard_switch(true);
        let mut data = rva_table(cases[..3].iter().map(|va| (va - BASE) as u32));
        data.extend_from_slice(&rva_table([DATA_RVA; 4]));
        assert_unresolved(&analyze(&code, &data, &[0], &[]), (jmp_va - TEXT) as u32);
    }

    /// `movzx eax, dx; shr eax, 4; cmp ax, 2; ja default` (a word guard on a shifted, zero-extended
    /// index); with `zero_extended` false the index is `mov eax, edx` (bits 16..31 unknown).
    fn shifted_word_guard_switch(zero_extended: bool) -> (Vec<u8>, [u64; 4], u64) {
        let code = assemble_at_text(|a| {
            let (mut c0, mut c1, mut c2, mut def) = (
                a.create_label(),
                a.create_label(),
                a.create_label(),
                a.create_label(),
            );
            if zero_extended {
                a.movzx(eax, dx)?;
            } else {
                a.mov(eax, edx)?;
            }
            a.shr(eax, 4)?;
            a.cmp(ax, 2)?;
            a.ja(def)?;
            lea_rip(a, Register::RDX, BASE)?;
            a.movsxd(rcx, dword_ptr(rdx + rax * 4 + DATA_RVA as i32))?;
            a.add(rcx, rdx)?;
            a.jmp(rcx)?;
            emit_four_cases(a, &mut c0, &mut c1, &mut c2, &mut def)
        });
        let cases = find_four_cases(&code);
        let jmp_va = find_indirect_jmp(&code);
        (code, cases, jmp_va)
    }

    #[test]
    fn word_guard_bounds_a_shifted_zero_extended_index() {
        let (code, cases, jmp_va) = shifted_word_guard_switch(true);
        let mut data = rva_table(cases[..3].iter().map(|va| (va - BASE) as u32));
        data.extend_from_slice(&rva_table([DATA_RVA; 4])); // past the bound: never read
        let analysis = analyze(&code, &data, &[0], &[]);
        let table = &analysis.jump_tables[&jmp_va];
        assert!(!table.inferred);
        assert_eq!(table.targets, cases[..3]);

        // Without the zero extension bits 16..31 of the shifted value are unknown: no bound.
        let (code, cases, jmp_va) = shifted_word_guard_switch(false);
        let mut data = rva_table(cases[..3].iter().map(|va| (va - BASE) as u32));
        data.extend_from_slice(&rva_table([DATA_RVA; 4]));
        assert_unresolved(&analyze(&code, &data, &[0], &[]), (jmp_va - TEXT) as u32);
    }

    /// `jmp` through a table whose index is a remainder left by a loop (no guard bounds it).
    fn loop_remainder_switch() -> (Vec<u8>, [u64; 4], u64) {
        let code = assemble_at_text(|a| {
            let (mut c0, mut c1, mut c2, mut def, mut top, mut tail) = (
                a.create_label(),
                a.create_label(),
                a.create_label(),
                a.create_label(),
                a.create_label(),
                a.create_label(),
            );
            a.set_label(&mut top)?;
            a.cmp(ecx, 3)?;
            a.jb(tail)?;
            a.sub(ecx, 3)?;
            a.jmp(top)?;
            a.set_label(&mut tail)?;
            lea_rip(a, Register::RDX, BASE)?;
            a.mov(eax, dword_ptr(rdx + rcx * 4 + DATA_RVA as i32))?;
            a.add(rax, rdx)?;
            a.jmp(rax)?;
            emit_four_cases(a, &mut c0, &mut c1, &mut c2, &mut def)
        });
        let cases = find_four_cases(&code);
        let jmp_va = find_indirect_jmp(&code);
        (code, cases, jmp_va)
    }

    /// `jmp` through a table indexed by a zero-extended byte whose value nothing on the slice
    /// bounds further (here a live-in register): the only bound is the byte's width.
    fn zero_extended_byte_switch() -> (Vec<u8>, [u64; 4], u64) {
        let code = assemble_at_text(|a| {
            let (mut c0, mut c1, mut c2, mut def) = (
                a.create_label(),
                a.create_label(),
                a.create_label(),
                a.create_label(),
            );
            a.movzx(ecx, r10b)?;
            lea_rip(a, Register::RDX, BASE)?;
            a.mov(eax, dword_ptr(rdx + rcx * 4 + DATA_RVA as i32))?;
            a.add(rax, rdx)?;
            a.jmp(rax)?;
            emit_four_cases(a, &mut c0, &mut c1, &mut c2, &mut def)
        });
        let cases = find_four_cases(&code);
        let jmp_va = find_indirect_jmp(&code);
        (code, cases, jmp_va)
    }

    #[test]
    fn infers_the_extent_within_a_zero_extended_byte_bound() {
        let (code, cases, jmp_va) = zero_extended_byte_switch();
        // Three cases inside the function, then an entry that leaves it: the byte range (256
        // entries) is wider than the table, so the extent is inferred instead of rejected.
        let mut data = rva_table(cases[..3].iter().map(|va| (va - BASE) as u32));
        data.extend_from_slice(&rva_table([DATA_RVA, 0]));
        let analysis = analyze(&code, &data, &[0], &[(0, code.len() as u32, 0)]);
        let table = &analysis.jump_tables[&jmp_va];
        assert!(table.inferred);
        assert_eq!(table.targets, cases[..3]);

        // A table that fills the whole byte range with valid cases keeps its proven bound.
        let mut data = rva_table(std::iter::repeat_n((cases[0] - BASE) as u32, 256));
        data.extend_from_slice(&rva_table([DATA_RVA]));
        let analysis = analyze(&code, &data, &[0], &[(0, code.len() as u32, 0)]);
        let table = &analysis.jump_tables[&jmp_va];
        assert!(!table.inferred);
        assert_eq!(table.targets.len(), 256);
    }

    #[test]
    fn inferred_extent_stops_at_an_entry_inside_an_instruction() {
        // A loop-state index with no provable bound, its table followed directly by the next
        // switch's table: read with the first table's base, the next entries land inside the
        // function but not on an instruction start. The extent ends there; that entry and every
        // later one (here a real case again) stay on the run-time dispatcher.
        let (code, cases, jmp_va) = loop_remainder_switch();
        let mut entries: Vec<u32> = cases[..3].iter().map(|va| (va - BASE) as u32).collect();
        entries.push((cases[0] + 1 - BASE) as u32);
        entries.push((cases[3] - BASE) as u32);
        let data = rva_table(entries);
        let analysis = analyze(&code, &data, &[0], &[(0, code.len() as u32, 0)]);
        let table = &analysis.jump_tables[&jmp_va];
        assert!(table.inferred);
        assert_eq!(table.targets, cases[..3]);
        assert_eq!(
            jump_successors(&analysis.functions[&TEXT], jmp_va),
            cases[..3]
                .iter()
                .map(|&c| Edge::JumpTable(c))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn infers_the_extent_of_a_table_without_a_provable_bound() {
        let (code, cases, jmp_va) = loop_remainder_switch();
        // Three cases inside the function, then an entry that leaves it (the next table, data).
        let mut data = rva_table(cases[..3].iter().map(|va| (va - BASE) as u32));
        data.extend_from_slice(&rva_table([DATA_RVA, 0]));
        let analysis = analyze(&code, &data, &[0], &[(0, code.len() as u32, 0)]);
        let table = &analysis.jump_tables[&jmp_va];
        assert!(table.inferred);
        assert_eq!(table.table, DATA);
        assert_eq!(table.targets, cases[..3]);
        assert_eq!(
            jump_successors(&analysis.functions[&TEXT], jmp_va),
            cases[..3]
                .iter()
                .map(|&c| Edge::JumpTable(c))
                .collect::<Vec<_>>()
        );

        // Without a `.pdata` function the extent is not inferred.
        let analysis = analyze(&code, &data, &[0], &[]);
        assert_unresolved(&analysis, (jmp_va - TEXT) as u32);

        // A leaf function (no `.pdata` record) followed by a `.pdata` function `next`: the
        // extent is [leaf entry, next entry) inside the `.pdata` gap.
        let end = code.len() as u32;
        let next = end + 0x10;
        let mut padded = code.clone();
        padded.resize(next as usize, 0xCC);
        padded.extend_from_slice(&[0xC3; 0x10]);
        let neighbours = [(0x0, 0x1, 0x0), (next, next + 0x10, next)];
        let analysis = analyze(&padded, &data, &[0, next], &neighbours);
        let table = &analysis.jump_tables[&jmp_va];
        assert!(table.inferred);
        assert_eq!(table.targets, cases[..3]);

        // `next` another leaf in the same gap: an entry at or past it ends the table.
        let mut two_leaves = padded.clone();
        two_leaves.extend_from_slice(&[0xC3; 0x10]);
        let gap_end = next + 0x10;
        let leaf_neighbours = [(0x0, 0x1, 0x0), (gap_end, gap_end + 0x10, gap_end)];
        let mut into_next = rva_table(cases[..2].iter().map(|va| (va - BASE) as u32));
        into_next.extend_from_slice(&rva_table([(TEXT - BASE) as u32 + next, 0]));
        let analysis = analyze(&two_leaves, &into_next, &[0, next], &leaf_neighbours);
        assert_eq!(analysis.jump_tables[&jmp_va].targets, cases[..2]);

        // The gap must be bounded on both sides, and the function by a next entry.
        let analysis = analyze(&padded, &data, &[0, next], &[(next, next + 0x10, next)]);
        assert_unresolved(&analysis, (jmp_va - TEXT) as u32);
        let analysis = analyze(&padded, &data, &[0], &neighbours);
        assert_unresolved(&analysis, (jmp_va - TEXT) as u32);

        // A single in-function entry is not a table.
        let mut short = rva_table([(cases[0] - BASE) as u32]);
        short.extend_from_slice(&rva_table([DATA_RVA, 0]));
        let analysis = analyze(&code, &short, &[0], &[(0, code.len() as u32, 0)]);
        assert!(analysis.jump_tables.is_empty());
    }

    /// Threaded bytecode dispatch: `jmp qword ptr [r12 + rbx*8]` with the index a scrambled
    /// 16-bit opcode (`imul` by an odd constant, then `movzx` of the low word). Case 0 fetches the
    /// next opcode and re-dispatches through the same table, with the base set before the merge.
    /// Returns the code, the four case VAs, both dispatch jumps and the VA of a separate function
    /// placed after the interpreter.
    fn threaded_dispatch() -> (Vec<u8>, [u64; 4], [u64; 2], u64) {
        let code = assemble_at_text(|a| {
            let (mut c0, mut c1, mut c2, mut def) = (
                a.create_label(),
                a.create_label(),
                a.create_label(),
                a.create_label(),
            );
            a.movzx(edi, word_ptr(rcx))?;
            a.mov(rsi, rcx)?;
            lea_rip(a, Register::R12, DATA)?;
            a.imul_3(eax, edi, 0x3315)?;
            a.movzx(ebx, ax)?;
            a.jmp(qword_ptr(r12 + rbx * 8))?;
            a.add(rsi, 2)?;
            a.movzx(eax, word_ptr(rsi))?;
            a.imul_3(eax, eax, 0x3315)?;
            a.movzx(ebx, ax)?;
            a.jmp(qword_ptr(r12 + rbx * 8))?;
            emit_four_cases(a, &mut c0, &mut c1, &mut c2, &mut def)?;
            a.mov(eax, 0x99)?;
            a.ret()
        });
        let mut dec = Decoder::with_ip(64, &code, TEXT, DecoderOptions::NONE);
        let mut jumps = Vec::new();
        let mut fetch = 0;
        let mut other = 0;
        while dec.can_decode() {
            let instr = dec.decode();
            if instr.flow_control() == iced_x86::FlowControl::IndirectBranch {
                jumps.push(instr.ip());
                if fetch == 0 {
                    fetch = instr.next_ip();
                }
            }
            if instr.mnemonic() == iced_x86::Mnemonic::Mov
                && instr.try_immediate(1).ok() == Some(0x99)
            {
                other = instr.ip();
            }
        }
        let four = find_four_cases(&code);
        (
            code,
            [fetch, four[0], four[1], four[2]],
            [jumps[0], jumps[1]],
            other,
        )
    }

    #[test]
    fn resolves_threaded_dispatch_through_a_pointer_table_with_decoy_entries() {
        let (code, cases, jumps, other) = threaded_dispatch();
        let interpreter_end = (other - TEXT) as u32;
        let ranges = [
            (0, interpreter_end, 0),
            (interpreter_end, code.len() as u32, interpreter_end),
        ];
        // The handlers, then entries that are no case of this function: another function's entry,
        // a value outside the image and a data address. They are left to run-time dispatch.
        let mut data: Vec<u8> = cases.iter().flat_map(|va| va.to_le_bytes()).collect();
        for decoy in [other, 0x1234_5678_9abc, DATA] {
            data.extend_from_slice(&decoy.to_le_bytes());
        }
        // An in-function address past the first decoy (`mov rsi, rcx`) is not read as a case.
        let past = TEXT + 3;
        data.extend_from_slice(&past.to_le_bytes());
        let analysis = analyze(&code, &data, &[0, interpreter_end], &ranges);
        let expected: Vec<Edge> = {
            let mut sorted = cases.to_vec();
            sorted.sort_unstable();
            sorted.into_iter().map(Edge::JumpTable).collect()
        };
        for jmp in jumps {
            let table = &analysis.jump_tables[&jmp];
            assert!(table.inferred, "the word width is the only bound");
            assert_eq!(table.load_site, jmp, "the jump reads the table itself");
            assert_eq!(table.table, DATA);
            assert_eq!(table.targets, cases);
            assert_eq!(jump_successors(&analysis.functions[&TEXT], jmp), expected);
        }
        let blocks = &analysis.functions[&TEXT].blocks;
        assert!(!blocks.contains_key(&other) && !blocks.contains_key(&past));
    }

    #[test]
    fn resolves_rva_table_with_separate_table_lea_and_lea_target() {
        // `lea rdx, [__ImageBase]; lea r8, [rdx + DATA_RVA]; mov eax, [r8 + rcx*4]; lea rax, [rdx + rax]; jmp rax`
        let code = assemble_at_text(|a| {
            let (mut c0, mut c1, mut c2, mut def) = (
                a.create_label(),
                a.create_label(),
                a.create_label(),
                a.create_label(),
            );
            a.cmp(ecx, 3)?;
            a.ja(def)?;
            a.mov(ecx, ecx)?; // zero-extend ecx -> rcx
            lea_rip(a, Register::RDX, BASE)?;
            a.lea(r8, ptr(rdx + DATA_RVA as i32))?;
            a.mov(eax, dword_ptr(r8 + rcx * 4))?;
            a.lea(rax, ptr(rdx + rax))?;
            a.jmp(rax)?;
            emit_four_cases(a, &mut c0, &mut c1, &mut c2, &mut def)
        });
        let cases = find_four_cases(&code);
        let jmp_va = find_indirect_jmp(&code);
        let data = rva_table(cases.map(|va| (va - BASE) as u32));
        let analysis = analyze(&code, &data, &[0], &[(0, code.len() as u32, 0)]);

        let jt = analysis
            .jump_tables
            .get(&jmp_va)
            .expect("separate lea rTab + lea target resolved");
        assert_eq!(jt.table, DATA);
        assert_eq!(jt.targets, cases);
    }

    #[test]
    fn resolves_rva_table_with_nonvolatile_base_across_call() {
        // `lea rbx, [__ImageBase]` preserved across `call helper` per Win64 ABI.
        let code = assemble_at_text(|a| {
            let (mut c0, mut c1, mut c2, mut def, mut helper) = (
                a.create_label(),
                a.create_label(),
                a.create_label(),
                a.create_label(),
                a.create_label(),
            );
            lea_rip(a, Register::RBX, BASE)?;
            a.call(helper)?;
            a.cmp(eax, 3)?;
            a.ja(def)?;
            a.movsxd(rcx, eax)?;
            a.mov(eax, dword_ptr(rbx + rcx * 4 + DATA_RVA as i32))?;
            a.add(rax, rbx)?;
            a.jmp(rax)?;
            emit_four_cases(a, &mut c0, &mut c1, &mut c2, &mut def)?;
            a.set_label(&mut helper)?;
            a.mov(eax, 1)?;
            a.ret()?;
            Ok(())
        });
        let cases = find_four_cases(&code);
        let jmp_va = find_indirect_jmp(&code);
        let fn_end = (cases[3] - TEXT + 3) as u32;
        let data = rva_table(cases.map(|va| (va - BASE) as u32));
        let analysis = analyze(
            &code,
            &data,
            &[0],
            &[(0, fn_end, 0), (fn_end, code.len() as u32, fn_end)],
        );

        let jt = analysis
            .jump_tables
            .get(&jmp_va)
            .expect("non-volatile base across call resolved");
        assert_eq!(jt.table, DATA);
        assert_eq!(jt.targets, cases);
    }

    /// A switch inside a loop whose table base was loaded before the loop:
    /// ```text
    ///   0: lea r14, [rip+DATA]         7: xor ebx, ebx
    ///   9: loop: cmp ebx, 3            c: ja done
    ///   e: mov eax, ebx               10: movsxd rax, dword [r14+rax*4]
    ///  14: add rax, r14               17: jmp rax
    ///  19..: four cases `<case_body>; jmp loop`, then done: ret
    /// ```
    /// The dispatch's predecessor chain stops at the loop header (entry + four back edges), so the
    /// `lea` is only visible through the definitions reaching the header.
    fn loop_switch(case3_body: &[u8], call_in_case0: bool) -> (Vec<u8>, [u64; 4], u64) {
        let mut code = Vec::new();
        let disp = (DATA as i64 - (TEXT as i64 + 7)) as i32;
        code.extend([0x4c, 0x8d, 0x35]);
        code.extend(disp.to_le_bytes());
        code.extend([0x31, 0xdb]); // xor ebx, ebx
        let loop_head = code.len();
        code.extend([0x83, 0xfb, 0x03, 0x77, 0x00]); // cmp ebx, 3; ja done (patched)
        let ja_at = code.len() - 1;
        code.extend([
            0x89, 0xd8, 0x49, 0x63, 0x04, 0x86, 0x4c, 0x01, 0xf0, 0xff, 0xe0,
        ]);
        let jmp_va = TEXT + code.len() as u64 - 2;
        let mut cases = [0u64; 4];
        let mut call_at = None;
        for (i, case) in cases.iter_mut().enumerate() {
            *case = TEXT + code.len() as u64;
            match i {
                0 if call_in_case0 => {
                    call_at = Some(code.len());
                    code.extend([0xe8, 0, 0, 0, 0, 0xff, 0xc3]); // call callee; inc ebx
                }
                3 => code.extend(case3_body),
                _ => code.extend([0xff, 0xc3]), // inc ebx
            }
            let rel = loop_head as i64 - (code.len() as i64 + 2);
            code.extend([0xeb, rel as i8 as u8]); // jmp loop
        }
        let done = code.len();
        code[ja_at] = (done - (ja_at + 1)) as u8;
        code.push(0xc3);
        if let Some(at) = call_at {
            // The callee: a separate function after `done`.
            let rel = (code.len() - (at + 5)) as i32;
            code[at + 1..at + 5].copy_from_slice(&rel.to_le_bytes());
            code.push(0xc3);
        }
        (code, cases, jmp_va)
    }

    fn signed_offsets(cases: &[u64]) -> Vec<u8> {
        cases
            .iter()
            .flat_map(|&va| ((va as i64 - DATA as i64) as i32).to_le_bytes())
            .collect()
    }

    #[test]
    fn resolves_switch_with_table_base_loaded_before_the_loop() {
        let (code, cases, jmp_va) = loop_switch(&[0xff, 0xc3], false);
        let analysis = analyze(
            &code,
            &signed_offsets(&cases),
            &[0],
            &[(0, code.len() as u32, 0)],
        );
        let jt = analysis
            .jump_tables
            .get(&jmp_va)
            .expect("loop switch resolved");
        assert_eq!(jt.table, DATA);
        assert_eq!(jt.targets, cases);
        assert_eq!(
            jump_successors(&analysis.functions[&TEXT], jmp_va),
            cases.map(Edge::JumpTable)
        );
    }

    #[test]
    fn loop_switch_stays_unresolved_when_a_path_changes_the_base() {
        // case 3: mov r14, rbx; inc ebx: the base differs on one back edge.
        let (code, cases, jmp_va) = loop_switch(&[0x49, 0x89, 0xde, 0xff, 0xc3], false);
        let analysis = analyze(
            &code,
            &signed_offsets(&cases),
            &[0],
            &[(0, code.len() as u32, 0)],
        );
        assert!(!analysis.jump_tables.contains_key(&jmp_va));
    }

    #[test]
    fn loop_switch_keeps_a_nonvolatile_base_across_calls() {
        // case 0 calls (a function entry at the next instruction); r14 survives the call.
        let (code, cases, jmp_va) = loop_switch(&[0xff, 0xc3], true);
        let analysis = analyze(
            &code,
            &signed_offsets(&cases),
            &[0],
            &[(0, code.len() as u32, 0)],
        );
        let jt = analysis
            .jump_tables
            .get(&jmp_va)
            .expect("resolved across the call");
        assert_eq!(jt.targets, cases);
    }

    #[test]
    fn resolves_table_relative_signed_offset_switch() {
        // `lea rdx, [rip + DATA]; movsxd rcx, dword ptr [rdx + rax*4]; add rcx, rdx; jmp rcx`
        // Note: DATA > TEXT, so all 32-bit offsets are NEGATIVE!
        let code = assemble_at_text(|a| {
            let (mut c0, mut c1, mut c2, mut def) = (
                a.create_label(),
                a.create_label(),
                a.create_label(),
                a.create_label(),
            );
            a.cmp(ecx, 3)?;
            a.ja(def)?;
            a.movsxd(rax, ecx)?;
            lea_rip(a, Register::RDX, DATA)?;
            a.movsxd(rcx, dword_ptr(rdx + rax * 4))?;
            a.add(rcx, rdx)?;
            a.jmp(rcx)?;
            emit_four_cases(a, &mut c0, &mut c1, &mut c2, &mut def)
        });
        let cases = find_four_cases(&code);
        let jmp_va = find_indirect_jmp(&code);
        let data: Vec<u8> = cases
            .iter()
            .flat_map(|&va| {
                let rel = (va as i64 - DATA as i64) as i32;
                assert!(rel < 0, "expected negative table-relative offset");
                rel.to_le_bytes()
            })
            .collect();
        let analysis = analyze(&code, &data, &[0], &[(0, code.len() as u32, 0)]);

        let jt = analysis
            .jump_tables
            .get(&jmp_va)
            .expect("table-relative signed offset switch resolved");
        assert_eq!(jt.table, DATA);
        assert_eq!(jt.targets, cases);
        assert_eq!(
            jump_successors(&analysis.functions[&TEXT], jmp_va),
            cases.map(Edge::JumpTable)
        );
    }

    #[test]
    fn resolves_arbitrary_anchor_relative_signed_table() {
        // `lea rdx, [rip + TEXT]` (section/function anchor, neither __ImageBase nor table_va),
        // `movsxd rax, dword ptr [rdx + rcx*4 + (DATA - TEXT)]`, `add rax, rdx`, `jmp rax`.
        let code = assemble_at_text(|a| {
            let (mut c0, mut c1, mut c2, mut def) = (
                a.create_label(),
                a.create_label(),
                a.create_label(),
                a.create_label(),
            );
            a.cmp(ecx, 3)?;
            a.ja(def)?;
            a.movsxd(rcx, ecx)?;
            lea_rip(a, Register::RDX, TEXT)?;
            a.movsxd(rax, dword_ptr(rdx + rcx * 4 + (DATA - TEXT) as i32))?;
            a.add(rax, rdx)?;
            a.jmp(rax)?;
            emit_four_cases(a, &mut c0, &mut c1, &mut c2, &mut def)
        });
        let cases = find_four_cases(&code);
        let jmp_va = find_indirect_jmp(&code);
        let data: Vec<u8> = cases
            .iter()
            .flat_map(|&va| ((va - TEXT) as i32).to_le_bytes())
            .collect();
        let analysis = analyze(&code, &data, &[0], &[(0, code.len() as u32, 0)]);

        let jt = analysis
            .jump_tables
            .get(&jmp_va)
            .expect("anchor-relative table resolved");
        assert_eq!(jt.table, DATA);
        assert_eq!(jt.targets, cases);
    }

    #[test]
    fn resolves_absolute_64bit_pointer_table_register_and_memory_jmp() {
        // Variant A: `lea rdx, [rip + DATA]; mov rax, qword ptr [rdx + rax*8]; jmp rax` (via discover)
        let code = assemble_at_text(|a| {
            let (mut c0, mut c1, mut c2, mut def) = (
                a.create_label(),
                a.create_label(),
                a.create_label(),
                a.create_label(),
            );
            a.cmp(ecx, 3)?;
            a.ja(def)?;
            a.movsxd(rax, ecx)?;
            lea_rip(a, Register::RDX, DATA)?;
            a.mov(rax, qword_ptr(rdx + rax * 8))?;
            a.jmp(rax)?;
            emit_four_cases(a, &mut c0, &mut c1, &mut c2, &mut def)
        });
        let cases = find_four_cases(&code);
        let jmp_va = find_indirect_jmp(&code);
        let data: Vec<u8> = cases.iter().flat_map(|&va| va.to_le_bytes()).collect();
        let analysis = analyze(&code, &data, &[0], &[(0, code.len() as u32, 0)]);

        let jt = analysis
            .jump_tables
            .get(&jmp_va)
            .expect("64-bit pointer table via register jmp resolved");
        assert_eq!(jt.table, DATA);
        assert_eq!(jt.targets, cases);

        // Variant B: direct `jmp qword ptr [rdx + rax*8]` resolved via `jump_table::resolve`.
        let code_mem = assemble_at_text(|a| {
            let (mut c0, mut c1, mut c2, mut def) = (
                a.create_label(),
                a.create_label(),
                a.create_label(),
                a.create_label(),
            );
            a.cmp(ecx, 3)?;
            a.ja(def)?;
            a.movsxd(rax, ecx)?;
            lea_rip(a, Register::RDX, DATA)?;
            a.jmp(qword_ptr(rdx + rax * 8))?;
            emit_four_cases(a, &mut c0, &mut c1, &mut c2, &mut def)
        });
        let cases_mem = find_four_cases(&code_mem);
        let data_mem: Vec<u8> = cases_mem.iter().flat_map(|&va| va.to_le_bytes()).collect();
        let mut image = CodeImage::new(
            BASE,
            vec![
                CodeSection::new(TEXT, &code_mem, true),
                CodeSection::new(DATA, &data_mem, false),
            ],
            vec![TEXT],
        );
        image.function_table = FunctionTable::new(vec![FunctionRange {
            start: TEXT,
            end: TEXT + code_mem.len() as u64,
            function: TEXT,
        }]);
        let mut dec = Decoder::with_ip(64, &code_mem, TEXT, DecoderOptions::NONE);
        let mut instrs = Vec::new();
        while dec.can_decode() {
            let i = dec.decode();
            let is_jmp = i.flow_control() == iced_x86::FlowControl::IndirectBranch;
            instrs.push(i);
            if is_jmp {
                break;
            }
        }
        let steps: Vec<super::Step<'_>> = instrs
            .iter()
            .map(|instr| super::Step {
                instr,
                taken: (instr.flow_control() == iced_x86::FlowControl::ConditionalBranch)
                    .then_some(false),
            })
            .collect();
        let jt_mem = super::resolve(&image, &steps).expect("direct jmp [rdx+rax*8] resolved");
        assert_eq!(jt_mem.site, instrs.last().unwrap().ip());
        assert_eq!(jt_mem.load_site, jt_mem.site);
        assert_eq!(jt_mem.table, DATA);
        assert_eq!(jt_mem.targets, cases_mem);
    }

    #[test]
    fn resolves_two_level_word_index_and_swapped_sib_byte_index() {
        // Variant A: 16-bit word index table `[rdx + rax*2 + idx_rva]` -> 32-bit RVA table.
        let code = assemble_at_text(|a| {
            let (mut c0, mut c1, mut c2, mut def) = (
                a.create_label(),
                a.create_label(),
                a.create_label(),
                a.create_label(),
            );
            a.cmp(ecx, 5)?;
            a.ja(def)?;
            a.movsxd(rax, ecx)?;
            lea_rip(a, Register::RDX, BASE)?;
            a.movzx(eax, word_ptr(rdx + rax * 2 + (DATA_RVA + 0x20) as i32))?;
            a.mov(ecx, dword_ptr(rdx + rax * 4 + DATA_RVA as i32))?;
            a.add(rcx, rdx)?;
            a.jmp(rcx)?;
            emit_four_cases(a, &mut c0, &mut c1, &mut c2, &mut def)
        });
        let cases = find_four_cases(&code);
        let jmp_va = find_indirect_jmp(&code);
        let mut data = rva_table(cases.map(|va| (va - BASE) as u32));
        data.resize(0x20, 0xcc);
        for idx in [0u16, 3, 1, 2, 3, 0] {
            data.extend_from_slice(&idx.to_le_bytes());
        }
        let analysis = analyze(&code, &data, &[0], &[(0, code.len() as u32, 0)]);
        let jt = analysis
            .jump_tables
            .get(&jmp_va)
            .expect("two-level word index table resolved");
        assert_eq!(jt.table, DATA);
        assert_eq!(jt.targets, cases);
        let idx = jt.index.expect("word JumpTableIndex present");
        assert_eq!(idx.table, DATA + 0x20);
        assert_eq!(idx.len, 12); // 6 u16 entries = 12 bytes

        // Variant B: Byte index table via `lea r8, [rip + idx_table]` with SWAPPED SIB `[rax + r8]`.
        let code_swapped = assemble_at_text(|a| {
            let (mut c0, mut c1, mut c2, mut def) = (
                a.create_label(),
                a.create_label(),
                a.create_label(),
                a.create_label(),
            );
            a.cmp(ecx, 5)?;
            a.ja(def)?;
            a.movsxd(rax, ecx)?;
            lea_rip(a, Register::R8, DATA + 0x20)?;
            lea_rip(a, Register::RDX, BASE)?;
            a.movzx(eax, byte_ptr(rax + r8))?; // swapped: index in base slot, table in index slot!
            a.mov(ecx, dword_ptr(rdx + rax * 4 + DATA_RVA as i32))?;
            a.add(rcx, rdx)?;
            a.jmp(rcx)?;
            emit_four_cases(a, &mut c0, &mut c1, &mut c2, &mut def)
        });
        let cases_s = find_four_cases(&code_swapped);
        let jmp_s = find_indirect_jmp(&code_swapped);
        let mut data_s = rva_table(cases_s.map(|va| (va - BASE) as u32));
        data_s.resize(0x20, 0xcc);
        data_s.extend_from_slice(&[0, 1, 2, 3, 1, 0]);
        let analysis_s = analyze(
            &code_swapped,
            &data_s,
            &[0],
            &[(0, code_swapped.len() as u32, 0)],
        );
        let jt_s = analysis_s
            .jump_tables
            .get(&jmp_s)
            .expect("swapped SIB byte index table resolved");
        assert_eq!(jt_s.targets, cases_s);
        assert_eq!(jt_s.index.unwrap().table, DATA + 0x20);
        assert_eq!(jt_s.index.unwrap().len, 6);
    }

    #[test]
    fn resolves_and_power_of_two_bounds() {
        // 32-bit `and ecx, 3` (no cmp/ja at all; also zero-extends rcx!).
        let code32 = assemble_at_text(|a| {
            let (mut c0, mut c1, mut c2, mut def) = (
                a.create_label(),
                a.create_label(),
                a.create_label(),
                a.create_label(),
            );
            a.and(ecx, 3)?;
            lea_rip(a, Register::RDX, BASE)?;
            a.mov(eax, dword_ptr(rdx + rcx * 4 + DATA_RVA as i32))?;
            a.add(rax, rdx)?;
            a.jmp(rax)?;
            emit_four_cases(a, &mut c0, &mut c1, &mut c2, &mut def)
        });
        let cases32 = find_four_cases(&code32);
        let jmp32 = find_indirect_jmp(&code32);
        let data32 = rva_table(cases32.map(|va| (va - BASE) as u32));
        let analysis32 = analyze(&code32, &data32, &[0], &[(0, code32.len() as u32, 0)]);
        assert_eq!(
            analysis32
                .jump_tables
                .get(&jmp32)
                .expect("32-bit `and` bound resolved")
                .targets,
            cases32
        );

        // 64-bit `and rcx, 3` (no cmp/ja at all).
        let code64 = assemble_at_text(|a| {
            let (mut c0, mut c1, mut c2, mut def) = (
                a.create_label(),
                a.create_label(),
                a.create_label(),
                a.create_label(),
            );
            a.and(rcx, 3)?;
            lea_rip(a, Register::RDX, BASE)?;
            a.mov(eax, dword_ptr(rdx + rcx * 4 + DATA_RVA as i32))?;
            a.add(rax, rdx)?;
            a.jmp(rax)?;
            emit_four_cases(a, &mut c0, &mut c1, &mut c2, &mut def)
        });
        let cases64 = find_four_cases(&code64);
        let jmp64 = find_indirect_jmp(&code64);
        let data64 = rva_table(cases64.map(|va| (va - BASE) as u32));
        let analysis64 = analyze(&code64, &data64, &[0], &[(0, code64.len() as u32, 0)]);
        assert_eq!(
            analysis64
                .jump_tables
                .get(&jmp64)
                .expect("64-bit `and` bound resolved")
                .targets,
            cases64
        );
    }

    #[test]
    fn resolves_sub_cmp_ja_and_signed_guards_in_predecessor_blocks() {
        // 1) `sub ecx, 10; cmp ecx, 3; ja default` WITHOUT `movsxd` (relies on 32-bit `sub` zero-extending `rcx`).
        let code_sub = assemble_at_text(|a| {
            let (mut c0, mut c1, mut c2, mut def) = (
                a.create_label(),
                a.create_label(),
                a.create_label(),
                a.create_label(),
            );
            a.sub(ecx, 10)?;
            a.cmp(ecx, 3)?;
            a.ja(def)?;
            lea_rip(a, Register::RDX, BASE)?;
            a.mov(eax, dword_ptr(rdx + rcx * 4 + DATA_RVA as i32))?;
            a.add(rax, rdx)?;
            a.jmp(rax)?;
            emit_four_cases(a, &mut c0, &mut c1, &mut c2, &mut def)
        });
        let cases_sub = find_four_cases(&code_sub);
        let jmp_sub = find_indirect_jmp(&code_sub);
        let data_sub = rva_table(cases_sub.map(|va| (va - BASE) as u32));
        let analysis_sub = analyze(&code_sub, &data_sub, &[0], &[(0, code_sub.len() as u32, 0)]);
        assert_eq!(
            analysis_sub
                .jump_tables
                .get(&jmp_sub)
                .expect("sub + cmp + ja without movsxd resolved")
                .targets,
            cases_sub
        );

        // 2) Range check BEFORE `lea eax, [rcx - 10]`: `cmp ecx, 10; jb def; cmp ecx, 13; ja def; lea eax, [rcx - 10]`.
        let code_range = assemble_at_text(|a| {
            let (mut c0, mut c1, mut c2, mut def) = (
                a.create_label(),
                a.create_label(),
                a.create_label(),
                a.create_label(),
            );
            a.cmp(ecx, 10)?;
            a.jb(def)?;
            a.cmp(ecx, 13)?;
            a.ja(def)?;
            a.lea(eax, ptr(rcx - 10))?;
            lea_rip(a, Register::RDX, BASE)?;
            a.mov(ecx, dword_ptr(rdx + rax * 4 + DATA_RVA as i32))?;
            a.add(rcx, rdx)?;
            a.jmp(rcx)?;
            emit_four_cases(a, &mut c0, &mut c1, &mut c2, &mut def)
        });
        let cases_range = find_four_cases(&code_range);
        let jmp_range = find_indirect_jmp(&code_range);
        let data_range = rva_table(cases_range.map(|va| (va - BASE) as u32));
        let analysis_range = analyze(
            &code_range,
            &data_range,
            &[0],
            &[(0, code_range.len() as u32, 0)],
        );
        assert_eq!(
            analysis_range
                .jump_tables
                .get(&jmp_range)
                .expect("jb + ja + lea [rcx-10] resolved")
                .targets,
            cases_range
        );

        // 3) Signed `test + js + cmp + jg` in predecessor blocks.
        let code_jg = assemble_at_text(|a| {
            let (mut c0, mut c1, mut c2, mut def) = (
                a.create_label(),
                a.create_label(),
                a.create_label(),
                a.create_label(),
            );
            a.test(ecx, ecx)?;
            a.js(def)?;
            a.cmp(ecx, 3)?;
            a.jg(def)?;
            a.movsxd(rax, ecx)?;
            lea_rip(a, Register::RDX, BASE)?;
            a.mov(ecx, dword_ptr(rdx + rax * 4 + DATA_RVA as i32))?;
            a.add(rcx, rdx)?;
            a.jmp(rcx)?;
            emit_four_cases(a, &mut c0, &mut c1, &mut c2, &mut def)
        });
        let cases_jg = find_four_cases(&code_jg);
        let jmp_jg = find_indirect_jmp(&code_jg);
        let data_jg = rva_table(cases_jg.map(|va| (va - BASE) as u32));
        let analysis_jg = analyze(&code_jg, &data_jg, &[0], &[(0, code_jg.len() as u32, 0)]);
        assert_eq!(
            analysis_jg
                .jump_tables
                .get(&jmp_jg)
                .expect("signed test+js+cmp+jg resolved")
                .targets,
            cases_jg
        );

        // 4) Signed `cmp ecx, 4; jge default` with `movsxd` scheduled BEFORE `cmp` + `jge`.
        let code_jge = assemble_at_text(|a| {
            let (mut c0, mut c1, mut c2, mut def) = (
                a.create_label(),
                a.create_label(),
                a.create_label(),
                a.create_label(),
            );
            a.movsxd(rax, ecx)?;
            a.cmp(ecx, 4)?;
            a.jge(def)?;
            lea_rip(a, Register::RDX, BASE)?;
            a.mov(ecx, dword_ptr(rdx + rax * 4 + DATA_RVA as i32))?;
            a.add(rcx, rdx)?;
            a.jmp(rcx)?;
            emit_four_cases(a, &mut c0, &mut c1, &mut c2, &mut def)
        });
        let cases_jge = find_four_cases(&code_jge);
        let jmp_jge = find_indirect_jmp(&code_jge);
        let data_jge = rva_table(cases_jge.map(|va| (va - BASE) as u32));
        let analysis_jge = analyze(&code_jge, &data_jge, &[0], &[(0, code_jge.len() as u32, 0)]);
        assert_eq!(
            analysis_jge
                .jump_tables
                .get(&jmp_jge)
                .expect("movsxd before cmp+jge resolved")
                .targets,
            cases_jge
        );

        // 5) Pre-scaled index (`shl rax, 2`) + pre-computed element pointer (`lea rcx, [rdx + rax + table_rva]`).
        let code_prescaled = assemble_at_text(|a| {
            let (mut c0, mut c1, mut c2, mut def) = (
                a.create_label(),
                a.create_label(),
                a.create_label(),
                a.create_label(),
            );
            a.cmp(ecx, 3)?;
            a.ja(def)?;
            a.movsxd(rax, ecx)?;
            a.shl(rax, 2)?;
            lea_rip(a, Register::RDX, BASE)?;
            a.lea(rcx, ptr(rdx + rax + DATA_RVA as i32))?;
            a.mov(eax, dword_ptr(rcx))?;
            a.add(rax, rdx)?;
            a.jmp(rax)?;
            emit_four_cases(a, &mut c0, &mut c1, &mut c2, &mut def)
        });
        let cases_ps = find_four_cases(&code_prescaled);
        let jmp_ps = find_indirect_jmp(&code_prescaled);
        let data_ps = rva_table(cases_ps.map(|va| (va - BASE) as u32));
        let analysis_ps = analyze(
            &code_prescaled,
            &data_ps,
            &[0],
            &[(0, code_prescaled.len() as u32, 0)],
        );
        assert_eq!(
            analysis_ps
                .jump_tables
                .get(&jmp_ps)
                .expect("pre-scaled shl + lea ptr resolved")
                .targets,
            cases_ps
        );
    }

    #[test]
    fn rejects_unsafe_and_invalid_extended_patterns() {
        // 1) Unguarded `sub` after upper-only check (`ecx` could be < 10 and wrap!).
        let code_wrap = assemble_at_text(|a| {
            let (mut c0, mut c1, mut c2, mut def) = (
                a.create_label(),
                a.create_label(),
                a.create_label(),
                a.create_label(),
            );
            a.cmp(ecx, 13)?;
            a.ja(def)?;
            a.sub(ecx, 10)?;
            lea_rip(a, Register::RDX, BASE)?;
            a.mov(eax, dword_ptr(rdx + rcx * 4 + DATA_RVA as i32))?;
            a.add(rax, rdx)?;
            a.jmp(rax)?;
            emit_four_cases(a, &mut c0, &mut c1, &mut c2, &mut def)
        });
        let cases_wrap = find_four_cases(&code_wrap);
        let jmp_wrap = (find_indirect_jmp(&code_wrap) - TEXT) as u32;
        let data_wrap = rva_table(cases_wrap.map(|va| (va - BASE) as u32));
        assert_not_proven(
            &analyze(
                &code_wrap,
                &data_wrap,
                &[0],
                &[(0, code_wrap.len() as u32, 0)],
            ),
            jmp_wrap,
        );

        // 2) Negative signed upper bound (`cmp ecx, -1; jg default`).
        let code_neg = assemble_at_text(|a| {
            let (mut c0, mut c1, mut c2, mut def) = (
                a.create_label(),
                a.create_label(),
                a.create_label(),
                a.create_label(),
            );
            a.cmp(ecx, -1)?;
            a.jg(def)?;
            a.movsxd(rax, ecx)?;
            lea_rip(a, Register::RDX, BASE)?;
            a.mov(ecx, dword_ptr(rdx + rax * 4 + DATA_RVA as i32))?;
            a.add(rcx, rdx)?;
            a.jmp(rcx)?;
            emit_four_cases(a, &mut c0, &mut c1, &mut c2, &mut def)
        });
        let cases_neg = find_four_cases(&code_neg);
        let jmp_neg = (find_indirect_jmp(&code_neg) - TEXT) as u32;
        let data_neg = rva_table(cases_neg.map(|va| (va - BASE) as u32));
        assert_not_proven(
            &analyze(&code_neg, &data_neg, &[0], &[(0, code_neg.len() as u32, 0)]),
            jmp_neg,
        );

        // 3) Volatile base register (`rdx`) clobbered by `call` before jump-table lookup.
        let code_clobber = assemble_at_text(|a| {
            let (mut c0, mut c1, mut c2, mut def, mut helper) = (
                a.create_label(),
                a.create_label(),
                a.create_label(),
                a.create_label(),
                a.create_label(),
            );
            lea_rip(a, Register::RDX, BASE)?;
            a.call(helper)?;
            a.cmp(eax, 3)?;
            a.ja(def)?;
            a.movsxd(rcx, eax)?;
            a.mov(eax, dword_ptr(rdx + rcx * 4 + DATA_RVA as i32))?;
            a.add(rax, rdx)?;
            a.jmp(rax)?;
            emit_four_cases(a, &mut c0, &mut c1, &mut c2, &mut def)?;
            a.set_label(&mut helper)?;
            a.mov(eax, 1)?;
            a.ret()?;
            Ok(())
        });
        let cases_cl = find_four_cases(&code_clobber);
        let jmp_cl = (find_indirect_jmp(&code_clobber) - TEXT) as u32;
        let data_cl = rva_table(cases_cl.map(|va| (va - BASE) as u32));
        assert_unresolved(&analyze(&code_clobber, &data_cl, &[0], &[]), jmp_cl);

        // 4) Table-relative signed offset pointing outside the containing `.pdata` function.
        let code_rel = assemble_at_text(|a| {
            let (mut c0, mut c1, mut c2, mut def) = (
                a.create_label(),
                a.create_label(),
                a.create_label(),
                a.create_label(),
            );
            a.cmp(ecx, 3)?;
            a.ja(def)?;
            a.movsxd(rax, ecx)?;
            lea_rip(a, Register::RDX, DATA)?;
            a.movsxd(rcx, dword_ptr(rdx + rax * 4))?;
            a.add(rcx, rdx)?;
            a.jmp(rcx)?;
            emit_four_cases(a, &mut c0, &mut c1, &mut c2, &mut def)
        });
        let cases_rel = find_four_cases(&code_rel);
        let jmp_rel = (find_indirect_jmp(&code_rel) - TEXT) as u32;
        let data_rel: Vec<u8> = cases_rel
            .iter()
            .flat_map(|&va| ((va as i64 - DATA as i64) as i32).to_le_bytes())
            .collect();
        let def_off = (cases_rel[3] - TEXT) as u32;
        assert_unresolved(
            &analyze(
                &code_rel,
                &data_rel,
                &[0],
                &[(0, def_off, 0), (def_off, code_rel.len() as u32, def_off)],
            ),
            jmp_rel,
        );

        // 5) Absolute 64-bit pointer table with a non-code target in `.rdata`.
        let code_abs = assemble_at_text(|a| {
            let (mut c0, mut c1, mut c2, mut def) = (
                a.create_label(),
                a.create_label(),
                a.create_label(),
                a.create_label(),
            );
            a.cmp(ecx, 3)?;
            a.ja(def)?;
            a.movsxd(rax, ecx)?;
            lea_rip(a, Register::RDX, DATA)?;
            a.mov(rax, qword_ptr(rdx + rax * 8))?;
            a.jmp(rax)?;
            emit_four_cases(a, &mut c0, &mut c1, &mut c2, &mut def)
        });
        let mut cases_abs = find_four_cases(&code_abs);
        cases_abs[2] = DATA + 0x20; // non-executable data target
        let jmp_abs = (find_indirect_jmp(&code_abs) - TEXT) as u32;
        let mut data_abs: Vec<u8> = cases_abs.iter().flat_map(|&va| va.to_le_bytes()).collect();
        data_abs.resize(0x30, 0);
        assert_unresolved(&analyze(&code_abs, &data_abs, &[0], &[]), jmp_abs);

        // 6) `and ecx, 7` when entries 4..7 point into data (out-of-code targets must reject all).
        let code_and = assemble_at_text(|a| {
            let (mut c0, mut c1, mut c2, mut def) = (
                a.create_label(),
                a.create_label(),
                a.create_label(),
                a.create_label(),
            );
            a.and(ecx, 7)?;
            lea_rip(a, Register::RDX, BASE)?;
            a.mov(eax, dword_ptr(rdx + rcx * 4 + DATA_RVA as i32))?;
            a.add(rax, rdx)?;
            a.jmp(rax)?;
            emit_four_cases(a, &mut c0, &mut c1, &mut c2, &mut def)
        });
        let cases_and = find_four_cases(&code_and);
        let jmp_and = (find_indirect_jmp(&code_and) - TEXT) as u32;
        let mut data_and = rva_table(cases_and.map(|va| (va - BASE) as u32));
        data_and.extend_from_slice(&rva_table([DATA_RVA; 4]));
        assert_unresolved(&analyze(&code_and, &data_and, &[0], &[]), jmp_and);
    }

    // ---- R29: tables behind control-flow merges (reaching values over the CFG) ----------------

    /// Assembles `build` at `TEXT`; returns the bytes and the address of every label in `labels`
    /// order (labels are created by `build` and pushed into the vector).
    fn assemble_labeled(
        build: impl FnOnce(&mut CodeAssembler, &mut Vec<CodeLabel>) -> Result<(), iced_x86::IcedError>,
    ) -> (Vec<u8>, Vec<u64>) {
        let mut a = CodeAssembler::new(64).expect("64-bit assembler");
        let mut labels = Vec::new();
        build(&mut a, &mut labels).expect("assemble");
        let result = a
            .assemble_options(
                TEXT,
                iced_x86::BlockEncoderOptions::RETURN_NEW_INSTRUCTION_OFFSETS,
            )
            .expect("encode at TEXT");
        let ips = labels
            .iter()
            .map(|l| result.label_ip(l).expect("label ip"))
            .collect();
        (result.inner.code_buffer, ips)
    }

    /// Table-relative signed 32-bit offsets of `targets` (table at `DATA`), followed by a zero
    /// entry (which points at the table itself, not at code).
    fn signed_table(targets: &[u64]) -> Vec<u8> {
        let mut out = signed_offsets(targets);
        out.extend(0i32.to_le_bytes());
        out
    }

    /// The boot's shape: a loop whose dispatch block starts at a merge. The table base `rsi` is
    /// set in the entry block; the index `ecx` arrives as a different constant on every edge into
    /// the merge; case 1 parks the base in `rbx`, reuses `rsi` for a stack address and restores it
    /// before looping, while `rbx` holds another table address on entry.
    /// ```text
    /// entry: lea rsi, [rip+DATA]; lea rbx, [rip+DATA+0x40]; xor ecx, ecx; xor eax, eax
    /// merge: test edi, edi; je done
    ///        dec edi; mov r8d, ecx; movsxd rdx, [rsi+r8*4]; add rdx, rsi; jmp rdx
    /// case0: add eax, 1;    mov ecx, 1; jmp merge
    /// case1: add eax, 10;   mov rbx, rsi; lea rsi, [rsp-8]; mov r9, [rsi]; mov rsi, rbx;
    ///        mov ecx, 2; jmp merge
    /// case2: add eax, 100;  mov ecx, 3; jmp merge
    /// case3: add eax, 1000; xor ecx, ecx; jmp merge
    /// done:  ret
    /// ```
    /// `edi` iterations give eax = sum of the visited cases in the order 0, 1, 2, 3, 0, ...
    fn merge_switch() -> (Vec<u8>, [u64; 4], u64) {
        let (code, ips) = assemble_labeled(|a, labels| {
            let mut merge = a.create_label();
            let mut done = a.create_label();
            let mut dispatch = a.create_label();
            let mut cases = [
                a.create_label(),
                a.create_label(),
                a.create_label(),
                a.create_label(),
            ];
            lea_rip(a, Register::RSI, DATA)?;
            lea_rip(a, Register::RBX, DATA + 0x40)?;
            a.xor(ecx, ecx)?;
            a.xor(eax, eax)?;
            a.set_label(&mut merge)?;
            a.test(edi, edi)?;
            a.je(done)?;
            a.dec(edi)?;
            a.mov(r8d, ecx)?;
            a.set_label(&mut dispatch)?;
            a.movsxd(rdx, dword_ptr(rsi + r8 * 4))?;
            a.add(rdx, rsi)?;
            a.jmp(rdx)?;
            a.set_label(&mut cases[0])?;
            a.add(eax, 1)?;
            a.mov(ecx, 1)?;
            a.jmp(merge)?;
            a.set_label(&mut cases[1])?;
            a.add(eax, 10)?;
            a.mov(rbx, rsi)?;
            a.lea(rsi, ptr(rsp - 8))?;
            a.mov(r9, qword_ptr(rsi))?;
            a.mov(rsi, rbx)?;
            a.mov(ecx, 2)?;
            a.jmp(merge)?;
            a.set_label(&mut cases[2])?;
            a.add(eax, 100)?;
            a.mov(ecx, 3)?;
            a.jmp(merge)?;
            a.set_label(&mut cases[3])?;
            a.add(eax, 1000)?;
            a.xor(ecx, ecx)?;
            a.jmp(merge)?;
            a.set_label(&mut done)?;
            a.ret()?;
            labels.extend([cases[0], cases[1], cases[2], cases[3], dispatch]);
            Ok(())
        });
        let cases = [ips[0], ips[1], ips[2], ips[3]];
        let jmp_va = find_indirect_jmp_after(&code, ips[4]);
        (code, cases, jmp_va)
    }

    fn find_indirect_jmp_after(code: &[u8], from: u64) -> u64 {
        let mut dec = Decoder::with_ip(64, code, TEXT, DecoderOptions::NONE);
        while dec.can_decode() {
            let instr = dec.decode();
            if instr.ip() >= from && instr.flow_control() == iced_x86::FlowControl::IndirectBranch {
                return instr.ip();
            }
        }
        panic!("no indirect jmp after {from:#x}");
    }

    #[test]
    fn resolves_merge_switch_with_base_carried_through_a_register_copy() {
        // With a `.pdata` owner the base alone resolves it (inferred extent, as before R29).
        let (code, cases, jmp_va) = merge_switch();
        let analysis = analyze(
            &code,
            &signed_table(&cases),
            &[0],
            &[(0, code.len() as u32, 0)],
        );
        let jt = analysis
            .jump_tables
            .get(&jmp_va)
            .expect("merge switch resolved");
        assert_eq!(jt.table, DATA);
        assert_eq!(jt.targets, cases);
    }

    #[test]
    fn resolves_merge_switch_from_the_index_constants_on_every_incoming_edge() {
        // Without `.pdata` no extent can be inferred: the bound [0, 3] comes from the constant
        // index on each of the five edges into the merge.
        let (code, cases, jmp_va) = merge_switch();
        let analysis = analyze(&code, &signed_table(&cases), &[0], &[]);
        let jt = analysis
            .jump_tables
            .get(&jmp_va)
            .expect("bounded by the reaching index values");
        assert!(!jt.inferred);
        assert_eq!(jt.targets, cases);
    }

    /// A merge whose dispatch reads `ecx`, entered from two guarded paths and (when `third` is
    /// set) a third path built by `third`:
    /// ```text
    /// entry: lea rsi, [rip+DATA]; mov ecx, [rsp+8]; test edx, edx; jne b
    ///   a:   cmp ecx, 2; ja done; jmp merge
    ///   b:   and ecx, 3; jmp merge            (or the `third` variant before `jmp merge`)
    /// merge: movsxd rax, [rsi+rcx*4]; add rax, rsi; jmp rax
    /// cases, done
    /// ```
    fn guarded_merge(
        path_b: impl FnOnce(&mut CodeAssembler, CodeLabel, CodeLabel) -> Result<(), iced_x86::IcedError>,
    ) -> (Vec<u8>, [u64; 4], u64) {
        let (code, ips) = assemble_labeled(|a, labels| {
            let mut b = a.create_label();
            let mut merge = a.create_label();
            let mut done = a.create_label();
            let mut helper = a.create_label();
            let mut cases = [
                a.create_label(),
                a.create_label(),
                a.create_label(),
                a.create_label(),
            ];
            lea_rip(a, Register::RSI, DATA)?;
            a.mov(ecx, dword_ptr(rsp + 8))?;
            a.test(edx, edx)?;
            a.jne(b)?;
            a.cmp(ecx, 2)?;
            a.ja(done)?;
            a.jmp(merge)?;
            a.set_label(&mut b)?;
            path_b(a, done, helper)?;
            a.jmp(merge)?;
            a.set_label(&mut merge)?;
            a.mov(ecx, ecx)?;
            a.movsxd(rax, dword_ptr(rsi + rcx * 4))?;
            a.add(rax, rsi)?;
            a.jmp(rax)?;
            for (i, case) in cases.iter_mut().enumerate() {
                a.set_label(case)?;
                a.mov(eax, 0x20 + i as u32)?;
                a.ret()?;
            }
            a.set_label(&mut done)?;
            a.xor(eax, eax)?;
            a.ret()?;
            a.set_label(&mut helper)?;
            a.ret()?;
            labels.extend([cases[0], cases[1], cases[2], cases[3], merge]);
            Ok(())
        });
        let cases = [ips[0], ips[1], ips[2], ips[3]];
        let jmp_va = find_indirect_jmp_after(&code, ips[4]);
        (code, cases, jmp_va)
    }

    #[test]
    fn resolves_switch_bounded_by_different_guards_on_each_incoming_path() {
        // Path a: `cmp ecx, 2; ja done` (<= 2); path b: `and ecx, 3` (<= 3): the larger bound.
        let (code, cases, jmp_va) = guarded_merge(|a, _, _| a.and(ecx, 3));
        let analysis = analyze(&code, &signed_table(&cases), &[0], &[]);
        let jt = analysis
            .jump_tables
            .get(&jmp_va)
            .expect("bounded on both paths");
        assert!(!jt.inferred);
        assert_eq!(jt.targets, cases);
    }

    #[test]
    fn merge_switch_with_an_unguarded_path_stays_unresolved() {
        // Path b leaves the loaded index unbounded.
        let (code, cases, jmp_va) = guarded_merge(|a, _, _| a.nop());
        let analysis = analyze(&code, &signed_table(&cases), &[0], &[]);
        assert!(!analysis.jump_tables.contains_key(&jmp_va));
        assert_eq!(
            analysis.unresolved_jumps.get(&jmp_va),
            Some(&UnresolvedJump::TableUnbounded)
        );
    }

    #[test]
    fn merge_switch_with_a_guard_on_the_wrong_register_stays_unresolved() {
        let (code, cases, jmp_va) = guarded_merge(|a, done, _| {
            a.cmp(edx, 3)?;
            a.ja(done)
        });
        let analysis = analyze(&code, &signed_table(&cases), &[0], &[]);
        assert!(!analysis.jump_tables.contains_key(&jmp_va));
    }

    #[test]
    fn merge_switch_with_the_base_overwritten_after_the_guard_stays_unresolved() {
        // Path b bounds the index, then replaces the base with a loaded pointer.
        let (code, cases, jmp_va) = guarded_merge(|a, _, _| {
            a.and(ecx, 3)?;
            a.mov(rsi, qword_ptr(rsp + 16))
        });
        let analysis = analyze(&code, &signed_table(&cases), &[0], &[]);
        assert!(!analysis.jump_tables.contains_key(&jmp_va));
        assert_eq!(
            analysis.unresolved_jumps.get(&jmp_va),
            Some(&UnresolvedJump::TableBaseUnknown)
        );
    }

    #[test]
    fn merge_switch_with_conflicting_bases_stays_unresolved() {
        // Path b bounds the index but points the base at another table.
        let (code, cases, jmp_va) = guarded_merge(|a, _, _| {
            a.and(ecx, 3)?;
            lea_rip(a, Register::RSI, DATA + 0x10)
        });
        let mut data = signed_table(&cases);
        data.extend(signed_table(&cases));
        let analysis = analyze(&code, &data, &[0], &[]);
        assert!(!analysis.jump_tables.contains_key(&jmp_va));
        assert_eq!(
            analysis.unresolved_jumps.get(&jmp_va),
            Some(&UnresolvedJump::TableBaseUnknown)
        );
    }

    #[test]
    fn merge_switch_with_an_index_reloaded_after_a_call_stays_unresolved() {
        // Path b bounds ecx, spills it, calls (which may change the slot) and reloads it.
        let (code, cases, jmp_va) = guarded_merge(|a, _, helper| {
            a.and(ecx, 3)?;
            a.mov(dword_ptr(rsp + 0x20), ecx)?;
            a.call(helper)?;
            a.mov(ecx, dword_ptr(rsp + 0x20))
        });
        let analysis = analyze(&code, &signed_table(&cases), &[0], &[]);
        assert!(!analysis.jump_tables.contains_key(&jmp_va));
    }

    #[test]
    fn large_loop_switch_keeps_its_table_once_the_cases_are_discovered() {
        // A dispatch loop (base in r14 before the loop, `cmp ecx, N-1; ja` guard at the merge)
        // whose cases, once discovered, put more than 512 blocks behind the merge: every case
        // branches and jumps back to the merge. The table must survive the final verification.
        const CASES: usize = 200;
        let (code, ips) = assemble_labeled(|a, labels| {
            let mut merge = a.create_label();
            let mut done = a.create_label();
            let mut cases: Vec<CodeLabel> = (0..CASES).map(|_| a.create_label()).collect();
            lea_rip(a, Register::R14, DATA)?;
            a.xor(eax, eax)?;
            a.set_label(&mut merge)?;
            a.test(edi, edi)?;
            a.je(done)?;
            a.dec(edi)?;
            a.movzx(ecx, byte_ptr(rsi))?;
            a.inc(rsi)?;
            a.cmp(ecx, (CASES - 1) as u32)?;
            a.ja(merge)?;
            a.movsxd(rcx, dword_ptr(r14 + rcx * 4))?;
            a.add(rcx, r14)?;
            a.jmp(rcx)?;
            for (i, case) in cases.iter_mut().enumerate() {
                let mut skip = a.create_label();
                a.set_label(case)?;
                a.test(edx, edx)?;
                a.je(skip)?;
                a.add(eax, i as u32)?;
                a.set_label(&mut skip)?;
                a.inc(eax)?;
                a.jmp(merge)?;
            }
            a.set_label(&mut done)?;
            a.ret()?;
            labels.extend(cases.iter().copied());
            Ok(())
        });
        let jmp_va = find_indirect_jmp(&code);
        let analysis = analyze(
            &code,
            &signed_table(&ips),
            &[0],
            &[(0, code.len() as u32, 0)],
        );
        let jt = analysis
            .jump_tables
            .get(&jmp_va)
            .expect("table kept after the cases join the loop");
        assert!(!jt.inferred);
        assert_eq!(jt.targets, ips);
        let blocks = analysis.functions[&TEXT].blocks.len();
        assert!(blocks > 512, "{blocks} blocks");
    }

    // ---- R39: `!=` guard at the end of a masked range, behind a loop merge ------------------

    /// A 15-case table indexed by `r10`, masked to `[0, 15]` and with `15` sent to the default
    /// before a loop that leaves `r10` alone, so the slice stops at the loop header. The index
    /// reaches `rcx` as `movzx ecx, r10b` (`byte_index`; the slice alone sees only the byte's width)
    /// or as `mov ecx, r10d` (the slice alone sees no bound at all):
    /// ```text
    /// entry: and r10d, 0xf; cmp r10, 0xf; je default
    ///        xor esi, esi
    /// loop:  inc rsi; cmp rsi, r11; jne loop
    ///        movzx ecx, r10b | mov ecx, r10d; lea rdx, [rip+DATA]; movsxd rcx, [rdx+rcx*4]; add rcx, rdx; jmp rcx
    /// cases 0..=14: mov eax, 0x20+i; ret
    /// default: xor eax, eax; ret
    /// ```
    fn not_equal_guarded_loop_switch(byte_index: bool) -> (Vec<u8>, Vec<u64>, u64) {
        const CASES: usize = 15;
        let (code, ips) = assemble_labeled(|a, labels| {
            let mut lp = a.create_label();
            let mut default = a.create_label();
            let mut cases: Vec<CodeLabel> = (0..CASES).map(|_| a.create_label()).collect();
            a.and(r10d, 0xf)?;
            a.cmp(r10, 0xf)?;
            a.je(default)?;
            a.xor(esi, esi)?;
            a.set_label(&mut lp)?;
            a.inc(rsi)?;
            a.cmp(rsi, r11)?;
            a.jne(lp)?;
            if byte_index {
                a.movzx(ecx, r10b)?;
            } else {
                a.mov(ecx, r10d)?;
            }
            lea_rip(a, Register::RDX, DATA)?;
            a.movsxd(rcx, dword_ptr(rdx + rcx * 4))?;
            a.add(rcx, rdx)?;
            a.jmp(rcx)?;
            for (i, case) in cases.iter_mut().enumerate() {
                a.set_label(case)?;
                a.mov(eax, 0x20 + i as u32)?;
                a.ret()?;
            }
            a.set_label(&mut default)?;
            a.xor(eax, eax)?;
            a.ret()?;
            labels.extend(cases.iter().copied());
            Ok(())
        });
        let jmp_va = find_indirect_jmp(&code);
        (code, ips, jmp_va)
    }

    #[test]
    fn resolves_masked_index_excluded_at_its_top_by_a_not_equal_guard_before_a_loop() {
        // With `.pdata` the slice alone infers the extent (from the byte width for `movzx`, from
        // nothing for `mov`); the bound proven by the reaching `!=` guard must replace it.
        for byte_index in [true, false] {
            let (code, cases, jmp_va) = not_equal_guarded_loop_switch(byte_index);
            assert_eq!(cases.len(), 15);
            // Entry 15 (right after the 15 cases) is a zero offset: the table itself, not code.
            let data = signed_table(&cases);
            assert_eq!(data.len(), 16 * 4);
            for ranges in [&[][..], &[(0, code.len() as u32, 0)]] {
                let analysis = analyze(&code, &data, &[0], ranges);
                let jt = analysis.jump_tables.get(&jmp_va).unwrap_or_else(|| {
                    panic!("15-case table resolved (byte {byte_index}, ranges {ranges:?})")
                });
                assert!(!jt.inferred, "bound proven, not inferred: {jt:#x?}");
                assert_eq!(jt.table, DATA);
                assert_eq!(jt.targets, cases);
                assert!(!analysis.unresolved_jumps.contains_key(&jmp_va));
                let succs = jump_successors(&analysis.functions[&TEXT], jmp_va);
                assert!(
                    !succs.contains(&Edge::IndirectJump { slot: None }),
                    "no run-time dispatch fallback: {succs:x?}"
                );
                assert_eq!(succs.len(), 15, "{succs:x?}");
            }
        }
    }

    #[test]
    fn not_equal_guard_on_an_interior_value_leaves_the_sentinel_unproven() {
        // `cmp r10, 7; je default` excludes an interior value only: [0, 15] still reaches the
        // non-code entry 15, so no proven table may be claimed.
        for byte_index in [true, false] {
            let (mut code, cases, jmp_va) = not_equal_guarded_loop_switch(byte_index);
            // `cmp r10, 0xf` is `49 83 fa 0f` right after `and r10d, 0xf` (`41 83 e2 0f`).
            assert_eq!(&code[4..8], &[0x49, 0x83, 0xfa, 0x0f]);
            code[7] = 7;
            for ranges in [&[][..], &[(0, code.len() as u32, 0)]] {
                let analysis = analyze(&code, &signed_table(&cases), &[0], ranges);
                assert_not_proven(&analysis, (jmp_va - TEXT) as u32);
            }
        }
    }

    #[test]
    fn unresolved_jumps_are_classified() {
        // Three functions: `jmp [rcx+8]`, `mov rax, [rcx]; jmp rax`, and `jmp rcx`.
        let (code, ips) = assemble_labeled(|a, labels| {
            let mut f1 = a.create_label();
            let mut f2 = a.create_label();
            let mut f3 = a.create_label();
            a.set_label(&mut f1)?;
            a.jmp(qword_ptr(rcx + 8))?;
            a.set_label(&mut f2)?;
            a.mov(rax, qword_ptr(rcx))?;
            a.jmp(rax)?;
            a.set_label(&mut f3)?;
            a.add(rcx, 0x40)?;
            a.jmp(rdx)?;
            labels.extend([f1, f2, f3]);
            Ok(())
        });
        let entries: Vec<u32> = ips.iter().map(|&va| (va - TEXT) as u32).collect();
        let analysis = analyze(&code, &[0; 16], &entries, &[]);
        let kinds: Vec<UnresolvedJump> = analysis.unresolved_jumps.values().copied().collect();
        assert_eq!(
            kinds,
            [
                UnresolvedJump::MemoryOperand,
                UnresolvedJump::LoadedPointer,
                UnresolvedJump::RegisterLiveIn,
            ]
        );
    }

    // ---- R44: checked bounds, determinism, entry caps & slice-budget hardening -----------------

    #[test]
    fn r44_checked_arithmetic_rejects_rva_va_wraparound_and_short_chunks() {
        use super::{EntryEncoding, checked_disp_add, checked_rip_target, infer_extent, read};

        // Truncated chunks never panic.
        let enc_u32 = EntryEncoding::Unsigned32 { base: BASE };
        let enc_i32 = EntryEncoding::Signed32 { base: BASE };
        let enc_abs = EntryEncoding::Absolute64;
        assert_eq!(enc_u32.decode(&[]), None);
        assert_eq!(enc_u32.decode(&[1, 2, 3]), None);
        assert_eq!(enc_i32.decode(&[1, 2]), None);
        assert_eq!(enc_abs.decode(&[0; 7]), None);

        // Unsigned32 overflow past u64::MAX is rejected instead of wrapping into low memory.
        let high_u32 = EntryEncoding::Unsigned32 {
            base: u64::MAX - 0x10,
        };
        assert_eq!(high_u32.decode(&0x20u32.to_le_bytes()), None);
        assert_eq!(high_u32.decode(&0x10u32.to_le_bytes()), Some(u64::MAX));

        // Signed32 underflow below 0 and overflow past u64::MAX are rejected.
        let low_i32 = EntryEncoding::Signed32 { base: 0x10 };
        assert_eq!(low_i32.decode(&(-0x20i32).to_le_bytes()), None);
        assert_eq!(low_i32.decode(&(-0x10i32).to_le_bytes()), Some(0));
        let high_i32 = EntryEncoding::Signed32 {
            base: u64::MAX - 0x10,
        };
        assert_eq!(high_i32.decode(&0x20i32.to_le_bytes()), None);
        assert_eq!(high_i32.decode(&0x10i32.to_le_bytes()), Some(u64::MAX));

        // Signed displacement addition rejects wraparound in both directions.
        assert_eq!(checked_disp_add(BASE, 0x2000), Some(BASE + 0x2000));
        assert_eq!(
            checked_disp_add(BASE + 0x2000, (-0x1000i64) as u64),
            Some(BASE + 0x1000)
        );
        assert_eq!(checked_disp_add(0x10, (-0x20i64) as u64), None);
        assert_eq!(checked_disp_add(u64::MAX - 0x10, 0x20), None);

        // Wrapping RIP-relative LEA near u64::MAX is rejected.
        let lea_plus_100 = [0x48, 0x8d, 0x15, 0x00, 0x01, 0x00, 0x00]; // lea rdx, [rip + 0x100]
        let mut dec_ok = Decoder::with_ip(64, &lea_plus_100, BASE + 0x1000, DecoderOptions::NONE);
        assert_eq!(
            checked_rip_target(&dec_ok.decode()),
            Some(BASE + 0x1007 + 0x100)
        );
        let mut dec_wrap = Decoder::with_ip(64, &lea_plus_100, u64::MAX - 0x20, DecoderOptions::NONE);
        assert_eq!(checked_rip_target(&dec_wrap.decode()), None);

        // `read` rejects zero-length reads and VA wraparound across u64::MAX.
        let tail_bytes = [0x90u8; 16];
        let wrap_image = CodeImage::new(
            u64::MAX - 16,
            vec![CodeSection::new(u64::MAX - 8, &tail_bytes, false)],
            vec![],
        );
        assert_eq!(read(&wrap_image, u64::MAX - 8, 0), None);
        assert_eq!(read(&wrap_image, u64::MAX - 8, 12), None);
        assert_eq!(read(&wrap_image, u64::MAX - 8, 8).map(<[u8]>::len), Some(8));

        // `infer_extent` near u64::MAX stops cleanly without overflowing `va += entry_size`.
        let code = [0xff, 0xe1, 0xc3, 0xc3]; // jmp rcx; ret; ret
        let code_va = u64::MAX - 0x200;
        let table_va = u64::MAX - 8;
        let target_a = code_va + 2;
        let target_b = code_va + 3;
        let mut table_bytes = Vec::new();
        table_bytes.extend_from_slice(&((target_a - code_va) as u32).to_le_bytes());
        table_bytes.extend_from_slice(&((target_b - code_va) as u32).to_le_bytes());
        let mut high_image = CodeImage::new(
            code_va,
            vec![
                CodeSection::new(code_va, &code, true),
                CodeSection::new(table_va, &table_bytes, false),
            ],
            vec![code_va],
        );
        high_image.function_table = FunctionTable::new(vec![FunctionRange {
            start: code_va,
            end: code_va + code.len() as u64,
            function: code_va,
        }]);
        let jmp = Decoder::with_ip(64, &code, code_va, DecoderOptions::NONE).decode();
        let inferred = infer_extent(
            &high_image,
            &jmp,
            code_va,
            table_va,
            EntryEncoding::Unsigned32 { base: code_va },
            None,
            16,
        )
        .expect("two valid entries before u64::MAX resolve without overflowing");
        assert_eq!(inferred.targets, [target_a, target_b]);
    }

    #[test]
    fn r44_bit_width_helpers_and_malformed_traces_never_panic() {
        use super::{Slicer, classify, mask, max_signed, min_signed, resolve, sign_extend};
        use iced_x86::Mnemonic;

        // Zero and oversized bit widths never panic on shift overflow.
        assert_eq!(mask(0), 0);
        assert_eq!(mask(64), u64::MAX);
        assert_eq!(mask(128), u64::MAX);
        assert_eq!(sign_extend(0xff, 0), 0);
        assert_eq!(sign_extend(0xff, 8), -1);
        assert_eq!(min_signed(0), 0);
        assert_eq!(max_signed(0), 0);
        assert!(Slicer::cmp_imm_fact(Mnemonic::Ja, false, 3, 0).is_none());
        assert!(Slicer::cmp_imm_fact(Mnemonic::Ja, false, 3, 128).is_none());

        // Empty trace and invalid Instruction::default() never panic.
        let image = CodeImage::new(BASE, vec![], vec![]);
        assert!(resolve(&image, &[]).is_none());
        assert_eq!(classify(&image, &[]), UnresolvedJump::Computed);

        let invalid = Instruction::default();
        let bad_trace = [super::Step {
            instr: &invalid,
            taken: None,
        }];
        assert!(resolve(&image, &bad_trace).is_none());
        assert_eq!(classify(&image, &bad_trace), UnresolvedJump::Computed);
    }

    #[test]
    fn r44_deterministic_target_ordering_and_sorted_unique_targets() {
        use super::sorted_unique_targets;

        // 6-entry table with reversed and duplicated case targets.
        let code = assemble_at_text(|a| {
            let (mut c0, mut c1, mut c2, mut def) = (
                a.create_label(),
                a.create_label(),
                a.create_label(),
                a.create_label(),
            );
            a.cmp(ecx, 5)?;
            a.ja(def)?;
            a.movsxd(rax, ecx)?;
            lea_rip(a, Register::RDX, BASE)?;
            a.mov(ecx, dword_ptr(rdx + rax * 4 + DATA_RVA as i32))?;
            a.add(rcx, rdx)?;
            a.jmp(rcx)?;
            emit_four_cases(a, &mut c0, &mut c1, &mut c2, &mut def)
        });
        let cases = find_four_cases(&code);
        let jmp_va = find_indirect_jmp(&code);
        let slot_targets = [cases[2], cases[0], cases[2], cases[1], cases[0], cases[3]];
        let data = rva_table(slot_targets.map(|va| (va - BASE) as u32));

        let run_a = analyze(&code, &data, &[0], &[(0, code.len() as u32, 0)]);
        let run_b = analyze(&code, &data, &[0], &[(0, code.len() as u32, 0)]);
        assert_eq!(run_a, run_b);

        let jt = run_a
            .jump_tables
            .get(&jmp_va)
            .expect("switch with reversed/duplicate targets resolved");
        // Table-slot order is preserved for JumpTableEntry relocations.
        assert_eq!(jt.targets, slot_targets);
        // Canonical sorted unique target view is strictly ascending and deduplicated.
        let canonical = sorted_unique_targets(jt);
        let mut expected = cases.to_vec();
        expected.sort_unstable();
        expected.dedup();
        assert_eq!(canonical, expected);
        assert_eq!(
            jump_successors(&run_a.functions[&TEXT], jmp_va),
            expected
                .iter()
                .copied()
                .map(Edge::JumpTable)
                .collect::<Vec<_>>()
        );

        // Duplicate `!=` guards on the upper bound still trim deterministically.
        let code_neq = assemble_at_text(|a| {
            let (mut c0, mut c1, mut c2, mut def) = (
                a.create_label(),
                a.create_label(),
                a.create_label(),
                a.create_label(),
            );
            a.and(ecx, 3)?;
            a.cmp(ecx, 3)?;
            a.je(def)?;
            a.cmp(ecx, 3)?;
            a.je(def)?;
            a.cmp(ecx, 2)?;
            a.je(def)?;
            lea_rip(a, Register::RDX, BASE)?;
            a.mov(eax, dword_ptr(rdx + rcx * 4 + DATA_RVA as i32))?;
            a.add(rax, rdx)?;
            a.jmp(rax)?;
            emit_four_cases(a, &mut c0, &mut c1, &mut c2, &mut def)
        });
        let neq_cases = find_four_cases(&code_neq);
        let neq_jmp = find_indirect_jmp(&code_neq);
        let neq_data = rva_table(neq_cases[..2].iter().map(|va| (va - BASE) as u32));
        let neq_analysis = analyze(&code_neq, &neq_data, &[0], &[(0, code_neq.len() as u32, 0)]);
        let neq_jt = neq_analysis
            .jump_tables
            .get(&neq_jmp)
            .expect("duplicate NotEqual guards trim deterministically");
        assert!(!neq_jt.inferred);
        assert_eq!(neq_jt.targets, neq_cases[..2]);
    }

    #[test]
    fn r44_bounded_entry_caps_and_slicer_step_budget() {
        use super::{
            EntryEncoding, MAX_INFERRED_ENTRIES, MAX_JUMP_TABLE_ENTRIES, classify, infer_extent,
            resolve,
        };

        // 1) Proven bound exceeding MAX_JUMP_TABLE_ENTRIES is rejected cleanly.
        let code_over_cap = assemble_at_text(|a| {
            let (mut c0, mut c1, mut c2, mut def) = (
                a.create_label(),
                a.create_label(),
                a.create_label(),
                a.create_label(),
            );
            a.cmp(ecx, MAX_JUMP_TABLE_ENTRIES as u32)?; // 0..=65536 => 65537 entries > cap
            a.ja(def)?;
            a.movsxd(rax, ecx)?;
            lea_rip(a, Register::RDX, BASE)?;
            a.mov(ecx, dword_ptr(rdx + rax * 4 + DATA_RVA as i32))?;
            a.add(rcx, rdx)?;
            a.jmp(rcx)?;
            emit_four_cases(a, &mut c0, &mut c1, &mut c2, &mut def)
        });
        let over_cases = find_four_cases(&code_over_cap);
        let over_jmp = (find_indirect_jmp(&code_over_cap) - TEXT) as u32;
        let over_data = rva_table(over_cases.map(|va| (va - BASE) as u32));
        assert_unresolved(&analyze(&code_over_cap, &over_data, &[0], &[]), over_jmp);

        // 2) Inferred extent strictly caps at MAX_INFERRED_ENTRIES and rejects limit < 2.
        let (loop_code, loop_cases, loop_jmp) = loop_remainder_switch();
        let big_data = rva_table(std::iter::repeat_n(
            (loop_cases[0] - BASE) as u32,
            MAX_INFERRED_ENTRIES + 32,
        ));
        let big_analysis = analyze(
            &loop_code,
            &big_data,
            &[0],
            &[(0, loop_code.len() as u32, 0)],
        );
        let big_jt = big_analysis
            .jump_tables
            .get(&loop_jmp)
            .expect("large inferred table resolved up to cap");
        assert!(big_jt.inferred);
        assert_eq!(big_jt.targets.len(), MAX_INFERRED_ENTRIES);

        let mut img = CodeImage::new(
            BASE,
            vec![
                CodeSection::new(TEXT, &loop_code, true),
                CodeSection::new(DATA, &big_data, false),
            ],
            vec![TEXT],
        );
        img.function_table = FunctionTable::new(vec![FunctionRange {
            start: TEXT,
            end: TEXT + loop_code.len() as u64,
            function: TEXT,
        }]);
        let mut dec = Decoder::with_ip(64, &loop_code, TEXT, DecoderOptions::NONE);
        let mut jmp_instr = Instruction::default();
        while dec.can_decode() {
            let i = dec.decode();
            if i.ip() == loop_jmp {
                jmp_instr = i;
                break;
            }
        }
        assert!(
            infer_extent(
                &img,
                &jmp_instr,
                loop_jmp,
                DATA,
                EntryEncoding::Unsigned32 { base: BASE },
                None,
                1,
            )
            .is_none()
        );

        // 3) Adversarial exponential-fanout `add rax, rax` chain hits MAX_SLICE_STEPS cleanly.
        let fanout_code = assemble_at_text(|a| {
            for _ in 0..24 {
                a.add(rax, rax)?;
            }
            a.jmp(rax)?;
            Ok(())
        });
        let fanout_img = CodeImage::new(
            BASE,
            vec![CodeSection::new(TEXT, &fanout_code, true)],
            vec![TEXT],
        );
        let mut dec = Decoder::with_ip(64, &fanout_code, TEXT, DecoderOptions::NONE);
        let mut instrs = Vec::new();
        while dec.can_decode() {
            instrs.push(dec.decode());
        }
        let steps: Vec<super::Step<'_>> = instrs
            .iter()
            .map(|instr| super::Step { instr, taken: None })
            .collect();
        assert!(resolve(&fanout_img, &steps).is_none());
        assert_eq!(classify(&fanout_img, &steps), UnresolvedJump::Computed);
    }

    #[test]
    fn r44_truncated_tables_and_malformed_leaf_gaps_return_none() {
        use super::{EntryEncoding, infer_extent};

        // 1) Truncated 32-bit RVA table (missing 1..3 bytes of final entry).
        let code = switch_code(TEXT_RVA, DATA_RVA);
        let full_data = case_table(TEXT_RVA);
        for cut in 1..=3 {
            let truncated = &full_data[..full_data.len() - cut];
            assert_unresolved(&analyze(&code, truncated, &[0], &[]), 0x19);
        }

        // 2) Truncated 64-bit pointer table (missing final 4 bytes).
        let code_abs = assemble_at_text(|a| {
            let (mut c0, mut c1, mut c2, mut def) = (
                a.create_label(),
                a.create_label(),
                a.create_label(),
                a.create_label(),
            );
            a.cmp(ecx, 3)?;
            a.ja(def)?;
            a.movsxd(rax, ecx)?;
            lea_rip(a, Register::RDX, DATA)?;
            a.mov(rax, qword_ptr(rdx + rax * 8))?;
            a.jmp(rax)?;
            emit_four_cases(a, &mut c0, &mut c1, &mut c2, &mut def)
        });
        let cases_abs = find_four_cases(&code_abs);
        let jmp_abs = (find_indirect_jmp(&code_abs) - TEXT) as u32;
        let data_abs: Vec<u8> = cases_abs.iter().flat_map(|&va| va.to_le_bytes()).collect();
        assert_unresolved(
            &analyze(&code_abs, &data_abs[..data_abs.len() - 4], &[0], &[]),
            jmp_abs,
        );

        // 3) Truncated 16-bit word index table (odd byte length, missing 1 byte of last u16).
        let code_word = assemble_at_text(|a| {
            let (mut c0, mut c1, mut c2, mut def) = (
                a.create_label(),
                a.create_label(),
                a.create_label(),
                a.create_label(),
            );
            a.cmp(ecx, 3)?;
            a.ja(def)?;
            a.movsxd(rax, ecx)?;
            lea_rip(a, Register::RDX, BASE)?;
            a.movzx(eax, word_ptr(rdx + rax * 2 + (DATA_RVA + 0x20) as i32))?;
            a.mov(ecx, dword_ptr(rdx + rax * 4 + DATA_RVA as i32))?;
            a.add(rcx, rdx)?;
            a.jmp(rcx)?;
            emit_four_cases(a, &mut c0, &mut c1, &mut c2, &mut def)
        });
        let cases_word = find_four_cases(&code_word);
        let jmp_word = (find_indirect_jmp(&code_word) - TEXT) as u32;
        let mut data_word = rva_table(cases_word.map(|va| (va - BASE) as u32));
        data_word.resize(0x20, 0xcc);
        data_word.extend_from_slice(&[0, 0, 1, 0, 2, 0, 3]); // 7 bytes instead of 8!
        assert_unresolved(&analyze(&code_word, &data_word, &[0], &[]), jmp_word);

        // 4) Malformed / inverted leaf gap returns None cleanly without panicking.
        let leaf_code = [0xff, 0xe1, 0xc3, 0xc3];
        let leaf_data = rva_table([TEXT_RVA + 2, TEXT_RVA + 3]);
        let mut leaf_img = CodeImage::new(
            BASE,
            vec![
                CodeSection::new(TEXT, &leaf_code, true),
                CodeSection::new(DATA, &leaf_data, false),
            ],
            vec![TEXT],
        );
        leaf_img.function_table = FunctionTable::new(vec![
            FunctionRange {
                start: TEXT - 0x20,
                end: TEXT - 0x10,
                function: TEXT - 0x20,
            },
            FunctionRange {
                start: TEXT + 0x10,
                end: TEXT + 0x20,
                function: TEXT + 0x10,
            },
        ]);
        let jmp = Decoder::with_ip(64, &leaf_code, TEXT, DecoderOptions::NONE).decode();
        // Inverted leaf extent (entry > next).
        assert!(
            infer_extent(
                &leaf_img,
                &jmp,
                TEXT,
                DATA,
                EntryEncoding::Unsigned32 { base: BASE },
                Some((TEXT + 0x10, TEXT)),
                16,
            )
            .is_none()
        );
        // Jump whose next_ip equals the end of the leaf extent (empty post-dispatch span).
        assert!(
            infer_extent(
                &leaf_img,
                &jmp,
                TEXT,
                DATA,
                EntryEncoding::Unsigned32 { base: BASE },
                Some((TEXT, jmp.next_ip())),
                16,
            )
            .is_none()
        );
    }
    /// Two copies of one value made before a merge (`mov ebx, edx; mov r14d, edx`), the guard on
    /// one and the table indexed by the other. On one incoming path a call clobbers the common
    /// source: the copies are still copies of each other, so the guard bounds the index.
    #[test]
    fn resolves_switch_indexed_by_a_sibling_copy_across_a_call() {
        let lea_disp = (i64::from(DATA_RVA) - i64::from(TEXT_RVA + 0x1a)) as i32;
        let code = [
            &[0x89, 0xd3][..],               // 0: mov ebx, edx
            &[0x41, 0x89, 0xd6],             // 2: mov r14d, edx
            &[0x85, 0xc9],                   // 5: test ecx, ecx
            &[0x74, 0x05],                   // 7: je e
            &[0xe8, 0x24, 0x00, 0x00, 0x00], // 9: call 32 (clobbers rdx)
            &[0x83, 0xfb, 0x03],             // e: cmp ebx, 3
            &[0x77, 0x1c],                   // 11: ja 2f
            &[0x48, 0x8d, 0x05],             // 13: lea rax, [table]
            &lea_disp.to_le_bytes(),         //
            &[0x4a, 0x63, 0x0c, 0xb0],       // 1a: movsxd rcx, [rax+r14*4]
            &[0x48, 0x01, 0xc1],             // 1e: add rcx, rax
            &[0xff, 0xe1],                   // 21: jmp rcx
            &[0x31, 0xc0, 0xc3],             // 23: case 0
            &[0x31, 0xc0, 0xc3],             // 26: case 1
            &[0x31, 0xc0, 0xc3],             // 29: case 2
            &[0x31, 0xc0, 0xc3],             // 2c: case 3
            &[0x31, 0xc0, 0xc3],             // 2f: default
            &[0xc3],                         // 32: callee
        ]
        .concat();
        let cases = [0x23u32, 0x26, 0x29, 0x2c];
        let mut data: Vec<u8> = cases
            .iter()
            .flat_map(|&c| ((TEXT_RVA + c) as i32 - DATA_RVA as i32).to_le_bytes())
            .collect();
        // A fifth in-code entry past the bound must not be read.
        data.extend_from_slice(&((TEXT_RVA + 0x2f) as i32 - DATA_RVA as i32).to_le_bytes());
        let analysis = analyze(&code, &data, &[0], &[]);
        let table = analysis
            .jump_tables
            .get(&(TEXT + 0x21))
            .expect("sibling-copy-indexed switch resolves");
        assert!(!table.inferred);
        assert_eq!(table.targets, cases.map(|c| TEXT + u64::from(c)).to_vec());
    }

    /// The merge in front of a switch gains its second incoming edge only once an earlier switch
    /// resolves (its case calls, then falls into the merge). The first match of the later switch
    /// sees the one early path, whose facts prove no bound; it is matched again once its slice
    /// changes, and then the guard on a sibling copy bounds it.
    #[test]
    fn retries_a_switch_whose_merge_gains_an_edge_from_a_later_table() {
        let t1 = DATA_RVA;
        let t2 = DATA_RVA + 16;
        let lea1 = (i64::from(t1) - i64::from(TEXT_RVA + 0x11)) as i32;
        let lea2 = (i64::from(t2) - i64::from(TEXT_RVA + 0x2b)) as i32;
        let code = [
            &[0x89, 0xd3][..],               // 0: mov ebx, edx
            &[0x41, 0x89, 0xd6],             // 2: mov r14d, edx
            &[0x83, 0xfa, 0x04],             // 5: cmp edx, 4
            &[0x73, 0x15],                   // 8: jae 1f
            &[0x48, 0x8d, 0x05],             // a: lea rax, [t1]
            &lea1.to_le_bytes(),             //
            &[0x4a, 0x63, 0x0c, 0xb0],       // 11: movsxd rcx, [rax+r14*4]
            &[0x48, 0x01, 0xc1],             // 15: add rcx, rax
            &[0xff, 0xe1],                   // 18: jmp rcx
            &[0xe8, 0x24, 0x00, 0x00, 0x00], // 1a: call 43 (the only case of t1)
            &[0x83, 0xfb, 0x03],             // 1f: cmp ebx, 3 (merge)
            &[0x77, 0x1c],                   // 22: ja 40
            &[0x48, 0x8d, 0x05],             // 24: lea rax, [t2]
            &lea2.to_le_bytes(),             //
            &[0x4a, 0x63, 0x0c, 0xb0],       // 2b: movsxd rcx, [rax+r14*4]
            &[0x48, 0x01, 0xc1],             // 2f: add rcx, rax
            &[0xff, 0xe1],                   // 32: jmp rcx
            &[0x31, 0xc0, 0xc3],             // 34: case 0
            &[0x31, 0xc0, 0xc3],             // 37: case 1
            &[0x31, 0xc0, 0xc3],             // 3a: case 2
            &[0x31, 0xc0, 0xc3],             // 3d: case 3
            &[0x31, 0xc0, 0xc3],             // 40: default
            &[0xc3],                         // 43: callee
        ]
        .concat();
        let rel = |c: u32, base: u32| ((TEXT_RVA + c) as i32 - base as i32).to_le_bytes();
        let mut data: Vec<u8> = (0..4).flat_map(|_| rel(0x1a, t1)).collect();
        let cases = [0x34u32, 0x37, 0x3a, 0x3d];
        data.extend(cases.iter().flat_map(|&c| rel(c, t2)));
        data.extend_from_slice(&rel(0x40, t2));
        let analysis = analyze(&code, &data, &[0], &[]);
        assert_eq!(analysis.jump_tables[&(TEXT + 0x18)].targets, [TEXT + 0x1a; 4]);
        let table = analysis
            .jump_tables
            .get(&(TEXT + 0x32))
            .expect("switch behind the later merge resolves");
        assert!(!table.inferred);
        assert_eq!(table.targets, cases.map(|c| TEXT + u64::from(c)).to_vec());
    }

    /// The index lives in a copy made before a merge (`mov r9d, eax` ahead of a branch around a
    /// block that reaches the guard): the guard on the source bounds the copy.
    #[test]
    fn resolves_switch_on_register_copy_made_before_a_merge() {
        let lea_disp = (i64::from(DATA_RVA) - i64::from(TEXT_RVA + 0x14)) as i32;
        let code = [
            &[0x41, 0x89, 0xc1][..],   // 0: mov r9d, eax
            &[0x85, 0xc9],             // 3: test ecx, ecx
            &[0x74, 0x01],             // 5: je 8
            &[0x90],                   // 7: nop
            &[0x83, 0xf8, 0x03],       // 8: cmp eax, 3
            &[0x77, 0x1d],             // b: ja 2a
            &[0x48, 0x8d, 0x15],       // d: lea rdx, [table]
            &lea_disp.to_le_bytes(),   //
            &[0x4e, 0x63, 0x0c, 0x8a], // 14: movsxd r9, [rdx+r9*4]
            &[0x49, 0x01, 0xd1],       // 18: add r9, rdx
            &[0x41, 0xff, 0xe1],       // 1b: jmp r9
            &[0x31, 0xc0, 0xc3],       // 1e: case 0
            &[0x31, 0xc0, 0xc3],       // 21: case 1
            &[0x31, 0xc0, 0xc3],       // 24: case 2
            &[0x31, 0xc0, 0xc3],       // 27: case 3
            &[0x31, 0xc0, 0xc3],       // 2a: default
        ]
        .concat();
        let cases = [0x1eu32, 0x21, 0x24, 0x27];
        let data: Vec<u8> = cases
            .iter()
            .flat_map(|&c| ((TEXT_RVA + c) as i32 - DATA_RVA as i32).to_le_bytes())
            .collect();
        let analysis = analyze(&code, &data, &[0], &[]);
        let table = analysis
            .jump_tables
            .get(&(TEXT + 0x1b))
            .expect("copy-indexed switch resolves");
        assert!(!table.inferred);
        assert_eq!(table.targets, cases.map(|c| TEXT + u64::from(c)).to_vec());
    }
}
