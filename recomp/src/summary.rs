//! Deterministic JSON summary of a discovery run (addresses and counts only, never code bytes).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use crate::cfg::{CallTarget, Edge};
use crate::code_image::CodeImage;
use crate::discover::Analysis;
use crate::lift::MemoryModel;
use crate::lift::memory::WindowCheck;
use crate::pipeline::{PipelineReport, ShardRecord};
use crate::reloc::RelocKind;

const EDGE_KINDS: [&str; 7] = [
    "fallthrough",
    "taken",
    "jump",
    "tail_call",
    "indirect_jump",
    "return",
    "jump_table",
];

const RELOC_KINDS: [RelocKind; 11] = [
    RelocKind::CallTarget,
    RelocKind::BranchTarget,
    RelocKind::RipData,
    RelocKind::RipAddress,
    RelocKind::CallSlot,
    RelocKind::JumpSlot,
    RelocKind::AbsoluteImmediate,
    RelocKind::AbsoluteDisplacement,
    RelocKind::JumpTable,
    RelocKind::JumpTableIndex,
    RelocKind::JumpTableEntry,
];

fn edge_index(edge: &Edge) -> usize {
    match edge {
        Edge::Fallthrough(_) => 0,
        Edge::Taken(_) => 1,
        Edge::Jump(_) => 2,
        Edge::TailCall(_) => 3,
        Edge::IndirectJump { .. } => 4,
        Edge::Return => 5,
        Edge::JumpTable(_) => 6,
    }
}

fn hex_list(values: impl IntoIterator<Item = u64>) -> String {
    let items: Vec<String> = values.into_iter().map(|v| format!("\"0x{v:x}\"")).collect();
    format!("[{}]", items.join(", "))
}

/// Render the summary as pretty-printed JSON with a trailing newline.
pub fn render_summary_json(image: &CodeImage<'_>, analysis: &Analysis) -> String {
    let mut edge_counts = [0usize; EDGE_KINDS.len()];
    let mut block_count = 0;
    let mut instruction_count = 0;
    let mut diagnostic_count = 0;
    // `jmp reg` / `jmp [computed]` sites left unresolved (blocks shared by several functions are
    // counted once).
    let mut unresolved_jumps = BTreeSet::new();
    let mut resume_targets = 0;
    let mut resume_functions = 0;
    for function in analysis.functions.values() {
        resume_targets += function.resume_targets.len();
        resume_functions += usize::from(!function.resume_targets.is_empty());
        block_count += function.blocks.len();
        instruction_count += function.instruction_count();
        diagnostic_count += function.diagnostics().count();
        for block in function.blocks.values() {
            for edge in &block.successors {
                edge_counts[edge_index(edge)] += 1;
                if *edge == (Edge::IndirectJump { slot: None }) {
                    unresolved_jumps.insert(block.start);
                }
            }
        }
    }

    let mut out = String::new();
    out.push_str("{\n");
    let _ = writeln!(out, "  \"image_base\": \"0x{:x}\",", image.base());
    let _ = writeln!(out, "  \"image_end\": \"0x{:x}\",", image.end());
    let entry_points: BTreeSet<u64> = image.entry_points.iter().copied().collect();
    let _ = writeln!(
        out,
        "  \"entry_points\": {},",
        hex_list(entry_points)
    );
    let _ = writeln!(out, "  \"function_count\": {},", analysis.functions.len());
    let _ = writeln!(out, "  \"block_count\": {block_count},");
    let _ = writeln!(out, "  \"instruction_count\": {instruction_count},");
    let _ = writeln!(out, "  \"diagnostic_count\": {diagnostic_count},");
    let _ = writeln!(
        out,
        "  \"pdata\": {{\"functions\": {}, \"ranges\": {}}},",
        image.function_table.function_count(),
        image.function_table.ranges().len()
    );
    let _ = writeln!(
        out,
        "  \"jump_tables\": {{\"resolved\": {}, \"unresolved\": {}}},",
        analysis.jump_tables.len(),
        unresolved_jumps.len()
    );
    let eh = &image.eh.stats;
    let _ = writeln!(
        out,
        "  \"eh\": {{\"c_specific\": {}, \"cxx3\": {}, \"cxx4\": {}, \"gs_check\": {}, \
         \"unknown_handler\": {}, \"parsed\": {}, \"funclets\": {}, \"metadata_resume_targets\": {}, \
         \"diagnostics\": {}, \"setjmp_slots\": {}, \"resume_targets\": {resume_targets}, \
         \"functions_with_resume_targets\": {resume_functions}}},",
        eh.c_specific,
        eh.cxx3,
        eh.cxx4,
        eh.gs_check,
        eh.unknown,
        eh.parsed,
        eh.funclets,
        eh.resume_targets,
        eh.diagnostics,
        image.eh.setjmp_slots.len(),
    );

    out.push_str("  \"edges\": {");
    for (i, (name, count)) in EDGE_KINDS.iter().zip(edge_counts).enumerate() {
        let sep = if i == 0 { "" } else { ", " };
        let _ = write!(out, "{sep}\"{name}\": {count}");
    }
    out.push_str("},\n");

    out.push_str("  \"relocations\": {");
    for (i, kind) in RELOC_KINDS.iter().enumerate() {
        let count = analysis
            .relocations
            .entries()
            .iter()
            .filter(|r| r.kind == *kind)
            .count();
        let sep = if i == 0 { "" } else { ", " };
        let _ = write!(out, "{sep}\"{}\": {count}", kind.as_str());
    }
    out.push_str("},\n");

    out.push_str("  \"functions\": [");
    for (i, function) in analysis.functions.values().enumerate() {
        out.push_str(if i == 0 { "\n" } else { ",\n" });
        let indirect_calls = function
            .calls()
            .filter(|c| matches!(c.target, CallTarget::Indirect { .. }))
            .count();
        let mut sorted_diags: Vec<_> = function.diagnostics().collect();
        sorted_diags.sort_by(|a, b| {
            a.site
                .cmp(&b.site)
                .then_with(|| a.address.cmp(&b.address))
                .then_with(|| a.kind.as_str().cmp(b.kind.as_str()))
        });
        let diagnostics: Vec<String> = sorted_diags
            .into_iter()
            .map(|d| {
                format!(
                    "{{\"site\": \"0x{:x}\", \"address\": \"0x{:x}\", \"kind\": \"{}\"}}",
                    d.site,
                    d.address,
                    d.kind.as_str()
                )
            })
            .collect();
        let tail_calls: BTreeSet<u64> = function.tail_calls().collect();
        let _ = write!(
            out,
            "    {{\"entry\": \"0x{:x}\", \"blocks\": {}, \"instructions\": {}, \"callees\": {}, \
             \"tail_calls\": {}, \"indirect_calls\": {indirect_calls}, \"diagnostics\": [{}]}}",
            function.entry,
            function.blocks.len(),
            function.instruction_count(),
            hex_list(function.callees.iter().copied()),
            hex_list(tail_calls),
            diagnostics.join(", ")
        );
    }
    if analysis.functions.is_empty() {
        out.push_str("]\n");
    } else {
        out.push_str("\n  ]\n");
    }
    out.push_str("}\n");
    out
}

/// Computes `numerator / denominator` as a percentage in `[0.0, 100.0]`, rounded half-up to
/// two decimal places using exact integer arithmetic.
///
/// Zero-division returns `0.0`, and numerators exceeding `denominator` are clamped to `100.0`.
pub fn percentage_2dp(numerator: usize, denominator: usize) -> f64 {
    percentage_u64_2dp(numerator as u64, denominator as u64)
}

/// Computes `numerator / denominator` as a percentage in `[0.0, 100.0]` for `u64` byte/count
/// totals, rounded half-up to two decimal places using exact `u128` arithmetic.
///
/// Zero-division returns `0.0`, and numerators exceeding `denominator` are clamped to `100.0`.
pub fn percentage_u64_2dp(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        return 0.0;
    }
    let clamped = u128::from(numerator.min(denominator));
    let denom = u128::from(denominator);
    let basis_points = (clamped * 20_000 + denom) / (denom * 2);
    basis_points as f64 / 100.0
}

fn ratio_u128_2dp(numerator: u128, denominator: u128) -> f64 {
    if denominator == 0 || numerator == 0 {
        return 0.0;
    }
    let scaled = match numerator
        .checked_mul(200)
        .and_then(|v| v.checked_add(denominator))
        .and_then(|v| denominator.checked_mul(2).map(|d2| v / d2))
    {
        Some(centi) => centi,
        None => {
            // Overflow-safe fallback for extreme u128 values: split quotient and remainder.
            let q = numerator / denominator;
            let r = numerator % denominator;
            let frac = (r.saturating_mul(200).saturating_add(denominator))
                / denominator.saturating_mul(2).max(1);
            q.saturating_mul(100).saturating_add(frac)
        }
    };
    scaled as f64 / 100.0
}

/// Computes an unclamped non-negative ratio `numerator / denominator`, rounded half-up to two
/// decimal places using exact `u128` arithmetic.
///
/// Zero-division always returns `0.0` (never `NaN` or infinity).
pub fn ratio_2dp(numerator: u64, denominator: u64) -> f64 {
    ratio_u128_2dp(u128::from(numerator), u128::from(denominator))
}

/// Converts a byte count to MiB (`bytes / 1_048_576`), rounded half-up to two decimal places.
pub fn mib_2dp(bytes: u64) -> f64 {
    ratio_2dp(bytes, 1_048_576)
}

/// Sanitizes a symbol, DLL, mnemonic, or source label so host directory paths, trailing
/// directory markers, relative traversal segments, and control characters can never leak into
/// CLI or program reports.
pub fn sanitize_label(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.ends_with('/') || trimmed.ends_with('\\') {
        return "<unknown>".to_string();
    }
    let tail = trimmed
        .rsplit(['/', '\\'])
        .find(|segment| !segment.is_empty())
        .unwrap_or("");
    let tail = if tail.len() >= 2
        && tail.as_bytes()[0].is_ascii_alphabetic()
        && tail.as_bytes()[1] == b':'
    {
        &tail[2..]
    } else {
        tail
    };

    let mut out = String::with_capacity(tail.len());
    let mut prev_space = false;
    for ch in tail.chars() {
        if matches!(ch, '\r' | '\n' | '\t' | ' ') {
            if !prev_space && !out.is_empty() {
                out.push(' ');
                prev_space = true;
            }
        } else if !ch.is_control() {
            out.push(ch);
            prev_space = false;
        }
    }
    let cleaned = out.trim();
    if cleaned.is_empty() || cleaned == "." || cleaned == ".." {
        "<unknown>".to_string()
    } else {
        cleaned.to_string()
    }
}

/// Returns `true` if `text` appears to contain a leaked Windows, Unix/macOS, or `file://` host
/// filesystem path.
pub fn contains_host_path(text: &str) -> bool {
    let bytes = text.as_bytes();

    // 1. file:// URI scheme (case-insensitive)
    if text.to_ascii_lowercase().contains("file://") {
        return true;
    }

    // 2. Windows drive-letter paths: `C:\`, `C:/`, and JSON-escaped `C:\\`, `C:\/`
    for i in 0..bytes.len().saturating_sub(2) {
        if bytes[i].is_ascii_alphabetic()
            && bytes[i + 1] == b':'
            && matches!(bytes[i + 2], b'\\' | b'/')
        {
            let preceded_by_ident = i > 0 && bytes[i - 1].is_ascii_alphanumeric();
            if !preceded_by_ident {
                return true;
            }
        }
    }

    // 3. Windows UNC and extended-length paths (`\\?\`, `\\.\`, JSON-escaped `\\\\`,
    //    or raw `\\host\share`).
    if text.contains(r"\\?\") || text.contains(r"\\.\") || text.contains(r"\\\\") {
        return true;
    }
    let mut i = 0;
    while i + 3 < bytes.len() {
        if bytes[i] == b'\\' && bytes[i + 1] == b'\\' && bytes[i + 2].is_ascii_alphanumeric() {
            let preceded_by_bs = i > 0 && bytes[i - 1] == b'\\';
            if !preceded_by_bs {
                let mut j = i + 2;
                while j < bytes.len()
                    && (bytes[j].is_ascii_alphanumeric() || matches!(bytes[j], b'.' | b'-' | b'_'))
                {
                    j += 1;
                }
                if j > i + 2 && j < bytes.len() && bytes[j] == b'\\' {
                    return true;
                }
            }
        }
        i += 1;
    }

    // 4. Common Unix/macOS host root prefixes at token boundaries
    const UNIX_ROOTS: &[&str] = &[
        "/Users/",
        "/home/",
        "/root/",
        "/tmp/",
        "/var/",
        "/opt/",
        "/etc/",
        "/mnt/",
        "/Volumes/",
        "/private/",
        "/proc/",
        "/sys/",
    ];
    for root in UNIX_ROOTS {
        let mut search_from = 0;
        while let Some(rel_idx) = text[search_from..].find(root) {
            let idx = search_from + rel_idx;
            let boundary_ok = idx == 0
                || matches!(
                    bytes[idx - 1],
                    b' ' | b'\t'
                        | b'\n'
                        | b'\r'
                        | b'"'
                        | b'\''
                        | b'`'
                        | b'('
                        | b'['
                        | b'='
                        | b','
                        | b':'
                );
            if boundary_ok {
                return true;
            }
            search_from = idx + root.len();
        }
    }

    false
}

fn json_string(input: &str) -> String {
    let mut out = String::with_capacity(input.len() + 2);
    out.push('"');
    for ch in input.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0C}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Sanitized, counts-only coverage summary for discovery, lifting, and import binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CoverageSummary {
    pub total_functions: usize,
    pub lifted_functions: usize,
    pub total_instructions: usize,
    pub decoded_instructions: usize,
    pub total_imports: usize,
    pub bound_imports: usize,
}

impl CoverageSummary {
    /// Derives a discovery-stage coverage summary from [`Analysis`].
    ///
    /// Functions without decode diagnostics count as lifted/clean, and `total_instructions`
    /// equals decoded instructions plus decode diagnostic sites.
    pub fn from_analysis(analysis: &Analysis) -> Self {
        let total_functions = analysis.functions.len();
        let mut lifted_functions = 0usize;
        let mut decoded_instructions = 0usize;
        let mut diagnostic_count = 0usize;

        for function in analysis.functions.values() {
            let fn_diags = function.diagnostics().count();
            if fn_diags == 0 {
                lifted_functions += 1;
            }
            decoded_instructions = decoded_instructions.saturating_add(function.instruction_count());
            diagnostic_count = diagnostic_count.saturating_add(fn_diags);
        }

        Self {
            total_functions,
            lifted_functions,
            total_instructions: decoded_instructions.saturating_add(diagnostic_count),
            decoded_instructions,
            total_imports: 0,
            bound_imports: 0,
        }
    }

    /// Derives a full pipeline coverage summary from [`PipelineReport`].
    pub fn from_pipeline_report(report: &PipelineReport) -> Self {
        let total_instructions = report.lift.instructions;
        let decoded_instructions = total_instructions.saturating_sub(report.lift.unsupported_total());
        let bound_imports = report.imports.bound.saturating_add(report.imports.guest);
        Self {
            total_functions: report.discovery.functions,
            lifted_functions: report.lift.functions,
            total_instructions,
            decoded_instructions,
            total_imports: report.imports.total,
            bound_imports: bound_imports.min(report.imports.total),
        }
    }

    /// Attaches import-binding counts to an existing summary.
    pub fn with_imports(mut self, total_imports: usize, bound_imports: usize) -> Self {
        self.total_imports = total_imports;
        self.bound_imports = bound_imports;
        self
    }

    /// Percentage of discovered functions lifted cleanly (`0.00..=100.00`).
    pub fn lifted_function_pct(&self) -> f64 {
        percentage_2dp(self.lifted_functions, self.total_functions)
    }

    /// Percentage of instructions decoded/supported (`0.00..=100.00`).
    pub fn decoded_instruction_pct(&self) -> f64 {
        percentage_2dp(self.decoded_instructions, self.total_instructions)
    }

    /// Percentage of imports bound to shims or guest modules (`0.00..=100.00`).
    pub fn bound_import_pct(&self) -> f64 {
        percentage_2dp(self.bound_imports, self.total_imports)
    }

    /// Count of functions not yet lifted.
    pub fn unlifted_functions(&self) -> usize {
        self.total_functions.saturating_sub(self.lifted_functions)
    }

    /// Count of undecoded or unsupported instructions.
    pub fn undecoded_instructions(&self) -> usize {
        self.total_instructions.saturating_sub(self.decoded_instructions)
    }

    /// Count of imports remaining unresolved.
    pub fn unresolved_imports(&self) -> usize {
        self.total_imports.saturating_sub(self.bound_imports)
    }

    /// Formats a crisp, sanitized one-line progress ratio with zero host paths or raw bytes.
    pub fn format_one_line(&self) -> String {
        format!(
            "functions {}/{} ({:.2}%) | instructions {}/{} ({:.2}%) | imports {}/{} ({:.2}%, {} unresolved)",
            self.lifted_functions,
            self.total_functions,
            self.lifted_function_pct(),
            self.decoded_instructions,
            self.total_instructions,
            self.decoded_instruction_pct(),
            self.bound_imports,
            self.total_imports,
            self.bound_import_pct(),
            self.unresolved_imports(),
        )
    }
}

/// One sanitized `(label, count)` entry in a deterministic top-N breakdown.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RankedCount {
    pub label: String,
    pub count: usize,
}

fn aggregate_and_rank<I, S>(entries: I) -> Vec<RankedCount>
where
    I: IntoIterator<Item = (S, usize)>,
    S: AsRef<str>,
{
    let mut totals: BTreeMap<String, usize> = BTreeMap::new();
    for (raw_label, count) in entries {
        if count == 0 {
            continue;
        }
        let label = sanitize_label(raw_label.as_ref());
        let slot = totals.entry(label).or_insert(0);
        *slot = slot.saturating_add(count);
    }

    let mut ranked: Vec<RankedCount> = totals
        .into_iter()
        .map(|(label, count)| RankedCount { label, count })
        .collect();
    ranked.sort_by(|a, b| {
        b.count
            .cmp(&a.count)
            .then_with(|| a.label.to_ascii_lowercase().cmp(&b.label.to_ascii_lowercase()))
            .then_with(|| a.label.cmp(&b.label))
    });
    ranked
}

/// Aggregates, sanitizes, and deterministically ranks up to `limit` non-zero `(label, count)`
/// entries by count descending, then label ascending (case-insensitive, then exact).
pub fn rank_top_n<I, S>(entries: I, limit: usize) -> Vec<RankedCount>
where
    I: IntoIterator<Item = (S, usize)>,
    S: AsRef<str>,
{
    if limit == 0 {
        return Vec::new();
    }
    let mut ranked = aggregate_and_rank(entries);
    ranked.truncate(limit);
    ranked
}

/// Formats a deterministic one-line top-N summary (`"a (5), b (2), +1 more"` or `"none"`).
pub fn format_top_n<I, S>(entries: I, limit: usize) -> String
where
    I: IntoIterator<Item = (S, usize)>,
    S: AsRef<str>,
{
    let ranked = aggregate_and_rank(entries);
    if ranked.is_empty() || limit == 0 {
        return "none".to_string();
    }

    let shown = ranked.len().min(limit);
    let mut out = String::new();
    for (idx, item) in ranked.iter().take(shown).enumerate() {
        if idx > 0 {
            out.push_str(", ");
        }
        let _ = write!(out, "{} ({})", item.label, item.count);
    }
    if ranked.len() > shown {
        let _ = write!(out, ", +{} more", ranked.len() - shown);
    }
    out
}

/// Deterministically ranks decode diagnostic kinds across an [`Analysis`].
pub fn diagnostic_breakdown(analysis: &Analysis) -> Vec<RankedCount> {
    let entries = analysis
        .functions
        .values()
        .flat_map(|f| f.diagnostics())
        .map(|d| (d.kind.as_str(), 1usize));
    aggregate_and_rank(entries)
}

/// Deterministically ranks up to `limit` DLLs with unresolved imports from `(dll, bound, unresolved)` tuples.
pub fn unresolved_import_breakdown(
    by_dll: &[(String, usize, usize)],
    limit: usize,
) -> Vec<RankedCount> {
    rank_top_n(
        by_dll
            .iter()
            .map(|(dll, _bound, unresolved)| (dll.as_str(), *unresolved)),
        limit,
    )
}

/// Formats a deterministic one-line top-N unresolved-import breakdown by DLL.
pub fn format_unresolved_imports_top_n(by_dll: &[(String, usize, usize)], limit: usize) -> String {
    format_top_n(
        by_dll
            .iter()
            .map(|(dll, _bound, unresolved)| (dll.as_str(), *unresolved)),
        limit,
    )
}

/// Default maximum shard skew ratio (`max / mean`) considered balanced for Wasm64 fast-boot
/// streaming and parallel compilation.
pub const DEFAULT_FAST_BOOT_MAX_SKEW: f64 = 1.50;

/// Default per-shard byte budget (4 MiB) for fast Wasm64 browser streaming and instantiation.
pub const DEFAULT_FAST_BOOT_SHARD_BUDGET_BYTES: u64 = 4 * 1024 * 1024;

/// Schema identifier emitted by [`FastBootSummary::render_json`].
pub const FAST_BOOT_SUMMARY_SCHEMA: &str = "playgta6-recomp-fast-boot-summary";

/// Schema version emitted by [`FastBootSummary::render_json`].
pub const FAST_BOOT_SUMMARY_SCHEMA_VERSION: u32 = 1;

fn memory_model_label(model: &MemoryModel) -> &'static str {
    match model {
        MemoryModel::Base(_) => "base",
        MemoryModel::Windowed(w) => match w.check {
            WindowCheck::Outline => "window-outline",
            _ if w.contiguous_runs => "window-runs",
            _ => "window",
        },
        MemoryModel::PageTable { .. } => "page-table",
    }
}

/// Sanitized per-shard measurement for deterministic Wasm64 fast-boot summaries.
///
/// Contains counts, sizes, and a sanitized shard file label only (never host directory paths).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FastBootShardEntry {
    pub index: usize,
    pub label: String,
    pub functions: usize,
    pub instructions: usize,
    pub ll_bytes: u64,
    pub obj_bytes: Option<u64>,
}

impl FastBootShardEntry {
    /// Creates a sanitized shard summary entry.
    pub fn new(
        index: usize,
        raw_label: &str,
        functions: usize,
        instructions: usize,
        ll_bytes: u64,
        obj_bytes: Option<u64>,
    ) -> Self {
        Self {
            index,
            label: sanitize_label(raw_label),
            functions,
            instructions,
            ll_bytes,
            obj_bytes,
        }
    }

    /// Derives a sanitized shard entry from a pipeline [`ShardRecord`].
    pub fn from_shard_record(record: &ShardRecord) -> Self {
        let raw_label = record
            .obj
            .as_ref()
            .map_or(record.ll.path.as_str(), |obj| obj.path.as_str());
        Self::new(
            record.index,
            raw_label,
            record.functions,
            record.instructions,
            record.ll.size,
            record.obj.as_ref().map(|obj| obj.size),
        )
    }

    /// Effective artifact byte size used for fast-boot budgets (`obj_bytes` if compiled, else `ll_bytes`).
    pub fn effective_bytes(&self) -> u64 {
        self.obj_bytes.unwrap_or(self.ll_bytes)
    }
}

fn sort_shards_deterministic(shards: &mut [FastBootShardEntry]) {
    shards.sort_by(|a, b| {
        b.effective_bytes()
            .cmp(&a.effective_bytes())
            .then_with(|| b.instructions.cmp(&a.instructions))
            .then_with(|| b.functions.cmp(&a.functions))
            .then_with(|| a.index.cmp(&b.index))
            .then_with(|| a.label.to_ascii_lowercase().cmp(&b.label.to_ascii_lowercase()))
            .then_with(|| a.label.cmp(&b.label))
    });
}

/// Aggregated Wasm64 shard distribution and budget statistics with zero-division guards.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FastBootShardStats {
    pub shard_count: usize,
    pub compiled_obj_shards: usize,
    pub total_functions: usize,
    pub total_instructions: usize,
    pub min_instructions: usize,
    pub max_instructions: usize,
    pub total_ll_bytes: u64,
    pub total_obj_bytes: u64,
    pub total_effective_bytes: u64,
    pub min_effective_bytes: u64,
    pub max_effective_bytes: u64,
    pub shard_budget_bytes: u64,
    pub oversized_shards: usize,
}

impl FastBootShardStats {
    /// Computes aggregate shard metrics from `entries` against `shard_budget_bytes`.
    pub fn from_entries(entries: &[FastBootShardEntry], shard_budget_bytes: u64) -> Self {
        if entries.is_empty() {
            return Self {
                shard_budget_bytes,
                ..Self::default()
            };
        }

        let mut stats = Self {
            shard_count: entries.len(),
            min_instructions: usize::MAX,
            min_effective_bytes: u64::MAX,
            shard_budget_bytes,
            ..Self::default()
        };

        for shard in entries {
            let eff = shard.effective_bytes();
            stats.total_functions = stats.total_functions.saturating_add(shard.functions);
            stats.total_instructions = stats.total_instructions.saturating_add(shard.instructions);
            stats.min_instructions = stats.min_instructions.min(shard.instructions);
            stats.max_instructions = stats.max_instructions.max(shard.instructions);
            stats.total_ll_bytes = stats.total_ll_bytes.saturating_add(shard.ll_bytes);
            if let Some(obj) = shard.obj_bytes {
                stats.compiled_obj_shards = stats.compiled_obj_shards.saturating_add(1);
                stats.total_obj_bytes = stats.total_obj_bytes.saturating_add(obj);
            }
            stats.total_effective_bytes = stats.total_effective_bytes.saturating_add(eff);
            stats.min_effective_bytes = stats.min_effective_bytes.min(eff);
            stats.max_effective_bytes = stats.max_effective_bytes.max(eff);
            if shard_budget_bytes > 0 && eff > shard_budget_bytes {
                stats.oversized_shards = stats.oversized_shards.saturating_add(1);
            }
        }

        stats
    }

    /// Percentage of shards compiled to object files (`0.00..=100.00`, `0.0` when empty).
    pub fn compiled_obj_pct(&self) -> f64 {
        percentage_2dp(self.compiled_obj_shards, self.shard_count)
    }

    /// Mean functions per shard (`0.0` when `shard_count == 0`).
    pub fn avg_functions_per_shard(&self) -> f64 {
        ratio_2dp(self.total_functions as u64, self.shard_count as u64)
    }

    /// Mean instructions per shard (`0.0` when `shard_count == 0`).
    pub fn avg_instructions_per_shard(&self) -> f64 {
        ratio_2dp(self.total_instructions as u64, self.shard_count as u64)
    }

    /// Mean `.ll` byte size per shard (`0.0` when `shard_count == 0`).
    pub fn avg_ll_bytes(&self) -> f64 {
        ratio_2dp(self.total_ll_bytes, self.shard_count as u64)
    }

    /// Mean `.o` byte size per compiled shard (`0.0` when `compiled_obj_shards == 0`).
    pub fn avg_obj_bytes(&self) -> f64 {
        ratio_2dp(self.total_obj_bytes, self.compiled_obj_shards as u64)
    }

    /// Mean effective byte size per shard (`0.0` when `shard_count == 0`).
    pub fn avg_effective_bytes(&self) -> f64 {
        ratio_2dp(self.total_effective_bytes, self.shard_count as u64)
    }

    /// Instruction skew ratio `max / mean = (max * count) / total`, rounded half-up to 2dp.
    ///
    /// Returns `0.0` when `shard_count == 0` or `total_instructions == 0`.
    pub fn instruction_skew_ratio(&self) -> f64 {
        if self.shard_count == 0 || self.total_instructions == 0 {
            return 0.0;
        }
        let num = (self.max_instructions as u128).saturating_mul(self.shard_count as u128);
        let den = self.total_instructions as u128;
        ratio_u128_2dp(num, den)
    }

    /// Effective byte-size skew ratio `max / mean = (max * count) / total`, rounded half-up to 2dp.
    ///
    /// Returns `0.0` when `shard_count == 0` or `total_effective_bytes == 0`.
    pub fn size_skew_ratio(&self) -> f64 {
        if self.shard_count == 0 || self.total_effective_bytes == 0 {
            return 0.0;
        }
        let num = u128::from(self.max_effective_bytes).saturating_mul(self.shard_count as u128);
        let den = u128::from(self.total_effective_bytes);
        ratio_u128_2dp(num, den)
    }
}

/// Deterministic, strictly sanitized Wasm64 fast-boot build summary.
///
/// Emits counts, ratios, sizes, and sanitized labels only — never host paths or binary bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FastBootSummary {
    pub module: String,
    pub target: String,
    pub memory_model: String,
    pub indirect_dispatch: String,
    pub coverage: CoverageSummary,
    pub shards: FastBootShardStats,
    pub input_bytes: u64,
    pub guest_image_bytes: u64,
    pub registration_bytes: u64,
    pub total_output_bytes: u64,
    pub heaviest_shards: Vec<FastBootShardEntry>,
    pub unresolved_imports: Vec<RankedCount>,
    pub unsupported_mnemonics: Vec<RankedCount>,
}

impl FastBootSummary {
    /// Creates an empty, all-zero fast-boot summary for a sanitized module label.
    pub fn empty(module: &str) -> Self {
        Self {
            module: sanitize_label(module),
            target: "wasm64".to_string(),
            memory_model: "base".to_string(),
            indirect_dispatch: "table".to_string(),
            coverage: CoverageSummary::default(),
            shards: FastBootShardStats {
                shard_budget_bytes: DEFAULT_FAST_BOOT_SHARD_BUDGET_BYTES,
                ..FastBootShardStats::default()
            },
            input_bytes: 0,
            guest_image_bytes: 0,
            registration_bytes: 0,
            total_output_bytes: 0,
            heaviest_shards: Vec::new(),
            unresolved_imports: Vec::new(),
            unsupported_mnemonics: Vec::new(),
        }
    }

    /// Derives a sanitized fast-boot summary from a [`PipelineReport`] using
    /// [`DEFAULT_FAST_BOOT_SHARD_BUDGET_BYTES`].
    pub fn from_pipeline_report(report: &PipelineReport) -> Self {
        Self::from_pipeline_report_with_budget(report, DEFAULT_FAST_BOOT_SHARD_BUDGET_BYTES)
    }

    /// Derives a sanitized fast-boot summary from a [`PipelineReport`] with a custom per-shard
    /// byte budget.
    pub fn from_pipeline_report_with_budget(
        report: &PipelineReport,
        shard_budget_bytes: u64,
    ) -> Self {
        let mut shard_entries: Vec<FastBootShardEntry> = report
            .lift
            .shards
            .iter()
            .map(FastBootShardEntry::from_shard_record)
            .collect();
        let shards = FastBootShardStats::from_entries(&shard_entries, shard_budget_bytes);
        sort_shards_deterministic(&mut shard_entries);

        let registration_bytes = report
            .lift
            .registration_obj
            .as_ref()
            .map_or(report.lift.registration_ll.size, |obj| obj.size);
        let total_output_bytes = report
            .files
            .iter()
            .fold(0u64, |acc, f| acc.saturating_add(f.size));

        let unresolved_imports = unresolved_import_breakdown(&report.imports.by_dll, usize::MAX);
        let unsupported_mnemonics = rank_top_n(
            report
                .lift
                .unsupported
                .iter()
                .map(|(m, &count)| (m.as_str(), count)),
            usize::MAX,
        );

        Self {
            module: sanitize_label(&report.module),
            target: sanitize_label(&report.target.to_string()),
            memory_model: sanitize_label(memory_model_label(&report.memory)),
            indirect_dispatch: sanitize_label(report.indirect_dispatch.as_str()),
            coverage: CoverageSummary::from_pipeline_report(report),
            shards,
            input_bytes: report.input_size,
            guest_image_bytes: report.image.container_bytes,
            registration_bytes,
            total_output_bytes,
            heaviest_shards: shard_entries,
            unresolved_imports,
            unsupported_mnemonics,
        }
    }

    /// Replaces the shard list (sanitizing all labels, recomputing stats, and sorting
    /// deterministically).
    pub fn with_shards(
        mut self,
        mut shards: Vec<FastBootShardEntry>,
        shard_budget_bytes: u64,
    ) -> Self {
        for shard in &mut shards {
            shard.label = sanitize_label(&shard.label);
        }
        self.shards = FastBootShardStats::from_entries(&shards, shard_budget_bytes);
        sort_shards_deterministic(&mut shards);
        self.heaviest_shards = shards;
        self
    }

    /// Attaches a sanitized, deterministically ranked unresolved-import breakdown.
    pub fn with_unresolved_imports<I, S>(mut self, entries: I) -> Self
    where
        I: IntoIterator<Item = (S, usize)>,
        S: AsRef<str>,
    {
        self.unresolved_imports = rank_top_n(entries, usize::MAX);
        self
    }

    /// Attaches a sanitized, deterministically ranked unsupported-mnemonic breakdown.
    pub fn with_unsupported_mnemonics<I, S>(mut self, entries: I) -> Self
    where
        I: IntoIterator<Item = (S, usize)>,
        S: AsRef<str>,
    {
        self.unsupported_mnemonics = rank_top_n(entries, usize::MAX);
        self
    }

    /// Total payload bytes required to boot: guest image + registration module + effective shards.
    pub fn boot_payload_bytes(&self) -> u64 {
        self.guest_image_bytes
            .saturating_add(self.registration_bytes)
            .saturating_add(self.shards.total_effective_bytes)
    }

    /// Estimates cold streaming time in milliseconds at `bandwidth_mbps` megabits per second,
    /// rounded half-up to 2dp.
    ///
    /// Zero-division returns `0.0` when `bandwidth_mbps == 0` or `boot_payload_bytes() == 0`.
    pub fn estimated_download_ms(&self, bandwidth_mbps: u64) -> f64 {
        let bytes = self.boot_payload_bytes();
        if bandwidth_mbps == 0 || bytes == 0 {
            return 0.0;
        }
        // 1 Mbps = 1_000_000 bits/s = 125_000 bytes/s = 125 bytes/ms.
        let bytes_per_ms = u128::from(bandwidth_mbps).saturating_mul(125);
        ratio_u128_2dp(u128::from(bytes), bytes_per_ms)
    }

    /// Returns `true` when there are no oversized shards and both instruction and size skew
    /// ratios are within `max_skew`.
    ///
    /// Returns `false` if `max_skew` is non-finite or `< 1.0`.
    pub fn is_fast_boot_balanced(&self, max_skew: f64) -> bool {
        if !max_skew.is_finite() || max_skew < 1.0 {
            return false;
        }
        self.shards.oversized_shards == 0
            && self.shards.instruction_skew_ratio() <= max_skew
            && self.shards.size_skew_ratio() <= max_skew
    }

    /// Renders a crisp, sanitized one-line summary with deterministic ordering.
    pub fn render_one_line(&self) -> String {
        let module = sanitize_label(&self.module);
        let target = sanitize_label(&self.target);
        let memory = sanitize_label(&self.memory_model);
        let dispatch = sanitize_label(&self.indirect_dispatch);
        format!(
            "[{}] target={target} memory={memory} dispatch={dispatch} | {} | shards {} ({:.2}% obj, instr_skew {:.2}x, size_skew {:.2}x, {} oversized) | boot_payload {:.2} MiB",
            module,
            self.coverage.format_one_line(),
            self.shards.shard_count,
            self.shards.compiled_obj_pct(),
            self.shards.instruction_skew_ratio(),
            self.shards.size_skew_ratio(),
            self.shards.oversized_shards,
            mib_2dp(self.boot_payload_bytes()),
        )
    }

    /// Renders a deterministic, strictly sanitized multi-line text summary with a trailing newline.
    pub fn render_text(&self, top_n: usize) -> String {
        let module = sanitize_label(&self.module);
        let target = sanitize_label(&self.target);
        let memory = sanitize_label(&self.memory_model);
        let dispatch = sanitize_label(&self.indirect_dispatch);

        let mut shards_sorted: Vec<FastBootShardEntry> = self
            .heaviest_shards
            .iter()
            .cloned()
            .map(|mut s| {
                s.label = sanitize_label(&s.label);
                s
            })
            .collect();
        sort_shards_deterministic(&mut shards_sorted);

        let top_unresolved = format_top_n(
            self.unresolved_imports
                .iter()
                .map(|r| (r.label.as_str(), r.count)),
            top_n,
        );
        let top_unsupported = format_top_n(
            self.unsupported_mnemonics
                .iter()
                .map(|r| (r.label.as_str(), r.count)),
            top_n,
        );

        let mut out = String::with_capacity(512);
        let _ = writeln!(out, "Fast-Boot Summary: {module}");
        let _ = writeln!(
            out,
            "Target: {target} | Memory: {memory} | Dispatch: {dispatch}"
        );
        let _ = writeln!(out, "Coverage: {}", self.coverage.format_one_line());
        let _ = writeln!(
            out,
            "Shards: {} total, {} obj ({:.2}%), functions {} (avg {:.2}), instructions {} (min {}, max {}, avg {:.2}, skew {:.2}x), bytes {} effective (min {}, max {}, avg {:.2}, skew {:.2}x), oversized {} (budget {} bytes)",
            self.shards.shard_count,
            self.shards.compiled_obj_shards,
            self.shards.compiled_obj_pct(),
            self.shards.total_functions,
            self.shards.avg_functions_per_shard(),
            self.shards.total_instructions,
            self.shards.min_instructions,
            self.shards.max_instructions,
            self.shards.avg_instructions_per_shard(),
            self.shards.instruction_skew_ratio(),
            self.shards.total_effective_bytes,
            self.shards.min_effective_bytes,
            self.shards.max_effective_bytes,
            self.shards.avg_effective_bytes(),
            self.shards.size_skew_ratio(),
            self.shards.oversized_shards,
            self.shards.shard_budget_bytes,
        );
        let boot_bytes = self.boot_payload_bytes();
        let _ = writeln!(
            out,
            "Payload: boot {} bytes ({:.2} MiB) | guest_image {} | registration {} | input {} | total_output {}",
            boot_bytes,
            mib_2dp(boot_bytes),
            self.guest_image_bytes,
            self.registration_bytes,
            self.input_bytes,
            self.total_output_bytes,
        );

        let shown_shards = shards_sorted.len().min(top_n);
        if shown_shards == 0 {
            let _ = writeln!(out, "Heaviest Shards (top {top_n}): none");
        } else {
            let mut items = Vec::with_capacity(shown_shards);
            for shard in shards_sorted.iter().take(shown_shards) {
                items.push(format!(
                    "#{} {} ({} bytes, {} instrs, {} fns)",
                    shard.index,
                    shard.label,
                    shard.effective_bytes(),
                    shard.instructions,
                    shard.functions
                ));
            }
            if shards_sorted.len() > shown_shards {
                items.push(format!("+{} more", shards_sorted.len() - shown_shards));
            }
            let _ = writeln!(out, "Heaviest Shards (top {top_n}): {}", items.join(", "));
        }
        let _ = writeln!(out, "Top Unresolved Imports (top {top_n}): {top_unresolved}");
        let _ = writeln!(
            out,
            "Top Unsupported Mnemonics (top {top_n}): {top_unsupported}"
        );
        out
    }

    /// Renders a deterministic, RFC 8259 compliant, strictly sanitized JSON summary with a
    /// trailing newline.
    pub fn render_json(&self, top_n: usize) -> String {
        let module = sanitize_label(&self.module);
        let target = sanitize_label(&self.target);
        let memory = sanitize_label(&self.memory_model);
        let dispatch = sanitize_label(&self.indirect_dispatch);

        let mut shards_sorted: Vec<FastBootShardEntry> = self
            .heaviest_shards
            .iter()
            .cloned()
            .map(|mut s| {
                s.label = sanitize_label(&s.label);
                s
            })
            .collect();
        sort_shards_deterministic(&mut shards_sorted);

        let unresolved = rank_top_n(
            self.unresolved_imports
                .iter()
                .map(|r| (r.label.as_str(), r.count)),
            top_n,
        );
        let unsupported = rank_top_n(
            self.unsupported_mnemonics
                .iter()
                .map(|r| (r.label.as_str(), r.count)),
            top_n,
        );

        let boot_bytes = self.boot_payload_bytes();

        let mut out = String::with_capacity(1024);
        out.push_str("{\n");
        let _ = writeln!(out, "  \"schema\": {},", json_string(FAST_BOOT_SUMMARY_SCHEMA));
        let _ = writeln!(
            out,
            "  \"schema_version\": {},",
            FAST_BOOT_SUMMARY_SCHEMA_VERSION
        );
        let _ = writeln!(out, "  \"module\": {},", json_string(&module));
        let _ = writeln!(out, "  \"target\": {},", json_string(&target));
        let _ = writeln!(out, "  \"memory_model\": {},", json_string(&memory));
        let _ = writeln!(out, "  \"indirect_dispatch\": {},", json_string(&dispatch));
        let _ = writeln!(out, "  \"coverage\": {{");
        let _ = writeln!(
            out,
            "    \"total_functions\": {},",
            self.coverage.total_functions
        );
        let _ = writeln!(
            out,
            "    \"lifted_functions\": {},",
            self.coverage.lifted_functions
        );
        let _ = writeln!(
            out,
            "    \"lifted_function_pct\": {:.2},",
            self.coverage.lifted_function_pct()
        );
        let _ = writeln!(
            out,
            "    \"total_instructions\": {},",
            self.coverage.total_instructions
        );
        let _ = writeln!(
            out,
            "    \"decoded_instructions\": {},",
            self.coverage.decoded_instructions
        );
        let _ = writeln!(
            out,
            "    \"decoded_instruction_pct\": {:.2},",
            self.coverage.decoded_instruction_pct()
        );
        let _ = writeln!(
            out,
            "    \"total_imports\": {},",
            self.coverage.total_imports
        );
        let _ = writeln!(
            out,
            "    \"bound_imports\": {},",
            self.coverage.bound_imports
        );
        let _ = writeln!(
            out,
            "    \"unresolved_imports\": {},",
            self.coverage.unresolved_imports()
        );
        let _ = writeln!(
            out,
            "    \"bound_import_pct\": {:.2}",
            self.coverage.bound_import_pct()
        );
        out.push_str("  },\n");

        let _ = writeln!(out, "  \"shards\": {{");
        let _ = writeln!(out, "    \"count\": {},", self.shards.shard_count);
        let _ = writeln!(
            out,
            "    \"compiled_obj_shards\": {},",
            self.shards.compiled_obj_shards
        );
        let _ = writeln!(
            out,
            "    \"compiled_obj_pct\": {:.2},",
            self.shards.compiled_obj_pct()
        );
        let _ = writeln!(
            out,
            "    \"total_functions\": {},",
            self.shards.total_functions
        );
        let _ = writeln!(
            out,
            "    \"avg_functions\": {:.2},",
            self.shards.avg_functions_per_shard()
        );
        let _ = writeln!(
            out,
            "    \"total_instructions\": {},",
            self.shards.total_instructions
        );
        let _ = writeln!(
            out,
            "    \"min_instructions\": {},",
            self.shards.min_instructions
        );
        let _ = writeln!(
            out,
            "    \"max_instructions\": {},",
            self.shards.max_instructions
        );
        let _ = writeln!(
            out,
            "    \"avg_instructions\": {:.2},",
            self.shards.avg_instructions_per_shard()
        );
        let _ = writeln!(
            out,
            "    \"instruction_skew_ratio\": {:.2},",
            self.shards.instruction_skew_ratio()
        );
        let _ = writeln!(
            out,
            "    \"total_ll_bytes\": {},",
            self.shards.total_ll_bytes
        );
        let _ = writeln!(
            out,
            "    \"total_obj_bytes\": {},",
            self.shards.total_obj_bytes
        );
        let _ = writeln!(
            out,
            "    \"total_effective_bytes\": {},",
            self.shards.total_effective_bytes
        );
        let _ = writeln!(
            out,
            "    \"min_effective_bytes\": {},",
            self.shards.min_effective_bytes
        );
        let _ = writeln!(
            out,
            "    \"max_effective_bytes\": {},",
            self.shards.max_effective_bytes
        );
        let _ = writeln!(
            out,
            "    \"avg_effective_bytes\": {:.2},",
            self.shards.avg_effective_bytes()
        );
        let _ = writeln!(
            out,
            "    \"size_skew_ratio\": {:.2},",
            self.shards.size_skew_ratio()
        );
        let _ = writeln!(
            out,
            "    \"shard_budget_bytes\": {},",
            self.shards.shard_budget_bytes
        );
        let _ = writeln!(
            out,
            "    \"oversized_shards\": {}",
            self.shards.oversized_shards
        );
        out.push_str("  },\n");

        let _ = writeln!(out, "  \"payload\": {{");
        let _ = writeln!(out, "    \"input_bytes\": {},", self.input_bytes);
        let _ = writeln!(out, "    \"guest_image_bytes\": {},", self.guest_image_bytes);
        let _ = writeln!(
            out,
            "    \"registration_bytes\": {},",
            self.registration_bytes
        );
        let _ = writeln!(out, "    \"boot_payload_bytes\": {boot_bytes},");
        let _ = writeln!(
            out,
            "    \"boot_payload_mib\": {:.2},",
            mib_2dp(boot_bytes)
        );
        let _ = writeln!(
            out,
            "    \"total_output_bytes\": {}",
            self.total_output_bytes
        );
        out.push_str("  },\n");

        let shown_shards = shards_sorted.len().min(top_n);
        if shown_shards == 0 {
            out.push_str("  \"heaviest_shards\": [],\n");
        } else {
            out.push_str("  \"heaviest_shards\": [\n");
            for (idx, shard) in shards_sorted.iter().take(shown_shards).enumerate() {
                let comma = if idx + 1 < shown_shards { "," } else { "" };
                let obj_val = match shard.obj_bytes {
                    Some(v) => v.to_string(),
                    None => "null".to_string(),
                };
                let _ = writeln!(
                    out,
                    "    {{\"index\": {}, \"label\": {}, \"functions\": {}, \"instructions\": {}, \"ll_bytes\": {}, \"obj_bytes\": {obj_val}, \"effective_bytes\": {}}}{comma}",
                    shard.index,
                    json_string(&shard.label),
                    shard.functions,
                    shard.instructions,
                    shard.ll_bytes,
                    shard.effective_bytes(),
                );
            }
            out.push_str("  ],\n");
        }

        write_ranked_json(&mut out, "top_unresolved_imports", &unresolved, true);
        write_ranked_json(&mut out, "top_unsupported_mnemonics", &unsupported, false);
        out.push_str("}\n");
        out
    }
}

fn write_ranked_json(out: &mut String, key: &str, items: &[RankedCount], trailing_comma: bool) {
    let comma_end = if trailing_comma { "," } else { "" };
    if items.is_empty() {
        let _ = writeln!(out, "  \"{key}\": []{comma_end}");
        return;
    }
    let _ = writeln!(out, "  \"{key}\": [");
    for (idx, item) in items.iter().enumerate() {
        let comma = if idx + 1 < items.len() { "," } else { "" };
        let _ = writeln!(
            out,
            "    {{\"label\": {}, \"count\": {}}}{comma}",
            json_string(&item.label),
            item.count
        );
    }
    let _ = writeln!(out, "  ]{comma_end}");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::code_image::CodeSection;
    use crate::discover::discover;

    #[test]
    fn summary_is_deterministic_and_has_no_code_bytes() {
        let code = [
            0xe8, 0x06, 0x00, 0x00, 0x00, // 0: call 0xb
            0xff, 0x25, 0x00, 0x00, 0x00, 0x00, // 5: jmp [rip+0] (slot at 0xb)
            0xc3, // b: ret
        ];
        let image = CodeImage::new(
            0x1000,
            vec![CodeSection::new(0x1000, &code, true)],
            vec![0x1000],
        );
        let first = render_summary_json(&image, &discover(&image));
        let second = render_summary_json(&image, &discover(&image));
        assert_eq!(first, second);
        assert_eq!(
            first,
            concat!(
                "{\n",
                "  \"image_base\": \"0x1000\",\n",
                "  \"image_end\": \"0x100c\",\n",
                "  \"entry_points\": [\"0x1000\"],\n",
                "  \"function_count\": 2,\n",
                "  \"block_count\": 2,\n",
                "  \"instruction_count\": 3,\n",
                "  \"diagnostic_count\": 0,\n",
                "  \"pdata\": {\"functions\": 0, \"ranges\": 0},\n",
                "  \"jump_tables\": {\"resolved\": 0, \"unresolved\": 0},\n",
                "  \"eh\": {\"c_specific\": 0, \"cxx3\": 0, \"cxx4\": 0, \"gs_check\": 0, ",
                "\"unknown_handler\": 0, \"parsed\": 0, \"funclets\": 0, \"metadata_resume_targets\": 0, ",
                "\"diagnostics\": 0, \"setjmp_slots\": 0, \"resume_targets\": 0, ",
                "\"functions_with_resume_targets\": 0},\n",
                "  \"edges\": {\"fallthrough\": 0, \"taken\": 0, \"jump\": 0, \"tail_call\": 0, ",
                "\"indirect_jump\": 1, \"return\": 1, \"jump_table\": 0},\n",
                "  \"relocations\": {\"call_target\": 1, \"branch_target\": 0, \"rip_data\": 0, ",
                "\"rip_address\": 0, \"call_slot\": 0, \"jump_slot\": 1, \"absolute_immediate\": 0, ",
                "\"absolute_displacement\": 0, \"jump_table\": 0, \"jump_table_index\": 0, ",
                "\"jump_table_entry\": 0},\n",
                "  \"functions\": [\n",
                "    {\"entry\": \"0x1000\", \"blocks\": 1, \"instructions\": 2, \"callees\": [\"0x100b\"], ",
                "\"tail_calls\": [], \"indirect_calls\": 0, \"diagnostics\": []},\n",
                "    {\"entry\": \"0x100b\", \"blocks\": 1, \"instructions\": 1, \"callees\": [], ",
                "\"tail_calls\": [], \"indirect_calls\": 0, \"diagnostics\": []}\n",
                "  ]\n",
                "}\n",
            )
        );
    }

    #[test]
    fn percentage_handles_zero_division_clamping_and_rounding_precision() {
        // Zero-division boundaries
        assert_eq!(percentage_2dp(0, 0), 0.0);
        assert_eq!(percentage_2dp(42, 0), 0.0);
        assert_eq!(percentage_2dp(0, 100), 0.0);

        // Exact endpoints and clamping above 100%
        assert_eq!(percentage_2dp(10, 10), 100.0);
        assert_eq!(percentage_2dp(15, 10), 100.0);
        assert_eq!(percentage_2dp(usize::MAX, usize::MAX), 100.0);

        // Rounding precision (half-up at 2 decimal places)
        assert_eq!(percentage_2dp(1, 3), 33.33);
        assert_eq!(percentage_2dp(2, 3), 66.67);
        assert_eq!(percentage_2dp(1, 6), 16.67);
        assert_eq!(percentage_2dp(5, 6), 83.33);
        assert_eq!(percentage_2dp(1, 8), 12.5);
        assert_eq!(percentage_2dp(3, 800), 0.38);
        assert_eq!(percentage_2dp(1, 20_000), 0.01);
        assert_eq!(percentage_2dp(1, 40_000), 0.0);
        assert_eq!(percentage_2dp(9_999, 10_000), 99.99);
        assert_eq!(percentage_2dp(19_999, 20_000), 100.0);
    }

    #[test]
    fn coverage_summary_ratios_and_one_line_formatting() {
        let empty = CoverageSummary::default();
        assert_eq!(empty.lifted_function_pct(), 0.0);
        assert_eq!(empty.decoded_instruction_pct(), 0.0);
        assert_eq!(empty.bound_import_pct(), 0.0);
        assert_eq!(empty.unlifted_functions(), 0);
        assert_eq!(empty.undecoded_instructions(), 0);
        assert_eq!(empty.unresolved_imports(), 0);
        assert_eq!(
            empty.format_one_line(),
            "functions 0/0 (0.00%) | instructions 0/0 (0.00%) | imports 0/0 (0.00%, 0 unresolved)"
        );

        let partial = CoverageSummary {
            total_functions: 3,
            lifted_functions: 2,
            total_instructions: 600,
            decoded_instructions: 599,
            total_imports: 12,
            bound_imports: 9,
        };
        assert_eq!(partial.lifted_function_pct(), 66.67);
        assert_eq!(partial.decoded_instruction_pct(), 99.83);
        assert_eq!(partial.bound_import_pct(), 75.0);
        assert_eq!(partial.unlifted_functions(), 1);
        assert_eq!(partial.undecoded_instructions(), 1);
        assert_eq!(partial.unresolved_imports(), 3);
        assert_eq!(
            partial.format_one_line(),
            "functions 2/3 (66.67%) | instructions 599/600 (99.83%) | imports 9/12 (75.00%, 3 unresolved)"
        );
    }

    #[test]
    fn coverage_summary_from_analysis_counts_clean_and_diagnostic_functions() {
        // Function 1 (0x1000): valid `ret`
        // Function 2 (0x1001): jumps to unmapped address 0x9000 -> produces a diagnostic
        let code = [
            0xc3, // 0x1000: ret
            0xe9, 0xfa, 0x7f, 0x00, 0x00, // 0x1001: jmp 0x9000 (unmapped)
        ];
        let image = CodeImage::new(
            0x1000,
            vec![CodeSection::new(0x1000, &code, true)],
            vec![0x1000, 0x1001],
        );
        let analysis = discover(&image);
        let summary = CoverageSummary::from_analysis(&analysis).with_imports(10, 7);

        assert_eq!(summary.total_functions, 2);
        assert_eq!(summary.lifted_functions, 1);
        assert_eq!(summary.lifted_function_pct(), 50.0);
        assert_eq!(summary.decoded_instructions, 2);
        assert_eq!(summary.total_instructions, 3);
        assert_eq!(summary.decoded_instruction_pct(), 66.67);
        assert_eq!(summary.bound_import_pct(), 70.0);

        let diags = diagnostic_breakdown(&analysis);
        assert_eq!(
            diags,
            vec![RankedCount {
                label: "unmapped".to_string(),
                count: 1,
            }]
        );
    }

    #[test]
    fn sanitize_label_strips_host_paths_and_control_characters() {
        assert_eq!(
            sanitize_label(r"C:\Users\dev\Games\GTA6\game_scarlett_master.exe"),
            "game_scarlett_master.exe"
        );
        assert_eq!(
            sanitize_label("/home/runner/work/build/d3d12_xs.dll"),
            "d3d12_xs.dll"
        );
        assert_eq!(sanitize_label("D:local_stub.dll"), "local_stub.dll");
        assert_eq!(
            sanitize_label("KERNEL32.dll!VirtualAlloc"),
            "KERNEL32.dll!VirtualAlloc"
        );
        assert_eq!(sanitize_label("  vmovdqu\r\n\t "), "vmovdqu");
        assert_eq!(sanitize_label("   /\\  \x07 "), "<unknown>");
    }

    #[test]
    fn top_n_ranking_and_formatting_are_deterministic_regardless_of_input_order() {
        let input_a = vec![
            ("zeta.dll", 2),
            ("Alpha.dll", 5),
            ("beta.dll", 5),
            ("zero.dll", 0),
            ("alpha.dll", 5),
            (r"C:\secret\dir\gamma.dll", 1),
            ("zeta.dll", 3), // aggregates with earlier zeta.dll -> 5
        ];
        let input_b: Vec<(&str, usize)> = input_a.iter().copied().rev().collect();

        let ranked_a = rank_top_n(input_a.clone(), 3);
        let ranked_b = rank_top_n(input_b.clone(), 3);
        assert_eq!(ranked_a, ranked_b);
        assert_eq!(
            ranked_a,
            vec![
                RankedCount {
                    label: "Alpha.dll".to_string(),
                    count: 5,
                },
                RankedCount {
                    label: "alpha.dll".to_string(),
                    count: 5,
                },
                RankedCount {
                    label: "beta.dll".to_string(),
                    count: 5,
                },
            ]
        );

        assert_eq!(
            format_top_n(input_a.clone(), 3),
            "Alpha.dll (5), alpha.dll (5), beta.dll (5), +2 more"
        );
        assert_eq!(
            format_top_n(input_b, 5),
            "Alpha.dll (5), alpha.dll (5), beta.dll (5), zeta.dll (5), gamma.dll (1)"
        );
        assert_eq!(format_top_n(input_a, 0), "none");
        assert_eq!(format_top_n(Vec::<(&str, usize)>::new(), 5), "none");
        assert_eq!(format_top_n(vec![("only_zero", 0)], 5), "none");
    }

    #[test]
    fn unresolved_import_breakdown_filters_fully_bound_dlls_and_sanitizes() {
        let by_dll = vec![
            ("KERNEL32.dll".to_string(), 10, 0),
            (r"C:\Xbox\d3d12_xs.dll".to_string(), 2, 4),
            ("bink2scarlett.dll".to_string(), 0, 4),
            ("WS2_32.dll".to_string(), 20, 1),
        ];

        assert_eq!(
            unresolved_import_breakdown(&by_dll, 2),
            vec![
                RankedCount {
                    label: "bink2scarlett.dll".to_string(),
                    count: 4,
                },
                RankedCount {
                    label: "d3d12_xs.dll".to_string(),
                    count: 4,
                },
            ]
        );
        assert_eq!(
            format_unresolved_imports_top_n(&by_dll, 2),
            "bink2scarlett.dll (4), d3d12_xs.dll (4), +1 more"
        );
        assert_eq!(
            format_unresolved_imports_top_n(&[("KERNEL32.dll".to_string(), 5, 0)], 5),
            "none"
        );
    }

    #[test]
    fn render_summary_json_canonicalizes_unsorted_and_duplicate_entry_points() {
        let code = [
            0xc3, // 0x1000: ret
            0xc3, // 0x1001: ret
            0xc3, // 0x1002: ret
        ];
        let image_a = CodeImage::new(
            0x1000,
            vec![CodeSection::new(0x1000, &code, true)],
            vec![0x1002, 0x1000, 0x1001, 0x1000],
        );
        let image_b = CodeImage::new(
            0x1000,
            vec![CodeSection::new(0x1000, &code, true)],
            vec![0x1000, 0x1001, 0x1002],
        );

        let json_a = render_summary_json(&image_a, &discover(&image_a));
        let json_b = render_summary_json(&image_b, &discover(&image_b));
        assert_eq!(json_a, json_b);
        assert!(json_a.contains("\"entry_points\": [\"0x1000\", \"0x1001\", \"0x1002\"],"));
        assert!(!contains_host_path(&json_a));
    }

    #[test]
    fn u64_percentage_ratio_and_mib_zero_division_and_rounding_guards() {
        // Zero-division guards never produce NaN or infinity
        assert_eq!(percentage_u64_2dp(0, 0), 0.0);
        assert_eq!(percentage_u64_2dp(u64::MAX, 0), 0.0);
        assert_eq!(ratio_2dp(0, 0), 0.0);
        assert_eq!(ratio_2dp(u64::MAX, 0), 0.0);
        assert_eq!(ratio_2dp(0, 12345), 0.0);
        assert_eq!(mib_2dp(0), 0.0);

        // Extreme values remain finite and overflow-free
        assert_eq!(percentage_u64_2dp(u64::MAX, u64::MAX), 100.0);
        assert!(ratio_2dp(u64::MAX, 1).is_finite());
        assert_eq!(ratio_2dp(u64::MAX, u64::MAX), 1.0);
        assert!(mib_2dp(u64::MAX).is_finite());

        // Exact half-up 2dp rounding
        assert_eq!(percentage_u64_2dp(1, 3), 33.33);
        assert_eq!(percentage_u64_2dp(2, 3), 66.67);
        assert_eq!(percentage_u64_2dp(500, 100), 100.0);
        assert_eq!(ratio_2dp(3, 2), 1.5);
        assert_eq!(ratio_2dp(5, 3), 1.67);
        assert_eq!(ratio_2dp(1, 200), 0.01);
        assert_eq!(ratio_2dp(1, 201), 0.0);
        assert_eq!(mib_2dp(1_048_576), 1.0);
        assert_eq!(mib_2dp(1_572_864), 1.5);
    }

    #[test]
    fn sanitize_label_and_contains_host_path_strict_adversarial_cases() {
        // Trailing directory slashes and traversal markers must never leak directory names
        assert_eq!(sanitize_label(r"C:\Users\alice\GTA6\"), "<unknown>");
        assert_eq!(sanitize_label("/home/runner/work/playgta6web/"), "<unknown>");
        assert_eq!(sanitize_label(r"\\buildbox\share\"), "<unknown>");
        assert_eq!(sanitize_label("file:///Users/bob/Desktop/"), "<unknown>");
        assert_eq!(sanitize_label("."), "<unknown>");
        assert_eq!(sanitize_label(".."), "<unknown>");
        assert_eq!(sanitize_label("../../secret/.."), "<unknown>");
        assert_eq!(sanitize_label(r"C:\Users\alice\."), "<unknown>");
        assert_eq!(sanitize_label("D:"), "<unknown>");

        // UNC, extended-length, and file:// URI paths strip to clean file names
        assert_eq!(
            sanitize_label(r"\\?\C:\Users\alice\Games\shard-00001.o"),
            "shard-00001.o"
        );
        assert_eq!(
            sanitize_label(r"\\buildbox\share\artifacts\game.exe"),
            "game.exe"
        );
        assert_eq!(
            sanitize_label("FILE:///Users/bob/builds/d3d12_xs.dll"),
            "d3d12_xs.dll"
        );
        assert_eq!(
            sanitize_label("line1\r\n\tline2\x00\x1b"),
            "line1 line2"
        );

        // Positive detections for every host-path class
        let leaked_samples = [
            r"C:\Users\alice\game.exe",
            "D:/Xbox/GTA6/main.exe",
            r#"{"label": "C:\\Users\\alice\\game.exe"}"#,
            r#"{"label": "E:\/builds\/shard.o"}"#,
            r"\\?\C:\secret\file.bin",
            r"\\.\pipe\debug",
            r"\\buildbox\share\file.o",
            r#"{"path": "\\\\server\\share\\file.o"}"#,
            "file:///home/runner/game.exe",
            "FILE://localhost/C:/game.exe",
            "/Users/dev/project/out.ll",
            "path=/home/runner/work/repo/file",
            "\"/root/secret.bin\"",
            "(/tmp/dump.bin)",
            "[/var/log/recomp.log]",
            "/opt/llvm/bin/llc",
            "/etc/passwd",
            "/mnt/c/Games/GTA6",
            "/Volumes/XboxSSD/game.exe",
            "/private/var/folders/tmp.o",
            "/proc/self/maps",
            "/sys/kernel/debug",
        ];
        for sample in leaked_samples {
            assert!(
                contains_host_path(sample),
                "expected contains_host_path to detect host path in: {sample}"
            );
        }

        // Clean sanitized reports and labels must not trigger false positives
        let clean_samples = [
            "game_scarlett_master.exe",
            "KERNEL32.dll!VirtualAlloc",
            "shard-00000.o",
            "lift/shard-00000.o",
            "https://example.com/assets/shard.wasm",
            "target=wasm64 memory=window-outline dispatch=shard",
            "{\"module\": \"main\", \"boot_payload_mib\": 12.50}",
            "prefix/home/not-at-root",
        ];
        for sample in clean_samples {
            assert!(
                !contains_host_path(sample),
                "unexpected false positive in contains_host_path for: {sample}"
            );
        }
    }

    #[test]
    fn fast_boot_summary_empty_and_zero_stats_have_no_nan_or_div_by_zero() {
        let empty = FastBootSummary::empty("main");
        assert_eq!(empty.shards.shard_count, 0);
        assert_eq!(empty.shards.compiled_obj_pct(), 0.0);
        assert_eq!(empty.shards.avg_functions_per_shard(), 0.0);
        assert_eq!(empty.shards.avg_instructions_per_shard(), 0.0);
        assert_eq!(empty.shards.avg_ll_bytes(), 0.0);
        assert_eq!(empty.shards.avg_obj_bytes(), 0.0);
        assert_eq!(empty.shards.avg_effective_bytes(), 0.0);
        assert_eq!(empty.shards.instruction_skew_ratio(), 0.0);
        assert_eq!(empty.shards.size_skew_ratio(), 0.0);
        assert_eq!(empty.boot_payload_bytes(), 0);
        assert_eq!(empty.estimated_download_ms(0), 0.0);
        assert_eq!(empty.estimated_download_ms(100), 0.0);
        assert!(empty.is_fast_boot_balanced(DEFAULT_FAST_BOOT_MAX_SKEW));
        assert!(!empty.is_fast_boot_balanced(0.5));
        assert!(!empty.is_fast_boot_balanced(f64::NAN));
        assert!(!empty.is_fast_boot_balanced(f64::INFINITY));

        // Shards that exist but have 0 instructions and 0 bytes must still guard skew division
        let zero_shards = FastBootSummary::empty("main").with_shards(
            vec![
                FastBootShardEntry::new(0, "shard-00000.ll", 0, 0, 0, None),
                FastBootShardEntry::new(1, "shard-00001.ll", 0, 0, 0, None),
            ],
            DEFAULT_FAST_BOOT_SHARD_BUDGET_BYTES,
        );
        assert_eq!(zero_shards.shards.shard_count, 2);
        assert_eq!(zero_shards.shards.compiled_obj_pct(), 0.0);
        assert_eq!(zero_shards.shards.avg_obj_bytes(), 0.0);
        assert_eq!(zero_shards.shards.instruction_skew_ratio(), 0.0);
        assert_eq!(zero_shards.shards.size_skew_ratio(), 0.0);
        assert!(zero_shards.is_fast_boot_balanced(DEFAULT_FAST_BOOT_MAX_SKEW));

        let json = empty.render_json(5);
        let text = empty.render_text(5);
        let one_line = empty.render_one_line();
        assert!(!json.contains("NaN") && !json.contains("inf"));
        assert!(!text.contains("NaN") && !text.contains("inf"));
        assert!(!one_line.contains("NaN") && !one_line.contains("inf"));
        assert!(!contains_host_path(&json));
        assert!(!contains_host_path(&text));
        assert!(!contains_host_path(&one_line));
        assert!(json.contains("\"heaviest_shards\": [],"));
        assert!(json.contains("\"top_unresolved_imports\": [],"));
        assert!(json.contains("\"top_unsupported_mnemonics\": []\n}"));
        assert!(text.contains("Heaviest Shards (top 5): none\n"));
        assert!(text.contains("Top Unresolved Imports (top 5): none\n"));
        assert!(text.contains("Top Unsupported Mnemonics (top 5): none\n"));
    }

    #[test]
    fn fast_boot_summary_json_and_text_are_deterministic_regardless_of_input_order() {
        let shards_a = vec![
            FastBootShardEntry::new(2, "shard-00002.o", 100, 10_000, 800_000, Some(200_000)),
            FastBootShardEntry::new(0, "shard-00000.o", 250, 30_000, 2_000_000, Some(600_000)),
            FastBootShardEntry::new(3, "shard-00003.o", 100, 10_000, 750_000, Some(200_000)),
            FastBootShardEntry::new(1, "shard-00001.o", 150, 10_000, 900_000, Some(200_000)),
        ];
        let mut shards_b = shards_a.clone();
        shards_b.reverse();

        let unresolved_a = vec![
            ("ws2_32.dll", 2),
            ("d3d12_xs.dll", 5),
            ("bink2scarlett.dll", 5),
            ("zero.dll", 0),
        ];
        let unresolved_b: Vec<(&str, usize)> = unresolved_a.iter().copied().rev().collect();

        let unsupported_a = vec![("vpdpbusd", 3), ("vfmadd231ps", 7), ("xsavec", 3)];
        let unsupported_b: Vec<(&str, usize)> = unsupported_a.iter().copied().rev().collect();

        let mut summary_a = FastBootSummary::empty("main")
            .with_shards(shards_a, 500_000)
            .with_unresolved_imports(unresolved_a)
            .with_unsupported_mnemonics(unsupported_a);
        summary_a.coverage = CoverageSummary {
            total_functions: 600,
            lifted_functions: 594,
            total_instructions: 60_000,
            decoded_instructions: 59_987,
            total_imports: 50,
            bound_imports: 38,
        };
        summary_a.input_bytes = 5_000_000;
        summary_a.guest_image_bytes = 2_000_000;
        summary_a.registration_bytes = 100_000;
        summary_a.total_output_bytes = 3_300_000;

        let mut summary_b = FastBootSummary::empty("main")
            .with_shards(shards_b, 500_000)
            .with_unresolved_imports(unresolved_b)
            .with_unsupported_mnemonics(unsupported_b);
        summary_b.coverage = summary_a.coverage;
        summary_b.input_bytes = summary_a.input_bytes;
        summary_b.guest_image_bytes = summary_a.guest_image_bytes;
        summary_b.registration_bytes = summary_a.registration_bytes;
        summary_b.total_output_bytes = summary_a.total_output_bytes;

        // Byte-identical JSON, text, and one-line summaries across input permutations
        assert_eq!(summary_a, summary_b);
        assert_eq!(summary_a.render_json(3), summary_b.render_json(3));
        assert_eq!(summary_a.render_text(3), summary_b.render_text(3));
        assert_eq!(summary_a.render_one_line(), summary_b.render_one_line());

        // Exact skew and budget metrics:
        // instructions: [30_000, 10_000, 10_000, 10_000], sum=60_000, mean=15_000 -> skew=2.00x
        // effective bytes: [600_000, 200_000, 200_000, 200_000], sum=1_200_000, mean=300_000 -> skew=2.00x
        assert_eq!(summary_a.shards.compiled_obj_pct(), 100.0);
        assert_eq!(summary_a.shards.avg_functions_per_shard(), 150.0);
        assert_eq!(summary_a.shards.avg_instructions_per_shard(), 15_000.0);
        assert_eq!(summary_a.shards.instruction_skew_ratio(), 2.0);
        assert_eq!(summary_a.shards.size_skew_ratio(), 2.0);
        assert_eq!(summary_a.shards.oversized_shards, 1); // 600_000 > 500_000
        assert!(!summary_a.is_fast_boot_balanced(DEFAULT_FAST_BOOT_MAX_SKEW));

        // Payload & download time: boot = 2_000_000 + 100_000 + 1_200_000 = 3_300_000 bytes
        // At 100 Mbps (12_500 bytes/ms): 3_300_000 / 12_500 = 264.00 ms
        assert_eq!(summary_a.boot_payload_bytes(), 3_300_000);
        assert_eq!(summary_a.estimated_download_ms(100), 264.0);
        assert_eq!(summary_a.estimated_download_ms(0), 0.0);

        // Deterministic tie-breaking order of heaviest shards:
        // #0 (600k), #1 (200k, 10k instr, 150 fns), #2 (200k, 10k instr, 100 fns, idx 2), #3 (idx 3)
        let indices: Vec<usize> = summary_a.heaviest_shards.iter().map(|s| s.index).collect();
        assert_eq!(indices, vec![0, 1, 2, 3]);

        let json = summary_a.render_json(2);
        let text = summary_a.render_text(2);
        assert!(json.contains("\"schema\": \"playgta6-recomp-fast-boot-summary\","));
        assert!(json.contains("\"instruction_skew_ratio\": 2.00,"));
        assert!(json.contains("\"size_skew_ratio\": 2.00,"));
        assert!(json.contains("\"oversized_shards\": 1"));
        assert!(text.contains(
            "Heaviest Shards (top 2): #0 shard-00000.o (600000 bytes, 30000 instrs, 250 fns), #1 shard-00001.o (200000 bytes, 10000 instrs, 150 fns), +2 more\n"
        ));
        assert!(text.contains(
            "Top Unresolved Imports (top 2): bink2scarlett.dll (5), d3d12_xs.dll (5), +1 more\n"
        ));
    }

    #[test]
    fn fast_boot_summary_sanitizes_host_paths_across_all_fields_and_mutated_state() {
        let mut summary = FastBootSummary::empty(r"C:\Users\alice\GTA6\game_scarlett_master.exe")
            .with_shards(
                vec![
                    FastBootShardEntry::new(
                        0,
                        r"\\?\C:\Users\alice\work\lift\shard-00000.o",
                        12,
                        400,
                        40_000,
                        Some(12_000),
                    ),
                    FastBootShardEntry::new(
                        1,
                        "/home/runner/work/playgta6web/work/lift/shard-00001.ll",
                        8,
                        300,
                        30_000,
                        None,
                    ),
                    FastBootShardEntry::new(
                        2,
                        "file:///Volumes/XboxSSD/secret_dir/",
                        4,
                        100,
                        10_000,
                        None,
                    ),
                ],
                DEFAULT_FAST_BOOT_SHARD_BUDGET_BYTES,
            )
            .with_unresolved_imports([
                (r"\\buildbox\share\sys\d3d12_xs.dll", 4),
                ("/Users/bob/sdk/xaudio2_9.dll\r\n", 2),
                (r"C:\Windows\System32\", 1),
            ])
            .with_unsupported_mnemonics([
                ("C:\\secret\\vgatherdps\x07", 3),
                ("../../bad/..", 1),
            ]);

        // Defense-in-depth: even if public fields are mutated with host paths after construction,
        // rendered outputs must still strip them cleanly.
        summary.target = r"C:\Users\alice\toolchains\wasm64".to_string();
        summary.memory_model = "/tmp/override/window-outline".to_string();
        summary.indirect_dispatch = "file:///opt/mode/shard".to_string();
        if let Some(shard0) = summary.heaviest_shards.iter_mut().find(|s| s.index == 0) {
            shard0.label = r"D:\Private\User\shard-00000.o".to_string();
        }

        let json = summary.render_json(10);
        let text = summary.render_text(10);
        let one_line = summary.render_one_line();

        for output in [&json, &text, &one_line] {
            assert!(
                !contains_host_path(output),
                "host path leaked in rendered output:\n{output}"
            );
            for forbidden in [
                "alice",
                "runner",
                "buildbox",
                "XboxSSD",
                "secret_dir",
                "Private",
                "System32",
                "file://",
            ] {
                assert!(
                    !output.contains(forbidden),
                    "forbidden host token '{forbidden}' leaked in output:\n{output}"
                );
            }
        }

        assert!(json.contains("\"module\": \"game_scarlett_master.exe\","));
        assert!(json.contains("\"target\": \"wasm64\","));
        assert!(json.contains("\"memory_model\": \"window-outline\","));
        assert!(json.contains("\"indirect_dispatch\": \"shard\","));
        assert!(json.contains("\"label\": \"shard-00000.o\""));
        assert!(json.contains("\"label\": \"shard-00001.ll\""));
        assert!(json.contains("\"label\": \"<unknown>\""));
        assert!(json.contains("\"label\": \"d3d12_xs.dll\""));
        assert!(json.contains("\"label\": \"xaudio2_9.dll\""));
        assert!(json.contains("\"label\": \"vgatherdps\""));
    }

    #[test]
    fn fast_boot_summary_from_pipeline_report_aggregates_and_sanitizes() {
        use crate::image::ImageOptions;
        use crate::lift::target::LiftTarget;
        use crate::pipeline::refine::{CodePointerOutcome, SlotCheck};
        use crate::pipeline::{
            DiscoveryStats, ImageStats, ImportStats, IndirectDispatch, LiftStats, OutputFile,
            PdataStats,
        };

        let mut unsupported = BTreeMap::new();
        unsupported.insert(r"C:\host\vmovntdq".to_string(), 2);
        unsupported.insert("xgetbv".to_string(), 5);

        let report = PipelineReport {
            module: r"C:\Users\dev\builds\main".to_string(),
            input_size: 4_000_000,
            input_hash: 0x1234_5678,
            image: ImageStats {
                image_base: 0x14000_0000,
                entry_point: 0x14000_1000,
                size_of_image: 0x20_0000,
                sections: Vec::new(),
                dir64: 0,
                highlow: 0,
                container_bytes: 1_500_000,
                diagnostics: BTreeMap::new(),
            },
            pdata: PdataStats {
                records: 10,
                valid: 10,
                functions: 10,
                diagnostics: BTreeMap::new(),
            },
            discovery: DiscoveryStats {
                rounds: 1,
                entry_points: 1,
                functions: 10,
                blocks: 25,
                instructions: 1_000,
                jump_tables_resolved: 0,
                jump_tables_unresolved: 0,
                jump_tables_inferred: 0,
                jump_tables_unresolved_by_reason: Vec::new(),
                trap_blocks: 0,
                invalid_blocks: 0,
                no_return_calls: 0,
                relocations: Vec::new(),
                diagnostics: Vec::new(),
            },
            no_return: Vec::new(),
            code_pointers: CodePointerOutcome {
                rounds: 1,
                round_limit_hit: false,
                no_return_thunks: Vec::new(),
                pointers: Vec::new(),
            },
            pdata_present: true,
            slots: SlotCheck::default(),
            imports: ImportStats {
                total: 20,
                bound: 14,
                guest: 2,
                unresolved: 4,
                by_dll: vec![
                    (r"C:\Xbox\d3d12_xs.dll".to_string(), 10, 3),
                    ("KERNEL32.dll".to_string(), 6, 1),
                ],
                headers: Vec::new(),
                shims: 14,
                catalogue_diagnostics: 0,
            },
            lift: LiftStats {
                shard_size: 2048,
                shard_instructions: 65_536,
                shards: vec![
                    ShardRecord {
                        index: 0,
                        first: 0x14000_1000,
                        last: 0x14000_2000,
                        functions: 6,
                        instructions: 600,
                        unsupported: BTreeMap::new(),
                        ll: OutputFile {
                            path: r"C:\Users\dev\work\lift\shard-00000.ll".to_string(),
                            size: 200_000,
                            hash: 1,
                        },
                        obj: Some(OutputFile {
                            path: r"C:\Users\dev\work\lift\shard-00000.o".to_string(),
                            size: 50_000,
                            hash: 2,
                        }),
                    },
                    ShardRecord {
                        index: 1,
                        first: 0x14000_3000,
                        last: 0x14000_4000,
                        functions: 4,
                        instructions: 400,
                        unsupported: BTreeMap::new(),
                        ll: OutputFile {
                            path: "/home/runner/work/lift/shard-00001.ll".to_string(),
                            size: 160_000,
                            hash: 3,
                        },
                        obj: Some(OutputFile {
                            path: "/home/runner/work/lift/shard-00001.o".to_string(),
                            size: 50_000,
                            hash: 4,
                        }),
                    },
                ],
                functions: 10,
                instructions: 1_000,
                unsupported,
                registration_ll: OutputFile {
                    path: "lift/functions.ll".to_string(),
                    size: 12_000,
                    hash: 5,
                },
                registration_obj: Some(OutputFile {
                    path: "lift/functions.o".to_string(),
                    size: 4_000,
                    hash: 6,
                }),
            },
            llvm: None,
            extra_entries: Vec::new(),
            emit_obj: true,
            image_options: ImageOptions::default(),
            target: LiftTarget::Wasm64,
            memory: MemoryModel::translated_window_outline(),
            indirect_dispatch: IndirectDispatch::Shard,
            files: vec![
                OutputFile {
                    path: "image/guest.pgim".to_string(),
                    size: 1_500_000,
                    hash: 7,
                },
                OutputFile {
                    path: "lift/shard-00000.o".to_string(),
                    size: 50_000,
                    hash: 2,
                },
                OutputFile {
                    path: "lift/shard-00001.o".to_string(),
                    size: 50_000,
                    hash: 4,
                },
                OutputFile {
                    path: "lift/functions.o".to_string(),
                    size: 4_000,
                    hash: 6,
                },
            ],
            function_entries: Vec::new(),
            report: String::new(),
            manifest: String::new(),
        };

        let summary = FastBootSummary::from_pipeline_report(&report);
        assert_eq!(summary.module, "main");
        assert_eq!(summary.target, "wasm64");
        assert_eq!(summary.memory_model, "window-outline");
        assert_eq!(summary.indirect_dispatch, "shard");
        assert_eq!(summary.coverage.lifted_function_pct(), 100.0);
        assert_eq!(summary.coverage.decoded_instruction_pct(), 99.3); // (1000 - 7) / 1000
        assert_eq!(summary.coverage.bound_import_pct(), 80.0); // (14 + 2) / 20
        assert_eq!(summary.shards.shard_count, 2);
        assert_eq!(summary.shards.compiled_obj_pct(), 100.0);
        assert_eq!(summary.shards.instruction_skew_ratio(), 1.2); // 600 / 500
        assert_eq!(summary.shards.size_skew_ratio(), 1.0); // 50_000 / 50_000
        assert_eq!(summary.shards.oversized_shards, 0);
        assert!(summary.is_fast_boot_balanced(DEFAULT_FAST_BOOT_MAX_SKEW));
        assert_eq!(summary.boot_payload_bytes(), 1_604_000);
        assert_eq!(summary.total_output_bytes, 1_604_000);

        let json = summary.render_json(5);
        let text = summary.render_text(5);
        assert!(!contains_host_path(&json));
        assert!(!contains_host_path(&text));
        assert!(json.contains("\"label\": \"xgetbv\", \"count\": 5"));
        assert!(json.contains("\"label\": \"vmovntdq\", \"count\": 2"));
        assert!(json.contains("\"label\": \"d3d12_xs.dll\", \"count\": 3"));
    }
}
