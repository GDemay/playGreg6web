//! Deterministic JSON summary of a discovery run (addresses and counts only, never code bytes).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use crate::cfg::{CallTarget, Edge};
use crate::code_image::CodeImage;
use crate::discover::Analysis;
use crate::pipeline::PipelineReport;
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
    let _ = writeln!(
        out,
        "  \"entry_points\": {},",
        hex_list(image.entry_points.iter().copied())
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
        let diagnostics: Vec<String> = function
            .diagnostics()
            .map(|d| {
                format!(
                    "{{\"site\": \"0x{:x}\", \"address\": \"0x{:x}\", \"kind\": \"{}\"}}",
                    d.site,
                    d.address,
                    d.kind.as_str()
                )
            })
            .collect();
        let _ = write!(
            out,
            "    {{\"entry\": \"0x{:x}\", \"blocks\": {}, \"instructions\": {}, \"callees\": {}, \
             \"tail_calls\": {}, \"indirect_calls\": {indirect_calls}, \"diagnostics\": [{}]}}",
            function.entry,
            function.blocks.len(),
            function.instruction_count(),
            hex_list(function.callees.iter().copied()),
            hex_list(function.tail_calls()),
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
    if denominator == 0 {
        return 0.0;
    }
    let clamped = numerator.min(denominator) as u128;
    let denom = denominator as u128;
    let basis_points = (clamped * 20_000 + denom) / (denom * 2);
    basis_points as f64 / 100.0
}

/// Sanitizes a symbol, DLL, mnemonic, or source label so host directory paths and control
/// characters can never leak into CLI or program reports.
pub fn sanitize_label(raw: &str) -> String {
    let trimmed = raw.trim();
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
    if cleaned.is_empty() {
        "<unknown>".to_string()
    } else {
        cleaned.to_string()
    }
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
}
