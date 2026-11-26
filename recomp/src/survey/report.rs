//! Deterministic text, Markdown, and JSON rendering of a [`Survey`].
//!
//! Only addresses, GUID constants, sanitized source labels, import names, counts, and ratios are
//! emitted. All free-form text fields are strictly redacted against host filesystem paths
//! (Windows drive letters, UNC/extended-length paths, POSIX roots, home/relative traversals, and
//! `file://` URIs), and all collections are canonicalized via `BTreeMap`/`BTreeSet` for
//! permutation-invariant output.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use super::{
    ArgKey, Confidence, KeyReport, SlotReport, SourceReport, Stats, Survey, Totals, evidence_str,
};

/// Replacement token emitted in place of any detected host filesystem path.
pub const REDACTED_PATH: &str = "<redacted-path>";

/// Fallback label when a source name is empty after sanitization.
const UNKNOWN_LABEL: &str = "<unknown>";

/// Common POSIX, macOS, WSL, and Cygwin root prefixes treated as host paths at token boundaries
/// (matched ASCII case-insensitively).
const POSIX_HOST_ROOTS: &[&str] = &[
    "/users/",
    "/home/",
    "/root/",
    "/tmp/",
    "/var/",
    "/opt/",
    "/etc/",
    "/mnt/",
    "/volumes/",
    "/private/",
    "/proc/",
    "/sys/",
    "/usr/",
    "/bin/",
    "/sbin/",
    "/lib/",
    "/lib64/",
    "/dev/",
    "/run/",
    "/srv/",
    "/applications/",
    "/library/",
    "/system/",
    "/cygdrive/",
    "/wsl$/",
    "/wsl.localhost/",
];

/// Computes `numerator / denominator` as a percentage in `[0.0, 100.0]`, rounded half-up to two
/// decimal places using exact `u128` integer arithmetic.
///
/// Zero-division returns `0.0` (never `NaN` or infinity), and numerators exceeding `denominator`
/// are clamped to `100.0`.
pub fn safe_percentage_2dp(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 || numerator == 0 {
        return 0.0;
    }
    let clamped = u128::from(numerator.min(denominator));
    let denom = u128::from(denominator);
    let basis_points = (clamped * 20_000 + denom) / (denom * 2);
    basis_points as f64 / 100.0
}

/// Convenience wrapper around [`safe_percentage_2dp`] for `u32` survey counts.
pub fn safe_percentage_u32(numerator: u32, denominator: u32) -> f64 {
    safe_percentage_2dp(u64::from(numerator), u64::from(denominator))
}

/// Computes an unclamped non-negative ratio `numerator / denominator`, rounded half-up to two
/// decimal places using exact overflow-safe `u128` arithmetic.
///
/// Zero-division always returns `0.0` (never `NaN`, `-0.0`, or infinity).
pub fn safe_ratio_2dp(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 || numerator == 0 {
        return 0.0;
    }
    let num = u128::from(numerator);
    let den = u128::from(denominator);
    let scaled = match num
        .checked_mul(200)
        .and_then(|v| v.checked_add(den))
        .and_then(|v| den.checked_mul(2).map(|d2| v / d2))
    {
        Some(centi) => centi,
        None => {
            let q = num / den;
            let r = num % den;
            let frac = (r.saturating_mul(200).saturating_add(den))
                / den.saturating_mul(2).max(1);
            q.saturating_mul(100).saturating_add(frac)
        }
    };
    scaled as f64 / 100.0
}

/// Convenience wrapper around [`safe_ratio_2dp`] for `u32` survey counts.
pub fn safe_ratio_u32(numerator: u32, denominator: u32) -> f64 {
    safe_ratio_2dp(u64::from(numerator), u64::from(denominator))
}

/// Zero-division safe summary ratios and percentages derived from a [`Survey`].
#[derive(Clone, Debug, PartialEq)]
pub struct SurveyMetrics {
    pub analyzed_function_pct: f64,
    pub aborted_function_pct: f64,
    pub confirmed_site_pct: f64,
    pub unconfirmed_site_pct: f64,
    pub avg_sites_per_slot: f64,
    pub avg_slots_per_key: f64,
    pub avg_calls_per_key: f64,
    pub avg_contexts_per_round: f64,
    pub total_unresolved: u64,
    pub unresolved_per_source_call: f64,
    pub total_slot_records: u64,
    pub confidence_counts: BTreeMap<&'static str, u32>,
    pub high_confidence_slot_pct: f64,
}

impl SurveyMetrics {
    /// Computes deterministic, zero-division and overflow-safe survey metrics.
    pub fn from_survey(s: &Survey) -> Self {
        let total_sites = u64::from(s.totals.sites).saturating_add(u64::from(s.totals.unconfirmed_sites));
        let total_unresolved = s
            .unresolved
            .values()
            .fold(0u64, |acc, &v| acc.saturating_add(u64::from(v)));

        let mut confidence_counts: BTreeMap<&'static str, u32> = BTreeMap::from([
            (Confidence::High.as_str(), 0),
            (Confidence::Medium.as_str(), 0),
            (Confidence::Low.as_str(), 0),
        ]);
        let mut total_slot_records = 0u64;
        for src in &s.sources {
            for k in &src.keys {
                for sl in &k.slots {
                    total_slot_records = total_slot_records.saturating_add(1);
                    let entry = confidence_counts
                        .entry(sl.confidence.as_str())
                        .or_insert(0);
                    *entry = entry.saturating_add(1);
                }
            }
        }
        let high_slots = u64::from(*confidence_counts.get(Confidence::High.as_str()).unwrap_or(&0));

        Self {
            analyzed_function_pct: safe_percentage_u32(
                s.stats.functions_analyzed,
                s.stats.functions_discovered,
            ),
            aborted_function_pct: safe_percentage_u32(
                s.stats.aborted,
                s.stats.functions_analyzed,
            ),
            confirmed_site_pct: safe_percentage_2dp(u64::from(s.totals.sites), total_sites),
            unconfirmed_site_pct: safe_percentage_2dp(
                u64::from(s.totals.unconfirmed_sites),
                total_sites,
            ),
            avg_sites_per_slot: safe_ratio_u32(s.totals.sites, s.totals.slots),
            avg_slots_per_key: safe_ratio_u32(s.totals.slots, s.totals.keys),
            avg_calls_per_key: safe_ratio_u32(s.totals.source_calls, s.totals.keys),
            avg_contexts_per_round: safe_ratio_u32(s.stats.contexts, s.stats.rounds),
            total_unresolved,
            unresolved_per_source_call: safe_ratio_2dp(
                total_unresolved,
                u64::from(s.totals.source_calls),
            ),
            total_slot_records,
            confidence_counts,
            high_confidence_slot_pct: safe_percentage_2dp(high_slots, total_slot_records),
        }
    }
}

fn is_path_boundary(b: u8) -> bool {
    matches!(
        b,
        b' ' | b'\t'
            | b'\n'
            | b'\r'
            | b'"'
            | b'\''
            | b'`'
            | b'('
            | b')'
            | b'['
            | b']'
            | b'{'
            | b'}'
            | b'<'
            | b'>'
            | b'='
            | b','
            | b';'
            | b':'
            | b'|'
    )
}

fn is_path_hard_stop(b: u8) -> bool {
    matches!(
        b,
        b'"' | b'\''
            | b'`'
            | b'<'
            | b'>'
            | b'|'
            | b','
            | b';'
            | b'!'
            | b'\n'
            | b'\r'
            | b'\t'
    )
}

fn starts_with_ascii_ignore_case(s: &str, prefix: &str) -> bool {
    let sb = s.as_bytes();
    let pb = prefix.as_bytes();
    sb.len() >= pb.len() && sb[..pb.len()].eq_ignore_ascii_case(pb)
}

/// Checks whether a host filesystem path begins at byte offset `i` (which must be a UTF-8 char
/// boundary). Returns `Some(min_prefix_len)` if a host path starts at `i`.
fn match_host_path_prefix(text: &str, i: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    let tail = &text[i..];
    let boundary = i == 0 || is_path_boundary(bytes[i - 1]);

    // 1. file:// or file:/ URI scheme (including percent-encoded slashes), case-insensitive.
    if boundary {
        if starts_with_ascii_ignore_case(tail, "file://") {
            return Some(7);
        }
        if starts_with_ascii_ignore_case(tail, "file:/") {
            return Some(6);
        }
        if starts_with_ascii_ignore_case(tail, "file:%2f") {
            return Some(8);
        }
    }

    // 2. Windows extended-length / device / JSON-escaped UNC prefixes (`\\?\`, `\\.\`, `//?/`,
    //    `//./`, `\\\\?\\`, `\\\\.\\`, `\\\\host\\`).
    if boundary {
        for prefix in [r"\\?\", r"\\.\", "//?/", "//./", r"\\\\?\\", r"\\\\.\\"] {
            if tail.starts_with(prefix) {
                return Some(prefix.len());
            }
        }
        // JSON-escaped UNC: `\\\\<host>\\`
        if tail.starts_with(r"\\\\") {
            let mut j = 4;
            let tb = tail.as_bytes();
            while j < tb.len()
                && (tb[j].is_ascii_alphanumeric() || matches!(tb[j], b'.' | b'-' | b'_'))
            {
                j += 1;
            }
            if j > 4 && j + 1 < tb.len() && tb[j] == b'\\' && tb[j + 1] == b'\\' {
                return Some(j + 2);
            }
            // Any bare `\\\\` followed by path characters is also unsafe.
            if tb.len() > 4 && !is_path_hard_stop(tb[4]) && tb[4] != b' ' {
                return Some(4);
            }
        }
        // Raw Windows UNC (`\\host\share`) or forward-slash UNC (`//host/share`, never preceded
        // by `:` such as `http://`).
        let preceded_by_colon = i > 0 && bytes[i - 1] == b':';
        if !preceded_by_colon
            && (tail.starts_with(r"\\") || tail.starts_with("//"))
            && tail.len() >= 4
        {
            let sep = tail.as_bytes()[0];
            let tb = tail.as_bytes();
            if tb[2].is_ascii_alphanumeric() {
                let mut j = 2;
                while j < tb.len()
                    && (tb[j].is_ascii_alphanumeric() || matches!(tb[j], b'.' | b'-' | b'_'))
                {
                    j += 1;
                }
                if j > 2 && j < tb.len() && tb[j] == sep {
                    return Some(j + 1);
                }
            }
        }
    }

    // 3. Windows drive-letter paths: `C:\`, `C:/`, and JSON-escaped `C:\\`, `C:\/`.
    if tail.len() >= 3 {
        let tb = tail.as_bytes();
        if tb[0].is_ascii_alphabetic()
            && tb[1] == b':'
            && matches!(tb[2], b'\\' | b'/')
            && !(i > 0 && bytes[i - 1].is_ascii_alphanumeric())
        {
            return Some(3);
        }
    }

    // 4. Home-relative (`~/`, `~\`, `~user/`) and parent-traversal (`../`, `..\`) paths.
    if boundary {
        if tail.starts_with("~/") || tail.starts_with(r"~\") {
            return Some(2);
        }
        if tail.starts_with('~') {
            let tb = tail.as_bytes();
            let mut j = 1;
            while j < tb.len()
                && (tb[j].is_ascii_alphanumeric() || matches!(tb[j], b'.' | b'-' | b'_'))
            {
                j += 1;
            }
            if j > 1 && j < tb.len() && matches!(tb[j], b'/' | b'\\') {
                return Some(j + 1);
            }
        }
        if tail.starts_with("../") || tail.starts_with(r"..\") {
            return Some(3);
        }
    }

    // 5. Common POSIX / macOS / WSL / Cygwin absolute host roots at token boundaries.
    if boundary && tail.starts_with('/') {
        for root in POSIX_HOST_ROOTS {
            if starts_with_ascii_ignore_case(tail, root) {
                return Some(root.len());
            }
        }
    }

    None
}

/// Scans forward from the end of a matched host-path prefix to find the full path span, keeping
/// embedded spaces and balanced `(x86)` segments when followed by another path separator before
/// the next hard terminator.
fn consume_host_path_end(text: &str, start: usize, min_prefix_len: usize) -> usize {
    let bytes = text.as_bytes();
    let mut end = (start + min_prefix_len).min(bytes.len());
    let mut paren_depth: u32 = 0;

    while end < bytes.len() {
        let b = bytes[end];
        if is_path_hard_stop(b) {
            break;
        }
        match b {
            b'(' => {
                // Absorb `(x86)` inside paths only if a `)` followed by `/` or `\` appears before
                // the next hard stop.
                let mut scan = end + 1;
                let mut found_close_sep = false;
                while scan < bytes.len() && !is_path_hard_stop(bytes[scan]) {
                    if bytes[scan] == b')'
                        && scan + 1 < bytes.len()
                        && matches!(bytes[scan + 1], b'\\' | b'/')
                    {
                        found_close_sep = true;
                        break;
                    }
                    scan += 1;
                }
                if found_close_sep {
                    paren_depth = paren_depth.saturating_add(1);
                    end += 1;
                } else {
                    break;
                }
            }
            b')' => {
                if paren_depth > 0 {
                    paren_depth -= 1;
                    end += 1;
                } else {
                    break;
                }
            }
            b'[' | b']' | b'{' | b'}' => break,
            b' ' => {
                // Absorb spaces inside paths like `C:\Program Files\Xbox\game.exe` only when the
                // next token does not start a brand-new host path and a path separator appears
                // before the next hard stop or unmatched closing bracket.
                let mut next = end + 1;
                while next < bytes.len() && bytes[next] == b' ' {
                    next += 1;
                }
                if next >= bytes.len()
                    || is_path_hard_stop(bytes[next])
                    || matches!(bytes[next], b')' | b']' | b'}')
                    || match_host_path_prefix(text, next).is_some()
                {
                    break;
                }
                let mut scan = next;
                let mut has_sep = false;
                while scan < bytes.len()
                    && !is_path_hard_stop(bytes[scan])
                    && !matches!(bytes[scan], b']' | b'}')
                {
                    if bytes[scan] == b')' && paren_depth == 0 {
                        // Allow `(x86)\` inside the lookahead, stop on closing outer `)`.
                        if !(scan + 1 < bytes.len() && matches!(bytes[scan + 1], b'\\' | b'/')) {
                            break;
                        }
                    }
                    if scan > next && bytes[scan - 1] == b' ' && match_host_path_prefix(text, scan).is_some() {
                        break;
                    }
                    if matches!(bytes[scan], b'\\' | b'/') {
                        has_sep = true;
                        break;
                    }
                    scan += 1;
                }
                if has_sep {
                    end += 1;
                } else {
                    break;
                }
            }
            _ => {
                // Advance one full UTF-8 character.
                let ch_len = text[end..]
                    .chars()
                    .next()
                    .map_or(1, char::len_utf8);
                end += ch_len;
            }
        }
    }

    end
}

/// Returns `true` if `text` appears to contain any Windows drive, UNC/extended-length, POSIX root,
/// home/traversal, or `file://` host filesystem path.
pub fn contains_host_path(text: &str) -> bool {
    for (idx, _) in text.char_indices() {
        if match_host_path_prefix(text, idx).is_some() {
            return true;
        }
    }
    false
}

/// Strips ANSI CSI escapes and control characters, then replaces every detected host filesystem
/// path (Windows drive, UNC/extended-length, POSIX root, home/traversal, or `file://` URI) with
/// [`REDACTED_PATH`].
pub fn redact_host_paths(input: &str) -> String {
    let cleaned = strip_ansi_and_controls(input);
    let mut out = String::with_capacity(cleaned.len());
    let mut idx = 0usize;
    while idx < cleaned.len() {
        if let Some(prefix_len) = match_host_path_prefix(&cleaned, idx) {
            let end = consume_host_path_end(&cleaned, idx, prefix_len);
            out.push_str(REDACTED_PATH);
            idx = end.max(idx + 1);
        } else {
            let ch = cleaned[idx..].chars().next().unwrap();
            out.push(ch);
            idx += ch.len_utf8();
        }
    }
    out.trim().to_string()
}

fn strip_ansi_and_controls(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\x1b' {
            // Drop ANSI CSI sequences (`ESC [ ... final_byte`).
            if matches!(chars.peek(), Some('[')) {
                chars.next();
                for c in chars.by_ref() {
                    if ('@'..='~').contains(&c) {
                        break;
                    }
                }
            }
            continue;
        }
        match ch {
            '\r' | '\n' | '\t' => out.push(' '),
            c if c.is_control() => {}
            c => out.push(c),
        }
    }
    out
}

fn sanitize_name(raw: &str) -> String {
    let sanitized = redact_host_paths(raw);
    if sanitized.is_empty() {
        UNKNOWN_LABEL.to_string()
    } else {
        sanitized
    }
}

fn sanitize_unresolved(map: &BTreeMap<&'static str, u32>) -> BTreeMap<String, u32> {
    let mut out: BTreeMap<String, u32> = BTreeMap::new();
    for (&reason, &count) in map {
        let key = sanitize_name(reason);
        let slot = out.entry(key).or_insert(0);
        *slot = slot.saturating_add(count);
    }
    out
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct CanonicalSlot {
    slot: u32,
    sites: u32,
    unconfirmed_sites: u32,
    functions: u32,
    caller_sites: u32,
    min_args: u8,
    max_args: u8,
    evidence: u8,
    confidence: Confidence,
    examples: BTreeSet<u64>,
    unconfirmed_examples: BTreeSet<u64>,
}

impl CanonicalSlot {
    fn from_slot(sl: &SlotReport) -> Self {
        let (min_args, max_args) = (sl.min_args.min(sl.max_args), sl.min_args.max(sl.max_args));
        Self {
            slot: sl.slot,
            sites: sl.sites,
            unconfirmed_sites: sl.unconfirmed_sites,
            functions: sl.functions,
            caller_sites: sl.caller_sites,
            min_args,
            max_args,
            evidence: sl.evidence,
            confidence: sl.confidence,
            examples: sl.examples.iter().copied().collect(),
            unconfirmed_examples: sl.unconfirmed_examples.iter().copied().collect(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct CanonicalKey {
    key_args: Vec<ArgKey>,
    source_calls: u32,
    source_functions: u32,
    out_stack: u32,
    out_global: u32,
    out_ret: u32,
    globals: BTreeSet<u64>,
    global_fields: BTreeSet<(u64, i64)>,
    slots: Vec<CanonicalSlot>,
    unresolved: BTreeMap<String, u32>,
}

impl CanonicalKey {
    fn from_key(k: &KeyReport) -> Self {
        let mut slots: Vec<CanonicalSlot> = k.slots.iter().map(CanonicalSlot::from_slot).collect();
        slots.sort();
        Self {
            key_args: k.key_args.clone(),
            source_calls: k.source_calls,
            source_functions: k.source_functions,
            out_stack: k.out_stack,
            out_global: k.out_global,
            out_ret: k.out_ret,
            globals: k.globals.iter().copied().collect(),
            global_fields: k.global_fields.iter().copied().collect(),
            slots,
            unresolved: sanitize_unresolved(&k.unresolved),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct CanonicalSource {
    name: String,
    description: String,
    keys: Vec<CanonicalKey>,
}

impl CanonicalSource {
    fn from_source(src: &SourceReport) -> Self {
        let mut keys: Vec<CanonicalKey> = src.keys.iter().map(CanonicalKey::from_key).collect();
        keys.sort();
        Self {
            name: sanitize_name(&src.name),
            description: redact_host_paths(&src.description),
            keys,
        }
    }
}

struct CanonicalSurvey {
    sources: Vec<CanonicalSource>,
    totals: Totals,
    unresolved: BTreeMap<String, u32>,
    guard_slots: BTreeSet<u64>,
    stats: Stats,
    metrics: SurveyMetrics,
}

impl CanonicalSurvey {
    fn from_survey(s: &Survey) -> Self {
        let mut sources: Vec<CanonicalSource> = s
            .sources
            .iter()
            .map(CanonicalSource::from_source)
            .collect();
        sources.sort();
        Self {
            sources,
            totals: s.totals.clone(),
            unresolved: sanitize_unresolved(&s.unresolved),
            guard_slots: s.guard_slots.iter().copied().collect(),
            stats: s.stats.clone(),
            metrics: SurveyMetrics::from_survey(s),
        }
    }
}

fn key_label(args: &[ArgKey]) -> String {
    if args.is_empty() {
        "-".to_string()
    } else {
        args.iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(" / ")
    }
}

fn hex_list(values: &BTreeSet<u64>) -> String {
    values
        .iter()
        .map(|v| format!("0x{v:x}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn counts(map: &BTreeMap<String, u32>) -> String {
    if map.is_empty() {
        return "none".to_string();
    }
    map.iter()
        .map(|(k, v)| format!("{k} {v}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// `0x<global>+0x<disp>` (or `-0x<disp>`) per field.
fn field_list(fields: &BTreeSet<(u64, i64)>) -> Vec<String> {
    fields
        .iter()
        .map(|&(g, d)| {
            if d < 0 {
                format!("0x{g:x}-0x{:x}", d.unsigned_abs())
            } else {
                format!("0x{g:x}+0x{d:x}")
            }
        })
        .collect()
}

fn format_arg_range(min_args: u8, max_args: u8) -> String {
    if min_args == max_args {
        format!(">={min_args}")
    } else {
        format!(">={min_args}..{max_args}")
    }
}

/// Fixed-width deterministic, sanitized text table.
pub fn render_text(s: &Survey) -> String {
    let cs = CanonicalSurvey::from_survey(s);
    let mut out = String::new();
    let _ = writeln!(
        out,
        "interface survey: {} functions discovered, {} analysed in {} contexts, {} rounds, \
         {} aborted{}",
        cs.stats.functions_discovered,
        cs.stats.functions_analyzed,
        cs.stats.contexts,
        cs.stats.rounds,
        cs.stats.aborted,
        if cs.stats.context_limit_hit {
            " (context limit hit)"
        } else {
            ""
        }
    );
    let _ = writeln!(out, "guard slots: {}", hex_list(&cs.guard_slots));
    for src in &cs.sources {
        let _ = writeln!(out, "\nsource {} ({})", src.name, src.description);
        if src.keys.is_empty() {
            let _ = writeln!(out, "  no interface found");
        }
        for k in &src.keys {
            render_key_text(&mut out, k);
        }
    }
    let t = &cs.totals;
    let _ = writeln!(
        out,
        "\ntotals: {} keys, {} source calls, {} slots, {} confirmed sites, {} unconfirmed sites, \
         {} functions",
        t.keys, t.source_calls, t.slots, t.sites, t.unconfirmed_sites, t.functions
    );
    let _ = writeln!(out, "unresolved uses: {}", counts(&cs.unresolved));
    out
}

fn render_key_text(out: &mut String, k: &CanonicalKey) {
    let _ = writeln!(
        out,
        "  key {}: {} source calls in {} functions (out: stack {}, global {}, ret {}){}",
        key_label(&k.key_args),
        k.source_calls,
        k.source_functions,
        k.out_stack,
        k.out_global,
        k.out_ret,
        if k.globals.is_empty() {
            String::new()
        } else {
            format!("; globals {}", hex_list(&k.globals))
        }
    );
    if !k.global_fields.is_empty() {
        let _ = writeln!(
            out,
            "    global object fields: {}",
            field_list(&k.global_fields).join(",")
        );
    }
    if !k.slots.is_empty() {
        let _ = writeln!(
            out,
            "    {:>5} {:>6} {:>7} {:>6} {:>8} {:>7}  {:<14} {:<6}  examples",
            "slot", "sites", "unconf", "funcs", "callers", "args", "evidence", "conf"
        );
    }
    for sl in &k.slots {
        let args = format_arg_range(sl.min_args, sl.max_args);
        let mut ex = hex_list(&sl.examples);
        if !sl.unconfirmed_examples.is_empty() {
            let _ = write!(ex, " unconfirmed:{}", hex_list(&sl.unconfirmed_examples));
        }
        let _ = writeln!(
            out,
            "    {:>5} {:>6} {:>7} {:>6} {:>8} {:>7}  {:<14} {:<6}  {}",
            sl.slot,
            sl.sites,
            sl.unconfirmed_sites,
            sl.functions,
            sl.caller_sites,
            args,
            evidence_str(sl.evidence),
            sl.confidence.as_str(),
            ex
        );
    }
    if !k.unresolved.is_empty() {
        let _ = writeln!(out, "    unresolved: {}", counts(&k.unresolved));
    }
}

fn md_escape_cell(raw: &str) -> String {
    let clean = redact_host_paths(raw);
    let mut out = String::with_capacity(clean.len());
    for ch in clean.chars() {
        match ch {
            '|' => out.push_str("\\|"),
            '`' => out.push('\''),
            c => out.push(c),
        }
    }
    out
}

/// Deterministic, sanitized Markdown report with canonical ordering and zero-division safe ratios.
pub fn render_markdown(s: &Survey) -> String {
    let cs = CanonicalSurvey::from_survey(s);
    let m = &cs.metrics;
    let st = &cs.stats;
    let t = &cs.totals;

    let mut out = String::with_capacity(1024);
    let _ = writeln!(out, "# Interface Survey Report");
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "> Sanitized static analysis report: addresses, GUID constants, counts, and ratios only (host paths redacted)."
    );
    let _ = writeln!(out);

    let _ = writeln!(out, "## Analysis Summary");
    let _ = writeln!(out);
    let _ = writeln!(out, "| Metric | Value |");
    let _ = writeln!(out, "| --- | ---: |");
    let _ = writeln!(out, "| Functions Discovered | {} |", st.functions_discovered);
    let _ = writeln!(
        out,
        "| Functions Analyzed | {} ({:.2}%) |",
        st.functions_analyzed, m.analyzed_function_pct
    );
    let _ = writeln!(
        out,
        "| Aborted Functions | {} ({:.2}%) |",
        st.aborted, m.aborted_function_pct
    );
    let _ = writeln!(
        out,
        "| Contexts / Rounds | {} / {} ({:.2} avg/round) |",
        st.contexts, st.rounds, m.avg_contexts_per_round
    );
    let _ = writeln!(
        out,
        "| Context Limit Hit | {} |",
        if st.context_limit_hit { "yes" } else { "no" }
    );
    let guard_display = if cs.guard_slots.is_empty() {
        "_none_".to_string()
    } else {
        format!("`{}`", hex_list(&cs.guard_slots))
    };
    let _ = writeln!(out, "| Guard Slots | {guard_display} |");
    let _ = writeln!(out);

    let _ = writeln!(out, "## Totals & Coverage Ratios");
    let _ = writeln!(out);
    let _ = writeln!(out, "| Metric | Value |");
    let _ = writeln!(out, "| --- | ---: |");
    let _ = writeln!(out, "| Keys | {} |", t.keys);
    let _ = writeln!(
        out,
        "| Source Calls | {} ({:.2} avg/key) |",
        t.source_calls, m.avg_calls_per_key
    );
    let _ = writeln!(
        out,
        "| Confirmed Slots | {} ({:.2} avg/key) |",
        t.slots, m.avg_slots_per_key
    );
    let _ = writeln!(
        out,
        "| Confirmed Sites | {} ({:.2}%, {:.2} avg/slot) |",
        t.sites, m.confirmed_site_pct, m.avg_sites_per_slot
    );
    let _ = writeln!(
        out,
        "| Unconfirmed Sites | {} ({:.2}%) |",
        t.unconfirmed_sites, m.unconfirmed_site_pct
    );
    let _ = writeln!(out, "| Wrapper Functions | {} |", t.functions);
    let _ = writeln!(
        out,
        "| High-Confidence Slots | {}/{} ({:.2}%) |",
        m.confidence_counts
            .get(Confidence::High.as_str())
            .copied()
            .unwrap_or(0),
        m.total_slot_records,
        m.high_confidence_slot_pct
    );
    let _ = writeln!(
        out,
        "| Unresolved Uses | {} ({:.2} per source call) |",
        m.total_unresolved, m.unresolved_per_source_call
    );
    let _ = writeln!(out);

    let _ = writeln!(out, "## Sources");
    let _ = writeln!(out);
    if cs.sources.is_empty() {
        let _ = writeln!(out, "_No sources recorded._");
        let _ = writeln!(out);
    } else {
        for src in &cs.sources {
            let _ = writeln!(
                out,
                "### `{}`",
                md_escape_cell(&src.name)
            );
            let _ = writeln!(out);
            if !src.description.is_empty() {
                let _ = writeln!(out, "- Description: `{}`", md_escape_cell(&src.description));
            }
            if src.keys.is_empty() {
                let _ = writeln!(out, "- _No interface found._");
                let _ = writeln!(out);
                continue;
            }
            let _ = writeln!(out);
            for k in &src.keys {
                let _ = writeln!(
                    out,
                    "#### Key `{}`",
                    md_escape_cell(&key_label(&k.key_args))
                );
                let _ = writeln!(out);
                let _ = writeln!(
                    out,
                    "- Source calls: {} in {} functions (out: stack {}, global {}, ret {})",
                    k.source_calls, k.source_functions, k.out_stack, k.out_global, k.out_ret
                );
                if !k.globals.is_empty() {
                    let _ = writeln!(out, "- Globals: `{}`", hex_list(&k.globals));
                }
                if !k.global_fields.is_empty() {
                    let _ = writeln!(
                        out,
                        "- Global object fields: `{}`",
                        field_list(&k.global_fields).join(",")
                    );
                }
                if !k.slots.is_empty() {
                    let _ = writeln!(out);
                    let _ = writeln!(
                        out,
                        "| Slot | Sites | Unconf | Confirmed % | Funcs | Callers | Args | Evidence | Confidence | Examples |"
                    );
                    let _ = writeln!(
                        out,
                        "| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |"
                    );
                    for sl in &k.slots {
                        let slot_total = u64::from(sl.sites).saturating_add(u64::from(sl.unconfirmed_sites));
                        let slot_conf_pct = safe_percentage_2dp(u64::from(sl.sites), slot_total);
                        let mut ex = hex_list(&sl.examples);
                        if !sl.unconfirmed_examples.is_empty() {
                            if !ex.is_empty() {
                                ex.push(' ');
                            }
                            let _ = write!(ex, "unconfirmed:{}", hex_list(&sl.unconfirmed_examples));
                        }
                        let ex_cell = if ex.is_empty() {
                            "-".to_string()
                        } else {
                            format!("`{}`", md_escape_cell(&ex))
                        };
                        let _ = writeln!(
                            out,
                            "| {} | {} | {} | {:.2}% | {} | {} | `{}` | `{}` | `{}` | {} |",
                            sl.slot,
                            sl.sites,
                            sl.unconfirmed_sites,
                            slot_conf_pct,
                            sl.functions,
                            sl.caller_sites,
                            format_arg_range(sl.min_args, sl.max_args),
                            evidence_str(sl.evidence),
                            sl.confidence.as_str(),
                            ex_cell
                        );
                    }
                }
                if !k.unresolved.is_empty() {
                    let _ = writeln!(out);
                    let _ = writeln!(out, "- Unresolved: `{}`", md_escape_cell(&counts(&k.unresolved)));
                }
                let _ = writeln!(out);
            }
        }
    }

    let _ = writeln!(out, "## Unresolved Uses");
    let _ = writeln!(out);
    if cs.unresolved.is_empty() {
        let _ = writeln!(out, "_None._");
    } else {
        let _ = writeln!(out, "| Reason | Count | Share |");
        let _ = writeln!(out, "| --- | ---: | ---: |");
        for (reason, &count) in &cs.unresolved {
            let share = safe_percentage_2dp(u64::from(count), m.total_unresolved);
            let _ = writeln!(
                out,
                "| `{}` | {} | {:.2}% |",
                md_escape_cell(reason),
                count,
                share
            );
        }
    }

    out
}

fn json_str(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\u{08}' => o.push_str("\\b"),
            '\u{0C}' => o.push_str("\\f"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(o, "\\u{:04x}", c as u32);
            }
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn json_hex_list(values: &BTreeSet<u64>) -> String {
    let items: Vec<String> = values.iter().map(|v| format!("\"0x{v:x}\"")).collect();
    format!("[{}]", items.join(", "))
}

fn json_counts(map: &BTreeMap<String, u32>) -> String {
    let items: Vec<String> = map
        .iter()
        .map(|(k, v)| format!("{}: {v}", json_str(k)))
        .collect();
    format!("{{{}}}", items.join(", "))
}

/// Pretty-printed deterministic, host-path redacted JSON with a trailing newline, preserving the
/// documented CLI report schema.
pub fn render_json(s: &Survey) -> String {
    let cs = CanonicalSurvey::from_survey(s);
    let mut o = String::new();
    o.push_str("{\n");
    let st = &cs.stats;
    let _ = writeln!(
        o,
        "  \"stats\": {{\"functions_discovered\": {}, \"functions_analyzed\": {}, \"contexts\": {}, \
         \"rounds\": {}, \"aborted\": {}, \"context_limit_hit\": {}}},",
        st.functions_discovered,
        st.functions_analyzed,
        st.contexts,
        st.rounds,
        st.aborted,
        st.context_limit_hit
    );
    let _ = writeln!(o, "  \"guard_slots\": {},", json_hex_list(&cs.guard_slots));
    let t = &cs.totals;
    let _ = writeln!(
        o,
        "  \"totals\": {{\"keys\": {}, \"source_calls\": {}, \"slots\": {}, \"sites\": {}, \
         \"unconfirmed_sites\": {}, \"functions\": {}}},",
        t.keys, t.source_calls, t.slots, t.sites, t.unconfirmed_sites, t.functions
    );
    let _ = writeln!(o, "  \"unresolved\": {},", json_counts(&cs.unresolved));
    o.push_str("  \"sources\": [");
    for (i, src) in cs.sources.iter().enumerate() {
        o.push_str(if i == 0 { "\n" } else { ",\n" });
        let _ = write!(
            o,
            "    {{\"name\": {}, \"description\": {}, \"keys\": [",
            json_str(&src.name),
            json_str(&src.description)
        );
        for (j, k) in src.keys.iter().enumerate() {
            o.push_str(if j == 0 { "\n" } else { ",\n" });
            let key: Vec<String> = k
                .key_args
                .iter()
                .map(|a| json_str(&a.to_string()))
                .collect();
            let _ = write!(
                o,
                "      {{\"key\": [{}], \"source_calls\": {}, \"source_functions\": {}, \
                 \"out\": {{\"stack\": {}, \"global\": {}, \"ret\": {}}}, \"globals\": {}, \
                 \"global_fields\": [{}], \"unresolved\": {}, \"slots\": [",
                key.join(", "),
                k.source_calls,
                k.source_functions,
                k.out_stack,
                k.out_global,
                k.out_ret,
                json_hex_list(&k.globals),
                field_list(&k.global_fields)
                    .iter()
                    .map(|s| json_str(s))
                    .collect::<Vec<_>>()
                    .join(", "),
                json_counts(&k.unresolved)
            );
            for (n, sl) in k.slots.iter().enumerate() {
                let _ = write!(
                    o,
                    "{}        {{\"slot\": {}, \"sites\": {}, \"unconfirmed_sites\": {}, \
                     \"functions\": {}, \"caller_sites\": {}, \"min_args\": {}, \"max_args\": {}, \
                     \"evidence\": {}, \"confidence\": {}, \"examples\": {}, \
                     \"unconfirmed_examples\": {}}}",
                    if n == 0 { "\n" } else { ",\n" },
                    sl.slot,
                    sl.sites,
                    sl.unconfirmed_sites,
                    sl.functions,
                    sl.caller_sites,
                    sl.min_args,
                    sl.max_args,
                    json_str(&evidence_str(sl.evidence)),
                    json_str(sl.confidence.as_str()),
                    json_hex_list(&sl.examples),
                    json_hex_list(&sl.unconfirmed_examples)
                );
            }
            o.push_str(if k.slots.is_empty() {
                "]}"
            } else {
                "\n      ]}"
            });
        }
        o.push_str(if src.keys.is_empty() {
            "]}"
        } else {
            "\n    ]}"
        });
    }
    o.push_str(if cs.sources.is_empty() {
        "]\n"
    } else {
        "\n  ]\n"
    });
    o.push_str("}\n");
    o
}

/// Strictly key-sorted JSON AST backed by [`BTreeMap`] for canonical serialization.
enum JsonVal {
    Bool(bool),
    U64(u64),
    F64Fixed2(f64),
    Str(String),
    Array(Vec<JsonVal>),
    Object(BTreeMap<String, JsonVal>),
}

impl JsonVal {
    fn write_pretty(&self, out: &mut String, indent: usize) {
        match self {
            Self::Bool(b) => {
                let _ = write!(out, "{b}");
            }
            Self::U64(n) => {
                let _ = write!(out, "{n}");
            }
            Self::F64Fixed2(f) => {
                let safe = if f.is_finite() && *f != 0.0 { *f } else { 0.0 };
                let _ = write!(out, "{safe:.2}");
            }
            Self::Str(s) => out.push_str(&json_str(s)),
            Self::Array(items) => {
                if items.is_empty() {
                    out.push_str("[]");
                    return;
                }
                let scalar_only = items
                    .iter()
                    .all(|v| matches!(v, Self::Bool(_) | Self::U64(_) | Self::F64Fixed2(_) | Self::Str(_)));
                if scalar_only {
                    out.push('[');
                    for (idx, item) in items.iter().enumerate() {
                        if idx > 0 {
                            out.push_str(", ");
                        }
                        item.write_pretty(out, indent);
                    }
                    out.push(']');
                    return;
                }
                out.push_str("[\n");
                let child_indent = indent + 2;
                for (idx, item) in items.iter().enumerate() {
                    for _ in 0..child_indent {
                        out.push(' ');
                    }
                    item.write_pretty(out, child_indent);
                    if idx + 1 < items.len() {
                        out.push(',');
                    }
                    out.push('\n');
                }
                for _ in 0..indent {
                    out.push(' ');
                }
                out.push(']');
            }
            Self::Object(map) => {
                if map.is_empty() {
                    out.push_str("{}");
                    return;
                }
                out.push_str("{\n");
                let child_indent = indent + 2;
                for (idx, (k, v)) in map.iter().enumerate() {
                    for _ in 0..child_indent {
                        out.push(' ');
                    }
                    out.push_str(&json_str(k));
                    out.push_str(": ");
                    v.write_pretty(out, child_indent);
                    if idx + 1 < map.len() {
                        out.push(',');
                    }
                    out.push('\n');
                }
                for _ in 0..indent {
                    out.push(' ');
                }
                out.push('}');
            }
        }
    }
}

fn json_u32_map(map: &BTreeMap<String, u32>) -> JsonVal {
    JsonVal::Object(
        map.iter()
            .map(|(k, &v)| (k.clone(), JsonVal::U64(u64::from(v))))
            .collect(),
    )
}

fn json_hex_array(values: &BTreeSet<u64>) -> JsonVal {
    JsonVal::Array(
        values
            .iter()
            .map(|v| JsonVal::Str(format!("0x{v:x}")))
            .collect(),
    )
}

/// Renders a strictly canonical RFC 8259 JSON report backed by [`BTreeMap`] / [`BTreeSet`] where
/// every object's keys are strictly lexicographically sorted, all arrays are canonically ordered,
/// zero-division safe [`SurveyMetrics`] are included, and all host paths are strictly redacted.
pub fn render_canonical_json(s: &Survey) -> String {
    let cs = CanonicalSurvey::from_survey(s);
    let m = &cs.metrics;
    let st = &cs.stats;
    let t = &cs.totals;

    let mut stats_obj = BTreeMap::new();
    stats_obj.insert("aborted".to_string(), JsonVal::U64(u64::from(st.aborted)));
    stats_obj.insert(
        "context_limit_hit".to_string(),
        JsonVal::Bool(st.context_limit_hit),
    );
    stats_obj.insert("contexts".to_string(), JsonVal::U64(u64::from(st.contexts)));
    stats_obj.insert(
        "functions_analyzed".to_string(),
        JsonVal::U64(u64::from(st.functions_analyzed)),
    );
    stats_obj.insert(
        "functions_discovered".to_string(),
        JsonVal::U64(u64::from(st.functions_discovered)),
    );
    stats_obj.insert("rounds".to_string(), JsonVal::U64(u64::from(st.rounds)));

    let mut totals_obj = BTreeMap::new();
    totals_obj.insert("functions".to_string(), JsonVal::U64(u64::from(t.functions)));
    totals_obj.insert("keys".to_string(), JsonVal::U64(u64::from(t.keys)));
    totals_obj.insert("sites".to_string(), JsonVal::U64(u64::from(t.sites)));
    totals_obj.insert("slots".to_string(), JsonVal::U64(u64::from(t.slots)));
    totals_obj.insert(
        "source_calls".to_string(),
        JsonVal::U64(u64::from(t.source_calls)),
    );
    totals_obj.insert(
        "unconfirmed_sites".to_string(),
        JsonVal::U64(u64::from(t.unconfirmed_sites)),
    );

    let confidence_obj: BTreeMap<String, JsonVal> = m
        .confidence_counts
        .iter()
        .map(|(&k, &v)| (k.to_string(), JsonVal::U64(u64::from(v))))
        .collect();

    let mut metrics_obj = BTreeMap::new();
    metrics_obj.insert(
        "aborted_function_pct".to_string(),
        JsonVal::F64Fixed2(m.aborted_function_pct),
    );
    metrics_obj.insert(
        "analyzed_function_pct".to_string(),
        JsonVal::F64Fixed2(m.analyzed_function_pct),
    );
    metrics_obj.insert(
        "avg_calls_per_key".to_string(),
        JsonVal::F64Fixed2(m.avg_calls_per_key),
    );
    metrics_obj.insert(
        "avg_contexts_per_round".to_string(),
        JsonVal::F64Fixed2(m.avg_contexts_per_round),
    );
    metrics_obj.insert(
        "avg_sites_per_slot".to_string(),
        JsonVal::F64Fixed2(m.avg_sites_per_slot),
    );
    metrics_obj.insert(
        "avg_slots_per_key".to_string(),
        JsonVal::F64Fixed2(m.avg_slots_per_key),
    );
    metrics_obj.insert(
        "confidence_counts".to_string(),
        JsonVal::Object(confidence_obj),
    );
    metrics_obj.insert(
        "confirmed_site_pct".to_string(),
        JsonVal::F64Fixed2(m.confirmed_site_pct),
    );
    metrics_obj.insert(
        "high_confidence_slot_pct".to_string(),
        JsonVal::F64Fixed2(m.high_confidence_slot_pct),
    );
    metrics_obj.insert(
        "total_slot_records".to_string(),
        JsonVal::U64(m.total_slot_records),
    );
    metrics_obj.insert(
        "total_unresolved".to_string(),
        JsonVal::U64(m.total_unresolved),
    );
    metrics_obj.insert(
        "unconfirmed_site_pct".to_string(),
        JsonVal::F64Fixed2(m.unconfirmed_site_pct),
    );
    metrics_obj.insert(
        "unresolved_per_source_call".to_string(),
        JsonVal::F64Fixed2(m.unresolved_per_source_call),
    );

    let sources_arr: Vec<JsonVal> = cs
        .sources
        .iter()
        .map(|src| {
            let keys_arr: Vec<JsonVal> = src
                .keys
                .iter()
                .map(|k| {
                    let mut out_obj = BTreeMap::new();
                    out_obj.insert("global".to_string(), JsonVal::U64(u64::from(k.out_global)));
                    out_obj.insert("ret".to_string(), JsonVal::U64(u64::from(k.out_ret)));
                    out_obj.insert("stack".to_string(), JsonVal::U64(u64::from(k.out_stack)));

                    let slots_arr: Vec<JsonVal> = k
                        .slots
                        .iter()
                        .map(|sl| {
                            let slot_total = u64::from(sl.sites)
                                .saturating_add(u64::from(sl.unconfirmed_sites));
                            let mut sl_obj = BTreeMap::new();
                            sl_obj.insert(
                                "caller_sites".to_string(),
                                JsonVal::U64(u64::from(sl.caller_sites)),
                            );
                            sl_obj.insert(
                                "confidence".to_string(),
                                JsonVal::Str(sl.confidence.as_str().to_string()),
                            );
                            sl_obj.insert(
                                "confirmed_site_pct".to_string(),
                                JsonVal::F64Fixed2(safe_percentage_2dp(
                                    u64::from(sl.sites),
                                    slot_total,
                                )),
                            );
                            sl_obj.insert(
                                "evidence".to_string(),
                                JsonVal::Str(evidence_str(sl.evidence)),
                            );
                            sl_obj.insert("examples".to_string(), json_hex_array(&sl.examples));
                            sl_obj.insert(
                                "functions".to_string(),
                                JsonVal::U64(u64::from(sl.functions)),
                            );
                            sl_obj.insert(
                                "max_args".to_string(),
                                JsonVal::U64(u64::from(sl.max_args)),
                            );
                            sl_obj.insert(
                                "min_args".to_string(),
                                JsonVal::U64(u64::from(sl.min_args)),
                            );
                            sl_obj.insert("sites".to_string(), JsonVal::U64(u64::from(sl.sites)));
                            sl_obj.insert("slot".to_string(), JsonVal::U64(u64::from(sl.slot)));
                            sl_obj.insert(
                                "unconfirmed_examples".to_string(),
                                json_hex_array(&sl.unconfirmed_examples),
                            );
                            sl_obj.insert(
                                "unconfirmed_sites".to_string(),
                                JsonVal::U64(u64::from(sl.unconfirmed_sites)),
                            );
                            JsonVal::Object(sl_obj)
                        })
                        .collect();

                    let mut k_obj = BTreeMap::new();
                    k_obj.insert(
                        "global_fields".to_string(),
                        JsonVal::Array(
                            field_list(&k.global_fields)
                                .into_iter()
                                .map(JsonVal::Str)
                                .collect(),
                        ),
                    );
                    k_obj.insert("globals".to_string(), json_hex_array(&k.globals));
                    k_obj.insert(
                        "key".to_string(),
                        JsonVal::Array(
                            k.key_args
                                .iter()
                                .map(|a| JsonVal::Str(a.to_string()))
                                .collect(),
                        ),
                    );
                    k_obj.insert("out".to_string(), JsonVal::Object(out_obj));
                    k_obj.insert("slots".to_string(), JsonVal::Array(slots_arr));
                    k_obj.insert(
                        "source_calls".to_string(),
                        JsonVal::U64(u64::from(k.source_calls)),
                    );
                    k_obj.insert(
                        "source_functions".to_string(),
                        JsonVal::U64(u64::from(k.source_functions)),
                    );
                    k_obj.insert("unresolved".to_string(), json_u32_map(&k.unresolved));
                    JsonVal::Object(k_obj)
                })
                .collect();

            let mut src_obj = BTreeMap::new();
            src_obj.insert(
                "description".to_string(),
                JsonVal::Str(src.description.clone()),
            );
            src_obj.insert("keys".to_string(), JsonVal::Array(keys_arr));
            src_obj.insert("name".to_string(), JsonVal::Str(src.name.clone()));
            JsonVal::Object(src_obj)
        })
        .collect();

    let mut root = BTreeMap::new();
    root.insert("guard_slots".to_string(), json_hex_array(&cs.guard_slots));
    root.insert("metrics".to_string(), JsonVal::Object(metrics_obj));
    root.insert("sources".to_string(), JsonVal::Array(sources_arr));
    root.insert("stats".to_string(), JsonVal::Object(stats_obj));
    root.insert("totals".to_string(), JsonVal::Object(totals_obj));
    root.insert("unresolved".to_string(), json_u32_map(&cs.unresolved));

    let mut out = String::with_capacity(1024);
    JsonVal::Object(root).write_pretty(&mut out, 0);
    out.push('\n');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Strict RFC 8259 validator that optionally enforces strictly ascending, duplicate-free keys
    /// inside every JSON object at every nesting depth.
    struct JsonValidator<'a> {
        bytes: &'a [u8],
        pos: usize,
        require_sorted_keys: bool,
    }

    impl<'a> JsonValidator<'a> {
        fn validate(input: &'a str, require_sorted_keys: bool) -> Result<(), String> {
            let mut v = Self {
                bytes: input.as_bytes(),
                pos: 0,
                require_sorted_keys,
            };
            v.skip_ws();
            v.parse_value()?;
            v.skip_ws();
            if v.pos != v.bytes.len() {
                return Err(format!("trailing bytes at offset {}", v.pos));
            }
            Ok(())
        }

        fn skip_ws(&mut self) {
            while self.pos < self.bytes.len()
                && matches!(self.bytes[self.pos], b' ' | b'\t' | b'\n' | b'\r')
            {
                self.pos += 1;
            }
        }

        fn peek(&self) -> Option<u8> {
            self.bytes.get(self.pos).copied()
        }

        fn parse_value(&mut self) -> Result<(), String> {
            self.skip_ws();
            match self.peek() {
                Some(b'{') => self.parse_object(),
                Some(b'[') => self.parse_array(),
                Some(b'"') => self.parse_string().map(|_| ()),
                Some(b't') => self.expect_literal(b"true"),
                Some(b'f') => self.expect_literal(b"false"),
                Some(b'n') => self.expect_literal(b"null"),
                Some(b'-' | b'0'..=b'9') => self.parse_number(),
                other => Err(format!("unexpected byte {other:?} at {}", self.pos)),
            }
        }

        fn expect_literal(&mut self, lit: &[u8]) -> Result<(), String> {
            if self.bytes[self.pos..].starts_with(lit) {
                self.pos += lit.len();
                Ok(())
            } else {
                Err(format!("expected literal at {}", self.pos))
            }
        }

        fn parse_object(&mut self) -> Result<(), String> {
            self.pos += 1; // '{'
            self.skip_ws();
            if self.peek() == Some(b'}') {
                self.pos += 1;
                return Ok(());
            }
            let mut seen = BTreeSet::new();
            let mut prev_key: Option<String> = None;
            loop {
                self.skip_ws();
                let key = self.parse_string()?;
                if !seen.insert(key.clone()) {
                    return Err(format!("duplicate JSON object key {key:?}"));
                }
                if self.require_sorted_keys
                    && let Some(prev) = &prev_key
                    && prev >= &key
                {
                    return Err(format!(
                        "unsorted JSON object keys: {prev:?} appeared before {key:?}"
                    ));
                }
                prev_key = Some(key);
                self.skip_ws();
                if self.peek() != Some(b':') {
                    return Err(format!("expected ':' at {}", self.pos));
                }
                self.pos += 1;
                self.parse_value()?;
                self.skip_ws();
                match self.peek() {
                    Some(b',') => self.pos += 1,
                    Some(b'}') => {
                        self.pos += 1;
                        return Ok(());
                    }
                    other => return Err(format!("expected ',' or '}}' got {other:?}")),
                }
            }
        }

        fn parse_array(&mut self) -> Result<(), String> {
            self.pos += 1; // '['
            self.skip_ws();
            if self.peek() == Some(b']') {
                self.pos += 1;
                return Ok(());
            }
            loop {
                self.parse_value()?;
                self.skip_ws();
                match self.peek() {
                    Some(b',') => self.pos += 1,
                    Some(b']') => {
                        self.pos += 1;
                        return Ok(());
                    }
                    other => return Err(format!("expected ',' or ']' got {other:?}")),
                }
            }
        }

        fn parse_string(&mut self) -> Result<String, String> {
            if self.peek() != Some(b'"') {
                return Err(format!("expected '\"' at {}", self.pos));
            }
            self.pos += 1;
            let mut out = String::new();
            while self.pos < self.bytes.len() {
                let b = self.bytes[self.pos];
                self.pos += 1;
                match b {
                    b'"' => return Ok(out),
                    b'\\' => {
                        if self.pos >= self.bytes.len() {
                            return Err("unterminated escape".to_string());
                        }
                        let esc = self.bytes[self.pos];
                        self.pos += 1;
                        match esc {
                            b'"' | b'\\' | b'/' => out.push(esc as char),
                            b'b' => out.push('\u{08}'),
                            b'f' => out.push('\u{0C}'),
                            b'n' => out.push('\n'),
                            b'r' => out.push('\r'),
                            b't' => out.push('\t'),
                            b'u' => {
                                if self.pos + 4 > self.bytes.len() {
                                    return Err("short \\u escape".to_string());
                                }
                                let hex = std::str::from_utf8(&self.bytes[self.pos..self.pos + 4])
                                    .map_err(|_| "invalid utf8 in \\u".to_string())?;
                                let code = u32::from_str_radix(hex, 16)
                                    .map_err(|_| format!("invalid hex {hex:?}"))?;
                                self.pos += 4;
                                out.push(char::from_u32(code).unwrap_or('\u{FFFD}'));
                            }
                            other => return Err(format!("bad escape \\{}", other as char)),
                        }
                    }
                    c if c < 0x20 => return Err(format!("unescaped control byte 0x{c:02x}")),
                    c => out.push(c as char),
                }
            }
            Err("unterminated string".to_string())
        }

        fn parse_number(&mut self) -> Result<(), String> {
            if self.peek() == Some(b'-') {
                self.pos += 1;
            }
            let start = self.pos;
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
            if self.pos == start {
                return Err(format!("expected digits at {}", self.pos));
            }
            if self.peek() == Some(b'.') {
                self.pos += 1;
                let frac_start = self.pos;
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.pos += 1;
                }
                if self.pos == frac_start {
                    return Err("expected fractional digits".to_string());
                }
            }
            if matches!(self.peek(), Some(b'e' | b'E')) {
                self.pos += 1;
                if matches!(self.peek(), Some(b'+' | b'-')) {
                    self.pos += 1;
                }
                let exp_start = self.pos;
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.pos += 1;
                }
                if self.pos == exp_start {
                    return Err("expected exponent digits".to_string());
                }
            }
            Ok(())
        }
    }

    fn empty_survey() -> Survey {
        Survey {
            sources: Vec::new(),
            totals: Totals::default(),
            unresolved: BTreeMap::new(),
            guard_slots: Vec::new(),
            stats: Stats::default(),
        }
    }

    fn synthetic_guid(seed: u8) -> ArgKey {
        ArgKey::Guid([
            seed,
            seed.wrapping_add(1),
            seed.wrapping_add(2),
            seed.wrapping_add(3),
            seed.wrapping_add(4),
            seed.wrapping_add(5),
            seed.wrapping_add(6),
            seed.wrapping_add(7),
            seed.wrapping_add(8),
            seed.wrapping_add(9),
            seed.wrapping_add(10),
            seed.wrapping_add(11),
            seed.wrapping_add(12),
            seed.wrapping_add(13),
            seed.wrapping_add(14),
            seed.wrapping_add(15),
        ])
    }

    #[test]
    fn empty_survey_and_zero_division_guards() {
        // Direct zero-division and boundary checks on math helpers.
        assert_eq!(safe_percentage_2dp(0, 0), 0.0);
        assert_eq!(safe_percentage_2dp(42, 0), 0.0);
        assert_eq!(safe_percentage_2dp(0, 100), 0.0);
        assert_eq!(safe_percentage_2dp(150, 100), 100.0);
        assert_eq!(safe_percentage_2dp(1, 3), 33.33);
        assert_eq!(safe_percentage_2dp(2, 3), 66.67);
        assert_eq!(safe_percentage_2dp(1, 200), 0.5);
        assert_eq!(safe_percentage_2dp(u64::MAX, u64::MAX), 100.0);
        assert_eq!(safe_percentage_u32(0, 0), 0.0);

        assert_eq!(safe_ratio_2dp(0, 0), 0.0);
        assert_eq!(safe_ratio_2dp(99, 0), 0.0);
        assert_eq!(safe_ratio_2dp(0, 99), 0.0);
        assert_eq!(safe_ratio_2dp(3, 2), 1.5);
        assert_eq!(safe_ratio_2dp(1, 3), 0.33);
        assert_eq!(safe_ratio_2dp(2, 3), 0.67);
        assert_eq!(safe_ratio_2dp(u64::MAX, u64::MAX), 1.0);
        assert!(safe_ratio_2dp(u64::MAX, 1).is_finite());
        assert_eq!(safe_ratio_u32(0, 0), 0.0);

        // Empty survey renders cleanly across every format with finite zero metrics.
        let s = empty_survey();
        let m = SurveyMetrics::from_survey(&s);
        assert_eq!(m.analyzed_function_pct, 0.0);
        assert_eq!(m.aborted_function_pct, 0.0);
        assert_eq!(m.confirmed_site_pct, 0.0);
        assert_eq!(m.unconfirmed_site_pct, 0.0);
        assert_eq!(m.avg_sites_per_slot, 0.0);
        assert_eq!(m.avg_slots_per_key, 0.0);
        assert_eq!(m.avg_calls_per_key, 0.0);
        assert_eq!(m.avg_contexts_per_round, 0.0);
        assert_eq!(m.unresolved_per_source_call, 0.0);
        assert_eq!(m.high_confidence_slot_pct, 0.0);

        let text = render_text(&s);
        let json = render_json(&s);
        let cjson = render_canonical_json(&s);
        let md = render_markdown(&s);

        JsonValidator::validate(&json, false).expect("empty render_json must be valid RFC 8259");
        JsonValidator::validate(&cjson, true)
            .expect("empty render_canonical_json must be valid with strictly sorted keys");
        for out in [&text, &json, &cjson, &md] {
            assert!(!out.contains("NaN"), "NaN leaked in:\n{out}");
            assert!(!out.contains("inf"), "inf leaked in:\n{out}");
            assert!(!contains_host_path(out), "false-positive host path in:\n{out}");
        }
        assert!(md.contains("_No sources recorded._"));
        assert!(md.contains("_None._"));
    }

    #[test]
    fn extreme_u32_max_counts_are_overflow_and_zero_div_safe() {
        let s = Survey {
            sources: vec![SourceReport {
                name: "stress".to_string(),
                description: "import stress.dll!Entry, out arg 0".to_string(),
                keys: vec![KeyReport {
                    key_args: vec![ArgKey::Unknown],
                    source_calls: u32::MAX,
                    source_functions: u32::MAX,
                    out_stack: u32::MAX,
                    out_global: 0,
                    out_ret: 0,
                    globals: vec![u64::MAX],
                    global_fields: vec![(0x1400_0000, i64::MIN), (0x1400_0000, i64::MAX)],
                    slots: vec![SlotReport {
                        slot: 0,
                        sites: u32::MAX,
                        unconfirmed_sites: u32::MAX,
                        functions: u32::MAX,
                        caller_sites: u32::MAX,
                        // Intentionally inverted bounds to verify normalization.
                        min_args: 6,
                        max_args: 2,
                        evidence: 0,
                        confidence: Confidence::High,
                        examples: vec![u64::MAX],
                        unconfirmed_examples: vec![0],
                    }],
                    unresolved: BTreeMap::from([("overflow_test", u32::MAX)]),
                }],
            }],
            totals: Totals {
                keys: u32::MAX,
                source_calls: u32::MAX,
                slots: u32::MAX,
                sites: u32::MAX,
                unconfirmed_sites: u32::MAX,
                functions: u32::MAX,
            },
            unresolved: BTreeMap::from([("a", u32::MAX), ("b", u32::MAX)]),
            guard_slots: vec![u64::MAX, 0],
            stats: Stats {
                functions_discovered: u32::MAX,
                functions_analyzed: u32::MAX,
                contexts: u32::MAX,
                rounds: 0, // deliberate zero denominator
                aborted: u32::MAX,
                context_limit_hit: true,
            },
        };

        let m = SurveyMetrics::from_survey(&s);
        assert_eq!(m.confirmed_site_pct, 50.0);
        assert_eq!(m.unconfirmed_site_pct, 50.0);
        assert_eq!(m.avg_contexts_per_round, 0.0);
        assert_eq!(m.total_unresolved, u64::from(u32::MAX) * 2);
        assert_eq!(m.unresolved_per_source_call, 2.0);
        assert_eq!(m.high_confidence_slot_pct, 100.0);

        let text = render_text(&s);
        let json = render_json(&s);
        let cjson = render_canonical_json(&s);
        let md = render_markdown(&s);

        assert!(text.contains(">=2..6"), "inverted min/max args must normalize");
        assert!(md.contains("0x14000000-0x8000000000000000"));
        JsonValidator::validate(&json, false).expect("stress render_json valid");
        JsonValidator::validate(&cjson, true).expect("stress render_canonical_json valid & sorted");
    }

    #[test]
    fn permutation_invariant_rendering_across_all_formats() {
        let slot_a = SlotReport {
            slot: 3,
            sites: 4,
            unconfirmed_sites: 1,
            functions: 2,
            caller_sites: 7,
            min_args: 1,
            max_args: 3,
            evidence: 0b0000_0001,
            confidence: Confidence::High,
            examples: vec![0x1400_3000, 0x1400_1000, 0x1400_3000],
            unconfirmed_examples: vec![0x1400_9000, 0x1400_8000],
        };
        let slot_a_perm = SlotReport {
            examples: vec![0x1400_1000, 0x1400_3000],
            unconfirmed_examples: vec![0x1400_8000, 0x1400_9000, 0x1400_8000],
            ..slot_a.clone()
        };
        let slot_b = SlotReport {
            slot: 14,
            sites: 2,
            unconfirmed_sites: 0,
            functions: 1,
            caller_sites: 3,
            min_args: 2,
            max_args: 2,
            evidence: 0,
            confidence: Confidence::Medium,
            examples: vec![0x1400_5000, 0x1400_4000],
            unconfirmed_examples: vec![],
        };
        let slot_b_perm = SlotReport {
            examples: vec![0x1400_4000, 0x1400_5000, 0x1400_4000],
            ..slot_b.clone()
        };

        let key_1 = KeyReport {
            key_args: vec![synthetic_guid(0x10), ArgKey::Const(2)],
            source_calls: 3,
            source_functions: 2,
            out_stack: 2,
            out_global: 1,
            out_ret: 0,
            globals: vec![0x1402_0000, 0x1401_0000, 0x1402_0000],
            global_fields: vec![(0x1402_0000, 0x48), (0x1401_0000, -0x10), (0x1402_0000, 0x48)],
            slots: vec![slot_b.clone(), slot_a.clone()],
            unresolved: BTreeMap::from([("stored_to_unknown", 2), ("passed_as_argument", 1)]),
        };
        let key_1_perm = KeyReport {
            globals: vec![0x1401_0000, 0x1402_0000],
            global_fields: vec![(0x1401_0000, -0x10), (0x1402_0000, 0x48)],
            slots: vec![slot_a_perm, slot_b_perm],
            ..key_1.clone()
        };
        let key_2 = KeyReport {
            key_args: vec![synthetic_guid(0x20), ArgKey::Unknown],
            source_calls: 1,
            source_functions: 1,
            out_stack: 0,
            out_global: 0,
            out_ret: 1,
            globals: vec![],
            global_fields: vec![],
            slots: vec![],
            unresolved: BTreeMap::new(),
        };

        let src_dev = SourceReport {
            name: "dev".to_string(),
            description: "import d3d12_xs.dll!D3D12XboxCreateDevice, out arg 3".to_string(),
            keys: vec![key_2.clone(), key_1.clone()],
        };
        let src_dev_perm = SourceReport {
            keys: vec![key_1_perm, key_2],
            ..src_dev.clone()
        };
        let src_queue = SourceReport {
            name: "queue".to_string(),
            description: "method dev:14, out arg 2".to_string(),
            keys: vec![],
        };

        let first = Survey {
            sources: vec![src_queue.clone(), src_dev],
            totals: Totals {
                keys: 2,
                source_calls: 4,
                slots: 2,
                sites: 6,
                unconfirmed_sites: 1,
                functions: 3,
            },
            unresolved: BTreeMap::from([("stored_to_unknown", 2), ("passed_as_argument", 1)]),
            guard_slots: vec![0x1405_0020, 0x1405_0010, 0x1405_0020],
            stats: Stats {
                functions_discovered: 100,
                functions_analyzed: 80,
                contexts: 160,
                rounds: 4,
                aborted: 2,
                context_limit_hit: false,
            },
        };
        let second = Survey {
            sources: vec![src_dev_perm, src_queue],
            guard_slots: vec![0x1405_0010, 0x1405_0020],
            ..first.clone()
        };

        assert_eq!(render_text(&first), render_text(&second));
        assert_eq!(render_json(&first), render_json(&second));
        assert_eq!(render_canonical_json(&first), render_canonical_json(&second));
        assert_eq!(render_markdown(&first), render_markdown(&second));

        let cjson = render_canonical_json(&first);
        JsonValidator::validate(&render_json(&first), false).expect("render_json valid");
        JsonValidator::validate(&cjson, true).expect("render_canonical_json strictly sorted");
        // Guard slots and globals must be deduplicated and sorted ascending.
        assert!(cjson.contains("\"guard_slots\": [\"0x14050010\", \"0x14050020\"]"));
        assert!(cjson.contains("\"globals\": [\"0x14010000\", \"0x14020000\"]"));
        assert!(cjson.contains("\"global_fields\": [\"0x14010000-0x10\", \"0x14020000+0x48\"]"));
    }

    #[test]
    fn adversarial_host_path_redaction_windows_unc_posix_and_file_uri() {
        let cases: &[(&str, &[&str])] = &[
            (
                r"C:\Users\alice\GTA6\default.exe",
                &["alice", "GTA6", "default.exe", "C:\\"],
            ),
            (
                "D:/XboxGames/Project/d3d12_xs.dll",
                &["XboxGames", "Project", "D:/"],
            ),
            (
                r"C:\Program Files (x86)\Rockstar Games\secret\title.exe",
                &["Program Files", "Rockstar", "secret", "title.exe"],
            ),
            (
                r"C:\\Users\\alice\\local-data\\dump.bin",
                &["alice", "local-data", "dump.bin"],
            ),
            (
                r"\\corp-fileserver\xbox-builds\dev\game.xvc",
                &["corp-fileserver", "xbox-builds", "game.xvc"],
            ),
            (
                "//buildbox.local/share/private/game.exe",
                &["buildbox.local", "private", "game.exe"],
            ),
            (
                r"\\\\nas01\\builds\\game.exe",
                &["nas01", "builds", "game.exe"],
            ),
            (
                r"\\?\C:\Users\alice\local-data\game.exe",
                &["alice", "local-data", "game.exe"],
            ),
            (
                r"\\?\UNC\srv\share\secret.bin",
                &["srv", "share", "secret.bin"],
            ),
            (r"\\.\pipe\xbox-debug", &["pipe", "xbox-debug"]),
            (
                "/Users/alice/Documents/gta6/default.exe",
                &["/Users/", "alice", "Documents", "default.exe"],
            ),
            (
                "/home/bob/work/local-data/game.rpf",
                &["/home/", "bob", "local-data", "game.rpf"],
            ),
            ("/root/.ssh/id_ed25519", &["/root/", "id_ed25519"]),
            ("/tmp/recomp-dump-42.bin", &["/tmp/", "recomp-dump"]),
            ("/var/log/xbox/trace.log", &["/var/", "trace.log"]),
            ("/opt/xbox-sdk/bin/tool", &["/opt/", "xbox-sdk"]),
            ("/etc/passwd", &["/etc/", "passwd"]),
            (
                "/mnt/c/Users/alice/game.exe",
                &["/mnt/", "alice", "game.exe"],
            ),
            (
                "/Volumes/ExternalSSD/GTA6/default.exe",
                &["/Volumes/", "ExternalSSD", "default.exe"],
            ),
            ("/private/tmp/secret.bin", &["/private/", "secret.bin"]),
            ("/proc/self/maps", &["/proc/", "maps"]),
            ("/users/Alice/mixed-case/game.exe", &["Alice", "mixed-case"]),
            ("/cygdrive/c/Users/alice/a.exe", &["/cygdrive/", "alice"]),
            (
                "/wsl$/Ubuntu/home/bob/game.exe",
                &["/wsl$/", "Ubuntu", "bob"],
            ),
            ("~/local-data/game/default.exe", &["~/", "local-data"]),
            ("~alice/builds/default.exe", &["~alice", "builds"]),
            (
                "../../local-data/game/default.exe",
                &["../", "local-data", "default.exe"],
            ),
            (r"..\local-data\keys.txt", &["..\\", "local-data", "keys.txt"]),
            (
                "file:///C:/Users/alice/game.exe",
                &["file://", "alice", "game.exe"],
            ),
            (
                "FILE://localhost/Users/alice/game.exe",
                &["FILE://", "localhost", "alice"],
            ),
            (
                "File:/home/bob/secret.bin",
                &["File:/", "bob", "secret.bin"],
            ),
        ];

        for &(raw, forbidden) in cases {
            assert!(
                contains_host_path(raw),
                "contains_host_path failed to detect {raw:?}"
            );
            let redacted = redact_host_paths(raw);
            assert!(
                redacted.contains(REDACTED_PATH),
                "expected {REDACTED_PATH} in {redacted:?} for {raw:?}"
            );
            assert!(
                !contains_host_path(&redacted),
                "host path still detected after redaction: {raw:?} -> {redacted:?}"
            );
            for secret in forbidden {
                assert!(
                    !redacted.contains(secret),
                    "secret {secret:?} leaked in {redacted:?} (from {raw:?})"
                );
            }
        }

        // Embedded context around host paths is preserved cleanly.
        assert_eq!(
            redact_host_paths(r"import C:\Users\alice\d3d12_xs.dll!D3D12CreateDevice, out arg 3"),
            "import <redacted-path>!D3D12CreateDevice, out arg 3"
        );
        assert_eq!(
            redact_host_paths(
                r"source (C:\Program Files (x86)\Xbox\game.exe) and (/home/bob/game.exe)"
            ),
            "source (<redacted-path>) and (<redacted-path>)"
        );
    }

    #[test]
    fn adversarial_host_paths_never_leak_into_rendered_reports() {
        let s = Survey {
            sources: vec![
                SourceReport {
                    name: "dev|\x1b[31mC:\\Users\\alice\\game.exe\n`injected`".to_string(),
                    description: r"import \\corp-nas\builds\d3d12_xs.dll!Create, out arg 3"
                        .to_string(),
                    keys: vec![KeyReport {
                        key_args: vec![ArgKey::Const(1)],
                        source_calls: 1,
                        source_functions: 1,
                        out_stack: 1,
                        out_global: 0,
                        out_ret: 0,
                        globals: vec![0x1400_1000],
                        global_fields: vec![],
                        slots: vec![SlotReport {
                            slot: 5,
                            sites: 1,
                            unconfirmed_sites: 0,
                            functions: 1,
                            caller_sites: 1,
                            min_args: 1,
                            max_args: 1,
                            evidence: 0,
                            confidence: Confidence::High,
                            examples: vec![0x1400_2000],
                            unconfirmed_examples: vec![],
                        }],
                        unresolved: BTreeMap::from([
                            ("C:\\Users\\alice\\secret_a.pdb", 2),
                            ("/home/bob/secret_b.pdb", 3),
                        ]),
                    }],
                },
                SourceReport {
                    name: "file:///Volumes/SecretSSD/GTA6/title.exe".to_string(),
                    description: "proc ~/local-data/game/api.dll, out ret".to_string(),
                    keys: vec![],
                },
            ],
            totals: Totals {
                keys: 1,
                source_calls: 1,
                slots: 1,
                sites: 1,
                unconfirmed_sites: 0,
                functions: 1,
            },
            // Two distinct adversarial paths that both redact to `<redacted-path>`: counts must
            // aggregate deterministically without emitting duplicate JSON keys!
            unresolved: BTreeMap::from([
                ("C:\\Users\\alice\\secret_a.pdb", 2),
                ("/home/bob/secret_b.pdb", 3),
            ]),
            guard_slots: vec![],
            stats: Stats {
                functions_discovered: 10,
                functions_analyzed: 10,
                contexts: 10,
                rounds: 1,
                aborted: 0,
                context_limit_hit: false,
            },
        };

        let text = render_text(&s);
        let json = render_json(&s);
        let cjson = render_canonical_json(&s);
        let md = render_markdown(&s);

        JsonValidator::validate(&json, false)
            .expect("sanitized render_json must be valid with no duplicate keys");
        JsonValidator::validate(&cjson, true)
            .expect("sanitized render_canonical_json must be valid & strictly sorted");

        for (label, out) in [
            ("text", &text),
            ("json", &json),
            ("canonical_json", &cjson),
            ("markdown", &md),
        ] {
            assert!(
                !contains_host_path(out),
                "{label} output leaked host path:\n{out}"
            );
            for secret in [
                "alice",
                "bob",
                "corp-nas",
                "SecretSSD",
                "local-data",
                "secret_a.pdb",
                "secret_b.pdb",
                "\x1b",
            ] {
                assert!(
                    !out.contains(secret),
                    "{label} output leaked forbidden token {secret:?}:\n{out}"
                );
            }
        }

        // Colliding redacted unresolved keys aggregate cleanly to count 5.
        assert!(cjson.contains("\"<redacted-path>\": 5"));
        assert!(json.contains("\"<redacted-path>\": 5"));
        // Markdown heading and table cells must not contain raw injected newlines or unescaped `|`.
        assert!(md.contains("dev\\|<redacted-path> 'injected'"));
    }

    #[test]
    fn legitimate_survey_labels_and_guids_are_preserved_without_false_positives() {
        let legitimate = [
            "dev",
            "q",
            "import d3d12_xs.dll!D3D12XboxCreateDevice, out arg 3",
            "GetProcAddress \"QueryApiImpl\", out arg 2, key args 0+1",
            "function 0x140001000, out ret",
            "method dev:14, out arg 2",
            "global 0x140050000",
            "{43322110-6554-8776-98a9-bacbdcedfe0f}",
            "0x140020000+0x48",
            "0x140020000-0x10",
            "regs+xmm+stack+tail+sret",
            "this_only",
            "stored_to_unknown",
            "key {43322110-6554-8776-98a9-bacbdcedfe0f} / 0x7",
            "unconfirmed:0x140001020,0x140001040",
        ];

        for item in legitimate {
            assert!(
                !contains_host_path(item),
                "false positive host-path detection on {item:?}"
            );
            assert_eq!(
                redact_host_paths(item),
                item,
                "legitimate survey text mutated: {item:?}"
            );
        }
    }

    #[test]
    fn markdown_and_canonical_json_include_accurate_metrics_and_structure() {
        let s = Survey {
            sources: vec![SourceReport {
                name: "device".to_string(),
                description: "import d3d12_xs.dll!D3D12XboxCreateDevice, out arg 3".to_string(),
                keys: vec![KeyReport {
                    key_args: vec![synthetic_guid(1)],
                    source_calls: 4,
                    source_functions: 2,
                    out_stack: 3,
                    out_global: 1,
                    out_ret: 0,
                    globals: vec![0x1401_0000],
                    global_fields: vec![(0x1401_0000, 0x48), (0x1401_0000, -0x10)],
                    slots: vec![
                        SlotReport {
                            slot: 1,
                            sites: 3,
                            unconfirmed_sites: 1,
                            functions: 2,
                            caller_sites: 5,
                            min_args: 1,
                            max_args: 2,
                            evidence: 0,
                            confidence: Confidence::High,
                            examples: vec![0x1400_1000],
                            unconfirmed_examples: vec![0x1400_1100],
                        },
                        SlotReport {
                            slot: 2,
                            sites: 1,
                            unconfirmed_sites: 0,
                            functions: 1,
                            caller_sites: 2,
                            min_args: 3,
                            max_args: 3,
                            evidence: 0,
                            confidence: Confidence::Medium,
                            examples: vec![0x1400_2000],
                            unconfirmed_examples: vec![],
                        },
                        SlotReport {
                            slot: 3,
                            sites: 0,
                            unconfirmed_sites: 3,
                            functions: 0,
                            caller_sites: 0,
                            min_args: 1,
                            max_args: 1,
                            evidence: 0,
                            confidence: Confidence::Low,
                            examples: vec![],
                            unconfirmed_examples: vec![0x1400_3000],
                        },
                    ],
                    unresolved: BTreeMap::from([("passed_as_argument", 2)]),
                }],
            }],
            totals: Totals {
                keys: 1,
                source_calls: 4,
                slots: 2,
                sites: 4,
                unconfirmed_sites: 4,
                functions: 3,
            },
            unresolved: BTreeMap::from([("passed_as_argument", 2), ("stored_to_unknown", 6)]),
            guard_slots: vec![0x1409_0008],
            stats: Stats {
                functions_discovered: 200,
                functions_analyzed: 150,
                contexts: 300,
                rounds: 3,
                aborted: 3,
                context_limit_hit: false,
            },
        };

        let m = SurveyMetrics::from_survey(&s);
        assert_eq!(m.analyzed_function_pct, 75.0);
        assert_eq!(m.aborted_function_pct, 2.0);
        assert_eq!(m.confirmed_site_pct, 50.0);
        assert_eq!(m.unconfirmed_site_pct, 50.0);
        assert_eq!(m.avg_sites_per_slot, 2.0);
        assert_eq!(m.avg_slots_per_key, 2.0);
        assert_eq!(m.avg_calls_per_key, 4.0);
        assert_eq!(m.avg_contexts_per_round, 100.0);
        assert_eq!(m.total_unresolved, 8);
        assert_eq!(m.unresolved_per_source_call, 2.0);
        assert_eq!(m.total_slot_records, 3);
        assert_eq!(m.high_confidence_slot_pct, 33.33);
        assert_eq!(m.confidence_counts.get("high"), Some(&1));
        assert_eq!(m.confidence_counts.get("medium"), Some(&1));
        assert_eq!(m.confidence_counts.get("low"), Some(&1));

        let md = render_markdown(&s);
        assert!(md.contains("# Interface Survey Report"));
        assert!(md.contains("| Functions Analyzed | 150 (75.00%) |"));
        assert!(md.contains("| Aborted Functions | 3 (2.00%) |"));
        assert!(md.contains("| Confirmed Sites | 4 (50.00%, 2.00 avg/slot) |"));
        assert!(md.contains("| High-Confidence Slots | 1/3 (33.33%) |"));
        assert!(md.contains("| `passed_as_argument` | 2 | 25.00% |"));
        assert!(md.contains("| `stored_to_unknown` | 6 | 75.00% |"));
        assert!(md.contains("0x14010000-0x10,0x14010000+0x48"));

        let cjson = render_canonical_json(&s);
        JsonValidator::validate(&cjson, true).expect("canonical JSON valid & strictly sorted");
        assert!(cjson.contains("\"analyzed_function_pct\": 75.00"));
        assert!(cjson.contains("\"confirmed_site_pct\": 50.00"));
        assert!(cjson.contains("\"high_confidence_slot_pct\": 33.33"));
        assert!(!contains_host_path(&md));
        assert!(!contains_host_path(&cjson));
    }
}
