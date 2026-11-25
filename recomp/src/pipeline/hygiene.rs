//! Output-directory hygiene: build outputs are derived from game code and data, so they may only
//! be written outside every git work tree, or inside one only below a gitignored directory (by
//! convention the repo's `work/`).
//!
//! The check is done in-process, without running `git`:
//! 1. `--out` is validated against empty/control-character boundary inputs, made absolute
//!    (relative to the current directory), and normalised lexically (`.` dropped, `..` pops without
//!    escaping past root); the longest existing prefix is then canonicalised so symlinks and
//!    (on Windows) drive-letter spellings resolve to the real location.
//! 2. The enclosing work tree is the nearest ancestor (or the directory itself) containing a
//!    `.git` entry: a directory for a normal checkout, a file for linked worktrees and
//!    submodules. No `.git` anywhere above means the directory is outside every repository.
//! 3. Inside a work tree, the directory is accepted only if it or one of its ancestors below the
//!    root is ignored. Ignore rules come from `.git/info/exclude` (normal checkouts only) and from
//!    every `.gitignore` between the root and the directory, applied with git's precedence
//!    (deeper files override shallower ones, the last matching line wins, `!` re-includes). The
//!    pattern syntax follows gitignore(5): `#` comments, `\` escapes, `!` negation, trailing `/`
//!    (directory only; every path tested here is a directory), leading or inner `/` anchors the
//!    pattern to the `.gitignore`'s directory, `*`/`?`/`[...]` globs that never cross `/`, and
//!    `**` segments that match any number of directories. Matching is case-sensitive and uses
//!    polynomial-time dynamic programming so adversarial patterns cannot trigger exponential
//!    backtracking.
//! 4. All user-facing diagnostics and structured [`HygieneReport`] outputs are deterministically
//!    ordered (`BTreeMap`/`BTreeSet`) and stripped of ANSI escapes, control characters, and host
//!    filesystem paths (Windows drive paths, UNC/device paths, POSIX/MSYS absolute paths,
//!    `file://` URIs, home-relative `~` paths, and `..` traversals).
//!
//! Global excludes (`core.excludesFile`) are deliberately not consulted: they live outside the
//! repository, so a directory ignored only there is still visible to other clones. Anything the
//! parser cannot prove ignored is refused, so every approximation errs on the side of refusing.

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::{Component, Path, PathBuf};

/// Placeholder emitted in place of redacted host filesystem paths in sanitized outputs.
pub const REDACTED_PATH_TOKEN: &str = "<path>";

/// Maximum character length preserved by [`sanitize_diagnostic_text`].
pub const MAX_SANITIZED_TEXT_CHARS: usize = 4096;

/// Upper bound on individual `.gitignore` / `.git/info/exclude` file sizes read in-process.
/// Oversized ignore files are skipped (failing closed so unverified paths are refused).
const MAX_IGNORE_FILE_BYTES: u64 = 1024 * 1024;

/// Classification of one hygiene check performed by [`inspect_output_dir`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum HygieneStatus {
    AllowedOutsideRepo,
    AllowedGitignored,
    RefusedEmptyPath,
    RefusedInvalidPath,
    RefusedNotDirectory,
    RefusedGitMetadata,
    RefusedUntrackedInRepo,
    RefusedIoError,
}

impl HygieneStatus {
    /// Whether this status permits writing build outputs.
    #[allow(dead_code)]
    pub const fn is_allowed(self) -> bool {
        matches!(
            self,
            Self::AllowedOutsideRepo | Self::AllowedGitignored
        )
    }

    /// Stable, deterministic snake_case identifier for reports and metrics.
    #[allow(dead_code)]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AllowedOutsideRepo => "allowed_outside_repo",
            Self::AllowedGitignored => "allowed_gitignored",
            Self::RefusedEmptyPath => "refused_empty_path",
            Self::RefusedInvalidPath => "refused_invalid_path",
            Self::RefusedNotDirectory => "refused_not_directory",
            Self::RefusedGitMetadata => "refused_git_metadata",
            Self::RefusedUntrackedInRepo => "refused_untracked_in_repo",
            Self::RefusedIoError => "refused_io_error",
        }
    }
}

/// Sanitized, deterministic outcome for a single checked output path.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub struct HygieneEntry {
    /// Sanitized path label (`<repo>/...`, clean relative path, or `<path>`).
    pub label: String,
    /// Structured decision status.
    pub status: HygieneStatus,
    /// Sanitized refusal message when `!status.is_allowed()`.
    pub reason: Option<String>,
    /// Sanitized repo-relative forward-slash path when inside a git work tree.
    pub relative_path: Option<String>,
    /// Deterministic sorted set of repo-relative ignore files consulted.
    pub rule_sources: BTreeSet<String>,
}

/// Deterministic, host-path-free aggregate report across one or more checked output directories.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[allow(dead_code)]
pub struct HygieneReport {
    /// Total number of path checks recorded (including duplicate inputs).
    pub total_checked: usize,
    /// Per-label outcomes keyed in canonical lexicographical order.
    pub entries: BTreeMap<String, HygieneEntry>,
    /// Counts per [`HygieneStatus::as_str`] key in canonical lexicographical order.
    pub outcome_counts: BTreeMap<String, usize>,
    /// Sorted unique sanitized labels that were allowed (and never refused).
    pub allowed_labels: BTreeSet<String>,
    /// Sorted unique sanitized labels that were refused.
    pub refused_labels: BTreeSet<String>,
    /// Sorted unique repo-relative ignore files consulted across all checks.
    pub rule_sources: BTreeSet<String>,
    /// Sorted unique sanitized refusal diagnostics.
    pub diagnostics: BTreeSet<String>,
}

#[allow(dead_code)]
impl HygieneReport {
    /// Creates an empty report.
    pub fn new() -> Self {
        Self::default()
    }

    /// Evaluates every path in `paths` and returns a deterministic, permutation-invariant report.
    pub fn from_paths<'a, I>(paths: I) -> Self
    where
        I: IntoIterator<Item = &'a Path>,
    {
        let mut report = Self::new();
        for path in paths {
            report.record_path(path);
        }
        report
    }

    /// Evaluates `out` with [`inspect_output_dir`] and merges the result into this report.
    pub fn record_path(&mut self, out: &Path) {
        self.record_entry(inspect_output_dir(out));
    }

    /// Merges a pre-evaluated [`HygieneEntry`] into this report deterministically.
    ///
    /// If two inputs collapse to the same sanitized label (for example two distinct external host
    /// paths both redacted to `<path>`), any refusal takes precedence over an allow so a redaction
    /// collision can never mask a refusal.
    pub fn record_entry(&mut self, mut entry: HygieneEntry) {
        entry.label = sanitize_report_label(&entry.label);
        entry.reason = entry.reason.map(|msg| sanitize_diagnostic_text(&msg));
        entry.rule_sources = entry
            .rule_sources
            .into_iter()
            .map(|src| sanitize_path_str(&src))
            .collect();

        self.total_checked = self.total_checked.saturating_add(1);
        *self
            .outcome_counts
            .entry(entry.status.as_str().to_string())
            .or_insert(0) += 1;
        self.rule_sources.extend(entry.rule_sources.iter().cloned());
        if let Some(reason) = &entry.reason {
            if !reason.is_empty() {
                self.diagnostics.insert(reason.clone());
            }
        }

        let label = entry.label.clone();
        match self.entries.get_mut(&label) {
            Some(existing) => {
                existing
                    .rule_sources
                    .extend(entry.rule_sources.iter().cloned());
                let replace = (existing.status.is_allowed() && !entry.status.is_allowed())
                    || (existing.status.is_allowed() == entry.status.is_allowed()
                        && (entry.status, entry.reason.as_deref())
                            < (existing.status, existing.reason.as_deref()));
                if replace {
                    entry.rule_sources = existing.rule_sources.clone();
                    *existing = entry;
                }
            }
            None => {
                self.entries.insert(label.clone(), entry);
            }
        }

        let final_allowed = self
            .entries
            .get(&label)
            .is_some_and(|e| e.status.is_allowed());
        if final_allowed {
            self.refused_labels.remove(&label);
            self.allowed_labels.insert(label);
        } else {
            self.allowed_labels.remove(&label);
            self.refused_labels.insert(label);
        }
    }

    /// Returns `true` when at least zero checks were recorded and none were refused.
    pub fn is_all_allowed(&self) -> bool {
        self.refused_labels.is_empty()
    }

    /// Renders a deterministic, host-path-free plain-text summary suitable for logs or reports.
    pub fn render_summary(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(
            out,
            "hygiene_report: checked={} allowed={} refused={}",
            self.total_checked,
            self.allowed_labels.len(),
            self.refused_labels.len()
        );
        for (status, count) in &self.outcome_counts {
            let _ = writeln!(out, "  outcome.{status}={count}");
        }
        for source in &self.rule_sources {
            let _ = writeln!(out, "  rule_source={source}");
        }
        for (label, entry) in &self.entries {
            match &entry.reason {
                Some(reason) => {
                    let _ = writeln!(out, "  entry[{label}]={}: {reason}", entry.status.as_str());
                }
                None => {
                    let _ = writeln!(out, "  entry[{label}]={}", entry.status.as_str());
                }
            }
        }
        sanitize_diagnostic_text(&out)
    }
}

/// Checks that `out` may receive build outputs (see the module docs). `Err` carries a sanitized,
/// host-path-free message for the user.
pub fn check_output_dir(out: &Path) -> Result<(), String> {
    let entry = inspect_output_dir(out);
    if entry.status.is_allowed() {
        Ok(())
    } else {
        Err(entry.reason.unwrap_or_else(|| {
            "refusing to write build outputs to an unverified output directory".to_string()
        }))
    }
}

/// Inspects `out` and returns a structured, sanitized [`HygieneEntry`].
pub fn inspect_output_dir(out: &Path) -> HygieneEntry {
    let raw_lossy = out.to_string_lossy();
    if raw_lossy.contains('\0')
        || raw_lossy
            .chars()
            .any(|c| c.is_control() && c != '\t')
    {
        let reason = sanitize_diagnostic_text(
            "refusing to write build outputs to an invalid output path containing control characters",
        );
        return HygieneEntry {
            label: REDACTED_PATH_TOKEN.to_string(),
            status: HygieneStatus::RefusedInvalidPath,
            reason: Some(reason),
            relative_path: None,
            rule_sources: BTreeSet::new(),
        };
    }

    if raw_lossy.trim().is_empty() {
        let reason = sanitize_diagnostic_text(
            "refusing to write build outputs to an empty output path",
        );
        return HygieneEntry {
            label: REDACTED_PATH_TOKEN.to_string(),
            status: HygieneStatus::RefusedEmptyPath,
            reason: Some(reason),
            relative_path: None,
            rule_sources: BTreeSet::new(),
        };
    }

    let safe_out_label = sanitize_path_label(out);
    let absolute = if out.is_absolute() {
        out.to_path_buf()
    } else {
        match env::current_dir() {
            Ok(cwd) => cwd.join(out),
            Err(e) => {
                let reason = sanitize_diagnostic_text(&format!(
                    "cannot determine the current directory: {e}"
                ));
                return HygieneEntry {
                    label: safe_out_label,
                    status: HygieneStatus::RefusedIoError,
                    reason: Some(reason),
                    relative_path: None,
                    rule_sources: BTreeSet::new(),
                };
            }
        }
    };

    let path = resolve(&normalize(&absolute));
    if path.exists() && !path.is_dir() {
        let reason = sanitize_diagnostic_text(&format!(
            "output path '{safe_out_label}' exists and is not a directory"
        ));
        return HygieneEntry {
            label: safe_out_label,
            status: HygieneStatus::RefusedNotDirectory,
            reason: Some(reason),
            relative_path: None,
            rule_sources: BTreeSet::new(),
        };
    }

    let Some(root) = path.ancestors().find(|dir| dir.join(".git").exists()) else {
        return HygieneEntry {
            label: safe_out_label,
            status: HygieneStatus::AllowedOutsideRepo,
            reason: None,
            relative_path: None,
            rule_sources: BTreeSet::new(),
        };
    };

    let relative: Vec<String> = match path.strip_prefix(root) {
        Ok(rel) => rel
            .components()
            .map(|c| c.as_os_str().to_string_lossy().into_owned())
            .collect(),
        Err(_) => {
            let reason = sanitize_diagnostic_text(&format!(
                "cannot relate '{safe_out_label}' to its work tree"
            ));
            return HygieneEntry {
                label: safe_out_label,
                status: HygieneStatus::RefusedIoError,
                reason: Some(reason),
                relative_path: None,
                rule_sources: BTreeSet::new(),
            };
        }
    };

    let rel_display = sanitize_repo_relative(&relative);
    let repo_label = if rel_display.is_empty() {
        "<repo>".to_string()
    } else {
        format!("<repo>/{rel_display}")
    };

    if relative.first().is_some_and(|c| c == ".git") {
        let reason = sanitize_diagnostic_text(&format!(
            "refusing to write build outputs into git metadata ('{repo_label}')"
        ));
        return HygieneEntry {
            label: repo_label,
            status: HygieneStatus::RefusedGitMetadata,
            reason: Some(reason),
            relative_path: Some(rel_display),
            rule_sources: BTreeSet::new(),
        };
    }

    let rules = IgnoreRules::load(root, &relative);
    let rule_sources = rules.rule_sources.clone();
    if (1..=relative.len()).any(|len| rules.is_ignored(&relative[..len])) {
        return HygieneEntry {
            label: repo_label,
            status: HygieneStatus::AllowedGitignored,
            reason: None,
            relative_path: Some(rel_display),
            rule_sources,
        };
    }

    let reason = sanitize_diagnostic_text(&format!(
        "refusing to write build outputs to '{repo_label}': it is inside a git work tree and not in a \
         gitignored directory. Build outputs are derived from game code and data; write them to \
         the gitignored work/ directory instead (e.g. --out work/recomp) or outside the repository"
    ));
    HygieneEntry {
        label: repo_label,
        status: HygieneStatus::RefusedUntrackedInRepo,
        reason: Some(reason),
        relative_path: Some(rel_display),
        rule_sources,
    }
}

/// Sanitizes a diagnostic or error string so it is deterministic, uses `\n` line endings, is free
/// of ANSI escape sequences and control characters, is stripped of host filesystem paths, and is
/// bounded to [`MAX_SANITIZED_TEXT_CHARS`] Unicode scalar values.
pub fn sanitize_diagnostic_text(text: &str) -> String {
    let cleaned = strip_ansi_and_control_chars(text);
    let redacted = redact_host_paths(&cleaned);
    let trimmed = redacted.trim();
    truncate_chars(trimmed, MAX_SANITIZED_TEXT_CHARS)
}

/// Sanitizes a filesystem [`Path`] for safe display in diagnostics or reports.
///
/// Any host-absolute path, Windows drive path, UNC/device path, `file://` URI, home-relative `~`
/// path, parent traversal (`..`), or empty path is redacted to [`REDACTED_PATH_TOKEN`]. Clean
/// relative forward-slash paths (e.g. `work/recomp`) are normalized and preserved.
pub fn sanitize_path_label(path: &Path) -> String {
    sanitize_path_str(&path.to_string_lossy())
}

/// Sanitizes a raw path string with strict host-path redaction (see [`sanitize_path_label`]).
pub fn sanitize_path_str(raw: &str) -> String {
    let cleaned = strip_ansi_and_control_chars(raw);
    let trimmed = cleaned.trim();
    if trimmed.is_empty() {
        return REDACTED_PATH_TOKEN.to_string();
    }

    let norm = trimmed.replace('\\', "/");
    let lower = norm.to_ascii_lowercase();
    let is_host_or_traversal = lower.starts_with("file://")
        || norm.starts_with('/')
        || norm == "~"
        || match_tilde_home_prefix(norm.as_bytes()).is_some()
        || has_windows_drive_prefix(&norm)
        || norm.split('/').any(|seg| seg == "..")
        || redact_host_paths(&norm) != norm;

    if is_host_or_traversal {
        return REDACTED_PATH_TOKEN.to_string();
    }

    let mut segments = Vec::new();
    for seg in norm.split('/') {
        if seg.is_empty() || seg == "." {
            continue;
        }
        segments.push(seg);
    }
    if segments.is_empty() {
        REDACTED_PATH_TOKEN.to_string()
    } else {
        truncate_chars(&segments.join("/"), MAX_SANITIZED_TEXT_CHARS)
    }
}

/// Sanitizes an entry label stored in [`HygieneReport`], preserving `<repo>` / `<repo>/...` and
/// `<path>` markers while strictly redacting any host path.
#[allow(dead_code)]
fn sanitize_report_label(raw: &str) -> String {
    let cleaned = strip_ansi_and_control_chars(raw);
    let trimmed = cleaned.trim();
    if trimmed.is_empty() || trimmed == REDACTED_PATH_TOKEN {
        return REDACTED_PATH_TOKEN.to_string();
    }
    if trimmed == "<repo>" {
        return "<repo>".to_string();
    }
    if let Some(rest) = trimmed.strip_prefix("<repo>/") {
        let safe_rest = sanitize_path_str(rest);
        if safe_rest == REDACTED_PATH_TOKEN {
            return REDACTED_PATH_TOKEN.to_string();
        }
        return format!("<repo>/{safe_rest}");
    }
    sanitize_path_str(trimmed)
}

/// Sanitizes repository-relative path components into a clean forward-slash string.
fn sanitize_repo_relative(relative: &[String]) -> String {
    let mut parts = Vec::with_capacity(relative.len());
    for comp in relative {
        let cleaned = strip_ansi_and_control_chars(comp);
        let trimmed = cleaned.trim();
        if trimmed.is_empty()
            || trimmed == "."
            || trimmed == ".."
            || trimmed.contains('/')
            || trimmed.contains('\\')
            || has_windows_drive_prefix(trimmed)
        {
            parts.push("<segment>".to_string());
        } else {
            parts.push(redact_host_paths(trimmed));
        }
    }
    parts.join("/")
}

/// Redacts embedded host filesystem paths from free-form diagnostic text while preserving clean
/// relative paths, non-file web URLs (`http://`, `https://`), and trailing `:line` / `:line:col`
/// suffixes.
///
/// Covers:
/// - `file://` URIs (case-insensitive, local/authority/percent-encoded/quoted)
/// - Windows UNC & extended-length/device paths (`\\server\share\...`, `\\?\C:\...`, `\\.\...`,
///   and forward-slash `//server/share/...`, `//?/C:/...`)
/// - Windows drive paths (`C:\...`, `D:/...`, quoted paths with spaces)
/// - Home-relative paths (`~/...`, `~\...`, `~user/...`, `~user\...`)
/// - Parent-directory traversal paths (`../...`, `..\...`)
/// - POSIX & MSYS/WSL absolute paths (`/Users/...`, `/home/...`, `/tmp/...`, `/c/Users/...`,
///   `/mnt/c/...`, and multi-segment `/a/b` or known Unix roots)
pub fn redact_host_paths(text: &str) -> String {
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut out = String::with_capacity(len);
    let mut i = 0usize;

    while i < len {
        let at_boundary = i == 0 || is_path_boundary_before(bytes[i - 1]);
        if at_boundary {
            // Preserve non-file web URLs (`http://`, `https://`) without redacting their slashes.
            if starts_with_ascii_ignore_case(&bytes[i..], b"http://")
                || starts_with_ascii_ignore_case(&bytes[i..], b"https://")
            {
                let end = scan_unquoted_token_end(bytes, i);
                out.push_str(&text[i..end]);
                i = end;
                continue;
            }

            let opening_quote = if i > 0
                && matches!(bytes[i - 1], b'"' | b'\'' | b'`')
            {
                Some(bytes[i - 1])
            } else {
                None
            };

            if let Some((path_end, suffix_start)) = match_host_path(bytes, i, opening_quote) {
                out.push_str(REDACTED_PATH_TOKEN);
                if suffix_start < path_end {
                    out.push_str(&text[suffix_start..path_end]);
                }
                i = path_end;
                continue;
            }
        }

        let ch = text[i..].chars().next().unwrap_or(' ');
        out.push(ch);
        i += ch.len_utf8();
    }

    out
}

fn has_windows_drive_prefix(s: &str) -> bool {
    let bytes = s.as_bytes();
    bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':'
}

fn strip_ansi_and_control_chars(text: &str) -> String {
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut out = String::with_capacity(len);
    let mut i = 0usize;

    while i < len {
        let b = bytes[i];
        // Strip ANSI escape sequences: CSI (`ESC [ ... final`) and OSC (`ESC ] ... BEL/ST`).
        if b == 0x1b {
            if i + 1 < len && bytes[i + 1] == b'[' {
                i += 2;
                while i < len {
                    let c = bytes[i];
                    i += 1;
                    if (0x40..=0x7e).contains(&c) {
                        break;
                    }
                }
            } else if i + 1 < len && bytes[i + 1] == b']' {
                i += 2;
                while i < len {
                    let c = bytes[i];
                    if c == 0x07 {
                        i += 1;
                        break;
                    }
                    if c == 0x1b && i + 1 < len && bytes[i + 1] == b'\\' {
                        i += 2;
                        break;
                    }
                    i += 1;
                }
            } else {
                i += 1;
            }
            continue;
        }
        if b == b'\r' {
            if i + 1 < len && bytes[i + 1] == b'\n' {
                i += 2;
            } else {
                i += 1;
            }
            out.push('\n');
            continue;
        }
        if b == b'\n' || b == b'\t' {
            out.push(b as char);
            i += 1;
            continue;
        }
        if b < 0x20 || b == 0x7f {
            i += 1;
            continue;
        }
        let ch = text[i..].chars().next().unwrap_or(' ');
        out.push(ch);
        i += ch.len_utf8();
    }

    out
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }
    let mut chars = text.chars();
    let mut kept = String::new();
    for _ in 0..max_chars {
        let Some(ch) = chars.next() else {
            return text.to_string();
        };
        kept.push(ch);
    }
    if chars.next().is_none() {
        return text.to_string();
    }
    if max_chars <= 3 {
        return ".".repeat(max_chars);
    }
    let trim_to = max_chars - 3;
    let mut out: String = text.chars().take(trim_to).collect();
    out.push_str("...");
    out
}

const fn is_path_boundary_before(b: u8) -> bool {
    matches!(
        b,
        b' ' | b'\t'
            | b'\n'
            | b'"'
            | b'\''
            | b'`'
            | b'('
            | b'['
            | b'{'
            | b'<'
            | b'='
            | b','
            | b';'
            | b':'
    )
}

const fn is_unquoted_path_stop(b: u8) -> bool {
    b.is_ascii_whitespace()
        || matches!(
            b,
            b'"' | b'\'' | b'`' | b'(' | b')' | b'[' | b']' | b'{' | b'}' | b'<' | b'>' | b','
                | b';'
        )
}

fn starts_with_ascii_ignore_case(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.len() >= needle.len()
        && haystack[..needle.len()]
            .iter()
            .zip(needle.iter())
            .all(|(a, b)| a.to_ascii_lowercase() == b.to_ascii_lowercase())
}

fn scan_unquoted_token_end(bytes: &[u8], start: usize) -> usize {
    let mut end = start;
    while end < bytes.len() && !is_unquoted_path_stop(bytes[end]) {
        end += 1;
    }
    end
}

fn scan_path_token_end(bytes: &[u8], start: usize, opening_quote: Option<u8>) -> usize {
    if let Some(q) = opening_quote {
        let mut cursor = start;
        while cursor < bytes.len() && bytes[cursor] != b'\n' {
            if bytes[cursor] == q {
                return cursor;
            }
            cursor += 1;
        }
    }
    let mut end = scan_unquoted_token_end(bytes, start);
    // Keep sentence-ending trailing `.` or prose `:` outside the redacted token unless it's
    // `..`, `/.`, `\.`, or part of a bare drive prefix.
    while end > start + 2 {
        let last = bytes[end - 1];
        let prev = bytes[end - 2];
        if last == b'.' && prev != b'.' && prev != b'/' && prev != b'\\' {
            end -= 1;
        } else if last == b':' {
            end -= 1;
        } else {
            break;
        }
    }
    end
}

/// Matches `~/`, `~\`, `~user/`, or `~user\` at the start of `rem`, returning the prefix length
/// including the separator.
fn match_tilde_home_prefix(rem: &[u8]) -> Option<usize> {
    if rem.len() < 2 || rem[0] != b'~' {
        return None;
    }
    if rem[1] == b'/' || rem[1] == b'\\' {
        return Some(2);
    }
    let mut k = 1usize;
    while k < rem.len()
        && (rem[k].is_ascii_alphanumeric() || matches!(rem[k], b'_' | b'-' | b'.'))
    {
        k += 1;
    }
    if k > 1 && k < rem.len() && (rem[k] == b'/' || rem[k] == b'\\') {
        Some(k + 1)
    } else {
        None
    }
}

/// If `bytes[start..]` begins a host filesystem path, returns `(token_end, suffix_start)`
/// where `bytes[start..suffix_start]` is replaced with `<path>` and `bytes[suffix_start..token_end]`
/// (such as `:42:10`) is preserved.
fn match_host_path(
    bytes: &[u8],
    start: usize,
    opening_quote: Option<u8>,
) -> Option<(usize, usize)> {
    let rem = &bytes[start..];

    // 1. `file://...` URI (case-insensitive).
    if starts_with_ascii_ignore_case(rem, b"file://") {
        let end = scan_path_token_end(bytes, start, opening_quote);
        if end >= start + 7 {
            let suffix = if end > start + 7 {
                find_line_col_suffix(bytes, start + 7, end)
            } else {
                end
            };
            return Some((end, suffix));
        }
        return None;
    }

    // 2. UNC or Windows extended-length/device path (`\\server\share\...`, `\\?\C:\...`, `\\.\...`).
    if rem.len() >= 3
        && rem[0] == b'\\'
        && rem[1] == b'\\'
        && (rem[2].is_ascii_alphanumeric() || matches!(rem[2], b'?' | b'.' | b'_'))
    {
        let end = scan_path_token_end(bytes, start, opening_quote);
        let suffix = find_line_col_suffix(bytes, start + 2, end);
        return Some((end, suffix));
    }

    // 2b. Forward-slash UNC / extended-length path (`//server/share/...`, `//?/C:/...`, `//./...`).
    if rem.len() >= 3
        && rem[0] == b'/'
        && rem[1] == b'/'
        && (rem[2].is_ascii_alphanumeric() || matches!(rem[2], b'?' | b'.' | b'_'))
    {
        let end = scan_path_token_end(bytes, start, opening_quote);
        let token = &bytes[start..end];
        if token[2..].contains(&b'/') {
            let suffix = find_line_col_suffix(bytes, start + 2, end);
            return Some((end, suffix));
        }
    }

    // 3. Windows drive path (`C:\...` or `C:/...`, including root `C:\` / `C:/`).
    if rem.len() >= 3
        && rem[0].is_ascii_alphabetic()
        && rem[1] == b':'
        && (rem[2] == b'\\' || rem[2] == b'/')
    {
        let end = scan_path_token_end(bytes, start, opening_quote);
        let suffix = find_line_col_suffix(bytes, start + 3, end);
        return Some((end, suffix));
    }

    // 4. Home-relative path (`~/...`, `~\...`, `~user/...`, `~user\...`).
    if let Some(prefix_len) = match_tilde_home_prefix(rem) {
        if rem.len() > prefix_len && !is_unquoted_path_stop(rem[prefix_len]) {
            let end = scan_path_token_end(bytes, start, opening_quote);
            let suffix = find_line_col_suffix(bytes, start + prefix_len, end);
            return Some((end, suffix));
        }
    }

    // 5. Parent-directory traversal (`../...` or `..\...`).
    if rem.len() >= 4
        && rem[0] == b'.'
        && rem[1] == b'.'
        && (rem[2] == b'/' || rem[2] == b'\\')
        && !is_unquoted_path_stop(rem[3])
    {
        let end = scan_path_token_end(bytes, start, opening_quote);
        let suffix = find_line_col_suffix(bytes, start + 3, end);
        return Some((end, suffix));
    }

    // 6. POSIX / MSYS / WSL absolute path (`/Users/...`, `/home/...`, `/c/Users/...`, or `/a/b`).
    if rem.len() >= 2
        && rem[0] == b'/'
        && (rem[1].is_ascii_alphanumeric() || matches!(rem[1], b'.' | b'_' | b'@' | b'-'))
    {
        let end = scan_path_token_end(bytes, start, opening_quote);
        let token = &bytes[start..end];
        let has_second_slash = token[1..].contains(&b'/');
        let is_known_unix_root = matches!(
            token,
            b"/Users"
                | b"/home"
                | b"/tmp"
                | b"/var"
                | b"/etc"
                | b"/opt"
                | b"/mnt"
                | b"/root"
                | b"/private"
                | b"/Volumes"
                | b"/workspace"
                | b"/usr"
                | b"/bin"
                | b"/sbin"
                | b"/lib"
                | b"/proc"
                | b"/sys"
                | b"/dev"
                | b"/srv"
                | b"/run"
        );
        if has_second_slash || is_known_unix_root {
            let suffix = find_line_col_suffix(bytes, start + 1, end);
            return Some((end, suffix));
        }
    }

    None
}

/// Detect optional trailing `:line` or `:line:col` suffix inside `bytes[min_prefix_end..token_end]`.
fn find_line_col_suffix(bytes: &[u8], min_prefix_end: usize, token_end: usize) -> usize {
    let mut split = token_end;
    for _ in 0..2 {
        let mut cursor = split;
        while cursor > min_prefix_end && bytes[cursor - 1].is_ascii_digit() {
            cursor -= 1;
        }
        if cursor < split && cursor > min_prefix_end && bytes[cursor - 1] == b':' {
            split = cursor - 1;
        } else {
            break;
        }
    }
    split
}

/// Lexical normalisation: drops `.`, resolves `..` against the preceding normal component, and
/// never pushes `..` past a root directory.
fn normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    let mut has_root = false;
    let mut normal_depth = 0usize;
    for component in path.components() {
        match component {
            Component::Prefix(_) => {
                out.push(component);
            }
            Component::RootDir => {
                out.push(component);
                has_root = true;
            }
            Component::CurDir => {}
            Component::ParentDir => {
                if normal_depth > 0 {
                    out.pop();
                    normal_depth -= 1;
                } else if !has_root {
                    out.push(component);
                }
            }
            Component::Normal(part) => {
                out.push(part);
                normal_depth += 1;
            }
        }
    }
    out
}

/// Canonicalises the longest existing prefix of `path` and appends the rest.
fn resolve(path: &Path) -> PathBuf {
    let mut existing = path;
    let mut rest = Vec::new();
    loop {
        if let Ok(canonical) = fs::canonicalize(existing) {
            let mut out = canonical;
            out.extend(rest.iter().rev());
            return out;
        }
        match (existing.parent(), existing.file_name()) {
            (Some(parent), Some(name)) => {
                rest.push(name.to_os_string());
                existing = parent;
            }
            _ => return path.to_path_buf(),
        }
    }
}

/// One parsed ignore line.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Pattern {
    negated: bool,
    /// Matched against the path relative to `base` (otherwise against the last component only).
    anchored: bool,
    segments: Vec<String>,
    /// Number of leading path components of the directory holding the ignore file.
    base: usize,
}

impl Pattern {
    fn parse(line: &str, base: usize) -> Option<Self> {
        if line.contains('\0') {
            return None;
        }
        let line = line.strip_suffix('\r').unwrap_or(line);
        let line = trim_trailing_spaces(line);
        if line.is_empty() || line.starts_with('#') {
            return None;
        }
        let (negated, body) = match line.strip_prefix('!') {
            Some(rest) => (true, rest),
            None => (false, line),
        };
        let body = body.trim_end_matches('/');
        if body.is_empty() {
            return None;
        }
        let anchored = body.contains('/');
        let mut segments: Vec<String> = Vec::new();
        for seg in body.trim_start_matches('/').split('/') {
            if seg.is_empty() {
                continue;
            }
            // Collapse consecutive `**` segments into one for canonical matching.
            if seg == "**" && segments.last().is_some_and(|prev| prev == "**") {
                continue;
            }
            segments.push(seg.to_string());
        }
        if segments.is_empty() {
            return None;
        }
        Some(Self {
            negated,
            anchored,
            segments,
            base,
        })
    }

    /// Whether the pattern matches the directory `path` (components relative to the work tree).
    fn matches(&self, path: &[String]) -> bool {
        if path.len() <= self.base {
            return false;
        }
        let relative = &path[self.base..];
        if self.anchored {
            match_segments(&self.segments, relative)
        } else {
            relative
                .last()
                .is_some_and(|name| glob(self.segments[0].as_bytes(), name.as_bytes()))
        }
    }
}

/// Trailing spaces are ignored unless escaped with a backslash.
fn trim_trailing_spaces(line: &str) -> &str {
    let mut end = line.len();
    while end > 0 && line.as_bytes()[end - 1] == b' ' {
        if end >= 2 && line.as_bytes()[end - 2] == b'\\' {
            break;
        }
        end -= 1;
    }
    &line[..end]
}

/// Matches pattern segments against path components; a `**` segment spans any number of them.
/// Implemented with iterative dynamic programming (`O(pattern.len() * path.len())`) to avoid
/// exponential backtracking on repeated `**` patterns.
fn match_segments(pattern: &[String], path: &[String]) -> bool {
    let n = path.len();
    let mut dp = vec![false; n + 1];
    dp[0] = true;

    for seg in pattern {
        if seg == "**" {
            let mut seen = false;
            for slot in &mut dp {
                seen |= *slot;
                *slot = seen;
            }
        } else {
            let mut next = vec![false; n + 1];
            for j in 0..n {
                if dp[j] && glob(seg.as_bytes(), path[j].as_bytes()) {
                    next[j + 1] = true;
                }
            }
            dp = next;
        }
    }
    dp[n]
}

enum GlobToken {
    Star,
    Any,
    Literal(u8),
    CharClass(Class),
}

fn compile_glob(pattern: &[u8]) -> Vec<GlobToken> {
    let mut tokens = Vec::with_capacity(pattern.len());
    let mut rest = pattern;
    while let Some((&b, tail)) = rest.split_first() {
        match b {
            b'*' => {
                if !matches!(tokens.last(), Some(GlobToken::Star)) {
                    tokens.push(GlobToken::Star);
                }
                rest = tail;
            }
            b'?' => {
                tokens.push(GlobToken::Any);
                rest = tail;
            }
            b'[' => match class(tail) {
                Some((set, after)) => {
                    tokens.push(GlobToken::CharClass(set));
                    rest = after;
                }
                // An unterminated class is a literal `[`.
                None => {
                    tokens.push(GlobToken::Literal(b'['));
                    rest = tail;
                }
            },
            b'\\' if !tail.is_empty() => {
                tokens.push(GlobToken::Literal(tail[0]));
                rest = &tail[1..];
            }
            other => {
                tokens.push(GlobToken::Literal(other));
                rest = tail;
            }
        }
    }
    tokens
}

/// Glob match of one segment: `*`, `?`, `[...]` (with `!`/`^` negation and ranges), `\` escapes.
/// Uses iterative dynamic programming so adversarial repeated `*` patterns run in polynomial time.
fn glob(pattern: &[u8], name: &[u8]) -> bool {
    let tokens = compile_glob(pattern);
    let n = name.len();
    let mut dp = vec![false; n + 1];
    dp[0] = true;

    for token in &tokens {
        match token {
            GlobToken::Star => {
                let mut seen = false;
                for slot in &mut dp {
                    seen |= *slot;
                    *slot = seen;
                }
            }
            GlobToken::Any => {
                let mut next = vec![false; n + 1];
                next[1..=n].copy_from_slice(&dp[..n]);
                dp = next;
            }
            GlobToken::Literal(expected) => {
                let mut next = vec![false; n + 1];
                for j in 0..n {
                    if dp[j] && name[j] == *expected {
                        next[j + 1] = true;
                    }
                }
                dp = next;
            }
            GlobToken::CharClass(set) => {
                let mut next = vec![false; n + 1];
                for j in 0..n {
                    if dp[j] && set.matches(name[j]) {
                        next[j + 1] = true;
                    }
                }
                dp = next;
            }
        }
    }
    dp[n]
}

struct Class {
    negated: bool,
    ranges: Vec<(u8, u8)>,
}

impl Class {
    fn matches(&self, c: u8) -> bool {
        self.ranges.iter().any(|&(lo, hi)| (lo..=hi).contains(&c)) != self.negated
    }
}

/// Parses the body of a `[...]` class (after the `[`). Returns the class and the pattern after `]`.
fn class(pattern: &[u8]) -> Option<(Class, &[u8])> {
    let (negated, mut rest) = match pattern.first() {
        Some(b'!' | b'^') => (true, &pattern[1..]),
        _ => (false, pattern),
    };
    let mut ranges = Vec::new();
    let mut first = true;
    loop {
        let (&c, tail) = rest.split_first()?;
        if c == b']' && !first {
            return Some((Class { negated, ranges }, tail));
        }
        first = false;
        if tail.len() >= 2 && tail[0] == b'-' && tail[1] != b']' {
            ranges.push((c, tail[1]));
            rest = &tail[2..];
        } else {
            ranges.push((c, c));
            rest = tail;
        }
    }
}

/// Every ignore rule that can apply to a path below the work tree, lowest precedence first.
struct IgnoreRules {
    patterns: Vec<Pattern>,
    /// Deterministic sorted set of repo-relative ignore sources consulted.
    rule_sources: BTreeSet<String>,
}

impl IgnoreRules {
    /// Loads `.git/info/exclude` and the `.gitignore` of the root and of every directory on the
    /// way to `relative` (exclusive of the last component: a directory's own `.gitignore` only
    /// applies to its contents).
    fn load(root: &Path, relative: &[String]) -> Self {
        let mut patterns = Vec::new();
        let mut rule_sources = BTreeSet::new();
        let mut add = |file: PathBuf, base: usize, rel_source: String| {
            let Ok(meta) = fs::metadata(&file) else {
                return;
            };
            if !meta.is_file() || meta.len() > MAX_IGNORE_FILE_BYTES {
                return;
            }
            if let Ok(text) = fs::read_to_string(&file) {
                rule_sources.insert(sanitize_path_str(&rel_source));
                patterns.extend(text.lines().filter_map(|line| Pattern::parse(line, base)));
            }
        };
        let git = root.join(".git");
        if git.is_dir() {
            add(
                git.join("info").join("exclude"),
                0,
                ".git/info/exclude".to_string(),
            );
        }
        let mut dir = root.to_path_buf();
        let mut rel_prefix: Vec<String> = Vec::new();
        for (depth, component) in relative.iter().enumerate() {
            let source_label = if rel_prefix.is_empty() {
                ".gitignore".to_string()
            } else {
                format!("{}/.gitignore", rel_prefix.join("/"))
            };
            add(dir.join(".gitignore"), depth, source_label);
            dir.push(component);
            rel_prefix.push(component.clone());
        }
        Self {
            patterns,
            rule_sources,
        }
    }

    fn is_ignored(&self, path: &[String]) -> bool {
        self.patterns
            .iter()
            .rev()
            .find(|p| p.matches(path))
            .is_some_and(|p| !p.negated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct ScopedTempDir(PathBuf);

    impl ScopedTempDir {
        fn new(tag: &str) -> Self {
            static NEXT_ID: AtomicU64 = AtomicU64::new(0);
            let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
            let dir = env::temp_dir().join(format!(
                "playgta6-hygiene-{tag}-{}-{id}",
                std::process::id()
            ));
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(&dir).expect("create temp dir");
            Self(dir)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for ScopedTempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn comps(path: &str) -> Vec<String> {
        path.split('/').map(str::to_string).collect()
    }

    fn rules(lines: &[(&str, usize)]) -> IgnoreRules {
        IgnoreRules {
            patterns: lines
                .iter()
                .filter_map(|&(line, base)| Pattern::parse(line, base))
                .collect(),
            rule_sources: BTreeSet::new(),
        }
    }

    #[test]
    fn gitignore_pattern_semantics() {
        let r = rules(&[
            ("# comment", 0),
            ("work/", 0),
            ("/build-*", 0),
            ("out*/", 0),
            ("!out-keep/", 0),
            ("docs/**/gen", 0),
            ("cache[0-9]", 0),
            ("local", 1),
            ("\\#hash", 0),
            ("trailing\\ ", 0),
        ]);
        let ignored = |p: &str| r.is_ignored(&comps(p));
        assert!(ignored("work"));
        assert!(
            ignored("a/b/work"),
            "unanchored patterns match at any depth"
        );
        assert!(ignored("build-x"));
        assert!(!ignored("sub/build-x"), "leading / anchors to the root");
        assert!(ignored("out-a"));
        assert!(!ignored("out-keep"), "! re-includes");
        assert!(ignored("docs/gen"));
        assert!(ignored("docs/a/b/gen"), "** spans directories");
        assert!(!ignored("docs/a/b/gen2"));
        assert!(ignored("cache7"));
        assert!(!ignored("cachex"));
        assert!(
            ignored("sub/local"),
            "base-relative pattern from sub/.gitignore"
        );
        assert!(
            !ignored("local"),
            "pattern of sub/.gitignore does not apply above it"
        );
        assert!(ignored("#hash"));
        assert!(ignored("trailing "));
        assert!(!ignored("src"));
    }

    #[test]
    fn glob_classes_and_escapes() {
        assert!(glob(b"a?c", b"abc"));
        assert!(!glob(b"a?c", b"ac"));
        assert!(glob(b"[!a]x", b"bx"));
        assert!(!glob(b"[!a]x", b"ax"));
        assert!(glob(b"[]]", b"]"));
        assert!(glob(b"[a", b"[a"), "unterminated class is literal");
        assert!(glob(b"\\*", b"*"));
        assert!(!glob(b"\\*", b"x"));
        assert!(glob(b"*.pgim", b"game.pgim"));
    }

    #[test]
    fn normalizes_dot_segments() {
        let p = normalize(Path::new("/a/./b/../c"));
        assert_eq!(p, PathBuf::from("/a/c"));
    }

    #[test]
    fn redacts_adversarial_windows_unc_posix_and_file_uri_paths() {
        let cases = [
            // Windows drive paths (backslashes, forward slashes, root, spaces, :line:col)
            (
                r"failed at C:\Users\alice\GTA6\out\main.ll:42:10",
                "failed at <path>:42:10",
            ),
            (
                "output=D:/Games/SecretBuild/work and E:\\ and F:\\work.",
                "output=<path> and <path> and <path>.",
            ),
            (
                "dot-suffix E:\\. and /tmp/.",
                "dot-suffix <path> and <path>",
            ),
            (
                r#"quoted "C:\Users\Alice Smith\My Game\out" and 'D:/Program Files/Xbox/out'"#,
                r#"quoted "<path>" and '<path>'"#,
            ),
            (
                r"inline `C:\Users\Alice Smith\repo\out` ok",
                "inline `<path>` ok",
            ),
            // Windows UNC, extended-length, device, and forward-slash UNC paths
            (
                r"unc \\fileserver\share\alice\dump.bin and \\?\C:\Users\alice\long\path",
                "unc <path> and <path>",
            ),
            (
                r"device \\.\pipe\playgta6 and \\?\UNC\nas01\builds\out",
                "device <path> and <path>",
            ),
            (
                "posix-unc //fileserver/share/builds/out and //?/C:/Users/alice/out",
                "posix-unc <path> and <path>",
            ),
            // POSIX, macOS, WSL, and MSYS absolute paths
            (
                "unix /Users/alice/dev/repo/src:19:4 and /home/bob/work/out and /tmp",
                "unix <path>:19:4 and <path> and <path>",
            ),
            (
                "msys /c/Users/alice/repo and wsl /mnt/c/Users/alice/game",
                "msys <path> and wsl <path>",
            ),
            (
                r#"spaced "/Users/Alice Smith/My Project/out": denied"#,
                r#"spaced "<path>": denied"#,
            ),
            // file:// URIs (mixed case, authority, percent-encoding, quoted, bare)
            (
                "uri file:///C:/Users/alice/My%20Game/out and FILE://localhost/home/bob/x.ll:7:2",
                "uri <path> and <path>:7:2",
            ),
            (
                r#"quoted "file:///C:/Users/Alice Smith/out" and bare file://"#,
                r#"quoted "<path>" and bare <path>"#,
            ),
            // Home-relative and parent-directory traversal paths
            (
                r"home ~/secret/dir, ~\win\dir, ~alice/proj/out and traversal ../escape/dir ..\escape\win",
                "home <path>, <path>, <path> and traversal <path> <path>",
            ),
            // Clean relative paths and HTTP(S) URLs stay intact
            (
                "use --out work/recomp (see https://github.com/PlayGta6Web/playgta6web and http://localhost:8080/app)",
                "use --out work/recomp (see https://github.com/PlayGta6Web/playgta6web and http://localhost:8080/app)",
            ),
        ];

        for (input, expected) in cases {
            let actual = redact_host_paths(input);
            assert_eq!(actual, expected, "input: {input}");
            for forbidden in ["alice", "Alice", "bob", "fileserver", "nas01", "C:\\", "D:/", "/Users/", "/home/", "file://"] {
                assert!(
                    !actual.contains(forbidden),
                    "leaked `{forbidden}` in `{actual}` from `{input}`"
                );
            }
        }
    }

    #[test]
    fn sanitizes_ansi_escapes_control_chars_and_bounds_length() {
        let noisy = "\x1b[31mERROR\x1b[0m\x1b]0;secret-title\x07: \x00bad\x01 path C:\\Users\\alice\\repo\\out\r\nline2\rline3\x7f  ";
        let sanitized = sanitize_diagnostic_text(noisy);
        assert_eq!(sanitized, "ERROR: bad path <path>\nline2\nline3");
        assert!(!sanitized.contains('\x1b'));
        assert!(!sanitized.contains('\0'));
        assert!(!sanitized.contains('\r'));
        assert!(!sanitized.contains("alice"));

        // Deterministic truncation respecting UTF-8 char boundaries.
        let long_input = "é".repeat(MAX_SANITIZED_TEXT_CHARS + 128);
        let truncated = sanitize_diagnostic_text(&long_input);
        assert_eq!(truncated.chars().count(), MAX_SANITIZED_TEXT_CHARS);
        assert!(truncated.ends_with("..."));
    }

    #[test]
    fn sanitize_path_str_handles_relative_and_adversarial_inputs() {
        assert_eq!(sanitize_path_str(""), REDACTED_PATH_TOKEN);
        assert_eq!(sanitize_path_str("   \t\r\n "), REDACTED_PATH_TOKEN);
        assert_eq!(sanitize_path_str("."), REDACTED_PATH_TOKEN);
        assert_eq!(sanitize_path_str("././"), REDACTED_PATH_TOKEN);
        assert_eq!(sanitize_path_str("/"), REDACTED_PATH_TOKEN);
        assert_eq!(sanitize_path_str("~"), REDACTED_PATH_TOKEN);
        assert_eq!(sanitize_path_str("~/work"), REDACTED_PATH_TOKEN);
        assert_eq!(sanitize_path_str("~alice/work"), REDACTED_PATH_TOKEN);
        assert_eq!(sanitize_path_str("../work"), REDACTED_PATH_TOKEN);
        assert_eq!(sanitize_path_str("work/../src"), REDACTED_PATH_TOKEN);
        assert_eq!(sanitize_path_str(r"C:\Users\alice\work"), REDACTED_PATH_TOKEN);
        assert_eq!(sanitize_path_str("C:"), REDACTED_PATH_TOKEN);
        assert_eq!(sanitize_path_str(r"\\server\share\out"), REDACTED_PATH_TOKEN);
        assert_eq!(sanitize_path_str("//server/share/out"), REDACTED_PATH_TOKEN);
        assert_eq!(sanitize_path_str("file:///home/alice/out"), REDACTED_PATH_TOKEN);
        assert_eq!(sanitize_path_str("/home/alice/out"), REDACTED_PATH_TOKEN);

        // Clean relative paths normalize separators and redundant `./` or `//`.
        assert_eq!(sanitize_path_str(r".\work\\recomp\."), "work/recomp");
        assert_eq!(sanitize_path_str("  work/recomp  "), "work/recomp");
    }

    #[test]
    fn guards_empty_invalid_and_adversarial_boundary_inputs() {
        // Empty and whitespace-only paths are explicitly refused.
        let empty = inspect_output_dir(Path::new(""));
        assert_eq!(empty.status, HygieneStatus::RefusedEmptyPath);
        assert!(check_output_dir(Path::new("")).unwrap_err().contains("refusing"));

        let blank = inspect_output_dir(Path::new("   \t  "));
        assert_eq!(blank.status, HygieneStatus::RefusedEmptyPath);

        // NUL and control characters in output path are explicitly refused.
        let with_nul = inspect_output_dir(Path::new("work/\0evil"));
        assert_eq!(with_nul.status, HygieneStatus::RefusedInvalidPath);
        let with_ansi = inspect_output_dir(Path::new("\x1b[31mwork/out"));
        assert_eq!(with_ansi.status, HygieneStatus::RefusedInvalidPath);

        // normalize clamps `..` at root and preserves leading relative `..`.
        assert_eq!(normalize(Path::new("")), PathBuf::new());
        assert_eq!(normalize(Path::new(".")), PathBuf::new());
        assert_eq!(normalize(Path::new("/../../a")), PathBuf::from("/a"));
        assert_eq!(normalize(Path::new("../a/../b")), PathBuf::from("../b"));
        assert_eq!(normalize(Path::new("a/../../b")), PathBuf::from("../b"));

        // Slash-only, blank, and NUL patterns are ignored (never produce empty-segment matches).
        for degenerate in ["", "   ", "\r", "/", "///", "!/", "!///", "work/\0bad"] {
            assert_eq!(Pattern::parse(degenerate, 0), None, "pattern: {degenerate:?}");
        }

        // Adversarial repeated `*` and `**` inputs finish immediately without backtracking blowup.
        let evil_glob = format!("{}z", "*".repeat(64));
        let long_name = "a".repeat(128);
        assert!(!glob(evil_glob.as_bytes(), long_name.as_bytes()));

        let evil_pat = Pattern::parse(&format!("{}z", "**/".repeat(32)), 0).expect("valid pattern");
        let deep_path: Vec<String> = (0..64).map(|i| format!("d{i}")).collect();
        assert!(!evil_pat.matches(&deep_path));
    }

    #[test]
    fn check_output_dir_errors_never_leak_host_paths() {
        let tmp = ScopedTempDir::new("redact-err");
        let repo = tmp.path().join("repo");
        fs::create_dir_all(repo.join(".git").join("info")).expect("fake repo");
        fs::write(repo.join(".gitignore"), "work/\n").expect("gitignore");
        let not_a_dir = tmp.path().join("regular-file.bin");
        fs::write(&not_a_dir, b"x").expect("regular file");

        let host_root = tmp.path().to_string_lossy().into_owned();
        let host_basename = tmp
            .path()
            .file_name()
            .expect("temp dir name")
            .to_string_lossy()
            .into_owned();

        let bad_paths = [
            repo.clone(),
            repo.join("src"),
            repo.join(".git").join("objects"),
            not_a_dir,
        ];

        for bad in &bad_paths {
            let err = check_output_dir(bad).expect_err("must be refused");
            assert!(
                !err.contains(&host_root) && !err.contains(&host_basename),
                "error leaked host path `{host_basename}`: {err}"
            );
            assert!(
                !err.contains(":\\") && !err.contains(":/"),
                "error leaked Windows drive prefix: {err}"
            );
            assert!(
                !err.contains("\\\\"),
                "error leaked UNC prefix: {err}"
            );
        }

        let untracked_err = check_output_dir(&repo.join("src")).unwrap_err();
        assert!(untracked_err.contains("refusing"));
        assert!(untracked_err.contains("<repo>/src"));
        assert!(untracked_err.contains("work/"));

        let git_err = check_output_dir(&repo.join(".git").join("objects")).unwrap_err();
        assert!(git_err.contains("refusing"));
        assert!(git_err.contains("<repo>/.git/objects"));
    }

    #[test]
    fn hygiene_report_is_deterministic_and_permutation_invariant() {
        let tmp = ScopedTempDir::new("determinism");
        let repo = tmp.path().join("repo");
        fs::create_dir_all(repo.join(".git").join("info")).expect("fake repo");
        fs::create_dir_all(repo.join("sub")).expect("subdir");
        fs::write(repo.join(".gitignore"), "work/\n").expect("root gitignore");
        fs::write(repo.join("sub").join(".gitignore"), "cache/\n").expect("sub gitignore");
        fs::write(repo.join(".git").join("info").join("exclude"), "scratch/\n").expect("exclude");

        let p_work = repo.join("work").join("recomp");
        let p_sub_cache = repo.join("sub").join("cache");
        let p_src = repo.join("src");
        let p_git = repo.join(".git").join("objects");
        let p_root = repo.clone();
        let p_empty = PathBuf::new();

        let order_a: Vec<&Path> = vec![
            &p_work,
            &p_sub_cache,
            &p_src,
            &p_git,
            &p_root,
            &p_empty,
        ];
        let order_b: Vec<&Path> = vec![
            &p_empty,
            &p_root,
            &p_git,
            &p_src,
            &p_sub_cache,
            &p_work,
        ];
        let order_c: Vec<&Path> = vec![
            &p_src,
            &p_work,
            &p_empty,
            &p_sub_cache,
            &p_root,
            &p_git,
        ];

        let r_a1 = HygieneReport::from_paths(order_a.iter().copied());
        let r_a2 = HygieneReport::from_paths(order_a.iter().copied());
        let r_b = HygieneReport::from_paths(order_b.iter().copied());
        let r_c = HygieneReport::from_paths(order_c.iter().copied());

        assert_eq!(r_a1, r_a2, "repeated runs must be identical");
        assert_eq!(r_a1, r_b, "report must be invariant under reverse order");
        assert_eq!(r_a1, r_c, "report must be invariant under arbitrary permutation");
        assert_eq!(r_a1.render_summary(), r_b.render_summary());
        assert_eq!(r_a1.render_summary(), r_c.render_summary());

        assert!(!r_a1.is_all_allowed());
        assert_eq!(r_a1.total_checked, 6);
        assert_eq!(
            r_a1.entries.keys().cloned().collect::<Vec<_>>(),
            vec![
                "<path>",
                "<repo>",
                "<repo>/.git/objects",
                "<repo>/src",
                "<repo>/sub/cache",
                "<repo>/work/recomp",
            ]
        );
        assert_eq!(
            r_a1.rule_sources.iter().cloned().collect::<Vec<_>>(),
            vec![".git/info/exclude", ".gitignore", "sub/.gitignore"]
        );

        // Redaction collision safety: if an external allowed path and an empty refused path both
        // redact to `<path>`, the refusal always wins regardless of input order.
        let p_outside = tmp.path().join("outside-build");
        let col_1 = HygieneReport::from_paths([p_outside.as_path(), p_empty.as_path()]);
        let col_2 = HygieneReport::from_paths([p_empty.as_path(), p_outside.as_path()]);
        assert_eq!(col_1, col_2);
        assert!(!col_1.is_all_allowed());
        assert!(col_1.refused_labels.contains(REDACTED_PATH_TOKEN));
        assert!(!col_1.allowed_labels.contains(REDACTED_PATH_TOKEN));

        // Summary output contains zero host path tokens.
        let summary = r_a1.render_summary();
        let host_basename = tmp
            .path()
            .file_name()
            .expect("temp dir name")
            .to_string_lossy()
            .into_owned();
        assert!(!summary.contains(&host_basename), "summary leaked temp dir: {summary}");
        assert!(!summary.contains(":\\"), "summary leaked drive prefix: {summary}");
    }
}
