//! Shim families, the runtime's export-registration tables and the shim audit.
//!
//! * [`SHIM_FAMILIES`] / [`shim_families_for_dll`]: which `pg6_<family>_*` shims may implement an
//!   import of a DLL, in priority order, and how names are matched inside each family.
//! * [`scan_tables`]: reads the export tables the runtime registers in the module registry
//!   (`register_export`/`register_exports` callers) from runtime sources: every `(DLL, export
//!   name, ordinal, symbol)` tuple, whatever the table's spelling (`E(TAG, "Name", symbol)`
//!   X-macros whose tag names a `#define`d DLL constant, `{kDll, "Name", proc_address(&symbol)}`
//!   arrays, `export_entry("Name", &symbol, ordinal)` lists passed to `register_exports("DLL",
//!   ...)`). These tables are what `GetProcAddress` resolves through, so they are the
//!   authoritative export-name -> symbol and ordinal -> export maps for the static bridge too.
//! * The audit tables of `runtime/os/shim_audit.h` (`PG6_SHIM_AUDIT`, `PG6_IMPORT_GAPS`).
//! * [`parse_import_list`]: the import list of `docs/imports_required.md`.

use std::collections::{BTreeMap, BTreeSet};

use super::{Tok, dll_key, tokenize_keep_directives};
use crate::pe::ImportedSymbol;

/// Maximum wrapper-call nesting depth followed by [`symbol_ref`] (e.g. `f(g(&pg6_x))`).
const MAX_SYMBOL_REF_DEPTH: usize = 32;

// ---------------------------------------------------------------------------------------------
// Checked offset / index bounds
// ---------------------------------------------------------------------------------------------

/// Checked `base + offset` bounded strictly below `len`. Returns `None` on overflow or OOB.
pub fn checked_index(base: usize, offset: usize, len: usize) -> Option<usize> {
    base.checked_add(offset).filter(|&idx| idx < len)
}

/// Checked `[start, start + len)` span bounded by `total`. Returns `None` on overflow or OOB.
pub fn checked_span(start: usize, len: usize, total: usize) -> Option<(usize, usize)> {
    let end = start.checked_add(len)?;
    (end <= total).then_some((start, end))
}

/// Checked byte offset `index * stride` for a fixed-stride table of `limit_bytes` total bytes.
/// Returns `None` if `stride == 0`, on multiplication/addition overflow, or when the entry at
/// `[offset, offset + stride)` would exceed `limit_bytes`.
pub fn checked_table_offset(index: usize, stride: usize, limit_bytes: usize) -> Option<usize> {
    if stride == 0 {
        return None;
    }
    let offset = index.checked_mul(stride)?;
    checked_span(offset, stride, limit_bytes).map(|(start, _)| start)
}

/// Checked subslice `slice[start..end]` returning `None` if `start > end` or `end > slice.len()`.
fn checked_slice<T>(slice: &[T], start: usize, end: usize) -> Option<&[T]> {
    (start <= end).then(|| slice.get(start..end)).flatten()
}

// ---------------------------------------------------------------------------------------------
// Families
// ---------------------------------------------------------------------------------------------

/// How a family maps an export name to one of its shims.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NameRule {
    /// The export tables first; failing that, the name key ([`super::name_key`]: ASCII-lowercase,
    /// underscores dropped) of the shim name after the prefix.
    TablesThenKey,
    /// The export tables only (exact export names). For families whose export names are not
    /// derivable from the symbol (`_exit` vs `exit`, C++-decorated names, ordinals).
    TablesOnly,
}

/// One shim family of the runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShimFamily {
    /// Symbol prefix (`pg6_crt_`).
    pub prefix: &'static str,
    pub rule: NameRule,
    /// Every shim of the family needs a `PG6_SHIM_AUDIT` row to bind: `ok` binds, `deny` and
    /// `pending` leave the import a logging thunk with the recorded reason.
    pub audited: bool,
    /// Where the family lives (documentation).
    pub header: &'static str,
}

pub const PREFIX_WIN32: &str = "pg6_win32_";
pub const PREFIX_GDK: &str = "pg6_gdk_";
pub const PREFIX_D3D12: &str = "pg6_d3d12_";
pub const PREFIX_CRT: &str = "pg6_crt_";
pub const PREFIX_MW: &str = "pg6_mw_";
pub const PREFIX_XBOX: &str = "pg6_xbox_";
pub const PREFIX_WS2: &str = "pg6_ws2_";

/// Every shim family the bridge knows.
pub const SHIM_FAMILIES: &[ShimFamily] = &[
    ShimFamily {
        prefix: PREFIX_WIN32,
        rule: NameRule::TablesThenKey,
        audited: false,
        header: "os/win32_*.h, os/xbox_offline.h (table: os/shim_exports.cpp)",
    },
    ShimFamily {
        prefix: PREFIX_GDK,
        rule: NameRule::TablesThenKey,
        audited: false,
        header: "os/xgameruntime.h (table: os/shim_exports.cpp)",
    },
    ShimFamily {
        prefix: PREFIX_D3D12,
        rule: NameRule::TablesThenKey,
        audited: false,
        header: "gpu/d3d12_xs.h",
    },
    ShimFamily {
        prefix: PREFIX_CRT,
        rule: NameRule::TablesOnly,
        audited: true,
        header: "os/ucrt_bridge.h (table: os/ucrt_bridge.cpp PG6_UCRT_EXPORTS)",
    },
    ShimFamily {
        prefix: PREFIX_MW,
        rule: NameRule::TablesOnly,
        audited: true,
        header: "os/xbox_middleware.h (table: os/xbox_middleware.cpp)",
    },
    ShimFamily {
        prefix: PREFIX_XBOX,
        rule: NameRule::TablesOnly,
        audited: true,
        header: "os/xbox_scarlett.h (table: os/xbox_scarlett.cpp)",
    },
    ShimFamily {
        prefix: PREFIX_WS2,
        rule: NameRule::TablesOnly,
        audited: true,
        header: "os/winsock_ordinals.h (table: os/winsock_ordinals.cpp, names and ordinals)",
    },
];

/// The family with symbol prefix `prefix`.
pub fn family(prefix: &str) -> Option<&'static ShimFamily> {
    SHIM_FAMILIES.iter().find(|f| f.prefix == prefix)
}

/// The family a shim symbol belongs to.
pub fn family_of_symbol(symbol: &str) -> Option<&'static ShimFamily> {
    SHIM_FAMILIES.iter().find(|f| symbol.starts_with(f.prefix))
}

#[derive(Debug, Clone, Copy)]
enum DllMatch {
    Exact(&'static str),
    Prefix(&'static str),
}

/// DLL (key: lowercase, `.dll` dropped) -> families in priority order. First matching row wins.
const DLL_ROUTES: &[(DllMatch, &[&str])] = &[
    // C runtime: the UCRT API sets, ucrtbase and the VC++ runtime.
    (DllMatch::Prefix("api-ms-win-crt-"), &[PREFIX_CRT]),
    (DllMatch::Exact("ucrtbase"), &[PREFIX_CRT]),
    (DllMatch::Exact("vcruntime140"), &[PREFIX_CRT]),
    (DllMatch::Exact("vcruntime140_1"), &[PREFIX_CRT]),
    // Winsock: the ordinal family first (one coherent socket / last-error state for every
    // WS2_32 import), the O5 offline sockets for names it does not export.
    (DllMatch::Exact("ws2_32"), &[PREFIX_WS2, PREFIX_WIN32]),
    (DllMatch::Exact("kernel32"), &[PREFIX_WIN32]),
    (DllMatch::Exact("kernelbase"), &[PREFIX_WIN32]),
    (DllMatch::Exact("advapi32"), &[PREFIX_WIN32]),
    (DllMatch::Exact("winhttp"), &[PREFIX_WIN32]),
    (DllMatch::Exact("bcrypt"), &[PREFIX_WIN32]),
    (DllMatch::Exact("user32"), &[PREFIX_WIN32]),
    (DllMatch::Exact("combase"), &[PREFIX_WIN32]),
    (DllMatch::Exact("ole32"), &[PREFIX_WIN32]),
    (DllMatch::Prefix("api-ms-win-"), &[PREFIX_WIN32]),
    (DllMatch::Prefix("ext-ms-win-"), &[PREFIX_WIN32]),
    (DllMatch::Exact("xgameruntime"), &[PREFIX_GDK]),
    (DllMatch::Prefix("microsoft.xbox.services"), &[PREFIX_GDK]),
    (DllMatch::Exact("d3d12"), &[PREFIX_D3D12]),
    (DllMatch::Exact("d3d12_x"), &[PREFIX_D3D12]),
    (DllMatch::Exact("d3d12_xs"), &[PREFIX_D3D12]),
    // Xbox Series system DLLs (O12).
    (DllMatch::Exact("xmem"), &[PREFIX_XBOX]),
    (DllMatch::Exact("xcrdapi"), &[PREFIX_XBOX]),
    (DllMatch::Exact("xg_xs"), &[PREFIX_XBOX]),
    (DllMatch::Exact("gameinput"), &[PREFIX_XBOX]),
    // Middleware (O14).
    (DllMatch::Exact("libhttpclient.gdk"), &[PREFIX_MW]),
    (DllMatch::Exact("bink2scarlett"), &[PREFIX_MW]),
    (DllMatch::Exact("acphal"), &[PREFIX_MW]),
    (DllMatch::Exact("xapu"), &[PREFIX_MW]),
    (DllMatch::Exact("chad"), &[PREFIX_MW]),
    (DllMatch::Exact("amd_fidelityfx_loader_xbox"), &[PREFIX_MW]),
    (DllMatch::Exact("xboxbvhbuilder"), &[PREFIX_MW]),
    (DllMatch::Exact("mfplat"), &[PREFIX_MW]),
    (DllMatch::Exact("mfreadwrite"), &[PREFIX_MW]),
];

/// Shim family prefixes that may implement imports of `dll`, in priority order (empty when the
/// runtime has no family for it). DLL names compare case-insensitively, `.dll` optional.
pub fn shim_families_for_dll(dll: &str) -> &'static [&'static str] {
    let key = dll_key(dll);
    DLL_ROUTES
        .iter()
        .find(|(m, _)| match m {
            DllMatch::Exact(name) => key == *name,
            DllMatch::Prefix(prefix) => key.starts_with(prefix),
        })
        .map_or(&[], |(_, families)| families)
}

// ---------------------------------------------------------------------------------------------
// Tables
// ---------------------------------------------------------------------------------------------

/// One `(DLL, export name, ordinal, symbol)` registration of a runtime export table.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ExportEntry {
    /// DLL as spelled in the table.
    pub dll: String,
    pub name: String,
    pub ordinal: Option<u16>,
    pub symbol: String,
    /// Include spelling of the scanned source.
    pub source: String,
}

impl ExportEntry {
    /// Canonical input-order-independent sort key:
    /// `(dll_key, name, ordinal, symbol, raw_dll, source)`.
    pub fn canonical_key(&self) -> (String, String, Option<u16>, String, String, String) {
        (
            dll_key(&self.dll),
            self.name.clone(),
            self.ordinal,
            self.symbol.clone(),
            self.dll.clone(),
            self.source.clone(),
        )
    }
}

/// Audit verdict of one shim (`PG6_SHIM_AUDIT`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Verdict {
    /// Audited: safe to call from recompiled code through a marshaling thunk. Binds.
    Ok,
    /// Audited: must not be bound (the note says why); the import stays a logging thunk.
    Deny,
    /// Not audited yet; does not bind.
    Pending,
}

impl Verdict {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Deny => "deny",
            Self::Pending => "pending",
        }
    }

    fn parse(word: &str) -> Option<Self> {
        match word {
            "ok" => Some(Self::Ok),
            "deny" => Some(Self::Deny),
            "pending" => Some(Self::Pending),
            _ => None,
        }
    }
}

/// One `PG6_SHIM_AUDIT` row.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct AuditEntry {
    pub symbol: String,
    pub verdict: Verdict,
    pub note: String,
    pub source: String,
}

impl AuditEntry {
    /// Canonical sort key: `(symbol, verdict, note, source)`.
    pub fn canonical_key(&self) -> (String, Verdict, String, String) {
        (
            self.symbol.clone(),
            self.verdict,
            self.note.clone(),
            self.source.clone(),
        )
    }
}

/// One `PG6_IMPORT_GAPS` row: an import the runtime knowingly has no shim for, with the reason
/// (and the task that will add it). `name` is an export name, `#<ordinal>` or `*`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct GapEntry {
    pub dll: String,
    pub name: String,
    pub reason: String,
    pub source: String,
}

impl GapEntry {
    pub fn matches(&self, dll: &str, symbol: &ImportedSymbol) -> bool {
        if dll_key(&self.dll) != dll_key(dll) {
            return false;
        }
        match symbol {
            _ if self.name == "*" => true,
            ImportedSymbol::ByName { name, .. } => *name == self.name,
            ImportedSymbol::ByOrdinal { ordinal } => self.name == format!("#{ordinal}"),
        }
    }

    /// Canonical sort key: `(dll_key, name, raw_dll, reason, source)`.
    pub fn canonical_key(&self) -> (String, String, String, String, String) {
        (
            dll_key(&self.dll),
            self.name.clone(),
            self.dll.clone(),
            self.reason.clone(),
            self.source.clone(),
        )
    }
}

/// Everything [`scan_tables`] found in one source.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SourceTables {
    pub exports: Vec<ExportEntry>,
    pub audit: Vec<AuditEntry>,
    pub gaps: Vec<GapEntry>,
    pub diagnostics: Vec<String>,
}

impl SourceTables {
    /// Sort and deduplicate exports, audit rows, gaps, and diagnostics into canonical order
    /// (invariant across input permutations).
    pub fn canonicalize(&mut self) {
        let mut exports = BTreeMap::new();
        for entry in self.exports.drain(..) {
            exports.entry(entry.canonical_key()).or_insert(entry);
        }
        self.exports = exports.into_values().collect();

        let mut audit = BTreeMap::new();
        for row in self.audit.drain(..) {
            audit.entry(row.canonical_key()).or_insert(row);
        }
        self.audit = audit.into_values().collect();

        let mut gaps = BTreeMap::new();
        for gap in self.gaps.drain(..) {
            gaps.entry(gap.canonical_key()).or_insert(gap);
        }
        self.gaps = gaps.into_values().collect();

        let diagnostics: BTreeSet<String> = self.diagnostics.drain(..).collect();
        self.diagnostics = diagnostics.into_iter().collect();
    }

    /// Return a canonically sorted and deduplicated copy of these tables.
    pub fn canonicalized(&self) -> Self {
        let mut copy = self.clone();
        copy.canonicalize();
        copy
    }

    /// Validate export and audit tables for duplicate or conflicting ordinals and symbols.
    pub fn validate(&self) -> TableValidation {
        let mut conflicts: BTreeSet<TableConflict> = validate_exports(&self.exports)
            .conflicts
            .into_iter()
            .collect();
        conflicts.extend(validate_audit(&self.audit).conflicts);
        TableValidation {
            conflicts: conflicts.into_iter().collect(),
        }
    }

    /// Build a deterministic `BTreeMap`/`BTreeSet`-backed lookup and validation index.
    pub fn index(&self) -> StubTableIndex {
        StubTableIndex::from_tables(self)
    }
}

// ---------------------------------------------------------------------------------------------
// Duplicate ordinal / symbol validation & canonical lookup index
// ---------------------------------------------------------------------------------------------

/// Kind of duplicate or conflict detected in stub export or audit tables.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum TableConflict {
    /// Two or more distinct export names claim the same `(dll_key, ordinal)`.
    ConflictingOrdinalName {
        dll_key: String,
        ordinal: u16,
        names: Vec<String>,
    },
    /// One `(dll_key, ordinal)` is registered more than once.
    DuplicateOrdinal {
        dll_key: String,
        ordinal: u16,
        symbols: Vec<String>,
        count: usize,
    },
    /// One `(dll_key, name)` is bound to multiple distinct symbols.
    DuplicateExportSymbol {
        dll_key: String,
        name: String,
        symbols: Vec<String>,
    },
    /// One `(dll_key, name)` is registered with multiple distinct non-zero ordinals.
    ConflictingExportOrdinal {
        dll_key: String,
        name: String,
        ordinals: Vec<u16>,
    },
    /// An exact `(dll_key, name, ordinal, symbol)` tuple appears more than once.
    DuplicateExportEntry {
        dll_key: String,
        name: String,
        ordinal: Option<u16>,
        symbol: String,
        count: usize,
    },
    /// A shim symbol appears more than once in `PG6_SHIM_AUDIT`.
    DuplicateAuditSymbol {
        symbol: String,
        verdicts: Vec<Verdict>,
        count: usize,
    },
}

impl TableConflict {
    /// Deterministic human-readable diagnostic message for this conflict.
    pub fn describe(&self) -> String {
        match self {
            Self::ConflictingOrdinalName {
                dll_key,
                ordinal,
                names,
            } => format!(
                "{dll_key}!#{ordinal} maps to conflicting export names: {}",
                names.join(", ")
            ),
            Self::DuplicateOrdinal {
                dll_key,
                ordinal,
                symbols,
                count,
            } => format!(
                "{dll_key}!#{ordinal} registered {count} times (symbols: {})",
                symbols.join(", ")
            ),
            Self::DuplicateExportSymbol {
                dll_key,
                name,
                symbols,
            } => format!(
                "{dll_key}!{name} bound to multiple symbols: {}",
                symbols.join(", ")
            ),
            Self::ConflictingExportOrdinal {
                dll_key,
                name,
                ordinals,
            } => {
                let list: Vec<String> = ordinals.iter().map(|o| format!("#{o}")).collect();
                format!(
                    "{dll_key}!{name} has conflicting ordinals: {}",
                    list.join(", ")
                )
            }
            Self::DuplicateExportEntry {
                dll_key,
                name,
                ordinal,
                symbol,
                count,
            } => match ordinal {
                Some(ord) => format!(
                    "{dll_key}!{name} (#{ord}) -> {symbol} registered {count} times"
                ),
                None => format!("{dll_key}!{name} -> {symbol} registered {count} times"),
            },
            Self::DuplicateAuditSymbol {
                symbol,
                verdicts,
                count,
            } => {
                let list: Vec<&str> = verdicts.iter().map(|v| v.as_str()).collect();
                format!(
                    "audit symbol {symbol} repeated {count} times (verdicts: {})",
                    list.join(", ")
                )
            }
        }
    }
}

/// Deterministic validation report for stub export and audit tables.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TableValidation {
    /// Canonically ordered list of detected conflicts.
    pub conflicts: Vec<TableConflict>,
}

impl TableValidation {
    pub fn is_clean(&self) -> bool {
        self.conflicts.is_empty()
    }

    pub fn has_ordinal_conflicts(&self) -> bool {
        self.conflicts.iter().any(|c| {
            matches!(
                c,
                TableConflict::ConflictingOrdinalName { .. }
                    | TableConflict::DuplicateOrdinal { .. }
                    | TableConflict::ConflictingExportOrdinal { .. }
            )
        })
    }

    pub fn has_symbol_conflicts(&self) -> bool {
        self.conflicts.iter().any(|c| {
            matches!(
                c,
                TableConflict::DuplicateExportSymbol { .. }
                    | TableConflict::DuplicateExportEntry { .. }
                    | TableConflict::DuplicateAuditSymbol { .. }
            )
        })
    }

    /// Deterministic diagnostic strings for every conflict.
    pub fn diagnostics(&self) -> Vec<String> {
        self.conflicts.iter().map(TableConflict::describe).collect()
    }
}

/// Validate export-table registrations for duplicate ordinals, conflicting ordinal names,
/// conflicting ordinals for a name, duplicate symbol bindings, and exact duplicate rows.
/// Results are canonically ordered and invariant across input permutations.
pub fn validate_exports(exports: &[ExportEntry]) -> TableValidation {
    let mut exact_counts: BTreeMap<(String, String, Option<u16>, String), usize> = BTreeMap::new();
    let mut by_name_symbols: BTreeMap<(String, String), BTreeSet<String>> = BTreeMap::new();
    let mut by_name_ordinals: BTreeMap<(String, String), BTreeSet<u16>> = BTreeMap::new();
    let mut by_ordinal: BTreeMap<(String, u16), (BTreeSet<String>, BTreeSet<String>, usize)> =
        BTreeMap::new();

    for entry in exports {
        let key = dll_key(&entry.dll);
        let exact_key = (
            key.clone(),
            entry.name.clone(),
            entry.ordinal,
            entry.symbol.clone(),
        );
        let exact = exact_counts.entry(exact_key).or_insert(0);
        *exact = exact.saturating_add(1);

        by_name_symbols
            .entry((key.clone(), entry.name.clone()))
            .or_default()
            .insert(entry.symbol.clone());

        if let Some(ordinal) = entry.ordinal {
            by_name_ordinals
                .entry((key.clone(), entry.name.clone()))
                .or_default()
                .insert(ordinal);

            let (names, symbols, count) = by_ordinal.entry((key, ordinal)).or_default();
            names.insert(entry.name.clone());
            symbols.insert(entry.symbol.clone());
            *count = count.saturating_add(1);
        }
    }

    let mut conflicts: BTreeSet<TableConflict> = BTreeSet::new();

    for ((key, ordinal), (names, symbols, count)) in by_ordinal {
        if names.len() > 1 {
            conflicts.insert(TableConflict::ConflictingOrdinalName {
                dll_key: key.clone(),
                ordinal,
                names: names.into_iter().collect(),
            });
        }
        if count > 1 {
            conflicts.insert(TableConflict::DuplicateOrdinal {
                dll_key: key,
                ordinal,
                symbols: symbols.into_iter().collect(),
                count,
            });
        }
    }

    for ((key, name), symbols) in by_name_symbols {
        if symbols.len() > 1 {
            conflicts.insert(TableConflict::DuplicateExportSymbol {
                dll_key: key,
                name,
                symbols: symbols.into_iter().collect(),
            });
        }
    }

    for ((key, name), ordinals) in by_name_ordinals {
        if ordinals.len() > 1 {
            conflicts.insert(TableConflict::ConflictingExportOrdinal {
                dll_key: key,
                name,
                ordinals: ordinals.into_iter().collect(),
            });
        }
    }

    for ((key, name, ordinal, symbol), count) in exact_counts {
        if count > 1 {
            conflicts.insert(TableConflict::DuplicateExportEntry {
                dll_key: key,
                name,
                ordinal,
                symbol,
                count,
            });
        }
    }

    TableValidation {
        conflicts: conflicts.into_iter().collect(),
    }
}

/// Validate `PG6_SHIM_AUDIT` rows for duplicate symbols (whether matching or conflicting
/// verdicts). Results are canonically ordered and invariant across input permutations.
pub fn validate_audit(audit: &[AuditEntry]) -> TableValidation {
    let mut by_symbol: BTreeMap<String, (BTreeSet<Verdict>, usize)> = BTreeMap::new();
    for row in audit {
        let (verdicts, count) = by_symbol.entry(row.symbol.clone()).or_default();
        verdicts.insert(row.verdict);
        *count = count.saturating_add(1);
    }

    let mut conflicts: BTreeSet<TableConflict> = BTreeSet::new();
    for (symbol, (verdicts, count)) in by_symbol {
        if count > 1 {
            conflicts.insert(TableConflict::DuplicateAuditSymbol {
                symbol,
                verdicts: verdicts.into_iter().collect(),
                count,
            });
        }
    }

    TableValidation {
        conflicts: conflicts.into_iter().collect(),
    }
}

/// Deterministic, `BTreeMap`/`BTreeSet`-backed lookup table over scanned [`ExportEntry`] rows.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StubTableIndex {
    entries: Vec<ExportEntry>,
    by_dll_name: BTreeMap<(String, String), BTreeSet<String>>,
    by_dll_ordinal: BTreeMap<(String, u16), BTreeSet<String>>,
    by_symbol: BTreeMap<String, BTreeSet<(String, String)>>,
    validation: TableValidation,
}

impl StubTableIndex {
    /// Build a canonical lookup index from raw export entries. Validation runs on the raw slice
    /// before canonical deduplication so duplicate registrations are preserved in [`Self::validation`].
    pub fn from_exports(exports: &[ExportEntry]) -> Self {
        let validation = validate_exports(exports);
        let mut canonical = BTreeMap::new();
        for entry in exports {
            canonical
                .entry(entry.canonical_key())
                .or_insert_with(|| entry.clone());
        }
        let entries: Vec<ExportEntry> = canonical.into_values().collect();
        let mut by_dll_name: BTreeMap<(String, String), BTreeSet<String>> = BTreeMap::new();
        let mut by_dll_ordinal: BTreeMap<(String, u16), BTreeSet<String>> = BTreeMap::new();
        let mut by_symbol: BTreeMap<String, BTreeSet<(String, String)>> = BTreeMap::new();

        for entry in &entries {
            let key = dll_key(&entry.dll);
            by_dll_name
                .entry((key.clone(), entry.name.clone()))
                .or_default()
                .insert(entry.symbol.clone());
            if let Some(ordinal) = entry.ordinal {
                by_dll_ordinal
                    .entry((key.clone(), ordinal))
                    .or_default()
                    .insert(entry.name.clone());
            }
            by_symbol
                .entry(entry.symbol.clone())
                .or_default()
                .insert((key, entry.name.clone()));
        }

        Self {
            entries,
            by_dll_name,
            by_dll_ordinal,
            by_symbol,
            validation,
        }
    }

    /// Build a canonical lookup index from scanned [`SourceTables`], validating both exports and
    /// audit rows.
    pub fn from_tables(tables: &SourceTables) -> Self {
        let mut index = Self::from_exports(&tables.exports);
        let mut conflicts: BTreeSet<TableConflict> =
            index.validation.conflicts.into_iter().collect();
        conflicts.extend(validate_audit(&tables.audit).conflicts);
        index.validation.conflicts = conflicts.into_iter().collect();
        index
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Canonically sorted, deduplicated export entries.
    pub fn entries(&self) -> &[ExportEntry] {
        &self.entries
    }

    /// Checked lookup of canonical entry at `index`.
    pub fn entry_at(&self, index: usize) -> Option<&ExportEntry> {
        self.entries.get(index)
    }

    /// Checked byte offset of entry `index` in a fixed-stride table of `table_bytes` bytes.
    /// Returns `None` if `index` is out of bounds of this table, `entry_stride == 0`, or the
    /// byte range overflows / exceeds `table_bytes`.
    pub fn entry_byte_offset(
        &self,
        index: usize,
        entry_stride: usize,
        table_bytes: usize,
    ) -> Option<usize> {
        if index >= self.entries.len() {
            return None;
        }
        checked_table_offset(index, entry_stride, table_bytes)
    }

    /// Sorted distinct shim symbols registered for `(dll, name)`.
    pub fn symbols_for_name(&self, dll: &str, name: &str) -> Vec<&str> {
        self.by_dll_name
            .get(&(dll_key(dll), name.to_string()))
            .map(|syms| syms.iter().map(String::as_str).collect())
            .unwrap_or_default()
    }

    /// Sorted distinct export names registered for `(dll, ordinal)`.
    pub fn ordinal_names(&self, dll: &str, ordinal: u16) -> Vec<&str> {
        self.by_dll_ordinal
            .get(&(dll_key(dll), ordinal))
            .map(|names| names.iter().map(String::as_str).collect())
            .unwrap_or_default()
    }

    /// Unique export name registered for `(dll, ordinal)`, or `None` when absent or conflicting.
    pub fn unique_ordinal_name(&self, dll: &str, ordinal: u16) -> Option<&str> {
        let names = self.by_dll_ordinal.get(&(dll_key(dll), ordinal))?;
        (names.len() == 1)
            .then(|| names.iter().next().map(String::as_str))
            .flatten()
    }

    /// Sorted distinct `(dll_key, export_name)` pairs bound to `symbol`.
    pub fn exports_for_symbol(&self, symbol: &str) -> Vec<(&str, &str)> {
        self.by_symbol
            .get(symbol)
            .map(|pairs| {
                pairs
                    .iter()
                    .map(|(d, n)| (d.as_str(), n.as_str()))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Validation findings collected when building this index.
    pub fn validation(&self) -> &TableValidation {
        &self.validation
    }
}

const AUDIT_MACRO: &str = "PG6_SHIM_AUDIT";
const GAPS_MACRO: &str = "PG6_IMPORT_GAPS";

fn is_dll_name(s: &str) -> bool {
    s.len() > 4
        && s.len()
            .checked_sub(4)
            .and_then(|start| s.as_bytes().get(start..))
            .is_some_and(|tail| tail.eq_ignore_ascii_case(b".dll"))
}

/// `NAME` -> string for `#define NAME "..."`, `NAME[] = "..."` and `NAME = "..."`.
fn string_constants(toks: &[Tok]) -> BTreeMap<String, String> {
    let mut consts = BTreeMap::new();
    let len = toks.len();
    for (i, tok) in toks.iter().enumerate() {
        let Tok::Ident(name) = tok else { continue };
        let at = |offset: usize| checked_index(i, offset, len).and_then(|idx| toks.get(idx));
        let value = if i
            .checked_sub(1)
            .and_then(|prev| toks.get(prev))
            .is_some_and(|t| t.is_ident("define"))
        {
            match at(1) {
                Some(Tok::Str(s)) => Some(s),
                _ => None,
            }
        } else if at(1).is_some_and(|t| t.is_punct('['))
            && at(2).is_some_and(|t| t.is_punct(']'))
            && at(3).is_some_and(|t| t.is_punct('='))
        {
            match at(4) {
                Some(Tok::Str(s)) => Some(s),
                _ => None,
            }
        } else if at(1).is_some_and(|t| t.is_punct('=')) {
            match (at(2), at(3)) {
                (Some(Tok::Str(s)), Some(end)) if end.is_punct(';') => Some(s),
                _ => None,
            }
        } else {
            None
        };
        if let Some(value) = value {
            consts.insert(name.clone(), value.clone());
        }
    }
    consts
}

/// A string constant by name, or by a unique `<prefix>_<name>` (token pasting: `E(K32, ...)` with
/// `#define PG6_MOD_K32 "KERNEL32.dll"` and `PG6_MOD_##mod`).
fn resolve_constant<'a>(consts: &'a BTreeMap<String, String>, name: &str) -> Option<&'a str> {
    if name.is_empty() {
        return None;
    }
    if let Some(value) = consts.get(name) {
        return Some(value);
    }
    let suffix = format!("_{name}");
    let mut found = consts
        .iter()
        .filter(|(k, _)| k.ends_with(&suffix))
        .map(|(_, v)| v.as_str());
    let first = found.next()?;
    found.next().is_none().then_some(first)
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Arg {
    Dll(String),
    Name(String),
    Ordinal(Option<u16>),
    Symbol(String),
    Other,
}

fn parse_ordinal(text: &str) -> Option<Option<u16>> {
    let digits = text.trim_end_matches(['u', 'U', 'l', 'L']);
    if digits.is_empty() {
        return None;
    }
    let value = if let Some(hex) = digits
        .strip_prefix("0x")
        .or_else(|| digits.strip_prefix("0X"))
    {
        if hex.is_empty() {
            return None;
        }
        u32::from_str_radix(hex, 16).ok()?
    } else {
        digits.parse::<u32>().ok()?
    };
    let ordinal = u16::try_from(value).ok()?;
    Some((ordinal != 0).then_some(ordinal))
}

/// `pg6_x`, `&pg6_x`, `f(&pg6_x)` / `f(pg6_x)` -> `pg6_x`.
fn symbol_ref(arg: &[Tok]) -> Option<String> {
    symbol_ref_bounded(arg, MAX_SYMBOL_REF_DEPTH)
}

fn symbol_ref_bounded(arg: &[Tok], depth_remaining: usize) -> Option<String> {
    match arg {
        [Tok::Ident(s)] if s.starts_with("pg6_") => Some(s.clone()),
        [amp, Tok::Ident(s)] if amp.is_punct('&') && s.starts_with("pg6_") => Some(s.clone()),
        [Tok::Ident(_), open, inner @ .., close] if open.is_punct('(') && close.is_punct(')') => {
            let next_depth = depth_remaining.checked_sub(1)?;
            symbol_ref_bounded(inner, next_depth)
        }
        _ => None,
    }
}

fn classify_arg(arg: &[Tok], consts: &BTreeMap<String, String>) -> Arg {
    if let Some(symbol) = symbol_ref(arg) {
        return Arg::Symbol(symbol);
    }
    match arg {
        [Tok::Str(s)] if is_dll_name(s) => Arg::Dll(s.clone()),
        [Tok::Str(s)] if !s.is_empty() => Arg::Name(s.clone()),
        [Tok::Num(n)] => parse_ordinal(n).map_or(Arg::Other, Arg::Ordinal),
        [Tok::Ident(id)] => match resolve_constant(consts, id) {
            Some(value) if is_dll_name(value) => Arg::Dll(value.to_string()),
            _ => Arg::Other,
        },
        _ => Arg::Other,
    }
}

/// Top-level comma-separated arguments of the group opened at `open` and closed at `close`.
/// Returns an empty vector if `open`/`close` are inverted, overflow, or out of bounds.
fn split_args(toks: &[Tok], open: usize, close: usize) -> Vec<&[Tok]> {
    let Some(inner_start) = open.checked_add(1) else {
        return Vec::new();
    };
    if close > toks.len() || inner_start >= close {
        return Vec::new();
    }
    let mut args = Vec::new();
    let mut depth: usize = 0;
    let mut start = inner_start;
    for j in inner_start..close {
        let Some(tok) = toks.get(j) else {
            break;
        };
        if tok.is_punct('(') || tok.is_punct('[') || tok.is_punct('{') {
            depth = depth.saturating_add(1);
        } else if tok.is_punct(')') || tok.is_punct(']') || tok.is_punct('}') {
            depth = depth.saturating_sub(1);
        } else if depth == 0 && tok.is_punct(',') {
            if let Some(slice) = checked_slice(toks, start, j) {
                args.push(slice);
            }
            let Some(next_start) = j.checked_add(1) else {
                return args;
            };
            start = next_start;
        }
    }
    if start < close
        && let Some(slice) = checked_slice(toks, start, close)
    {
        args.push(slice);
    }
    args
}

/// Matching closer and enclosing opener of every `(`/`{`/`[` opener.
struct Groups {
    close: Vec<Option<usize>>,
    parent: Vec<Option<usize>>,
}

fn groups(toks: &[Tok]) -> Groups {
    let mut close = vec![None; toks.len()];
    let mut parent = vec![None; toks.len()];
    let mut stack: Vec<usize> = Vec::new();
    // Stack depth at the start of the current preprocessor directive.
    let mut directive: Option<usize> = None;
    for (i, tok) in toks.iter().enumerate() {
        if tok.is_punct('(') || tok.is_punct('{') || tok.is_punct('[') {
            if let Some(slot) = parent.get_mut(i) {
                *slot = stack.last().copied();
            }
            stack.push(i);
        } else if tok.is_punct(')') || tok.is_punct('}') || tok.is_punct(']') {
            if directive.is_none_or(|depth| stack.len() > depth)
                && let Some(open) = stack.pop()
                && let Some(slot) = close.get_mut(open)
            {
                *slot = Some(i);
            }
        } else if tok.is_punct('#') {
            directive.get_or_insert(stack.len());
        } else if tok.is_punct('\n') {
            // End of a directive: groups opened inside it never span it.
            let depth = directive.take().unwrap_or(0);
            while stack.len() > depth {
                if let Some(open) = stack.pop()
                    && let Some(slot) = close.get_mut(open)
                {
                    *slot = Some(i);
                }
            }
        }
    }
    Groups { close, parent }
}

/// Token ranges `[start, end)` of `#define <MACRO>(...)` directives.
fn macro_body(toks: &[Tok], name: &str) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let len = toks.len();
    for i in 1..len {
        let is_define = i
            .checked_sub(1)
            .and_then(|prev| toks.get(prev))
            .is_some_and(|t| t.is_ident("define"));
        let is_name = toks.get(i).is_some_and(|t| t.is_ident(name));
        if is_define && is_name {
            let Some(start) = i.checked_add(1) else {
                continue;
            };
            let end = toks
                .get(i..)
                .and_then(|tail| tail.iter().position(|t| t.is_punct('\n')))
                .and_then(|p| i.checked_add(p))
                .map_or(len, |e| e.min(len));
            if start <= end {
                out.push((start, end));
            }
        }
    }
    out
}

/// Scan one runtime source (header or `.cpp`) for export tables and audit tables.
/// `source` is its include spelling (for diagnostics and provenance).
pub fn scan_tables(text: &str, source: &str) -> SourceTables {
    let toks = tokenize_keep_directives(text);
    let consts = string_constants(&toks);
    let g = groups(&toks);
    let mut out = SourceTables::default();

    // Audit and gap tables (macro bodies), excluded from the export-table scan.
    let mut excluded = vec![false; toks.len()];
    for (start, end) in macro_body(&toks, AUDIT_MACRO) {
        if let Some(mask) = excluded.get_mut(start..end) {
            mask.fill(true);
        }
        for open in start..end {
            let Some(close) = g
                .close
                .get(open)
                .copied()
                .flatten()
                .filter(|_| toks.get(open).is_some_and(|t| t.is_punct('(')))
            else {
                continue;
            };
            match split_args(&toks, open, close).as_slice() {
                [
                    [Tok::Ident(symbol)],
                    [Tok::Ident(verdict)],
                    [Tok::Str(note)],
                ] if symbol.starts_with("pg6_") => match Verdict::parse(verdict) {
                    Some(verdict) => out.audit.push(AuditEntry {
                        symbol: symbol.clone(),
                        verdict,
                        note: note.clone(),
                        source: source.to_string(),
                    }),
                    None => out.diagnostics.push(format!(
                        "{source}: audit row for {symbol} has unknown verdict `{verdict}`"
                    )),
                },
                _ => {}
            }
        }
    }
    for (start, end) in macro_body(&toks, GAPS_MACRO) {
        if let Some(mask) = excluded.get_mut(start..end) {
            mask.fill(true);
        }
        for open in start..end {
            let Some(close) = g
                .close
                .get(open)
                .copied()
                .flatten()
                .filter(|_| toks.get(open).is_some_and(|t| t.is_punct('(')))
            else {
                continue;
            };
            if let [[Tok::Str(dll)], [Tok::Str(name)], [Tok::Str(reason)]] =
                split_args(&toks, open, close).as_slice()
            {
                out.gaps.push(GapEntry {
                    dll: dll.clone(),
                    name: name.clone(),
                    reason: reason.clone(),
                    source: source.to_string(),
                });
            }
        }
    }

    // Export tuples.
    let mut dll_arg_memo: BTreeMap<usize, Option<String>> = BTreeMap::new();
    let mut dll_arg = |open: usize| -> Option<String> {
        if let Some(cached) = dll_arg_memo.get(&open) {
            return cached.clone();
        }
        let found = g.close.get(open).copied().flatten().and_then(|close| {
            let dlls: Vec<String> = split_args(&toks, open, close)
                .into_iter()
                .filter_map(|a| match classify_arg(a, &consts) {
                    Arg::Dll(d) => Some(d),
                    _ => None,
                })
                .collect();
            (dlls.len() == 1).then(|| dlls[0].clone())
        });
        dll_arg_memo.insert(open, found.clone());
        found
    };
    for open in 0..toks.len() {
        if excluded.get(open).copied().unwrap_or(false)
            || !toks
                .get(open)
                .is_some_and(|t| t.is_punct('(') || t.is_punct('{'))
        {
            continue;
        }
        let Some(close) = g.close.get(open).copied().flatten() else {
            continue;
        };
        // Cheap filter: a direct `pg6_` identifier argument somewhere inside.
        let Some(inner) = open
            .checked_add(1)
            .and_then(|inner_start| checked_slice(&toks, inner_start, close))
        else {
            continue;
        };
        if !inner
            .iter()
            .any(|t| t.ident().is_some_and(|s| s.starts_with("pg6_")))
        {
            continue;
        }
        let args: Vec<Arg> = split_args(&toks, open, close)
            .into_iter()
            .map(|a| classify_arg(a, &consts))
            .collect();
        if args.contains(&Arg::Other) {
            continue;
        }
        let count = |f: fn(&Arg) -> bool| args.iter().filter(|a| f(a)).count();
        if count(|a| matches!(a, Arg::Symbol(_))) != 1
            || count(|a| matches!(a, Arg::Name(_))) != 1
            || count(|a| matches!(a, Arg::Dll(_))) > 1
            || count(|a| matches!(a, Arg::Ordinal(_))) > 1
        {
            continue;
        }
        let mut dll = None;
        let mut name = String::new();
        let mut symbol = String::new();
        let mut ordinal = None;
        for arg in args {
            match arg {
                Arg::Dll(d) => dll = Some(d),
                Arg::Name(n) => name = n,
                Arg::Symbol(s) => symbol = s,
                Arg::Ordinal(o) => ordinal = o,
                Arg::Other => {}
            }
        }
        // No DLL in the tuple: an enclosing call's DLL argument (`register_exports("xmem.dll",
        // {...})`), else the next DLL literal in the enclosing top-level block (an entry array
        // registered after its definition: `register_exports("WS2_32.dll", entries, n)`).
        if dll.is_none() {
            let mut outer = g.parent.get(open).copied().flatten();
            let mut top = open;
            let mut steps = 0usize;
            while let Some(p) = outer {
                if steps >= toks.len() {
                    break;
                }
                steps = steps.saturating_add(1);
                if dll.is_none() {
                    dll = dll_arg(p);
                }
                top = p;
                outer = g.parent.get(p).copied().flatten();
            }
            if dll.is_none() {
                let end = g.close.get(top).copied().flatten().unwrap_or(toks.len());
                if let Some(tail) = close
                    .checked_add(1)
                    .and_then(|tail_start| checked_slice(&toks, tail_start, end))
                {
                    dll = tail.iter().find_map(|t| match t {
                        Tok::Str(s) if is_dll_name(s) => Some(s.clone()),
                        Tok::Ident(id) => resolve_constant(&consts, id)
                            .filter(|v| is_dll_name(v))
                            .map(str::to_string),
                        _ => None,
                    });
                }
            }
        }
        match dll {
            Some(dll) => out.exports.push(ExportEntry {
                dll,
                name,
                ordinal,
                symbol,
                source: source.to_string(),
            }),
            None => out.diagnostics.push(format!(
                "{source}: export entry \"{name}\" -> {symbol} has no DLL"
            )),
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------
// Import lists
// ---------------------------------------------------------------------------------------------

/// Imports listed in a text file: the per-DLL tables of `docs/imports_required.md`
/// (`### \`DLL\` (...)` headings, `| name | \`Symbol\` | ... |` and
/// `| ordinal | \`#N\` | ... |` rows) or plain `DLL!Name` / `DLL!#N` lines. In file order.
pub fn parse_import_list(text: &str) -> Vec<(String, ImportedSymbol)> {
    let mut out = Vec::new();
    let mut dll: Option<String> = None;
    let ticked = |s: &str| -> Option<String> {
        let open = s.find('`')?;
        let start = open.checked_add(1)?;
        let rest = s.get(start..)?;
        let rel_end = rest.find('`')?;
        let end = start.checked_add(rel_end)?;
        s.get(start..end).map(str::to_string)
    };
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("### ") {
            dll = ticked(rest).filter(|d| is_dll_name(d));
            continue;
        }
        if line.starts_with('#') {
            dll = None;
            continue;
        }
        if let Some(row) = line.strip_prefix('|') {
            let Some(current) = &dll else { continue };
            let cells: Vec<&str> = row.split('|').map(str::trim).collect();
            let (Some(kind), Some(cell)) = (cells.first(), cells.get(1)) else {
                continue;
            };
            let Some(symbol) = ticked(cell) else { continue };
            if symbol.is_empty() {
                continue;
            }
            match *kind {
                "name" => out.push((
                    current.clone(),
                    ImportedSymbol::ByName {
                        hint: 0,
                        name: symbol,
                    },
                )),
                "ordinal" => {
                    if let Some(ordinal) = symbol
                        .strip_prefix('#')
                        .and_then(|o| o.parse::<u16>().ok())
                        .filter(|&o| o != 0)
                    {
                        out.push((current.clone(), ImportedSymbol::ByOrdinal { ordinal }));
                    }
                }
                _ => {}
            }
            continue;
        }
        if let Some((d, s)) = line.split_once('!')
            && is_dll_name(d)
            && !s.is_empty()
            && !s.contains(char::is_whitespace)
        {
            let symbol = if let Some(ord_str) = s.strip_prefix('#') {
                let Some(ordinal) = ord_str.parse::<u16>().ok().filter(|&o| o != 0) else {
                    continue;
                };
                ImportedSymbol::ByOrdinal { ordinal }
            } else {
                ImportedSymbol::ByName {
                    hint: 0,
                    name: s.to_string(),
                }
            };
            out.push((d.to_string(), symbol));
        }
    }
    out
}

/// Return a canonically ordered and deduplicated import list (invariant across input permutations).
#[allow(dead_code)]
pub fn canonicalize_import_list(
    imports: &[(String, ImportedSymbol)],
) -> Vec<(String, ImportedSymbol)> {
    let mut canonical: BTreeMap<(String, u8, String, u16, u16, String), (String, ImportedSymbol)> =
        BTreeMap::new();
    for (dll, symbol) in imports {
        let key = match symbol {
            ImportedSymbol::ByName { hint, name } => (
                dll_key(dll),
                0u8,
                name.clone(),
                0u16,
                *hint,
                dll.clone(),
            ),
            ImportedSymbol::ByOrdinal { ordinal } => (
                dll_key(dll),
                1u8,
                String::new(),
                *ordinal,
                0u16,
                dll.clone(),
            ),
        };
        canonical
            .entry(key)
            .or_insert_with(|| (dll.clone(), symbol.clone()));
    }
    canonical.into_values().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dll_routes_pick_families_in_priority_order() {
        assert_eq!(
            shim_families_for_dll("api-ms-win-crt-stdio-l1-1-0.dll"),
            [PREFIX_CRT]
        );
        assert_eq!(shim_families_for_dll("VCRUNTIME140.dll"), [PREFIX_CRT]);
        assert_eq!(shim_families_for_dll("vcruntime140_1"), [PREFIX_CRT]);
        assert_eq!(shim_families_for_dll("ucrtbase.DLL"), [PREFIX_CRT]);
        assert_eq!(
            shim_families_for_dll("WS2_32.dll"),
            [PREFIX_WS2, PREFIX_WIN32]
        );
        assert_eq!(
            shim_families_for_dll("api-ms-win-core-synch-l1-2-0.dll"),
            [PREFIX_WIN32]
        );
        assert_eq!(shim_families_for_dll("libHttpClient.GDK.dll"), [PREFIX_MW]);
        assert_eq!(shim_families_for_dll("XboxBvhBuilder.dll"), [PREFIX_MW]);
        assert_eq!(shim_families_for_dll("MFPlat.DLL"), [PREFIX_MW]);
        assert_eq!(shim_families_for_dll("GameInput.dll"), [PREFIX_XBOX]);
        assert_eq!(shim_families_for_dll("xmem"), [PREFIX_XBOX]);
        assert!(shim_families_for_dll("msvcp140.dll").is_empty());
        assert!(shim_families_for_dll("oleaut32.dll").is_empty());
        assert!(shim_families_for_dll("kernel32.dll.bak").is_empty());
        for f in SHIM_FAMILIES {
            assert_eq!(family(f.prefix), Some(f));
        }
        assert_eq!(family_of_symbol("pg6_mw_bink_close"), family(PREFIX_MW));
        assert_eq!(family_of_symbol("pg6_gthunk_qsort"), None);
    }

    const XMACRO: &str = r#"
#define PG6_UCRT_MOD_RUNTIME "api-ms-win-crt-runtime-l1-1-0.dll"
#define PG6_UCRT_MOD_VCR140 "VCRUNTIME140.dll"
#define PG6_UCRT_EXPORTS(E)                                  \
    E(RUNTIME, "_exit", pg6_crt__exit)                       \
    E(RUNTIME, "exit", pg6_crt_exit)                         \
    /* comment */                                            \
    E(VCR140, "memcpy", pg6_crt_memcpy)
#define PG6_UCRT_ENTRY(mod, name, fn) \
    Pg6UcrtExport{PG6_UCRT_MOD_##mod, name, #fn, proc_address(&fn)},
static const Pg6UcrtExport kUcrtTable[] = {PG6_UCRT_EXPORTS(PG6_UCRT_ENTRY)};
"#;

    const ARRAYS: &str = r#"
constexpr const char kDllBvh[] = "XboxBvhBuilder.dll";
const MwExportSpec* table() {
    static const MwExportSpec kExports[] = {
        {kDllBvh, "??1XboxBvhBuilder@@QEAA@XZ", raw_proc_address(&pg6_mw_bvh_builder_dtor)},
        {"MFPlat.DLL", "MFShutdown", raw_proc_address(&pg6_mw_mf_shutdown)},
    };
    return kExports;
}
size_t reg() {
    size_t total = register_exports(
        "GameInput.dll",
        {
            export_entry("GameInputCreate", &pg6_xbox_GameInputCreate, 1u),
        });
    log("not an entry", 3);
    return total;
}
uint32_t pg6_winsock_register(void) {
    std::call_once(once, [] {
        const Pg6ExportEntry entries[] = {
            export_entry("closesocket", &pg6_ws2_closesocket, 3u),
            export_entry("getaddrinfo", &pg6_ws2_getaddrinfo),
        };
        registered = register_exports("WS2_32.dll", entries, 2);
    });
    return registered;
}
static const GuestExport kGuest[] = {
    {"GameInput.dll", "GameInputCreate", 1, &pg6_gthunk_game_input_create},
};
void orphan() { use(export_entry("Lonely", &pg6_xbox_Lonely)); }
"#;

    fn entries(t: &SourceTables) -> Vec<(String, String, Option<u16>, String)> {
        t.exports
            .iter()
            .map(|e| (e.dll.clone(), e.name.clone(), e.ordinal, e.symbol.clone()))
            .collect()
    }

    fn e(
        dll: &str,
        name: &str,
        ordinal: Option<u16>,
        symbol: &str,
    ) -> (String, String, Option<u16>, String) {
        (dll.into(), name.into(), ordinal, symbol.into())
    }

    #[test]
    fn scans_x_macro_tables_with_token_pasted_dll_tags() {
        let t = scan_tables(XMACRO, "os/x.cpp");
        assert_eq!(
            entries(&t),
            [
                e(
                    "api-ms-win-crt-runtime-l1-1-0.dll",
                    "_exit",
                    None,
                    "pg6_crt__exit"
                ),
                e(
                    "api-ms-win-crt-runtime-l1-1-0.dll",
                    "exit",
                    None,
                    "pg6_crt_exit"
                ),
                e("VCRUNTIME140.dll", "memcpy", None, "pg6_crt_memcpy"),
            ]
        );
        assert!(t.diagnostics.is_empty(), "{:?}", t.diagnostics);
        assert_eq!(t.exports[0].source, "os/x.cpp");
    }

    #[test]
    fn scans_struct_arrays_register_calls_and_ordinals() {
        let t = scan_tables(ARRAYS, "os/y.cpp");
        assert_eq!(
            entries(&t),
            [
                e(
                    "XboxBvhBuilder.dll",
                    "??1XboxBvhBuilder@@QEAA@XZ",
                    None,
                    "pg6_mw_bvh_builder_dtor"
                ),
                e("MFPlat.DLL", "MFShutdown", None, "pg6_mw_mf_shutdown"),
                e(
                    "GameInput.dll",
                    "GameInputCreate",
                    Some(1),
                    "pg6_xbox_GameInputCreate"
                ),
                e("WS2_32.dll", "closesocket", Some(3), "pg6_ws2_closesocket"),
                e("WS2_32.dll", "getaddrinfo", None, "pg6_ws2_getaddrinfo"),
                e(
                    "GameInput.dll",
                    "GameInputCreate",
                    Some(1),
                    "pg6_gthunk_game_input_create"
                ),
            ]
        );
        assert_eq!(t.diagnostics.len(), 1, "{:?}", t.diagnostics);
        assert!(t.diagnostics[0].contains("Lonely"));
    }

    #[test]
    fn scans_audit_and_gap_tables() {
        let text = r##"
#pragma once
// X(symbol, verdict, "note")
#define PG6_SHIM_AUDIT(X)                                          \
    X(pg6_crt_strlen, ok, "pure; char* in, size_t out")            \
    X(pg6_crt_longjmp, deny, "jmp_buf belongs to guest VCRUNTIME") \
    X(pg6_crt_fopen, pending, "")                                  \
    X(pg6_crt_bogus, maybe, "x")
#define PG6_IMPORT_GAPS(X) \
    X("d3d12_xs.dll", "*", "G24") \
    X("WS2_32.dll", "#99", "none")
"##;
        let t = scan_tables(text, "os/shim_audit.h");
        let rows: Vec<(&str, Verdict)> = t
            .audit
            .iter()
            .map(|a| (a.symbol.as_str(), a.verdict))
            .collect();
        assert_eq!(
            rows,
            [
                ("pg6_crt_strlen", Verdict::Ok),
                ("pg6_crt_longjmp", Verdict::Deny),
                ("pg6_crt_fopen", Verdict::Pending)
            ]
        );
        assert_eq!(t.audit[1].note, "jmp_buf belongs to guest VCRUNTIME");
        assert_eq!(t.diagnostics.len(), 1);
        assert!(t.exports.is_empty(), "audit rows are not export entries");
        assert_eq!(t.gaps.len(), 2);
        let any = ImportedSymbol::ByName {
            hint: 0,
            name: "D3D12XboxCreateDevice".into(),
        };
        assert!(t.gaps[0].matches("D3D12_XS.DLL", &any));
        assert!(!t.gaps[0].matches("d3d12.dll", &any));
        assert!(t.gaps[1].matches("ws2_32", &ImportedSymbol::ByOrdinal { ordinal: 99 }));
        assert!(!t.gaps[1].matches("ws2_32", &ImportedSymbol::ByOrdinal { ordinal: 9 }));
    }

    #[test]
    fn parses_markdown_and_plain_import_lists() {
        let md = "# Report\n\n## Imports\n\n### `WS2_32.dll` (2 symbols)\n\n| Kind | Symbol | Hint / Ordinal |\n| --- | --- | --- |\n| ordinal | `#111` | ordinal `111` |\n| name | `getaddrinfo` | hint `7` |\n\n### `XboxBvhBuilder.dll` (1 symbol)\n\n| name | `??1XboxBvhBuilder@@QEAA@XZ` | hint `1` |\n\n## Exports\n\n| name | `NotAnImport` | x |\nKERNEL32.dll!Sleep\nGameInput.dll!#1\n";
        let list = parse_import_list(md);
        let labels: Vec<String> = list
            .iter()
            .map(|(d, s)| match s {
                ImportedSymbol::ByName { name, .. } => format!("{d}!{name}"),
                ImportedSymbol::ByOrdinal { ordinal } => format!("{d}!#{ordinal}"),
            })
            .collect();
        assert_eq!(
            labels,
            [
                "WS2_32.dll!#111",
                "WS2_32.dll!getaddrinfo",
                "XboxBvhBuilder.dll!??1XboxBvhBuilder@@QEAA@XZ",
                "KERNEL32.dll!Sleep",
                "GameInput.dll!#1"
            ]
        );
    }

    fn mk_export(
        dll: &str,
        name: &str,
        ordinal: Option<u16>,
        symbol: &str,
        source: &str,
    ) -> ExportEntry {
        ExportEntry {
            dll: dll.to_string(),
            name: name.to_string(),
            ordinal,
            symbol: symbol.to_string(),
            source: source.to_string(),
        }
    }

    fn mk_audit(symbol: &str, verdict: Verdict, note: &str, source: &str) -> AuditEntry {
        AuditEntry {
            symbol: symbol.to_string(),
            verdict,
            note: note.to_string(),
            source: source.to_string(),
        }
    }

    #[test]
    fn canonicalize_and_index_are_invariant_under_input_permutations() {
        let base_exports = vec![
            mk_export("WS2_32.dll", "send", Some(19), "pg6_ws2_send", "b.cpp"),
            mk_export(
                "api-ms-win-crt-heap-l1-1-0.dll",
                "malloc",
                None,
                "pg6_crt_malloc",
                "a.cpp",
            ),
            mk_export(
                "ws2_32.DLL",
                "closesocket",
                Some(3),
                "pg6_ws2_closesocket",
                "b.cpp",
            ),
            mk_export(
                "GameInput.dll",
                "GameInputCreate",
                Some(1),
                "pg6_xbox_GameInputCreate",
                "c.cpp",
            ),
            // Exact duplicate to confirm deterministic deduplication.
            mk_export("WS2_32.dll", "send", Some(19), "pg6_ws2_send", "b.cpp"),
        ];
        let base_audit = vec![
            mk_audit("pg6_ws2_send", Verdict::Ok, "safe", "audit.h"),
            mk_audit("pg6_crt_malloc", Verdict::Pending, "wip", "audit.h"),
            mk_audit("pg6_ws2_send", Verdict::Ok, "safe", "audit.h"),
        ];
        let base_gaps = vec![
            GapEntry {
                dll: "WS2_32.dll".into(),
                name: "#99".into(),
                reason: "none".into(),
                source: "audit.h".into(),
            },
            GapEntry {
                dll: "d3d12_xs.dll".into(),
                name: "*".into(),
                reason: "G24".into(),
                source: "audit.h".into(),
            },
        ];
        let base_diags = vec!["z diag".to_string(), "a diag".to_string(), "z diag".to_string()];

        let t_forward = SourceTables {
            exports: base_exports.clone(),
            audit: base_audit.clone(),
            gaps: base_gaps.clone(),
            diagnostics: base_diags.clone(),
        };

        let mut rev_exports = base_exports.clone();
        rev_exports.reverse();
        let mut rev_audit = base_audit.clone();
        rev_audit.reverse();
        let mut rev_gaps = base_gaps.clone();
        rev_gaps.reverse();
        let mut rev_diags = base_diags.clone();
        rev_diags.reverse();

        let t_reverse = SourceTables {
            exports: rev_exports,
            audit: rev_audit,
            gaps: rev_gaps,
            diagnostics: rev_diags,
        };

        let mut rot_exports = base_exports;
        rot_exports.rotate_left(2);
        let mut rot_audit = base_audit;
        rot_audit.rotate_left(1);
        let mut rot_gaps = base_gaps;
        rot_gaps.rotate_left(1);
        let mut rot_diags = base_diags;
        rot_diags.rotate_left(1);

        let t_rotated = SourceTables {
            exports: rot_exports,
            audit: rot_audit,
            gaps: rot_gaps,
            diagnostics: rot_diags,
        };

        let c_fwd = t_forward.canonicalized();
        let c_rev = t_reverse.canonicalized();
        let c_rot = t_rotated.canonicalized();
        assert_eq!(c_fwd, c_rev);
        assert_eq!(c_fwd, c_rot);
        assert_eq!(c_fwd.exports.len(), 4);
        assert_eq!(c_fwd.audit.len(), 2);
        assert_eq!(c_fwd.diagnostics, vec!["a diag", "z diag"]);

        // Canonical export order follows lowercase dll_key then name.
        let ordered_names: Vec<&str> = c_fwd.exports.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(
            ordered_names,
            ["malloc", "GameInputCreate", "closesocket", "send"]
        );

        // StubTableIndex and TableValidation are also permutation-invariant.
        let idx_fwd = t_forward.index();
        let idx_rev = t_reverse.index();
        let idx_rot = t_rotated.index();
        assert_eq!(idx_fwd, idx_rev);
        assert_eq!(idx_fwd, idx_rot);
        assert_eq!(t_forward.validate(), t_reverse.validate());
        assert_eq!(t_forward.validate(), t_rotated.validate());

        // Import list canonicalization is permutation-invariant and deduplicates.
        let imports_a = vec![
            (
                "WS2_32.dll".to_string(),
                ImportedSymbol::ByOrdinal { ordinal: 111 },
            ),
            (
                "KERNEL32.dll".to_string(),
                ImportedSymbol::ByName {
                    hint: 0,
                    name: "Sleep".into(),
                },
            ),
            (
                "ws2_32.dll".to_string(),
                ImportedSymbol::ByName {
                    hint: 0,
                    name: "getaddrinfo".into(),
                },
            ),
            (
                "KERNEL32.dll".to_string(),
                ImportedSymbol::ByName {
                    hint: 0,
                    name: "Sleep".into(),
                },
            ),
        ];
        let mut imports_b = imports_a.clone();
        imports_b.reverse();
        let can_a = canonicalize_import_list(&imports_a);
        let can_b = canonicalize_import_list(&imports_b);
        assert_eq!(can_a, can_b);
        assert_eq!(can_a.len(), 3);
    }

    #[test]
    fn validate_detects_duplicate_and_conflicting_ordinals() {
        let exports = vec![
            mk_export(
                "WS2_32.dll",
                "closesocket",
                Some(3),
                "pg6_ws2_closesocket",
                "a.cpp",
            ),
            // Different export name claiming the same (ws2_32, #3) ordinal (case-insensitive DLL).
            mk_export(
                "ws2_32.DLL",
                "connect",
                Some(3),
                "pg6_ws2_connect",
                "b.cpp",
            ),
            // Same (ws2_32, send) registered with two distinct ordinals.
            mk_export("WS2_32.dll", "send", Some(19), "pg6_ws2_send", "a.cpp"),
            mk_export("WS2_32.dll", "send", Some(20), "pg6_ws2_send", "b.cpp"),
            // Unique valid ordinal.
            mk_export(
                "GameInput.dll",
                "GameInputCreate",
                Some(1),
                "pg6_xbox_GameInputCreate",
                "c.cpp",
            ),
        ];

        let report = validate_exports(&exports);
        assert!(!report.is_clean());
        assert!(report.has_ordinal_conflicts());
        assert!(report.conflicts.contains(&TableConflict::ConflictingOrdinalName {
            dll_key: "ws2_32".into(),
            ordinal: 3,
            names: vec!["closesocket".into(), "connect".into()],
        }));
        assert!(report.conflicts.contains(&TableConflict::DuplicateOrdinal {
            dll_key: "ws2_32".into(),
            ordinal: 3,
            symbols: vec!["pg6_ws2_closesocket".into(), "pg6_ws2_connect".into()],
            count: 2,
        }));
        assert!(report.conflicts.contains(&TableConflict::ConflictingExportOrdinal {
            dll_key: "ws2_32".into(),
            name: "send".into(),
            ordinals: vec![19, 20],
        }));
        assert!(!report.diagnostics().is_empty());

        // StubTableIndex rejects conflicting ordinals in unique_ordinal_name while preserving
        // sorted ordinal_names for diagnostics.
        let index = StubTableIndex::from_exports(&exports);
        assert_eq!(index.unique_ordinal_name("WS2_32.dll", 3), None);
        assert_eq!(
            index.ordinal_names("ws2_32", 3),
            vec!["closesocket", "connect"]
        );
        assert_eq!(
            index.unique_ordinal_name("gameinput", 1),
            Some("GameInputCreate")
        );
        assert_eq!(index.unique_ordinal_name("gameinput", 99), None);
    }

    #[test]
    fn validate_detects_duplicate_export_and_audit_symbols() {
        let clean = SourceTables {
            exports: vec![
                mk_export("KERNEL32.dll", "Sleep", None, "pg6_win32_sleep", "k.cpp"),
                mk_export("WS2_32.dll", "send", Some(19), "pg6_ws2_send", "w.cpp"),
            ],
            audit: vec![mk_audit("pg6_ws2_send", Verdict::Ok, "audited", "audit.h")],
            gaps: Vec::new(),
            diagnostics: Vec::new(),
        };
        assert!(clean.validate().is_clean());
        assert!(!clean.validate().has_ordinal_conflicts());
        assert!(!clean.validate().has_symbol_conflicts());

        let dirty = SourceTables {
            exports: vec![
                mk_export(
                    "GameInput.dll",
                    "GameInputCreate",
                    Some(1),
                    "pg6_xbox_GameInputCreate",
                    "a.cpp",
                ),
                mk_export(
                    "gameinput.DLL",
                    "GameInputCreate",
                    Some(1),
                    "pg6_gthunk_game_input_create",
                    "b.cpp",
                ),
                mk_export("KERNEL32.dll", "Sleep", None, "pg6_win32_sleep", "k1.cpp"),
                mk_export("kernel32.dll", "Sleep", None, "pg6_win32_sleep", "k2.cpp"),
            ],
            audit: vec![
                mk_audit("pg6_crt_fopen", Verdict::Pending, "first", "a.h"),
                mk_audit("pg6_crt_fopen", Verdict::Deny, "second", "b.h"),
            ],
            gaps: Vec::new(),
            diagnostics: Vec::new(),
        };

        let report = dirty.validate();
        assert!(!report.is_clean());
        assert!(report.has_symbol_conflicts());
        assert!(report.conflicts.contains(&TableConflict::DuplicateExportSymbol {
            dll_key: "gameinput".into(),
            name: "GameInputCreate".into(),
            symbols: vec![
                "pg6_gthunk_game_input_create".into(),
                "pg6_xbox_GameInputCreate".into(),
            ],
        }));
        assert!(report.conflicts.contains(&TableConflict::DuplicateExportEntry {
            dll_key: "kernel32".into(),
            name: "Sleep".into(),
            ordinal: None,
            symbol: "pg6_win32_sleep".into(),
            count: 2,
        }));
        assert!(report.conflicts.contains(&TableConflict::DuplicateAuditSymbol {
            symbol: "pg6_crt_fopen".into(),
            verdicts: vec![Verdict::Deny, Verdict::Pending],
            count: 2,
        }));
    }

    #[test]
    fn empty_and_boundary_tables_never_panic() {
        // Empty source and empty index.
        let empty = scan_tables("", "empty.cpp");
        assert!(empty.exports.is_empty());
        assert!(empty.audit.is_empty());
        assert!(empty.gaps.is_empty());
        assert!(empty.diagnostics.is_empty());
        assert!(empty.validate().is_clean());
        let empty_idx = empty.index();
        assert!(empty_idx.is_empty());
        assert_eq!(empty_idx.len(), 0);
        assert_eq!(empty_idx.entry_at(0), None);
        assert_eq!(empty_idx.entry_byte_offset(0, 16, 64), None);

        // Boundary ordinals: 0 -> None, 1 and 65535 (u16::MAX) -> Some, 65536 / invalid -> None.
        assert_eq!(parse_ordinal("0"), Some(None));
        assert_eq!(parse_ordinal("0u"), Some(None));
        assert_eq!(parse_ordinal("1"), Some(Some(1)));
        assert_eq!(parse_ordinal("65535u"), Some(Some(u16::MAX)));
        assert_eq!(parse_ordinal("0xFFFFul"), Some(Some(u16::MAX)));
        assert_eq!(parse_ordinal("65536"), None);
        assert_eq!(parse_ordinal("0x10000"), None);
        assert_eq!(parse_ordinal("0x"), None);
        assert_eq!(parse_ordinal("u"), None);
        assert_eq!(parse_ordinal(""), None);

        // Deeply nested wrapper calls exceeding MAX_SYMBOL_REF_DEPTH must not stack overflow.
        let mut nested = String::from("&pg6_win32_sleep");
        for _ in 0..=MAX_SYMBOL_REF_DEPTH + 4 {
            nested = format!("wrap({nested})");
        }
        let nested_toks = tokenize_keep_directives(&nested);
        assert_eq!(symbol_ref(&nested_toks), None);

        // Unbalanced delimiters and `#define PG6_SHIM_AUDIT` right at EOF with no trailing newline.
        let malformed = "))) {{{ [[[\n#define PG6_SHIM_AUDIT";
        let t = scan_tables(malformed, "malformed.h");
        assert!(t.exports.is_empty());
        assert!(t.audit.is_empty());

        // Malformed backticks and #0 ordinals in parse_import_list must not panic.
        let weird_md = "### `WS2_32.dll`\n| name | `` | x |\n| name | ` | x |\n| ordinal | `#0` | x |\n| ordinal | `#65536` | x |\n| ordinal | `#65535` | x |\nWS2_32.dll!#0\nWS2_32.dll!#65535\n.dll!Bad\n";
        let parsed = parse_import_list(weird_md);
        assert_eq!(
            parsed,
            vec![
                (
                    "WS2_32.dll".into(),
                    ImportedSymbol::ByOrdinal { ordinal: u16::MAX }
                ),
                (
                    "WS2_32.dll".into(),
                    ImportedSymbol::ByOrdinal { ordinal: u16::MAX }
                ),
            ]
        );
    }

    #[test]
    fn top_level_orphan_export_emits_diagnostic_without_panicking() {
        // Previously panicked at `toks[close + 1..end]` when `top == open` (`end == close`).
        let src = r#"export_entry("TopLevelOrphan", &pg6_xbox_TopLevelOrphan);"#;
        let t = scan_tables(src, "os/orphan.cpp");
        assert!(t.exports.is_empty());
        assert_eq!(t.diagnostics.len(), 1);
        assert!(t.diagnostics[0].contains("TopLevelOrphan"));
        assert!(t.diagnostics[0].contains("has no DLL"));
    }

    #[test]
    fn checked_offset_and_index_bounds_reject_overflow_and_oob() {
        // checked_index
        assert_eq!(checked_index(0, 0, 1), Some(0));
        assert_eq!(checked_index(2, 3, 6), Some(5));
        assert_eq!(checked_index(2, 3, 5), None);
        assert_eq!(checked_index(usize::MAX, 1, usize::MAX), None);

        // checked_span
        assert_eq!(checked_span(4, 4, 8), Some((4, 8)));
        assert_eq!(checked_span(4, 5, 8), None);
        assert_eq!(checked_span(usize::MAX, 1, usize::MAX), None);

        // checked_table_offset
        assert_eq!(checked_table_offset(0, 0, 64), None);
        assert_eq!(checked_table_offset(0, 16, 15), None);
        assert_eq!(checked_table_offset(0, 16, 16), Some(0));
        assert_eq!(checked_table_offset(3, 16, 64), Some(48));
        assert_eq!(checked_table_offset(4, 16, 64), None);
        assert_eq!(checked_table_offset(usize::MAX, 16, usize::MAX), None);
        assert_eq!(checked_table_offset(1, usize::MAX, usize::MAX), None);

        // split_args with inverted, OOB, and overflowing indices returns empty without panicking.
        let toks = tokenize_keep_directives("(a, b)");
        assert!(split_args(&toks, usize::MAX, toks.len()).is_empty());
        assert!(split_args(&toks, 2, 2).is_empty());
        assert!(split_args(&toks, 4, 1).is_empty());
        assert!(split_args(&toks, 0, toks.len() + 10).is_empty());

        // StubTableIndex::entry_at and entry_byte_offset enforce both table length and byte bounds.
        let idx = StubTableIndex::from_exports(&[
            mk_export("KERNEL32.dll", "ExitProcess", None, "pg6_win32_exit_process", "k.cpp"),
            mk_export("KERNEL32.dll", "Sleep", None, "pg6_win32_sleep", "k.cpp"),
        ]);
        assert_eq!(idx.len(), 2);
        assert_eq!(idx.entry_at(0).map(|e| e.name.as_str()), Some("ExitProcess"));
        assert_eq!(idx.entry_at(1).map(|e| e.name.as_str()), Some("Sleep"));
        assert_eq!(idx.entry_at(2), None);
        assert_eq!(idx.entry_byte_offset(0, 32, 64), Some(0));
        assert_eq!(idx.entry_byte_offset(1, 32, 64), Some(32));
        assert_eq!(idx.entry_byte_offset(1, 32, 63), None);
        assert_eq!(idx.entry_byte_offset(2, 32, 96), None);
        assert_eq!(idx.entry_byte_offset(0, 0, 64), None);
        assert_eq!(idx.entry_byte_offset(1, usize::MAX, usize::MAX), None);
    }

    #[test]
    fn stub_table_index_lookup_handles_case_insensitive_dlls_and_symbols() {
        let exports = vec![
            mk_export("WS2_32.dll", "closesocket", Some(3), "pg6_ws2_closesocket", "w.cpp"),
            mk_export("KERNEL32.DLL", "Sleep", None, "pg6_win32_sleep", "k.cpp"),
            mk_export("KernelBase.dll", "Sleep", None, "pg6_win32_sleep", "k.cpp"),
        ];
        let idx = StubTableIndex::from_exports(&exports);
        assert_eq!(
            idx.symbols_for_name("ws2_32", "closesocket"),
            vec!["pg6_ws2_closesocket"]
        );
        assert_eq!(
            idx.symbols_for_name("WS2_32.DLL", "closesocket"),
            vec!["pg6_ws2_closesocket"]
        );
        assert!(idx.symbols_for_name("ws2_32", "nonexistent").is_empty());
        assert_eq!(
            idx.exports_for_symbol("pg6_win32_sleep"),
            vec![("kernel32", "Sleep"), ("kernelbase", "Sleep")]
        );
        assert!(idx.exports_for_symbol("pg6_missing").is_empty());
    }
}
