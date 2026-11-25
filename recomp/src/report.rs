//! Sanitized Markdown and JSON report renderers for parsed PE32+ metadata.
//!
//! Outputs symbol names, section metadata, and addresses ONLY — never raw binary bytes.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use playgta6_recomp::summary::{
    CoverageSummary, RankedCount, format_top_n, percentage_2dp, rank_top_n, sanitize_label,
};

use crate::pe::{ImportedSymbol, Pe64};

/// Renders a sanitized Markdown report (headers, sections, TLS callbacks, grouped imports,
/// grouped delay-load imports, and exports) containing symbol names & addresses only.
pub fn render_markdown(pe: &Pe64, source_label: &str) -> String {
    let mut out = String::with_capacity(4096);

    let import_sym_count: usize = pe.imports.iter().map(|m| m.symbols.len()).sum();
    let delay_sym_count: usize = pe.delay_imports.iter().map(|m| m.symbols.len()).sum();
    let export_sym_count: usize = pe.exports.as_ref().map_or(0, |e| e.symbols.len());
    let tls_cb_count: usize = pe.tls.as_ref().map_or(0, |t| t.callbacks.len());
    let entry_file_off = pe.rva_to_file_offset(pe.optional.entry_point_rva);

    let _ = writeln!(out, "# PE64 Inspection Report — `{}`", md_inline(source_label));
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "> Sanitized static analysis output: headers, section metadata, addresses, and symbol names only (zero raw binary bytes)."
    );
    let _ = writeln!(out);

    // --- Headers ---
    let _ = writeln!(out, "## Headers");
    let _ = writeln!(out);
    let _ = writeln!(out, "| Field | Value |");
    let _ = writeln!(out, "| --- | --- |");
    let _ = writeln!(out, "| File Size | {} bytes |", pe.file_size);
    let _ = writeln!(out, "| DOS `e_lfanew` | `0x{:08X}` |", pe.e_lfanew);
    let _ = writeln!(
        out,
        "| Machine | `{}` (`0x{:04X}`) |",
        pe.coff.machine_name(),
        pe.coff.machine
    );
    let _ = writeln!(
        out,
        "| Subsystem | `{}` (`0x{:04X}`) |",
        pe.optional.subsystem_name(),
        pe.optional.subsystem
    );
    let _ = writeln!(
        out,
        "| Optional Magic | `0x{:04X}` (PE32+) |",
        pe.optional.magic
    );
    let _ = writeln!(
        out,
        "| Linker Version | {}.{} |",
        pe.optional.major_linker_version, pe.optional.minor_linker_version
    );
    let _ = writeln!(
        out,
        "| Timestamp | `0x{:08X}` |",
        pe.coff.time_date_stamp
    );
    let _ = writeln!(
        out,
        "| Image Base | `0x{:016X}` |",
        pe.optional.image_base
    );
    match entry_file_off {
        Some(off) => {
            let _ = writeln!(
                out,
                "| Entry Point RVA | `0x{:08X}` (file offset `0x{:X}`) |",
                pe.optional.entry_point_rva, off
            );
        }
        None => {
            let _ = writeln!(
                out,
                "| Entry Point RVA | `0x{:08X}` (unmapped) |",
                pe.optional.entry_point_rva
            );
        }
    }
    let _ = writeln!(out, "| Base of Code | `0x{:08X}` |", pe.optional.base_of_code);
    let _ = writeln!(out, "| Size of Image | `0x{:08X}` ({} bytes) |", pe.optional.size_of_image, pe.optional.size_of_image);
    let _ = writeln!(out, "| Size of Headers | `0x{:08X}` |", pe.optional.size_of_headers);
    let _ = writeln!(
        out,
        "| Alignment (Section / File) | `0x{:X}` / `0x{:X}` |",
        pe.optional.section_alignment, pe.optional.file_alignment
    );
    let _ = writeln!(
        out,
        "| COFF Characteristics | `0x{:04X}` |",
        pe.coff.characteristics
    );
    let _ = writeln!(
        out,
        "| DLL Characteristics | `0x{:04X}` |",
        pe.optional.dll_characteristics
    );
    let _ = writeln!(out, "| Checksum | `0x{:08X}` |", pe.optional.checksum);
    let _ = writeln!(out, "| Sections | {} |", pe.sections.len());
    let _ = writeln!(
        out,
        "| Standard Imports | {} DLL(s), {} symbol(s) |",
        pe.imports.len(),
        import_sym_count
    );
    let _ = writeln!(
        out,
        "| Delay-Load Imports | {} DLL(s), {} symbol(s) |",
        pe.delay_imports.len(),
        delay_sym_count
    );
    let _ = writeln!(out, "| Exports | {} symbol(s) |", export_sym_count);
    let _ = writeln!(out, "| TLS Callbacks | {} |", tls_cb_count);
    let _ = writeln!(out);

    // --- Sections ---
    let _ = writeln!(out, "## Sections");
    let _ = writeln!(out);
    if pe.sections.is_empty() {
        let _ = writeln!(out, "_No sections present._");
    } else {
        let _ = writeln!(
            out,
            "| Name | Virtual Address | Virtual Size | Raw Offset | Raw Size | Perms | Characteristics |"
        );
        let _ = writeln!(out, "| --- | --- | --- | --- | --- | --- | --- |");
        for sec in &pe.sections {
            let _ = writeln!(
                out,
                "| `{}` | `0x{:08X}` | `0x{:08X}` | `0x{:08X}` | `0x{:08X}` | `{}` | `0x{:08X}` |",
                md_table_cell(&sec.name),
                sec.virtual_address,
                sec.virtual_size,
                sec.raw_offset,
                sec.raw_size,
                sec.permissions_str(),
                sec.characteristics
            );
        }
    }
    let _ = writeln!(out);

    // --- TLS Callbacks ---
    let _ = writeln!(out, "## TLS Callbacks");
    let _ = writeln!(out);
    match &pe.tls {
        None => {
            let _ = writeln!(out, "_No TLS directory present._");
        }
        Some(tls) => {
            let _ = writeln!(
                out,
                "- Raw Data Range: `0x{:016X}` .. `0x{:016X}` (zero-fill: {} bytes, characteristics: `0x{:08X}`)",
                tls.start_address_of_raw_data,
                tls.end_address_of_raw_data,
                tls.size_of_zero_fill,
                tls.characteristics
            );
            let _ = writeln!(
                out,
                "- Index VA: `0x{:016X}` | Callbacks Array VA: `0x{:016X}`",
                tls.address_of_index, tls.address_of_callbacks
            );
            let _ = writeln!(out);
            if tls.callbacks.is_empty() {
                let _ = writeln!(out, "_No TLS callbacks registered._");
            } else {
                let _ = writeln!(out, "| # | Callback RVA | Callback VA |");
                let _ = writeln!(out, "| --- | --- | --- |");
                for (idx, cb) in tls.callbacks.iter().enumerate() {
                    let _ = writeln!(
                        out,
                        "| {} | `0x{:08X}` | `0x{:016X}` |",
                        idx, cb.rva, cb.va
                    );
                }
            }
        }
    }
    let _ = writeln!(out);

    // --- Standard Imports (grouped by DLL) ---
    let _ = writeln!(out, "## Imports");
    let _ = writeln!(out);
    if pe.imports.is_empty() {
        let _ = writeln!(out, "_No standard imports._");
        let _ = writeln!(out);
    } else {
        let grouped = group_standard_imports(pe);
        for (dll_name, symbols) in grouped {
            let _ = writeln!(
                out,
                "### `{}` ({} symbol{})",
                md_inline(&dll_name),
                symbols.len(),
                if symbols.len() == 1 { "" } else { "s" }
            );
            let _ = writeln!(out);
            write_symbol_table(&mut out, &symbols);
            let _ = writeln!(out);
        }
    }

    // --- Delay-Load Imports (grouped by DLL) ---
    let _ = writeln!(out, "## Delay-Load Imports");
    let _ = writeln!(out);
    if pe.delay_imports.is_empty() {
        let _ = writeln!(out, "_No delay-load imports._");
        let _ = writeln!(out);
    } else {
        let grouped = group_delay_imports(pe);
        for group in grouped {
            let mode = if group.uses_rva { "RVA" } else { "VA" };
            let _ = writeln!(
                out,
                "### `{}` ({} symbol{}, mode: {}, attrs: `0x{:X}`)",
                md_inline(&group.dll_name),
                group.symbols.len(),
                if group.symbols.len() == 1 { "" } else { "s" },
                mode,
                group.attributes
            );
            let _ = writeln!(out);
            write_symbol_table(&mut out, &group.symbols);
            let _ = writeln!(out);
        }
    }

    // --- Exports ---
    let _ = writeln!(out, "## Exports");
    let _ = writeln!(out);
    match &pe.exports {
        None => {
            let _ = writeln!(out, "_No export directory present._");
        }
        Some(exports) => {
            if let Some(mod_name) = &exports.module_name {
                let _ = writeln!(
                    out,
                    "- Module Name: `{}` | Ordinal Base: {}",
                    md_inline(mod_name),
                    exports.ordinal_base
                );
            } else {
                let _ = writeln!(out, "- Ordinal Base: {}", exports.ordinal_base);
            }
            let _ = writeln!(out);
            if exports.symbols.is_empty() {
                let _ = writeln!(out, "_No exported symbols._");
            } else {
                let _ = writeln!(out, "| Ordinal | RVA | Name | Forwarder |");
                let _ = writeln!(out, "| --- | --- | --- | --- |");
                for sym in &exports.symbols {
                    let name_col = match &sym.name {
                        Some(n) => format!("`{}`", md_table_cell(n)),
                        None => "_(ordinal only)_".to_string(),
                    };
                    let fwd_col = match &sym.forwarder {
                        Some(fwd) => format!("`{}`", md_table_cell(fwd)),
                        None => "-".to_string(),
                    };
                    let _ = writeln!(
                        out,
                        "| {} | `0x{:08X}` | {} | {} |",
                        sym.ordinal, sym.rva, name_col, fwd_col
                    );
                }
            }
        }
    }

    out
}

struct GroupedDelayImport<'a> {
    dll_name: String,
    uses_rva: bool,
    attributes: u32,
    symbols: Vec<&'a ImportedSymbol>,
}

fn group_standard_imports(pe: &Pe64) -> Vec<(String, Vec<&ImportedSymbol>)> {
    let mut map: BTreeMap<(String, String), Vec<&ImportedSymbol>> = BTreeMap::new();
    for module in &pe.imports {
        let key = (module.dll_name.to_ascii_lowercase(), module.dll_name.clone());
        map.entry(key).or_default().extend(module.symbols.iter());
    }
    map.into_iter()
        .map(|((_, dll_name), syms)| (dll_name, syms))
        .collect()
}

fn group_delay_imports(pe: &Pe64) -> Vec<GroupedDelayImport<'_>> {
    let mut map: BTreeMap<(String, String), GroupedDelayImport<'_>> = BTreeMap::new();
    for module in &pe.delay_imports {
        let key = (module.dll_name.to_ascii_lowercase(), module.dll_name.clone());
        let entry = map.entry(key).or_insert_with(|| GroupedDelayImport {
            dll_name: module.dll_name.clone(),
            uses_rva: module.uses_rva,
            attributes: module.attributes,
            symbols: Vec::new(),
        });
        entry.symbols.extend(module.symbols.iter());
    }
    map.into_values().collect()
}

fn write_symbol_table(out: &mut String, symbols: &[&ImportedSymbol]) {
    if symbols.is_empty() {
        let _ = writeln!(out, "_No symbols listed._");
        return;
    }
    let _ = writeln!(out, "| Kind | Symbol | Hint / Ordinal |");
    let _ = writeln!(out, "| --- | --- | --- |");
    for sym in symbols {
        match sym {
            ImportedSymbol::ByName { hint, name } => {
                let _ = writeln!(
                    out,
                    "| name | `{}` | hint `0x{:04X}` |",
                    md_table_cell(name),
                    hint
                );
            }
            ImportedSymbol::ByOrdinal { ordinal } => {
                let _ = writeln!(out, "| ordinal | `#{ordinal}` | ordinal `{ordinal}` |");
            }
        }
    }
}

fn md_inline(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '`' => out.push('\''),
            '\r' | '\n' => out.push(' '),
            c if c.is_control() => {}
            c => out.push(c),
        }
    }
    out
}

fn md_table_cell(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '|' => out.push_str("\\|"),
            '`' => out.push('\''),
            '\r' | '\n' => out.push(' '),
            c if c.is_control() => {}
            c => out.push(c),
        }
    }
    out
}

/// Renders a sanitized JSON report for a single parsed PE32+ binary (zero external crates).
pub fn render_json(pe: &Pe64, source_label: &str) -> String {
    let mut out = String::with_capacity(4096);
    write_pe_json(&mut out, pe, source_label, 0);
    out.push('\n');
    out
}

/// Renders a JSON array of sanitized reports when inspecting multiple binaries.
pub fn render_json_many(items: &[(String, Pe64)]) -> String {
    let mut out = String::with_capacity(4096 * items.len().max(1));
    out.push_str("[\n");
    for (idx, (label, pe)) in items.iter().enumerate() {
        write_pe_json(&mut out, pe, label, 2);
        if idx + 1 < items.len() {
            out.push(',');
        }
        out.push('\n');
    }
    out.push_str("]\n");
    out
}

fn write_pe_json(out: &mut String, pe: &Pe64, source_label: &str, base_indent: usize) {
    let i0 = " ".repeat(base_indent);
    let i1 = " ".repeat(base_indent + 2);
    let i2 = " ".repeat(base_indent + 4);
    let i3 = " ".repeat(base_indent + 6);
    let i4 = " ".repeat(base_indent + 8);

    let entry_file_off = pe.rva_to_file_offset(pe.optional.entry_point_rva);

    let _ = writeln!(out, "{i0}{{");
    let _ = writeln!(out, "{i1}\"source\": {},", json_str(source_label));
    let _ = writeln!(out, "{i1}\"file_size\": {},", pe.file_size);
    let _ = writeln!(out, "{i1}\"e_lfanew\": {},", pe.e_lfanew);

    // COFF header
    let _ = writeln!(out, "{i1}\"coff\": {{");
    let _ = writeln!(out, "{i2}\"machine\": {},", pe.coff.machine);
    let _ = writeln!(
        out,
        "{i2}\"machine_name\": {},",
        json_str(pe.coff.machine_name())
    );
    let _ = writeln!(
        out,
        "{i2}\"number_of_sections\": {},",
        pe.coff.number_of_sections
    );
    let _ = writeln!(
        out,
        "{i2}\"time_date_stamp\": {},",
        pe.coff.time_date_stamp
    );
    let _ = writeln!(
        out,
        "{i2}\"pointer_to_symbol_table\": {},",
        pe.coff.pointer_to_symbol_table
    );
    let _ = writeln!(
        out,
        "{i2}\"number_of_symbols\": {},",
        pe.coff.number_of_symbols
    );
    let _ = writeln!(
        out,
        "{i2}\"size_of_optional_header\": {},",
        pe.coff.size_of_optional_header
    );
    let _ = writeln!(
        out,
        "{i2}\"characteristics\": {}",
        pe.coff.characteristics
    );
    let _ = writeln!(out, "{i1}}},");

    // Optional header
    let _ = writeln!(out, "{i1}\"optional_header\": {{");
    let _ = writeln!(out, "{i2}\"magic\": {},", pe.optional.magic);
    let _ = writeln!(
        out,
        "{i2}\"major_linker_version\": {},",
        pe.optional.major_linker_version
    );
    let _ = writeln!(
        out,
        "{i2}\"minor_linker_version\": {},",
        pe.optional.minor_linker_version
    );
    let _ = writeln!(out, "{i2}\"size_of_code\": {},", pe.optional.size_of_code);
    let _ = writeln!(
        out,
        "{i2}\"size_of_initialized_data\": {},",
        pe.optional.size_of_initialized_data
    );
    let _ = writeln!(
        out,
        "{i2}\"size_of_uninitialized_data\": {},",
        pe.optional.size_of_uninitialized_data
    );
    let _ = writeln!(
        out,
        "{i2}\"entry_point_rva\": {},",
        pe.optional.entry_point_rva
    );
    match entry_file_off {
        Some(off) => {
            let _ = writeln!(out, "{i2}\"entry_point_file_offset\": {off},");
        }
        None => {
            let _ = writeln!(out, "{i2}\"entry_point_file_offset\": null,");
        }
    }
    let _ = writeln!(out, "{i2}\"base_of_code\": {},", pe.optional.base_of_code);
    let _ = writeln!(out, "{i2}\"image_base\": {},", pe.optional.image_base);
    let _ = writeln!(
        out,
        "{i2}\"image_base_hex\": {},",
        json_str(&format!("0x{:016X}", pe.optional.image_base))
    );
    let _ = writeln!(
        out,
        "{i2}\"section_alignment\": {},",
        pe.optional.section_alignment
    );
    let _ = writeln!(
        out,
        "{i2}\"file_alignment\": {},",
        pe.optional.file_alignment
    );
    let _ = writeln!(out, "{i2}\"size_of_image\": {},", pe.optional.size_of_image);
    let _ = writeln!(
        out,
        "{i2}\"size_of_headers\": {},",
        pe.optional.size_of_headers
    );
    let _ = writeln!(out, "{i2}\"checksum\": {},", pe.optional.checksum);
    let _ = writeln!(out, "{i2}\"subsystem\": {},", pe.optional.subsystem);
    let _ = writeln!(
        out,
        "{i2}\"subsystem_name\": {},",
        json_str(pe.optional.subsystem_name())
    );
    let _ = writeln!(
        out,
        "{i2}\"dll_characteristics\": {},",
        pe.optional.dll_characteristics
    );
    let _ = writeln!(
        out,
        "{i2}\"number_of_rva_and_sizes\": {},",
        pe.optional.number_of_rva_and_sizes
    );
    let _ = writeln!(out, "{i2}\"data_directories\": [");
    for (idx, dd) in pe.optional.data_directories.iter().enumerate() {
        let comma = if idx + 1 < pe.optional.data_directories.len() {
            ","
        } else {
            ""
        };
        let _ = writeln!(
            out,
            "{i3}{{\"index\": {idx}, \"virtual_address\": {}, \"size\": {}}}{comma}",
            dd.virtual_address, dd.size
        );
    }
    let _ = writeln!(out, "{i2}]");
    let _ = writeln!(out, "{i1}}},");

    // Sections
    let _ = writeln!(out, "{i1}\"sections\": [");
    for (idx, sec) in pe.sections.iter().enumerate() {
        let comma = if idx + 1 < pe.sections.len() { "," } else { "" };
        let _ = writeln!(out, "{i2}{{");
        let _ = writeln!(out, "{i3}\"name\": {},", json_str(&sec.name));
        let _ = writeln!(out, "{i3}\"virtual_address\": {},", sec.virtual_address);
        let _ = writeln!(out, "{i3}\"virtual_size\": {},", sec.virtual_size);
        let _ = writeln!(out, "{i3}\"raw_offset\": {},", sec.raw_offset);
        let _ = writeln!(out, "{i3}\"raw_size\": {},", sec.raw_size);
        let _ = writeln!(out, "{i3}\"permissions\": {},", json_str(&sec.permissions_str()));
        let _ = writeln!(out, "{i3}\"characteristics\": {}", sec.characteristics);
        let _ = writeln!(out, "{i2}}}{comma}");
    }
    let _ = writeln!(out, "{i1}],");

    // TLS
    match &pe.tls {
        None => {
            let _ = writeln!(out, "{i1}\"tls\": null,");
        }
        Some(tls) => {
            let _ = writeln!(out, "{i1}\"tls\": {{");
            let _ = writeln!(
                out,
                "{i2}\"start_address_of_raw_data\": {},",
                tls.start_address_of_raw_data
            );
            let _ = writeln!(
                out,
                "{i2}\"end_address_of_raw_data\": {},",
                tls.end_address_of_raw_data
            );
            let _ = writeln!(out, "{i2}\"address_of_index\": {},", tls.address_of_index);
            let _ = writeln!(
                out,
                "{i2}\"address_of_callbacks\": {},",
                tls.address_of_callbacks
            );
            let _ = writeln!(out, "{i2}\"size_of_zero_fill\": {},", tls.size_of_zero_fill);
            let _ = writeln!(out, "{i2}\"characteristics\": {},", tls.characteristics);
            let _ = writeln!(out, "{i2}\"callbacks\": [");
            for (idx, cb) in tls.callbacks.iter().enumerate() {
                let comma = if idx + 1 < tls.callbacks.len() {
                    ","
                } else {
                    ""
                };
                let _ = writeln!(
                    out,
                    "{i3}{{\"rva\": {}, \"rva_hex\": {}, \"va\": {}, \"va_hex\": {}}}{comma}",
                    cb.rva,
                    json_str(&format!("0x{:08X}", cb.rva)),
                    cb.va,
                    json_str(&format!("0x{:016X}", cb.va))
                );
            }
            let _ = writeln!(out, "{i2}]");
            let _ = writeln!(out, "{i1}}},");
        }
    }

    // Standard Imports
    let _ = writeln!(out, "{i1}\"imports\": [");
    for (idx, module) in pe.imports.iter().enumerate() {
        let comma = if idx + 1 < pe.imports.len() { "," } else { "" };
        let _ = writeln!(out, "{i2}{{");
        let _ = writeln!(out, "{i3}\"dll\": {},", json_str(&module.dll_name));
        let _ = writeln!(out, "{i3}\"symbols\": [");
        write_symbols_json(out, &module.symbols, &i4);
        let _ = writeln!(out, "{i3}]");
        let _ = writeln!(out, "{i2}}}{comma}");
    }
    let _ = writeln!(out, "{i1}],");

    // Delay-Load Imports
    let _ = writeln!(out, "{i1}\"delay_imports\": [");
    for (idx, module) in pe.delay_imports.iter().enumerate() {
        let comma = if idx + 1 < pe.delay_imports.len() {
            ","
        } else {
            ""
        };
        let _ = writeln!(out, "{i2}{{");
        let _ = writeln!(out, "{i3}\"dll\": {},", json_str(&module.dll_name));
        let _ = writeln!(out, "{i3}\"attributes\": {},", module.attributes);
        let _ = writeln!(out, "{i3}\"uses_rva\": {},", module.uses_rva);
        let _ = writeln!(out, "{i3}\"symbols\": [");
        write_symbols_json(out, &module.symbols, &i4);
        let _ = writeln!(out, "{i3}]");
        let _ = writeln!(out, "{i2}}}{comma}");
    }
    let _ = writeln!(out, "{i1}],");

    // Exports
    match &pe.exports {
        None => {
            let _ = writeln!(out, "{i1}\"exports\": null");
        }
        Some(exports) => {
            let _ = writeln!(out, "{i1}\"exports\": {{");
            match &exports.module_name {
                Some(name) => {
                    let _ = writeln!(out, "{i2}\"module_name\": {},", json_str(name));
                }
                None => {
                    let _ = writeln!(out, "{i2}\"module_name\": null,");
                }
            }
            let _ = writeln!(out, "{i2}\"ordinal_base\": {},", exports.ordinal_base);
            let _ = writeln!(out, "{i2}\"symbols\": [");
            for (idx, sym) in exports.symbols.iter().enumerate() {
                let comma = if idx + 1 < exports.symbols.len() {
                    ","
                } else {
                    ""
                };
                let name_json = sym
                    .name
                    .as_deref()
                    .map(json_str)
                    .unwrap_or_else(|| "null".to_string());
                let fwd_json = sym
                    .forwarder
                    .as_deref()
                    .map(json_str)
                    .unwrap_or_else(|| "null".to_string());
                let _ = writeln!(
                    out,
                    "{i3}{{\"ordinal\": {}, \"rva\": {}, \"rva_hex\": {}, \"name\": {name_json}, \"forwarder\": {fwd_json}}}{comma}",
                    sym.ordinal,
                    sym.rva,
                    json_str(&format!("0x{:08X}", sym.rva))
                );
            }
            let _ = writeln!(out, "{i2}]");
            let _ = writeln!(out, "{i1}}}");
        }
    }

    let _ = write!(out, "{i0}}}");
}

fn write_symbols_json(out: &mut String, symbols: &[ImportedSymbol], indent: &str) {
    for (idx, sym) in symbols.iter().enumerate() {
        let comma = if idx + 1 < symbols.len() { "," } else { "" };
        match sym {
            ImportedSymbol::ByName { hint, name } => {
                let _ = writeln!(
                    out,
                    "{indent}{{\"kind\": \"name\", \"name\": {}, \"hint\": {hint}}}{comma}",
                    json_str(name)
                );
            }
            ImportedSymbol::ByOrdinal { ordinal } => {
                let _ = writeln!(
                    out,
                    "{indent}{{\"kind\": \"ordinal\", \"ordinal\": {ordinal}}}{comma}"
                );
            }
        }
    }
}

/// Properly escapes a Rust string into a valid RFC 8259 JSON string literal.
pub fn json_str(input: &str) -> String {
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
                let _ = write!(out, "\\u{:04X}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Sanitizes a report source label (stripping host directory prefixes and control characters).
#[allow(dead_code)]
pub fn sanitize_source_label(raw: &str) -> String {
    sanitize_label(raw)
}

/// Percentage of discovered functions lifted (`0.00..=100.00`, zero-division safe).
#[allow(dead_code)]
pub fn lifted_function_pct(lifted: usize, total: usize) -> f64 {
    percentage_2dp(lifted, total)
}

/// Percentage of instructions decoded/supported (`0.00..=100.00`, zero-division safe).
#[allow(dead_code)]
pub fn decoded_instruction_pct(decoded: usize, total: usize) -> f64 {
    percentage_2dp(decoded, total)
}

/// Percentage of imports bound to shims or guest modules (`0.00..=100.00`, zero-division safe).
#[allow(dead_code)]
pub fn bound_import_pct(bound: usize, total: usize) -> f64 {
    percentage_2dp(bound, total)
}

/// Formats a deterministic, sanitized one-line top-N breakdown (`"a (5), b (2), +1 more"`).
#[allow(dead_code)]
pub fn format_top_n_breakdown<I, S>(entries: I, limit: usize) -> String
where
    I: IntoIterator<Item = (S, usize)>,
    S: AsRef<str>,
{
    format_top_n(entries, limit)
}

/// Sanitized instruction-coverage & unresolved-import breakdown for CLI/program reports.
///
/// Holds counts, percentages, and sanitized labels only — never host paths or raw binary bytes.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoverageReport {
    pub source: String,
    pub summary: CoverageSummary,
    pub unresolved_by_dll: Vec<RankedCount>,
    pub unsupported_mnemonics: Vec<RankedCount>,
}

#[allow(dead_code)]
impl CoverageReport {
    /// Creates a new sanitized coverage report from a source label and [`CoverageSummary`].
    pub fn new(source_label: &str, summary: CoverageSummary) -> Self {
        Self {
            source: sanitize_source_label(source_label),
            summary,
            unresolved_by_dll: Vec::new(),
            unsupported_mnemonics: Vec::new(),
        }
    }

    /// Attaches a sanitized, deterministically ordered unresolved-import breakdown by DLL.
    pub fn with_unresolved_imports<I, S>(mut self, entries: I) -> Self
    where
        I: IntoIterator<Item = (S, usize)>,
        S: AsRef<str>,
    {
        self.unresolved_by_dll = rank_top_n(entries, usize::MAX);
        self
    }

    /// Attaches a sanitized, deterministically ordered unsupported-mnemonic breakdown.
    pub fn with_unsupported_mnemonics<I, S>(mut self, entries: I) -> Self
    where
        I: IntoIterator<Item = (S, usize)>,
        S: AsRef<str>,
    {
        self.unsupported_mnemonics = rank_top_n(entries, usize::MAX);
        self
    }

    /// Builds an import-coverage breakdown directly from a parsed [`Pe64`] (combining standard
    /// and delay-load imports) and a per-DLL map of bound import counts (matched
    /// case-insensitively by DLL name).
    pub fn from_pe_import_counts(
        pe: &Pe64,
        source_label: &str,
        bound_by_dll: &BTreeMap<String, usize>,
    ) -> Self {
        let mut bound_lookup: BTreeMap<String, usize> = BTreeMap::new();
        for (dll, &count) in bound_by_dll {
            let key = sanitize_label(dll).to_ascii_lowercase();
            let slot = bound_lookup.entry(key).or_insert(0);
            *slot = slot.saturating_add(count);
        }

        let mut totals_by_dll: BTreeMap<String, (String, usize)> = BTreeMap::new();
        for module in &pe.imports {
            let clean = sanitize_label(&module.dll_name);
            let key = clean.to_ascii_lowercase();
            let entry = totals_by_dll.entry(key).or_insert((clean, 0));
            entry.1 = entry.1.saturating_add(module.symbols.len());
        }
        for module in &pe.delay_imports {
            let clean = sanitize_label(&module.dll_name);
            let key = clean.to_ascii_lowercase();
            let entry = totals_by_dll.entry(key).or_insert((clean, 0));
            entry.1 = entry.1.saturating_add(module.symbols.len());
        }

        let mut total_imports = 0usize;
        let mut bound_imports = 0usize;
        let mut unresolved = Vec::new();
        for (key, (display_dll, count)) in totals_by_dll {
            let bound = bound_lookup.get(&key).copied().unwrap_or(0).min(count);
            let missing = count.saturating_sub(bound);
            total_imports = total_imports.saturating_add(count);
            bound_imports = bound_imports.saturating_add(bound);
            if missing > 0 {
                unresolved.push((display_dll, missing));
            }
        }

        let summary = CoverageSummary::default().with_imports(total_imports, bound_imports);
        Self::new(source_label, summary).with_unresolved_imports(unresolved)
    }

    /// Percentage of discovered functions lifted (`0.00..=100.00`).
    pub fn lifted_function_pct(&self) -> f64 {
        self.summary.lifted_function_pct()
    }

    /// Percentage of instructions decoded/supported (`0.00..=100.00`).
    pub fn decoded_instruction_pct(&self) -> f64 {
        self.summary.decoded_instruction_pct()
    }

    /// Percentage of imports bound (`0.00..=100.00`).
    pub fn bound_import_pct(&self) -> f64 {
        self.summary.bound_import_pct()
    }

    /// Formats a crisp one-line progress ratio prefixed with the sanitized source label.
    pub fn render_one_line(&self) -> String {
        format!("[{}] {}", self.source, self.summary.format_one_line())
    }

    /// Formats a crisp one-line progress ratio plus deterministic top-N unresolved-import and
    /// unsupported-mnemonic breakdowns.
    pub fn render_breakdown_line(&self, top_n: usize) -> String {
        let mut out = self.render_one_line();
        if !self.unresolved_by_dll.is_empty() {
            let top = format_top_n(
                self.unresolved_by_dll
                    .iter()
                    .map(|r| (r.label.as_str(), r.count)),
                top_n,
            );
            let _ = write!(out, " | top unresolved: {top}");
        }
        if !self.unsupported_mnemonics.is_empty() {
            let top = format_top_n(
                self.unsupported_mnemonics
                    .iter()
                    .map(|r| (r.label.as_str(), r.count)),
                top_n,
            );
            let _ = write!(out, " | top unsupported: {top}");
        }
        out
    }

    /// Renders a sanitized Markdown coverage summary with deterministic top-N breakdown tables.
    pub fn render_coverage_markdown(&self, top_n: usize) -> String {
        let mut out = String::with_capacity(512);
        let _ = writeln!(out, "## Coverage Summary — `{}`", md_inline(&self.source));
        let _ = writeln!(out);
        let _ = writeln!(out, "- {}", self.summary.format_one_line());
        let _ = writeln!(out);
        let _ = writeln!(out, "| Metric | Covered | Total | Coverage |");
        let _ = writeln!(out, "| --- | ---: | ---: | ---: |");
        let _ = writeln!(
            out,
            "| Lifted Functions | {} | {} | {:.2}% |",
            self.summary.lifted_functions,
            self.summary.total_functions,
            self.lifted_function_pct()
        );
        let _ = writeln!(
            out,
            "| Decoded Instructions | {} | {} | {:.2}% |",
            self.summary.decoded_instructions,
            self.summary.total_instructions,
            self.decoded_instruction_pct()
        );
        let _ = writeln!(
            out,
            "| Bound Imports | {} | {} | {:.2}% |",
            self.summary.bound_imports,
            self.summary.total_imports,
            self.bound_import_pct()
        );

        let top_unresolved = format_top_n(
            self.unresolved_by_dll
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
        let _ = writeln!(out);
        let _ = writeln!(
            out,
            "- Top Unresolved Imports (top {top_n}): {}",
            md_inline(&top_unresolved)
        );
        let _ = writeln!(
            out,
            "- Top Unsupported Mnemonics (top {top_n}): {}",
            md_inline(&top_unsupported)
        );
        out
    }

    /// Renders a sanitized RFC 8259 JSON coverage summary with deterministic top-N breakdowns.
    pub fn render_coverage_json(&self, top_n: usize) -> String {
        let mut out = String::with_capacity(512);
        let _ = writeln!(out, "{{");
        let _ = writeln!(out, "  \"source\": {},", json_str(&self.source));
        let _ = writeln!(
            out,
            "  \"summary_line\": {},",
            json_str(&self.render_breakdown_line(top_n))
        );
        let _ = writeln!(
            out,
            "  \"total_functions\": {},",
            self.summary.total_functions
        );
        let _ = writeln!(
            out,
            "  \"lifted_functions\": {},",
            self.summary.lifted_functions
        );
        let _ = writeln!(
            out,
            "  \"lifted_function_pct\": {:.2},",
            self.lifted_function_pct()
        );
        let _ = writeln!(
            out,
            "  \"total_instructions\": {},",
            self.summary.total_instructions
        );
        let _ = writeln!(
            out,
            "  \"decoded_instructions\": {},",
            self.summary.decoded_instructions
        );
        let _ = writeln!(
            out,
            "  \"decoded_instruction_pct\": {:.2},",
            self.decoded_instruction_pct()
        );
        let _ = writeln!(out, "  \"total_imports\": {},", self.summary.total_imports);
        let _ = writeln!(out, "  \"bound_imports\": {},", self.summary.bound_imports);
        let _ = writeln!(
            out,
            "  \"unresolved_imports\": {},",
            self.summary.unresolved_imports()
        );
        let _ = writeln!(
            out,
            "  \"bound_import_pct\": {:.2},",
            self.bound_import_pct()
        );
        write_ranked_json(&mut out, "top_unresolved_imports", &self.unresolved_by_dll, top_n, true);
        write_ranked_json(
            &mut out,
            "top_unsupported_mnemonics",
            &self.unsupported_mnemonics,
            top_n,
            false,
        );
        let _ = writeln!(out, "}}");
        out
    }
}

#[allow(dead_code)]
fn write_ranked_json(
    out: &mut String,
    key: &str,
    items: &[RankedCount],
    top_n: usize,
    trailing_comma: bool,
) {
    let comma_end = if trailing_comma { "," } else { "" };
    let shown = items.len().min(top_n);
    if shown == 0 {
        let _ = writeln!(out, "  \"{key}\": []{comma_end}");
        return;
    }
    let _ = writeln!(out, "  \"{key}\": [");
    for (idx, item) in items.iter().take(shown).enumerate() {
        let comma = if idx + 1 < shown { "," } else { "" };
        let _ = writeln!(
            out,
            "    {{\"label\": {}, \"count\": {}}}{comma}",
            json_str(&item.label),
            item.count
        );
    }
    let _ = writeln!(out, "  ]{comma_end}");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pe::test_util::build_synthetic_pe64;

    #[test]
    fn markdown_report_contains_all_expected_sections_and_symbols() {
        let raw = build_synthetic_pe64(false);
        let pe = Pe64::parse(&raw).expect("synthetic PE should parse");
        let md = render_markdown(&pe, "SyntheticGame.exe");

        assert!(md.contains("# PE64 Inspection Report — `SyntheticGame.exe`"));
        assert!(md.contains("| Machine | `AMD64` (`0x8664`) |"));
        assert!(md.contains("| Subsystem | `XBOX` (`0x000E`) |"));
        assert!(md.contains("| Image Base | `0x0000000140000000` |"));
        assert!(md.contains("| `.text` | `0x00001000` |"));
        assert!(md.contains("| `.rdata` | `0x00002000` |"));

        // TLS callbacks
        assert!(md.contains("## TLS Callbacks"));
        assert!(md.contains("| 0 | `0x00001030` | `0x0000000140001030` |"));
        assert!(md.contains("| 1 | `0x00001040` | `0x0000000140001040` |"));

        // Standard imports
        assert!(md.contains("### `KERNEL32.dll` (2 symbols)"));
        assert!(md.contains("| name | `VirtualAlloc` | hint `0x0123` |"));
        assert!(md.contains("| ordinal | `#42` | ordinal `42` |"));
        assert!(md.contains("### `xgameruntime.dll` (1 symbol)"));
        assert!(md.contains("| name | `XTaskQueueCreate` | hint `0x0007` |"));

        // Delay-load imports
        assert!(md.contains("## Delay-Load Imports"));
        assert!(md.contains("### `d3d12_xs.dll` (2 symbols, mode: RVA, attrs: `0x1`)"));
        assert!(md.contains("| name | `D3D12CreateDevice` | hint `0x0015` |"));
        assert!(md.contains("| ordinal | `#99` | ordinal `99` |"));

        // Exports
        assert!(md.contains("## Exports"));
        assert!(md.contains("Module Name: `SyntheticGame.exe` | Ordinal Base: 10"));
        assert!(md.contains("| 10 | `0x00001010` | `GameMain` | - |"));
        assert!(md.contains("| 11 | `0x00001020` | _(ordinal only)_ | - |"));
        assert!(md.contains("| 12 | `0x00002340` | `ForwardedLog` | `KERNEL32.OutputDebugStringA` |"));
    }

    /// Strict RFC 8259 JSON syntax validator (no external crates) to ensure our hand-written
    /// serializer never produces trailing commas, bad escapes, or invalid tokens.
    struct JsonValidator<'a> {
        bytes: &'a [u8],
        pos: usize,
    }

    impl<'a> JsonValidator<'a> {
        fn validate(input: &'a str) -> Result<(), String> {
            let mut v = Self {
                bytes: input.as_bytes(),
                pos: 0,
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
                && matches!(self.bytes[self.pos], b' ' | b'\t' | b'\r' | b'\n')
            {
                self.pos += 1;
            }
        }

        fn parse_value(&mut self) -> Result<(), String> {
            self.skip_ws();
            let b = *self.bytes.get(self.pos).ok_or("unexpected EOF")?;
            match b {
                b'{' => self.parse_object(),
                b'[' => self.parse_array(),
                b'"' => self.parse_string(),
                b't' => self.expect_bytes(b"true"),
                b'f' => self.expect_bytes(b"false"),
                b'n' => self.expect_bytes(b"null"),
                b'-' | b'0'..=b'9' => self.parse_number(),
                other => Err(format!("unexpected byte 0x{other:02X} at {}", self.pos)),
            }
        }

        fn expect_bytes(&mut self, lit: &[u8]) -> Result<(), String> {
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
            if self.bytes.get(self.pos) == Some(&b'}') {
                self.pos += 1;
                return Ok(());
            }
            loop {
                self.skip_ws();
                self.parse_string()?;
                self.skip_ws();
                if self.bytes.get(self.pos) != Some(&b':') {
                    return Err(format!("expected ':' at {}", self.pos));
                }
                self.pos += 1;
                self.parse_value()?;
                self.skip_ws();
                match self.bytes.get(self.pos) {
                    Some(b',') => self.pos += 1,
                    Some(b'}') => {
                        self.pos += 1;
                        return Ok(());
                    }
                    _ => return Err(format!("expected ',' or '}}' at {}", self.pos)),
                }
            }
        }

        fn parse_array(&mut self) -> Result<(), String> {
            self.pos += 1; // '['
            self.skip_ws();
            if self.bytes.get(self.pos) == Some(&b']') {
                self.pos += 1;
                return Ok(());
            }
            loop {
                self.parse_value()?;
                self.skip_ws();
                match self.bytes.get(self.pos) {
                    Some(b',') => self.pos += 1,
                    Some(b']') => {
                        self.pos += 1;
                        return Ok(());
                    }
                    _ => return Err(format!("expected ',' or ']' at {}", self.pos)),
                }
            }
        }

        fn parse_string(&mut self) -> Result<(), String> {
            if self.bytes.get(self.pos) != Some(&b'"') {
                return Err(format!("expected '\"' at {}", self.pos));
            }
            self.pos += 1;
            while let Some(&b) = self.bytes.get(self.pos) {
                self.pos += 1;
                match b {
                    b'"' => return Ok(()),
                    b'\\' => {
                        let esc = *self.bytes.get(self.pos).ok_or("unterminated escape")?;
                        self.pos += 1;
                        match esc {
                            b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => {}
                            b'u' => {
                                for _ in 0..4 {
                                    let h = *self.bytes.get(self.pos).ok_or("short \\u escape")?;
                                    if !h.is_ascii_hexdigit() {
                                        return Err(format!("bad hex digit at {}", self.pos));
                                    }
                                    self.pos += 1;
                                }
                            }
                            _ => return Err(format!("invalid escape at {}", self.pos)),
                        }
                    }
                    0x00..=0x1F => return Err(format!("unescaped control char at {}", self.pos)),
                    _ => {}
                }
            }
            Err("unterminated string".to_string())
        }

        fn parse_number(&mut self) -> Result<(), String> {
            if self.bytes.get(self.pos) == Some(&b'-') {
                self.pos += 1;
            }
            let start = self.pos;
            while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
            if self.pos == start {
                return Err(format!("invalid number at {}", self.pos));
            }
            if self.bytes.get(self.pos) == Some(&b'.') {
                self.pos += 1;
                let frac_start = self.pos;
                while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_digit() {
                    self.pos += 1;
                }
                if self.pos == frac_start {
                    return Err(format!("invalid fractional number at {}", self.pos));
                }
            }
            Ok(())
        }
    }

    #[test]
    fn json_report_renders_valid_structure_and_escapes_strings() {
        let raw = build_synthetic_pe64(false);
        let pe = Pe64::parse(&raw).expect("synthetic PE should parse");
        let json = render_json(&pe, "Synthetic\"Game\\v1.exe");

        JsonValidator::validate(&json).expect("single PE JSON must be valid RFC 8259 JSON");
        assert!(json.contains("\"source\": \"Synthetic\\\"Game\\\\v1.exe\""));
        assert!(json.contains("\"machine_name\": \"AMD64\""));
        assert!(json.contains("\"subsystem_name\": \"XBOX\""));
        assert!(json.contains("\"name\": \"VirtualAlloc\""));
        assert!(json.contains("\"ordinal\": 42"));
        assert!(json.contains("\"dll\": \"d3d12_xs.dll\""));
        assert!(json.contains("\"uses_rva\": true"));
        assert!(json.contains("\"forwarder\": \"KERNEL32.OutputDebugStringA\""));
        assert!(json.contains("\"rva_hex\": \"0x00001030\""));

        // Verify json_str escaping on control chars
        assert_eq!(
            json_str("a\"b\\c\n\r\t\x01"),
            "\"a\\\"b\\\\c\\n\\r\\t\\u0001\""
        );
    }

    #[test]
    fn json_many_and_empty_tables_produce_valid_json_and_markdown() {
        let raw = build_synthetic_pe64(false);
        let pe = Pe64::parse(&raw).expect("synthetic PE should parse");
        let mut minimal = pe.clone();
        minimal.imports.clear();
        minimal.delay_imports.clear();
        minimal.exports = None;
        minimal.tls = None;

        let md_min = render_markdown(&minimal, "minimal|test`bin.exe");
        assert!(md_min.contains("_No TLS directory present._"));
        assert!(md_min.contains("_No standard imports._"));
        assert!(md_min.contains("_No delay-load imports._"));
        assert!(md_min.contains("_No export directory present._"));

        let json_min = render_json(&minimal, "minimal.exe");
        JsonValidator::validate(&json_min).expect("minimal PE JSON must be valid");
        assert!(json_min.contains("\"tls\": null"));
        assert!(json_min.contains("\"exports\": null"));

        let json_many = render_json_many(&[
            ("a.exe".to_string(), pe),
            ("b.exe".to_string(), minimal),
        ]);
        JsonValidator::validate(&json_many).expect("multi-PE JSON array must be valid");
        assert!(json_many.contains("\"source\": \"a.exe\""));
        assert!(json_many.contains("\"source\": \"b.exe\""));
    }

    #[test]
    fn coverage_pct_helpers_handle_zero_division_and_rounding() {
        assert_eq!(lifted_function_pct(0, 0), 0.0);
        assert_eq!(decoded_instruction_pct(10, 0), 0.0);
        assert_eq!(bound_import_pct(0, 0), 0.0);

        assert_eq!(lifted_function_pct(1, 3), 33.33);
        assert_eq!(decoded_instruction_pct(2, 3), 66.67);
        assert_eq!(bound_import_pct(3, 8), 37.5);
        assert_eq!(lifted_function_pct(9, 5), 100.0);
        assert_eq!(decoded_instruction_pct(1, 20_000), 0.01);
        assert_eq!(bound_import_pct(1, 40_000), 0.0);
    }

    #[test]
    fn coverage_report_from_pe_computes_deterministic_sanitized_breakdowns() {
        let raw = build_synthetic_pe64(false);
        let pe = Pe64::parse(&raw).expect("synthetic PE should parse");

        // Synthetic PE has:
        //   KERNEL32.dll: 2 standard imports
        //   xgameruntime.dll: 1 standard import
        //   d3d12_xs.dll: 2 delay-load imports
        let mut bound = BTreeMap::new();
        bound.insert("kernel32.dll".to_string(), 1);
        bound.insert("xgameruntime.dll".to_string(), 1);

        let report = CoverageReport::from_pe_import_counts(
            &pe,
            r"C:\Private\HostPath\SyntheticGame.exe",
            &bound,
        )
        .with_unsupported_mnemonics(vec![
            ("vmovdqu", 3),
            ("aesenc", 7),
            ("cpuid", 7),
            ("zero_op", 0),
        ]);

        // Host path must never appear in any output.
        assert_eq!(report.source, "SyntheticGame.exe");
        assert_eq!(report.summary.total_imports, 5);
        assert_eq!(report.summary.bound_imports, 2);
        assert_eq!(report.summary.unresolved_imports(), 3);
        assert_eq!(report.bound_import_pct(), 40.0);
        assert_eq!(report.lifted_function_pct(), 0.0);
        assert_eq!(report.decoded_instruction_pct(), 0.0);

        let one_line = report.render_one_line();
        assert_eq!(
            one_line,
            "[SyntheticGame.exe] functions 0/0 (0.00%) | instructions 0/0 (0.00%) | imports 2/5 (40.00%, 3 unresolved)"
        );
        assert!(!one_line.contains("Private"));
        assert!(!one_line.contains("HostPath"));

        let breakdown_top1 = report.render_breakdown_line(1);
        assert_eq!(
            breakdown_top1,
            "[SyntheticGame.exe] functions 0/0 (0.00%) | instructions 0/0 (0.00%) | imports 2/5 (40.00%, 3 unresolved) | top unresolved: d3d12_xs.dll (2), +1 more | top unsupported: aesenc (7), +2 more"
        );

        let breakdown_all = report.render_breakdown_line(5);
        assert!(breakdown_all.contains("top unresolved: d3d12_xs.dll (2), KERNEL32.dll (1)"));
        assert!(breakdown_all.contains("top unsupported: aesenc (7), cpuid (7), vmovdqu (3)"));

        let md = report.render_coverage_markdown(2);
        assert!(md.contains("## Coverage Summary — `SyntheticGame.exe`"));
        assert!(md.contains("| Bound Imports | 2 | 5 | 40.00% |"));
        assert!(md.contains("- Top Unresolved Imports (top 2): d3d12_xs.dll (2), KERNEL32.dll (1)"));
        assert!(md.contains("- Top Unsupported Mnemonics (top 2): aesenc (7), cpuid (7), +1 more"));
        assert!(!md.contains("Private"));

        let json = report.render_coverage_json(2);
        JsonValidator::validate(&json).expect("coverage report JSON must be valid RFC 8259 JSON");
        assert!(json.contains("\"source\": \"SyntheticGame.exe\""));
        assert!(json.contains("\"bound_import_pct\": 40.00"));
        assert!(json.contains("\"unresolved_imports\": 3"));
        assert!(json.contains("{\"label\": \"d3d12_xs.dll\", \"count\": 2}"));
        assert!(json.contains("{\"label\": \"aesenc\", \"count\": 7}"));
        assert!(!json.contains("Private"));
    }

    #[test]
    fn coverage_report_top_n_order_is_independent_of_input_order_and_handles_zero_limit() {
        let summary = CoverageSummary {
            total_functions: 12,
            lifted_functions: 10,
            total_instructions: 300,
            decoded_instructions: 299,
            total_imports: 8,
            bound_imports: 6,
        };
        let r1 = CoverageReport::new("/tmp/secret/game.exe", summary)
            .with_unresolved_imports(vec![("b.dll", 2), ("a.dll", 2), ("c.dll", 5)]);
        let r2 = CoverageReport::new(r"C:\tmp\secret\game.exe", summary)
            .with_unresolved_imports(vec![("a.dll", 2), ("c.dll", 5), ("b.dll", 2)]);

        assert_eq!(r1, r2);
        assert_eq!(r1.render_breakdown_line(2), r2.render_breakdown_line(2));
        assert_eq!(
            r1.render_breakdown_line(2),
            "[game.exe] functions 10/12 (83.33%) | instructions 299/300 (99.67%) | imports 6/8 (75.00%, 2 unresolved) | top unresolved: c.dll (5), a.dll (2), +1 more"
        );
        assert_eq!(
            r1.render_breakdown_line(0),
            "[game.exe] functions 10/12 (83.33%) | instructions 299/300 (99.67%) | imports 6/8 (75.00%, 2 unresolved) | top unresolved: none"
        );
        assert_eq!(
            format_top_n_breakdown(vec![("z", 1), ("a", 1)], 2),
            "a (1), z (1)"
        );

        let json_zero = r1.render_coverage_json(0);
        JsonValidator::validate(&json_zero).expect("zero top-N JSON must be valid");
        assert!(json_zero.contains("\"top_unresolved_imports\": []"));
        assert!(json_zero.contains("\"top_unsupported_mnemonics\": []"));
    }
}
