//! Sanitized Markdown and JSON report renderers for parsed PE32+ metadata.
//!
//! Outputs symbol names, section metadata, and addresses ONLY — never raw binary bytes.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use playgta6_recomp::pipeline::ShardRecord;
use playgta6_recomp::summary::{
    CoverageSummary, FunctionSizeStats, RankedCount, V8_MAX_WASM_FUNCTION_BODY_BYTES,
    V8_MAX_WASM_FUNCTION_LOCALS, V8_MAX_WASM_MODULE_BYTES, Wasm64HeadroomReport, format_top_n,
    mib_2dp, percentage_2dp, percentage_u64_2dp, rank_top_n, ratio_2dp, sanitize_label,
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

/// Sanitizes a report source label (stripping Windows/Unix host directory prefixes,
/// bare directory trails, relative traversal segments, and control characters).
#[allow(dead_code)]
pub fn sanitize_source_label(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.ends_with('/') || trimmed.ends_with('\\') {
        return "<unknown>".to_string();
    }
    let cleaned = sanitize_label(trimmed);
    if cleaned == "." || cleaned == ".." {
        return "<unknown>".to_string();
    }
    cleaned
}

/// Returns `true` if `text` appears to contain a leaked Windows or Unix host filesystem path
/// (drive letters like `C:\` or `C:/`, UNC/extended paths like `\\server\share` or `\\?\`,
/// `file://` URIs, or common Unix root directories like `/home/` or `/Users/`).
#[allow(dead_code)]
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

    // 3. Windows UNC / extended-length paths (`\\?\`, `\\.\`, JSON-escaped `\\\\`,
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

    // 4. Common Unix/macOS host root prefixes
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

/// Default maximum skew ratio (`max / mean`) at or below which a Wasm64 shard layout is
/// considered balanced for parallel compilation and fast-boot streaming.
#[allow(dead_code)]
pub const DEFAULT_MAX_SKEW_RATIO: f64 = 1.50;

/// Sanitized per-shard measurement for Wasm64 fast-boot and parallel-compile diagnostics.
///
/// Holds a sanitized shard label (never a host directory path), function count, and byte size.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardEntry {
    pub name: String,
    pub functions: usize,
    pub bytes: u64,
}

#[allow(dead_code)]
impl ShardEntry {
    /// Creates a sanitized shard measurement, stripping any host directory prefixes from `name`.
    pub fn new(name: &str, functions: usize, bytes: u64) -> Self {
        Self {
            name: sanitize_source_label(name),
            functions,
            bytes,
        }
    }
}

/// Pure, deterministic Wasm64 shard skew & fast-boot balance metrics.
///
/// Computes function-count and byte-size balance ratios (`max / mean`) with exact 2dp rounding
/// and zero-division guards for empty or zero-function shard sets.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub struct ShardBalanceSummary {
    pub shard_count: usize,
    pub total_functions: usize,
    pub min_shard_functions: usize,
    pub max_shard_functions: usize,
    pub mean_shard_functions: f64,
    pub skew_ratio: f64,
    pub total_bytes: u64,
    pub min_shard_bytes: u64,
    pub max_shard_bytes: u64,
    pub mean_shard_bytes: f64,
    pub byte_skew_ratio: f64,
    pub max_skew_threshold: f64,
    pub balanced: bool,
    pub shards: Vec<ShardEntry>,
}

#[allow(dead_code)]
impl ShardBalanceSummary {
    /// Computes shard balance metrics from raw per-shard function counts using
    /// [`DEFAULT_MAX_SKEW_RATIO`].
    pub fn from_function_counts(counts: &[usize]) -> Self {
        Self::from_function_counts_with_threshold(counts, DEFAULT_MAX_SKEW_RATIO)
    }

    /// Computes shard balance metrics from raw per-shard function counts using a custom maximum
    /// acceptable skew threshold (finite values `< 1.0` are clamped to `1.0`; non-finite values
    /// fall back to [`DEFAULT_MAX_SKEW_RATIO`]).
    pub fn from_function_counts_with_threshold(counts: &[usize], max_skew_threshold: f64) -> Self {
        let width = counts
            .len()
            .saturating_sub(1)
            .to_string()
            .len()
            .max(5);
        let entries = counts.iter().enumerate().map(|(idx, &functions)| {
            ShardEntry::new(&format!("shard-{idx:0width$}"), functions, 0)
        });
        Self::from_shards_with_threshold(entries, max_skew_threshold)
    }

    /// Computes shard balance metrics from sanitized [`ShardEntry`] items using
    /// [`DEFAULT_MAX_SKEW_RATIO`].
    pub fn from_shards<I>(entries: I) -> Self
    where
        I: IntoIterator<Item = ShardEntry>,
    {
        Self::from_shards_with_threshold(entries, DEFAULT_MAX_SKEW_RATIO)
    }

    /// Computes shard balance metrics from [`ShardEntry`] items using a custom maximum acceptable
    /// skew threshold (finite values `< 1.0` are clamped to `1.0`; non-finite values fall back to
    /// [`DEFAULT_MAX_SKEW_RATIO`]).
    ///
    /// All shard names are re-sanitized and deterministically sorted regardless of input order.
    pub fn from_shards_with_threshold<I>(entries: I, max_skew_threshold: f64) -> Self
    where
        I: IntoIterator<Item = ShardEntry>,
    {
        let threshold = if max_skew_threshold.is_finite() {
            (max_skew_threshold.max(1.0) * 100.0).round() / 100.0
        } else {
            DEFAULT_MAX_SKEW_RATIO
        };

        let mut shards: Vec<ShardEntry> = entries
            .into_iter()
            .map(|entry| ShardEntry::new(&entry.name, entry.functions, entry.bytes))
            .collect();
        shards.sort_by(|a, b| {
            a.name
                .to_ascii_lowercase()
                .cmp(&b.name.to_ascii_lowercase())
                .then_with(|| a.name.cmp(&b.name))
                .then_with(|| a.functions.cmp(&b.functions))
                .then_with(|| a.bytes.cmp(&b.bytes))
        });

        let shard_count = shards.len();
        if shard_count == 0 {
            return Self {
                shard_count: 0,
                total_functions: 0,
                min_shard_functions: 0,
                max_shard_functions: 0,
                mean_shard_functions: 0.0,
                skew_ratio: 0.0,
                total_bytes: 0,
                min_shard_bytes: 0,
                max_shard_bytes: 0,
                mean_shard_bytes: 0.0,
                byte_skew_ratio: 0.0,
                max_skew_threshold: threshold,
                balanced: true,
                shards,
            };
        }

        let mut min_shard_functions = usize::MAX;
        let mut max_shard_functions = 0usize;
        let mut total_fn_u128 = 0u128;

        let mut min_shard_bytes = u64::MAX;
        let mut max_shard_bytes = 0u64;
        let mut total_bytes_u128 = 0u128;

        for shard in &shards {
            min_shard_functions = min_shard_functions.min(shard.functions);
            max_shard_functions = max_shard_functions.max(shard.functions);
            total_fn_u128 = total_fn_u128.saturating_add(shard.functions as u128);

            min_shard_bytes = min_shard_bytes.min(shard.bytes);
            max_shard_bytes = max_shard_bytes.max(shard.bytes);
            total_bytes_u128 = total_bytes_u128.saturating_add(shard.bytes as u128);
        }

        let count_u128 = shard_count as u128;
        let total_functions = usize::try_from(total_fn_u128).unwrap_or(usize::MAX);
        let mean_shard_functions = ratio_2dp_u128(total_fn_u128, count_u128);
        let skew_ratio = if total_fn_u128 == 0 {
            0.0
        } else {
            ratio_2dp_product(max_shard_functions as u128, count_u128, total_fn_u128)
        };

        let total_bytes = u64::try_from(total_bytes_u128).unwrap_or(u64::MAX);
        let mean_shard_bytes = ratio_2dp_u128(total_bytes_u128, count_u128);
        let byte_skew_ratio = if total_bytes_u128 == 0 {
            0.0
        } else {
            ratio_2dp_product(max_shard_bytes as u128, count_u128, total_bytes_u128)
        };

        let balanced = skew_ratio <= threshold && byte_skew_ratio <= threshold;

        Self {
            shard_count,
            total_functions,
            min_shard_functions,
            max_shard_functions,
            mean_shard_functions,
            skew_ratio,
            total_bytes,
            min_shard_bytes,
            max_shard_bytes,
            mean_shard_bytes,
            byte_skew_ratio,
            max_skew_threshold: threshold,
            balanced,
            shards,
        }
    }

    /// Computes shard skew & fast-boot balance metrics directly from pipeline [`ShardRecord`]s.
    ///
    /// Uses each shard's compiled object byte size when present, otherwise its LLVM IR byte size.
    pub fn from_shard_records(records: &[ShardRecord]) -> Self {
        Self::from_shard_records_with_threshold(records, DEFAULT_MAX_SKEW_RATIO)
    }

    /// Computes shard skew & fast-boot balance metrics from pipeline [`ShardRecord`]s using a
    /// custom maximum acceptable skew threshold.
    pub fn from_shard_records_with_threshold(
        records: &[ShardRecord],
        max_skew_threshold: f64,
    ) -> Self {
        let entries = records.iter().map(|rec| {
            let file = rec.obj.as_ref().unwrap_or(&rec.ll);
            ShardEntry::new(&file.path, rec.functions, file.size)
        });
        Self::from_shards_with_threshold(entries, max_skew_threshold)
    }

    /// Formats a compact, deterministic one-line Wasm64 shard balance summary.
    pub fn render_one_line(&self) -> String {
        let status = if self.balanced { "balanced" } else { "skewed" };
        if self.total_bytes > 0 {
            format!(
                "shards {} (total {} fns, min {}, max {}, mean {:.2}, skew {:.2}x | bytes {}, min {}, max {}, mean {:.2}, byte_skew {:.2}x, {})",
                self.shard_count,
                self.total_functions,
                self.min_shard_functions,
                self.max_shard_functions,
                self.mean_shard_functions,
                self.skew_ratio,
                self.total_bytes,
                self.min_shard_bytes,
                self.max_shard_bytes,
                self.mean_shard_bytes,
                self.byte_skew_ratio,
                status,
            )
        } else {
            format!(
                "shards {} (total {} fns, min {}, max {}, mean {:.2}, skew {:.2}x, {})",
                self.shard_count,
                self.total_functions,
                self.min_shard_functions,
                self.max_shard_functions,
                self.mean_shard_functions,
                self.skew_ratio,
                status,
            )
        }
    }

    /// Renders a sanitized, deterministically ordered Markdown Wasm64 shard balance report.
    pub fn render_markdown(&self) -> String {
        let mut out = String::with_capacity(512);
        let status = if self.balanced { "balanced" } else { "skewed" };
        let _ = writeln!(out, "### Wasm64 Shard Balance");
        let _ = writeln!(out);
        let _ = writeln!(out, "- {}", self.render_one_line());
        let _ = writeln!(out);
        let _ = writeln!(out, "| Metric | Value |");
        let _ = writeln!(out, "| --- | ---: |");
        let _ = writeln!(out, "| Shards | {} |", self.shard_count);
        let _ = writeln!(out, "| Total Functions | {} |", self.total_functions);
        let _ = writeln!(out, "| Min / Max Functions | {} / {} |", self.min_shard_functions, self.max_shard_functions);
        let _ = writeln!(out, "| Mean Functions | {:.2} |", self.mean_shard_functions);
        let _ = writeln!(out, "| Function Skew Ratio | {:.2}x |", self.skew_ratio);
        let _ = writeln!(out, "| Total Bytes | {} |", self.total_bytes);
        let _ = writeln!(out, "| Min / Max Bytes | {} / {} |", self.min_shard_bytes, self.max_shard_bytes);
        let _ = writeln!(out, "| Mean Bytes | {:.2} |", self.mean_shard_bytes);
        let _ = writeln!(out, "| Byte Skew Ratio | {:.2}x |", self.byte_skew_ratio);
        let _ = writeln!(out, "| Max Skew Threshold | {:.2}x |", self.max_skew_threshold);
        let _ = writeln!(out, "| Status | {status} |");

        if !self.shards.is_empty() {
            let _ = writeln!(out);
            let _ = writeln!(out, "| Shard | Functions | Bytes |");
            let _ = writeln!(out, "| --- | ---: | ---: |");
            for shard in &self.shards {
                let _ = writeln!(
                    out,
                    "| `{}` | {} | {} |",
                    md_table_cell(&shard.name),
                    shard.functions,
                    shard.bytes
                );
            }
        }
        out
    }

    /// Renders a sanitized RFC 8259 JSON Wasm64 shard balance report with strictly sorted keys.
    pub fn render_json(&self) -> String {
        let mut out = String::with_capacity(512);
        write_shard_balance_json(&mut out, self, 0, true);
        out.push('\n');
        out
    }

    /// Total shard size in MiB, rounded half-up to 2 decimal places.
    pub fn total_bytes_mib(&self) -> f64 {
        mib_2dp(self.total_bytes)
    }

    /// Heaviest shard size in MiB, rounded half-up to 2 decimal places.
    pub fn max_shard_bytes_mib(&self) -> f64 {
        mib_2dp(self.max_shard_bytes)
    }

    /// Remaining total byte headroom under [`V8_MAX_WASM_MODULE_BYTES`] (`0` when at or over cap).
    pub fn v8_module_headroom_bytes(&self) -> u64 {
        V8_MAX_WASM_MODULE_BYTES.saturating_sub(self.total_bytes)
    }

    /// Total shard byte utilization against [`V8_MAX_WASM_MODULE_BYTES`] (`0.00..=100.00`).
    pub fn v8_module_utilization_pct(&self) -> f64 {
        percentage_u64_2dp(self.total_bytes, V8_MAX_WASM_MODULE_BYTES)
    }

    /// Returns `true` if `total_bytes` strictly exceeds [`V8_MAX_WASM_MODULE_BYTES`].
    pub fn exceeds_v8_module_cap(&self) -> bool {
        self.total_bytes > V8_MAX_WASM_MODULE_BYTES
    }

    /// Ratio `min_shard_functions / max_shard_functions`, rounded half-up to 2dp (`0.0` when max is 0).
    pub fn min_to_max_function_ratio(&self) -> f64 {
        if self.max_shard_functions == 0 {
            return 0.0;
        }
        ratio_2dp(
            self.min_shard_functions as u64,
            self.max_shard_functions as u64,
        )
    }

    /// Ratio `min_shard_bytes / max_shard_bytes`, rounded half-up to 2dp (`0.0` when max is 0).
    pub fn min_to_max_byte_ratio(&self) -> f64 {
        if self.max_shard_bytes == 0 {
            return 0.0;
        }
        ratio_2dp(self.min_shard_bytes, self.max_shard_bytes)
    }

    /// Returns up to `top_n` heaviest shards sorted deterministically by
    /// `bytes desc, functions desc, name asc`.
    pub fn heaviest_shards(&self, top_n: usize) -> Vec<&ShardEntry> {
        if top_n == 0 || self.shards.is_empty() {
            return Vec::new();
        }
        let mut refs: Vec<&ShardEntry> = self.shards.iter().collect();
        refs.sort_by(|a, b| {
            b.bytes
                .cmp(&a.bytes)
                .then_with(|| b.functions.cmp(&a.functions))
                .then_with(|| a.name.to_ascii_lowercase().cmp(&b.name.to_ascii_lowercase()))
                .then_with(|| a.name.cmp(&b.name))
        });
        refs.truncate(top_n);
        refs
    }

    /// Renders a sanitized Markdown Wasm64 shard & V8 headroom report against `module_cap_bytes`
    /// and optional largest function body bytes / max function locals.
    pub fn render_headroom_markdown(
        &self,
        module_cap_bytes: u64,
        largest_fn_bytes: Option<u64>,
        max_locals: Option<usize>,
    ) -> String {
        let module_headroom_bytes = module_cap_bytes.saturating_sub(self.total_bytes);
        let module_util_pct = percentage_u64_2dp(self.total_bytes, module_cap_bytes);
        let exceeds_module_cap = self.total_bytes > module_cap_bytes;
        let within_caps = !exceeds_module_cap
            && largest_fn_bytes.is_none_or(|b| b <= V8_MAX_WASM_FUNCTION_BODY_BYTES)
            && max_locals.is_none_or(|l| l <= V8_MAX_WASM_FUNCTION_LOCALS);
        let status = if within_caps {
            "within-caps"
        } else {
            "over-cap"
        };

        let mut out = String::with_capacity(512);
        let _ = writeln!(out, "### Wasm64 Shard & V8 Headroom");
        let _ = writeln!(out);
        let _ = writeln!(out, "| Metric | Value |");
        let _ = writeln!(out, "| --- | ---: |");
        let _ = writeln!(out, "| Shards | {} |", self.shard_count);
        let _ = writeln!(
            out,
            "| Total Bytes | {} ({:.2} MiB) |",
            self.total_bytes,
            self.total_bytes_mib()
        );
        let _ = writeln!(
            out,
            "| Module Cap | {} ({:.2} MiB) |",
            module_cap_bytes,
            mib_2dp(module_cap_bytes)
        );
        let _ = writeln!(
            out,
            "| Module Headroom | {} ({:.2} MiB) |",
            module_headroom_bytes,
            mib_2dp(module_headroom_bytes)
        );
        let _ = writeln!(out, "| Module Utilization | {:.2}% |", module_util_pct);
        let _ = writeln!(
            out,
            "| Max Shard Bytes | {} ({:.2} MiB) |",
            self.max_shard_bytes,
            self.max_shard_bytes_mib()
        );
        let _ = writeln!(
            out,
            "| Min/Max Function Ratio | {:.2} |",
            self.min_to_max_function_ratio()
        );
        let _ = writeln!(
            out,
            "| Min/Max Byte Ratio | {:.2} |",
            self.min_to_max_byte_ratio()
        );
        let _ = writeln!(out, "| Function Skew Ratio | {:.2}x |", self.skew_ratio);
        let _ = writeln!(out, "| Byte Skew Ratio | {:.2}x |", self.byte_skew_ratio);
        if let Some(fn_bytes) = largest_fn_bytes {
            let fn_headroom = V8_MAX_WASM_FUNCTION_BODY_BYTES.saturating_sub(fn_bytes);
            let fn_util = percentage_u64_2dp(fn_bytes, V8_MAX_WASM_FUNCTION_BODY_BYTES);
            let _ = writeln!(
                out,
                "| Largest Function Body | {} / {} bytes ({:.2}%, headroom {} bytes) |",
                fn_bytes, V8_MAX_WASM_FUNCTION_BODY_BYTES, fn_util, fn_headroom
            );
        }
        if let Some(locals) = max_locals {
            let locals_headroom = V8_MAX_WASM_FUNCTION_LOCALS.saturating_sub(locals);
            let locals_util = percentage_2dp(locals, V8_MAX_WASM_FUNCTION_LOCALS);
            let _ = writeln!(
                out,
                "| Max Function Locals | {} / {} ({:.2}%, headroom {}) |",
                locals, V8_MAX_WASM_FUNCTION_LOCALS, locals_util, locals_headroom
            );
        }
        let _ = writeln!(out, "| V8 Cap Status | {status} |");

        let heaviest = self.heaviest_shards(self.shards.len());
        if !heaviest.is_empty() {
            let _ = writeln!(out);
            let _ = writeln!(out, "| Heaviest Shard | Functions | Bytes | MiB |");
            let _ = writeln!(out, "| --- | ---: | ---: | ---: |");
            for shard in heaviest {
                let clean_name = sanitize_source_label(&shard.name);
                let _ = writeln!(
                    out,
                    "| `{}` | {} | {} | {:.2} |",
                    md_table_cell(&clean_name),
                    shard.functions,
                    shard.bytes,
                    mib_2dp(shard.bytes),
                );
            }
        }
        out
    }

    /// Renders a sanitized RFC 8259 JSON Wasm64 headroom report with strictly sorted top-level ASCII keys.
    pub fn render_headroom_json(
        &self,
        module_cap_bytes: u64,
        largest_fn_bytes: Option<u64>,
        max_locals: Option<usize>,
    ) -> String {
        let module_headroom_bytes = module_cap_bytes.saturating_sub(self.total_bytes);
        let module_util_pct = percentage_u64_2dp(self.total_bytes, module_cap_bytes);
        let exceeds_module_cap = self.total_bytes > module_cap_bytes;
        let within_caps = !exceeds_module_cap
            && largest_fn_bytes.is_none_or(|b| b <= V8_MAX_WASM_FUNCTION_BODY_BYTES)
            && max_locals.is_none_or(|l| l <= V8_MAX_WASM_FUNCTION_LOCALS);

        let fn_body_headroom = largest_fn_bytes
            .map(|b| V8_MAX_WASM_FUNCTION_BODY_BYTES.saturating_sub(b).to_string())
            .unwrap_or_else(|| "null".to_string());
        let fn_body_util = largest_fn_bytes
            .map(|b| format!("{:.2}", percentage_u64_2dp(b, V8_MAX_WASM_FUNCTION_BODY_BYTES)))
            .unwrap_or_else(|| "null".to_string());
        let largest_fn_json = largest_fn_bytes
            .map(|b| b.to_string())
            .unwrap_or_else(|| "null".to_string());

        let locals_headroom = max_locals
            .map(|l| V8_MAX_WASM_FUNCTION_LOCALS.saturating_sub(l).to_string())
            .unwrap_or_else(|| "null".to_string());
        let locals_util = max_locals
            .map(|l| format!("{:.2}", percentage_2dp(l, V8_MAX_WASM_FUNCTION_LOCALS)))
            .unwrap_or_else(|| "null".to_string());
        let max_locals_json = max_locals
            .map(|l| l.to_string())
            .unwrap_or_else(|| "null".to_string());

        let heaviest = self.heaviest_shards(self.shards.len());

        let mut out = String::with_capacity(1024);
        out.push_str("{\n");
        let _ = writeln!(out, "  \"byte_skew_ratio\": {:.2},", self.byte_skew_ratio);
        let _ = writeln!(out, "  \"exceeds_module_cap\": {exceeds_module_cap},");
        let _ = writeln!(
            out,
            "  \"function_body_cap_bytes\": {},",
            V8_MAX_WASM_FUNCTION_BODY_BYTES
        );
        let _ = writeln!(
            out,
            "  \"function_body_headroom_bytes\": {fn_body_headroom},"
        );
        let _ = writeln!(out, "  \"function_body_utilization_pct\": {fn_body_util},");
        let _ = writeln!(
            out,
            "  \"function_locals_cap\": {},",
            V8_MAX_WASM_FUNCTION_LOCALS
        );
        if heaviest.is_empty() {
            out.push_str("  \"heaviest_shards\": [],\n");
        } else {
            out.push_str("  \"heaviest_shards\": [\n");
            for (idx, shard) in heaviest.iter().enumerate() {
                let comma = if idx + 1 < heaviest.len() { "," } else { "" };
                let clean_name = sanitize_source_label(&shard.name);
                let _ = writeln!(
                    out,
                    "    {{\"bytes\": {}, \"bytes_mib\": {:.2}, \"functions\": {}, \"name\": {}}}{comma}",
                    shard.bytes,
                    mib_2dp(shard.bytes),
                    shard.functions,
                    json_str(&clean_name),
                );
            }
            out.push_str("  ],\n");
        }
        let _ = writeln!(out, "  \"largest_function_body_bytes\": {largest_fn_json},");
        let _ = writeln!(out, "  \"locals_headroom\": {locals_headroom},");
        let _ = writeln!(out, "  \"locals_utilization_pct\": {locals_util},");
        let _ = writeln!(out, "  \"max_function_locals\": {max_locals_json},");
        let _ = writeln!(out, "  \"max_shard_bytes\": {},", self.max_shard_bytes);
        let _ = writeln!(
            out,
            "  \"max_shard_bytes_mib\": {:.2},",
            self.max_shard_bytes_mib()
        );
        let _ = writeln!(
            out,
            "  \"min_to_max_byte_ratio\": {:.2},",
            self.min_to_max_byte_ratio()
        );
        let _ = writeln!(
            out,
            "  \"min_to_max_function_ratio\": {:.2},",
            self.min_to_max_function_ratio()
        );
        let _ = writeln!(out, "  \"module_cap_bytes\": {module_cap_bytes},");
        let _ = writeln!(out, "  \"module_cap_mib\": {:.2},", mib_2dp(module_cap_bytes));
        let _ = writeln!(out, "  \"module_headroom_bytes\": {module_headroom_bytes},");
        let _ = writeln!(
            out,
            "  \"module_headroom_mib\": {:.2},",
            mib_2dp(module_headroom_bytes)
        );
        let _ = writeln!(out, "  \"module_utilization_pct\": {:.2},", module_util_pct);
        let _ = writeln!(out, "  \"shard_count\": {},", self.shard_count);
        let _ = writeln!(out, "  \"skew_ratio\": {:.2},", self.skew_ratio);
        let _ = writeln!(out, "  \"total_bytes\": {},", self.total_bytes);
        let _ = writeln!(out, "  \"total_bytes_mib\": {:.2},", self.total_bytes_mib());
        let _ = writeln!(out, "  \"within_caps\": {within_caps}");
        out.push_str("}\n");
        out
    }
}

#[allow(dead_code)]
fn ratio_2dp_u128(numerator: u128, denominator: u128) -> f64 {
    if denominator == 0 {
        return 0.0;
    }
    if let Some(scaled) = numerator.checked_mul(200)
        && let Some(num) = scaled.checked_add(denominator)
        && let Some(den) = denominator.checked_mul(2)
    {
        return (num / den) as f64 / 100.0;
    }
    ((numerator as f64 / denominator as f64) * 100.0).round() / 100.0
}

#[allow(dead_code)]
fn ratio_2dp_product(a: u128, b: u128, denominator: u128) -> f64 {
    if denominator == 0 {
        return 0.0;
    }
    if let Some(prod) = a.checked_mul(b) {
        return ratio_2dp_u128(prod, denominator);
    }
    let ratio = (a as f64 / denominator as f64) * (b as f64);
    (ratio * 100.0).round() / 100.0
}

#[allow(dead_code)]
fn write_shard_balance_json(
    out: &mut String,
    summary: &ShardBalanceSummary,
    base_indent: usize,
    indent_opening: bool,
) {
    let i0 = " ".repeat(base_indent);
    let i1 = " ".repeat(base_indent + 2);
    let i2 = " ".repeat(base_indent + 4);

    if indent_opening {
        let _ = writeln!(out, "{i0}{{");
    } else {
        let _ = writeln!(out, "{{");
    }
    let _ = writeln!(out, "{i1}\"balanced\": {},", summary.balanced);
    let _ = writeln!(
        out,
        "{i1}\"byte_skew_ratio\": {:.2},",
        summary.byte_skew_ratio
    );
    let _ = writeln!(out, "{i1}\"max_shard_bytes\": {},", summary.max_shard_bytes);
    let _ = writeln!(
        out,
        "{i1}\"max_shard_functions\": {},",
        summary.max_shard_functions
    );
    let _ = writeln!(
        out,
        "{i1}\"max_skew_threshold\": {:.2},",
        summary.max_skew_threshold
    );
    let _ = writeln!(
        out,
        "{i1}\"mean_shard_bytes\": {:.2},",
        summary.mean_shard_bytes
    );
    let _ = writeln!(
        out,
        "{i1}\"mean_shard_functions\": {:.2},",
        summary.mean_shard_functions
    );
    let _ = writeln!(out, "{i1}\"min_shard_bytes\": {},", summary.min_shard_bytes);
    let _ = writeln!(
        out,
        "{i1}\"min_shard_functions\": {},",
        summary.min_shard_functions
    );
    let _ = writeln!(out, "{i1}\"shard_count\": {},", summary.shard_count);
    if summary.shards.is_empty() {
        let _ = writeln!(out, "{i1}\"shards\": [],");
    } else {
        let _ = writeln!(out, "{i1}\"shards\": [");
        for (idx, shard) in summary.shards.iter().enumerate() {
            let comma = if idx + 1 < summary.shards.len() {
                ","
            } else {
                ""
            };
            let _ = writeln!(
                out,
                "{i2}{{\"bytes\": {}, \"functions\": {}, \"name\": {}}}{comma}",
                shard.bytes,
                shard.functions,
                json_str(&shard.name)
            );
        }
        let _ = writeln!(out, "{i1}],");
    }
    let _ = writeln!(out, "{i1}\"skew_ratio\": {:.2},", summary.skew_ratio);
    let _ = writeln!(out, "{i1}\"total_bytes\": {},", summary.total_bytes);
    let _ = writeln!(out, "{i1}\"total_functions\": {}", summary.total_functions);
    let _ = write!(out, "{i0}}}");
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

/// Sanitized instruction-coverage, unresolved-import, and optional Wasm64 shard-balance breakdown
/// for CLI/program reports.
///
/// Holds counts, percentages, and sanitized labels only — never host paths or raw binary bytes.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub struct CoverageReport {
    pub source: String,
    pub summary: CoverageSummary,
    pub unresolved_by_dll: Vec<RankedCount>,
    pub unsupported_mnemonics: Vec<RankedCount>,
    pub shard_balance: Option<ShardBalanceSummary>,
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
            shard_balance: None,
        }
    }

    /// Attaches a sanitized, deterministically ordered unresolved-import breakdown by DLL.
    pub fn with_unresolved_imports<I, S>(mut self, entries: I) -> Self
    where
        I: IntoIterator<Item = (S, usize)>,
        S: AsRef<str>,
    {
        let sanitized = entries
            .into_iter()
            .map(|(label, count)| (sanitize_source_label(label.as_ref()), count));
        self.unresolved_by_dll = rank_top_n(sanitized, usize::MAX);
        self
    }

    /// Attaches a sanitized, deterministically ordered unsupported-mnemonic breakdown.
    pub fn with_unsupported_mnemonics<I, S>(mut self, entries: I) -> Self
    where
        I: IntoIterator<Item = (S, usize)>,
        S: AsRef<str>,
    {
        let sanitized = entries
            .into_iter()
            .map(|(label, count)| (sanitize_source_label(label.as_ref()), count));
        self.unsupported_mnemonics = rank_top_n(sanitized, usize::MAX);
        self
    }

    /// Attaches Wasm64 shard skew & fast-boot balance metrics to this coverage report.
    pub fn with_shard_balance(mut self, balance: ShardBalanceSummary) -> Self {
        self.shard_balance = Some(balance);
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
            let key = sanitize_source_label(dll).to_ascii_lowercase();
            let slot = bound_lookup.entry(key).or_insert(0);
            *slot = slot.saturating_add(count);
        }

        let mut totals_by_dll: BTreeMap<String, (String, usize)> = BTreeMap::new();
        for module in &pe.imports {
            let clean = sanitize_source_label(&module.dll_name);
            let key = clean.to_ascii_lowercase();
            let entry = totals_by_dll.entry(key).or_insert((clean, 0));
            entry.1 = entry.1.saturating_add(module.symbols.len());
        }
        for module in &pe.delay_imports {
            let clean = sanitize_source_label(&module.dll_name);
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

    /// Formats a crisp one-line progress ratio plus deterministic top-N unresolved-import,
    /// unsupported-mnemonic, and optional Wasm64 shard-balance breakdowns.
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
        if let Some(ref balance) = self.shard_balance {
            let _ = write!(out, " | {}", balance.render_one_line());
        }
        out
    }

    /// Renders a sanitized Markdown coverage summary with deterministic top-N breakdown tables
    /// and optional Wasm64 shard-balance diagnostics.
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
        if let Some(ref balance) = self.shard_balance {
            let _ = writeln!(out);
            out.push_str(&balance.render_markdown());
        }
        out
    }

    /// Renders a sanitized RFC 8259 JSON coverage summary with strictly sorted top-level keys
    /// and deterministic top-N breakdowns.
    pub fn render_coverage_json(&self, top_n: usize) -> String {
        let mut out = String::with_capacity(512);
        let _ = writeln!(out, "{{");
        let _ = writeln!(
            out,
            "  \"bound_import_pct\": {:.2},",
            self.bound_import_pct()
        );
        let _ = writeln!(out, "  \"bound_imports\": {},", self.summary.bound_imports);
        let _ = writeln!(
            out,
            "  \"decoded_instruction_pct\": {:.2},",
            self.decoded_instruction_pct()
        );
        let _ = writeln!(
            out,
            "  \"decoded_instructions\": {},",
            self.summary.decoded_instructions
        );
        let _ = writeln!(
            out,
            "  \"lifted_function_pct\": {:.2},",
            self.lifted_function_pct()
        );
        let _ = writeln!(
            out,
            "  \"lifted_functions\": {},",
            self.summary.lifted_functions
        );
        if let Some(ref balance) = self.shard_balance {
            let _ = write!(out, "  \"shard_balance\": ");
            write_shard_balance_json(&mut out, balance, 2, false);
            let _ = writeln!(out, ",");
        }
        let _ = writeln!(out, "  \"source\": {},", json_str(&self.source));
        let _ = writeln!(
            out,
            "  \"summary_line\": {},",
            json_str(&self.render_breakdown_line(top_n))
        );
        write_ranked_json(
            &mut out,
            "top_unresolved_imports",
            &self.unresolved_by_dll,
            top_n,
            true,
        );
        write_ranked_json(
            &mut out,
            "top_unsupported_mnemonics",
            &self.unsupported_mnemonics,
            top_n,
            true,
        );
        let _ = writeln!(
            out,
            "  \"total_functions\": {},",
            self.summary.total_functions
        );
        let _ = writeln!(out, "  \"total_imports\": {},", self.summary.total_imports);
        let _ = writeln!(
            out,
            "  \"total_instructions\": {},",
            self.summary.total_instructions
        );
        let _ = writeln!(
            out,
            "  \"unresolved_imports\": {}",
            self.summary.unresolved_imports()
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

#[allow(dead_code)]
fn write_indented_json_block(out: &mut String, block: &str, indent_spaces: usize) {
    let prefix = " ".repeat(indent_spaces);
    let trimmed = block.trim_end_matches(['\r', '\n']);
    for (idx, line) in trimmed.lines().enumerate() {
        if idx == 0 {
            out.push_str(line);
        } else {
            out.push('\n');
            out.push_str(&prefix);
            out.push_str(line);
        }
    }
}

/// Sanitized combined report for function-size distribution, optional Wasm64 V8 headroom,
/// and optional Wasm64 shard balance.
///
/// Outputs addresses, counts, ratios, byte sizes, and sanitized labels only — never host paths
/// or raw binary bytes.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionHeadroomReport {
    pub source: String,
    pub function_sizes: FunctionSizeStats,
    pub wasm64_headroom: Option<Wasm64HeadroomReport>,
    pub shard_balance: Option<ShardBalanceSummary>,
}

#[allow(dead_code)]
impl FunctionHeadroomReport {
    /// Creates a new sanitized report from a source label and [`FunctionSizeStats`].
    pub fn new(source_label: &str, function_sizes: FunctionSizeStats) -> Self {
        Self {
            source: sanitize_source_label(source_label),
            function_sizes,
            wasm64_headroom: None,
            shard_balance: None,
        }
    }

    /// Attaches a [`Wasm64HeadroomReport`] to this report.
    pub fn with_wasm64_headroom(mut self, headroom: Wasm64HeadroomReport) -> Self {
        self.wasm64_headroom = Some(headroom);
        self
    }

    /// Attaches a [`ShardBalanceSummary`] to this report.
    pub fn with_shard_balance(mut self, balance: ShardBalanceSummary) -> Self {
        self.shard_balance = Some(balance);
        self
    }

    /// Renders a crisp, deterministic, sanitized one-line summary.
    pub fn render_one_line(&self) -> String {
        let source = sanitize_source_label(&self.source);
        let mut out = format!("[{source}] {}", self.function_sizes.render_one_line());
        if let Some(ref headroom) = self.wasm64_headroom {
            let _ = write!(out, " | {}", headroom.render_one_line());
        }
        if let Some(ref balance) = self.shard_balance {
            let _ = write!(out, " | {}", balance.render_one_line());
        }
        out
    }

    /// Renders a sanitized, deterministically ordered Markdown report.
    pub fn render_markdown(&self) -> String {
        let source = sanitize_source_label(&self.source);
        let mut out = String::with_capacity(1024);
        let _ = writeln!(
            out,
            "## Function Size & Wasm64 Headroom Report — `{}`",
            md_inline(&source)
        );
        let _ = writeln!(out);
        let _ = writeln!(out, "- {}", md_inline(&self.render_one_line()));
        let _ = writeln!(out);

        // --- Function Size Distribution ---
        let fs = &self.function_sizes;
        let _ = writeln!(out, "### Function Size Distribution");
        let _ = writeln!(out);
        let _ = writeln!(out, "| Metric | Value |");
        let _ = writeln!(out, "| --- | ---: |");
        let _ = writeln!(out, "| Total Functions | {} |", fs.total_functions);
        let _ = writeln!(out, "| Total Blocks | {} |", fs.total_blocks);
        let _ = writeln!(out, "| Total Instructions | {} |", fs.total_instructions);
        let _ = writeln!(
            out,
            "| Min / Max Instructions | {} / {} |",
            fs.min_instructions, fs.max_instructions
        );
        let mean_insns = if fs.mean_instructions.is_finite() {
            fs.mean_instructions
        } else {
            0.0
        };
        let _ = writeln!(out, "| Mean Instructions | {:.2} |", mean_insns);
        let _ = writeln!(
            out,
            "| p50 / p95 / p99 Instructions | {} / {} / {} |",
            fs.p50_instructions, fs.p95_instructions, fs.p99_instructions
        );
        let _ = writeln!(
            out,
            "| Functions > 1k Instructions | {} |",
            fs.functions_over_1k_insns
        );
        let _ = writeln!(
            out,
            "| Functions > 10k Instructions | {} |",
            fs.functions_over_10k_insns
        );

        let mut sorted_top = fs.top_functions.clone();
        sorted_top.sort_by(|a, b| {
            b.instructions
                .cmp(&a.instructions)
                .then_with(|| b.blocks.cmp(&a.blocks))
                .then_with(|| a.entry.cmp(&b.entry))
                .then_with(|| b.indirect_calls.cmp(&a.indirect_calls))
                .then_with(|| b.diagnostics.cmp(&a.diagnostics))
        });
        if !sorted_top.is_empty() {
            let _ = writeln!(out);
            let _ = writeln!(
                out,
                "| Rank | Entry | Instructions | Blocks | Indirect Calls | Diagnostics |"
            );
            let _ = writeln!(out, "| ---: | --- | ---: | ---: | ---: | ---: |");
            for (idx, func) in sorted_top.iter().enumerate() {
                let _ = writeln!(
                    out,
                    "| {} | `0x{:x}` | {} | {} | {} | {} |",
                    idx + 1,
                    func.entry,
                    func.instructions,
                    func.blocks,
                    func.indirect_calls,
                    func.diagnostics,
                );
            }
        }

        // --- Optional Wasm64 V8 Headroom ---
        if let Some(ref headroom) = self.wasm64_headroom {
            let status = if headroom.within_v8_caps() {
                "within-caps"
            } else {
                "over-cap"
            };
            let _ = writeln!(out);
            let _ = writeln!(out, "### Wasm64 V8 Headroom");
            let _ = writeln!(out);
            let _ = writeln!(out, "| Metric | Value |");
            let _ = writeln!(out, "| --- | ---: |");
            let _ = writeln!(out, "| Side Modules | {} |", headroom.side_modules.len());
            let _ = writeln!(
                out,
                "| Total Module Bytes | {} ({:.2} MiB) |",
                headroom.total_module_bytes,
                mib_2dp(headroom.total_module_bytes)
            );
            let _ = writeln!(
                out,
                "| Largest Side Module | {} ({:.2} MiB, {:.2}% of {:.2} MiB cap) |",
                headroom.largest_module_bytes,
                mib_2dp(headroom.largest_module_bytes),
                headroom.module_utilization_pct(),
                mib_2dp(headroom.module_cap_bytes),
            );
            let _ = writeln!(
                out,
                "| Min Module Headroom | {} ({:.2} MiB) |",
                headroom.min_module_headroom_bytes,
                mib_2dp(headroom.min_module_headroom_bytes),
            );
            let _ = writeln!(
                out,
                "| Asymmetric Split Ratio | {:.2}x |",
                headroom.asymmetric_split_ratio()
            );
            let _ = writeln!(
                out,
                "| Largest Function Body | {} / {} bytes ({:.2}%, headroom {} bytes) |",
                headroom.largest_function_body_bytes,
                headroom.function_body_cap_bytes,
                headroom.function_body_utilization_pct(),
                headroom.function_body_headroom_bytes(),
            );
            let _ = writeln!(
                out,
                "| Functions > 1 MiB | {} |",
                headroom.functions_over_1mib
            );
            let _ = writeln!(
                out,
                "| Max Function Locals | {} / {} ({:.2}%, headroom {}) |",
                headroom.max_function_locals,
                headroom.function_locals_cap,
                headroom.locals_utilization_pct(),
                headroom.locals_headroom(),
            );
            let _ = writeln!(out, "| V8 Cap Status | {status} |");

            let mut modules_sorted = headroom.side_modules.clone();
            for m in &mut modules_sorted {
                m.label = sanitize_source_label(&m.label);
            }
            modules_sorted.sort_by(|a, b| {
                b.bytes
                    .cmp(&a.bytes)
                    .then_with(|| b.shards.cmp(&a.shards))
                    .then_with(|| a.label.to_ascii_lowercase().cmp(&b.label.to_ascii_lowercase()))
                    .then_with(|| a.label.cmp(&b.label))
            });
            if !modules_sorted.is_empty() {
                let _ = writeln!(out);
                let _ = writeln!(
                    out,
                    "| Side Module | Shards | Bytes | MiB | Cap Util | Headroom (MiB) | Status |"
                );
                let _ = writeln!(out, "| --- | ---: | ---: | ---: | ---: | ---: | --- |");
                for module in &modules_sorted {
                    let mod_status = if module.exceeds_cap(headroom.module_cap_bytes) {
                        "over-cap"
                    } else {
                        "ok"
                    };
                    let _ = writeln!(
                        out,
                        "| `{}` | {} | {} | {:.2} | {:.2}% | {:.2} | {mod_status} |",
                        md_table_cell(&module.label),
                        module.shards,
                        module.bytes,
                        module.bytes_mib(),
                        module.utilization_pct(headroom.module_cap_bytes),
                        mib_2dp(module.headroom_bytes(headroom.module_cap_bytes)),
                    );
                }
            }
        }

        // --- Optional Shard Balance ---
        if let Some(ref balance) = self.shard_balance {
            let sanitized_balance = ShardBalanceSummary::from_shards_with_threshold(
                balance.shards.iter().cloned(),
                balance.max_skew_threshold,
            );
            let _ = writeln!(out);
            out.push_str(&sanitized_balance.render_markdown());
        }

        out
    }

    /// Renders a sanitized RFC 8259 JSON report with strictly sorted top-level ASCII keys.
    pub fn render_json(&self) -> String {
        let source = sanitize_source_label(&self.source);
        let summary_line = self.render_one_line();
        let function_sizes_json = self.function_sizes.render_json();

        let mut out = String::with_capacity(1024);
        out.push_str("{\n");
        out.push_str("  \"function_sizes\": ");
        write_indented_json_block(&mut out, &function_sizes_json, 2);
        out.push_str(",\n");

        match &self.shard_balance {
            Some(balance) => {
                let sanitized_balance = ShardBalanceSummary::from_shards_with_threshold(
                    balance.shards.iter().cloned(),
                    balance.max_skew_threshold,
                );
                out.push_str("  \"shard_balance\": ");
                write_shard_balance_json(&mut out, &sanitized_balance, 2, false);
                out.push_str(",\n");
            }
            None => {
                out.push_str("  \"shard_balance\": null,\n");
            }
        }

        let _ = writeln!(out, "  \"source\": {},", json_str(&source));
        let _ = writeln!(out, "  \"summary_line\": {},", json_str(&summary_line));

        match &self.wasm64_headroom {
            Some(headroom) => {
                let headroom_json = headroom.render_json();
                out.push_str("  \"wasm64_headroom\": ");
                write_indented_json_block(&mut out, &headroom_json, 2);
                out.push('\n');
            }
            None => {
                out.push_str("  \"wasm64_headroom\": null\n");
            }
        }

        out.push_str("}\n");
        out
    }
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

    /// Extracts top-level JSON object keys in appearance order (for deterministic schema tests).
    fn extract_top_level_json_keys(json: &str) -> Vec<String> {
        let bytes = json.as_bytes();
        let mut keys = Vec::new();
        let mut depth = 0usize;
        let mut i = 0usize;
        let mut expect_key = false;
        while i < bytes.len() {
            match bytes[i] {
                b'{' => {
                    depth += 1;
                    if depth == 1 {
                        expect_key = true;
                    }
                    i += 1;
                }
                b'}' => {
                    depth = depth.saturating_sub(1);
                    expect_key = false;
                    i += 1;
                }
                b',' if depth == 1 => {
                    expect_key = true;
                    i += 1;
                }
                b'"' => {
                    let start = i + 1;
                    i += 1;
                    while i < bytes.len() {
                        if bytes[i] == b'\\' {
                            i += 2;
                        } else if bytes[i] == b'"' {
                            break;
                        } else {
                            i += 1;
                        }
                    }
                    let text = &json[start..i];
                    i += 1;
                    if depth == 1 && expect_key {
                        keys.push(text.to_string());
                        expect_key = false;
                    }
                }
                _ => {
                    i += 1;
                }
            }
        }
        keys
    }

    #[test]
    fn shard_balance_handles_empty_zero_and_single_shard_without_panicking() {
        let empty = ShardBalanceSummary::from_function_counts(&[]);
        assert_eq!(empty.shard_count, 0);
        assert_eq!(empty.total_functions, 0);
        assert_eq!(empty.min_shard_functions, 0);
        assert_eq!(empty.max_shard_functions, 0);
        assert_eq!(empty.mean_shard_functions, 0.0);
        assert_eq!(empty.skew_ratio, 0.0);
        assert_eq!(empty.byte_skew_ratio, 0.0);
        assert!(empty.balanced);
        let empty_json = empty.render_json();
        JsonValidator::validate(&empty_json).expect("empty shard balance JSON must be valid");
        assert!(empty_json.contains("\"shard_count\": 0"));
        assert!(empty_json.contains("\"shards\": []"));

        let zeros = ShardBalanceSummary::from_function_counts(&[0, 0, 0]);
        assert_eq!(zeros.shard_count, 3);
        assert_eq!(zeros.total_functions, 0);
        assert_eq!(zeros.min_shard_functions, 0);
        assert_eq!(zeros.max_shard_functions, 0);
        assert_eq!(zeros.skew_ratio, 0.0);
        assert!(zeros.balanced);
        JsonValidator::validate(&zeros.render_json()).expect("all-zero shard JSON must be valid");

        let single = ShardBalanceSummary::from_function_counts(&[42]);
        assert_eq!(single.shard_count, 1);
        assert_eq!(single.total_functions, 42);
        assert_eq!(single.min_shard_functions, 42);
        assert_eq!(single.max_shard_functions, 42);
        assert_eq!(single.mean_shard_functions, 42.0);
        assert_eq!(single.skew_ratio, 1.0);
        assert!(single.balanced);

        // Zero-min shard must never divide by zero.
        let mixed_zero = ShardBalanceSummary::from_function_counts(&[100, 0]);
        assert_eq!(mixed_zero.min_shard_functions, 0);
        assert_eq!(mixed_zero.max_shard_functions, 100);
        assert_eq!(mixed_zero.mean_shard_functions, 50.0);
        assert_eq!(mixed_zero.skew_ratio, 2.0);
        assert!(!mixed_zero.balanced);

        // Non-finite and sub-1.0 thresholds are safely normalized.
        let nan_thresh = ShardBalanceSummary::from_function_counts_with_threshold(&[10, 10], f64::NAN);
        assert_eq!(nan_thresh.max_skew_threshold, DEFAULT_MAX_SKEW_RATIO);
        assert!(nan_thresh.balanced);

        let sub_one = ShardBalanceSummary::from_function_counts_with_threshold(&[10, 10], 0.25);
        assert_eq!(sub_one.max_skew_threshold, 1.0);
        assert!(sub_one.balanced);
    }

    #[test]
    fn shard_balance_computes_function_and_byte_skew_and_pipeline_records() {
        let balanced = ShardBalanceSummary::from_function_counts(&[2048, 2048, 1536]);
        assert_eq!(balanced.shard_count, 3);
        assert_eq!(balanced.total_functions, 5632);
        assert_eq!(balanced.min_shard_functions, 1536);
        assert_eq!(balanced.max_shard_functions, 2048);
        assert_eq!(balanced.mean_shard_functions, 1877.33);
        assert_eq!(balanced.skew_ratio, 1.09);
        assert!(balanced.balanced);

        let skewed = ShardBalanceSummary::from_function_counts(&[2048, 2048, 100, 100]);
        assert_eq!(skewed.mean_shard_functions, 1074.0);
        assert_eq!(skewed.skew_ratio, 1.91);
        assert!(!skewed.balanced);

        // Exact threshold boundary at 1.50x
        let boundary_ok = ShardBalanceSummary::from_function_counts_with_threshold(&[30, 10], 1.50);
        assert_eq!(boundary_ok.skew_ratio, 1.5);
        assert!(boundary_ok.balanced);

        let boundary_fail = ShardBalanceSummary::from_function_counts_with_threshold(&[30, 10], 1.49);
        assert_eq!(boundary_fail.skew_ratio, 1.5);
        assert!(!boundary_fail.balanced);

        // Uniform function count with skewed byte sizes flips balanced to false.
        let byte_skewed = ShardBalanceSummary::from_shards(vec![
            ShardEntry::new("shard-00000.wasm", 100, 1_000_000),
            ShardEntry::new("shard-00001.wasm", 100, 10_000),
        ]);
        assert_eq!(byte_skewed.skew_ratio, 1.0);
        assert_eq!(byte_skewed.total_bytes, 1_010_000);
        assert_eq!(byte_skewed.min_shard_bytes, 10_000);
        assert_eq!(byte_skewed.max_shard_bytes, 1_000_000);
        assert_eq!(byte_skewed.byte_skew_ratio, 1.98);
        assert!(!byte_skewed.balanced);
        assert!(byte_skewed.render_one_line().contains("byte_skew 1.98x, skewed"));

        // Direct construction from pipeline ShardRecord prefers obj.size when present.
        use playgta6_recomp::pipeline::OutputFile;
        let records = vec![
            ShardRecord {
                index: 1,
                first: 0x2000,
                last: 0x2FFF,
                functions: 120,
                instructions: 3000,
                unsupported: BTreeMap::new(),
                ll: OutputFile {
                    path: "lift/shard-00001.ll".to_string(),
                    size: 90_000,
                    hash: 1,
                },
                obj: Some(OutputFile {
                    path: "lift/shard-00001.o".to_string(),
                    size: 40_000,
                    hash: 2,
                }),
            },
            ShardRecord {
                index: 0,
                first: 0x1000,
                last: 0x1FFF,
                functions: 100,
                instructions: 2500,
                unsupported: BTreeMap::new(),
                ll: OutputFile {
                    path: "lift/shard-00000.ll".to_string(),
                    size: 50_000,
                    hash: 3,
                },
                obj: None,
            },
        ];
        let from_rec = ShardBalanceSummary::from_shard_records(&records);
        assert_eq!(from_rec.shard_count, 2);
        assert_eq!(from_rec.total_functions, 220);
        assert_eq!(from_rec.total_bytes, 90_000);
        assert_eq!(from_rec.shards[0].name, "shard-00000.ll");
        assert_eq!(from_rec.shards[1].name, "shard-00001.o");
        assert!(from_rec.balanced);
    }

    #[test]
    fn deterministic_json_and_text_ordering_independent_of_hashmap_or_input_order() {
        use std::collections::HashMap;

        let mut shard_map = HashMap::new();
        shard_map.insert("shard-00002.wasm", (512usize, 120_000u64));
        shard_map.insert("shard-00000.wasm", (2048usize, 480_000u64));
        shard_map.insert("shard-00001.wasm", (2048usize, 460_000u64));

        let balance_from_map = ShardBalanceSummary::from_shards(
            shard_map
                .iter()
                .map(|(name, &(fns, bytes))| ShardEntry::new(name, fns, bytes)),
        );
        let balance_from_rev = ShardBalanceSummary::from_shards(vec![
            ShardEntry::new("shard-00001.wasm", 2048, 460_000),
            ShardEntry::new("shard-00002.wasm", 512, 120_000),
            ShardEntry::new("shard-00000.wasm", 2048, 480_000),
        ]);

        assert_eq!(balance_from_map, balance_from_rev);
        assert_eq!(
            balance_from_map.render_one_line(),
            balance_from_rev.render_one_line()
        );
        assert_eq!(
            balance_from_map.render_markdown(),
            balance_from_rev.render_markdown()
        );
        let shard_json_a = balance_from_map.render_json();
        let shard_json_b = balance_from_rev.render_json();
        assert_eq!(shard_json_a, shard_json_b);
        JsonValidator::validate(&shard_json_a).expect("shard balance JSON must be valid");

        // Verify shard balance top-level JSON keys are strictly sorted in ascending ASCII order.
        let shard_keys = extract_top_level_json_keys(&shard_json_a);
        assert!(shard_keys.len() >= 10);
        for pair in shard_keys.windows(2) {
            assert!(
                pair[0] < pair[1],
                "shard_balance JSON keys must be strictly sorted: {:?} >= {:?}",
                pair[0],
                pair[1]
            );
        }

        // Verify CoverageReport with attached shard balance is order-independent and key-sorted.
        let mut unresolved_map = HashMap::new();
        unresolved_map.insert("d3d12_xs.dll", 4usize);
        unresolved_map.insert("bink2scarlett.dll", 4usize);
        unresolved_map.insert("ws2_32.dll", 1usize);

        let mut unsupported_map = HashMap::new();
        unsupported_map.insert("vmovdqu", 9usize);
        unsupported_map.insert("aesenc", 9usize);
        unsupported_map.insert("cpuid", 2usize);

        let summary = CoverageSummary {
            total_functions: 4608,
            lifted_functions: 4600,
            total_instructions: 100_000,
            decoded_instructions: 99_980,
            total_imports: 20,
            bound_imports: 11,
        };
        let rep_a = CoverageReport::new("game.exe", summary)
            .with_unresolved_imports(unresolved_map.iter().map(|(&k, &v)| (k, v)))
            .with_unsupported_mnemonics(unsupported_map.iter().map(|(&k, &v)| (k, v)))
            .with_shard_balance(balance_from_map);
        let rep_b = CoverageReport::new("game.exe", summary)
            .with_unresolved_imports(vec![
                ("ws2_32.dll", 1),
                ("d3d12_xs.dll", 4),
                ("bink2scarlett.dll", 4),
            ])
            .with_unsupported_mnemonics(vec![("cpuid", 2), ("vmovdqu", 9), ("aesenc", 9)])
            .with_shard_balance(balance_from_rev);

        assert_eq!(rep_a, rep_b);
        assert_eq!(rep_a.render_breakdown_line(5), rep_b.render_breakdown_line(5));
        assert_eq!(
            rep_a.render_coverage_markdown(5),
            rep_b.render_coverage_markdown(5)
        );
        let cov_json_a = rep_a.render_coverage_json(5);
        let cov_json_b = rep_b.render_coverage_json(5);
        assert_eq!(cov_json_a, cov_json_b);
        JsonValidator::validate(&cov_json_a).expect("coverage+shard JSON must be valid");

        let cov_keys = extract_top_level_json_keys(&cov_json_a);
        assert!(cov_keys.contains(&"shard_balance".to_string()));
        for pair in cov_keys.windows(2) {
            assert!(
                pair[0] < pair[1],
                "coverage JSON keys must be strictly sorted: {:?} >= {:?}",
                pair[0],
                pair[1]
            );
        }
    }

    #[test]
    fn sanitization_blocks_windows_and_unix_host_path_leaks_in_all_reports() {
        // Direct label sanitization edge cases (Windows, UNC, Unix, URIs, directory trails, traversal)
        assert_eq!(
            sanitize_source_label(r"C:\Users\alice\Games\GTA6\game.exe"),
            "game.exe"
        );
        assert_eq!(
            sanitize_source_label(r"\\?\C:\Private\Builds\shard-00001.wasm"),
            "shard-00001.wasm"
        );
        assert_eq!(
            sanitize_source_label(r"\\fileserver\share\secret\d3d12_xs.dll"),
            "d3d12_xs.dll"
        );
        assert_eq!(sanitize_source_label(r"C:\Users\alice\"), "<unknown>");
        assert_eq!(
            sanitize_source_label("/home/runner/work/playgta6web/local-data/game.exe"),
            "game.exe"
        );
        assert_eq!(
            sanitize_source_label("/Users/alice/Documents/shard-00000.ll"),
            "shard-00000.ll"
        );
        assert_eq!(sanitize_source_label("/home/alice/"), "<unknown>");
        assert_eq!(
            sanitize_source_label("file:///C:/Users/alice/game.exe"),
            "game.exe"
        );
        assert_eq!(sanitize_source_label("../../secret/game.exe"), "game.exe");
        assert_eq!(sanitize_source_label("../.."), "<unknown>");
        assert_eq!(sanitize_source_label("."), "<unknown>");

        // Host path leak detector positive & negative cases
        assert!(contains_host_path(r"C:\Users\alice\game.exe"));
        assert!(contains_host_path("\"source\": \"C:\\\\Users\\\\alice\\\\game.exe\""));
        assert!(contains_host_path("D:/local-data/game.exe"));
        assert!(contains_host_path(r"\\?\C:\secret\file.wasm"));
        assert!(contains_host_path(r"\\fileserver\share\file.wasm"));
        assert!(contains_host_path("\"path\": \"\\\\\\\\fileserver\\\\share\\\\file.wasm\""));
        assert!(contains_host_path("/home/runner/work/game.exe"));
        assert!(contains_host_path("\"source\": \"/Users/alice/game.exe\""));
        assert!(contains_host_path("file:///tmp/game.exe"));
        assert!(!contains_host_path("{\"source\": \"game.exe\", \"shard\": \"shard-00000.wasm\"}"));
        assert!(!contains_host_path("\"source\": \"Synthetic\\\"Game\\\\v1.exe\""));

        // Full serialized report audit: even raw struct-literal ShardEntry paths must be sanitized.
        let summary = CoverageSummary {
            total_functions: 200,
            lifted_functions: 195,
            total_instructions: 5000,
            decoded_instructions: 4990,
            total_imports: 10,
            bound_imports: 8,
        };
        let shards = ShardBalanceSummary::from_shards(vec![
            ShardEntry {
                name: r"C:\Users\alice\Private\work\lift\shard-00000.wasm".to_string(),
                functions: 100,
                bytes: 200_000,
            },
            ShardEntry {
                name: "/home/runner/secret_build/lift/shard-00001.wasm".to_string(),
                functions: 100,
                bytes: 210_000,
            },
            ShardEntry {
                name: r"\\fileserver\share\hidden\shard-00002.wasm".to_string(),
                functions: 0,
                bytes: 0,
            },
        ]);
        let report = CoverageReport::new(r"C:\Users\alice\Games\Title\game.exe", summary)
            .with_unresolved_imports(vec![
                (r"C:\XboxSDK\Private\d3d12_xs.dll", 2),
                ("/Users/alice/libs/bink2.dll", 1),
            ])
            .with_unsupported_mnemonics(vec![("/tmp/secret_dir/vmovdqu", 3)])
            .with_shard_balance(shards);

        let one_line = report.render_breakdown_line(5);
        let markdown = report.render_coverage_markdown(5);
        let json = report.render_coverage_json(5);
        JsonValidator::validate(&json).expect("sanitized report JSON must be valid");

        for output in [&one_line, &markdown, &json] {
            assert!(
                !contains_host_path(output),
                "host path leaked into output: {output}"
            );
            for forbidden in [
                "Users",
                "alice",
                "Private",
                "runner",
                "secret_build",
                "fileserver",
                "hidden",
                "XboxSDK",
                "secret_dir",
                r"C:\",
                "/home/",
                "/tmp/",
            ] {
                assert!(
                    !output.contains(forbidden),
                    "forbidden host path segment `{forbidden}` leaked into output: {output}"
                );
            }
        }
        assert!(json.contains("\"source\": \"game.exe\""));
        assert!(json.contains("\"name\": \"shard-00000.wasm\""));
        assert!(json.contains("\"name\": \"shard-00001.wasm\""));
        assert!(json.contains("\"name\": \"shard-00002.wasm\""));
    }

    #[test]
    fn shard_balance_headroom_helpers_and_json_validation() {
        use playgta6_recomp::summary::{FunctionSizeEntry, SideModuleEntry};

        // Empty shard balance: zero-division guards
        let empty = ShardBalanceSummary::from_function_counts(&[]);
        assert_eq!(empty.total_bytes_mib(), 0.0);
        assert_eq!(empty.max_shard_bytes_mib(), 0.0);
        assert_eq!(empty.v8_module_headroom_bytes(), V8_MAX_WASM_MODULE_BYTES);
        assert_eq!(empty.v8_module_utilization_pct(), 0.0);
        assert!(!empty.exceeds_v8_module_cap());
        assert_eq!(empty.min_to_max_function_ratio(), 0.0);
        assert_eq!(empty.min_to_max_byte_ratio(), 0.0);
        assert!(empty.heaviest_shards(5).is_empty());

        let empty_headroom_json =
            empty.render_headroom_json(V8_MAX_WASM_MODULE_BYTES, None, None);
        JsonValidator::validate(&empty_headroom_json)
            .expect("empty headroom JSON must be valid RFC 8259 JSON");
        let empty_keys = extract_top_level_json_keys(&empty_headroom_json);
        assert!(empty_keys.len() >= 20);
        for pair in empty_keys.windows(2) {
            assert!(
                pair[0] < pair[1],
                "render_headroom_json keys must be strictly sorted: {:?} >= {:?}",
                pair[0],
                pair[1]
            );
        }
        assert!(empty_headroom_json.contains("\"largest_function_body_bytes\": null,"));
        assert!(empty_headroom_json.contains("\"max_function_locals\": null,"));
        assert!(empty_headroom_json.contains("\"within_caps\": true"));

        // Non-empty shards with deterministic heaviest_shards tie-breaking
        let balance = ShardBalanceSummary::from_shards(vec![
            ShardEntry::new("shard-00002.wasm", 100, 2 * 1024 * 1024),
            ShardEntry::new("shard-00000.wasm", 200, 4 * 1024 * 1024),
            ShardEntry::new("shard-00001.wasm", 150, 2 * 1024 * 1024),
        ]);
        assert_eq!(balance.total_bytes_mib(), 8.0);
        assert_eq!(balance.max_shard_bytes_mib(), 4.0);
        assert_eq!(
            balance.v8_module_headroom_bytes(),
            V8_MAX_WASM_MODULE_BYTES - 8 * 1024 * 1024
        );
        assert_eq!(balance.v8_module_utilization_pct(), 0.78);
        assert!(!balance.exceeds_v8_module_cap());
        assert_eq!(balance.min_to_max_function_ratio(), 0.5);
        assert_eq!(balance.min_to_max_byte_ratio(), 0.5);
        assert!(balance.heaviest_shards(0).is_empty());
        let top2 = balance.heaviest_shards(2);
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0].name, "shard-00000.wasm");
        assert_eq!(top2[1].name, "shard-00001.wasm"); // tied bytes -> higher functions wins

        let headroom_md = balance.render_headroom_markdown(
            V8_MAX_WASM_MODULE_BYTES,
            Some(1_700_000),
            Some(1_306),
        );
        assert!(headroom_md.contains("### Wasm64 Shard & V8 Headroom"));
        assert!(headroom_md.contains("| V8 Cap Status | within-caps |"));
        assert!(headroom_md.contains("| `shard-00000.wasm` | 200 |"));

        let headroom_json = balance.render_headroom_json(
            V8_MAX_WASM_MODULE_BYTES,
            Some(1_700_000),
            Some(1_306),
        );
        JsonValidator::validate(&headroom_json)
            .expect("shard headroom JSON must be valid RFC 8259 JSON");
        let headroom_keys = extract_top_level_json_keys(&headroom_json);
        for pair in headroom_keys.windows(2) {
            assert!(
                pair[0] < pair[1],
                "render_headroom_json keys must be strictly sorted: {:?} >= {:?}",
                pair[0],
                pair[1]
            );
        }
        assert!(headroom_json.contains("\"largest_function_body_bytes\": 1700000,"));
        assert!(headroom_json.contains("\"max_function_locals\": 1306,"));
        assert!(headroom_json.contains("\"within_caps\": true"));

        // Over-V8-cap shard balance
        let over_cap = ShardBalanceSummary::from_shards(vec![
            ShardEntry::new("huge-0.wasm", 10_000, 600 * 1024 * 1024),
            ShardEntry::new("huge-1.wasm", 10_000, 500 * 1024 * 1024),
        ]);
        assert!(over_cap.exceeds_v8_module_cap());
        assert_eq!(over_cap.v8_module_headroom_bytes(), 0);
        assert_eq!(over_cap.v8_module_utilization_pct(), 100.0);
        let over_json = over_cap.render_headroom_json(V8_MAX_WASM_MODULE_BYTES, None, None);
        JsonValidator::validate(&over_json).expect("over-cap headroom JSON must be valid");
        assert!(over_json.contains("\"exceeds_module_cap\": true,"));
        assert!(over_json.contains("\"within_caps\": false"));

        // Validate summary.rs JSON renderers (FunctionSizeStats & Wasm64HeadroomReport) with JsonValidator
        let fn_stats = FunctionSizeStats::from_entries(
            vec![
                FunctionSizeEntry {
                    entry: 0x0001_4000_1000,
                    blocks: 50,
                    instructions: 2_500,
                    indirect_calls: 4,
                    diagnostics: 0,
                },
                FunctionSizeEntry {
                    entry: 0x0001_4000_9000,
                    blocks: 4,
                    instructions: 30,
                    indirect_calls: 0,
                    diagnostics: 1,
                },
            ],
            5,
        );
        let fn_json = fn_stats.render_json();
        JsonValidator::validate(&fn_json).expect("FunctionSizeStats JSON must be valid RFC 8259");
        let fn_keys = extract_top_level_json_keys(&fn_json);
        for pair in fn_keys.windows(2) {
            assert!(
                pair[0] < pair[1],
                "FunctionSizeStats JSON keys must be strictly sorted: {:?} >= {:?}",
                pair[0],
                pair[1]
            );
        }

        let wasm_headroom = Wasm64HeadroomReport::new(
            vec![
                SideModuleEntry::new("guest_part_0.wasm", 383, 890 * 1024 * 1024),
                SideModuleEntry::new("guest_part_2.wasm", 1, 7_025_459),
            ],
            1_700_000,
            1_306,
            4,
        );
        let wasm_json = wasm_headroom.render_json();
        JsonValidator::validate(&wasm_json)
            .expect("Wasm64HeadroomReport JSON must be valid RFC 8259");
        let wasm_keys = extract_top_level_json_keys(&wasm_json);
        for pair in wasm_keys.windows(2) {
            assert!(
                pair[0] < pair[1],
                "Wasm64HeadroomReport JSON keys must be strictly sorted: {:?} >= {:?}",
                pair[0],
                pair[1]
            );
        }
    }

    #[test]
    fn function_headroom_report_order_independence_and_json_key_sorting() {
        use playgta6_recomp::summary::{FunctionSizeEntry, SideModuleEntry};

        let entries_a = vec![
            FunctionSizeEntry {
                entry: 0x0001_4000_4000,
                blocks: 10,
                instructions: 500,
                indirect_calls: 1,
                diagnostics: 0,
            },
            FunctionSizeEntry {
                entry: 0x0001_4000_2000,
                blocks: 45,
                instructions: 12_000,
                indirect_calls: 6,
                diagnostics: 0,
            },
            FunctionSizeEntry {
                entry: 0x0001_4000_1000,
                blocks: 45,
                instructions: 12_000,
                indirect_calls: 2,
                diagnostics: 1,
            },
        ];
        let mut entries_b = entries_a.clone();
        entries_b.reverse();

        let modules_a = vec![
            SideModuleEntry::new("guest_part_1.wasm", 1, 146_801),
            SideModuleEntry::new("guest_part_0.wasm", 383, 890 * 1024 * 1024),
            SideModuleEntry::new("guest_part_2.wasm", 1, 7_025_459),
        ];
        let mut modules_b = modules_a.clone();
        modules_b.reverse();

        let shards_a = vec![
            ShardEntry::new("shard-00001.wasm", 100, 200_000),
            ShardEntry::new("shard-00000.wasm", 120, 240_000),
        ];
        let mut shards_b = shards_a.clone();
        shards_b.reverse();

        let rep_a = FunctionHeadroomReport::new(
            "game.exe",
            FunctionSizeStats::from_entries(entries_a, 5),
        )
        .with_wasm64_headroom(Wasm64HeadroomReport::new(modules_a, 1_700_000, 1_306, 4))
        .with_shard_balance(ShardBalanceSummary::from_shards(shards_a));

        let rep_b = FunctionHeadroomReport::new(
            "game.exe",
            FunctionSizeStats::from_entries(entries_b, 5),
        )
        .with_wasm64_headroom(Wasm64HeadroomReport::new(modules_b, 1_700_000, 1_306, 4))
        .with_shard_balance(ShardBalanceSummary::from_shards(shards_b));

        assert_eq!(rep_a, rep_b);
        assert_eq!(rep_a.render_one_line(), rep_b.render_one_line());
        assert_eq!(rep_a.render_markdown(), rep_b.render_markdown());
        let json_a = rep_a.render_json();
        let json_b = rep_b.render_json();
        assert_eq!(json_a, json_b);
        JsonValidator::validate(&json_a)
            .expect("FunctionHeadroomReport JSON must be valid RFC 8259");

        let keys = extract_top_level_json_keys(&json_a);
        assert_eq!(
            keys,
            vec![
                "function_sizes",
                "shard_balance",
                "source",
                "summary_line",
                "wasm64_headroom",
            ]
        );
        for pair in keys.windows(2) {
            assert!(pair[0] < pair[1]);
        }

        // Minimal report (no optional sections)
        let minimal = FunctionHeadroomReport::new("minimal.exe", FunctionSizeStats::default());
        let min_json = minimal.render_json();
        JsonValidator::validate(&min_json).expect("minimal FunctionHeadroomReport JSON must be valid");
        let min_keys = extract_top_level_json_keys(&min_json);
        assert_eq!(min_keys, keys);
        assert!(min_json.contains("\"shard_balance\": null,"));
        assert!(min_json.contains("\"wasm64_headroom\": null\n}"));
    }

    #[test]
    fn function_headroom_and_shard_headroom_block_adversarial_host_paths() {
        use playgta6_recomp::summary::{FunctionSizeEntry, SideModuleEntry};

        let mut balance = ShardBalanceSummary::from_shards(vec![
            ShardEntry::new(r"C:\Users\alice\Private\shard-00000.wasm", 100, 200_000),
            ShardEntry::new("/home/runner/work/secret/shard-00001.wasm", 120, 240_000),
            ShardEntry::new(r"\\fileserver\share\hidden\shard-00002.wasm", 80, 160_000),
            ShardEntry::new("file:///Users/bob/Desktop/", 10, 20_000),
        ]);
        // Defense-in-depth: mutate shard name after construction
        balance.shards[0].name = r"\\?\C:\SecretAdmin\shard-00000.wasm".to_string();

        let headroom_md = balance.render_headroom_markdown(
            V8_MAX_WASM_MODULE_BYTES,
            Some(1_700_000),
            Some(1_306),
        );
        let headroom_json = balance.render_headroom_json(
            V8_MAX_WASM_MODULE_BYTES,
            Some(1_700_000),
            Some(1_306),
        );
        JsonValidator::validate(&headroom_json).expect("sanitized shard headroom JSON must be valid");

        let mut full = FunctionHeadroomReport::new(
            r"C:\Users\alice\Games\GTA6\game.exe",
            FunctionSizeStats::from_entries(
                vec![FunctionSizeEntry {
                    entry: 0x0001_4000_1000,
                    blocks: 5,
                    instructions: 50,
                    indirect_calls: 1,
                    diagnostics: 0,
                }],
                5,
            ),
        )
        .with_wasm64_headroom(Wasm64HeadroomReport::new(
            vec![
                SideModuleEntry::new(
                    r"\\buildbox\share\artifacts\guest_part_0.wasm",
                    383,
                    890 * 1024 * 1024,
                ),
                SideModuleEntry::new(
                    "file:///tmp/private_build/guest_part_1.wasm",
                    1,
                    146_801,
                ),
            ],
            1_700_000,
            1_306,
            4,
        ))
        .with_shard_balance(balance);
        full.source = "/Users/bob/Private/game.exe".to_string();

        let one_line = full.render_one_line();
        let md = full.render_markdown();
        let json = full.render_json();
        JsonValidator::validate(&json).expect("sanitized FunctionHeadroomReport JSON must be valid");

        for output in [&headroom_md, &headroom_json, &one_line, &md, &json] {
            assert!(
                !contains_host_path(output),
                "host path leaked into output:\n{output}"
            );
            for forbidden in [
                "alice",
                "bob",
                "runner",
                "Private",
                "SecretAdmin",
                "fileserver",
                "hidden",
                "buildbox",
                "private_build",
                "file://",
                r"C:\",
                "/home/",
                "/Users/",
                "/tmp/",
            ] {
                assert!(
                    !output.contains(forbidden),
                    "forbidden host path segment `{forbidden}` leaked into output:\n{output}"
                );
            }
        }
        assert!(json.contains("\"source\": \"game.exe\","));
        assert!(json.contains("\"label\": \"guest_part_0.wasm\""));
        assert!(json.contains("\"name\": \"shard-00000.wasm\""));
    }
}
