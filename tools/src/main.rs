use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const ABI_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
struct FileEntry {
    rel_path: String,
    size: u64,
}

fn collect_files(root: &Path, dir: &Path, out: &mut Vec<FileEntry>) -> io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let meta = entry.metadata()?;
        if meta.is_dir() {
            collect_files(root, &path, out)?;
        } else if meta.is_file() {
            let rel = path
                .strip_prefix(root)
                .map_err(io::Error::other)?
                .to_string_lossy()
                .replace('\\', "/");
            out.push(FileEntry {
                rel_path: rel,
                size: meta.len(),
            });
        }
    }
    Ok(())
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn render_manifest_json(entries: &[FileEntry]) -> String {
    let total_bytes: u64 = entries.iter().map(|e| e.size).sum();
    let mut out = String::new();
    out.push_str("{\n");
    out.push_str(&format!("  \"abi_version\": {ABI_VERSION},\n"));
    out.push_str(&format!("  \"file_count\": {},\n", entries.len()));
    out.push_str(&format!("  \"total_bytes\": {},\n", total_bytes));
    out.push_str("  \"files\": [\n");
    for (idx, e) in entries.iter().enumerate() {
        let comma = if idx + 1 == entries.len() { "" } else { "," };
        out.push_str(&format!(
            "    {{\"path\": \"{}\", \"size\": {}}}{}\n",
            json_escape(&e.rel_path),
            e.size,
            comma
        ));
    }
    out.push_str("  ]\n");
    out.push_str("}\n");
    out
}

fn build_manifest(game_dir: &Path, out_file: &Path) -> io::Result<usize> {
    let mut entries = Vec::new();
    collect_files(game_dir, game_dir, &mut entries)?;
    entries.sort_by(|a, b| a.rel_path.cmp(&b.rel_path));
    if let Some(parent) = out_file.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    let json = render_manifest_json(&entries);
    fs::write(out_file, json)?;
    Ok(entries.len())
}

fn read_dotenv_key(key: &str) -> Option<String> {
    if let Ok(val) = env::var(key)
        && !val.trim().is_empty()
    {
        return Some(val);
    }
    let text = fs::read_to_string(".env").ok()?;
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((k, v)) = line.split_once('=')
            && k.trim() == key
        {
            let cleaned = v.trim().trim_matches('"').trim_matches('\'');
            if !cleaned.is_empty() {
                return Some(cleaned.to_string());
            }
        }
    }
    None
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        eprintln!("Usage: playgta6-manifest [--game <game-dir>] [--out <local-data/data-manifest.json>]");
        eprintln!("Falls back to GTA6_GAME_DIR and GTA6_DATA_DIR from env or .env.");
        return ExitCode::SUCCESS;
    }

    let mut game_dir: Option<PathBuf> = None;
    let mut out_file: Option<PathBuf> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--game" if i + 1 < args.len() => {
                game_dir = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--out" if i + 1 < args.len() => {
                out_file = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            other => {
                eprintln!("Unknown argument: {other}");
                return ExitCode::FAILURE;
            }
        }
    }

    let Some(game_dir) = game_dir.or_else(|| read_dotenv_key("GTA6_GAME_DIR").map(PathBuf::from))
    else {
        eprintln!("Error: --game <dir> or GTA6_GAME_DIR (in .env) is required.");
        return ExitCode::FAILURE;
    };

    let out_file = out_file.unwrap_or_else(|| {
        let data_dir = read_dotenv_key("GTA6_DATA_DIR").unwrap_or_else(|| "local-data".to_string());
        PathBuf::from(data_dir).join("data-manifest.json")
    });

    match build_manifest(&game_dir, &out_file) {
        Ok(count) => {
            println!("Wrote {} entries to {}", count, out_file.display());
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("Manifest generation failed: {err}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_sorted_manifest_json() {
        let base = env::temp_dir().join(format!("playgta6_manifest_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        let game = base.join("game");
        let sub = game.join("sub");
        fs::create_dir_all(&sub).unwrap();
        fs::write(game.join("b.bin"), b"12345").unwrap();
        fs::write(sub.join("a.bin"), b"xy").unwrap();

        let out_file = base.join("out/data-manifest.json");
        let count = build_manifest(&game, &out_file).unwrap();
        assert_eq!(count, 2);

        let content = fs::read_to_string(&out_file).unwrap();
        assert!(content.contains("\"abi_version\": 1"));
        assert!(content.contains("\"file_count\": 2"));
        assert!(content.contains("\"total_bytes\": 7"));
        assert!(content.contains("\"path\": \"b.bin\""));
        assert!(content.contains("\"path\": \"sub/a.bin\""));

        let _ = fs::remove_dir_all(&base);
    }
}
