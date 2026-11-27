// Shader quarantine parsing and drop-table accounting (gpu/native_exec/shader_quarantine.h).
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "gpu/drop_table.h"
#include "gpu/native_exec/shader_quarantine.h"

namespace {
int g_failures = 0;
void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_native_exec_quarantine.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}
#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)
using playgta6::gpu::d3d12_guest::DropTable;
using playgta6::gpu::native_exec::ShaderQuarantine;
bool has(const std::string& s, const char* part) { return s.find(part) != std::string::npos; }
std::string slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
size_t count(const std::string& s, const std::string& part) {
    size_t n = 0;
    for (size_t at = s.find(part); at != std::string::npos; at = s.find(part, at + 1)) ++n;
    return n;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc >= 4 && std::string(argv[1]) == "--persist-child") {  // child of the kill test below
        std::vector<uint64_t> f;
        for (int i = 3; i < argc; ++i) f.push_back(std::strtoull(argv[i], nullptr, 16));
        ShaderQuarantine::persist_auto_file(argv[2], f);
        std::_Exit(0);  // no destructors, no atexit flush: only what persist_auto_file flushed survives
    }
    uint64_t h = 0;
    CHECK(ShaderQuarantine::parse_hash("0xDEADbeef", &h) && h == 0xdeadbeefull);
    CHECK(ShaderQuarantine::parse_hash("ff", &h) && h == 0xff);
    CHECK(!ShaderQuarantine::parse_hash("", &h));
    CHECK(!ShaderQuarantine::parse_hash("0x", &h));
    CHECK(!ShaderQuarantine::parse_hash("xyz", &h));
    CHECK(!ShaderQuarantine::parse_hash("11111111111111111", &h));  // 17 digits

    ShaderQuarantine q;
    std::string err;
    CHECK(q.add_list("0x1, abc,,0x00000000000000ff", &err));
    CHECK(q.size() == 3 && q.contains(1) && q.contains(0xabc) && q.contains(0xff) && !q.contains(2));
    CHECK(!q.contains(0));
    ShaderQuarantine bad;
    CHECK(!bad.add_list("0x1,nothex", &err) && has(err, "nothex") && bad.empty());  // all or nothing

    // Per-run file: bare hashes and comments allowed.
    ShaderQuarantine f;
    CHECK(f.add_text("# c\n\n0x10\r\n0x20 hangs here\n", false, "run.txt", &err));
    CHECK(f.size() == 2 && f.contains(0x10) && f.contains(0x20));

    // Repo list: reason + issue link required; '#123' mid-line is a reference, not a comment.
    ShaderQuarantine r;
    CHECK(r.add_text("# header only\n", true, "list", &err) && r.empty());  // empty by default
    CHECK(r.add_text("0xaa hair pass hang https://example.org/issues/1\n0xbb crashes DXC #42\n", true, "list", &err));
    CHECK(r.size() == 2 && r.entries().at(0xbb) == "crashes DXC #42");
    ShaderQuarantine r2;
    CHECK(!r2.add_text("0xaa\n", true, "list", &err) && has(err, "list line 1") && r2.empty());
    CHECK(!r2.add_text("0xaa only a reason\n", true, "list", &err));
    CHECK(!r2.add_text("0xcc ok #1\nzz bad #2\n", true, "list", &err) && has(err, "line 2") && r2.empty());

    // Accounting: skips land in the drop table as "quarantined" with the hash, split draws / dispatches.
    DropTable t;
    t.record(0xabc, DropTable::Reason::Quarantined, false, 2);
    t.record(0xabc, DropTable::Reason::Quarantined, true);
    const std::string d = t.detail_json();
    CHECK(has(d, "\"hash\": \"0x0000000000000abc\", \"draws\": 2, \"dispatches\": 1"));
    CHECK(has(d, "\"quarantined\":3"));
    CHECK(!has(d, "no_shader_module\":1"));

    // Auto-quarantine file: new failed hashes are appended once, existing notes kept, 0 ignored.
    CHECK(ShaderQuarantine::merge_auto_file("", {}).empty());
    const std::string a1 = ShaderQuarantine::merge_auto_file("", {0x10, 0x10, 0});
    CHECK(has(a1, "0x0000000000000010 pipeline_failed auto\n") && a1[0] == '#');
    CHECK(ShaderQuarantine::merge_auto_file(a1, {0x10}).empty());
    const std::string a2 = ShaderQuarantine::merge_auto_file(a1, {0x10, 0x20});
    CHECK(has(a2, a1.c_str()) && has(a2, "0x0000000000000020 pipeline_failed auto\n"));
    ShaderQuarantine rt;
    CHECK(rt.add_text(a2, false, "auto", &err) && rt.size() == 2 && rt.contains(0x20));
    CHECK(ShaderQuarantine::merge_auto_file("zz\n", {0x30}).empty());  // unparsable file is never rewritten

    // persist_auto_file (the --auto-quarantine write in boot_main.cpp).
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / ("pg6_aq_" + std::to_string(std::rand()));
    fs::create_directories(dir);
    const std::string file = (dir / "auto.txt").string();
    CHECK(ShaderQuarantine::persist_auto_file(file, {}) == 0 && !fs::exists(file));  // nothing to add: no file
    CHECK(ShaderQuarantine::persist_auto_file(file, {0x10}) == 1);                    // created on first failure
    std::string body = slurp(file);
    CHECK(body[0] == '#' && count(body, "0x0000000000000010 pipeline_failed auto\n") == 1);
    CHECK(ShaderQuarantine::persist_auto_file(file, {0x10}) == 0 && slurp(file) == body);  // no duplicates
    CHECK(ShaderQuarantine::persist_auto_file(file, {0x10, 0x20}) == 1);
    body = slurp(file);
    CHECK(count(body, "0x0000000000000010") == 1 && count(body, "0x0000000000000020") == 1);
    const std::string hand = (dir / "hand.txt").string();
    std::ofstream(hand, std::ios::binary) << "# my note\n0x99 hangs it https://example.invalid/1\n";
    CHECK(ShaderQuarantine::persist_auto_file(hand, {0x99, 0x30}) == 1);  // 0x99 already there: only 0x30 added
    body = slurp(hand);
    CHECK(body.rfind("# my note\n0x99 hangs it https://example.invalid/1\n", 0) == 0 &&
          count(body, "0x0000000000000030") == 1 && !has(body, "0000000000000099"));
    const std::string killed = (dir / "killed.txt").string();  // flushed before the process is killed right after
    const std::string cmd = "\"\"" + std::string(argv[0]) + "\" --persist-child \"" + killed + "\" 1a 2b\"";
    CHECK(std::system(cmd.c_str()) == 0);
    body = slurp(killed);
    CHECK(has(body.c_str(), "0x000000000000001a pipeline_failed auto\n") && has(body.c_str(), "0x000000000000002b pipeline_failed auto\n"));
    // an unwritable path (a directory in place of the file, and a missing parent) just reports -1
    CHECK(ShaderQuarantine::persist_auto_file(dir.string(), {0x40}) == -1);
    CHECK(ShaderQuarantine::persist_auto_file((dir / "no" / "such" / "f.txt").string(), {0x40}) == -1);
    std::error_code ec;
    fs::remove_all(dir, ec);

    if (g_failures) return 1;
    std::printf("test_native_exec_quarantine: ok\n");
    return 0;
}
