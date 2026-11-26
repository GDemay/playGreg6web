// tests/data/bvh_vectors.json is the language-neutral copy of the shared BVH node-test vectors (contract
// C5, tests/rt_bvh_vectors.h) for other consumers (translator tests, WebGPU harnesses). Every float is an
// IEEE-754 bit pattern (u32); inv_dir is not stored: inv = (dir == 0 ? +inf : 1 / dir) per component.
//
//   test_rt_vectors_json <path>                 checks the file equals rt_bvh_vectors::all() and that the
//                                               CPU twin reproduces every `expected` from the file alone
//   test_rt_vectors_json --write <path>         regenerates the file
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>

#include "gpu/rt_bvh.h"
#include "gpu/rt_json.h"
#include "rt_bvh_vectors.h"

namespace {
using namespace rt_bvh_vectors;
namespace js = playgta6::gpu::rt_json;
int g_failures = 0;
void fail(const std::string& m) {
    std::fprintf(stderr, "FAIL: %s\n", m.c_str());
    ++g_failures;
}

std::string hex(uint64_t v) {
    char b[24];
    std::snprintf(b, sizeof b, "0x%llx", (unsigned long long)v);
    return b;
}

std::string quote(const std::string& s) {
    std::string o = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') o += '\\';
        o += c;
    }
    return o + "\"";
}

std::string render() {
    std::string o = "{\n  \"version\": 1,\n  \"note\": \"floats are IEEE-754 bit patterns; inv_dir = dir == 0 ? +inf : 1 / dir "
                    "per component; node and addresses are hex strings\",\n  \"vectors\": [\n";
    const auto vs = all();
    for (size_t i = 0; i < vs.size(); ++i) {
        const Vector& v = vs[i];
        const auto arr = [](const uint32_t* p, int n, bool f, const float* fp) {
            std::string s = "[";
            for (int k = 0; k < n; ++k) {
                if (k) s += ", ";
                s += std::to_string(f ? fb(fp[k]) : p[k]);
            }
            return s + "]";
        };
        o += "    {\"name\": " + quote(v.name) + ",\n     \"desc\": " + arr(v.desc, 4, false, nullptr) +
             ", \"node\": \"" + hex(v.node) + "\", \"extent\": " + std::to_string(fb(v.extent)) +
             ",\n     \"origin\": " + arr(nullptr, 3, true, v.origin) + ", \"dir\": " + arr(nullptr, 3, true, v.dir) +
             ",\n     \"expected\": " + arr(v.expected, 4, false, nullptr) + ",\n     \"memory\": [";
        for (size_t m = 0; m < v.memory.size(); ++m) {
            if (m) o += ", ";
            o += "{\"addr\": \"" + hex(v.memory[m].addr) + "\", \"dwords\": [";
            for (size_t d = 0; d < v.memory[m].dwords.size(); ++d) {
                if (d) o += ", ";
                o += std::to_string(v.memory[m].dwords[d]);
            }
            o += "]}";
        }
        o += "]}";
        o += i + 1 < vs.size() ? ",\n" : "\n";
    }
    return o + "  ]\n}\n";
}

float bf(uint32_t u) {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
uint64_t hx(const js::Value* v) { return v ? std::strtoull(v->str.c_str(), nullptr, 16) : 0; }
}  // namespace

int main(int argc, char** argv) {
    if (argc == 3 && std::strcmp(argv[1], "--write") == 0) {
        std::ofstream(argv[2], std::ios::binary) << render();
        return 0;
    }
    if (argc != 2) {
        std::fprintf(stderr, "usage: test_rt_vectors_json [--write] <path>\n");
        return 2;
    }
    std::ifstream f(argv[1], std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    std::string text = ss.str();
    text.erase(std::remove(text.begin(), text.end(), '\x0d'), text.end());  // checkout may use CRLF
    if (text != render()) fail("tests/data/bvh_vectors.json is stale: run test_rt_vectors_json --write tests/data/bvh_vectors.json");

    js::Value root;
    std::string err;
    if (!js::parse(text, &root, &err)) {
        fail("bvh_vectors.json: " + err);
        return 1;
    }
    const js::Value* vecs = root.get("vectors");
    if (vecs == nullptr || vecs->arr.size() < 18) {
        fail("too few vectors");
        return 1;
    }
    for (const js::Value& v : vecs->arr) {
        std::map<uint64_t, uint32_t> mem;
        for (const js::Value& m : v.get("memory")->arr) {
            const uint64_t a = hx(m.get("addr"));
            const auto& d = m.get("dwords")->arr;
            for (size_t i = 0; i < d.size(); ++i) mem[a + 4 * i] = static_cast<uint32_t>(d[i].u64());
        }
        uint32_t desc[4], exp[4];
        float o[3], dr[3], inv[3];
        for (int k = 0; k < 4; ++k) {
            desc[k] = static_cast<uint32_t>(v.get("desc")->arr[k].u64());
            exp[k] = static_cast<uint32_t>(v.get("expected")->arr[k].u64());
        }
        for (int k = 0; k < 3; ++k) {
            o[k] = bf(static_cast<uint32_t>(v.get("origin")->arr[k].u64()));
            dr[k] = bf(static_cast<uint32_t>(v.get("dir")->arr[k].u64()));
            inv[k] = dr[k] == 0.0f ? INFINITY : 1.0f / dr[k];
        }
        const auto load = [&](uint64_t a) -> uint32_t {
            const auto it = mem.find(a);
            return it == mem.end() ? 0u : it->second;
        };
        const auto r = playgta6::gpu::rt_bvh::intersect(desc, hx(v.get("node")),
                                                        bf(static_cast<uint32_t>(v.get("extent")->u64())),
                                                        {o[0], o[1], o[2]}, {dr[0], dr[1], dr[2]},
                                                        {inv[0], inv[1], inv[2]}, load);
        if (std::memcmp(r.w, exp, 16) != 0) fail("json vector " + v.get("name")->str + " does not reproduce");
    }
    if (g_failures == 0) std::printf("test_rt_vectors_json: OK (%zu vectors)\n", vecs->arr.size());
    return g_failures == 0 ? 0 : 1;
}
