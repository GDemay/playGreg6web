#include "gpu/shader_pack_index.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <limits>

namespace playgta6::gpu {

namespace {

uint32_t u32_at(const uint8_t* p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}

void set_error(std::string* error, std::string message) {
    if (error != nullptr) {
        *error = std::move(message);
    }
}

}  // namespace

bool parse_shader_pack_header(const uint8_t* data,
                              size_t size,
                              uint32_t* out_count,
                              std::string* error) {
    if (data == nullptr || size < kShaderPackHeaderBytes || std::memcmp(data, "WGPK", 4) != 0) {
        set_error(error, "not a WGPK shader pack");
        return false;
    }
    const uint32_t version = u32_at(data + 4);
    const uint32_t count = u32_at(data + 8);
    if (version != 1) {
        set_error(error, "unsupported WGPK version " + std::to_string(version));
        return false;
    }
    if (out_count != nullptr) {
        *out_count = count;
    }
    return true;
}

bool parse_shader_pack_index_toc(const uint8_t* toc_data,
                                 size_t toc_size,
                                 uint64_t total_file_size,
                                 ShaderPackIndex* out,
                                 std::string* error) {
    if (out == nullptr) {
        set_error(error, "null shader pack index output");
        return false;
    }
    out->clear();

    uint32_t count = 0;
    if (!parse_shader_pack_header(toc_data, toc_size, &count, error)) {
        return false;
    }
    if (count > (toc_size - kShaderPackHeaderBytes) / kShaderPackEntryBytes) {
        set_error(error, "WGPK entry table exceeds the file");
        return false;
    }
    const uint64_t toc_bytes =
        uint64_t{kShaderPackHeaderBytes} + uint64_t{count} * uint64_t{kShaderPackEntryBytes};
    if (total_file_size > 0 && toc_bytes > total_file_size) {
        set_error(error, "WGPK entry table exceeds the file");
        return false;
    }

    out->reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        const uint8_t* e = toc_data + kShaderPackHeaderBytes + size_t{i} * kShaderPackEntryBytes;
        uint64_t hash;
        std::memcpy(&hash, e, 8);
        const uint8_t stage = e[8];
        const uint32_t offset = u32_at(e + 12);
        const uint32_t length = u32_at(e + 16);
        if (stage < 1 || stage > 3) {
            out->clear();
            set_error(error, "WGPK entry " + std::to_string(i) + " has stage " + std::to_string(stage));
            return false;
        }
        const uint64_t payload_end = uint64_t{offset} + uint64_t{length};
        if (payload_end > std::numeric_limits<uint32_t>::max()) {
            out->clear();
            set_error(error, "WGPK entry " + std::to_string(i) + " payload range overflows");
            return false;
        }
        if (length != 0 && uint64_t{offset} < toc_bytes) {
            out->clear();
            set_error(error, "WGPK entry " + std::to_string(i) + " payload overlaps header/TOC");
            return false;
        }
        if (total_file_size > 0 && payload_end > total_file_size) {
            out->clear();
            set_error(error, "WGPK entry " + std::to_string(i) + " payload exceeds the file");
            return false;
        }
        out->emplace(hash, static_cast<PackStage>(stage));
    }
    return true;
}

bool parse_shader_pack_index(const uint8_t* data, size_t size, ShaderPackIndex* out, std::string* error) {
    return parse_shader_pack_index_toc(data, size, static_cast<uint64_t>(size), out, error);
}

std::string pack_key_code(uint64_t key) {
    char b[24];
    std::snprintf(b, sizeof(b), "0x%016" PRIx64, key);
    return b;
}

}  // namespace playgta6::gpu
