#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <type_traits>

#include "abi/playgta6_abi.h"

namespace playgta6::gpu {

class CommandEncoder {
public:
    static constexpr size_t kMinCapacity = 4096;

    CommandEncoder() noexcept = default;
    ~CommandEncoder() noexcept;
    CommandEncoder(const CommandEncoder& other);
    CommandEncoder& operator=(const CommandEncoder& other);
    CommandEncoder(CommandEncoder&& other) noexcept;
    CommandEncoder& operator=(CommandEncoder&& other) noexcept;

    void clear() noexcept;
    void reserve(size_t byte_capacity);
    void reset_and_shrink(size_t max_retain_bytes = 0);
    void append_raw(const void* bytes, size_t byte_size);

    void nop();
    // A Nop packet carrying `payload_bytes` of zero padding (a patchable placeholder).
    void nop_padded(uint32_t payload_bytes);
    void begin_frame();
    void clear_color(float r, float g, float b, float a);
    void create_buffer(uint32_t id, uint32_t usage_flags, uint64_t byte_size);
    void write_buffer(uint32_t id, uint32_t dst_offset, const void* data, uint32_t data_len);
    void create_shader_module(uint32_t id, GpuShaderStage stage, std::string_view wgsl_code);
    void create_render_pipeline(
        uint32_t id,
        uint32_t vs_shader_id,
        uint32_t fs_shader_id,
        uint32_t vertex_stride,
        GpuPrimitiveTopology topology = GpuPrimitiveTopology::TriangleList,
        uint32_t flags = 0);
    void set_pipeline(uint32_t pipeline_id);
    void set_vertex_buffer(uint32_t slot, uint32_t buffer_id, uint64_t offset = 0);
    void draw(
        uint32_t vertex_count,
        uint32_t instance_count = 1,
        uint32_t first_vertex = 0,
        uint32_t first_instance = 0);
    void present();
    void destroy_resource(uint32_t id);
    void create_texture(
        uint32_t id,
        uint32_t width,
        uint32_t height,
        uint32_t depth_or_layers,
        uint32_t mip_levels,
        GpuTextureFormat format,
        uint32_t usage_flags,
        GpuTextureDimension dimension = GpuTextureDimension::Dim2D);
    void write_texture(
        uint32_t texture_id,
        uint32_t mip_level,
        uint32_t origin_x,
        uint32_t origin_y,
        uint32_t origin_z,
        uint32_t width,
        uint32_t height,
        uint32_t depth_or_layers,
        uint32_t bytes_per_row,
        uint32_t rows_per_image,
        const void* data,
        uint32_t data_len);
    void create_sampler(
        uint32_t id,
        GpuAddressMode address_u = GpuAddressMode::ClampToEdge,
        GpuAddressMode address_v = GpuAddressMode::ClampToEdge,
        GpuAddressMode address_w = GpuAddressMode::ClampToEdge,
        GpuFilterMode mag_filter = GpuFilterMode::Linear,
        GpuFilterMode min_filter = GpuFilterMode::Linear,
        GpuFilterMode mipmap_filter = GpuFilterMode::Linear,
        uint32_t max_anisotropy = 1);
    void create_bind_group(
        uint32_t id,
        uint32_t pipeline_id,
        uint32_t group_index,
        const GpuBindGroupEntry* entries,
        uint32_t entry_count);
    void set_bind_group(uint32_t group_index, uint32_t bind_group_id);
    void set_index_buffer(
        uint32_t buffer_id,
        GpuIndexFormat index_format = GpuIndexFormat::Uint16,
        uint64_t offset = 0);
    void draw_indexed(
        uint32_t index_count,
        uint32_t instance_count = 1,
        uint32_t first_index = 0,
        int32_t base_vertex = 0,
        uint32_t first_instance = 0);
    void create_compute_pipeline(uint32_t id, uint32_t cs_shader_id, uint32_t flags = 0);
    void set_compute_pipeline(uint32_t pipeline_id);
    void dispatch_compute(uint32_t workgroup_x, uint32_t workgroup_y = 1, uint32_t workgroup_z = 1);
    // DrawIndirect / DrawIndexedIndirect / DispatchIndirect (GpuIndirectPayload).
    void indirect(GpuOpcode opcode, uint32_t buffer_id, uint64_t offset);
    // Generic packet for opcodes without a dedicated method: payload = `head` followed by `tail`.
    void emit_packet(
        GpuOpcode opcode,
        const void* head,
        uint32_t head_bytes,
        const void* tail = nullptr,
        uint32_t tail_bytes = 0);

    const uint8_t* data() const noexcept { return data_; }
    // Overwrites `bytes` already-encoded bytes at `offset` (false when out of range).
    bool patch(size_t offset, const void* bytes, size_t size);
    size_t size() const noexcept { return size_; }
    size_t capacity() const noexcept { return capacity_; }
    bool empty() const noexcept { return size_ == 0; }

private:
    uint8_t* reserve_append(size_t extra_bytes) {
        if (extra_bytes > capacity_ - size_) [[unlikely]] {
            grow_append(extra_bytes);
        }
        uint8_t* dst = data_ + size_;
        size_ += extra_bytes;
        return dst;
    }

    void grow_append(size_t extra_bytes);
    void grow_for(size_t required_capacity);

    void write_empty_packet(GpuOpcode opcode) {
        const GpuPacketHeader header{
            static_cast<uint16_t>(opcode),
            0,
            0,
        };
        uint8_t* dst = reserve_append(sizeof(GpuPacketHeader));
        std::memcpy(dst, &header, sizeof(GpuPacketHeader));
    }

    template <typename T>
    void write_pod_packet(GpuOpcode opcode, const T& payload) {
        static_assert(std::is_trivially_copyable_v<T>);
        static_assert(sizeof(T) <= 0xFFFFFFFFu);
        const GpuPacketHeader header{
            static_cast<uint16_t>(opcode),
            0,
            static_cast<uint32_t>(sizeof(T)),
        };
        uint8_t* dst = reserve_append(sizeof(GpuPacketHeader) + sizeof(T));
        std::memcpy(dst, &header, sizeof(GpuPacketHeader));
        std::memcpy(dst + sizeof(GpuPacketHeader), &payload, sizeof(T));
    }

    void write_packet_with_tail(
        GpuOpcode opcode,
        const void* head,
        uint32_t head_bytes,
        const void* tail,
        uint32_t tail_bytes);

    uint8_t* data_ = nullptr;
    size_t size_ = 0;
    size_t capacity_ = 0;
};

}  // namespace playgta6::gpu
