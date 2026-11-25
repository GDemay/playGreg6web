#pragma once

#include <cstddef>
#include <cstdint>

// Stereo downmix of the guest audio endpoint and the PCM ring a host reads it from.
//
// The guest renders a 7.1.4 static-object bed (ISpatialAudioClient, one mono float32 buffer per
// AudioObjectType) and/or a stereo IAudioRenderClient stream, all 48 kHz float32. The browser audio
// worker takes interleaved float32 at 48 kHz, so everything is mixed here into one stereo stream
// and pushed into a single-producer / single-consumer ring in runtime memory that the host maps.

namespace playgta6::os::audio_downmix {

constexpr uint32_t kSampleRate = 48000;
constexpr uint32_t kChannels = 2;
constexpr uint32_t kRingSamples = 1u << 17;  // floats (power of two): 65536 frames, about 1.36 s

// AudioObjectType bit values (audioclient / spatialaudioclient.h).
enum ObjectType : uint32_t {
    kFrontLeft = 0x2,
    kFrontRight = 0x4,
    kFrontCenter = 0x8,
    kLowFrequency = 0x10,
    kSideLeft = 0x20,
    kSideRight = 0x40,
    kBackLeft = 0x80,
    kBackRight = 0x100,
    kTopFrontLeft = 0x200,
    kTopFrontRight = 0x400,
    kTopBackLeft = 0x800,
    kTopBackRight = 0x1000,
};

// Gains of one static object into left/right (ITU-R BS.775 style): fronts 1.0, centre, sides and
// backs -3 dB, LFE dropped (as ITU does); top-front folds into the fronts at -3 dB and top-back
// into its side at -6 dB. Unknown types (and dynamic objects) get {0, 0}.
struct StereoGain {
    float left;
    float right;
};
StereoGain downmix_gain(uint32_t object_type);

// Adds `frames` mono samples of one object into interleaved stereo `out` (accumulates).
void mix_object_into_stereo(uint32_t object_type, const float* mono, uint32_t frames, float* out);
// Adds interleaved stereo `in` into interleaved stereo `out` (accumulates).
void mix_stereo_into_stereo(const float* in, uint32_t frames, float* out);

// Single-producer / single-consumer float ring. Indices are free-running sample counters (they
// wrap at 2^32); capacity is a power of two. The producer only writes `*write_index`, the
// consumer only `*read_index`. A full ring drops the new samples (never the unread ones).
struct PcmRing {
    float* data;
    uint32_t capacity;
    uint32_t* write_index;
    uint32_t* read_index;
};

// Producer side: pushes up to `count` samples, returns how many fitted.
size_t ring_push(PcmRing& ring, const float* samples, size_t count);
// Consumer side: pops up to `count` samples, returns how many were read.
size_t ring_pop(PcmRing& ring, float* out, size_t count);
// Samples currently readable.
size_t ring_readable(const PcmRing& ring);

// The runtime's one ring (fed by the mixer below).
PcmRing& runtime_ring();

// Mixer: stages sources between guest threads and the period tick. push_* may be called from
// any guest thread; flush_period() (the engine tick) sums one period of every source into the
// ring and the optional stereo sink.
void push_spatial_pass(const uint32_t* types, const float* const* buffers, uint32_t objects,
                       uint32_t frames);
void push_render_stereo(const float* interleaved, uint32_t frames);
// Mixes up to `period_frames` staged frames per source; nothing is pushed when no source has
// data. Returns the frames delivered.
uint32_t flush_period(uint32_t period_frames);
// Tests and shutdown: drops staged audio and the sink, zeroes the ring indices.
void reset();

using StereoSink = void (*)(void* user, const float* stereo, uint32_t frames, uint32_t sample_rate);
void set_stereo_sink(StereoSink sink, void* user);

}  // namespace playgta6::os::audio_downmix

// Host view of the PCM ring. All pointers are into runtime memory (wasm: linear-memory offsets).
// Samples are interleaved float32 stereo at `sample_rate`; `capacity` is in floats (a power of
// two). The ring is SPSC: the runtime owns `*write_index`, the host owns `*read_index`; both are
// free-running uint32 sample counters, so readable = (write - read) mod 2^32 and the slot of an
// index is index & (capacity - 1). Read with Atomics.load on write / Atomics.store on read.
extern "C" {
struct Playgta6AudioRing {
    uintptr_t base;             // float32 data
    uint32_t capacity;          // floats
    uint32_t channels;          // 2
    uint32_t sample_rate;       // 48000
    uintptr_t read_index_ptr;   // uint32 owned by the host
    uintptr_t write_index_ptr;  // uint32 owned by the runtime
};
const Playgta6AudioRing* playgta6_audio_ring(void);
}
