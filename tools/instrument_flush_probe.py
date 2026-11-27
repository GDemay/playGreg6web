#!/usr/bin/env python3
"""Scratch-only baseline diagnostic: apply after copying bench_native_flush.cpp into checkout.
No product patch. Build an otherwise unchanged baseline, run one bounded diagnostic workload,
retain stderr PG6_FLUSH_PROBE totals and the benchmark stdout correctness/digest JSON.
Optional --resident-shadow prepopulates shadows in untimed warmup as a separate control.
"""
from pathlib import Path
import argparse

p = argparse.ArgumentParser()
p.add_argument('checkout', type=Path)
p.add_argument('--resident-shadow', action='store_true', help='legacy always-resident control; otherwise PG6_PROFILE_RESIDENT_SHADOW enables it per invocation')
a = p.parse_args()
root = a.checkout

def replace(s, old, new):
    if s.count(old) != 1:
        raise RuntimeError(f'Expected exactly one anchor ({s.count(old)} found): {old[:100]}')
    return s.replace(old, new, 1)

r = root / 'runtime/gpu/d3d12_guest_resources.cpp'
s = r.read_text()
anchor = 'namespace playgta6::gpu::d3d12_guest {\n'
probe = r'''
// SCRATCH DIAGNOSTIC ONLY: thread-local aggregated timings, no per-flush logging.
struct FlushProbeTotals {
    uint64_t calls=0,total_ns=0,batch_ns=0,take_ns=0,index_ns=0,buffer_loop_ns=0,shadow_ns=0;
    uint64_t watch_calls=0,watch_ns=0,watch_bytes=0,watch_pages=0,alloc_calls=0,alloc_ns=0,alloc_bytes=0;
};
thread_local FlushProbeTotals t_flush_probe;
using ProbeClock = std::chrono::steady_clock;
static uint64_t probe_elapsed(ProbeClock::time_point start) {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(ProbeClock::now()-start).count());
}
extern "C" void pg6_flush_probe_reset() { t_flush_probe = {}; }
extern "C" void pg6_flush_probe_watch(uint64_t ns, uint64_t bytes, uint64_t pages) {
    auto& x=t_flush_probe; ++x.watch_calls; x.watch_ns+=ns; x.watch_bytes+=bytes; x.watch_pages+=pages;
}
extern "C" void pg6_flush_probe_dump() {
    const auto& x=t_flush_probe;
    std::fprintf(stderr,"PG6_FLUSH_PROBE {\"flushes\":%llu,\"total_ns\":%llu,\"batch_ns\":%llu,\"take_ns\":%llu,\"index_ns\":%llu,\"buffer_loop_ns\":%llu,\"shadow_ns\":%llu,\"watch_calls\":%llu,\"watch_ns\":%llu,\"watch_bytes\":%llu,\"watch_pages\":%llu,\"shadow_alloc_calls\":%llu,\"shadow_alloc_ns\":%llu,\"shadow_alloc_bytes\":%llu}\n",
        (unsigned long long)x.calls,(unsigned long long)x.total_ns,(unsigned long long)x.batch_ns,
        (unsigned long long)x.take_ns,(unsigned long long)x.index_ns,(unsigned long long)x.buffer_loop_ns,
        (unsigned long long)x.shadow_ns,(unsigned long long)x.watch_calls,(unsigned long long)x.watch_ns,
        (unsigned long long)x.watch_bytes,(unsigned long long)x.watch_pages,(unsigned long long)x.alloc_calls,
        (unsigned long long)x.alloc_ns,(unsigned long long)x.alloc_bytes);
}
struct ProbeWholeFlush {
    ProbeClock::time_point start=ProbeClock::now();
    ~ProbeWholeFlush() { ++t_flush_probe.calls; t_flush_probe.total_ns+=probe_elapsed(start); }
};
'''
s = replace(s, anchor, anchor + probe)
s = replace(s, 'void Device::flush_cpu_buffers(CommandEncoder& encoder) {',
            'void Device::flush_cpu_buffers(CommandEncoder& encoder) {\n    ProbeWholeFlush probe_whole;')
s = replace(s, '            write_batch.emplace(&host_ranges);',
            '            const auto probe_batch_start=ProbeClock::now();\n            write_batch.emplace(&host_ranges);\n            t_flush_probe.batch_ns+=probe_elapsed(probe_batch_start);')
s = replace(s, '                os::take_written_pages_ranges(take_ranges, &flush_cursor_, &dirty_pages, &ok);',
            '                const auto probe_take_start=ProbeClock::now();\n                os::take_written_pages_ranges(take_ranges, &flush_cursor_, &dirty_pages, &ok);\n                t_flush_probe.take_ns+=probe_elapsed(probe_take_start);')
s = replace(s, '                std::vector<const void*> visit;\n                buffer_page_index_.collect_pages',
            '                const auto probe_index_start=ProbeClock::now();\n                std::vector<const void*> visit;\n                buffer_page_index_.collect_pages')
s = replace(s, '                uint64_t visited = 0, skipped = 0, untracked = 0;',
            '                t_flush_probe.index_ns+=probe_elapsed(probe_index_start);\n                const auto probe_loop_start=ProbeClock::now();\n                uint64_t visited = 0, skipped = 0, untracked = 0;')
s = replace(s, '                    const uint64_t uploaded = r->flush_shadow(encoder, &mark_pass_);',
            '                    const auto probe_shadow_start=ProbeClock::now();\n                    const uint64_t uploaded = r->flush_shadow(encoder, &mark_pass_);\n                    t_flush_probe.shadow_ns+=probe_elapsed(probe_shadow_start);')
s = replace(s, '                count_work(Work::FlushBuffersVisited, visited);',
            '                t_flush_probe.buffer_loop_ns+=probe_elapsed(probe_loop_start);\n                count_work(Work::FlushBuffersVisited, visited);')
s = replace(s, '            chunk = std::make_unique<uint8_t[]>(static_cast<size_t>(chunk_bytes));  // zeroed',
            '            const auto probe_alloc_start=ProbeClock::now();\n            chunk = std::make_unique<uint8_t[]>(static_cast<size_t>(chunk_bytes));  // zeroed\n            t_flush_probe.alloc_ns+=probe_elapsed(probe_alloc_start);\n            ++t_flush_probe.alloc_calls; t_flush_probe.alloc_bytes+=chunk_bytes;')
r.write_text(s)

r = root / 'runtime/os/vmem_backing.cpp'
s = r.read_text()
anchor = 'namespace playgta6::os::backing {\n'
s = s.replace(anchor, 'extern "C" void pg6_flush_probe_watch(uint64_t, uint64_t, uint64_t);\n' + anchor, 1)
old = '''        if (GetWriteWatch(WRITE_WATCH_FLAG_RESET, reinterpret_cast<PVOID>(start), static_cast<SIZE_T>(end - start),
                          written.data(), &count, &granularity) != 0 ||
            granularity != kWrittenPageSize) {'''
new = '''        const auto probe_watch_start=std::chrono::steady_clock::now();
        const auto probe_watch_result=GetWriteWatch(WRITE_WATCH_FLAG_RESET, reinterpret_cast<PVOID>(start), static_cast<SIZE_T>(end - start),
                          written.data(), &count, &granularity);
        pg6_flush_probe_watch(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-probe_watch_start).count()), end-start, static_cast<uint64_t>(count));
        if (probe_watch_result != 0 || granularity != kWrittenPageSize) {'''
s = replace(s, old, new)
r.write_text(s)

r = root / 'tests/bench_native_flush.cpp'
s = r.read_text()
s = replace(s, 'namespace gpu = playgta6::gpu;',
            'extern "C" void pg6_flush_probe_reset();\nextern "C" void pg6_flush_probe_dump();\nnamespace gpu = playgta6::gpu;')
s = replace(s, '        const auto visits0 = dg::work_total',
            '        pg6_flush_probe_reset();\n        const auto visits0 = dg::work_total')
s = replace(s, '        std::printf("{\\"buffers\\":',
            '        pg6_flush_probe_dump();\n        std::printf("{\\"buffers\\":')
resident = 'true' if a.resident_shadow else 'std::getenv("PG6_PROFILE_RESIDENT_SHADOW") != nullptr'
s = replace(s, '        const std::vector<uint8_t> zero(bytes, 0);',
            '        const bool resident_shadow = ' + resident + ';\n        std::vector<uint8_t> zero(bytes, 0);\n        if (resident_shadow) zero[0] = 0xA5;  // control: warmup creates all chunks')
s = replace(s, '        std::vector<std::array<uint32_t, 12>> expected(count);',
            '        std::vector<std::array<uint32_t, 12>> expected(count);\n        if (resident_shadow) for (auto& words : expected) words[0] = 0xA5;')
r.write_text(s)
print('Applied scratch instrumentation; do not publish as a product patch.')
