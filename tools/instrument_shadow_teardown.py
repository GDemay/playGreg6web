#!/usr/bin/env python3
"""Scratch-only matched Resource destructor timer; changes benchmark source, not production code."""
import argparse
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('source', type=Path)
a = p.parse_args()
s = a.source.read_text()

def replace(old, new):
    global s
    if s.count(old) != 1:
        raise RuntimeError(f'expected one benchmark anchor, got {s.count(old)}: {old[:80]}')
    s = s.replace(old, new, 1)

replace('        std::printf("{\\"buffers\\":', '''        const double teardown_cpu_start = process_cpu_ms();
        const auto teardown_start = std::chrono::steady_clock::now();
        resources.clear();  // Ref release destroys every Resource and its shadow allocations.
        const double teardown_wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - teardown_start).count();
        const double teardown_cpu_ms = process_cpu_ms() - teardown_cpu_start;
        const uint64_t shadow_after_destroy = device->stats().shadow_resident_bytes;
        if (shadow_after_destroy != 0) throw std::runtime_error("shadow payload not released");
        std::printf("{\\"buffers\\":''')
replace('\\"cpu_ms\\":%.6f,', '\\"cpu_ms\\":%.6f,\\"teardown_wall_ms\\":%.6f,\\"teardown_cpu_ms\\":%.6f,\\"shadow_payload_after_destroy_bytes\\":%llu,')
replace('wall_ms, wall_ms / frames, cpu_ms,', 'wall_ms, wall_ms / frames, cpu_ms, teardown_wall_ms, teardown_cpu_ms, static_cast<unsigned long long>(shadow_after_destroy),')
replace('        resources.clear(); device.reset();', '        device.reset();')
a.source.write_text(s)
print('Applied benchmark-only Resource teardown timers; production code unchanged.')
