#!/usr/bin/env python3
"""Matched actual Device flush shadow-memory/timing probes, including single-flush dense first uploads."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before')
    parser.add_argument('after')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--case-filter', choices=('all', 'resident-sparse'), default='all')
    args = parser.parse_args()
    cases = []
    workloads = [
        (4096, 64, 80, 'sparse-first'),
        (4096, 64, 80, 'sparse-resident'),
        (4096, 64, 0, 'sparse-resident'),
        (1024, 1, 1024, 'dense-first'),
        (4096, 1, 4096, 'dense-first'),
        (512, 16, 512, 'dense-resident'),
        (4096, 64, 80, 'sparse-fully-resident'),
    ]
    if args.case_filter == 'resident-sparse':
        workloads = [x for x in workloads if x[3] == 'sparse-fully-resident']
    for workload in workloads:
        print('Measuring', workload, flush=True)
        def run(label):
            result = subprocess.run([getattr(args, label), *map(str, workload)], text=True,
                                    capture_output=True, check=True)
            sample = json.loads(result.stdout.strip().splitlines()[-1])
            if not sample['correct'] or not sample['tracking']:
                raise RuntimeError('unsupported/incorrect measurement lane')
            return sample
        for label in ('before', 'after'):
            run(label)
        samples = {'before': [], 'after': []}
        for i in range(7):
            for label in (('before', 'after') if i % 2 == 0 else ('after', 'before')):
                samples[label].append(run(label))
        invariant_fields = ('packet_digest', 'visited', 'skipped', 'compared_bytes', 'take_ranges', 'uploads')
        for field in invariant_fields:
            if len({x[field] for group in samples.values() for x in group}) != 1:
                raise RuntimeError('baseline/patch mismatch: ' + field)
        for label, group in samples.items():
            for field in ('shadow_resident_before_bytes', 'shadow_resident_after_bytes'):
                if len({x[field] for x in group}) != 1:
                    raise RuntimeError('shadow allocation metric not repeatable: ' + label + '/' + field)
        cases.append({'arguments': workload, 'samples': samples})
    environment = json.loads(Path('runner-environment.json').read_text(encoding='utf-8-sig'))
    result = {
        'benchmark': 'synthetic actual native Device::flush_cpu_buffers with Windows write tracking; page-sized shadow probe',
        'environment': environment,
        'sampling': 'separate prebuilt MSVC Release binaries; one untimed invocation each, then seven alternating A/B pairs; four warmup flushes per invocation; setup, guest writes, output checks and metric collection excluded from timed flush; dense-first times ONE flush and is not averaged with resident flushes',
        'metrics': 'elapsed wall and GetProcessTimes user+kernel CPU; allocated shadow payload bytes from Resource::shadow_resident_bytes, excluding pointer-vector and allocator metadata; 4KiB pointer slots require up to 12 pointers instead of one per48KiB buffer',
        'unmeasured': 'real game allocation/write distribution, shared XMem views, browser/Wasm, GPU execution, actual process RSS and sustained game FPS',
        'cases': cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
