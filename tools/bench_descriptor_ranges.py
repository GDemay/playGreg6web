#!/usr/bin/env python3
"""Matched synthetic public DescriptorHeapRegistry calls; no GPU or game data."""
import argparse
import json
import platform
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before')
    parser.add_argument('after')
    parser.add_argument('--samples', type=int, default=7)
    parser.add_argument('--baseline-commit', required=True)
    parser.add_argument('--patch-commit', required=True)
    args = parser.parse_args()
    if args.samples < 3:
        parser.error('at least three samples required')
    names = ['interleaved-disjoint', 'different-heaps', 'disjoint-bounds', 'early-overlap',
             'multiple-source-heaps', 'later-overlap', 'shuffled-disjoint', 'shuffled-later-overlap']
    cases = []
    for count in [64, 256, 1024, 4096, 8192]:
        for mode, name in enumerate(names):
            iterations = max(8, 256000 // count) if mode in (0, 4, 6) else max(256, 1024000 // count)
            def run(which):
                sample = json.loads(subprocess.check_output(
                    [getattr(args, which), str(count), str(iterations), str(mode)], text=True))
                if not sample['correct']:
                    raise RuntimeError('descriptor oracle failed')
                return sample
            run('before')
            run('after')
            samples = {'before': [], 'after': []}
            for i in range(args.samples):
                for which in (['before', 'after'] if i % 2 == 0 else ['after', 'before']):
                    samples[which].append(run(which))
            if len({s['digest'] for group in samples.values() for s in group}) != 1:
                raise RuntimeError('baseline/patch descriptor hashes differ')
            cases.append({'workload': name, 'ranges': count, 'samples': samples})
    print(json.dumps({
        'benchmark': 'synthetic actual DescriptorHeapRegistry::copy_descriptors elapsed wall time',
        'baseline': args.baseline_commit, 'patch': args.patch_commit,
        'environment': {'os': platform.system(), 'release': platform.release(), 'arch': platform.machine()},
        'sampling': 'separate prebuilt binaries; one untimed outer invocation each, then alternating A/B pairs; '
                    'each invocation constructs heaps and warms one copy before timing; non-alias workloads '
                    'alternate two distinct source streams so destinations change on every timed call; alias '
                    'controls repeat identical sources; complete descriptor checks and hashing outside timing',
        'unmeasured': 'game descriptor distributions/frequency, browser/Wasm, GPU work, game frame times, '
                      'first-call latency and process RSS',
        'cases': cases,
    }, indent=2))


if __name__ == '__main__':
    main()
