#!/usr/bin/env python3
"""Matched synthetic public XG COM slot 8/9 benchmarks; no game data or browser FPS.

Build the same tests/test_xbox_scarlett_guest.cpp against baseline and patch core
libraries, then pass the two resulting test_xbox_scarlett_guest binaries here.
The test's --benchmark-texture-copy mode runs only the requested transfer workload.
"""
import argparse
import json
import platform
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline_binary')
    parser.add_argument('patch_binary')
    parser.add_argument('--samples', type=int, default=7)
    parser.add_argument('--iterations', type=int, default=3)
    args = parser.parse_args()
    if args.samples < 3 or args.iterations < 1:
        parser.error('samples >= 3 and iterations >= 1 required')
    binaries = {'before': args.baseline_binary, 'after': args.patch_binary}
    cases = []
    for name, width, height, fmt, padding, layout in [
        ('4K RGBA tiled tight', 3840, 2160, 28, 0, 0x105),
        ('4K BC1 tiled tight', 3840, 2160, 71, 0, 0x105),
        ('4096 square RGBA exact-footprint', 4096, 4096, 28, 0, 0x105),
        ('4096 square BC1 exact-footprint', 4096, 4096, 71, 0, 0x105),
        ('4K RGBA tiled padded control', 3840, 2160, 28, 64, 0x105),
        ('4K RGBA console-linear tight', 3840, 2160, 28, 0, 0x100),
        ('small RGBA tiled tight control', 64, 64, 28, 0, 0x105),
    ]:
        for slot in [8, 9]:
            flags = ['--benchmark-texture-copy', str(width), str(height), str(fmt), str(padding),
                     str(slot), str(args.iterations), str(layout)]
            def run(label):
                completed = subprocess.run([binaries[label], *flags], check=True, text=True, capture_output=True)
                result = json.loads(completed.stdout.strip().splitlines()[-1])
                if not result['correct']:
                    raise RuntimeError('transfer output mismatch')
                return result
            for label in binaries:
                run(label)
            samples = {'before': [], 'after': []}
            for i in range(args.samples):
                for label in (['before', 'after'] if i % 2 == 0 else ['after', 'before']):
                    samples[label].append(run(label))
            hashes = {sample['output_hash'] for values in samples.values() for sample in values}
            if len(hashes) != 1:
                raise RuntimeError('baseline and patch output hashes differ')
            cases.append({'workload': name, 'slot': slot, 'samples': samples})
    print(json.dumps({
        'benchmark': 'synthetic native public XG texture copy COM calls',
        'environment': {'system': platform.system(), 'release': platform.release(), 'machine': platform.machine()},
        'sampling': 'separate prebuilt binaries; one untimed invocation per variant/workload; alternating A/B sample order; '
                    'each invocation constructs identical fixtures and warms one transfer before timing; full output/padding checks outside timing',
        'unmeasured': 'real title call frequency, browser/Wasm execution, real assets and game FPS; no bridge-call counter instrumentation',
        'cases': cases,
    }, indent=2))


if __name__ == '__main__':
    main()
