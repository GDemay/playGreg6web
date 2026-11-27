#!/usr/bin/env python3
"""Measure public XG CopyFromSubresource slot 9 with synthetic mip chains.

Build identical tests/test_xbox_scarlett_guest.cpp harnesses against baseline and
patch core libraries. Pass the resulting two binaries, then immutable commit IDs.
No game files, GPU rendering or browser execution are used.

Reproduce from the PR checkout (replace PATCH with the measured code commit):
  git worktree add ../mip-base be4c597725345de8005196ef4b60c64b747208ba
  cp tests/test_xbox_scarlett_guest.cpp ../mip-base/tests/test_xbox_scarlett_guest.cpp
  cmake -S ../mip-base/runtime -B ../mip-base-build -DCMAKE_BUILD_TYPE=Release -DPLAYGTA6_OPUS=ON
  cmake --build ../mip-base-build --parallel 4 --target test_xbox_scarlett_guest
  cmake -S runtime -B work/mip-build -DCMAKE_BUILD_TYPE=Release -DPLAYGTA6_OPUS=ON
  cmake --build work/mip-build --parallel 4 --target test_xbox_scarlett_guest
  python3 tools/bench_xg_mip_copy.py ../mip-base-build/test_xbox_scarlett_guest work/mip-build/test_xbox_scarlett_guest --baseline-commit be4c597725345de8005196ef4b60c64b747208ba --patch-commit PATCH --samples 7 --iterations 3
"""
import argparse
import json
import platform
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline_binary')
    parser.add_argument('patch_binary')
    parser.add_argument('--baseline-commit', required=True)
    parser.add_argument('--patch-commit', required=True)
    parser.add_argument('--samples', type=int, default=7)
    parser.add_argument('--iterations', type=int, default=3)
    args = parser.parse_args()
    if args.samples < 3 or args.iterations < 1:
        parser.error('samples >= 3 and iterations >= 1 required')
    binaries = {'before': args.baseline_binary, 'after': args.patch_binary}
    cases = []
    for name, control, width, height, fmt, mips, mip, translated in [
        ('4096 RGBA mip 4', False, 4096, 4096, 28, 13, 4, 0),
        ('4096 BC1 mip 4', False, 4096, 4096, 71, 13, 4, 0),
        ('odd 4093x2163 RGBA mip 4', False, 4093, 2163, 28, 12, 4, 0),
        ('3840x2160 RGBA mip 0 control', True, 3840, 2160, 28, 12, 0, 0),
        ('4096 RGBA mip 3 above-cap control', True, 4096, 4096, 28, 13, 3, 0),
        ('4096 RGBA tail mip 12 control', True, 4096, 4096, 28, 13, 12, 0),
        ('1024 RGBA translated mip 3 control', True, 1024, 1024, 28, 11, 3, 1),
    ]:
        flags = ['--benchmark-texture-mip', str(width), str(height), str(fmt), str(0x105), str(mips),
                 str(mip), str(13), str(args.iterations), str(translated)]
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
        if len({s['output_hash'] for variant in samples.values() for s in variant}) != 1:
            raise RuntimeError('baseline and patch output hashes differ')
        cases.append({'workload': name, 'control': control, 'samples': samples})
    print(json.dumps({
        'benchmark': 'synthetic native public XG CopyFromSubresource small mip extraction',
        'baselineCommit': args.baseline_commit, 'patchCommit': args.patch_commit,
        'environment': {'system': platform.system(), 'release': platform.release(), 'machine': platform.machine()},
        'sampling': 'separate prebuilt binaries; one untimed invocation per variant/workload; seven alternating A/B samples by default; '
                    'each invocation constructs identical fixtures and warms one transfer before timing; full output/padding checks outside timing; '
                    'wall_ms and cpu_ms are totals across iterations (CPU is process user+system)',
        'unmeasured': 'real title call frequency, browser/Wasm execution, real assets and game FPS',
        'cases': cases,
    }, indent=2))


if __name__ == '__main__':
    main()
