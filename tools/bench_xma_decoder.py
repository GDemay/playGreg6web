#!/usr/bin/env python3
"""Matched public WMA Pro FrameDecoder::decode timings on synthetic valid frames.

Compile the same tests/test_xma_decoder.cpp harness against baseline and patch
runtime/os/xma_decoder.cpp plus runtime/os/xapu_opus.cpp, with identical C++20
Release flags. Opus support is unnecessary for these XMA/WMA Pro workloads.
"""
import argparse
import json
import platform
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before')
    parser.add_argument('after')
    parser.add_argument('--baseline-commit', required=True)
    parser.add_argument('--patch-commit', required=True)
    parser.add_argument('--samples', type=int, default=7)
    args = parser.parse_args()
    if args.samples < 3:
        parser.error('at least 3 samples required')
    binaries = {'before': args.before, 'after': args.after}
    cases = []
    for size, channels, frames, kind in [
        (64, 2, 2000, 'nonzero'), (128, 2, 2000, 'nonzero'),
        (256, 2, 2000, 'nonzero'), (512, 2, 2000, 'nonzero'),
        (512, 1, 2000, 'nonzero'),
        (512, 2, 100000, 'silence'), (512, 2, 100000, 'malformed'),
    ]:
        flags = ['--bench-wmapro', str(size), str(channels), str(frames), kind]
        samples = {'before': [], 'after': []}
        for index in range(args.samples):
            for label in (['before', 'after'] if index % 2 == 0 else ['after', 'before']):
                completed = subprocess.run([binaries[label], *flags], text=True, capture_output=True, check=True)
                result = json.loads(completed.stdout.strip().splitlines()[-1])
                if not result['correct']:
                    raise RuntimeError('decoder correctness check failed')
                samples[label].append(result)
        for key in ['pcm_hash', 'first_pcm_hash']:
            if len({s[key] for variant in samples.values() for s in variant}) != 1:
                raise RuntimeError('baseline/patch PCM output differs: ' + key)
        cases.append({'subframe': size, 'channels': channels, 'kind': kind,
                      'controlWorkload': kind != 'nonzero', 'samples': samples})
    print(json.dumps({
        'benchmark': 'synthetic native public WMA Pro FrameDecoder decode',
        'baselineCommit': args.baseline_commit, 'patchCommit': args.patch_commit,
        'environment': {'system': platform.system(), 'release': platform.release(), 'arch': platform.machine()},
        'sampling': 'separate prebuilt binaries; alternating A/B order across fresh-process samples; first decode timed separately; '
                    'three untimed decodes then state reset before warmed batch; fixture encoding, allocation and all PCM hashing outside timing',
        'metrics': 'wall_ms/cpu_ms are warmed batch totals; first_decode_us/first_decode_cpu_us include cold production initialization',
        'unmeasured': 'real active XMA voices/call frequency, audio-device latency, browser/Wasm, other codecs and game FPS; no game assets used',
        'cases': cases,
    }, indent=2))


if __name__ == '__main__':
    main()
