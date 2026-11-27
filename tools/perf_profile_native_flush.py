#!/usr/bin/env python3
"""Bounded baseline-only flush attribution, using scratch instrumentation (not a speedup benchmark)."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('binary')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    cases = []
    for resident in (False, True):
        for dirty in (80, 0):
            samples = []
            for _ in range(3):
                env = os.environ.copy()
                env.pop('PG6_PROFILE_RESIDENT_SHADOW', None)
                if resident:
                    env['PG6_PROFILE_RESIDENT_SHADOW'] = '1'
                run = subprocess.run([args.binary, '4096', '64', str(dirty), 'index'], env=env,
                                     text=True, capture_output=True, check=True)
                benchmark = json.loads(run.stdout.strip().splitlines()[-1])
                probes = [json.loads(line.split('PG6_FLUSH_PROBE ', 1)[1])
                          for line in run.stderr.splitlines() if line.startswith('PG6_FLUSH_PROBE ')]
                if not benchmark['correct'] or len(probes) != 1 or probes[0]['flushes'] != 64:
                    raise RuntimeError('diagnostic/correctness output invalid')
                samples.append({'benchmark': benchmark, 'profile': probes[0]})
            if len({s['benchmark']['packet_digest'] for s in samples}) != 1:
                raise RuntimeError('packet digest not repeatable')
            cases.append({'buffers': 4096, 'logical_bytes': 4096 * 49152, 'flushes': 64,
                          'dirty_buffers_per_flush': dirty, 'resident_shadow_control': resident,
                          'samples': samples})
    env_file = Path('runner-environment.json')
    result = {
        'kind': 'baseline-only scratch diagnostic, not a performance improvement measurement',
        'base_ref': '14d1b86644dbc35fbf32d9522c505972545a7d8a',
        'environment': json.loads(env_file.read_text(encoding='utf-8-sig')) if env_file.exists() else {},
        'sampling': 'one instrumented Release binary, three invocations per workload, four untimed warmup flushes per invocation; aggregate timers reset after warmup; logging outside timed loop; no concurrent build/tests',
        'timer_units': 'aggregate steady-clock nanoseconds over 64 flushes; nested watch and allocation times are subsets, not additive stages',
        'unmeasured': 'real game/frame distribution, shared XMem view workloads, browser/Wasm and GPU execution; instrumentation overhead is included in diagnostic wall times',
        'cases': cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
