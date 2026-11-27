#!/usr/bin/env python3
"""Matched public Naga optimizer timings with complete WGSL/stats oracles.

Build the identical shaders/examples/bench_live_local_analysis.rs against both revisions.
"""
import argparse
import hashlib
import json
import pathlib
import platform
import statistics
import subprocess
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('before')
parser.add_argument('after')
parser.add_argument('--samples', type=int, default=7)
parser.add_argument('--baseline-commit', required=True)
parser.add_argument('--patch-commit', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
if args.samples < 3:
    parser.error('at least 3 pairs required')


def run(binary, kind, operations, output=None):
    command = [binary, kind, str(operations), '3' if output is None else '1']
    if output is not None:
        command.append(str(output))
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(rows) == (3 if output is None else 1)
    assert len({(row['stats'], row['output_fnv64']) for row in rows}) == 1
    return rows


cases = []
for kind in ['live', 'aggregate', 'forwarded', 'no-local', 'sparse-live', 'dead-store']:
    for operations in [256, 1024, 4096]:
        with tempfile.TemporaryDirectory() as temporary:
            before_path = pathlib.Path(temporary) / 'before.wgsl'
            after_path = pathlib.Path(temporary) / 'after.wgsl'
            before = run(args.before, kind, operations, before_path)
            after = run(args.after, kind, operations, after_path)
            assert before_path.read_bytes() == after_path.read_bytes(), (kind, operations, 'WGSL differs')
            assert before[0]['stats'] == after[0]['stats'], (kind, operations, 'stats differ')
            output_sha256 = hashlib.sha256(before_path.read_bytes()).hexdigest()
        samples = {'before': [], 'after': []}
        binaries = {'before': args.before, 'after': args.after}
        for index in range(args.samples):
            order = ['before', 'after'] if index % 2 == 0 else ['after', 'before']
            for variant in order:
                samples[variant].append(run(binaries[variant], kind, operations))
        assert len({
            (row['stats'], row['output_fnv64'])
            for batches in samples.values() for batch in batches for row in batch
        }) == 1
        medians = {
            variant: statistics.median(statistics.median(row['wall_ms'] for row in batch) for batch in batches)
            for variant, batches in samples.items()
        }
        change = (medians['after'] / medians['before'] - 1) * 100
        print(kind, operations, round(medians['before'], 4), round(medians['after'], 4), round(change, 2), flush=True)
        cases.append({'kind': kind, 'operations': operations, 'output_sha256': output_sha256, 'samples': samples})
        evidence = {
            'baselineCommit': args.baseline_commit,
            'patchCommit': args.patch_commit,
            'complete': len(cases) == 18,
            'expectedCases': 18,
            'scope': ('synthetic native production playgta6_shaders::ir_opt::optimize_module; whole public optimizer; '
                      f'one untimed warmup then 3 timed clones per fresh process; {args.samples} alternating process pairs; '
                      'clone/parse/validation/emission/hash outside timing'),
            'summaryMetric': f'median across {args.samples} process medians of 3 whole-optimizer calls; all values retained',
            'unmeasured': ('real shader distribution/hit rate, end-to-end RDNA translation and driver compilation, '
                           'cold initialization, GPU execution, browser/Wasm and game FPS'),
            'environment': {'system': platform.system(), 'release': platform.release(), 'arch': platform.machine(),
                            'rustc': subprocess.check_output(['rustc', '--version'], text=True).strip(),
                            'profile': 'opt-level=3; lto=thin; codegen-units=1'},
            'cases': cases,
        }
        pathlib.Path(args.output).write_text(json.dumps(evidence, indent=2) + '\n')
