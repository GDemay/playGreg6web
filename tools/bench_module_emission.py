#!/usr/bin/env python3
"""Matched synthetic public emit_module timings and requested allocations.
Build recomp/examples/bench_module_emission.rs identically against both revisions.
No lifting, filesystem writes, output hashing or LLVM compilation is timed.
"""
import argparse
import hashlib
import json
import platform
import subprocess
import tempfile
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('before')
p.add_argument('after')
p.add_argument('--baseline-commit', required=True)
p.add_argument('--patch-commit', required=True)
p.add_argument('--samples', type=int, default=7)
a = p.parse_args()
if a.samples < 3:
    p.error('at least 3 samples required')
binaries = {'before': a.before, 'after': a.after}
cases = []
for kind, mib in [('dispatch-long', 0), ('dispatch-many', 0), ('dispatch-many', 40), ('helper-dispatch-many', 40), ('dispatch-long-many', 40), ('dispatch-skew', 40), ('raw-dispatch', 40), ('dispatch-control', 0), ('helper-many', 40), ('control-many', 40), ('skew', 40), ('helper', 40), ('raw', 40), ('crlf', 40), ('control', 0)]:
    samples = {'before': [], 'after': []}
    with tempfile.TemporaryDirectory() as tmp:
        outputs = []
        for label in ['before', 'after']:
            path = Path(tmp) / (label + '.ll')
            subprocess.run([binaries[label], kind, str(mib), str(path)], capture_output=True, check=True)
            outputs.append(path.read_bytes())
        if outputs[0] != outputs[1]:
            raise RuntimeError('byte-for-byte output mismatch: ' + kind)
        sha256 = hashlib.sha256(outputs[0]).hexdigest()
    for i in range(a.samples):
        for label in (['before', 'after'] if i % 2 == 0 else ['after', 'before']):
            result = json.loads(subprocess.run([binaries[label], kind, str(mib)], capture_output=True, text=True, check=True).stdout)
            samples[label].append(result)
    if len({s['output_fnv64'] for values in samples.values() for s in values}) != 1:
        raise RuntimeError('sample output hash mismatch: ' + kind)
    cases.append({'kind': kind, 'fixture_mib': mib, 'output_sha256': sha256, 'samples': samples})
print(json.dumps({'benchmark': 'synthetic native Rust public emit_module', 'baselineCommit': a.baseline_commit,
                  'patchCommit': a.patch_commit, 'environment': {'system': platform.system(), 'release': platform.release(), 'arch': platform.machine()},
                  'sampling': 'separate prebuilt binaries; alternating order; fresh process each sample; untimed byte-for-byte complete output oracle per case',
                  'allocationMetric': 'peak live allocation requested sizes tracked through System allocator wrapper; realloc net requested size; excludes fixtures; not RSS or allocator internals',
                  'unmeasured': 'real title module sizes/distribution, LLVM compile/link time, browser/Wasm execution and game FPS', 'cases': cases}, indent=2))
