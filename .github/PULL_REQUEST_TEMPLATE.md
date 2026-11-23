## Summary
- **Track:** <!-- recomp / os / gpu / shaders / web / abi -->
- **Task ID:** <!-- e.g. R1, O2, G1, S1, W1 from docs/WORKSTREAMS.md -->
- **What changed:**

## Verification
- [ ] `cargo test --workspace`
- [ ] `python -m unittest discover -s tests -v`
- [ ] `node --check` on modified `web/` JS files
- [ ] `ctest` on `runtime/` (if C++ changed)

## Safety Checklist
- [ ] Confirmed **zero** copyrighted game files, binaries, keys, `.rpf`/`.xvd`/`.ttpf`/`.awc`, extracted assets, dumps, or `.env` are included.
- [ ] Did not bloat `README.md` unnecessarily.
