#!/usr/bin/env python3
"""Temporary source attribution only; never apply to a product branch."""
import sys
from pathlib import Path
root=Path(sys.argv[1]);p=root/'runtime/gpu/d3d12_guest_resources.cpp';s='#include <sys/resource.h>\n'+p.read_text()
start=s.index('uint64_t Resource::flush_shadow(');end=s.index('\nHResult Resource::Map(',start)
f=s[start:end]
head='''
namespace {
struct FragmentProbe {
 bool enabled = false;
 uint64_t ns[7]{}, calls[7]{}, sparse = 0, bulk = 0, candidates = 0, flushes = 0;
} fragment_probe;
rusage fragment_usage_begin{};
struct FragmentTimer {
 unsigned phase; bool on;
 std::chrono::steady_clock::time_point begin;
 explicit FragmentTimer(unsigned p) : phase(p), on(fragment_probe.enabled) {
  if(on) begin=std::chrono::steady_clock::now();
 }
 ~FragmentTimer() { if(on) {
  fragment_probe.ns[phase]+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-begin).count();
  ++fragment_probe.calls[phase];
 }}
};
}
extern "C" void pg6_fragment_profile_reset() { fragment_probe={}; fragment_probe.enabled=true; getrusage(RUSAGE_SELF,&fragment_usage_begin); }
extern "C" void pg6_fragment_profile_dump() {
 fragment_probe.enabled=false;
 rusage usage_end{}; getrusage(RUSAGE_SELF,&usage_end);
 std::fprintf(stderr,"FLUSH_PROFILE {\\\"flushes\\\":%llu,\\\"sparse\\\":%llu,\\\"bulk\\\":%llu,\\\"candidates\\\":%llu,\\\"ns\\\":[",
 (unsigned long long)fragment_probe.flushes,(unsigned long long)fragment_probe.sparse,
 (unsigned long long)fragment_probe.bulk,(unsigned long long)fragment_probe.candidates);
 for(unsigned i=0;i<7;++i) std::fprintf(stderr,"%s%llu",i?",":"",(unsigned long long)fragment_probe.ns[i]);
 std::fprintf(stderr,"],\\\"calls\\\":[");
 for(unsigned i=0;i<7;++i) std::fprintf(stderr,"%s%llu",i?",":"",(unsigned long long)fragment_probe.calls[i]);
 std::fprintf(stderr,"],\\\"minor_faults\\\":%ld}\\n",usage_end.ru_minflt-fragment_usage_begin.ru_minflt);
}
'''
# Correct C++ escaping from the Python literal above.
head=head.replace('\\"','\\"')
# 0 mark preparation; 1 staging allocation/zero; 2 shadow restore; 3 guest read;
# 4 compare+emit; 5 total; 6 sparse page restore/read setup (loop overhead).
f=f.replace('    const uint64_t size = shadow_size_;','    FragmentTimer total_probe(5);\n    if(fragment_probe.enabled) ++fragment_probe.flushes;\n    const uint64_t size = shadow_size_;',1)
if 'auto prepare_window = [&] {' in f:
 f=f.replace('auto prepare_window = [&] {','auto prepare_window = [&] {\n        FragmentTimer mark_probe(0);',1)
else:
 f=f.replace('if (!tracked && os::guest_window::translated(cpu_va_)) {','if (!tracked && os::guest_window::translated(cpu_va_)) {\n        FragmentTimer mark_probe(0);',1)
f=f.replace('        copy.assign(static_cast<size_t>(size), 0);','        if(fragment_probe.enabled) ++fragment_probe.bulk;\n        { FragmentTimer alloc_probe(1); copy.assign(static_cast<size_t>(size), 0); }',1)
def wrap_block(text,marker,phase):
 a=text.index(marker);opening=text.index('{',a);depth=1;i=opening+1
 while depth:
  depth+=(text[i]=='{')-(text[i]=='}');i+=1
 return text[:a]+'{ FragmentTimer phase_probe('+str(phase)+');\n'+text[a:i]+'\n}\n'+text[i:]
f=wrap_block(f,'        for (size_t i = 0; i < shadow_chunks_.size(); ++i)',2)
f=wrap_block(f,'        if (!com::read_guest(cpu_va_, copy.data(), copy.size()))',3)
if 'sparse_copy = std::make_unique_for_overwrite' in f:
 f=f.replace('            sparse_copy = std::make_unique_for_overwrite<uint8_t[]>(static_cast<size_t>(size));','            if(fragment_probe.enabled) ++fragment_probe.sparse;\n            { FragmentTimer alloc_probe(1); sparse_copy = std::make_unique_for_overwrite<uint8_t[]>(static_cast<size_t>(size)); }',1)
 f=f.replace('                const uint8_t* shadow = shadow_chunks_[off / kShadowChunk].get();','                FragmentTimer setup_probe(6);\n                const uint8_t* shadow = shadow_chunks_[off / kShadowChunk].get();',1)
 f=f.replace('                com::read_guest(cpu_va_ + off, dst, n);','                { FragmentTimer read_probe(3); com::read_guest(cpu_va_ + off, dst, n); }',1)
f=f.replace('    // Verification:', '    if(fragment_probe.enabled) fragment_probe.candidates += std::count(candidate.begin(),candidate.end(),true);\n    FragmentTimer compare_probe(4);\n    // Verification:',1)
p.write_text(s[:start]+head+f+s[end:])
b=root/'tests/bench_fragmented_buffer_flush.cpp';t=b.read_text();t=t.replace('int main(', 'extern "C" void pg6_fragment_profile_reset();\nextern "C" void pg6_fragment_profile_dump();\nint main(',1).replace(' double ms=0;', ' pg6_fragment_profile_reset();\n double ms=0;',1).replace(' std::printf(', ' pg6_fragment_profile_dump();\n std::printf(',1);b.write_text(t)
