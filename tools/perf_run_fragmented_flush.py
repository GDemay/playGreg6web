#!/usr/bin/env python3
"""Fork-only matched actual Device flush benchmark driver; no game data."""
import argparse,json,os,platform,statistics,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--environment-only',action='store_true');p.add_argument('--output',required=True);p.add_argument('--baseline');p.add_argument('--candidate');p.add_argument('--case-filter',default='all');p.add_argument('--base');p.add_argument('--patch');a=p.parse_args()
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
if a.environment_only:
 cpu=next((l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),'unknown')
 value={'os':platform.platform(),'cpu':cpu,'logical_cpus':os.cpu_count(),'memory_bytes':os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES'),'compiler':subprocess.check_output(['c++','--version'],text=True).splitlines()[0],'cmake':subprocess.check_output(['cmake','--version'],text=True).splitlines()[0],'flags':'Release; PLAYGTA6_OPUS=OFF; PLAYGTA6_PERFORMANCE_BENCHMARKS=ON; -j4 builds only','baseline':a.base,'patch':a.patch}
else:
 cases=[(n,64,m) for n in (8,32,64) for m in ('fragmented','contiguous')]+[(n,f,m) for n in (8,32,64) for f,m in ((1,'fragmented-dense'),(4,'fragmented-dense-resident'))]+[(2,4,'fragmented-unaligned')]
 if a.case_filter=='bounded': cases=[(8,8,'fragmented-dense-resident'),(64,16,'fragmented'),(8,16,'contiguous')]
 samples=[];summary=[]
 for mib,frames,mode in cases:
  for binary in (a.baseline,a.candidate):subprocess.check_output([binary,str(mib),'2',mode],text=True)
  group=[]
  for pair in range(7):
   for label in (('baseline','patch') if pair%2==0 else ('patch','baseline')):
    binary=a.baseline if label=='baseline' else a.candidate
    row=json.loads(subprocess.check_output([binary,str(mib),str(frames),mode],text=True));row.update(implementation=label,pair=pair);samples.append(row);group.append(row)
   out.write_text(json.dumps({'sampling':'7 alternating pairs; one separate untimed 2-flush invocation per binary/case; writes, hashing and packet verification excluded from timed Device flush','samples':samples,'summary':summary},indent=2)+'\n')
  assert len({r['digest'] for r in group})==1,(mib,mode,'digest mismatch')
  assert len({r['compared_bytes'] for r in group})==1,(mib,mode,'counter mismatch')
  s={'mib':mib,'frames':frames,'mode':mode}
  for label in ('baseline','patch'):
   xs=[r['wall_ms']/frames for r in group if r['implementation']==label];s[label]={'median_ms_per_flush':statistics.median(xs),'min_ms_per_flush':min(xs),'max_ms_per_flush':max(xs)}
  summary.append(s);print(json.dumps(s),flush=True)
 value={'sampling':'7 alternating pairs; one separate untimed 2-flush invocation per binary/case; writes, hashing and packet verification excluded from timed Device flush','samples':samples,'summary':summary}
out.write_text(json.dumps(value,indent=2)+'\n')
