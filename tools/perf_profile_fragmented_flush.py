#!/usr/bin/env python3
import argparse,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('baseline');p.add_argument('patch');p.add_argument('--output',required=True);a=p.parse_args()
rows=[]
for mib,frames,mode in ((8,8,'fragmented-dense-resident'),(64,16,'fragmented'),(8,16,'contiguous')):
 group=[]
 for pair in range(3):
  for label in (('baseline','patch') if pair%2==0 else ('patch','baseline')):
   r=subprocess.run([getattr(a,label),str(mib),str(frames),mode],text=True,capture_output=True,check=True)
   value=json.loads(r.stdout);line=next(l for l in r.stderr.splitlines() if l.startswith('FLUSH_PROFILE '));profile=json.loads(line[len('FLUSH_PROFILE '):]);row={'implementation':label,'pair':pair,'result':value,'profile':profile};rows.append(row);group.append(row)
 assert len({r['result']['digest'] for r in group})==1
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps({'diagnosticOnly':True,'phaseIndices':['mark_prepare','staging_alloc_zero','full_shadow_restore','guest_read','compare_emit','resource_total','sparse_restore_plus_guest_read'],'note':'phase6 includes phase3; total includes phases. Temporary instrumentation excluded from final performance evidence. Counters cover measured invocations after initial warm flush. Minor faults include untimed verification between flushes.','samples':rows},indent=2)+'\n')
