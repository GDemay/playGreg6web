import argparse,json,subprocess,platform,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('before');p.add_argument('after');p.add_argument('--output',required=True);a=p.parse_args()
results=[]
for workload in [(4096,64,80,'index'),(1024,64,80,'index'),(4096,64,0,'index'),(128,16,16,'full')]:
 print('Measuring',workload,flush=True)
 def run(exe):
  r=subprocess.run([exe,*map(str,workload)],text=True,capture_output=True)
  if r.returncode:raise RuntimeError('benchmark returned '+str(r.returncode)+': '+r.stderr)
  x=json.loads(r.stdout.strip().splitlines()[-1]);assert x['correct'];return x
 for exe in [a.before,a.after]:run(exe)
 samples={'before':[],'after':[]}
 for i in range(7):
  for key in (['before','after'] if i%2==0 else ['after','before']):samples[key].append(run(getattr(a,key)))
 assert len({x['packet_digest'] for group in samples.values() for x in group})==1,'packet digest mismatch'
 results.append({'arguments':workload,'samples':samples})
metadata=json.loads(Path('runner-environment.json').read_text())
output={'benchmark':'synthetic actual native Device::flush_cpu_buffers, Windows write tracking','environment':metadata,'sampling':'separate Release MSVC binaries; setup, dirty writes and correctness outside timed flush; one untimed invocation each then seven alternating A/B samples; each sample totals64 (or16control) actual flushes; CPU GetProcessTimes user+kernel and elapsed wall separately','unmeasured':'real-game frame rate, real guest writes, browser/Wasm path, GPU execution; production profile motivates buffer/page/hull counts but fixture is synthetic','cases':results}
Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(output,indent=2)+'\n')
