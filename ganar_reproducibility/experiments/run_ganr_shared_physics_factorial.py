"""Run all models with/without the common physics linearization skip."""
from __future__ import annotations
import argparse
import os, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
worker=r'''
import os
from nonlinear_geometry.config import get_profile
from nonlinear_geometry.experiments import ExperimentRunner, enumerate_jobs
import sys
r=ExperimentRunner(os.environ['GANR_REPRODUCTION_ROOT'],get_profile('full'),device=os.environ['GANR_REPRODUCTION_DEVICE'])
s,m,v,seed=sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4])
for j in enumerate_jobs(r.profile):
    if j.family=='all_physics_factorial' and j.system==s and j.model==m and j.variant==v and j.seed==seed:
        r.run_job(j,force=True)
'''
def main() -> None:
 parser = argparse.ArgumentParser()
 parser.add_argument('--output-root', default='local_reproduction_results/ganr')
 parser.add_argument('--device', default='cuda:0')
 args = parser.parse_args()
 root = Path(args.output_root).resolve()
 root.mkdir(parents=True, exist_ok=True)
 jobs=[(s,m,v,seed) for s in ('controlled','ac_power','duffing','ieee118','allen_cahn','shallow_water')
      for m in ('relu','gelu','silu','swiglu','resmlp','fourier','ganr')
      for v in ('no_physics','physics') for seed in (11,29,47)]
 for start in range(0,len(jobs),8):
    ps=[]
    for i,job in enumerate(jobs[start:start+8]):
        e=os.environ.copy(); e['PYTHONPATH']=str(ROOT/'src'); e['CUDA_VISIBLE_DEVICES']=str(i)
        e['GANR_REPRODUCTION_ROOT'] = str(root)
        e['GANR_REPRODUCTION_DEVICE'] = args.device
        ps.append(subprocess.Popen([sys.executable,'-c',worker,*map(str,job)],cwd=ROOT,env=e))
    codes=[p.wait() for p in ps]
    if any(c for c in codes): raise SystemExit(codes)

if __name__ == '__main__':
 main()
