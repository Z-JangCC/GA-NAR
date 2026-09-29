from __future__ import annotations
import json
import csv
from pathlib import Path
import numpy as np

SETTINGS=[('ieee118_acpf','relu'),('ieee118_acpf','gelu'),('ieee118_acpf','swiglu'),('kuramoto64','relu'),('kuramoto64','gelu'),('nonlinear_heat_fem','relu'),('nonlinear_heat_fem','softplus')]

def vals(system,model):
 out=[]
 for s in range(3):
  f=Path('local_training_results')/system/model/str(s)/'metrics.json'
  if f.exists(): out.append(json.loads(f.read_text())['mse_std'])
 return np.asarray(out)

def main():
 physics=json.loads(Path('local_training_results/physics_satisfaction.json').read_text())
 pidx={(r['system'],r['model'],r['seed']):r for r in physics}
 rows=[]
 for system,a in SETTINGS:
  names={'Original Baseline':'pm_'+a,'Capacity Control':'capacity_'+a,'Generic Geometry Control':'generic_geometry_'+a,'Full GA-NAR':'ganar_'+a}
  data={}
  for label,m in names.items():
   v=vals(system,m); data[label]={'model':m,'n':len(v),'mse_by_seed':v.tolist(),'mean':float(v.mean()) if len(v) else None,'sd':float(v.std(ddof=1)) if len(v)>1 else None}
   prs=[pidx[(system,m,s)] for s in range(3)]
   for metric in ('residual_norm_mean','residual_norm_p95','residual_norm_max','high_nonlinearity_residual_mean'):
    pv=np.asarray([x[metric] for x in prs]); data[label][metric+'_by_seed']=pv.tolist(); data[label][metric+'_mean']=float(pv.mean()); data[label][metric+'_sd']=float(pv.std(ddof=1))
  base=data['Original Baseline']['mean']; full=data['Full GA-NAR']['mean']
  for label in ('Capacity Control','Generic Geometry Control','Full GA-NAR'):
   data[label]['relative_improvement_vs_baseline']=None if base is None else float((base-data[label]['mean'])/base)
   # Seed-paired improvement is the primary comparison statistic.  It avoids
   # conflating a method effect with a different seed composition.
   if len(v:=data[label]['mse_by_seed']) == len(data['Original Baseline']['mse_by_seed']):
    b=np.asarray(data['Original Baseline']['mse_by_seed']); c=np.asarray(v)
    pair=(b-c)/b
    data[label]['paired_mse_improvement_mean']=float(pair.mean())
    data[label]['paired_mse_improvement_sd']=float(pair.std(ddof=1))
    b=np.asarray(data['Original Baseline']['residual_norm_mean_by_seed'])
    c=np.asarray(data[label]['residual_norm_mean_by_seed'])
    pair=(b-c)/b
    data[label]['paired_physics_improvement_mean']=float(pair.mean())
    data[label]['paired_physics_improvement_sd']=float(pair.std(ddof=1))
  method_labels=list(names.keys())
  data['ordering_by_mean']=sorted(method_labels,key=lambda k:data[k]['mean'] if data[k]['mean'] is not None else float('inf'))
  data['ordering_by_physics_residual']=sorted(method_labels,key=lambda k:data[k]['residual_norm_mean_mean'])
  rows.append({'system':system,'activation':a,'methods':data})
 Path('local_training_results/final_ga_ablation_summary.json').write_text(json.dumps(rows,indent=2))
 per_seed_path=Path('local_training_results/final_ga_ablation_per_seed.csv')
 with per_seed_path.open('w',newline='') as handle:
  fields=['system','activation','method','model','seed','mse_std','residual_rms',
          'residual_norm_mean','residual_norm_p95','residual_norm_max',
          'high_nonlinearity_residual_mean','admissible_fraction']
  writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader()
  for r in rows:
   for label in ('Original Baseline','Capacity Control','Generic Geometry Control','Full GA-NAR'):
    model=r['methods'][label]['model']
    for seed in range(3):
     pr=pidx[(r['system'],model,seed)]
     writer.writerow({'system':r['system'],'activation':r['activation'],'method':label,
                      'model':model,'seed':seed,
                      'mse_std':r['methods'][label]['mse_by_seed'][seed],
                      **{k:pr[k] for k in fields[6:]}})
 improvement_path=Path('local_training_results/final_ga_ablation_improvements_per_seed.csv')
 with improvement_path.open('w',newline='') as handle:
  fields=['system','activation','method','model','seed','baseline_mse','method_mse',
          'paired_mse_improvement','baseline_residual_norm_mean',
          'method_residual_norm_mean','paired_physics_improvement']
  writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader()
  for r in rows:
   baseline=r['methods']['Original Baseline']
   for label in ('Capacity Control','Generic Geometry Control','Full GA-NAR'):
    method=r['methods'][label]
    for seed in range(3):
     bm=baseline['mse_by_seed'][seed]; mm=method['mse_by_seed'][seed]
     bp=baseline['residual_norm_mean_by_seed'][seed]
     mp=method['residual_norm_mean_by_seed'][seed]
     writer.writerow({'system':r['system'],'activation':r['activation'],
                      'method':label,'model':method['model'],'seed':seed,
                      'baseline_mse':bm,'method_mse':mm,
                      'paired_mse_improvement':(bm-mm)/bm,
                      'baseline_residual_norm_mean':bp,
                      'method_residual_norm_mean':mp,
                      'paired_physics_improvement':(bp-mp)/bp})
 lines=['# Final GA-NAR Four-way Controlled Ablation','','The four methods are Original Baseline, Capacity Control, Generic Geometry Control, and Full GA-NAR. Capacity and generic controls reuse the correction-branch budget; only Full GA-NAR uses structured layer-wise geometry modulation. All entries below aggregate the same three seeds (0, 1, 2); values are mean ± sample SD.','', '## Prediction MSE','', '| System | Activation | Baseline | Capacity | Generic geometry | Full GA-NAR | Mean ordering |','|---|---|---:|---:|---:|---:|---|']
 for r in rows:
  d=r['methods']; lines.append(f"| {r['system']} | {r['activation']} | {d['Original Baseline']['mean']:.8g} ± {d['Original Baseline']['sd']:.3g} | {d['Capacity Control']['mean']:.8g} ± {d['Capacity Control']['sd']:.3g} | {d['Generic Geometry Control']['mean']:.8g} ± {d['Generic Geometry Control']['sd']:.3g} | {d['Full GA-NAR']['mean']:.8g} ± {d['Full GA-NAR']['sd']:.3g} | {' < '.join(d['ordering_by_mean'])} |")
 lines += ['', '### Paired MSE improvement versus Original Baseline', '', '| System | Activation | Capacity | Generic geometry | Full GA-NAR |','|---|---|---:|---:|---:|']
 for r in rows:
  d=r['methods']; lines.append(f"| {r['system']} | {r['activation']} | {100*d['Capacity Control']['paired_mse_improvement_mean']:.3f}% ± {100*d['Capacity Control']['paired_mse_improvement_sd']:.3f}% | {100*d['Generic Geometry Control']['paired_mse_improvement_mean']:.3f}% ± {100*d['Generic Geometry Control']['paired_mse_improvement_sd']:.3f}% | {100*d['Full GA-NAR']['paired_mse_improvement_mean']:.3f}% ± {100*d['Full GA-NAR']['paired_mse_improvement_sd']:.3f}% |")
 lines += ['', '## Physical-equation residual mean', '', '| System | Activation | Baseline | Capacity | Generic geometry | Full GA-NAR | Residual ordering |','|---|---|---:|---:|---:|---:|---|']
 for r in rows:
  d=r['methods']; lines.append(f"| {r['system']} | {r['activation']} | {d['Original Baseline']['residual_norm_mean_mean']:.8g} ± {d['Original Baseline']['residual_norm_mean_sd']:.3g} | {d['Capacity Control']['residual_norm_mean_mean']:.8g} ± {d['Capacity Control']['residual_norm_mean_sd']:.3g} | {d['Generic Geometry Control']['residual_norm_mean_mean']:.8g} ± {d['Generic Geometry Control']['residual_norm_mean_sd']:.3g} | {d['Full GA-NAR']['residual_norm_mean_mean']:.8g} ± {d['Full GA-NAR']['residual_norm_mean_sd']:.3g} | {' < '.join(d['ordering_by_physics_residual'])} |")
 lines += ['', '### Paired physical-residual improvement versus Original Baseline', '', '| System | Activation | Capacity | Generic geometry | Full GA-NAR |','|---|---|---:|---:|---:|']
 for r in rows:
  d=r['methods']; lines.append(f"| {r['system']} | {r['activation']} | {100*d['Capacity Control']['paired_physics_improvement_mean']:.3f}% ± {100*d['Capacity Control']['paired_physics_improvement_sd']:.3f}% | {100*d['Generic Geometry Control']['paired_physics_improvement_mean']:.3f}% ± {100*d['Generic Geometry Control']['paired_physics_improvement_sd']:.3f}% | {100*d['Full GA-NAR']['paired_physics_improvement_mean']:.3f}% ± {100*d['Full GA-NAR']['paired_physics_improvement_sd']:.3f}% |")
 lines += ['', '### Additional residual diagnostics', '', '| System | Activation | Method | P95 | Maximum | High-nonlinearity mean |','|---|---|---|---:|---:|---:|']
 for r in rows:
  d=r['methods']
  for label in ('Original Baseline','Capacity Control','Generic Geometry Control','Full GA-NAR'):
   m=d[label]
   lines.append(f"| {r['system']} | {r['activation']} | {label} | {m['residual_norm_p95_mean']:.8g} ± {m['residual_norm_p95_sd']:.3g} | {m['residual_norm_max_mean']:.8g} ± {m['residual_norm_max_sd']:.3g} | {m['high_nonlinearity_residual_mean_mean']:.8g} ± {m['high_nonlinearity_residual_mean_sd']:.3g} |")
 lines += ['', 'All 84 individual runs and every residual field are available in `local_training_results/final_ga_ablation_per_seed.csv`; all 63 seed-paired method improvements are in `local_training_results/final_ga_ablation_improvements_per_seed.csv`; the nested summary including all seed vectors is in `local_training_results/final_ga_ablation_summary.json`. The residuals are recomputed from the independent Kuramoto, PYPOWER IEEE-118 AC power-flow, and triangular FEM heat equations; they are not training losses.', '', '## Interpretation', '', 'The intended causal decomposition is capacity → generic geometry information → structured GA-NAR mechanism. The report uses observed ordering only; it does not assume Full GA-NAR must win before inspecting results.', '']
 Path('FINAL_GA_NAR_ABLATION_REPORT.md').write_text('\n'.join(lines))
 print(json.dumps(rows,indent=2))

if __name__=='__main__': main()
