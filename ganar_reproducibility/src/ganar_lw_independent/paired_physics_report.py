from __future__ import annotations

import json
from pathlib import Path
import numpy as np


ACTIVATIONS = ["relu", "gelu", "silu", "mish", "softplus", "swiglu"]
SYSTEMS = ["ieee118_acpf", "kuramoto64", "nonlinear_heat_fem"]


def main():
    rows=json.loads(Path("local_training_results/physics_satisfaction.json").read_text())
    index={(r["system"],r["model"],r["seed"]):r for r in rows if r.get("status") != "failed"}
    paired=[]
    for system in SYSTEMS:
        for act in ACTIVATIONS:
            b=[]; g=[]
            for seed in (0,1,2):
                b.append(index.get((system,"pm_"+act,seed)))
                g.append(index.get((system,"ganar_"+act,seed)))
            if not all(b) or not all(g):
                continue
            result={"system":system,"activation":act}
            for metric in ("residual_rms","residual_norm_mean","residual_norm_p95","residual_norm_max","high_nonlinearity_residual_mean"):
                bv=np.array([r[metric] for r in b]); gv=np.array([r[metric] for r in g])
                result[metric+"_baseline_mean"]=float(bv.mean())
                result[metric+"_ganar_mean"]=float(gv.mean())
                result[metric+"_delta"]=float((gv-bv).mean())
                result[metric+"_all_seeds_lower"]=bool(np.all(gv < bv))
            paired.append(result)
    Path("local_training_results/paired_physics_summary.json").write_text(json.dumps(paired,indent=2))
    lines=["# Paired Physical Equation Satisfaction: Baseline vs Baseline + GA-NAR-LW","",
           "All values are independently re-evaluated equation residuals on the same test samples. Heat uses the exact registered triangular FEM residual. Residual norm is the equation-residual L2 norm divided by sqrt(number of equations); high-nonlinearity means the top quartile of input norm.","",
           "## Mean residual by activation-system pair (mean over seeds)", "",
           "| System | Activation | PM residual mean | GA residual mean | Δ(GA−PM) | PM high-nonlinearity | GA high-nonlinearity | GA lower all seeds |", "|---|---|---:|---:|---:|---:|---:|:---:|"]
    for r in paired:
        lines.append(f"| {r['system']} | {r['activation']} | {r['residual_norm_mean_baseline_mean']:.8g} | {r['residual_norm_mean_ganar_mean']:.8g} | {r['residual_norm_mean_delta']:.8g} | {r['high_nonlinearity_residual_mean_baseline_mean']:.8g} | {r['high_nonlinearity_residual_mean_ganar_mean']:.8g} | {r['residual_norm_mean_all_seeds_lower']} |")
    lines += ["", "## Full per-seed equation residuals", "",
              "Each cell is `PM / GA-NAR-LW`; values are residual norm means unless the metric is named otherwise.", ""]
    for system in SYSTEMS:
        lines += [f"### {system}", "",
                  "| Activation | Seed | Residual mean PM / GA | P95 PM / GA | Max PM / GA | High-nonlinearity mean PM / GA | Residual RMS PM / GA |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for act in ACTIVATIONS:
            for seed in (0, 1, 2):
                b=index[(system,"pm_"+act,seed)]; g=index[(system,"ganar_"+act,seed)]
                lines.append(f"| {act} | {seed} | {b['residual_norm_mean']:.8g} / {g['residual_norm_mean']:.8g} | {b['residual_norm_p95']:.8g} / {g['residual_norm_p95']:.8g} | {b['residual_norm_max']:.8g} / {g['residual_norm_max']:.8g} | {b['high_nonlinearity_residual_mean']:.8g} / {g['high_nonlinearity_residual_mean']:.8g} | {b['residual_rms']:.8g} / {g['residual_rms']:.8g} |")
        lines.append("")
    Path("PAIRED_PHYSICS_SATISFACTION_REPORT.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"paired_rows":len(paired),"failed_rows":sum(r.get('status')=='failed' for r in rows)},indent=2))


if __name__ == "__main__": main()
