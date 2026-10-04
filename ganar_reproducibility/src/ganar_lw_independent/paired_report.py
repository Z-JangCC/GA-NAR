from __future__ import annotations

import json
from pathlib import Path
import numpy as np


ACTIVATIONS = ["relu", "gelu", "silu", "mish", "softplus"]
SYSTEMS = ["ieee118_acpf", "kuramoto64", "nonlinear_heat_fem"]


def values(system, model):
    out=[]
    for s in (0,1,2):
        f=Path("local_training_results")/system/model/str(s)/"metrics.json"
        if f.exists(): out.append(json.loads(f.read_text())["mse_std"])
    return np.asarray(out)


def main():
    rows=[]
    for system in SYSTEMS:
        for activation in ACTIVATIONS:
            b=values(system,f"pm_{activation}"); g=values(system,f"ganar_{activation}")
            if len(b)==len(g)==3:
                d=g-b; rows.append({"system":system,"activation":activation,
                    "baseline_mean":float(b.mean()),"composed_mean":float(g.mean()),
                    "delta_mean":float(d.mean()),"relative_improvement":float(np.mean((b-g)/b)),
                    "all_seeds_win":bool(np.all(d<0)),"delta_by_seed":d.tolist()})
        if system == "kuramoto64":
            b=values(system,"pm_silu"); g=values(system,"ganar_lw")
            if len(b)==len(g)==3:
                d=g-b; rows.append({"system":system,"activation":"silu_anchor",
                    "baseline_mean":float(b.mean()),"composed_mean":float(g.mean()),
                    "delta_mean":float(d.mean()),"relative_improvement":float(np.mean((b-g)/b)),
                    "all_seeds_win":bool(np.all(d<0)),"delta_by_seed":d.tolist()})
        b=values(system,"pm_swiglu"); g=values(system,"ganar_swiglu")
        if len(b)==len(g)==3:
            d=g-b; rows.append({"system":system,"activation":"swiglu",
                "baseline_mean":float(b.mean()),"composed_mean":float(g.mean()),
                "delta_mean":float(d.mean()),"relative_improvement":float(np.mean((b-g)/b)),
                "all_seeds_win":bool(np.all(d<0)),"delta_by_seed":d.tolist()})
    Path("local_training_results/paired_activation_summary.json").write_text(json.dumps(rows,indent=2))
    lines=["# Paired Baseline vs Baseline + GA-NAR-LW", "",
           "Every row uses identical system, split, seed, batch order, epoch budget, and checkpoint rule.","",
           "| System | Activation | Baseline MSE | Composed MSE | Relative improvement | All seeds win |", "|---|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['system']} | {r['activation']} | {r['baseline_mean']:.8g} | {r['composed_mean']:.8g} | {r['relative_improvement']:.2%} | {r['all_seeds_win']} |")
    Path("PAIRED_GA_NAR_LW_ACTIVATION_REPORT.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(rows,indent=2))


if __name__ == "__main__": main()
