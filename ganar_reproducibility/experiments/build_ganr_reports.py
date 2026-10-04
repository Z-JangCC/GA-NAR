#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/ganr_shared_physics_factorial/runs.csv'
OUTPUT = ROOT / 'reports/ganr_shared_physics_factorial_summary.json'

def main() -> None:
    runs = pd.read_csv(SOURCE)
    selected = runs[(runs.family == 'all_physics_factorial') & runs.system.isin(['controlled','duffing','ieee118','allen_cahn','shallow_water'])]
    summary = []
    for (system, model, variant), group in selected.groupby(['system','model','variant']):
        summary.append({'system': system, 'model': model, 'condition': variant, 'seed_count': int(len(group)), 'nrmse_mean': float(group.nrmse.mean()), 'nrmse_sd': float(group.nrmse.std(ddof=1))})
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(summary, indent=2) + '\n')
    print(OUTPUT)

if __name__ == '__main__':
    main()
