#!/usr/bin/env python3
from __future__ import annotations
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/ga_nar_lw_controlled_ablation/final_ga_ablation_summary.json'
OUTPUT = ROOT / 'reports/ga_nar_lw_controlled_ablation.md'

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(ROOT / 'reports/ga_nar_lw_controlled_ablation.md'))
    args = parser.parse_args()
    rows = json.loads(SOURCE.read_text())
    lines = ['# GA-NAR-LW controlled ablation', '', '| System | Activation | Baseline | Capacity | Generic geometry | Full GA-NAR |', '|---|---|---:|---:|---:|---:|']
    for row in rows:
        methods = row['methods']
        lines.append('| {} | {} | {:.8g} | {:.8g} | {:.8g} | {:.8g} |'.format(row['system'], row['activation'], methods['Original Baseline']['mean'], methods['Capacity Control']['mean'], methods['Generic Geometry Control']['mean'], methods['Full GA-NAR']['mean']))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(lines) + '\n')
    print(output)

if __name__ == '__main__':
    main()
