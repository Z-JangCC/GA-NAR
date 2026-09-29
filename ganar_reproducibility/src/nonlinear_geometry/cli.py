"""Command-line entry point for generation, training, plotting, and status."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import get_profile
from .experiments import ExperimentRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "status", "consolidate", "plot", "analyze"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--profile", choices=("smoke", "full", "modular"), default="full")
        sub.add_argument("--root", type=Path, default=Path.cwd())
        sub.add_argument("--device", default="auto")
        if command == "run":
            sub.add_argument(
                "--families",
                nargs="*",
                help="Optional exact family names; omission runs the complete matrix",
            )
            sub.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profile = get_profile(args.profile)
    runner = ExperimentRunner(args.root, profile, device=args.device)
    if args.command == "run":
        runner.run(families=args.families, force=args.force)
        return 0
    if args.command == "status":
        print(json.dumps(runner.status(), indent=2, ensure_ascii=False))
        return 0
    if args.command == "consolidate":
        runs, quintiles = runner.consolidate()
        print(json.dumps({"runs": len(runs), "quintiles": len(quintiles)}, indent=2))
        return 0
    if args.command == "plot":
        from .plotting import generate_all_figures

        generated = generate_all_figures(
            runner.result_dir / "runs.csv",
            runner.result_dir / "quintiles.csv",
            runner.result_dir / "figures",
        )
        print(json.dumps([str(path) for path in generated], indent=2))
        return 0
    if args.command == "analyze":
        from .analysis import analyze_profile

        outputs = analyze_profile(runner.result_dir)
        print(json.dumps(outputs, indent=2, ensure_ascii=False, default=str))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
