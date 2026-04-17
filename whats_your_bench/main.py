import os
import sys
import argparse
import datetime

import pandas as pd

from .problem_set import PROBLEM_CONFIGS, ParameterizedProblem

_ALL_PROBLEM_NAMES = [f"Problem{str(i+1).zfill(2)}" for i in range(len(PROBLEM_CONFIGS))]


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="wyb",
        description=(
            "What's Your Bench — benchmark PyMC, Pyro, and Stan against "
            "analytically exact posteriors."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  wyb                        run all 21 problems\n"
            "  wyb 1 2 3                  run problems 1, 2, and 3\n"
            "  wyb --list                 list available problems\n"
            "  wyb 1 --format md          write only a Markdown table\n"
            "  wyb --output-dir ./out     write results to ./out/\n"
        ),
    )
    parser.add_argument(
        "problems",
        metavar="N",
        type=int,
        nargs="*",
        help="problem numbers to run (default: all)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list available problems and exit",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        default="results",
        help="directory to write result files (default: ./results)",
    )
    parser.add_argument(
        "--format",
        choices=["md", "tex", "csv", "all"],
        default="all",
        help="output format (default: all)",
    )
    return parser


def _run_problem(name, cfg):
    p = ParameterizedProblem(cfg)
    print(f"Running {name}...", flush=True)

    p.run_models()

    retries = 0
    while True:
        try:
            p.evaluate_models()
            break
        except Exception:
            retries += 1
            if retries >= 3:
                raise RuntimeError(f"{name}: evaluate_models failed after 3 retries")

    p.results.insert(0, "Problem #", [name] * len(p.results))
    return p.results


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.list:
        print("Available problems:")
        for name in _ALL_PROBLEM_NAMES:
            print(f"  {name}")
        return

    all_problems = list(enumerate(PROBLEM_CONFIGS, start=1))

    if args.problems:
        invalid = [n for n in args.problems if n < 1 or n > len(PROBLEM_CONFIGS)]
        if invalid:
            parser.error(
                f"invalid problem number(s): {invalid}. "
                f"Valid range: 1–{len(PROBLEM_CONFIGS)}"
            )
        all_problems = [(i, cfg) for i, cfg in all_problems if i in args.problems]

    results = pd.DataFrame()
    for i, cfg in all_problems:
        name = f"Problem{str(i).zfill(2)}"
        df = _run_problem(name, cfg)
        results = pd.concat([results, df], axis=0)

    results.reset_index(drop=True, inplace=True)

    os.makedirs(args.output_dir, exist_ok=True)
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    base = os.path.join(args.output_dir, f"results_{now}")

    written = []
    if args.format in ("md", "all"):
        path = f"{base}.md"
        results.to_markdown(path)
        written.append(path)
    if args.format in ("tex", "all"):
        path = f"{base}.tex"
        results.to_latex(path)
        written.append(path)
    if args.format in ("csv", "all"):
        path = f"{base}.csv"
        results.to_csv(path)
        written.append(path)

    print(f"\nResults written to:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
