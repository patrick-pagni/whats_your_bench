import problem_set
from problem_set import PROBLEM_CONFIGS, ParameterizedProblem
import os
import pandas as pd
import datetime
import argparse

parser = argparse.ArgumentParser(description="CLI for what's your bench")
parser.add_argument("problems", metavar="problems", type=int, nargs="+", help="Problems to be run by benchmarking suite")
args = parser.parse_args()

# Build the full ordered list of problems from the config table.
all_problems = [
    (f"Problem{str(i+1).zfill(2)}", cfg)
    for i, cfg in enumerate(PROBLEM_CONFIGS)
]

if args.problems:
    all_problems = [(name, cfg) for i, (name, cfg) in enumerate(all_problems) if i + 1 in args.problems]

results = pd.DataFrame()

for name, cfg in all_problems:

    p = ParameterizedProblem(cfg)
    print(name)

    p.run_models()
    success = False
    retries = 0

    while not success:
        try:
            p.evaluate_models()
            success = True
        except:
            if retries < 3:
                retries += 1
            else:
                raise RuntimeError

    p.results.insert(loc=0, column="Problem #", value=[name] * p.results.shape[0])
    results = pd.concat([results, p.results], axis=0)

results.reset_index(drop=True, inplace=True)

path = "../results"
os.makedirs(path, exist_ok=True)

now = datetime.datetime.now()
results.to_markdown(f"{path}/results_{now}.md")
results.to_latex(f"{path}/results_{now}.tex")
results.to_csv(f"{path}/results_{now}.csv")
