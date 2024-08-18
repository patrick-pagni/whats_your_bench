import problem_set
import os
import sys
import inspect
import pandas as pd
import datetime
import argparse

parser = argparse.ArgumentParser(description="CLI for what's your bench")
parser.add_argument("problems", metavar="problems", type = int, nargs = "+", help = "Problems to be run by benchmarking suite")
args = parser.parse_args()

problems = [cls_obj for cls_name, cls_obj in inspect.getmembers(sys.modules['problem_set']) if inspect.isclass(cls_obj)][1:]

if args.problems:
     problems = [problem for i, problem in enumerate(problems) if i+1 in args.problems]

results = pd.DataFrame()

for i, problem in enumerate(problems):

    p = problem()
    print(type(p).__name__)

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

    p.results.insert(loc = 0, column = "Problem #", value = [type(p).__name__]*p.results.shape[0])
    results = pd.concat([results, p.results], axis = 0)

results.reset_index(drop=True, inplace=True)

path = "../results"
os.makedirs(path, exist_ok=True) 

now = datetime.datetime.now()
results.to_markdown(f"{path}/results_{now}.md")
results.to_latex(f"{path}/results_{now}.tex")
results.to_csv(f"{path}/results_{now}.csv")