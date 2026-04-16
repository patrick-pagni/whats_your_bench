# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

**What's Your Bench** is a benchmarking suite that evaluates three probabilistic programming languages (PPLs) — PyMC, Pyro, and Stan — by comparing their inferred posterior distributions against analytically exact posteriors derived from conjugate prior models. Distance is measured using KL Divergence and an N-dimensional Kolmogorov-Smirnov test (from the external `DDKS` package).

## Installation

```bash
conda create -c conda-forge -n <ENV_NAME> "pymc>=5"
# Apple Silicon: add "libblas=*=*accelerate" to the above
pip install -r requirements.txt
pip install git+https://github.com/patrick-pagni/DDKS
# Then install CmdStan for Stan support
```

## Running the Benchmark

The entry point must be run from inside `whats_your_bench/whats_your_bench/` (imports are relative):

```bash
cd whats_your_bench/whats_your_bench/
python main.py 1          # run Problem01 only
python main.py 1 2 3      # run multiple specific problems
python main.py            # run all 21 problems (slow — high-dim problems take a long time)
```

Results are written to `results/` as `.csv`, `.md`, and `.tex` files timestamped at run time.

## Architecture

### Core Data Flow

```
Problem (problem_set.py)
  ├── conjugate_prior (conjugate_priors.py) → exact posterior + predictive distribution
  ├── run_models() → pymc_models / pyro_models / stan_models → fitted SimpleNamespace params
  └── evaluate_models() → distance.py (KL divergence + KS test) → self.results DataFrame
```

### Key Design Conventions

**`SimpleNamespace` for distribution params**: Every model function (conjugate, PyMC, Pyro, Stan) returns a `SimpleNamespace` whose attribute names match `scipy.stats` distribution constructor kwargs (e.g., `loc`, `scale`, `df`, `mean`, `cov`). `Problem._model_dist()` unpacks these via `**params.__dict__` directly into the scipy distribution constructor. New models must return a `SimpleNamespace` with exactly the right attribute names for the target `predictive_dist`.

**`@timer` decorator**: All model functions and distance functions are wrapped with `utils.timer`, which returns `(result, elapsed_seconds)`. Callers always unpack two values: `result, time = func(...)`.

**Problem structure**: Each `ProblemNN` in `problem_set.py` is a standalone subclass of `Problem`. It hard-codes its own conjugate prior type, prior parameters, sample size, data-generating distribution, and random seed. Problems 01–06 vary sample size for Normal/Known-Variance; 07–15 vary priors and sample size for Normal/Known-Mean; 16–21 are multivariate (2D, 5D, 10D). MvNormal with known mean (commented out) is unimplemented across all three PPLs.

**Stan model files**: Stan `.stan` files live in `whats_your_bench/stan_models/`. The Python wrappers in `stan_models.py` reference them via a relative path (`../whats_your_bench/stan_models/...`), so they must be run from `whats_your_bench/whats_your_bench/`.

### Adding a New Problem

1. Add a conjugate prior class to `conjugate_priors.py` if needed — implement `find_predictive_posterior(data)` setting `self.posterior_predictive_params`, `self.predictive_dist`, and any known parameter attribute (`self.sigma`, `self.mu`, etc.).
2. Add matching model functions to `pymc_models.py`, `pyro_models.py`, `stan_models.py` (and a `.stan` file if needed). Decorate with `@timer`; return a `SimpleNamespace` with attrs matching the scipy dist constructor.
3. Add a `ProblemNN` class to `problem_set.py`; `main.py` discovers all classes in that module via `inspect.getmembers` in definition order.
