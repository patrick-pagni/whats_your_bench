# Benchmarking Probabilistic Programming Languages

*Patrick Pagni*

*230246906*

This code is accompanied by the research paper submitted for grading over Turnitin.

## Installation instructions

1. Create new conda env using this tool

Conda env is recommended as it allows for easy istallation of PyMC

```python
conda create -c conda-forge -n <ENV_NAME> "pymc>=5"
```

**n.b Apple Silicon: when creating your venv for; create it with this command:**

```python
conda create -c conda-forge -n <ENV_NAME> "pymc>=5" "libblas=*=*accelerate"
```

2. Install core packages from requirements file

```python
pip install -r requirements.txt
```

3. Install DDKS package

```python
pip install git+https://github.com/patrick-pagni/DDKS
```

4. Install CmdStan by following the instructions given [here](https://arc.net/l/quote/jctmbzzm)

## Running the benchmarking suite

After completing all the steps above, navigate to:
`path-to-repository/whats_your_bench/whats_your_bench/`

And run:

```python
python main.py [optional integer arguments]
```

It should be noted that running `python main.py` will run all Problems in the problem set which is not recommended since it the high-dimensional problems take a long time to run. (Feel free to try it of course.) To avoid this I encourage supplying an integer argument from 1 - 21.

e.g.

```python
python main.py 1
```
