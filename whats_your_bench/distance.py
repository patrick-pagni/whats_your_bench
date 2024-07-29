import numpy as np
import torch
from ddks.methods import adKS

def _generate_array(start, end, n):
    # Generate an array with values from start to end with decreasing intervals
    if end == np.inf:
        arr = np.cumsum(np.exp(np.linspace(0, 5, n)))
        # No need to normalize to end value, since it is infinite
        arr = (arr - arr[0])
    else:
        arr = np.cumsum(np.exp(-np.linspace(0, 5, n)))
        # Normalize the array to ensure it ranges from start to end
        arr = (arr - arr[0]) / (arr[-1] - arr[0]) * (end - start) + start

    return arr

def _pdf_space(start, end, n = 10000):
    mean = np.array([start, end]).mean()
    arr2 = _generate_array(mean, end, int(n/2))
    arr1 = _generate_array(-mean, -start, int(n/2)) * -1
    return np.concatenate([arr1[::-1], arr2])

def integrate(function, space):
    y = function(space)

    h = np.diff(space)

    return np.sum((y[1:] + y[:-1]) * h / 2)

# calculate kullback leibler divergence using integrate function
def kl_divergence(p, q, support_lim):
    start, end = support_lim
    space = _pdf_space(start, end)
    return integrate(lambda x: p.pdf(x) * np.log(p.pdf(x) / q.pdf(x)), space)

# Function to get N-dimensional ks distance
# TODO: Implement significance test
def ks_test(p, q, support_lim):

    pred = torch.Tensor(p.rvs(1000))
    true = torch.Tensor(q.rvs(1000))

    a = adKS()

    return a(pred, true, q, support_lim)