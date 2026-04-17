from .utils import timer
from config import PDF_GRID_SIZE, KS_SAMPLE_SIZE

import numpy as np
import torch
from ddks.methods import adKS

def _generate_array(start, end, n, reverse = False):
    # Generate an array with values from start to end with decreasing intervals
    try:
        d = start.shape[0]
    except IndexError:
        d = 1
        start = np.array([start])
        end = np.array([end])

    limits = np.column_stack((start, end))
    axes = []

    for i in range(d):
        arr = np.cumsum(np.exp(-np.linspace(0, 5, n)))
        if reverse:
            arr = arr[::-1]
        axes.append((arr - arr[0]) / (arr[-1] - arr[0]) * (limits[i, 1] - limits[i, 0]) + limits[i, 0])

    return np.array([np.array(x) for x in list(zip(*axes))])

def _adaptive_pdf_grid(start, end, n = PDF_GRID_SIZE):
    """Generate a non-uniform grid concentrated near the center, suitable for PDF evaluation."""
    mean = np.array([start, end]).mean(axis = 0)

    arr1 = _generate_array(start, mean, int(n/2), reverse = True)
    arr2 = _generate_array(mean, end, int(n/2))

    return np.concatenate([arr1, arr2[1:, :]])

def numerical_integrate_pdf(function, space):
    """Numerically integrate a PDF over the given space using the trapezoid rule."""
    y = function(space)
    h = np.diff(space, axis = 0)

    if y.shape[-1] != h.shape[-1]:
        y = y.reshape(-1, 1)
    
    y = np.abs(y)

    y = np.hstack((h, y[1:]))

    return np.sum(np.prod(y, axis = 1))

# calculate kullback leibler divergence using integrate function
@timer
def kl_divergence(p, q, support_lim):
    start, end = support_lim
    space = _adaptive_pdf_grid(start, end)
    return numerical_integrate_pdf(lambda x: p.pdf(x) * np.log(p.pdf(x) / q.pdf(x)), space)

# Function to get N-dimensional ks distance
@timer
def ks_test(p, q, support_lim, random_state, method = "all"):

    pred = p.rvs(KS_SAMPLE_SIZE, random_state = random_state)
    true = q.rvs(KS_SAMPLE_SIZE, random_state = random_state)

    if len(true.shape) == 1:
        true = true.reshape(-1, 1)
        pred = pred.reshape(-1, 1)

    a = adKS(method = method)

    ks_distance = a(pred, true, q, support_lim).item()
    ks_pvalue = a.p_D()

    return ks_distance, ks_pvalue
