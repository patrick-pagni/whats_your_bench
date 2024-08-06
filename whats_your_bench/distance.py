import numpy as np
import torch
from ddks.methods import adKS

def _generate_array(start, end, n):
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
        axes.append((arr - arr[0]) / (arr[-1] - arr[0]) * (limits[i, 1] - limits[i, 0]) + limits[i, 0])

    return np.array([np.array(x) for x in list(zip(*axes))])

def _pdf_space(start, end, n = 10):
    mean = np.array([start, end]).mean(axis = 0)

    arr1 = _generate_array(-mean, -start, int(n/2)) * -1
    arr2 = _generate_array(mean, end, int(n/2))

    return np.concatenate([arr1, arr2[1:, :]])

def integrate(function, space):

    y = function(space)
    h = np.diff(space, axis = 0)

    if y.shape[-1] != h.shape[-1]:
        y = y.reshape(-1, 1)

    y = np.hstack((h, y[1:]))

    return np.sum(np.prod(y, axis = 1))

# calculate kullback leibler divergence using integrate function
def kl_divergence(p, q, support_lim):
    start, end = support_lim
    space = _pdf_space(start, end)
    return integrate(lambda x: p.pdf(x) * np.log(p.pdf(x) / q.pdf(x)), space)

# Function to get N-dimensional ks distance
# TODO: Implement significance test
def ks_test(p, q, support_lim):

    ks_distances = []

    for i in range(30):

        try:
            pred = torch.Tensor(p.rvs(100))
            true = torch.Tensor(q.rvs(100))

            if len(true.shape) == 1:
                true = true.reshape(-1, 1)
                pred = pred.reshape(-1, 1)

            a = adKS()

            ks_distances.append(a(pred, true, q, support_lim))
        except:
            pass

    return sum(ks_distances)/len(ks_distances)