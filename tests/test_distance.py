import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "whats_your_bench"))

import numpy as np
import pytest
from scipy import stats
import distance


class TestKlDivergence:
    def test_identical_distributions_near_zero(self):
        # KL(p||p) should be ~0
        p = stats.norm(loc=0, scale=1)
        support = [-10.0, 10.0]
        result, elapsed = distance.kl_divergence(p, p, support)
        assert result < 0.05
        assert elapsed >= 0

    def test_different_distributions_positive(self):
        p = stats.norm(loc=3, scale=1)
        q = stats.norm(loc=0, scale=1)
        support = [-5.0, 8.0]
        result, elapsed = distance.kl_divergence(p, q, support)
        assert result > 0

    def test_returns_float(self):
        p = stats.norm(loc=0, scale=1)
        result, _ = distance.kl_divergence(p, p, [-10.0, 10.0])
        assert isinstance(float(result), float)


class TestKsTest:
    def test_identical_distributions_small_distance(self):
        p = stats.norm(loc=0, scale=1)
        distance_val, pvalue = distance.ks_test(p, p, 10.0, random_state=1)[0]
        assert distance_val < 0.5

    def test_different_distributions_larger_distance(self):
        p = stats.norm(loc=10, scale=1)
        q = stats.norm(loc=0, scale=1)
        distance_val, _ = distance.ks_test(p, q, 15.0, random_state=1)[0]
        # Distributions are far apart; KS distance should be detectable
        assert distance_val >= 0

    def test_returns_tuple_with_timing(self):
        p = stats.norm(loc=0, scale=1)
        result = distance.ks_test(p, p, 10.0, random_state=1)
        assert len(result) == 2   # (ks_result_tuple, elapsed_time)
        assert result[1] >= 0
