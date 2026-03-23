import time

import numpy as np

from dqa.analysis.information import KLJSDivergenceDriftAnalyzer
from dqa.domain.models import DriftLevel

RNG = np.random.default_rng(42)
kl_js_divergence = KLJSDivergenceDriftAnalyzer().analyze


def test_js_near_zero_for_identical_data():
    data = RNG.normal(0, 1, 3000)
    result = kl_js_divergence(data, data)
    assert result.details["js_divergence"] < 0.01
    assert result.level == DriftLevel.STABLE


def test_js_high_for_disjoint_distributions():
    ref = RNG.normal(loc=0, scale=1, size=3000)
    prod = RNG.normal(loc=10, scale=1, size=3000)
    result = kl_js_divergence(ref, prod)
    assert result.details["js_divergence"] > 0.6
    assert result.level == DriftLevel.ALERT


def test_result_has_all_keys():
    data = RNG.normal(0, 1, 1000)
    result = kl_js_divergence(data, data)
    assert "kl_ref_to_prod" in result.details
    assert "kl_prod_to_ref" in result.details
    assert "js_divergence" in result.details


def test_runs_under_500ms_for_large_arrays():
    ref = RNG.normal(0, 1, 10_000)
    prod = RNG.normal(0, 1, 10_000)
    t0 = time.perf_counter()
    kl_js_divergence(ref, prod)
    assert time.perf_counter() - t0 < 0.5
