import numpy as np
import pytest

pytest.importorskip("pymc", reason="Requiere: uv sync --extra bayesian")

from dqa.adapters.bayesian.pymc_analyzer import BayesianAnalyzer
from dqa.domain.models import DriftLevel

RNG = np.random.default_rng(42)


@pytest.mark.slow
def test_stable_for_same_distribution():
    ref  = RNG.normal(0, 1, 300)
    prod = RNG.normal(0, 1, 300)
    result = BayesianAnalyzer().analyze(ref, prod)
    assert result.details["hdi_overlap"] > 0.5
    assert result.level in (DriftLevel.STABLE, DriftLevel.WARNING)


@pytest.mark.slow
def test_alert_for_large_drift():
    ref  = RNG.normal(0, 1, 300)
    prod = RNG.normal(5, 1, 300)
    result = BayesianAnalyzer().analyze(ref, prod)
    assert result.details["hdi_overlap"] < 0.2
    assert result.level == DriftLevel.ALERT


@pytest.mark.slow
def test_rhat_below_convergence_threshold():
    """r-hat < 1.05 indica que las cadenas MCMC convergieron correctamente."""
    ref  = RNG.normal(0, 1, 300)
    prod = RNG.normal(0, 1, 300)
    result = BayesianAnalyzer().analyze(ref, prod)
    assert result.details["rhat_mu_ref"]  < 1.05
    assert result.details["rhat_mu_prod"] < 1.05
