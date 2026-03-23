import numpy as np

from dqa.analysis.classical import (
    KSTestDriftAnalyzer,
    PopulationStabilityIndexDriftAnalyzer,
)
from dqa.domain.models import DriftLevel

RNG = np.random.default_rng(42)
REF = RNG.normal(loc=0, scale=1, size=2000)
PROD_SAME = RNG.normal(loc=0, scale=1, size=2000)
PROD_DRIFT = RNG.normal(loc=3.0, scale=1, size=2000)

ks_test = KSTestDriftAnalyzer().analyze
population_stability_index = PopulationStabilityIndexDriftAnalyzer().analyze


def test_ks_stable_for_same_distribution():
    result = ks_test(REF, PROD_SAME)
    assert result.level == DriftLevel.STABLE
    assert result.details["p_value"] > 0.05


def test_ks_alert_for_drifted_distribution():
    result = ks_test(REF, PROD_DRIFT)
    assert result.level == DriftLevel.ALERT
    assert result.details["p_value"] < 0.05


def test_ks_result_has_required_keys():
    result = ks_test(REF, PROD_SAME)
    assert "statistic" in result.details
    assert "p_value" in result.details


def test_psi_stable_for_identical_data():
    result = population_stability_index(REF, REF)
    assert result.level == DriftLevel.STABLE
    assert result.details["psi"] < 0.1


def test_psi_alert_for_large_drift():
    result = population_stability_index(REF, PROD_DRIFT)
    assert result.level == DriftLevel.ALERT
    assert result.details["psi"] > 0.2


def test_psi_warning_band():
    prod_moderate = RNG.normal(loc=0.8, scale=1, size=2000)
    result = population_stability_index(REF, prod_moderate)
    assert result.level >= DriftLevel.WARNING
