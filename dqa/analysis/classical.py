"""
Segunda capa de detección de schema drift, incluye tests clásicos
de drift — KS test y PSI.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from dqa.domain.models import AnalysisResult, DriftLevel


class KSTestDriftAnalyzer:
    name: str = "KS Test"

    def analyze(self, ref: np.ndarray, prod: np.ndarray) -> AnalysisResult:
        """
        Kolmogorov-Smirnov de dos muestras: ¿provienen de la misma distribución?

        Detecta diferencias en cualquier parte de la distribución (cola, centro, forma).
        Sensible a n pequeño; complementar con PSI para confirmación.

        Args:
            ref:  Array de la distribución de referencia.
            prod: Array de la distribución de producción.

        Returns:
            ALERT si p_value < 0.05, STABLE en caso contrario.
        """
        stat, p_value = stats.ks_2samp(ref, prod)
        level = DriftLevel.ALERT if p_value < 0.05 else DriftLevel.STABLE
        return AnalysisResult(
            level=level,
            details={
                "statistic": round(float(stat), 4),
                "p_value": round(float(p_value), 4),
            },
        )


class PopulationStabilityIndexDriftAnalyzer:
    name: str = "PSI"

    def analyze(
        self, ref: np.ndarray, prod: np.ndarray, n_bins: int = 10
    ) -> AnalysisResult:
        """
        Population Stability Index — estándar de la industria financiera para monitoreo.

        Los bins se calculan sobre los percentiles de `ref` para que la referencia
        defina la escala. Valores de producción fuera del rango de referencia caen
        en los bins extremos (–∞, +∞).

        Umbrales convencionales:
            PSI < 0.1  → STABLE  (sin cambio significativo)
            PSI 0.1–0.2 → WARNING (cambio moderado, monitorear)
            PSI > 0.2  → ALERT   (cambio significativo, investigar)

        Args:
            ref:    Array de referencia.
            prod:   Array de producción.
            n_bins: Número de bins (default 10, estándar de industria).

        Returns:
            AnalysisResult con el valor PSI y su nivel de severidad.
        """
        breakpoints = np.percentile(ref, np.linspace(0, 100, n_bins + 1))
        breakpoints[0], breakpoints[-1] = -np.inf, np.inf

        ref_freq = np.histogram(ref, bins=breakpoints)[0] / len(ref)
        prod_freq = np.histogram(prod, bins=breakpoints)[0] / len(prod)

        # Reemplazar ceros para evitar log(0)
        ref_freq = np.where(ref_freq == 0, 1e-6, ref_freq)
        prod_freq = np.where(prod_freq == 0, 1e-6, prod_freq)

        psi = float(np.sum((prod_freq - ref_freq) * np.log(prod_freq / ref_freq)))

        if psi < 0.1:
            level = DriftLevel.STABLE
        elif psi < 0.2:
            level = DriftLevel.WARNING
        else:
            level = DriftLevel.ALERT

        return AnalysisResult(level=level, details={"psi": round(psi, 4)})
