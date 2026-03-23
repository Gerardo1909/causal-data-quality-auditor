"""
Tercera capa de detección de schema drift, implementa KL Divergence
y Jensen-Shannon Divergence via KDE.
"""

from __future__ import annotations

import numpy as np
from scipy.special import rel_entr
from scipy.stats import gaussian_kde

from dqa.domain.models import AnalysisResult, DriftLevel


class KLJSDivergenceDriftAnalyzer:
    name: str = "kl_js"

    _JS_WARNING_THRESHOLD = 0.05
    _JS_ALERT_THRESHOLD = 0.10

    def analyze(self, ref: np.ndarray, prod: np.ndarray) -> AnalysisResult:
        """
        Calcula KL Divergence y Jensen-Shannon Divergence entre dos distribuciones continuas.

        Usa Kernel Density Estimation (KDE gaussiano) para estimar las densidades de
        probabilidad antes de calcular las divergencias. JS es la métrica principal:
        es simétrica y acotada en [0, 1], a diferencia de KL que es asimétrica y
        puede ser infinita si los soportes no se superponen.

        JS = 0: distribuciones idénticas.
        JS = 1: distribuciones con soporte completamente disjunto.
        Umbral de alerta: JS > 0.10 (empírico, configurable por columna en v0.2).

        Args:
            ref:      Array de referencia (distribución base).
            prod:     Array de producción (distribución a comparar).

        Returns:
            AnalysisResult con kl_ref_to_prod, kl_prod_to_ref, js_divergence y nivel.
        """
        x_grid = np.linspace(
            min(ref.min(), prod.min()),
            max(ref.max(), prod.max()),
            500,
        )

        p = gaussian_kde(ref)(x_grid) + 1e-10
        q = gaussian_kde(prod)(x_grid) + 1e-10
        p, q = p / p.sum(), q / q.sum()

        m = (p + q) / 2
        kl_pq = float(np.sum(rel_entr(p, q)))
        kl_qp = float(np.sum(rel_entr(q, p)))
        js = float(np.sum(rel_entr(p, m)) / 2 + np.sum(rel_entr(q, m)) / 2)

        if js < self._JS_WARNING_THRESHOLD:
            level = DriftLevel.STABLE
        elif js < self._JS_ALERT_THRESHOLD:
            level = DriftLevel.WARNING
        else:
            level = DriftLevel.ALERT

        return AnalysisResult(
            level=level,
            details={
                "kl_ref_to_prod": round(kl_pq, 4),
                "kl_prod_to_ref": round(kl_qp, 4),
                "js_divergence": round(js, 4),
            },
        )
