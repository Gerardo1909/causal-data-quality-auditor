"""
Adapter bayesiano opcional: ajusta Normal(μ, σ) a cada dataset con PyMC
y compara los parámetros posteriores usando HPD overlap.

Solo se importa si el usuario usa --bayesian y tiene dqa[bayesian] instalado.
"""
from __future__ import annotations

import numpy as np
import pymc as pm
import arviz as az

from dqa.domain.models import AnalysisResult, DriftLevel


class BayesianAnalyzer:
    """
    Implementa ColumnAnalyzer usando inferencia bayesiana con PyMC.

    En lugar de métricas escalares, ajusta una distribución paramétrica a cada
    dataset y compara los intervalos de alta densidad posterior (HDI) de μ.
    Esto permite responder: "el μ de producción está fuera del rango plausible
    del μ de referencia", lo cual es más informativo que un p-value.
    """

    name = "bayesian"

    def analyze(self, ref: np.ndarray, prod: np.ndarray) -> AnalysisResult:
        """
        Ajusta Normal(μ, σ) a ref y prod por separado y compara sus posteriors de μ.

        La métrica principal es el overlap del HDI 94% entre μ_ref y μ_prod:
        - overlap > 0.6 → STABLE
        - overlap 0.3–0.6 → WARNING
        - overlap < 0.3 o μ_prod fuera del HDI de μ_ref → ALERT

        Args:
            ref:  Array de la distribución de referencia.
            prod: Array de la distribución de producción.

        Returns:
            AnalysisResult con mu_means, HDIs, hdi_overlap y r-hat de ambos fits.
        """
        idata_ref  = self._fit_normal(ref,  "ref")
        idata_prod = self._fit_normal(prod, "prod")

        mu_ref  = idata_ref.posterior["mu"].values.flatten()
        mu_prod = idata_prod.posterior["mu"].values.flatten()

        hdi_ref  = az.hdi(mu_ref,  hdi_prob=0.94)
        hdi_prod = az.hdi(mu_prod, hdi_prob=0.94)

        # Proporción del rango total que comparten ambos HDIs
        overlap = max(0.0, min(hdi_ref[1], hdi_prod[1]) - max(hdi_ref[0], hdi_prod[0]))
        span    = max(hdi_ref[1], hdi_prod[1]) - min(hdi_ref[0], hdi_prod[0]) + 1e-9
        overlap_ratio = overlap / span

        mu_prod_mean = float(mu_prod.mean())
        prod_outside_ref_hdi = mu_prod_mean < hdi_ref[0] or mu_prod_mean > hdi_ref[1]

        if overlap_ratio < 0.3 or prod_outside_ref_hdi:
            level = DriftLevel.ALERT
        elif overlap_ratio < 0.6:
            level = DriftLevel.WARNING
        else:
            level = DriftLevel.STABLE

        return AnalysisResult(
            level=level,
            details={
                "mu_ref_mean":  round(float(mu_ref.mean()), 4),
                "mu_prod_mean": round(mu_prod_mean, 4),
                "hdi_ref":      [round(float(hdi_ref[0]), 4),  round(float(hdi_ref[1]), 4)],
                "hdi_prod":     [round(float(hdi_prod[0]), 4), round(float(hdi_prod[1]), 4)],
                "hdi_overlap":  round(float(overlap_ratio), 4),
                "rhat_mu_ref":  round(float(az.rhat(idata_ref)["mu"].values),  4),
                "rhat_mu_prod": round(float(az.rhat(idata_prod)["mu"].values), 4),
            },
        )

    @staticmethod
    def _fit_normal(data: np.ndarray, tag: str) -> az.InferenceData:
        """Ajusta Normal(μ, σ) a data con 2 cadenas MCMC. Guarda en /tmp para auditoría."""
        with pm.Model():
            mu    = pm.Normal("mu",    mu=float(data.mean()), sigma=float(data.std() * 2))
            sigma = pm.HalfNormal("sigma", sigma=float(data.std()))
            pm.Normal("obs", mu=mu, sigma=sigma, observed=data)
            idata = pm.sample(
                draws=500, tune=500, chains=2,
                nuts_sampler="nutpie",
                random_seed=42,
                progressbar=False,
            )
        idata.to_netcdf(f"/tmp/dqa_bayesian_{tag}.nc")
        return idata
