"""
Genera los datasets de ejemplo para dqa.

Uso:
    uv run python examples/generate_fixtures.py

Archivos generados:
    examples/reference.parquet          — dataset de referencia (entrenamiento)
    examples/reference.csv              — mismo dataset en formato CSV
    examples/production_stable.parquet  — producción sin drift (exit 0)
    examples/production_drifted.parquet — producción con drift en income y score (exit 1)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

OUT = Path(__file__).parent

rng_ref   = np.random.default_rng(42)
rng_stab  = np.random.default_rng(1)
rng_drift = np.random.default_rng(99)

# --- Referencia (entrenamiento) ---
reference = pl.DataFrame({
    "age":    rng_ref.normal(35, 10, 1_000).tolist(),
    "income": rng_ref.normal(60_000, 15_000, 1_000).tolist(),
    "score":  rng_ref.normal(0.7, 0.1, 1_000).tolist(),
})
reference.write_parquet(OUT / "reference.parquet")
reference.write_csv(OUT / "reference.csv")

# --- Producción estable (sin drift significativo) ---
pl.DataFrame({
    "age":    rng_stab.normal(35, 10, 1_000).tolist(),
    "income": rng_stab.normal(60_000, 15_000, 1_000).tolist(),
    "score":  rng_stab.normal(0.7, 0.1, 1_000).tolist(),
}).write_parquet(OUT / "production_stable.parquet")

# --- Producción con drift ---
# income: media sube un 33% (N(60k→80k))  → ALERT
# score:  media baja de 0.7 a 0.45        → ALERT
# score_v2: columna nueva                  → Schema Drift (Added)
pl.DataFrame({
    "age":      rng_drift.normal(35, 10, 1_000).tolist(),
    "income":   rng_drift.normal(80_000, 15_000, 1_000).tolist(),
    "score":    rng_drift.normal(0.45, 0.15, 1_000).tolist(),
    "score_v2": rng_drift.uniform(0, 1, 1_000).tolist(),
}).write_parquet(OUT / "production_drifted.parquet")

print("Fixtures generados:")
for f in sorted(OUT.glob("*.parquet")) + sorted(OUT.glob("*.csv")):
    print(f"  {f.relative_to(OUT.parent)}")
