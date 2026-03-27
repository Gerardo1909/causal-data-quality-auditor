# Datasets de ejemplo

Datasets sintéticos para probar `dqa compare` localmente.

## Generar los archivos

```sh
uv run python examples/generate_fixtures.py
```

## Archivos generados

| Archivo | Descripción | Columnas | Exit code esperado |
|---------|-------------|----------|-------------------|
| `reference.parquet` | Dataset de referencia (entrenamiento) | age, income, score | — |
| `reference.csv` | Mismo dataset en formato CSV | age, income, score | — |
| `production_stable.parquet` | Producción sin drift | age, income, score | `0` |
| `production_drifted.parquet` | Producción con drift en income y score | age, income, score, score_v2 | `1` |

## Distribuciones

| Columna | Referencia | Producción estable | Producción con drift |
|---------|------------|-------------------|---------------------|
| `age` | N(35, 10) | N(35, 10) | N(35, 10) — STABLE |
| `income` | N(60 000, 15 000) | N(60 000, 15 000) | N(80 000, 15 000) — **ALERT** (+33% en media) |
| `score` | N(0.7, 0.1) | N(0.7, 0.1) | N(0.45, 0.15) — **ALERT** (degradación del modelo) |
| `score_v2` | — | — | Uniform(0, 1) — **Schema drift** (columna añadida) |

## Comandos de prueba

```sh
# Comparación estable — exit 0
dqa compare examples/reference.parquet examples/production_stable.parquet

# Comparación con drift — exit 1
dqa compare examples/reference.parquet examples/production_drifted.parquet

# Formatos mixtos: CSV como referencia
dqa compare examples/reference.csv examples/production_drifted.parquet

# Reporte Markdown
dqa compare examples/reference.parquet examples/production_drifted.parquet \
  --format markdown --output reporte.md

# Solo columnas específicas
dqa compare examples/reference.parquet examples/production_drifted.parquet \
  --columns income,score
```
