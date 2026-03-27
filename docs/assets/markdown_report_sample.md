# DQA Drift Report

**Overall status:** 🔴 ALERT

## Schema Drift

- **Added:** `score_v2`

## Column Analysis

### `age` — 🟢 STABLE

| Metric | Key | Value |
|--------|-----|-------|
| KS Test | statistic | 0.049 |
| KS Test | p_value | 0.1812 |
| PSI | psi | 0.0206 |
| kl_js | kl_ref_to_prod | 0.0046 |
| kl_js | kl_prod_to_ref | 0.0047 |
| kl_js | js_divergence | 0.0011 |

### `income` — 🔴 ALERT

| Metric | Key | Value |
|--------|-----|-------|
| KS Test | statistic | 0.517 |
| KS Test | p_value | 0.0 |
| PSI | psi | 1.847 |
| kl_js | kl_ref_to_prod | 0.9843 |
| kl_js | kl_prod_to_ref | 0.9626 |
| kl_js | js_divergence | 0.1895 |

### `score` — 🔴 ALERT

| Metric | Key | Value |
|--------|-----|-------|
| KS Test | statistic | 0.702 |
| KS Test | p_value | 0.0 |
| PSI | psi | 2.8753 |
| kl_js | kl_ref_to_prod | 1.4322 |
| kl_js | kl_prod_to_ref | 4.7675 |
| kl_js | js_divergence | 0.3245 |
