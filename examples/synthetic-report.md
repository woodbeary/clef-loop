# clef-loop eval: synthetic-demo

> Synthetic data: these numbers test the pipeline, not any model.

60 held-out records, 2026-09-26T08:59:45.232Z to 2026-10-10T04:28:24.156Z. Metric: agreement with the human's final decision. Failed calls and unparseable answers count as wrong.

| Arm | Kind | Exact match | Answered | p50 ms | p95 ms | Errors |
|---|---|---|---|---|---|---|
| logged | replay | 60.0% | 60/60 | n/a | n/a | 0 |
| majority | majority | 53.3% | 60/60 | n/a | n/a | 0 |

## Question `decision`

| Arm | Accuracy (95% CI) | Macro-F1 | Per class: precision / recall |
|---|---|---|---|
| logged | 60.0% (47.4% to 71.4%) | 0.328 | accept 0.57 / 1.00; skip 1.00 / 0.15; needs_info n/a / 0.00 |
| majority | 53.3% (40.9% to 65.4%) | 0.232 | accept 0.53 / 1.00; skip n/a / 0.00; needs_info n/a / 0.00 |

## Paired comparison (exact McNemar, same records)

| A | B | A right, B wrong | B right, A wrong | p |
|---|---|---|---|---|
| logged | majority | 4 | 0 | 0.125 |

Latency is client wall time from the machine that ran this eval, network included. It is not production latency; see docs/MEASURING.md for how to measure that.
