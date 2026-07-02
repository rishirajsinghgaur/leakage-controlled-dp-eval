# Reproducing the spurious deduplication "gain" (the protocol-violating pipeline)

The paper's case study shows that a natural but protocol-violating pipeline
produces a spurious +0.05 to +0.16 F1 deduplication advantage on SWaT. That result
is preserved in `results/paper_results.json` (conditions `fedavg_dp_dedup` vs
`fedavg_dp_nodedup`, dataset `swat`). To regenerate it, the two protocol violations
must be re-introduced into the leakage-controlled pipeline:

1. **Fit the selection gate on private + anomaly data** (violates requirement i).
   In the leakage-controlled pipeline the 3-sigma gate is fit on public normal-only data. The
   uncontrolled version fits it on the full local training data, anomalies included
   (`global_train_X = X_train` instead of `X_train[y_train == 0]` in
   `experiments/run_full_paper_sweep.py`).

2. **Train the detector on the anomaly-contaminated selected set** (violates
   requirement ii). In the leakage-controlled pipeline the reconstruction detector is trained on
   normal-only data (`train_normal_only=True` in `fl/client.py`); the uncontrolled version
   trains on the mixed deduplicated set.

Re-applying these two changes reproduces the inflated gain; reverting them (the
shipped default) reproduces the corrected null. The magnitude per budget is in
`paper_results.json`:

| eps | dedup (uncontrolled) | no-dedup (uncontrolled) | spurious gain |
|-----|-------------|----------------|---------------|
| 0.5 | 0.273 | 0.219 | +0.054 |
| 1.0 | 0.301 | 0.203 | +0.098 |
| 2.0 | 0.333 | 0.201 | +0.132 |
| 4.0 | 0.352 | 0.196 | +0.156 |

(Verifiable directly from `results/paper_results.json`.)
