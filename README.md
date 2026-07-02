# A Leakage-Controlled Evaluation Protocol for Data Selection in DP Federated IoT Anomaly Detection

Reproducibility package for the paper of the above title. Every number in the paper
is produced by the scripts here and stored as a JSON file under `results/`; no value
is hand-entered.

## What this paper is
We give a **leakage-controlled evaluation protocol** for data selection (deduplication,
coresets) under differential privacy (DP) in federated industrial-IoT anomaly
detection, with four requirements: (i) compute selection statistics on **public data
only**; (ii) train the reconstruction detector on **normal data only**; (iii)
**account for the privacy cost** of data-dependent selection; and (iv) evaluate
membership inference with a **randomized, seeded** member/non-member split. We use it
to expose two evaluation artifacts: (a) a **utility artifact** — a protocol-violating
pipeline produces a spurious deduplication "gain" (+0.05 to +0.16 F1 on SWaT) that the
protocol erases; and (b) a **privacy artifact** — a contiguous temporal split reports
membership-inference AUC up to 0.73 on SWaT, but this collapses to chance (≈0.50) under
a randomized split, so no membership leakage is detectable under the attacks we run.

## The two pipelines (the heart of the case study)
- **Uncontrolled (protocol-violating) pipeline** — fits the 3-sigma selection gate on
  the full private (anomaly-containing) data and trains the detector on that
  contaminated set. Produces the spurious +0.05 to +0.16 F1 SWaT "gain" recorded in
  `results/paper_results.json`. See `REPRODUCE_ARTIFACT.md`.
- **Leakage-controlled pipeline** — gate fit on public normal-only data, detector
  trained on normal-only data, selection privacy accounted. Produces the corrected
  results in `results/honest_rerun.json`, `characterization.json`, etc.

Releasing both lets anyone reproduce the false positive *and* its correction.

## Environment
```
python -m venv venv && source venv/bin/activate   # (Windows: venv\Scripts\activate)
pip install -r requirements.txt
```
Pinned versions are in `requirements.txt` (Python 3.13, Opacus 1.6, PyTorch 2.12,
faiss-cpu 1.13). Note: `pyarrow` must be importable *before* the deep-learning stack
on some platforms; the scripts read cached `.npz` archives via
`experiments/cached_load.py` to avoid a native load-order crash.

## Reproducing each result (script -> JSON -> paper element)
| Paper element | Script | Output JSON |
|---|---|---|
| Table 1 (selection comparison, centralised) | `experiments/characterization.py` | `characterization.json` |
| Table 2 (artifact audit, federated) | `experiments/honest_rerun.py` | `honest_rerun.json` |
| Table 3 (longer-trained detector) | `experiments/strong_selection.py` | `strong_selection.json` |
| Federated all-arms table | `experiments/fed_allarms.py` | `fed_allarms.json` |
| Floor-effect / make-or-break | `experiments/high_signal_selection.py` | `high_signal_selection.json` |
| Alt-detector robustness (IF, OC-SVM) | `experiments/alt_detector.py` | `alt_detector.json` |
| MIA (weak vs LiRA) | `experiments/lira_mia.py` | `lira_mia.json` |
| Non-private baselines | (see `run_full_paper_sweep.py`) | `reference_baselines.json` |
| Runtime | `experiments/runtime_benchmark.py` | `runtime_benchmark.json` |
| Statistical tests (Wilcoxon/TOST) | (computed from `characterization.json`) | `stats_tests.json` |
| Federation scale (K=10,20) | `experiments/fed_kablation.py` | `fed_kablation.json` |
| MIA artifact (contiguous split) | `experiments/mia_artifact.py` | `mia_artifact.json` |
| Protocol ablation (2x2 of rules) | `experiments/protocol_ablation.py` | `protocol_ablation.json` |
| N=64 LiRA validation | `experiments/lira_n64_check.py`, `lira_n64_skab.py` | `lira_n64_check.json`, `lira_n64_skab.json` |
| Figures | `experiments/make_paper_figs.py` | `paper/figs/*` |
| Spurious gain (uncontrolled pipeline) | see `REPRODUCE_ARTIFACT.md` | `paper_results.json` |

## Datasets — required directory layout
Place the raw datasets under a `Dataset/` folder at the repository root (the loaders in
`data/loaders.py` read from `<repo>/Dataset/`):
- **SKAB** (Skoltech, public) and **TEP** (Rieth et al. 2017, public) — download and place
  under `Dataset/`.
- **SWaT** (iTrust, SUTD, July-2019) — requires their usage agreement and is **not**
  redistributed here; the attack-window label reconstruction (GMT+8 -> GMT+0) is in the
  paper supplement.

On first run, SWaT/TEP are cached to `results/cache_<ds>.npz` (loaded by
`experiments/cached_load.py`) to avoid a parquet/torch native load-order crash; the
`results/*.json` outputs in this package were produced from those caches.

## Citation / status
See `MANIFEST.md` for the exact file inventory. This package is staged for archival;
a frozen DOI will be minted on acceptance.
