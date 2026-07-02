# How to reproduce the results

This guide lists the exact steps to set up the environment, place the data, and
regenerate every number and figure in the paper. Each result is written to a JSON file
under `results/`; the paper reads its tables from those files.

## 1. Environment

Tested on Python 3.13, CPU only. Create a fresh environment and install the pinned
versions:

```
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` pins PyTorch 2.12.0, Opacus 1.6.0, scikit-learn 1.7.2, NumPy 2.2.0,
SciPy 1.16.2, pyarrow 24.0.0, faiss-cpu 1.13.0, matplotlib 3.10.6.

## 2. Data

Create a `Dataset/` folder at the repository root and place the raw data there
(`data/loaders.py` reads from `<repo>/Dataset/`):

- SKAB: download from the Skoltech SKAB release and place under `Dataset/`.
- TEP: download the Rieth et al. (2017) Tennessee Eastman simulation and place under `Dataset/`.
- SWaT: request from iTrust (SUTD) under their usage agreement; not redistributable here.
  The attack-window labelling (GMT+8 to GMT+0) is described in the paper supplement.

On the first run, SWaT and TEP are cached to `results/cache_<dataset>.npz`. This avoids a
native-library load-order crash when the parquet reader is imported after PyTorch.

## 3. Run the experiments

Each command runs from the repository root and writes one JSON file. Scripts that train
federated or 40-epoch models are slow on CPU; all are resumable (re-running continues
from the last completed cell).

| Command | Output | Paper element |
|---------|--------|---------------|
| `python experiments/characterization.py skab swat tep` | `results/characterization.json` | Table 3, Fig 2 (selection comparison, 10 seeds) |
| `python experiments/honest_rerun.py` | `results/honest_rerun.json` | Table 2 (artifact audit, 10 seeds) |
| `python experiments/strong_selection.py` | `results/strong_selection.json` | Table 4 (longer-trained detector) |
| `python experiments/fed_allarms.py` | `results/fed_allarms.json` | Federated all-arms table |
| `python experiments/fed_kablation.py` | `results/fed_kablation.json` | Federation-scale table (K=10,20) |
| `python experiments/fed_dirichlet01.py` | `results/fed_dirichlet01.json` | Dirichlet(0.1) heterogeneity note |
| `python experiments/protocol_ablation.py` | `results/protocol_ablation.json` | Factorial ablation (supplement) |
| `python experiments/lira_mia.py` | `results/lira_mia.json` | MIA, honest split |
| `python experiments/mia_artifact.py` | `results/mia_artifact.json` | MIA, naive contiguous split |
| `python experiments/lira_n64_check.py` / `lira_n64_skab.py` | `results/lira_n64_*.json` | N=64 validation |
| `python experiments/alt_detector.py` | `results/alt_detector.json` | Isolation Forest / One-Class SVM |
| `python experiments/high_signal_selection.py` | `results/high_signal_selection.json` | Floor-effect / high-signal |
| `python experiments/runtime_benchmark.py` | `results/runtime_benchmark.json` | Runtime |
| `python experiments/convergence_curve.py` | `results/convergence_curve.json` | Convergence (R=15) |

## 4. Statistics and figures

```
python experiments/make_paper_figs.py        # writes paper/figs/{schematic,frontier,mia}.{pdf,eps}
```

The Wilcoxon / TOST / Cohen's d_z values (`results/stats_tests.json`) are computed from
`characterization.json` and `fed_allarms.json`.

## 5. Reproducing the false positive (the naive pipeline)

`REPRODUCE_ARTIFACT.md` explains how to re-introduce the two protocol violations and
regenerate the spurious deduplication gain stored in `results/paper_results.json`.

## 6. Building the paper

```
cd paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
pdflatex supplement.tex && bibtex supplement && pdflatex supplement.tex && pdflatex supplement.tex
```

Requires a LaTeX installation with the Springer Nature `sn-jnl` class (included files) and
the `sttools` package.

## Notes

- All JSON result files are included, so the tables and figures can be regenerated without
  re-running the (slow) experiments.
- The released code contains both the protocol-violating and the leakage-controlled pipelines,
  so the false positive and its correction are both reproducible.
