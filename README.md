# Leakage-Controlled Evaluation of Data Selection in DP Federated Anomaly Detection

Code for the paper *"A Leakage-Controlled Evaluation Protocol for Data Selection in
Differentially Private Federated Anomaly Detection for Industrial IoT."*

## Setup
```
pip install -r requirements.txt
```
Python 3.11+. DP-SGD via Opacus; anomaly detection via a reconstruction autoencoder.

## Run
```
python -m experiments.characterization      # main selection-vs-random sweep
python -m experiments.dp_aware_selection     # privacy-paid selector
python -m experiments.mia_privacy_final      # membership-inference audit
python -m experiments.reproduce_check        # seeded reproducibility check
```
Each script writes its results as JSON to a `results/` folder (created on first run).

## Layout
- `experiments/` — all sweeps and audits (one script per table)
- `privacy/` — DP-SGD trainer, RDP accountant, MIA
- `fl/`, `models/`, `data/`, `dedup/`, `audit/` — federated loop, detector, loaders, selection rules

## Data
SKAB, TEP, and HAI are public; place raw files under `Dataset/` at the repo root.
SWaT (iTrust, SUTD, July-2019) requires a signed access request and is not
redistributed here.

## Note
DP-SGD on CPU is not bit-deterministic (Opacus noise + threaded float sums); reruns
fall within the reported per-cell standard deviation and the conclusions are invariant.
