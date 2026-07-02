# MANIFEST — reproducibility package

Generated inventory of everything staged in this folder. Push the whole folder as the paper's repo.

## Top-level
- MANIFEST.md
- README.md
- REPRODUCE_ARTIFACT.md
- requirements.txt

## Core modules (required for imports)
- `fl/`: __init__.py client.py server.py strategies.py 
- `models/`: __init__.py mlp.py siamese.py 
- `privacy/`: __init__.py accountant.py dp_trainer.py mia.py 
- `dedup/`: __init__.py local_dedup.py sil_gate.py 
- `data/`: __init__.py loaders.py partitioner.py 
- `audit/`: __init__.py ledger.py 

## Experiment scripts (`experiments/`)
- __init__.py
- alt_detector.py
- cached_load.py
- characterization.py
- diversity_coreset.py
- fed_allarms.py
- high_signal_selection.py
- honest_rerun.py
- honest_sweep.py
- lira_mia.py
- make_paper_figs.py
- principled_method.py
- run_full_paper_sweep.py
- runtime_benchmark.py
- strong_selection.py

## Result artifacts (`results/`) — every paper number traces here
- alt_detector.json
- characterization.json
- fed_allarms.json
- high_signal_selection.json
- honest_rerun.json
- lira_mia.json
- paper_results.json
- reference_baselines.json
- runtime_benchmark.json
- stats_tests.json
- strong_selection.json

## Paper sources (`paper/`)
- main.tex, supplement.tex, references.bib
- figs/: fig_frontier.eps fig_frontier.pdf fig_mia.eps fig_mia.pdf fig_schematic.eps fig_schematic.pdf 

## Totals
- Python files: 34
- Result JSONs: 11
- NOT included (kept out deliberately): legacy pre-pivot result files (ablations, gradient_diversity, fedprox, rho_*, clean_numbers) and legacy figures from the abandoned positive-result draft.
