"""Cache-aware dataset load to avoid the pyarrow-after-torch segfault on parquet
datasets (SWaT/TEP). Loads results/cache_{ds}.npz (pyarrow-free) when present;
falls back to the normal loader otherwise. SKAB (CSV) is unaffected either way."""
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent


def cached_load(ds, max_samples, random_state=42):
    cpath = _ROOT / "results" / f"cache_{ds}.npz"
    if cpath.exists():
        z = np.load(cpath)
        return z["X"].astype("float32"), z["y"].astype("int64")
    from data.loaders import DATASET_REGISTRY
    X, y, _ = DATASET_REGISTRY[ds](max_samples=max_samples, random_state=random_state)
    return X.astype("float32"), y.astype("int64")
