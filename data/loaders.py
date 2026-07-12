"""
Dataset loaders for DP-FL anomaly detection.

Each loader returns (X, y, feature_names) where:
  X : np.ndarray  shape (N, D), float32, already z-score normalised
  y : np.ndarray  shape (N,),   int {0=normal, 1=anomaly/fault}
  feature_names : list[str]
"""

import logging
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent  # PAPER_DISCOVER_IOT/
DATASET_ROOT = _ROOT / "Dataset"

TEP_DIR   = DATASET_ROOT / "TEP"
SWAT_DIR  = DATASET_ROOT / "swat" / "extracted" / "SWaT.A4 & A5_Jul 2019"
SKAB_DIR  = DATASET_ROOT / "SKAB"


# ─────────────────────────────────────────────────────────────────────────────
# TEP — Tennessee Eastman Process
# ─────────────────────────────────────────────────────────────────────────────

def load_tep(
    max_samples: int = 50_000,
    fault_types: List[int] = None,
    random_state: int = 42,
    target_fault_rate: float = 0.35,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load Tennessee Eastman Process dataset.

    Prioritises parquet siblings; falls back to RData via pyreadr.
    Samples proportionally from fault-free and faulty files so the final
    fault rate matches target_fault_rate (~35% by default).
    Returns X (float32, z-scored), y (0/1), feature_names.
    """
    files = {
        "ff_train": TEP_DIR / "TEP_FaultFree_Training.RData",
        "ff_test":  TEP_DIR / "TEP_FaultFree_Testing.RData",
        "f_train":  TEP_DIR / "TEP_Faulty_Training.RData",
        "f_test":   TEP_DIR / "TEP_Faulty_Testing.RData",
    }

    # Load each file separately with a generous cap, then stratify-sample
    dfs_normal, dfs_fault = [], []
    for key, path in files.items():
        per_file_cap = max_samples   # load up to max_samples per file; trim later
        df = _load_rdata_or_parquet(path, max_samples=per_file_cap,
                                    random_state=random_state)
        log.info("TEP %s: %d rows", key, len(df))
        y_col = "faultNumber"
        if y_col in df.columns:
            is_normal = df[y_col].astype(int) == 0
            dfs_normal.append(df[is_normal])
            dfs_fault.append(df[~is_normal])
        else:
            dfs_normal.append(df)

    df_normal = pd.concat(dfs_normal, ignore_index=True)
    df_fault  = pd.concat(dfs_fault,  ignore_index=True) if dfs_fault else pd.DataFrame()

    # Stratified subsample to hit target_fault_rate
    rng = np.random.default_rng(random_state)
    n_fault  = min(int(max_samples * target_fault_rate), len(df_fault))
    n_normal = min(max_samples - n_fault, len(df_normal))

    df_parts = []
    if n_normal > 0 and len(df_normal) > 0:
        idx = rng.choice(len(df_normal), size=n_normal, replace=False)
        df_parts.append(df_normal.iloc[idx])
    if n_fault > 0 and len(df_fault) > 0:
        idx = rng.choice(len(df_fault), size=n_fault, replace=False)
        df_parts.append(df_fault.iloc[idx])

    data = pd.concat(df_parts, ignore_index=True) if df_parts else df_normal

    # Feature columns: xmeas_1..41, xmv_1..11
    feat_cols = [c for c in data.columns if c.startswith("xmeas_") or c.startswith("xmv_")]
    if not feat_cols:
        meta = {"faultNumber", "simulationRun", "sample"}
        feat_cols = [c for c in data.columns if c not in meta]

    y_col = "faultNumber"
    labels = data[y_col].astype(int).values if y_col in data.columns else np.zeros(len(data), dtype=int)
    labels = (labels > 0).astype(int)

    X_raw = data[feat_cols].values.astype(np.float32)

    X = _zscore(X_raw)
    log.info("TEP loaded: %d samples, %d features, %.1f%% fault",
             len(X), X.shape[1], 100 * labels.mean())
    return X, labels, feat_cols


def _load_rdata_or_parquet(path: Path, max_samples=None, random_state=42) -> pd.DataFrame:
    parquet = path.with_suffix(".parquet")
    if parquet.exists():
        log.info("Parquet fast-path: %s", parquet.name)
        df = pd.read_parquet(parquet)
    else:
        try:
            import pyreadr
        except ImportError:
            raise ImportError("Install pyreadr: pip install pyreadr")
        log.info("Loading RData (slow): %s", path.name)
        result = pyreadr.read_r(str(path))
        df = list(result.values())[0]
        # Cache as parquet for next time
        df.to_parquet(parquet, index=False)
        log.info("Cached parquet: %s", parquet.name)

    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=random_state).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SWaT — Secure Water Treatment A4/A5 Jul 2019
# ─────────────────────────────────────────────────────────────────────────────

# SWaT A4/A5 Jul 2019 attack windows (from the official attack-documentation PDF
# "SWaT data collection_20-07-2019 v2.pdf"). Times are GMT+8 on 2019-07-20; the
# data timestamps are GMT+0, so we convert each window to UTC when labelling.
# Six attacks were carried out 3:08 PM--4:16 PM (GMT+8); plant ran normally before.
_SWAT_ATTACKS_GMT8 = [
    ("15:08:46", "15:10:31"),  # 1. FIT401 spoof 0.8->0.5
    ("15:15:00", "15:19:32"),  # 2. LIT301 spoof 835->1024
    ("15:26:57", "15:30:48"),  # 3. P601 OFF->ON
    ("15:38:50", "15:46:20"),  # 4. MV201 CLOSE->OPEN + P101 OFF->ON
    ("15:54:00", "15:56:00"),  # 5. MV501 OPEN->CLOSE
    ("16:02:56", "16:16:18"),  # 6. P301 ON->OFF
]
_SWAT_ATTACK_DATE = "2019-07-20"


def load_swat(
    max_samples: int = 80_000,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load the SWaT A4/A5 (Jul 2019) Secure Water Treatment testbed dataset.

    This file ships WITHOUT an inline attack label; the six attacks are documented
    only in the companion PDF (in GMT+8). We reconstruct binary labels by mapping
    those six attack windows onto the data's GMT+0 timestamps. Verified coverage:
    ~13% anomaly rate (1981 / 14996 rows), all six attacks represented.

    Returns X (float32, z-scored, temporal order preserved), y (0=normal/1=attack),
    feature_names. Temporal order is NOT shuffled because the Siamese encoder relies
    on temporal-proximity windows (continuous 1 Hz process; lag-1 autocorr approx 1.0).
    """
    parquet_path = SWAT_DIR / "SWaT_dataset_Jul19_v2.parquet"
    if parquet_path.exists():
        log.info("SWaT parquet fast-path")
        df = pd.read_parquet(parquet_path)
    else:
        xlsx_path = SWAT_DIR / "SWaT_dataset_Jul 19 v2.xlsx"
        if not xlsx_path.exists():
            candidates = list(SWAT_DIR.glob("*.xlsx"))
            if not candidates:
                raise FileNotFoundError(f"SWaT file not found in {SWAT_DIR}")
            xlsx_path = candidates[0]
        log.info("Loading SWaT XLSX (slow): %s", xlsx_path.name)
        df = pd.read_excel(xlsx_path, header=1, dtype=str)
        df.to_parquet(parquet_path, index=False)
        log.info("Cached SWaT parquet")

    # The first data row is a junk header artifact (every cell == "value"); drop it.
    ts_col = "GMT +0"
    if ts_col not in df.columns:
        ts_col = next((c for c in df.columns if "gmt" in c.lower() or "time" in c.lower()), df.columns[0])
    _first = str(df[ts_col].iloc[0]).strip().strip("'").lower()
    if _first in ("value", "timestamp"):
        df = df.iloc[1:].reset_index(drop=True)

    # Parse timestamps: values are wrapped in literal single quotes and have
    # variable fractional-second precision -> strip quotes and use ISO8601.
    ts = (df[ts_col].astype(str).str.strip().str.strip("'"))
    ts = pd.to_datetime(ts, format="ISO8601", utc=True, errors="coerce")
    n_bad_ts = int(ts.isna().sum())
    if n_bad_ts > 0:
        log.warning("SWaT: %d timestamps failed to parse (kept, label=0 unless in window)", n_bad_ts)

    # Build binary labels from the six documented attack windows (GMT+8 -> UTC).
    y = np.zeros(len(df), dtype=int)
    for s, e in _SWAT_ATTACKS_GMT8:
        s_utc = pd.Timestamp(f"{_SWAT_ATTACK_DATE} {s}", tz="Etc/GMT-8").tz_convert("UTC")
        e_utc = pd.Timestamp(f"{_SWAT_ATTACK_DATE} {e}", tz="Etc/GMT-8").tz_convert("UTC")
        in_window = ((ts >= s_utc) & (ts <= e_utc)).fillna(False).values
        y[in_window] = 1

    if y.sum() == 0:
        raise ValueError("SWaT labelling produced 0 anomalies — check timestamp parsing/windows.")

    # Feature columns: every non-timestamp column, coerced to numeric.
    # String state columns (Active/Inactive) are encoded 1/0 before coercion.
    feat_cols = []
    for col in df.columns:
        if col == ts_col:
            continue
        s = df[col]
        if s.dtype == object:
            s = (s.astype(str).str.strip()
                   .replace({"Active": "1", "Inactive": "0",
                             "Normal": "1", "Attack": "0"}))
        s = pd.to_numeric(s, errors="coerce")
        if s.notna().sum() > 100 and s.std(skipna=True) > 0:
            df[col] = s
            feat_cols.append(col)

    if not feat_cols:
        raise ValueError("SWaT: no usable numeric feature columns found.")

    # Fill any residual NaNs in features with column medians (keep all rows/labels).
    feat = df[feat_cols].apply(lambda c: c.fillna(c.median()))
    X_raw = feat.values.astype(np.float32)

    # Subsample only if needed; keep sorted indices to preserve temporal order.
    rng = np.random.default_rng(random_state)
    if max_samples and len(X_raw) > max_samples:
        idx = np.sort(rng.choice(len(X_raw), size=max_samples, replace=False))
        X_raw = X_raw[idx]
        y     = y[idx]

    X = _zscore(X_raw)
    log.info("SWaT loaded: %d samples, %d features, %.1f%% attack",
             len(X), X.shape[1], 100 * y.mean())
    return X, y, feat_cols


# ─────────────────────────────────────────────────────────────────────────────
# SKAB — Skoltech Anomaly Benchmark
# ─────────────────────────────────────────────────────────────────────────────

def load_skab(
    max_samples: int = 46_000,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load all SKAB CSV files (valve1/, valve2/, other/, anomaly-free/).
    Returns X (float32, z-scored), y (0/1), feature_names.
    """
    all_dfs = []

    subdirs = ["anomaly-free", "valve1", "valve2", "other"]
    for sub in subdirs:
        sub_path = SKAB_DIR / sub
        if not sub_path.exists():
            continue
        for csv_file in sorted(sub_path.glob("*.csv")):
            try:
                df = pd.read_csv(csv_file, sep=";", parse_dates=["datetime"],
                                 index_col="datetime", on_bad_lines="skip")
            except Exception:
                try:
                    df = pd.read_csv(csv_file, sep=",", on_bad_lines="skip")
                except Exception:
                    continue
            all_dfs.append(df)

    if not all_dfs:
        raise FileNotFoundError(f"No SKAB CSVs found in {SKAB_DIR}")

    data = pd.concat(all_dfs, ignore_index=True)

    # Anomaly column names used in SKAB
    anomaly_col = None
    for col in ["anomaly", "Anomaly", "changepoint", "Changepoint"]:
        if col in data.columns:
            anomaly_col = col
            break

    feat_cols = [c for c in data.columns
                 if c not in {"anomaly", "Anomaly", "changepoint", "Changepoint", "datetime"}
                 and data[c].dtype in [np.float64, np.float32, np.int64, np.int32]]

    # Coerce to numeric
    for col in feat_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=feat_cols).reset_index(drop=True)

    if anomaly_col is not None:
        y = data[anomaly_col].fillna(0).astype(int).values
    else:
        y = np.zeros(len(data), dtype=int)

    X_raw = data[feat_cols].values.astype(np.float32)

    rng = np.random.default_rng(random_state)
    if max_samples and len(X_raw) > max_samples:
        idx = rng.choice(len(X_raw), size=max_samples, replace=False)
        idx.sort()
        X_raw = X_raw[idx]
        y     = y[idx]

    X = _zscore(X_raw)
    log.info("SKAB loaded: %d samples, %d features, %.1f%% anomaly",
             len(X), X.shape[1], 100 * y.mean())
    return X, y, feat_cols


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _zscore(X: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(X).astype(np.float32)


def load_iiot(
    max_samples: int = 80_000,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load Smart Manufacturing IIoT dataset (100k samples, 50 machines).

    Sensor features: temperature, vibration, humidity, pressure, energy_consumption,
    predicted_remaining_life, machine_status, downtime_risk, maintenance_required.
    Label: anomaly_flag (binary 0/1, ~9% anomaly rate).

    The machine_id column is NOT used as a feature; it is used externally as
    the natural Dirichlet-like partition basis for federated split.
    """
    path = DATASET_ROOT / "IIOT_dataset" / "smart_manufacturing_data.csv"
    if not path.exists():
        raise FileNotFoundError(f"IIOT dataset not found: {path}")

    df = pd.read_csv(path)

    feat_cols = [
        "temperature", "vibration", "humidity", "pressure",
        "energy_consumption", "predicted_remaining_life",
        "machine_status", "downtime_risk", "maintenance_required",
    ]
    feat_cols = [c for c in feat_cols if c in df.columns]

    for col in feat_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=feat_cols).reset_index(drop=True)
    y  = df["anomaly_flag"].astype(int).values
    X_raw = df[feat_cols].values.astype(np.float32)

    rng = np.random.default_rng(random_state)
    if max_samples and len(X_raw) > max_samples:
        idx = rng.choice(len(X_raw), size=max_samples, replace=False)
        idx.sort()
        X_raw = X_raw[idx]
        y     = y[idx]

    X = _zscore(X_raw)
    log.info("IIOT loaded: %d samples, %d features, %.1f%% anomaly",
             len(X), X.shape[1], 100 * y.mean())
    return X, y, feat_cols


# ─────────────────────────────────────────────────────────────────────────────
# NASA CMAPSS — Commercial Modular Aero-Propulsion System Simulation
# ─────────────────────────────────────────────────────────────────────────────

# Sensors that are constant across operating conditions (remove before analysis)
_CMAPSS_CONSTANT_SENSORS = {"sensor1", "sensor5", "sensor6", "sensor10",
                              "sensor16", "sensor18", "sensor19"}

_CMAPSS_DIR = (Path(__file__).resolve().parent.parent.parent
               / "Dataset" / "nasa_datasets"
               / "6. Turbofan Engine Degradation Simulation Data Set")


def load_cmapss(
    max_samples: int = 60_000,
    rul_threshold: int = 30,
    fd_sets: tuple = ("FD001", "FD002"),
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load NASA CMAPSS turbofan engine degradation dataset.

    Returns (X, y, feature_names) where:
      y = 1 if Remaining Useful Life (RUL) < rul_threshold (default 30 cycles)
          → "anomaly" = engine is in its final degradation phase
      X = 14 non-constant sensor features, z-score normalised

    Uses FD001 (single operating condition, 100 engines) and
    FD002 (6 operating conditions, 260 engines) by default (~74k rows).
    Constant sensors [1,5,6,10,16,18,19] are removed per standard practice.

    Temporal redundancy: consecutive readings of healthy engines are similar
    (raw cosine similarity ≈ 0.59) but NOT as high as SKAB (≈0.99 before dedup).
    This dataset is designed to test DP-FL anomaly detection under LOW redundancy conditions.
    """
    base = _CMAPSS_DIR
    cols = ["engine_id", "cycle", "s1", "s2", "s3"] + \
           [f"sensor{i}" for i in range(1, 22)]

    frames = []
    for fd in fd_sets:
        path = base / f"train_{fd}.txt"
        if not path.exists():
            log.warning("CMAPSS %s not found at %s; skipping", fd, path)
            continue
        df = pd.read_csv(path, sep=" ", header=None).dropna(axis=1)
        n_cols = min(len(cols), df.shape[1])
        df.columns = cols[:n_cols]
        df["fd"] = fd
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No CMAPSS train files found in {base}")

    train = pd.concat(frames, ignore_index=True)

    # Compute per-engine RUL and binary anomaly label
    train["max_cycle"] = train.groupby(["fd", "engine_id"])["cycle"].transform("max")
    train["rul"]       = train["max_cycle"] - train["cycle"]
    train["anomaly"]   = (train["rul"] < rul_threshold).astype(int)

    # Feature columns: 14 variable sensors (remove 7 constant ones)
    sensor_cols = [c for c in cols if c.startswith("sensor")
                   and c not in _CMAPSS_CONSTANT_SENSORS
                   and c in train.columns]

    for col in sensor_cols:
        train[col] = pd.to_numeric(train[col], errors="coerce")
    train = train.dropna(subset=sensor_cols).reset_index(drop=True)

    y = train["anomaly"].values.astype(int)
    X_raw = train[sensor_cols].values.astype(np.float32)

    rng = np.random.default_rng(random_state)
    if max_samples and len(X_raw) > max_samples:
        idx = rng.choice(len(X_raw), size=max_samples, replace=False)
        idx.sort()
        X_raw = X_raw[idx]
        y     = y[idx]

    X = _zscore(X_raw)
    log.info("CMAPSS loaded: %d samples, %d sensors, %.1f%% anomaly (RUL<%d)",
             len(X), X.shape[1], 100 * y.mean(), rul_threshold)
    return X, y, sensor_cols


# ─────────────────────────────────────────────────────────────────────────────
# HAI — HIL-based Augmented ICS Security dataset (power + thermal testbed)
# ─────────────────────────────────────────────────────────────────────────────

# The HAI download ships several releases (hai-20.07 / 21.03 / 22.04 / 23.05), each with a
# DIFFERENT sensor schema, so we pin ONE release rather than mixing them. Within a release
# the labelled TEST files (test*.csv) form a single temporal stream containing both normal
# operation and the injected attacks; the pure-normal train files are not needed because the
# test stream already supplies an abundant normal pool. Preference order below.
_HAI_BASE = DATASET_ROOT / "HAI"
_HAI_RELEASES = ["hai-22.04", "hai-21.03", "hai-20.07", "hai-23.05", "."]


def load_hai(
    max_samples: int = 40_000,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load the HAI (HIL-based Augmented ICS) dataset: a power-generation and thermal testbed
    coupling a boiler (P1, Emerson), a steam turbine (P2, GE Mark VIe), a pumped-storage
    hydropower unit (P3, Siemens), and a dSPACE HIL model (P4), with real injected attacks.
    A different ICS domain from water treatment (SWaT/SKAB) for cross-domain generalisation
    of the selection null.

    We pin a single release (hai-22.04 by default) and use its labelled TEST files as one
    temporally ordered stream (~3.3% attack across test1--test4). The timestamp column and the
    'Attack' label column are auto-detected ('attack_P1/P2/P3' are OR-ed if a single 'Attack'
    column is absent). Feature columns are the numeric process variables; temporal order is
    preserved (like SWaT) because the streams are strongly autocorrelated.

    Returns X (float32, z-scored), y (0=normal/1=attack), feature_names.
    """
    base = _HAI_BASE if _HAI_BASE.exists() else next(
        (d for d in [DATASET_ROOT / "hai", DATASET_ROOT / "HAI"] if d.exists()), None)
    if base is None:
        raise FileNotFoundError(
            f"HAI dataset not found. Place the official HAI release under {_HAI_BASE}")

    rel_dir = next((base / r for r in _HAI_RELEASES if (base / r).exists()
                    and list((base / r).glob("*test*.csv"))), None)
    if rel_dir is None:
        raise FileNotFoundError(f"No HAI release with test*.csv found under {base}")

    csvs = sorted(rel_dir.glob("*test*.csv"))
    log.info("HAI release: %s (%d test files)", rel_dir.name, len(csvs))

    frames = []
    for f in csvs:
        df = None
        for sep in (",", ";", "\t"):
            try:
                tmp = pd.read_csv(f, sep=sep, engine="python", on_bad_lines="skip")
                if tmp.shape[1] > 3:
                    df = tmp; break
            except Exception:
                continue
        if df is None:
            continue
        df.columns = [str(c).strip() for c in df.columns]
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"HAI test CSVs under {rel_dir} could not be parsed")

    data = pd.concat(frames, ignore_index=True)

    # Identify timestamp and label columns.
    ts_cols = [c for c in data.columns if c.lower() in ("time", "timestamp", "datetime")]
    atk_single = [c for c in data.columns if c.lower() == "attack"]
    atk_multi  = [c for c in data.columns if c.lower().startswith("attack_")]

    if atk_single:
        y = pd.to_numeric(data[atk_single[0]], errors="coerce").fillna(0)
        y = (y > 0).astype(int).values
    elif atk_multi:
        m = np.zeros(len(data), dtype=int)
        for c in atk_multi:
            m |= (pd.to_numeric(data[c], errors="coerce").fillna(0) > 0).astype(int).values
        y = m
    else:
        y = np.zeros(len(data), dtype=int)

    exclude = set(ts_cols) | set(atk_single) | set(atk_multi)
    feat_cols = []
    for c in data.columns:
        if c in exclude:
            continue
        s = pd.to_numeric(data[c], errors="coerce")
        if s.notna().sum() > 100 and s.std(skipna=True) > 0:
            data[c] = s
            feat_cols.append(c)

    if not feat_cols:
        raise ValueError("HAI: no usable numeric feature columns found.")

    feat = data[feat_cols].apply(lambda c: c.fillna(c.median()))
    X_raw = feat.values.astype(np.float32)

    # Subsample if needed; keep sorted indices to preserve temporal order.
    rng = np.random.default_rng(random_state)
    if max_samples and len(X_raw) > max_samples:
        idx = np.sort(rng.choice(len(X_raw), size=max_samples, replace=False))
        X_raw = X_raw[idx]; y = y[idx]

    X = _zscore(X_raw)
    log.info("HAI loaded: %d samples, %d features, %.1f%% attack",
             len(X), X.shape[1], 100 * y.mean())
    return X, y, feat_cols


DATASET_REGISTRY = {
    "tep":    load_tep,
    "swat":   load_swat,
    "skab":   load_skab,
    "iiot":   load_iiot,
    "cmapss": load_cmapss,
    "hai":    load_hai,
}
