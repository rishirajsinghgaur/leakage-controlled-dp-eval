r"""SWaT December-2015 (A1 & A2) replication of the core selection-vs-random characterization.

Independent SWaT release from the July-2019 data used in the main sweep: a different
year, a different attack campaign (36 attacks), and the full 51-tag physical record.
Runs the IDENTICAL code path as experiments/characterization.py (same encoder, same
tdedup/FPS selectors, same DP-SGD dp_eval, same swat hyperparameters) so the result is
directly comparable. Writes results/swat2015_characterization.json. Does not touch any
committed result.
"""
import sys, json, logging
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.ERROR)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models.siamese import SiameseEncoder, train_siamese
from experiments.characterization import dp_eval, fps_coreset
from experiments.run_full_paper_sweep import DS_CFG
from experiments.principled_method import seq_temporal_dedup

XLSX = ROOT.parent / "Dataset" / "swat" / "SWaT.A1 & A2_Dec 2015" / "Physical" / "SWaT_Dataset_Attack_v0.xlsx"
CACHE = ROOT / "results" / "cache_swat2015.npz"
OUT = ROOT / "results" / "swat2015_characterization.json"
SEEDS = [0, 1, 2]; EPS = [0.5, 1.0, 2.0, 4.0]
MAX_SAMPLES = 20_000; RANDOM_STATE = 42


def load_swat2015():
    """Mirror data.loaders.load_swat: z-scored numeric features, y=1 for Attack,
    temporal-order-preserving subsample to MAX_SAMPLES. Cache to npz after first read."""
    if CACHE.exists():
        z = np.load(CACHE); return z["X"].astype("float32"), z["y"].astype("int64")
    import pandas as pd
    print(f"Reading {XLSX.name} (~116 MB, slow)...", flush=True)
    df = pd.read_excel(XLSX, header=1)                       # row 1 blank, row 2 header
    df.columns = [str(c).strip() for c in df.columns]
    label_col = next(c for c in df.columns if c.lower().replace(" ", "") in ("normal/attack", "normal/attack "))
    ts_col = next(c for c in df.columns if "time" in c.lower())
    y = (df[label_col].astype(str).str.strip().str.lower() != "normal").astype(int).values
    feat_cols = []
    for col in df.columns:
        if col in (ts_col, label_col):
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() > 100 and s.std(skipna=True) > 0:
            df[col] = s.fillna(s.median()); feat_cols.append(col)
    X_raw = df[feat_cols].values.astype(np.float32)
    rng = np.random.default_rng(RANDOM_STATE)
    if len(X_raw) > MAX_SAMPLES:                             # sorted -> preserve temporal order
        idx = np.sort(rng.choice(len(X_raw), size=MAX_SAMPLES, replace=False))
        X_raw, y = X_raw[idx], y[idx]
    X = StandardScaler().fit_transform(X_raw).astype(np.float32)
    CACHE.parent.mkdir(exist_ok=True)
    np.savez_compressed(CACHE, X=X, y=y)
    print(f"Cached: {len(X)} samples, {X.shape[1]} features, {100*y.mean():.1f}% attack "
          f"({int(y.sum())}/{len(y)})", flush=True)
    return X, y


def main():
    cfg = dict(DS_CFG["swat"]); cfg["siamese_epochs"] = min(cfg["siamese_epochs"], 6)
    X, y = load_swat2015()
    print(f"loaded swat2015: n={len(X)} d={X.shape[1]} attack={100*y.mean():.1f}%", flush=True)
    idx = np.arange(len(X)); tr, te = train_test_split(idx, test_size=0.2, random_state=0, stratify=y); tr = np.sort(tr)
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    enc = SiameseEncoder(Xtr.shape[1], cfg["siamese_emb"])
    enc = train_siamese(enc, Xtr, ytr, epochs=cfg["siamese_epochs"], window=cfg["siamese_window"])
    Xn = Xtr[ytr == 0]
    if len(Xn) > 6000: Xn = Xn[:6000]
    E = enc.encode(Xn)
    kept_t, _ = seq_temporal_dedup(Xn, enc, keep_quantile=0.5); Xd = Xn[kept_t]; budget = len(Xd)
    Xf = Xn[fps_coreset(E, budget)]
    print(f"normal={len(Xn)} budget={budget} (rho={budget/len(Xn):.3f})", flush=True)
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(1000 + seed)
        Xr = Xn[np.sort(rng.choice(len(Xn), size=budget, replace=False))]
        for eps in EPS:
            for mode, Xt in [("full", Xn), ("random", Xr), ("tdedup", Xd), ("fps", Xf)]:
                f1, ap = dp_eval(Xt, Xte, yte, cfg, eps, seed)
                rows.append({"dataset": "swat2015", "mode": mode, "epsilon": eps, "seed": seed,
                             "f1": round(float(f1), 4), "auprc": round(float(ap), 4)})
                print(f"  swat2015 {mode} eps={eps} s={seed}: f1={f1:.4f} ap={ap:.4f}", flush=True)
                OUT.write_text(json.dumps(rows, indent=2))
    # summary: mean F1 per (mode, eps), and does any learned selector beat random?
    import statistics as st
    print("\n=== SUMMARY (mean F1 over seeds) ===", flush=True)
    beats = 0; total = 0
    for eps in EPS:
        cell = {m: st.mean(r["f1"] for r in rows if r["mode"] == m and r["epsilon"] == eps)
                for m in ["full", "random", "tdedup", "fps"]}
        for m in ["tdedup", "fps"]:
            total += 1; beats += cell[m] > cell["random"] + 1e-9
        print(f"  eps={eps}: full={cell['full']:.3f} random={cell['random']:.3f} "
              f"tdedup={cell['tdedup']:.3f} fps={cell['fps']:.3f}", flush=True)
    print(f"\nlearned-selector-beats-random cells: {beats}/{total}", flush=True)
    print("VERDICT:", "null holds (selection does not beat random)" if beats == 0
          else "some selector beat random - investigate", flush=True)


if __name__ == "__main__":
    main()
