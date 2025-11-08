#!/usr/bin/env python
"""
End-to-end MLB pipeline runner (incremental updates)
---------------------------------------------------
Flow:
  data_ingest  ->  eda_scaffold  ->  prepare_features  ->  model_train_allinone

This runner is signature-aware and robust:
- If data_ingest exposes a feature builder (build_feature_table / build_features / build_feature_model),
  we call it with only the arguments it accepts.
- If not, we fall back to a leak-safe feature builder from out/raw_games.csv.

Outputs
- out/raw_games.csv                      (raw ingest across years)
- out/mlb_features_combined.csv          (ever-growing combined features, deduped by game_pk)
- out/mlb_features_prepared.csv          (prepared for modeling)
- model_all_ts/                          (reports, figs, and CSVs from model_train_allinone)
"""

import argparse, os, sys, inspect
from datetime import datetime, timedelta, date
from pathlib import Path
import importlib
import pandas as pd
import numpy as np

# ---------------- utils ----------------
def ensure_dir(p: str) -> str:
    if p:
        os.makedirs(p, exist_ok=True)
    return p

def read_csv_safe(path: str, parse_dates=None) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, parse_dates=parse_dates or [])
    except Exception:
        return pd.DataFrame()

def max_game_date(df: pd.DataFrame, date_col: str) -> date | None:
    if df.empty or date_col not in df.columns:
        return None
    s = pd.to_datetime(df[date_col], errors="coerce").dt.date
    return None if s.isna().all() else s.max()

def safe_today() -> date:
    # use UTC today; change to -1 day if you want to avoid partial same-day games
    return datetime.utcnow().date()

def call_if_accepts(fn, **kwargs):
    """Call fn with only the kwargs it accepts (by name)."""
    sig = inspect.signature(fn)
    allowed = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(**allowed)

# ---------------- leak-safe fallback feature builder ----------------
LEAK_SUBSTR = (
    "score", "result", "win_prob", "prob", "implied", "moneyline",
    "odds", "final", "post", "series", "outcome", "label", "target"
)

def fallback_build_features_from_raw(
    raw_csv: str, out_csv: str, date_col: str, id_col: str, target: str,
    start_date: str, end_date: str
) -> str:
    """
    Build a minimal, leak-safe feature set directly from raw_games.csv.
    - Keeps {date_col,id_col,target} + numeric features w/ data.
    - Drops obvious leakage columns (odds/scores/results/etc.).
    - Filters to [start_date, end_date].
    """
    if not os.path.exists(raw_csv):
        raise RuntimeError(f"[fallback] raw file missing: {raw_csv}")

    df = pd.read_csv(raw_csv, low_memory=False)
    # normalize/locate date column
    if date_col not in df.columns:
        for alt in ["date", "gameDate", "game_datetime", "start_time", "start_date"]:
            if alt in df.columns:
                date_col = alt
                break
    if date_col not in df.columns:
        raise RuntimeError(f"[fallback] cannot find a date column in raw (looked for {date_col})")

    # filter by date span
    dt = pd.to_datetime(df[date_col], errors="coerce")
    mask = (dt >= pd.to_datetime(start_date)) & (dt <= pd.to_datetime(end_date))
    df = df.loc[mask].copy()

    # try to ensure id/target exist
    if id_col not in df.columns:
        for alt in ["game_id", "gamePk", "id"]:
            if alt in df.columns: id_col = alt; break
    if id_col not in df.columns:
        # fabricate a stable id if truly missing
        df[id_col] = np.arange(len(df))

    if target not in df.columns:
        # best-effort target from scores if available
        h, a = None, None
        for cand in ["home_score", "homeFinal", "home_runs", "homeRuns"]:
            if cand in df.columns: h = cand; break
        for cand in ["away_score", "awayFinal", "away_runs", "awayRuns"]:
            if cand in df.columns: a = cand; break
        if h and a:
            df[target] = (pd.to_numeric(df[h], errors="coerce") > pd.to_numeric(df[a], errors="coerce")).astype(int)
        else:
            raise RuntimeError(f"[fallback] target '{target}' not found and cannot be derived from scores.")

    # select leak-safe numeric features
    feats = []
    for c in df.columns:
        lc = c.lower()
        if c in (date_col, id_col, target):
            continue
        if any(s in lc for s in LEAK_SUBSTR):
            continue
        if "_score" in lc or lc.endswith("_w") or lc.endswith("_l"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any():
            feats.append(c)

    keep_cols = [date_col, id_col, target] + feats
    out = df[keep_cols].copy().sort_values(date_col).reset_index(drop=True)
    ensure_dir(os.path.dirname(out_csv) or ".")
    out.to_csv(out_csv, index=False)
    print(f"[fallback] wrote features → {out_csv} | rows={len(out):,} | feats={len(feats)}")
    return out_csv

# ---------------- main ----------------
def main(a):
    # IO dirs
    ensure_dir(os.path.dirname(a.raw) or ".")
    ensure_dir(os.path.dirname(a.features_combined) or ".")
    ensure_dir(os.path.dirname(a.prepared) or ".")
    ensure_dir(a.model_out)

    # import data_ingest (may or may not export feature builders)
    data_ingest = importlib.import_module("data_ingest")
    print("[diag] using data_ingest at:", inspect.getsourcefile(data_ingest))

    # ---------- 0) Ingest raw by years ----------
    update_to = datetime.strptime(a.update_to, "%Y-%m-%d").date() if a.update_to else safe_today()
    y0, y1 = a.base_year, update_to.year
    print(f"\n== RAW INGEST: ensuring years {y0}..{y1} are present in {a.raw} ==")

    # Prefer your _ingest_year_to_raw if it exists; otherwise just skip (assume you create raw elsewhere)
    ingest_fn = getattr(data_ingest, "_ingest_year_to_raw", None)
    if ingest_fn is not None:
        for y in range(y0, y1 + 1):
            call_if_accepts(
                ingest_fn,
                year=y,
                backfill_days=a.ingest_backfill_days,
                out_path=a.raw,
                include_finished=True,
                fetch_handedness=True,
                max_workers=a.max_workers,
            )
    else:
        print("[warn] data_ingest._ingest_year_to_raw not found — skipping raw refresh (expect raw already on disk).")

    raw_df = read_csv_safe(a.raw, parse_dates=["game_date", "game_datetime"])
    if raw_df.empty:
        print("!! No raw data available after ingest; aborting.")
        sys.exit(2)

    # ---------- 1) Compute span to (re)build features ----------
    combined_df = read_csv_safe(a.features_combined, parse_dates=[a.date_col])
    prev_end = max_game_date(combined_df, a.date_col)
    if prev_end is None:
        start_build = date(a.base_year, 1, 1) - timedelta(days=a.feature_backfill_days)
        print(f"[span] No combined features yet; building from {start_build} → {update_to}")
    else:
        start_build = max(
            prev_end + timedelta(days=1) - timedelta(days=a.feature_backfill_days),
            date(a.base_year, 1, 1) - timedelta(days=a.feature_backfill_days),
        )
        print(f"[span] Combined through {prev_end}; building NEW from {start_build} → {update_to} (warmup {a.feature_backfill_days}d)")

    if start_build > update_to:
        print("[span] Nothing new to build.")
        new_csv = None
    else:
        # ---------- 2) Build features for span (real builder if available; else fallback) ----------
        tmp_new_csv = os.path.join("out", f"_tmp_features_{start_build}_to_{update_to}.csv").replace(":", "-")
        ensure_dir("out")

        builder = None
        for name in ("build_feature_table", "build_features", "build_feature_model"):
            if hasattr(data_ingest, name):
                builder = getattr(data_ingest, name)
                break

        if builder is None:
            print("[warn] data_ingest has no feature builder; using leak-safe fallback from raw.")
            fallback_build_features_from_raw(
                raw_csv=a.raw,
                out_csv=tmp_new_csv,
                date_col=a.date_col,
                id_col=a.id_col,
                target=a.target,
                start_date=start_build.strftime("%Y-%m-%d"),
                end_date=update_to.strftime("%Y-%m-%d"),
            )
            new_csv = tmp_new_csv
        else:
            print(f"[info] using data_ingest.{builder.__name__} for feature build")
            ret = call_if_accepts(
                builder,
                start=start_build.strftime("%Y-%m-%d"),
                start_date=start_build.strftime("%Y-%m-%d"),
                end=update_to.strftime("%Y-%m-%d"),
                end_date=update_to.strftime("%Y-%m-%d"),
                out_csv=tmp_new_csv,
                out=tmp_new_csv,
                verbose=True,
            )
            # Some builders return a path; if so, use it
            if isinstance(ret, (list, tuple)) and len(ret) > 0 and isinstance(ret[-1], str):
                new_csv = ret[-1]
            elif isinstance(ret, str) and os.path.exists(ret):
                new_csv = ret
            else:
                new_csv = tmp_new_csv  # trust our requested output

    # ---------- 3) Append/dedupe into combined ----------
    if new_csv and os.path.exists(new_csv):
        new_feat = read_csv_safe(new_csv, parse_dates=[a.date_col])
        if prev_end is not None and not new_feat.empty:
            gd = pd.to_datetime(new_feat[a.date_col], errors="coerce").dt.date
            new_feat = new_feat.loc[gd > prev_end].copy()
        if new_feat.empty:
            print("[combine] No post-warmup new rows to append.")
        else:
            combined_old = read_csv_safe(a.features_combined, parse_dates=[a.date_col])
            combined_new = pd.concat([combined_old, new_feat], ignore_index=True) if not combined_old.empty else new_feat
            if "game_pk" in combined_new.columns:
                combined_new = combined_new.sort_values([a.date_col, "game_pk"])
                combined_new = combined_new.drop_duplicates(subset=["game_pk"], keep="last")
            else:
                combined_new = combined_new.drop_duplicates()
            combined_new = combined_new.sort_values(a.date_col).reset_index(drop=True)
            combined_new.to_csv(a.features_combined, index=False)
            print(f"[combine] wrote → {a.features_combined} | rows={len(combined_new):,}")
    else:
        print("[combine] No new feature CSV produced; keeping existing combined as-is.")

    # ---------- 4) EDA (optional) ----------
    if a.run_eda:
        print("\n== EDA ==")
        cmd = f'{a.pybin} {a.eda_script} --data "{a.features_combined}" --date-col {a.date_col} --id-col {a.id_col} --target {a.target}'
        print("[exec]", cmd)
        if os.system(cmd) != 0:
            print("!! EDA failed (continuing).")

    # ---------- 5) Prepare features ----------
    print("\n== PREPARE FEATURES ==")
    cmd = f'{a.pybin} {a.prepare_script} --in "{a.features_combined}" --out "{a.prepared}" --date-col {a.date_col} --id-col {a.id_col} --target {a.target}'
    print("[exec]", cmd)
    if os.system(cmd) != 0:
        print("!! prepare_features failed.")
        sys.exit(3)

    # ---------- 6) Model training ----------
    print("\n== MODEL TRAINING ==")
    train_args = [
        f'--data "{a.prepared}"',
        f'--date-col {a.date_col}',
        f'--id-col {a.id_col}',
        f'--target {a.target}',
        f'--outdir "{a.model_out}"',
        f'--topk {a.topk}',
        f'--threshold-strategy {a.threshold_strategy}',
        "--add-rolling" if a.add_rolling else "",
        "--add-interactions" if a.add_interactions else "",
        f'--calibration {a.calibration}',
        f'--walkforward {a.walkforward}',
        f'--n-splits {a.n_splits}',
        f'--home-odds-col {a.home_odds_col}' if a.home_odds_col else "",
        f'--away-odds-col {a.away_odds_col}' if a.away_odds_col else "",
        f'--min-edge {a.min_edge:.3f}',
        f'--kelly-cap {a.kelly_cap:.3f}',
    ]
    cmd = f'{a.pybin} {a.model_script} ' + " ".join([t for t in train_args if t])
    print("[exec]", cmd)
    if os.system(cmd) != 0:
        print("!! model_train_allinone failed.")
        sys.exit(4)

    print("\n✅ Pipeline complete.")
    print("Artifacts:")
    print("  Combined features:", a.features_combined)
    print("  Prepared dataset :", a.prepared)
    print("  Model out dir    :", a.model_out)

# ---------------- CLI ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # IO
    ap.add_argument("--raw", default="out/raw_games.csv")
    ap.add_argument("--features-combined", default="out/mlb_features_combined.csv")
    ap.add_argument("--prepared", default="out/mlb_features_prepared.csv")
    ap.add_argument("--model-out", default="model_all_ts")
    # Columns
    ap.add_argument("--date-col", default="game_date")
    ap.add_argument("--id-col", default="game_pk")
    ap.add_argument("--target", default="home_win")
    # Time span
    ap.add_argument("--base-year", type=int, default=2024)
    ap.add_argument("--update-to", type=str, default=None, help="YYYY-MM-DD (default: today UTC)")
    # Backfills
    ap.add_argument("--ingest-backfill-days", type=int, default=35)
    ap.add_argument("--feature-backfill-days", type=int, default=35)
    # Concurrency hint for ingest
    ap.add_argument("--max-workers", type=int, default=20)
    # Script paths/binaries
    ap.add_argument("--pybin", default="py")  # use "python" on mac/linux
    ap.add_argument("--eda-script", default="eda_scaffold.py")
    ap.add_argument("--prepare-script", default="prepare_features.py")
    ap.add_argument("--model-script", default="model_train_allinone.py")
    ap.add_argument("--run-eda", action="store_true")
    # Modeling knobs
    ap.add_argument("--topk", type=int, default=25)
    ap.add_argument("--threshold-strategy", choices=["f1","youden","balanced"], default="f1")
    ap.add_argument("--add-rolling", action="store_true")
    ap.add_argument("--add-interactions", action="store_true")
    ap.add_argument("--calibration", choices=["raw","platt","isotonic"], default="isotonic")
    ap.add_argument("--walkforward", choices=["monthly","ts","none"], default="monthly")
    ap.add_argument("--n-splits", type=int, default=6)
    # Betting (optional)
    ap.add_argument("--home-odds-col", default=None)
    ap.add_argument("--away-odds-col", default=None)
    ap.add_argument("--min-edge", type=float, default=0.02)
    ap.add_argument("--kelly-cap", type=float, default=0.05)
    args = ap.parse_args()
    main(args)
