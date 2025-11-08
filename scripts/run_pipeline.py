#!/usr/bin/env python
"""
End-to-end MLB pipeline runner (incremental updates)
---------------------------------------------------
Flow:
  data_ingest  ->  eda_scaffold  ->  prepare_features  ->  model_train_allinone

Usage (Windows PowerShell example):
  py run_pipeline.py --base-year 2024 ^
    --raw out/raw_games.csv ^
    --features-base out/mlb_features_2024.csv ^
    --features-combined out/mlb_features_combined.csv ^
    --prepared out/mlb_features_prepared.csv ^
    --model-out model_all_ts

Notes:
- We keep an always-growing COMBINED features file but ensure no duplicates (game_pk).
- Each update builds features only for the new span, with backfill warmup to keep rolling stats valid.
"""

from __future__ import annotations

import argparse, os, sys, json
from datetime import datetime, timedelta, date
import pandas as pd
from pathlib import Path
import importlib, inspect

# import your ingest module
data_ingest = importlib.import_module("data_ingest")

# ---------------- utils ----------------
def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def read_csv_safe(path: str, parse_dates=None) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, parse_dates=parse_dates or [])
    except Exception:
        return pd.DataFrame()

def max_game_date(df: pd.DataFrame, col: str) -> date | None:
    if col not in df.columns or df.empty:
        return None
    s = pd.to_datetime(df[col], errors="coerce").dt.date
    if s.isna().all():
        return None
    return s.max()

def normalize_dates(df: pd.DataFrame, cols=("game_date","game_datetime")) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def merge_dedupe_by_game_pk(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        base = new.copy()
    else:
        base = pd.concat([existing, new], ignore_index=True)
    if "game_datetime" in base.columns:
        base = base.sort_values(["game_pk", "game_datetime"], na_position="last")
    base = base.drop_duplicates(subset=["game_pk"], keep="last")
    return base

def safe_today() -> date:
    # You can choose to stop at yesterday to avoid same-day incomplete games:
    return (datetime.utcnow() - timedelta(days=0)).date()

def daterange_y(year_start: int, year_end: int):
    for y in range(year_start, year_end + 1):
        yield y

# ---------------- signature-aware shims (NEW) ----------------
def _safe_call_ingest_year(
    year: int,
    backfill_days: int,
    out_path: str,
    include_finished: bool = True,
    fetch_handedness: bool = True,
    max_workers: int = 20,
):
    """
    Call whichever ingest function exists and pass only the kwargs it accepts.
    Handles cases where your _ingest_year_to_raw doesn't accept max_workers, etc.
    """
    fn = getattr(data_ingest, "_ingest_year_to_raw", None) or getattr(data_ingest, "ingest_year_to_raw", None)
    if fn is None:
        raise RuntimeError("data_ingest.py is missing _ingest_year_to_raw/ingest_year_to_raw")

    sig = inspect.signature(fn)
    kwargs = {
        "year": year,
        "backfill_days": backfill_days,
        "out_path": out_path,
        "include_finished": include_finished,
        "fetch_handedness": fetch_handedness,
        "max_workers": max_workers,
    }
    allowed = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(**allowed)

def _safe_build_features(
    out_csv: str,
    start_str: str,
    end_str: str,
    verbose: bool = True,
):
    """
    Call build_feature_table (or similar) passing only the params it accepts,
    mapping start/end to start_date/end_date if needed.
    """
    fn = getattr(data_ingest, "build_feature_table", None) or \
         getattr(data_ingest, "build_features", None) or \
         getattr(data_ingest, "build_feature_model", None)
    if fn is None:
        raise RuntimeError("data_ingest.py is missing build_feature_table/build_features/build_feature_model")

    sig = inspect.signature(fn)
    kwargs = {"out_csv": out_csv, "verbose": verbose}

    # start/end argument names differ across your versions; map intelligently
    if "start_date" in sig.parameters:
        kwargs["start_date"] = start_str
    elif "start" in sig.parameters:
        kwargs["start"] = start_str

    if "end_date" in sig.parameters:
        kwargs["end_date"] = end_str
    elif "end" in sig.parameters:
        kwargs["end"] = end_str

    allowed = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(**allowed)

# ---------------- main orchestration ----------------
def main(a):
    ensure_dir(os.path.dirname(a.raw) or ".")
    ensure_dir(os.path.dirname(a.features_base) or ".")
    ensure_dir(os.path.dirname(a.features_combined) or ".")
    ensure_dir(os.path.dirname(a.prepared) or ".")
    ensure_dir(a.model_out)

    print("[diag] using data_ingest at:", inspect.getsourcefile(data_ingest))

    # 0) Read current artifacts (if they exist)
    raw_df      = read_csv_safe(a.raw, parse_dates=["game_date", "game_datetime"])
    features_df = read_csv_safe(a.features_combined, parse_dates=[a.date_col])

    # 1) Ingest/refresh RAW for each year up to --update-to
    update_to = datetime.strptime(a.update_to, "%Y-%m-%d").date() if a.update_to else safe_today()
    y0 = a.base_year
    y1 = update_to.year

    print(f"\n== RAW INGEST: ensuring years {y0}..{y1} are present in {a.raw} ==")
    for y in daterange_y(y0, y1):
        # Use signature-aware shim (fixes 'unexpected keyword argument max_workers')
        _safe_call_ingest_year(
            year=y,
            backfill_days=a.ingest_backfill_days,
            out_path=a.raw,
            include_finished=True,
            fetch_handedness=True,
            max_workers=a.max_workers,
        )

    # Reload raw after writes
    raw_df = read_csv_safe(a.raw, parse_dates=["game_date", "game_datetime"])
    if raw_df.empty:
        print("!! No raw data available after ingest; aborting.")
        sys.exit(2)

    # 2) Determine feature-build span
    combined_exists = os.path.exists(a.features_combined)
    if not combined_exists and os.path.exists(a.features_base):
        # initialize COMBINED from BASE (2024) on first run
        print(f"[init] Seeding combined features from base: {a.features_base}")
        base_2024 = read_csv_safe(a.features_base, parse_dates=[a.date_col])
        if base_2024.empty:
            print("!! Base 2024 features is empty; continuing without seeding.")
            features_df = pd.DataFrame()
        else:
            base_2024.to_csv(a.features_combined, index=False)
            features_df = base_2024.copy()

    prev_end = max_game_date(features_df, a.date_col)
    if prev_end is None:
        # no combined yet: build from Jan 1 of base_year, with warmup
        start_build = date(a.base_year, 1, 1) - timedelta(days=a.feature_backfill_days)
        print(f"[span] No combined features yet; building from {start_build} → {update_to}")
    else:
        # build from prev_end + 1 day, but extend back by warmup days for rolling stats
        start_build = max(prev_end + timedelta(days=1) - timedelta(days=a.feature_backfill_days),
                          date(a.base_year, 1, 1) - timedelta(days=a.feature_backfill_days))
        print(f"[span] Existing combined through {prev_end}. Building NEW features from {start_build} → {update_to} (warmup {a.feature_backfill_days}d)")

    if start_build > update_to:
        print("[span] Nothing new to build. Proceeding to EDA/prepare/model with current combined.")
        new_feat = pd.DataFrame()
    else:
        # 3) Build NEW features for the span (to a temp file)
        tmp_new_csv = os.path.join("out", f"_tmp_features_{start_build}_to_{update_to}.csv").replace(":", "-")
        os.makedirs("out", exist_ok=True)

        # Use signature-aware builder (fixes start vs start_date args)
        _safe_build_features(
            out_csv=tmp_new_csv,
            start_str=start_build.strftime("%Y-%m-%d"),
            end_str=update_to.strftime("%Y-%m-%d"),
            verbose=True,
        )

        new_feat = read_csv_safe(tmp_new_csv, parse_dates=[a.date_col])

        # Keep only truly NEW rows after prev_end (so we don't double-count warmup)
        if prev_end is not None and not new_feat.empty:
            gd = pd.to_datetime(new_feat[a.date_col], errors="coerce").dt.date
            new_feat = new_feat.loc[gd > prev_end].copy()

        print(f"[new] Built {len(new_feat):,} NEW feature rows (post-filter).")

    # 4) Append to COMBINED, dedupe by game_pk, sort by date
    if not new_feat.empty:
        combined_old = read_csv_safe(a.features_combined, parse_dates=[a.date_col])
        combined_new = pd.concat([combined_old, new_feat], ignore_index=True) if not combined_old.empty else new_feat
        if "game_pk" in combined_new.columns:
            combined_new = combined_new.sort_values([a.date_col, "game_pk"])
            combined_new = combined_new.drop_duplicates(subset=["game_pk"], keep="last")
        else:
            combined_new = combined_new.drop_duplicates()
        combined_new = combined_new.sort_values(a.date_col).reset_index(drop=True)
        combined_new.to_csv(a.features_combined, index=False)
        print(f"[combine] Wrote combined features → {a.features_combined}  (rows: {len(combined_new):,})")
    else:
        print("[combine] No new rows to append.")

    # 5) EDA (on COMBINED features)
    if a.run_eda:
        print("\n== EDA ==")
        cmd = f'{a.pybin} {a.eda_script} --data "{a.features_combined}" --date-col {a.date_col} --id-col {a.id_col} --target {a.target}'
        print("[exec]", cmd)
        code = os.system(cmd)
        if code != 0:
            print("!! EDA script failed (continuing).")

    # 6) Prepare features (encoding/cleanup for modeling)
    print("\n== PREPARE FEATURES ==")
    cmd = f'{a.pybin} {a.prepare_script} --in "{a.features_combined}" --out "{a.prepared}" --date-col {a.date_col} --id-col {a.id_col} --target {a.target}'
    print("[exec]", cmd)
    code = os.system(cmd)
    if code != 0:
        print("!! prepare_features failed.")
        sys.exit(3)

    # 7) Model training
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
    code = os.system(cmd)
    if code != 0:
        print("!! model_train_allinone failed.")
        sys.exit(4)

    print("\n✅ Pipeline complete.")
    print("Artifacts:")
    print("  Combined features:", a.features_combined)
    print("  Prepared dataset :", a.prepared)
    print("  Model out dir    :", a.model_out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # IO
    ap.add_argument("--raw", default="out/raw_games.csv")
    ap.add_argument("--features-base", default="out/mlb_features_2024.csv",
                    help="Your existing 2024 features CSV to seed the combined file on first run.")
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
    ap.add_argument("--ingest-backfill-days", type=int, default=35, help="For raw ingest year warmup")
    ap.add_argument("--feature-backfill-days", type=int, default=35, help="Warmup for rolling feature validity")
    # Concurrency for ingest
    ap.add_argument("--max-workers", type=int, default=20)
    # Script paths/binaries
    ap.add_argument("--pybin", default="py")  # use "python" on macOS/Linux
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
    # Betting optional
    ap.add_argument("--home-odds-col", default=None)
    ap.add_argument("--away-odds-col", default=None)
    ap.add_argument("--min-edge", type=float, default=0.02)
    ap.add_argument("--kelly-cap", type=float, default=0.05)

    args = ap.parse_args()
    main(args)
