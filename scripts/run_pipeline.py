#!/usr/bin/env python
"""
End-to-end MLB pipeline orchestrator
------------------------------------
Flow:
  RAW ingest (per year, with warmup)
  → feature build (calls data_ingest.build_feature_table)
  → EDA (optional)
  → prepare_features
  → model_train_allinone

Usage (PowerShell example):
  py scripts/run_pipeline.py --base-year 2024 ^
    --features-base out/mlb_features_2024.csv ^
    --features-combined out/mlb_features_combined.csv ^
    --prepared out/mlb_features_prepared.csv ^
    --model-out model_all_ts ^
    --run-eda --add-rolling --add-interactions
"""

from __future__ import annotations

import argparse, os, sys, importlib, inspect, shutil
from datetime import datetime, timedelta, date
import pandas as pd

# ---------------- utils ----------------
def ensure_dir(p: str) -> str:
    if p:
        os.makedirs(p, exist_ok=True)
    return p or "."

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
    # use UTC "today" (you can subtract 1 if you prefer to avoid same-day partials)
    return (datetime.utcnow() - timedelta(days=0)).date()

def discover_feature_builder(mod):
    """Return (callable, name) for a feature builder found in data_ingest, else (None, None)."""
    for name in (
        "build_feature_table",
        "build_features",
        "build_feature_model",
        "build_feature_table_model",
    ):
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn, name
    return None, None

def discover_year_ingest(mod):
    """Return a callable year ingestor if available; prefer public wrapper."""
    for name in ("build_raw_dataset_year", "_ingest_year_to_raw"):
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn, name
    return None, None

# ---------------- main orchestration ----------------
def main(a):
    for p in (
        os.path.dirname(a.raw),
        os.path.dirname(a.features_base),
        os.path.dirname(a.features_combined),
        os.path.dirname(a.prepared),
        a.model_out,
        "out",
    ):
        ensure_dir(p)

    # import user's data_ingest next to this script or on PYTHONPATH
    data_ingest = importlib.import_module("scripts.data_ingest" if os.path.exists("scripts/data_ingest.py") else "data_ingest")
    print("[diag] using data_ingest at:", inspect.getsourcefile(data_ingest))

    builder, builder_name = discover_feature_builder(data_ingest)
    ingestor, ingestor_name = discover_year_ingest(data_ingest)

    # 0) load existing artifacts
    combined = read_csv_safe(a.features_combined, parse_dates=[a.date_col])
    base2024 = read_csv_safe(a.features_base, parse_dates=[a.date_col])
    prev_end = max_game_date(combined, a.date_col)

    # If no combined yet but a base features file exists, seed it.
    if combined.empty and not base2024.empty:
        print(f"[init] Seeding combined features from base → {a.features_base}")
        base2024.to_csv(a.features_combined, index=False)
        combined = base2024.copy()
        prev_end = max_game_date(combined, a.date_col)

    # 1) Ingest RAW per year (with warmup) so builder has everything
    update_to = datetime.strptime(a.update_to, "%Y-%m-%d").date() if a.update_to else safe_today()
    y0, y1 = a.base_year, update_to.year
    if ingestor:
        print(f"[ingest] using {ingestor_name} for years {y0}..{y1}")
        for y in range(y0, y1 + 1):
            # keep kwargs flexible for either wrapper signature
            try:
                ingestor(
                    year=y,
                    backfill_days=a.ingest_backfill_days,
                    out_csv=a.raw,
                    out_path=a.raw,  # works for _ingest_year_to_raw
                    include_finished=True,
                    fetch_handedness=True,
                    max_workers=a.max_workers,
                )
            except TypeError:
                # try minimal signature
                ingestor(y)
    else:
        print("[warn] No year ingestor exposed in data_ingest; skipping raw ingest step.")

    # 2) Decide feature span with warmup
    warmup = timedelta(days=a.feature_backfill_days)
    if prev_end is None:
        start_build = date(a.base_year, 1, 1) - warmup
        print(f"[span] No combined features yet; building from {start_build} → {update_to} (warmup {a.feature_backfill_days}d)")
    else:
        start_build = max(prev_end + timedelta(days=1) - warmup, date(a.base_year, 1, 1) - warmup)
        print(f"[span] Combined through {prev_end}; building from {start_build} → {update_to} (warmup {a.feature_backfill_days}d)")

    # 3) Build features via data_ingest builder (preferred)
    new_feat = pd.DataFrame()
    if start_build <= update_to:
        if builder:
            print(f"[builder] calling {builder_name}(start,end,out_csv,verbose)")
            tmp_csv = os.path.join("out", f"_tmp_features_{start_build}_to_{update_to}.csv").replace(":", "-")
            out_path = builder(
                start=start_build.strftime("%Y-%m-%d"),
                end=update_to.strftime("%Y-%m-%d"),
                out_csv=tmp_csv,
                verbose=True,
                max_workers=a.max_workers,
            )
            built = read_csv_safe(out_path, parse_dates=[a.date_col])
            if prev_end is not None and not built.empty:
                gd = pd.to_datetime(built[a.date_col], errors="coerce").dt.date
                built = built.loc[gd > prev_end].copy()
            new_feat = built
            print(f"[builder] new rows post-filter: {len(new_feat):,}")
        else:
            # Fallback: minimal, leak-safe features from raw (date/id/target only)
            print("[fallback] data_ingest has no feature builder; writing minimal features from raw.")
            raw_df = read_csv_safe(a.raw, parse_dates=["game_date", "game_datetime"])
            keep = [c for c in (a.date_col, a.id_col, a.target) if c in raw_df.columns]
            new_feat = raw_df[keep].copy() if keep else pd.DataFrame()
    else:
        print("[span] Nothing to build; up to date.")

    # 4) Append + dedupe combined by game_pk
    if not new_feat.empty:
        current = read_csv_safe(a.features_combined, parse_dates=[a.date_col])
        merged = pd.concat([current, new_feat], ignore_index=True) if not current.empty else new_feat
        if "game_pk" in merged.columns:
            sort_cols = [c for c in [a.date_col, "game_pk"] if c in merged.columns]
            if sort_cols:
                merged = merged.sort_values(sort_cols)
            merged = merged.drop_duplicates(subset=["game_pk"], keep="last")
        else:
            merged = merged.drop_duplicates()
        merged.to_csv(a.features_combined, index=False)
        print(f"[combine] wrote → {a.features_combined}  (rows: {len(merged):,}, cols: {merged.shape[1]})")
    else:
        print("[combine] No new rows to append.")

    # 5) EDA (optional)
    if a.run_eda:
        eda_path = a.eda_script if os.path.exists(a.eda_script) else os.path.join("scripts", a.eda_script)
        if os.path.exists(eda_path):
            print("\n== EDA ==")
            cmd = f'{a.pybin} "{eda_path}" --data "{a.features_combined}" --date-col {a.date_col} --id-col {a.id_col} --target {a.target}'
            print("[exec]", cmd)
            code = os.system(cmd)
            if code != 0:
                print("!! EDA failed (continuing).")
        else:
            print(f"[warn] EDA script not found at {eda_path}; skipping.")

    # 6) Prepare features
    prep_path = a.prepare_script if os.path.exists(a.prepare_script) else os.path.join("scripts", a.prepare_script)
    print("\n== PREPARE FEATURES ==")
    cmd = f'{a.pybin} "{prep_path}" --data "{a.features_combined}" --out "{a.prepared}" --date-col {a.date_col} --id-col {a.id_col} --target {a.target}'
    print("[exec]", cmd)
    code = os.system(cmd)
    if code != 0:
        print("!! prepare_features failed.")
        sys.exit(3)

    # 7) Model training
    model_path = a.model_script if os.path.exists(a.model_script) else os.path.join("scripts", a.model_script)
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
        (f'--home-odds-col {a.home_odds_col}' if a.home_odds_col else ""),
        (f'--away-odds-col {a.away_odds_col}' if a.away_odds_col else ""),
        f'--min-edge {a.min_edge:.3f}',
        f'--kelly-cap {a.kelly_cap:.3f}',
    ]
    cmd = f'{a.pybin} "{model_path}" ' + " ".join([t for t in train_args if t])
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

# ---------------- CLI ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # IO
    ap.add_argument("--raw", default="out/raw_games.csv")
    ap.add_argument("--features-base", default="out/mlb_features_2024.csv",
                    help="Existing 2024 features to seed combined on first run.")
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
    ap.add_argument("--ingest-backfill-days", type=int, default=35, help="Raw ingest warmup days per season")
    ap.add_argument("--feature-backfill-days", type=int, default=35, help="Warmup days for rolling features")
    # Concurrency
    ap.add_argument("--max-workers", type=int, default=20)
    # Script paths/binaries
    ap.add_argument("--pybin", default="python")  # use "py" on Windows if preferred
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

