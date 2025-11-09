#!/usr/bin/env python
"""
End-to-end MLB pipeline runner (incremental updates)
- Uses your scripts/data_ingest.py to ingest/build features (no leakage).
- Keeps an always-growing combined features CSV (dedup by game_pk).
- Runs EDA -> prepare_features -> model_train_allinone.

Usage (PowerShell):
  py scripts\\run_pipeline.py --base-year 2024 --run-eda --add-rolling --add-interactions
"""

from __future__ import annotations
import argparse, os, sys, json
from datetime import datetime, timedelta, date
from pathlib import Path
import pandas as pd
import importlib.util

# ---------------- Robust import of data_ingest ----------------
def _load_data_ingest():
    """
    Make `scripts.data_ingest` importable whether the runner is on CI or locally.
    Falls back to loading by exact file path if package import fails.
    """
    # 1) Try normal import if PYTHONPATH already contains repo root.
    try:
        import scripts.data_ingest as data_ingest  # type: ignore
        return data_ingest
    except Exception:
        pass

    # 2) Ensure repo root on sys.path, try again.
    here = Path(__file__).resolve()
    repo_root = here.parent.parent  # repo/
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        import scripts.data_ingest as data_ingest  # type: ignore
        return data_ingest
    except Exception:
        pass

    # 3) Final fallback: load by file path
    di_path = repo_root / "scripts" / "data_ingest.py"
    if not di_path.exists():
        raise ModuleNotFoundError("Could not import scripts.data_ingest and data_ingest.py not found.")
    spec = importlib.util.spec_from_file_location("data_ingest", str(di_path))
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    assert spec and spec.loader
    spec.loader.exec_module(mod)                 # type: ignore
    return mod

data_ingest = _load_data_ingest()
# ---------------------------------------------------------------


# ---------------- utils ----------------
def ensure_dir(p: str | os.PathLike) -> str:
    p = str(p)
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
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
    # Stop at today; change to -1 day if you prefer.
    return (datetime.utcnow()).date()

# ---------------- orchestrator ----------------
def main(a):
    # Ensure dirs
    for p in (a.raw, a.features_base, a.features_combined, a.prepared):
        ensure_dir(p)
    os.makedirs(a.model_out, exist_ok=True)

    print(f"[diag] using data_ingest at: {getattr(data_ingest, '__file__', 'unknown')}")

    # 0) Load any existing combined
    features_df = read_csv_safe(a.features_combined, parse_dates=[a.date_col])

    # 1) Choose update end date and (later) cap the window
    update_to = datetime.strptime(a.update_to, "%Y-%m-%d").date() if a.update_to else safe_today()

    # We'll compute start_build first based on existing data, then CAP to last N days.
    prev_end = max_game_date(features_df, a.date_col)
    if prev_end is None:
        start_build = date(a.base_year, 1, 1) - timedelta(days=a.feature_backfill_days)
        print(f"[span] No combined features yet; initial build from {start_build} → {update_to}")
    else:
        start_build = max(
            prev_end + timedelta(days=1) - timedelta(days=a.feature_backfill_days),
            date(a.base_year, 1, 1) - timedelta(days=a.feature_backfill_days),
        )
        print(f"[span] Existing combined through {prev_end}. Planned NEW features from {start_build} → {update_to} (warmup {a.feature_backfill_days}d)")

    # ---- NEW: hard cap incremental work to last N days ----
    cap_start = update_to - timedelta(days=a.max_update_days)
    if start_build < cap_start:
        print(f"[cap] Capping build window to last {a.max_update_days} days: {cap_start} → {update_to}")
        start_build = cap_start

    # 1b) Ingest/refresh RAW only for years that intersect our capped window (+ ingest warmup)
    # We extend a bit earlier for ingest, so feature builder has enough context.
    ingest_start = max(date(a.base_year, 1, 1), start_build - timedelta(days=a.ingest_backfill_days))
    y0, y1 = ingest_start.year, update_to.year
    print(f"\n== RAW INGEST: ensuring years {y0}..{y1} cover { ingest_start } → { update_to } (ingest backfill {a.ingest_backfill_days}d) ==")
    for year in range(y0, y1 + 1):
        data_ingest._ingest_year_to_raw(
            year=year,
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

    # 3) Build features for the (capped) span
    if start_build > update_to:
        new_feat = pd.DataFrame()
        print("[span] Nothing new to build.")
    else:
        tmp_new_csv = os.path.join("out", f"_tmp_features_{start_build}_to_{update_to}.csv").replace(":", "-")
        os.makedirs("out", exist_ok=True)

        # Prefer your own full feature builder if it exists
        build_fn = getattr(data_ingest, "build_feature_table", None)
        if callable(build_fn):
            built = build_fn(
                start=start_build.strftime("%Y-%m-%d"),
                end=update_to.strftime("%Y-%m-%d"),
                out_csv=tmp_new_csv,
                verbose=True,
            )
            if isinstance(built, (list, tuple)):
                tmp_new_csv = built[-1]
        else:
            # leak-safe minimal fallback directly from raw
            print("[warn] data_ingest has no feature builder; using leak-safe fallback from raw.")
            sub = raw_df.copy()
            sub = sub[(pd.to_datetime(sub["game_date"]).dt.date >= start_build) &
                      (pd.to_datetime(sub["game_date"]).dt.date <= update_to)]
            keep = [c for c in ("game_date","game_pk","home_win") if c in sub.columns]
            sub = sub[keep].drop_duplicates("game_pk")
            sub.to_csv(tmp_new_csv, index=False)

        new_feat = read_csv_safe(tmp_new_csv, parse_dates=[a.date_col])
        if prev_end is not None and not new_feat.empty:
            gd = pd.to_datetime(new_feat[a.date_col], errors="coerce").dt.date
            new_feat = new_feat.loc[gd > prev_end].copy()
        print(f"[new] Built {len(new_feat):,} NEW feature rows (post-filter).")

    # 4) Append to COMBINED
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
        print(f"[combine] Wrote combined → {a.features_combined}  (rows: {len(combined_new):,})")
    else:
        print("[combine] No new rows to append.")

    # 5) EDA (optional)
    if a.run_eda and os.path.exists(a.eda_script):
        print("\n== EDA ==")
        cmd = f'{a.pybin} {a.eda_script} --data "{a.features_combined}" --date-col {a.date_col} --id-col {a.id_col} --target {a.target}'
        print("[exec]", cmd)
        os.system(cmd)

    # 6) Prepare features
    print("\n== PREPARE FEATURES ==")
    cmd = f'{a.pybin} {a.prepare_script} --data "{a.features_combined}" --out "{a.prepared}" --date-col {a.date_col} --id-col {a.id_col} --target {a.target}'
    print("[exec]", cmd)
    rc = os.system(cmd)
    if rc != 0:
        print("!! prepare_features failed.")
        sys.exit(3)

    # 7) Train
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
    rc = os.system(cmd)
    if rc != 0:
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
    ap.add_argument("--features-base", default="out/mlb_features_2024.csv")
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
    ap.add_argument("--max-update-days", type=int, default=30)
    ap.add_argument("--ingest-backfill-days", type=int, default=35)
    ap.add_argument("--feature-backfill-days", type=int, default=35)
    # ---- NEW: hard cap for incremental updates ----
    ap.add_argument("--max-update-days", type=int, default=30,
                    help="Maximum span (days) to update/build per run (default: 30).")
    # Concurrency for ingest
    ap.add_argument("--max-workers", type=int, default=20)
    # Script paths/binaries
    ap.add_argument("--pybin", default="python")
    ap.add_argument("--eda-script", default="scripts/eda_scaffold.py")
    ap.add_argument("--prepare-script", default="scripts/prepare_features.py")
    ap.add_argument("--model-script", default="scripts/model_train_allinone.py")
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
    # Create out/ directory and seed missing combined file
    os.makedirs("out", exist_ok=True)
    if not os.path.exists("out/mlb_features_combined.csv"):
        print("[init] Seeding empty combined file (first CI run)")
        pd.DataFrame(columns=["game_date", "game_pk", "home_win"]).to_csv("out/mlb_features_combined.csv", index=False)

    main(args)

    #main(ap.parse_args())
