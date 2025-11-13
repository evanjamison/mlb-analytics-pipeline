#!/usr/bin/env python
"""
run_pipeline.py — End-to-end MLB pipeline (incremental + 2024 base merge)

- Seeds/merges a permanent 2024 base features CSV into out/mlb_features_combined.csv
- Ingests raw (capped window), builds NEW features via scripts/data_ingest.py
- Appends new rows to combined (dedup by game_pk), runs EDA -> prepare -> train
"""
from __future__ import annotations
import argparse, os, sys
from datetime import datetime, timedelta, date
from pathlib import Path
import importlib.util
import pandas as pd

# ---------------- helpers ----------------
def ensure_parent(p: str | os.PathLike) -> str:
    p = str(p); d = os.path.dirname(p) or "."
    os.makedirs(d, exist_ok=True); return p

def read_csv_safe(path: str, parse_dates=None) -> pd.DataFrame:
    try:
        if path and os.path.exists(path):
            return pd.read_csv(path, parse_dates=parse_dates or [])
    except Exception:
        pass
    return pd.DataFrame()

def max_date(df: pd.DataFrame, col: str) -> date | None:
    if df.empty or col not in df.columns: return None
    s = pd.to_datetime(df[col], errors="coerce").dt.date
    return None if s.isna().all() else s.max()

def today_utc_date() -> date:
    return datetime.utcnow().date()

def _load_data_ingest():
    # normal import first
    try:
        import scripts.data_ingest as data_ingest  # type: ignore
        return data_ingest
    except Exception:
        pass
    # add repo root and retry
    here = Path(__file__).resolve()
    repo_root = here.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        import scripts.data_ingest as data_ingest  # type: ignore
        return data_ingest
    except Exception:
        pass
    # final fallback by file path
    di_path = repo_root / "scripts" / "data_ingest.py"
    if not di_path.exists():
        raise ModuleNotFoundError("scripts.data_ingest not importable and scripts/data_ingest.py not found.")
    spec = importlib.util.spec_from_file_location("data_ingest", str(di_path))
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod

data_ingest = _load_data_ingest()

# ---------------- base merge logic (Plan B) ----------------
def seed_or_merge_base(features_base: str, features_combined: str, date_col: str, require_base: bool) -> None:
    """
    Ensure combined contains all rows from the 2024 base CSV.
    - If combined missing/empty: write base directly (if present), else create empty skeleton.
    - If combined exists: concat(base, combined) -> dedup by game_pk (keep last) -> sort by date.
    """
    ensure_parent(features_combined)
    base_df = read_csv_safe(features_base, parse_dates=[date_col])
    comb_df = read_csv_safe(features_combined, parse_dates=[date_col])

    if base_df.empty:
        msg = f"[base] WARNING: base file not found or empty at: {features_base}"
        if require_base:
            raise FileNotFoundError(msg + "  (use --no-require-base to allow running without it)")
        print(msg)
        if comb_df.empty:
            # create a minimal header if literally nothing exists
            pd.DataFrame(columns=[date_col, "game_pk", "home_win"]).to_csv(features_combined, index=False)
        return

    # if combined empty -> write base
    if comb_df.empty:
        print(f"[base] Seeding combined with base ({len(base_df):,} rows)")
        base_df.sort_values(date_col).to_csv(features_combined, index=False)
        return

    # unify columns, concat, dedup
    print(f"[base] Merging base ({len(base_df):,}) + combined ({len(comb_df):,})")
    all_cols = sorted(set(base_df.columns).union(set(comb_df.columns)))
    base_df = base_df.reindex(columns=all_cols)
    comb_df = comb_df.reindex(columns=all_cols)
    merged = pd.concat([base_df, comb_df], ignore_index=True)
    if "game_pk" in merged.columns:
        merged = merged.sort_values([date_col, "game_pk"]).drop_duplicates("game_pk", keep="last")
    else:
        merged = merged.drop_duplicates()
    merged = merged.sort_values(date_col).reset_index(drop=True)
    merged.to_csv(features_combined, index=False)
    print(f"[base] Combined now has {len(merged):,} rows (after base merge)")

# ---------------- orchestrator ----------------
def main(a):
    # Make sure key paths exist
    for p in (a.raw, a.features_combined, a.prepared):
        ensure_parent(p)
    os.makedirs(a.model_out, exist_ok=True)
    print(f"[diag] data_ingest from: {getattr(data_ingest, '__file__', 'unknown')}")

    # 0) Seed/merge the 2024 base into combined (Plan B)
    seed_or_merge_base(a.features_base, a.features_combined, a.date_col, a.require_base)

    # 1) Load combined and determine update window
    combined_df = read_csv_safe(a.features_combined, parse_dates=[a.date_col])
    prev_end = max_date(combined_df, a.date_col)
    update_to = datetime.strptime(a.update_to, "%Y-%m-%d").date() if a.update_to else today_utc_date()

    if prev_end is None:
        start_build = date(a.base_year, 1, 1) - timedelta(days=a.feature_backfill_days)
        print(f"[span] No combined rows yet; build from {start_build} → {update_to}")
    else:
        start_build = max(
            prev_end + timedelta(days=1) - timedelta(days=a.feature_backfill_days),
            date(a.base_year, 1, 1) - timedelta(days=a.feature_backfill_days),
        )
        print(f"[span] Combined through {prev_end}. New features from {start_build} → {update_to} (warmup {a.feature_backfill_days}d)")

    # hard cap on incremental build
    cap_start = update_to - timedelta(days=a.max_update_days)
    if start_build < cap_start:
        print(f"[cap] Limiting build to last {a.max_update_days} days: {cap_start} → {update_to}")
        start_build = cap_start

    # 2) RAW ingest for years that overlap the (capped) window (+ingest warmup)
    ingest_start = max(date(a.base_year, 1, 1), start_build - timedelta(days=a.ingest_backfill_days))
    y0, y1 = ingest_start.year, update_to.year
    print(f"\n== RAW INGEST == covering {ingest_start} → {update_to} (years {y0}..{y1}, backfill {a.ingest_backfill_days}d)")
    for yr in range(y0, y1 + 1):
        data_ingest._ingest_year_to_raw(
            year=yr,
            backfill_days=a.ingest_backfill_days,
            out_path=a.raw,
            include_finished=True,
            fetch_handedness=True,
            max_workers=a.max_workers,
        )

    raw_df = read_csv_safe(a.raw, parse_dates=["game_date", "game_datetime"])
    if raw_df.empty:
        print("!! No raw data after ingest; abort.")
        sys.exit(2)

    # 3) Build features for the (capped) span
    if start_build > update_to:
        new_feat = pd.DataFrame()
        print("[build] Nothing to build.")
    else:
        tmp_new_csv = os.path.join("out", f"_tmp_features_{start_build}_to_{update_to}.csv").replace(":", "-")
        os.makedirs("out", exist_ok=True)

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
            print("[warn] No build_feature_table in data_ingest; writing minimal leak-safe features.")
            sub = raw_df.copy()
            gd = pd.to_datetime(sub["game_date"]).dt.date
            sub = sub[(gd >= start_build) & (gd <= update_to)]
            keep = [c for c in ("game_date","game_pk","home_win") if c in sub.columns]
            sub = sub[keep].drop_duplicates("game_pk")
            sub.to_csv(tmp_new_csv, index=False)

        new_feat = read_csv_safe(tmp_new_csv, parse_dates=[a.date_col])
        if prev_end is not None and not new_feat.empty:
            gd = pd.to_datetime(new_feat[a.date_col], errors="coerce").dt.date
            new_feat = new_feat.loc[gd > prev_end].copy()
        print(f"[build] Built {len(new_feat):,} NEW feature rows after post-filter.")

    # 4) Append to COMBINED (dedup)
    if not new_feat.empty:
        combined_old = read_csv_safe(a.features_combined, parse_dates=[a.date_col])
        all_cols = sorted(set(combined_old.columns).union(set(new_feat.columns)))
        combined_old = combined_old.reindex(columns=all_cols)
        new_feat = new_feat.reindex(columns=all_cols)
        merged = pd.concat([combined_old, new_feat], ignore_index=True)
        if "game_pk" in merged.columns:
            merged = merged.sort_values([a.date_col, "game_pk"]).drop_duplicates("game_pk", keep="last")
        else:
            merged = merged.drop_duplicates()
        merged = merged.sort_values(a.date_col).reset_index(drop=True)
        merged.to_csv(a.features_combined, index=False)
        print(f"[combine] Updated combined → {a.features_combined}  (rows: {len(merged):,})")
    else:
        print("[combine] No new rows to append.")

    # 5) EDA (optional)
    if a.run_eda and os.path.exists(a.eda_script):
        print("\n== EDA ==")
        cmd = f'{a.pybin} {a.eda_script} --data "{a.features_combined}" --date-col {a.date_col} --id-col {a.id_col} --target {a.target}'
        print("[exec]", cmd); os.system(cmd)

    # 6) Prepare features
    print("\n== PREPARE FEATURES ==")
    cmd = f'{a.pybin} {a.prepare_script} --data "{a.features_combined}" --out "{a.prepared}" --date-col {a.date_col} --id-col {a.id_col} --target {a.target}'
    print("[exec]", cmd)
    if os.system(cmd) != 0:
        print("!! prepare_features failed."); sys.exit(3)

    # 7) Train
    print("\n== MODEL TRAINING ==")
    args = [
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
        f'--min-edge {a.min_edge:.3f}',
        f'--kelly-cap {a.kelly_cap:.3f}',
    ]
    cmd = f'{a.pybin} {a.model_script} ' + " ".join([t for t in args if t])
    print("[exec]", cmd)
    if os.system(cmd) != 0:
        print("!! model_train_allinone failed."); sys.exit(4)

    print("\n✅ Pipeline complete.")
    print("Artifacts:")
    print("  Base features     :", a.features_base)
    print("  Combined features :", a.features_combined)
    print("  Prepared dataset  :", a.prepared)
    print("  Model out dir     :", a.model_out)

# ---------------- CLI ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # IO
    ap.add_argument("--raw", default="out/raw_games.csv")
    ap.add_argument("--features-base", default="out/mlb_features_full_20240328_to_20240929.csv",
                    help="Your 2024 season base features CSV")
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
    # Backfills + cap
    ap.add_argument("--ingest-backfill-days", type=int, default=7)
    ap.add_argument("--feature-backfill-days", type=int, default=30)
    ap.add_argument("--max-update-days", type=int, default=40)
    # Concurrency
    ap.add_argument("--max-workers", type=int, default=20)
    # Scripts / runtime
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
    ap.add_argument("--min-edge", type=float, default=0.02)
    ap.add_argument("--kelly-cap", type=float, default=0.05)
    # Base file strictness
    ap.add_argument("--require-base", dest="require_base", action="store_true")
    ap.add_argument("--no-require-base", dest="require_base", action="store_false")
    ap.set_defaults(require_base=False)

    args = ap.parse_args()

    # ensure out/ exists and combined file is present (empty skeleton OK)
    os.makedirs("out", exist_ok=True)
    if not os.path.exists(args.features_combined):
        pd.DataFrame(columns=[args.date_col, "game_pk", args.target]).to_csv(args.features_combined, index=False)

    main(args)
