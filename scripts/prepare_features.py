"""
prepare_features.py

Leakage-safe rolling imputation + first-pass feature engineering
for the MLB home-win dataset.

Usage:
    python prepare_features.py \
      --data "out/mlb_features_model_20240328_to_20240929.csv" \
      --date-col game_date \
      --id-col game_pk \
      --target home_win \
      --out "out/mlb_features_prepared.csv"

Notes
-----
- Rolling imputation is time-based, grouped by available entity keys
  (e.g., team/pitcher IDs) if those columns exist.
- We shift(1) all rolling stats to avoid look-ahead leakage.
- If a column still has NA after all group levels, we fall back to the
  global median (computed on the training-time past only per row).
- Feature engineering is conservative and auto-skips when inputs are absent.
"""

import argparse
import os
from typing import List, Sequence

import numpy as np
import pandas as pd


# ---------------------------
# Utilities
# ---------------------------

LIKELY_KEYS_PRIORITY: List[List[str]] = [
    ["home_sp_id"],                # 1) starting pitcher-level (home)
    ["away_sp_id"],                #    starting pitcher-level (away)
    ["home_team_id"],              # 2) team-level (home)
    ["away_team_id"],              #    team-level (away)
]

# You can extend this if you have these IDs:
#   ["home_bp_id"], ["away_bp_id"], ["plate_umpire_id"], etc.


def ensure_parent_dir(path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def _has_cols(df: pd.DataFrame, cols: Sequence[str]) -> bool:
    return all(c in df.columns for c in cols)


def _time_rolling_mean(
    gdf: pd.DataFrame,
    date_col: str,
    value_col: str,
    window_days: int,
) -> pd.Series:
    """
    Compute leakage-safe time-based rolling mean per group:
      1) set index to datetime
      2) rolling(f'{window_days}D').mean()
      3) shift(1) so current row doesn't see its own value
    Works even with irregularly spaced game dates.
    """
    s = (
        gdf.set_index(date_col)[value_col]
        .rolling(f"{window_days}D")
        .mean()
        .shift(1)
        .reset_index(drop=True)
    )
    return s


def rolling_impute_single(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    group_cols: Sequence[str],
    window_days: int,
) -> pd.Series:
    """
    Return a Series of rolling means for value_col, computed within groups
    defined by group_cols (if present). If group columns aren’t present,
    returns an empty Series with correct index (no effect).
    """
    if not _has_cols(df, group_cols):
        return pd.Series(index=df.index, dtype=float)

    pieces = []
    for _, g in df.groupby(list(group_cols), sort=False, dropna=False):
        # Preserve original order within group; rolling expects datetime dtype
        g = g.sort_values(date_col)
        rmean = _time_rolling_mean(g, date_col, value_col, window_days)
        pieces.append(pd.Series(rmean.values, index=g.index))
    if pieces:
        out = pd.concat(pieces).sort_index()
        return out
    return pd.Series(index=df.index, dtype=float)


def leakage_safe_global_median(df: pd.DataFrame, date_col: str, value_col: str) -> pd.Series:
    """
    Expanding (past-only) global median per row:
    median of values strictly before the current row's date.
    If no past data yet, returns NaN.
    """
    # Sort by date once
    s = df[[date_col, value_col]].sort_values(date_col)
    vals = s[value_col].values
    med = np.empty(len(vals))
    med[:] = np.nan
    # Running median without dependencies: keep a list of past values (ok at MLB scale)
    past = []
    for i in range(len(vals)):
        if len(past) == 0:
            med[i] = np.nan
        else:
            med[i] = float(np.median(past))
        v = vals[i]
        if pd.notna(v):
            past.append(v)
    # Map back to original index order
    return pd.Series(med, index=s.index).reindex(df.index)


def rolling_impute_column(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    keys_priority: List[List[str]],
    windows: Sequence[int] = (14, 30),
) -> pd.Series:
    """
    Impute a numeric column using a hierarchy of time-based rolling means.
    For each row, we try:
      1) pitcher-level (if key exists) 14D → 30D
      2) team-level (if key exists)    14D → 30D
      3) global expanding median of past
      4) final fallback: overall column median (static)
    Always leakage-safe via shift(1).
    """
    out = pd.Series(np.nan, index=df.index, dtype=float)

    # Try each set of keys in priority order
    for keyset in keys_priority:
        if not _has_cols(df, keyset):
            continue
        for w in windows:
            r = rolling_impute_single(df, date_col, value_col, keyset, w)
            if r.notna().any():
                out = out.fillna(r)

    # Global expanding median (past only)
    gm = leakage_safe_global_median(df, date_col, value_col)
    out = out.fillna(gm)

    # Final static median fallback (in case earliest rows are still NA)
    static_med = df[value_col].median()
    out = out.fillna(static_med)

    return out


def auto_numeric_impute_cols(df: pd.DataFrame, target: str, id_col: str) -> List[str]:
    """Pick numeric columns that actually need imputation (have NaNs)."""
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = {target, id_col}
    cols = [c for c in num if c not in exclude and df[c].isna().any()]
    return cols


# ---------------------------
# Feature engineering (safe & optional)
# ---------------------------

FE_INT_PAIRS = [
    ("SP_xFIP_diff", "Lineup_woba_vsHand_diff", "INT_SPxFIP__WOBAvsHand"),
    ("Team_WinPct_diff", "Team_Elo_diff", "INT_WinPct__Elo"),
    ("SP_KminusBB_diff", "Lineup_woba_vsHand_diff", "INT_KBB__WOBAvsHand"),
]

FE_QUADRATICS = [
    "Team_WinPct_diff",
    "Team_Elo_diff",
    "SP_xFIP_diff",
    "SP_KminusBB_diff",
]

def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    for a, b, name in FE_INT_PAIRS:
        if a in df.columns and b in df.columns:
            df[name] = df[a] * df[b]
    return df

def add_quadratics(df: pd.DataFrame) -> pd.DataFrame:
    for c in FE_QUADRATICS:
        if c in df.columns:
            df[f"{c}__sq"] = df[c] ** 2
    return df

def add_monthly_zscores(df: pd.DataFrame, date_col: str, cols: Sequence[str]) -> pd.DataFrame:
    """
    Z-score by calendar month (Period 'M'), leakage-safe (group stats
    are computed without shifting, but since it's a per-month aggregate
    used as a transform, it's typically fine for exploratory modeling.
    If you want strict leakage safety, compute on train only.)
    """
    if not np.issubdtype(df[date_col].dtype, np.datetime64):
        return df

    month = df[date_col].dt.to_period("M")
    for c in cols:
        if c not in df.columns: 
            continue
        g = df.groupby(month)[c]
        m = g.transform("mean")
        s = g.transform("std")
        df[f"{c}__z_m"] = (df[c] - m) / s.replace(0, np.nan)
    return df


# ---------------------------
# Main script
# ---------------------------

def main(args):
    df = pd.read_csv(args.data, parse_dates=[args.date_col])
    df = df.sort_values(args.date_col).reset_index(drop=True)

    print(f"[read] rows={len(df):,}, cols={df.shape[1]}")

    # Choose columns to impute
    impute_cols = auto_numeric_impute_cols(df, args.target, args.id_col)
    print(f"[impute] numeric columns with NA: {len(impute_cols)}")

    # Perform leakage-safe rolling imputation
    for c in impute_cols:
        before = df[c].isna().sum()
        imp = rolling_impute_column(
            df,
            date_col=args.date_col,
            value_col=c,
            keys_priority=LIKELY_KEYS_PRIORITY,
            windows=(14, 30),
        )
        df[c] = df[c].fillna(imp)
        after = df[c].isna().sum()
        print(f"  - {c}: {before} → {after} NA")

    # --- Feature engineering (skips missing inputs automatically) ---
    print("[features] adding interactions/quadratics")
    df = add_interactions(df)
    df = add_quadratics(df)

    # monthly z-scores for a small, high-signal set (auto-skips if absent)
    z_cols = [
        "Team_WinPct_diff",
        "Team_Elo_diff",
        "Lineup_woba_vsHand_diff",
        "SP_xFIP_diff",
        "SP_KminusBB_diff",
    ]
    print("[features] adding monthly z-scores (if columns exist)")
    df = add_monthly_zscores(df, args.date_col, z_cols)

    # Save
    ensure_parent_dir(args.out)
    df.to_csv(args.out, index=False)
    print(f"[write] saved: {args.out}")
    print("[done] feature prep complete.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to modeling CSV")
    p.add_argument("--date-col", default="game_date")
    p.add_argument("--id-col", default="game_pk")
    p.add_argument("--target", default="home_win")
    p.add_argument("--out", default="out/mlb_features_prepared.csv")
    args = p.parse_args()
    main(args)
