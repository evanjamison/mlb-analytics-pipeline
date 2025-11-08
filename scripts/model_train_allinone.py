#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
model_train_allinone.py — walk-forward training + leaderboard + optional betting fields

Key robustness fixes:
  1) Safe preprocessor: build ColumnTransformer ONLY with non-empty column lists.
     - If no numeric OR no categoricals, we skip that branch (no SimpleImputer on empty X).
     - If both are empty, we raise a clear error telling you which columns remained.
  2) Safe splits: walk-forward by time (monthly or ts) with clear diagnostics.
  3) Threshold strategies: f1 / youden / balanced.
  4) Optional calibration: raw / platt / isotonic.

Usage (PowerShell one-liner):
  py scripts/model_train_allinone.py ^
    --data "out\\mlb_features_prepared.csv" --date-col game_date --id-col game_pk --target home_win ^
    --outdir "model_all_ts" --topk 25 --threshold-strategy f1 --add-rolling --add-interactions ^
    --calibration isotonic --walkforward monthly --n-splits 6
"""

from __future__ import annotations

import argparse, json, os, sys, math
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, accuracy_score, log_loss, brier_score_loss, f1_score,
    precision_recall_curve, roc_curve
)

# ---------------------------
# Utilities
# ---------------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p

def _coerce_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")

def _is_numeric_dtype(dtype) -> bool:
    return np.issubdtype(dtype, np.number) or dtype == bool

def _pick_feature_cols(df: pd.DataFrame, date_col: str, id_col: str, target: str) -> List[str]:
    drop = {c for c in [date_col, id_col, target] if c in df.columns}
    # Drop any columns that are obviously non-features we introduced earlier
    extras = {'home_moneyline','away_moneyline'}
    drop |= {c for c in extras if c in df.columns}
    cols = [c for c in df.columns if c not in drop]
    return cols

def _split_types(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[List[str], List[str]]:
    num_cols, cat_cols = [], []
    for c in feature_cols:
        dt = df[c].dtype
        if _is_numeric_dtype(dt):
            num_cols.append(c)
        else:
            cat_cols.append(c)
    return num_cols, cat_cols

def make_preprocessor(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """
    Build a ColumnTransformer that is guaranteed to have no empty transformers.
    If either branch has no columns, it is omitted. If both are empty, error.
    """
    num_cols, cat_cols = _split_types(df, feature_cols)
    transformers = []
    if len(num_cols) > 0:
        num_pipe = Pipeline(
            steps=[
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler(with_mean=True, with_std=True)),
            ]
        )
        transformers.append(("num", num_pipe, num_cols))
    if len(cat_cols) > 0:
        cat_pipe = Pipeline(
            steps=[
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("cat", cat_pipe, cat_cols))

    if not transformers:
        raise ValueError(
            "No usable feature columns after filtering.\n"
            f"Candidate columns were: {feature_cols}\n"
            "Check that at least one numeric or categorical column remains."
        )

    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    return pre, num_cols, cat_cols

def _best_threshold(y_true: np.ndarray, p: np.ndarray, mode: str = "f1") -> float:
    """
    Pick a probability threshold based on strategy:
      - f1: maximize F1 on validation
      - youden: maximize (TPR - FPR) on ROC (Youden's J)
      - balanced: threshold where precision ≈ recall (closest in abs difference)
    """
    p = np.asarray(p).ravel()
    y_true = np.asarray(y_true).astype(int).ravel()

    if mode == "f1":
        pr, rc, thr = precision_recall_curve(y_true, p)
        f1s = 2 * (pr * rc) / np.clip(pr + rc, 1e-12, None)
        idx = np.nanargmax(f1s)
        # precision_recall_curve returns thresholds of length n-1
        return float(thr[max(0, idx - 1)]) if len(thr) > 0 else 0.5

    elif mode == "youden":
        fpr, tpr, thr = roc_curve(y_true, p)
        j = tpr - fpr
        idx = int(np.nanargmax(j))
        return float(thr[idx]) if idx < len(thr) else 0.5

    elif mode == "balanced":
        pr, rc, thr = precision_recall_curve(y_true, p)
        # choose threshold where |precision - recall| is minimized
        diffs = np.abs(pr[:-1] - rc[:-1])
        idx = int(np.nanargmin(diffs)) if len(diffs) else 0
        return float(thr[idx]) if len(thr) else 0.5

    return 0.5

@dataclass
class FoldResult:
    fold_idx: int
    train_rows: int
    valid_rows: int
    test_rows: int
    auc_valid: float
    auc_test: float
    f1_valid: float
    f1_test: float
    logloss_test: float
    brier_test: float
    acc_test: float
    threshold: float

# ---------------------------
# Walk-forward splits
# ---------------------------

def _time_splits_monthly(df: pd.DataFrame, date_col: str, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Split by month boundaries; last split is test. Each split advances by one month.
    Returns list of (train_idx, valid_idx, test_idx).
    """
    d = pd.to_datetime(df[date_col], errors="coerce")
    df = df.assign(_mon=d.dt.to_period("M").astype(str))
    months = sorted(df["_mon"].unique().tolist())
    if len(months) < n_splits + 2:
        n_splits = max(1, len(months) - 2)
    splits = []
    for i in range(n_splits):
        train_months = months[: i + 1]
        valid_month = months[i + 1]
        test_month = months[i + 2] if i + 2 < len(months) else months[-1]
        tr_idx = df.index[df["_mon"].isin(train_months)].to_numpy()
        va_idx = df.index[df["_mon"] == valid_month].to_numpy()
        te_idx = df.index[df["_mon"] == test_month].to_numpy()
        splits.append((tr_idx, va_idx, te_idx))
    return splits

def _time_splits_progressive(df: pd.DataFrame, date_col: str, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Progressive expanding window: split the sorted index into train/valid/test blocks.
    """
    idx = np.arange(len(df))
    blocks = np.array_split(idx, n_splits + 2)  # train blocks grow
    splits = []
    for i in range(n_splits):
        tr = np.concatenate(blocks[: i + 1])
        va = blocks[i + 1]
        te = blocks[i + 2] if i + 2 < len(blocks) else blocks[-1]
        splits.append((tr, va, te))
    return splits

# ---------------------------
# Main training
# ---------------------------

def run(args):
    ensure_dir(args.outdir)

    df = pd.read_csv(args.data)
    if args.date_col not in df.columns:
        raise ValueError(f"{args.date_col} not in data.")
    if args.target not in df.columns:
        raise ValueError(f"{args.target} not in data.")

    # Sort by date and clean target
    df[args.date_col] = _coerce_datetime(df[args.date_col])
    df = df.sort_values(args.date_col).reset_index(drop=True)
    df = df.dropna(subset=[args.target]).copy()
    y = df[args.target].astype(int).values

    # Choose feature columns and build a SAFE preprocessor
    feat_cols = _pick_feature_cols(df, args.date_col, args.id_col, args.target)
    pre, num_cols, cat_cols = make_preprocessor(df, feat_cols)

    # Define candidate models (lightweight, sklearn-only)
    models = [
        ("LogReg_l2", LogisticRegression(max_iter=1000, n_jobs=None, solver="lbfgs")),
        ("RandomForest", RandomForestClassifier(n_estimators=400, max_depth=None, n_jobs=-1, class_weight=None, random_state=42)),
        ("GradBoost", GradientBoostingClassifier(random_state=42)),
    ]

    # Calibration wrapper if requested
    def maybe_calibrate(est):
        if args.calibration == "raw":
            return est
        method = "sigmoid" if args.calibration == "platt" else "isotonic"
        # Calibrate on validation inside fold — we’ll fit per fold
        return CalibratedClassifierCV(est, method=method, cv="prefit")

    # Build walk-forward splits
    if args.walkforward == "monthly":
        splits = _time_splits_monthly(df, args.date_col, args.n_splits)
    elif args.walkforward == "ts":
        splits = _time_splits_progressive(df, args.date_col, args.n_splits)
    else:
        # Single split: 70/15/15 by order
        n = len(df)
        tr_end = int(n * 0.7)
        va_end = int(n * 0.85)
        splits = [(np.arange(0, tr_end), np.arange(tr_end, va_end), np.arange(va_end, n))]

    leaderboard: List[Dict[str, Any]] = []
    fold_summaries: List[FoldResult] = []

    # Per-model loop across splits
    for name, base_est in models:
        print(f"\n=== Training: {name} ===")
        model_metrics = []
        fold = 0

        for tr_idx, va_idx, te_idx in splits:
            fold += 1
            Xtr = df.iloc[tr_idx][feat_cols]
            ytr = y[tr_idx]
            Xva = df.iloc[va_idx][feat_cols]; yva = y[va_idx]
            Xte = df.iloc[te_idx][feat_cols]; yte = y[te_idx]

            # Fit preprocessor + base estimator
            pipe = Pipeline([("pre", pre), ("clf", base_est)])
            pipe.fit(Xtr, ytr)

            # Optional calibration using validation set
            clf = pipe
            if args.calibration != "raw":
                # Transform val data once
                Xva_tr = pipe.named_steps["pre"].transform(Xva)
                cal = maybe_calibrate(base_est)
                cal.fit(Xva_tr, yva)
                clf = Pipeline([("pre", pipe.named_steps["pre"]), ("cal", cal)])

            # Predict probabilities
            p_va = clf.predict_proba(Xva)[:, 1]
            p_te = clf.predict_proba(Xte)[:, 1]

            # Metrics
            auc_v = float(roc_auc_score(yva, p_va)) if len(np.unique(yva)) > 1 else float("nan")
            auc_t = float(roc_auc_score(yte, p_te)) if len(np.unique(yte)) > 1 else float("nan")
            thr = _best_threshold(yva, p_va, args.threshold_strategy)
            yhat_t = (p_te >= thr).astype(int)

            ll = float(log_loss(yte, np.clip(p_te, 1e-6, 1 - 1e-6)))
            br = float(brier_score_loss(yte, p_te))
            acc = float(accuracy_score(yte, yhat_t))
            f1t = float(f1_score(yte, yhat_t)) if len(np.unique(yte)) > 1 else float("nan")
            f1v = float(f1_score(yva, (p_va >= thr).astype(int))) if len(np.unique(yva)) > 1 else float("nan")

            model_metrics.append({
                "fold": fold, "model": name,
                "train_rows": int(len(tr_idx)), "valid_rows": int(len(va_idx)), "test_rows": int(len(te_idx)),
                "auc_valid": auc_v, "auc_test": auc_t,
                "f1_valid": f1v, "f1_test": f1t,
                "logloss_test": ll, "brier_test": br, "acc_test": acc, "threshold": thr
            })

            fold_summaries.append(FoldResult(
                fold_idx=fold, train_rows=len(tr_idx), valid_rows=len(va_idx), test_rows=len(te_idx),
                auc_valid=auc_v, auc_test=auc_t, f1_valid=f1v, f1_test=f1t,
                logloss_test=ll, brier_test=br, acc_test=acc, threshold=thr
            ))

        # Aggregate per-model
        dfm = pd.DataFrame(model_metrics)
        agg = dfm.mean(numeric_only=True).to_dict()
        agg.update({"model": name, "folds": len(dfm)})
        leaderboard.append(agg)

        # Save per-model fold metrics
        out_metrics_csv = os.path.join(args.outdir, f"metrics_folds_{name}.csv")
        dfm.to_csv(out_metrics_csv, index=False)

    # Leaderboard (rank by mean AUC on test, fallback to F1)
    lb = pd.DataFrame(leaderboard)
    if "auc_test" in lb.columns and lb["auc_test"].notna().any():
        lb = lb.sort_values(["auc_test","f1_test"], ascending=False)
    else:
        lb = lb.sort_values(["f1_test"], ascending=False)

    topk = min(args.topk, len(lb))
    lb_top = lb.head(topk).reset_index(drop=True)

    ensure_dir(args.outdir)
    lb_csv = os.path.join(args.outdir, "leaderboard.csv")
    lb_top.to_csv(lb_csv, index=False)

    # Save a compact JSON report
    report = {
        "data": args.data,
        "rows": int(len(df)),
        "date_range": {
            "min": str(df[args.date_col].min().date()) if len(df) else None,
            "max": str(df[args.date_col].max().date()) if len(df) else None,
        },
        "target_mean": float(np.mean(y)) if len(y) else None,
        "features_total": len(feat_cols),
        "num_features": len(num_cols),
        "cat_features": len(cat_cols),
        "walkforward": args.walkforward,
        "n_splits": args.n_splits,
        "calibration": args.calibration,
        "threshold_strategy": args.threshold_strategy,
        "leaderboard_head": lb_top.to_dict(orient="records"),
    }
    with open(os.path.join(args.outdir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== Leaderboard (top) ===")
    print(lb_top.head(min(10, len(lb_top))).to_string(index=False))
    print(f"\nSaved: {lb_csv}")
    print(f"Report: {os.path.join(args.outdir,'report.json')}")
    print("✅ Done.")

# ---------------------------
# CLI
# ---------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--date-col", default="game_date")
    ap.add_argument("--id-col", default="game_pk")
    ap.add_argument("--target", default="home_win")

    ap.add_argument("--outdir", default="model_all_ts")
    ap.add_argument("--topk", type=int, default=25)
    ap.add_argument("--threshold-strategy", choices=["f1","youden","balanced"], default="f1")

    ap.add_argument("--add-rolling", action="store_true")       # kept for CLI compatibility (no-op here)
    ap.add_argument("--add-interactions", action="store_true")  # kept for CLI compatibility (no-op here)

    ap.add_argument("--calibration", choices=["raw","platt","isotonic"], default="isotonic")
    ap.add_argument("--walkforward", choices=["monthly","ts","none"], default="monthly")
    ap.add_argument("--n-splits", type=int, default=6)

    args = ap.parse_args()
    run(args)


