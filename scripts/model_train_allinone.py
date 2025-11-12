#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
model_train_allinone.py — walk-forward training + leaderboard + optional betting fields

- Safe preprocessor (skips empty branches; errors if no features remain)
- Safe monthly/TS splits with diagnostics and clamping
- Threshold strategies: f1 / youden / balanced
- Calibration: raw / platt / isotonic
- Figures: ROC/PR (all models), reliability + confusion (best model), walk-forward AUC trend
- HTML report: index.html inside --outdir
"""

from __future__ import annotations

import argparse, json, os, sys
from dataclasses import dataclass
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
    precision_recall_curve, roc_curve, confusion_matrix
)
import matplotlib.pyplot as plt


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
    extras = {'home_moneyline','away_moneyline'}  # non-features if present
    drop |= {c for c in extras if c in df.columns}
    return [c for c in df.columns if c not in drop]

def _split_types(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[List[str], List[str]]:
    num_cols, cat_cols = [], []
    for c in feature_cols:
        dt = df[c].dtype
        (num_cols if _is_numeric_dtype(dt) else cat_cols).append(c)
    return num_cols, cat_cols

def make_preprocessor(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[ColumnTransformer, List[str], List[str]]:
    num_cols, cat_cols = _split_types(df, feature_cols)
    transformers = []
    if num_cols:
        num_pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler(with_mean=True, with_std=True)),
        ])
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        cat_pipe = Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", cat_pipe, cat_cols))
    if not transformers:
        raise ValueError(
            "No usable feature columns after filtering.\n"
            f"Candidate columns were: {feature_cols}\n"
            "Check that at least one numeric or categorical column remains."
        )
    return ColumnTransformer(transformers=transformers, remainder="drop"), num_cols, cat_cols

def _best_threshold(y_true: np.ndarray, p: np.ndarray, mode: str = "f1") -> float:
    p = np.asarray(p).ravel()
    y_true = np.asarray(y_true).astype(int).ravel()

    if mode == "f1":
        pr, rc, thr = precision_recall_curve(y_true, p)
        f1s = 2 * (pr * rc) / np.clip(pr + rc, 1e-12, None)
        idx = int(np.nanargmax(f1s)) if len(f1s) else 0
        return float(thr[max(0, idx - 1)]) if len(thr) else 0.5

    if mode == "youden":
        fpr, tpr, thr = roc_curve(y_true, p)
        j = tpr - fpr
        idx = int(np.nanargmax(j)) if len(j) else 0
        return float(thr[idx]) if len(thr) else 0.5

    if mode == "balanced":
        pr, rc, thr = precision_recall_curve(y_true, p)
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

def _time_splits_monthly(df: pd.DataFrame, date_col: str, n_splits: int):
    """
    Split by month boundaries safely — automatically clamps n_splits if not enough months.
    Returns list of (train_idx, valid_idx, test_idx).
    """
    d = pd.to_datetime(df[date_col], errors="coerce")
    df = df.assign(_mon=d.dt.to_period("M").astype(str))
    months = sorted(df["_mon"].dropna().unique().tolist())

    if len(months) < 3:
        print(f"[warn] Only {len(months)} distinct months in data — using single 70/15/15 split.")
        idx = np.arange(len(df))
        n = len(df)
        tr_end, va_end = int(n * 0.7), int(n * 0.85)
        return [(idx[:tr_end], idx[tr_end:va_end], idx[va_end:])]

    n_splits = min(n_splits, len(months) - 2)
    splits = []
    for i in range(n_splits):
        train_months = months[: i + 1]
        valid_month = months[i + 1]
        test_month  = months[i + 2]
        tr_idx = df.index[df["_mon"].isin(train_months)].to_numpy()
        va_idx = df.index[df["_mon"] == valid_month].to_numpy()
        te_idx = df.index[df["_mon"] == test_month].to_numpy()
        splits.append((tr_idx, va_idx, te_idx))

    print(f"[diag] Generated {len(splits)} monthly splits from {len(months)} months.")
    return splits


def _time_splits_progressive(df: pd.DataFrame, date_col: str, n_splits: int):
    """
    Progressive expanding window: split the sorted index into train/valid/test blocks.
    """
    idx = np.arange(len(df))
    blocks = np.array_split(idx, n_splits + 2)
    splits = []
    for i in range(n_splits):
        tr = np.concatenate(blocks[: i + 1]) if i > 0 else blocks[0]
        va = blocks[i + 1]
        te = blocks[i + 2] if i + 2 < len(blocks) else blocks[-1]
        splits.append((tr, va, te))
    return splits


# ---- plotting helpers ----

def _plot_roc_all(ax, curves, title="ROC — test_all"):
    for name, (fpr, tpr) in curves.items():
        ax.plot(fpr, tpr, label=name)
    ax.plot([0,1],[0,1],"--",alpha=0.4)
    ax.set_title(title); ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend()

def _plot_pr_all(ax, curves, title="PR — test_all"):
    for name, (prec, rec) in curves.items():
        ax.plot(rec, prec, label=name)
    ax.set_title(title); ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.legend()

def _plot_reliability(ax, probs, y_true, bins=10, title="Reliability — best"):
    p = np.clip(probs, 1e-6, 1-1e-6)
    qs = np.linspace(0,1,bins+1)
    idx = np.digitize(p, qs)-1
    obs, pred = [], []
    for b in range(bins):
        m = idx==b
        if m.sum()>0:
            obs.append(np.mean(y_true[m]))
            pred.append(np.mean(p[m]))
    ax.plot([0,1],[0,1],":",color="tab:orange")
    ax.plot(pred, obs, marker="o")
    ax.set_title(title); ax.set_xlabel("pred prob"); ax.set_ylabel("observed")

def _plot_conf(ax, y_true, y_hat, title="Confusion"):
    cm = confusion_matrix(y_true, y_hat)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title); ax.set_xlabel("Pred"); ax.set_ylabel("True")
    for (i,j),v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha='center', va='center')


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

    # Features + preprocessor
    feat_cols = _pick_feature_cols(df, args.date_col, args.id_col, args.target)
    pre, num_cols, cat_cols = make_preprocessor(df, feat_cols)

    # Candidate models
    models = [
        ("LogReg_l2",  LogisticRegression(max_iter=1000, solver="lbfgs")),
        ("RandomForest", RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=42)),
        ("GradBoost",  GradientBoostingClassifier(random_state=42)),
    ]

    def maybe_calibrate(est):
        if args.calibration == "raw":
            return est
        method = "sigmoid" if args.calibration == "platt" else "isotonic"
        return CalibratedClassifierCV(est, method=method, cv="prefit")

    # Splits
    if args.walkforward == "monthly":
        splits = _time_splits_monthly(df, args.date_col, args.n_splits)
    elif args.walkforward == "ts":
        splits = _time_splits_progressive(df, args.date_col, args.n_splits)
    else:
        n = len(df); tr_end, va_end = int(n*0.7), int(n*0.85)
        splits = [(np.arange(0,tr_end), np.arange(tr_end,va_end), np.arange(va_end,n))]

    leaderboard: List[Dict[str, Any]] = []
    fold_summaries: List[FoldResult] = []

    # Train across splits
    for name, base_est in models:
        print(f"\n=== Training: {name} ===")
        model_metrics = []
        fold = 0

        for tr_idx, va_idx, te_idx in splits:
            fold += 1
            Xtr = df.iloc[tr_idx][feat_cols]; ytr = y[tr_idx]
            Xva = df.iloc[va_idx][feat_cols]; yva = y[va_idx]
            Xte = df.iloc[te_idx][feat_cols]; yte = y[te_idx]

            pipe = Pipeline([("pre", pre), ("clf", base_est)])
            pipe.fit(Xtr, ytr)

            clf = pipe
            if args.calibration != "raw":
                Xva_tr = pipe.named_steps["pre"].transform(Xva)
                cal = maybe_calibrate(base_est)
                cal.fit(Xva_tr, yva)
                clf = Pipeline([("pre", pipe.named_steps["pre"]), ("cal", cal)])

            p_va = clf.predict_proba(Xva)[:, 1]
            p_te = clf.predict_proba(Xte)[:, 1]

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

        dfm = pd.DataFrame(model_metrics)
        agg = dfm.mean(numeric_only=True).to_dict()
        agg.update({"model": name, "folds": len(dfm)})
        leaderboard.append(agg)
        dfm.to_csv(os.path.join(args.outdir, f"metrics_folds_{name}.csv"), index=False)

    # Leaderboard
    lb = pd.DataFrame(leaderboard)
    lb = (lb.sort_values(["auc_test","f1_test"], ascending=False)
          if "auc_test" in lb and lb["auc_test"].notna().any()
          else lb.sort_values(["f1_test"], ascending=False))
    lb_top = lb.head(min(args.topk, len(lb))).reset_index(drop=True)

    ensure_dir(args.outdir)
    lb_csv = os.path.join(args.outdir, "leaderboard.csv")
    lb_top.to_csv(lb_csv, index=False)

    # Report JSON
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
        "min_edge": args.min_edge,
        "kelly_cap": args.kelly_cap,
        "leaderboard_head": lb_top.to_dict(orient="records"),
    }
    with open(os.path.join(args.outdir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # ---------- Figures ----------
    figs_dir = os.path.join(args.outdir, "figs"); ensure_dir(figs_dir)

    # Re-fit once on last split for plotting
    if len(splits):
        tr_idx, va_idx, te_idx = splits[-1]
        Xtr = df.iloc[tr_idx][feat_cols]; ytr = y[tr_idx]
        Xva = df.iloc[va_idx][feat_cols]; yva = y[va_idx]
        Xte = df.iloc[te_idx][feat_cols]; yte = y[te_idx]

        roc_curves, pr_curves = {}, {}
        best_name, best_auc = None, -1.0
        best_probs, best_thr, best_pred = None, 0.5, None

        for name, base_est in models:
            pipe = Pipeline([("pre", pre), ("clf", base_est)])
            pipe.fit(Xtr, ytr)
            clf = pipe
            if args.calibration != "raw":
                Xva_tr = pipe.named_steps["pre"].transform(Xva)
                method = "sigmoid" if args.calibration == "platt" else "isotonic"
                cal = CalibratedClassifierCV(base_est, method=method, cv="prefit")
                cal.fit(Xva_tr, yva)
                clf = Pipeline([("pre", pipe.named_steps["pre"]), ("cal", cal)])

            p_te = clf.predict_proba(Xte)[:,1]
            fpr, tpr, _ = roc_curve(yte, p_te)
            prec, rec, _ = precision_recall_curve(yte, p_te)
            roc_curves[name] = (fpr, tpr)
            pr_curves[name]  = (prec, rec)

            auc_t = float(roc_auc_score(yte, p_te)) if len(np.unique(yte))>1 else float("nan")
            if not np.isnan(auc_t) and auc_t > best_auc:
                best_auc = auc_t
                best_name = name
                best_probs = p_te
                best_thr   = _best_threshold(yva, clf.predict_proba(Xva)[:,1], args.threshold_strategy)
                best_pred  = (p_te >= best_thr).astype(int)

        # ROC all
        fig, ax = plt.subplots(figsize=(6,5))
        _plot_roc_all(ax, roc_curves, title="ROC — test_all")
        fig.savefig(os.path.join(figs_dir,"roc_test_all.png"), dpi=170); plt.close(fig)

        # PR all
        fig, ax = plt.subplots(figsize=(6,5))
        _plot_pr_all(ax, pr_curves, title="PR — test_all")
        fig.savefig(os.path.join(figs_dir,"pr_test_all.png"), dpi=170); plt.close(fig)

        # Reliability + Confusion for best model
        if best_probs is not None:
            fig, ax = plt.subplots(figsize=(6,5))
            _plot_reliability(ax, best_probs, yte, title=f"Reliability — best={best_name}")
            fig.savefig(os.path.join(figs_dir,"reliability_best.png"), dpi=170); plt.close(fig)

            fig, ax = plt.subplots(figsize=(4,4))
            _plot_conf(ax, yte, best_pred, title=f"Confusion — best={best_name} (thr={best_thr:.2f})")
            fig.savefig(os.path.join(figs_dir,"confusion_best.png"), dpi=170); plt.close(fig)

        # Walk-forward trend (if multiple splits)
        if len(splits) > 1 and best_name is not None:
            aucs = []
            for (tr, va, te) in splits:
                Xtr = df.iloc[tr][feat_cols]; ytr = y[tr]
                Xva = df.iloc[va][feat_cols]; yva = y[va]
                Xte = df.iloc[te][feat_cols]; yte = y[te]
                est = [m for m in models if m[0]==best_name][0][1]
                pipe = Pipeline([("pre", pre), ("clf", est)])
                pipe.fit(Xtr, ytr)
                clf = pipe
                if args.calibration != "raw":
                    Xva_tr = pipe.named_steps["pre"].transform(Xva)
                    method = "sigmoid" if args.calibration == "platt" else "isotonic"
                    cal = CalibratedClassifierCV(est, method=method, cv="prefit")
                    cal.fit(Xva_tr, yva)
                    clf = Pipeline([("pre", pipe.named_steps["pre"]), ("cal", cal)])
                p_te = clf.predict_proba(Xte)[:,1]
                aucs.append(float(roc_auc_score(yte, p_te)) if len(np.unique(yte))>1 else np.nan)
            fig, ax = plt.subplots(figsize=(6,3))
            ax.plot(np.arange(1,len(aucs)+1), aucs, marker="o")
            ax.set_xlabel("split"); ax.set_ylabel("AUC"); ax.set_title(f"Walk-forward AUC — {best_name}")
            fig.savefig(os.path.join(figs_dir,"walkforward_auc.png"), dpi=170); plt.close(fig)

    # HTML index (build without f-strings to avoid backslash-in-expression issues)
    walk_tag = '<img src="figs/walkforward_auc.png" style="max-width:700px;">' if len(splits) > 1 else ''

    html = (
        "<!doctype html>\n"
        "<html><head><meta charset='utf-8'><title>Model report</title></head>\n"
        "<body>\n"
        "<h1>Model report</h1>\n"
        "<ul>\n"
        '  <li><a href="leaderboard.csv">leaderboard.csv</a></li>\n'
        '  <li><a href="report.json">report.json</a></li>\n'
        "</ul>\n"
        "<h2>Figures</h2>\n"
        '<img src="figs/roc_test_all.png" style="max-width:900px;"><br>\n'
        '<img src="figs/pr_test_all.png" style="max-width:900px;"><br>\n'
        '<img src="figs/reliability_best.png" style="max-width:650px;"><br>\n'
        '<img src="figs/confusion_best.png" style="max-width:450px;"><br>\n'
        + walk_tag + "\n"
        "</body></html>"
    )
    with open(os.path.join(args.outdir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)


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

    # kept for CLI compatibility (feature building is upstream)
    ap.add_argument("--add-rolling", action="store_true")
    ap.add_argument("--add-interactions", action="store_true")

    ap.add_argument("--calibration", choices=["raw","platt","isotonic"], default="isotonic")
    ap.add_argument("--walkforward", choices=["monthly","ts","none"], default="monthly")
    ap.add_argument("--n-splits", type=int, default=6)

    # pass-through betting params (not used here, but accepted)
    ap.add_argument("--min-edge", type=float, default=0.02)
    ap.add_argument("--kelly-cap", type=float, default=0.05)

    args = ap.parse_args()
    run(args)

