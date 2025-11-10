#!/usr/bin/env python
# scripts/eda_scaffold.py
import argparse, os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def _safe_read(path, parse_dates=None):
    return pd.read_csv(path, parse_dates=parse_dates or [])

def _pick_feature_cols(df, date_col, id_col, target):
    drop = {c for c in [date_col, id_col, target] if c in df.columns}
    extras = {'home_moneyline','away_moneyline'}
    drop |= {c for c in extras if c in df.columns}
    return [c for c in df.columns if c not in drop]

def _pointbiserial(df, target, features):
    # numeric-only correlation against binary target (0/1)
    y = df[target].astype(float).values
    out = []
    for c in features:
        s = df[c]
        if np.issubdtype(s.dtype, np.number):
            x = pd.to_numeric(s, errors='coerce')
            if x.notna().sum() > 5 and np.nanstd(x) > 0:
                r = np.corrcoef(x.fillna(x.mean()), y)[0,1]
                out.append((c, float(r)))
    return pd.DataFrame(out, columns=["feature","corr_to_target"]).sort_values("corr_to_target", key=np.abs, ascending=False)

def _mutual_info(df, target, features):
    X = df[features].copy()
    # basic numeric cast; drop purely non-numeric for MI here (keeps script snappy)
    for c in X.columns:
        if not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.factorize(X[c].astype(str), sort=True)[0]
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = df[target].astype(int).values
    mi = mutual_info_classif(X.values, y, random_state=42)
    return pd.DataFrame({"feature": X.columns, "mutual_info": mi}).sort_values("mutual_info", ascending=False)

def _perm_importance(df, target, features, n_repeats=10):
    # tiny baseline: standardized logistic
    X = df[features].copy()
    for c in X.columns:
        if not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.factorize(X[c].astype(str), sort=True)[0]
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
    y = df[target].astype(int).values

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    imps = np.zeros(X.shape[1])
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X[tr], y[tr])
        r = permutation_importance(clf, X[te], y[te], n_repeats=n_repeats, random_state=42, scoring="roc_auc")
        imps += r.importances_mean
    imps /= 3.0
    return pd.DataFrame({"feature": features, "perm_auc_drop": imps}).sort_values("perm_auc_drop", ascending=False)

def _barh(df, col, title, outpng, top=15, figsize=(8,6)):
    topdf = df.head(top).iloc[::-1]
    plt.figure(figsize=figsize)
    plt.barh(topdf["feature"], topdf[col])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpng, dpi=170)
    plt.close()

def main(a):
    outdir = ensure_dir(a.outdir)
    df = _safe_read(a.data, parse_dates=[a.date_col])
    feats = _pick_feature_cols(df, a.date_col, a.id_col, a.target)

    # views
    corr = _pointbiserial(df, a.target, feats)
    mi   = _mutual_info(df, a.target, feats)
    pim  = _perm_importance(df, a.target, feats, n_repeats=8)

    # join & save
    merged = corr.merge(mi, on="feature", how="outer").merge(pim, on="feature", how="outer")
    merged.to_csv(os.path.join(outdir, "feature_ranking.csv"), index=False)

    # plots
    _barh(corr, "corr_to_target", "Top correlated features", os.path.join(outdir, "top_corr.png"))
    _barh(mi, "mutual_info", "Top mutual information features", os.path.join(outdir, "top_mi.png"))
    _barh(pim, "perm_auc_drop", "Top permutation-importance (AUC drop)", os.path.join(outdir, "top_perm.png"))

    # tiny HTML
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>EDA — Feature Importance</title></head>
<body>
<h1>EDA — Feature Importance</h1>
<p><b>Data:</b> {a.data}</p>
<ul>
  <li><a href="feature_ranking.csv">feature_ranking.csv</a></li>
</ul>
<h2>Top correlated features</h2><img src="top_corr.png" style="max-width:900px;">
<h2>Top mutual info features</h2><img src="top_mi.png" style="max-width:900px;">
<h2>Top permutation-importance (AUC drop)</h2><img src="top_perm.png" style="max-width:900px;">
</body></html>"""
    with open(os.path.join(outdir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--date-col", default="game_date")
    ap.add_argument("--id-col", default="game_pk")
    ap.add_argument("--target", default="home_win")
    ap.add_argument("--outdir", default="eda_out")
    args = ap.parse_args()
    main(args)
