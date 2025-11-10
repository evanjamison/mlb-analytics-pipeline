
"""
Enhanced EDA Scaffold for MLB Modeling (Drop-in Fix)
----------------------------------------------------

Usage:
    python eda_scaffold.py --data "out/mlb_features_model_20240328_to_20240929.csv" \
        --date-col game_date --id-col game_pk --target home_win

Outputs:
    - eda_out_plus/  (plots, tables, summaries)
    - eda_out_plus/report.html  (rich HTML summary)
"""

import argparse, os, warnings, math
from datetime import datetime
import numpy as np, pandas as pd, matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)
plt.rcParams.update({"figure.dpi": 120})

# --------------------------
# Utilities
# --------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def fig_path(out_dir, name):
    return os.path.join(out_dir, f"{name}.png")

def save_csv(df, out_dir, name):
    path = os.path.join(out_dir, f"{name}.csv")
    df.to_csv(path, index=False)
    return path

def _coerce_datetime(df, date_col):
    if date_col in df.columns:
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        except Exception:
            pass
    return df

def _replace_nonfinite(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number]).columns
    if len(num):
        df[num] = df[num].replace([np.inf, -np.inf], np.nan)
    return df

def _trim_for_plot(series: pd.Series, q=0.995):
    """Winsorize for plotting only (returns numpy array); raw data is not modified."""
    s = series.dropna().values
    if s.size == 0:
        return s
    lo, hi = np.nanpercentile(s, [(1-q)*100, q*100])
    return np.clip(s, lo, hi)

# --------------------------
# Core Helpers
# --------------------------
def summary_stats(df, target=None):
    try:
        desc = df.describe(include="all", datetime_is_numeric=True).T
    except TypeError:
        desc = df.describe(include="all").T
    miss_pct = (df.isna().mean() * 100).round(2)
    desc["missing_%"] = miss_pct.reindex(desc.index).values
    if target and target in df.columns and target in desc.index:
        desc.loc[target, "note"] = "Target variable"
    return desc.reset_index().rename(columns={"index": "feature"})

def corr_matrix(df, out_dir, max_cols_plot=60):
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] == 0:
        return
    corr = num.corr(method="spearman")
    corr.to_csv(os.path.join(out_dir, "corr_spearman.csv"))
    # If too many columns, just skip the heatmap (CSV still saved)
    if corr.shape[0] > max_cols_plot:
        return
    plt.figure(figsize=(max(8, corr.shape[1]*0.25), max(6, corr.shape[0]*0.25)))
    im = plt.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title("Spearman correlation")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=6)
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=6)
    plt.tight_layout()
    plt.savefig(fig_path(out_dir, "corr_spearman"))
    plt.close()

def target_relationships(df, target, out_dir, max_show=15):
    if target not in df.columns:
        return
    if not pd.api.types.is_numeric_dtype(df[target]):
        try:
            df[target] = pd.to_numeric(df[target], errors="coerce")
        except Exception:
            return
    num_cols = df.select_dtypes(include=[np.number]).columns.drop(target, errors="ignore")
    if len(num_cols) == 0:
        return
    corr = df[num_cols].corrwith(df[target], method="spearman").sort_values(key=lambda s: s.abs(), ascending=False)
    corr_df = corr.reset_index().rename(columns={"index": "feature", 0: "corr_with_target"})
    save_csv(corr_df, out_dir, "target_corr_rank")
    top = corr_df.head(max_show)
    plt.figure(figsize=(8, 4))
    plt.barh(top["feature"][::-1], top["corr_with_target"][::-1])
    plt.xlabel("Spearman correlation with target")
    plt.title("Top correlated features")
    plt.tight_layout()
    plt.savefig(fig_path(out_dir, "target_corr_rank"))
    plt.close()

def missingness_overview(df, out_dir, top_n=30):
    miss = df.isna().mean().sort_values(ascending=False)
    miss_df = miss.reset_index().rename(columns={"index": "feature", 0: "missing_pct"})
    miss_df["missing_pct"] = (miss_df["missing_pct"] * 100).round(2)
    save_csv(miss_df, out_dir, "missingness")
    plt.figure(figsize=(8, 4))
    labels = miss.index[:top_n][::-1]
    vals = miss.values[:top_n][::-1] * 100
    plt.barh(labels, vals)
    plt.xlabel("Missing %")
    plt.title(f"Top missing columns (up to {top_n})")
    plt.tight_layout()
    plt.savefig(fig_path(out_dir, "missingness"))
    plt.close()

def variance_inflation_factors(X: pd.DataFrame) -> pd.DataFrame:
    """Compute simple VIFs with mean-imputation + standardization.
    Correct R^2 = 1 - SSE/SST. Assumes X has no constant columns."""
    from numpy.linalg import lstsq
    X = X.apply(lambda s: s.fillna(s.mean()))
    # avoid 0 std
    std = X.std(ddof=0).replace(0, np.nan)
    Xs = (X - X.mean()) / std
    Xs = Xs.dropna(axis=1, how="any")  # drop any column that ended up NaN after std
    cols = list(Xs.columns)
    res = []
    for col in cols:
        y = Xs[col].values.reshape(-1, 1)
        Xj = Xs.drop(columns=[col]).values
        if Xj.shape[1] == 0:
            continue
        # add intercept
        Xj_i = np.column_stack([np.ones(len(Xj)), Xj])
        beta, *_ = lstsq(Xj_i, y, rcond=None)
        yhat = Xj_i @ beta
        sse = ((y - yhat) ** 2).sum()
        sst = ((y - y.mean()) ** 2).sum()
        R2 = float(1.0 - sse / sst) if sst > 0 else 0.0
        vif = np.inf if (1.0 - R2) <= 1e-12 else 1.0 / (1.0 - R2)
        res.append((col, vif))
    return pd.DataFrame(res, columns=["feature", "VIF"]).sort_values("VIF", ascending=False)

def vif_summary(df: pd.DataFrame, out_dir: str, max_rows: int = 5000, miss_thresh: float = 0.60):
    """Safe VIF: drop all-NaN/constant cols, allow some missingness,
    mean-impute, then sample rows if big."""
    num = df.select_dtypes(include=[np.number]).copy()
    if num.shape[1] < 2:
        return
    nun = num.nunique(dropna=True)
    cols = nun[nun > 1].index.tolist()
    if len(cols) < 2:
        return
    miss = num[cols].isna().mean()
    cols = miss[miss < miss_thresh].index.tolist()
    if len(cols) < 2:
        return
    X = num[cols]
    if len(X) == 0:
        return
    n = len(X)
    if n > max_rows:
        X = X.sample(max_rows, random_state=42)
    vif = variance_inflation_factors(X)
    save_csv(vif, out_dir, "vif")

def monthly_trends(df, date_col, out_dir, features, limit=6):
    if date_col not in df.columns:
        return
    if not np.issubdtype(df[date_col].dtype, np.datetime64):
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        except Exception:
            return
    month = df[date_col].dt.to_period("M")
    take = [c for c in features if c in df.columns][:limit]
    for c in take:
        s = pd.to_numeric(df[c], errors="coerce")
        m = s.groupby(month).mean()
        if m.dropna().empty:
            continue
        plt.figure()
        m.plot(marker="o")
        plt.title(f"Monthly mean: {c}")
        plt.xlabel("Month"); plt.ylabel(c)
        plt.tight_layout()
        plt.savefig(fig_path(out_dir, f"monthly__{c}"))
        plt.close()

def id_and_date_audits(df, id_col, date_col, out_dir):
    lines = []
    if id_col in df.columns:
        nuniq = df[id_col].nunique(dropna=True)
        dup = len(df) - nuniq
        lines.append(f"Duplicate {id_col} count: {dup:,}")
        if dup > 0:
            dup_ids = (df[id_col].value_counts().loc[lambda s: s > 1].reset_index()
                       .rename(columns={"index": id_col, id_col: "count"}))
            save_csv(dup_ids, out_dir, "duplicate_ids")
    else:
        lines.append(f"{id_col} not in columns — duplicate audit skipped.")
    if date_col in df.columns:
        try:
            d = pd.to_datetime(df[date_col], errors="coerce")
            lines.append(f"Date coverage: {str(pd.Series([d.min(), d.max()]).dt.strftime('%Y-%m-%d').tolist())}")
            # rows per month
            pm = d.dt.to_period("M").value_counts().sort_index()
            pm_df = pm.reset_index().rename(columns={"index": "month", 0: "rows"})
            save_csv(pm_df, out_dir, "rows_per_month")
            plt.figure()
            pm.sort_index().plot(kind="bar", rot=45)
            plt.ylabel("Rows")
            plt.title("Rows per month")
            plt.tight_layout()
            plt.savefig(fig_path(out_dir, "rows_per_month"))
            plt.close()
        except Exception:
            lines.append("Failed to parse date_col for coverage.")
    else:
        lines.append(f"{date_col} not in columns — date coverage skipped.")
    return lines

def target_by_month(df, target, date_col, out_dir):
    if target not in df.columns or date_col not in df.columns:
        return
    try:
        d = pd.to_datetime(df[date_col], errors="coerce")
        y = pd.to_numeric(df[target], errors="coerce")
        pm = d.dt.to_period("M")
        g = y.groupby(pm).mean()
        if not g.dropna().empty:
            plt.figure()
            g.plot(marker="o")
            plt.ylabel(f"{target} mean")
            plt.title(f"{target} by month")
            plt.tight_layout()
            plt.savefig(fig_path(out_dir, f"{target}_by_month"))
            plt.close()
            save_csv(g.reset_index().rename(columns={target: "mean"}), out_dir, f"{target}_by_month")
    except Exception:
        pass

def simple_leakage_flags(df, out_dir):
    """Flag columns that look post-game or overly informative."""
    flags = []
    cols = set([c.lower() for c in df.columns])
    suspicious_tokens = [
        "score", "final", "result", "status", "win", "loss", "wl", "outcome",
        "prob_after", "post", "settled", "closing", "implied", "total_runs"
    ]
    for t in suspicious_tokens:
        hits = [c for c in df.columns if t in c.lower()]
        if hits:
            flags.append(f"Found suspicious token '{t}' in columns: {hits[:10]}{'...' if len(hits)>10 else ''}")
    if flags:
        with open(os.path.join(out_dir, "leakage_flags.txt"), "w") as f:
            f.write("\n".join(flags))

# --------------------------
# Main
# --------------------------
def main(args):
    out_dir = ensure_dir("eda_out_plus")
    df = pd.read_csv(args.data)
    df = _replace_nonfinite(df)
    df = _coerce_datetime(df, args.date_col)
    df = df.sort_values(args.date_col, na_position="last").reset_index(drop=True)

    # 0. Overview + audits
    overview = [
        f"Rows: {len(df):,}",
        f"Columns: {df.shape[1]}",
        f"Has {args.id_col}: {args.id_col in df.columns}",
        f"Has {args.target}: {args.target in df.columns}",
        f"Has {args.date_col}: {args.date_col in df.columns}",
    ]
    overview.extend(id_and_date_audits(df, args.id_col, args.date_col, out_dir))

    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write("\n".join(overview))

    # 1. Summary statistics
    desc = summary_stats(df, args.target)
    save_csv(desc, out_dir, "summary_stats")

    # 2. Missingness overview
    missingness_overview(df, out_dir)

    # 3. Correlation matrix (CSV always, heatmap if <= 60 cols)
    corr_matrix(df, out_dir)

    # 4. Feature-target relationships
    target_relationships(df, args.target, out_dir)

    # 5. VIF for multicollinearity (safe)
    vif_summary(df, out_dir)

    # 6. Monthly feature stability
    candidates = [
        "SP_xFIP_diff","SP_FIP_diff","SP_KminusBB_diff",
        "Lineup_woba_vsHand_diff","Team_WinPct_diff","Team_Elo_diff",
        "BP_xFIP_lastN_diff","BP_IP_lastN_diff"
    ]
    monthly_trends(df, args.date_col, out_dir, [c for c in candidates if c in df.columns])

    # 7. Target by month (class balance drift)
    target_by_month(df, args.target, args.date_col, out_dir)

    # 8. Simple leakage flags
    simple_leakage_flags(df, out_dir)

    # 9. HTML report
    imgs = [f for f in os.listdir(out_dir) if f.endswith(".png")]
    html = f"""
    <html>
    <head>
    <meta charset="utf-8"/><title>MLB Enhanced EDA</title>
    <style>
        body{{font-family:Arial;margin:24px}}
        img{{max-width:100%;height:auto;margin:6px 0}}
        pre{{background:#f5f5f5;padding:8px}}
        a{{text-decoration:none}}
        ul{{line-height:1.6}}
        code{{background:#f0f0f0;padding:2px 4px;border-radius:3px}}
    </style>
    </head><body>
        <h1>MLB Enhanced EDA Report</h1>
        <p><b>Generated:</b> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <h2>Summary</h2><pre>{os.linesep.join(overview)}</pre>
        <h2>Artifacts</h2>
        <ul>
            <li><a href="summary_stats.csv">Summary statistics</a></li>
            <li><a href="missingness.csv">Missingness</a></li>
            <li><a href="corr_spearman.csv">Correlation matrix (Spearman)</a></li>
            <li><a href="target_corr_rank.csv">Target correlations</a></li>
            <li><a href="vif.csv">Variance inflation factors</a></li>
            <li><a href="rows_per_month.csv">Rows per month</a> (if available)</li>
            <li><a href="duplicate_ids.csv">Duplicate IDs</a> (if any)</li>
            <li><a href="leakage_flags.txt">Leakage flags</a> (if any)</li>
        </ul>
        <h2>Plots</h2>
        {''.join(f'<img src="{f}"/>' for f in sorted(imgs))}
    </body></html>
    """
    with open(os.path.join(out_dir, "report.html"), "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✅ Enhanced EDA complete. See: {out_dir}/report.html")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--date-col", default="game_date")
    p.add_argument("--id-col", default="game_pk")
    p.add_argument("--target", default="home_win")
    args = p.parse_args()
    main(args)
