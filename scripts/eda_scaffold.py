
"""
Enhanced EDA Scaffold for MLB Modeling
--------------------------------------

Usage:
    python eda_plus.py --data "out/mlb_features_model_20240328_to_20240929.csv" \
        --date-col game_date --id-col game_pk --target home_win

Outputs:
    - eda_out_plus/  (plots, tables, summaries)
    - eda_out_plus/report.html  (rich HTML summary)
"""

import argparse, os, warnings
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

# --------------------------
# Core Helpers
# --------------------------
def summary_stats(df, target=None):
    try:
        desc = df.describe(include='all', datetime_is_numeric=True).T
    except TypeError:
        desc = df.describe(include='all').T
    desc["missing_%"] = (df.isna().mean() * 100).round(2)
    if target and target in df.columns:
        desc.loc[target, "note"] = "Target variable"
    return desc.reset_index().rename(columns={"index":"feature"})


def corr_matrix(df, out_dir):
    num = df.select_dtypes(include=[np.number])
    corr = num.corr(method='spearman')
    plt.figure(figsize=(8,6))
    im = plt.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title("Spearman correlation")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=6)
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=6)
    plt.tight_layout()
    plt.savefig(fig_path(out_dir, "corr_spearman"))
    plt.close()
    corr.to_csv(os.path.join(out_dir, "corr_spearman.csv"))

def target_relationships(df, target, out_dir, max_show=15):
    if target not in df.columns: return
    num_cols = df.select_dtypes(include=[np.number]).columns.drop(target, errors="ignore")
    # correlation with target
    corr = df[num_cols].corrwith(df[target], method="spearman").sort_values(key=abs, ascending=False)
    corr_df = corr.reset_index().rename(columns={"index":"feature",0:"corr_with_target"})
    save_csv(corr_df, out_dir, "target_corr_rank")
    top_feats = corr_df.head(max_show)["feature"]
    # plot
    plt.figure(figsize=(8,4))
    plt.barh(top_feats[::-1], corr_df.head(max_show)["corr_with_target"][::-1])
    plt.xlabel("Spearman correlation with target")
    plt.title("Top correlated features")
    plt.tight_layout()
    plt.savefig(fig_path(out_dir, "target_corr_rank"))
    plt.close()

def missingness_heatmap(df, out_dir):
    miss = df.isna().mean().sort_values(ascending=False)
    plt.figure(figsize=(8,4))
    plt.barh(miss.index[:30][::-1], miss.values[:30][::-1]*100)
    plt.xlabel("Missing %")
    plt.title("Top missing columns (up to 30)")
    plt.tight_layout()
    plt.savefig(fig_path(out_dir, "missingness"))
    plt.close()
    save_csv(miss.reset_index().rename(columns={"index":"feature",0:"missing_pct"}), out_dir, "missingness")

def variance_inflation_factors(X: pd.DataFrame) -> pd.DataFrame:
    """Compute simple VIFs with mean-imputation + standardization.
    Assumes X has no constant columns."""
    from numpy.linalg import lstsq

    # mean-impute just for VIF math
    X = X.apply(lambda s: s.fillna(s.mean()))
    # standardize (avoid divide-by-zero — constants already removed)
    Xs = (X - X.mean()) / X.std(ddof=0)

    cols = list(Xs.columns)
    res = []
    for col in cols:
        y = Xs[col].values.reshape(-1, 1)
        Xj = Xs.drop(columns=[col]).values
        # add intercept
        Xj_i = np.column_stack([np.ones(len(Xj)), Xj])
        beta, *_ = lstsq(Xj_i, y, rcond=None)
        yhat = Xj_i @ beta
        ssr = ((yhat - y.mean()) ** 2).sum()
        sst = ((y - y.mean()) ** 2).sum()
        R2 = float(ssr / sst) if sst != 0 else 0.0
        vif = np.inf if (1 - R2) <= 1e-9 else 1.0 / (1.0 - R2)
        res.append((col, vif))
    return pd.DataFrame(res, columns=["feature", "VIF"]).sort_values("VIF", ascending=False)


def vif_summary(df: pd.DataFrame, out_dir: str, max_rows: int = 5000, miss_thresh: float = 0.60):
    """Safe VIF: drop all-NaN/constant cols, allow some missingness,
    mean-impute, then sample rows if big."""
    num = df.select_dtypes(include=[np.number]).copy()

    # drop all-NaN and constant columns
    nun = num.nunique(dropna=True)
    cols = nun[nun > 1].index.tolist()
    if len(cols) < 2:
        # too few usable numeric columns; skip
        return

    # keep columns with acceptable missingness
    miss = num[cols].isna().mean()
    cols = miss[miss < miss_thresh].index.tolist()
    if len(cols) < 2:
        return

    X = num[cols]
    if len(X) == 0:
        return

    # downsample rows ONLY if we have some
    n = len(X)
    if n > max_rows:
        X = X.sample(max_rows, random_state=42)

    vif = variance_inflation_factors(X)
    save_csv(vif, out_dir, "vif")


def monthly_trends(df, date_col, out_dir, features, limit=6):
    month = df[date_col].dt.to_period("M")
    for c in features[:limit]:
        if c not in df.columns: continue
        m = df.groupby(month)[c].mean()
        plt.figure()
        m.plot(marker="o")
        plt.title(f"Monthly mean: {c}")
        plt.xlabel("Month"); plt.ylabel(c)
        plt.tight_layout()
        plt.savefig(fig_path(out_dir, f"monthly__{c}"))
        plt.close()

# --------------------------
# Main
# --------------------------
def main(args):
    out_dir = ensure_dir("eda_out_plus")
    df = pd.read_csv(args.data, parse_dates=[args.date_col])
    df = df.sort_values(args.date_col).reset_index(drop=True)

    # 1. Overview
    overview = [
        f"Rows: {len(df):,}",
        f"Columns: {df.shape[1]}",
        f"Unique {args.id_col}: {df[args.id_col].nunique():,}" if args.id_col in df.columns else "",
        f"Target mean: {df[args.target].mean():.3f}" if args.target in df.columns else ""
    ]
    with open(os.path.join(out_dir, "summary.txt"), "w") as f: f.write("\n".join(overview))

    # 2. Summary statistics
    desc = summary_stats(df, args.target)
    save_csv(desc, out_dir, "summary_stats")

    # 3. Missingness overview
    missingness_heatmap(df, out_dir)

    # 4. Correlation matrix
    corr_matrix(df, out_dir)

    # 5. Feature-target relationships
    target_relationships(df, args.target, out_dir)

    # 6. VIF for multicollinearity
    vif_summary(df, out_dir)

    # 7. Monthly feature stability
    candidates = [
        "SP_xFIP_diff","SP_FIP_diff","SP_KminusBB_diff",
        "Lineup_woba_vsHand_diff","Team_WinPct_diff","Team_Elo_diff",
        "BP_xFIP_lastN_diff","BP_IP_lastN_diff"
    ]
    monthly_trends(df, args.date_col, out_dir, [c for c in candidates if c in df.columns])

    # 8. HTML report
    imgs = [f for f in os.listdir(out_dir) if f.endswith(".png")]
    html = f"""
    <html>
    <head>
    <meta charset="utf-8"/><title>MLB Enhanced EDA</title>
    <style>
        body{{font-family:Arial;margin:24px}}
        img{{max-width:100%;height:auto;margin:6px 0}}
        pre{{background:#f5f5f5;padding:8px}}
    </style>
    </head><body>
        <h1>MLB Enhanced EDA Report</h1>
        <p><b>Generated:</b> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <h2>Summary</h2><pre>{os.linesep.join(overview)}</pre>
        <h2>Artifacts</h2>
        <ul>
            <li><a href="summary_stats.csv">Summary statistics</a></li>
            <li><a href="missingness.csv">Missingness</a></li>
            <li><a href="corr_spearman.csv">Correlation matrix</a></li>
            <li><a href="target_corr_rank.csv">Target correlations</a></li>
            <li><a href="vif.csv">Variance inflation factors</a></li>
        </ul>
        <h2>Plots</h2>
        {''.join(f'<img src="{f}"/>' for f in imgs)}
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
