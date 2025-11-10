"""
model_train_refined.py — TIMESERIES + MONTHLY WF + BETTING

Includes:
- Feature pruning by valid-set permutation importance
- Logistic + Boosting (LGBM if available else GB), isotonic calibration
- Threshold strategies: f1 / youden / balanced
- Rolling-origin CV (chronological k-fold)
- Optional leakage-safe rolling features (shifted, grouped)
- NEW: Monthly walk-forward retrain evaluation
- NEW: Betting/EV module with American moneylines, flat & Kelly

Example:
  python model_train_refined.py --data "out\\mlb_features_prepared.csv" ^
    --date-col game_date --id-col game_pk --target home_win ^
    --outdir "model_out_refined" --topk 25 --cv-folds 6 --threshold-strategy f1 ^
    --add-rolling ^
    --home-odds-col home_moneyline --away-odds-col away_moneyline --kelly-cap 0.05
"""

import argparse, os, warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import (
    roc_auc_score, accuracy_score, log_loss, brier_score_loss,
    f1_score, precision_recall_curve, roc_curve, auc, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore", category=FutureWarning)
plt.rcParams.update({"figure.dpi": 120})

# Optional LightGBM
try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False


# --------------------------
# Utilities
# --------------------------
def ensure_dir(p): os.makedirs(p, exist_ok=True); return p
def fig_path(out_dir, name): return os.path.join(out_dir, f"{name}.png")
def save_csv(df, out_dir, name):
    p = os.path.join(out_dir, f"{name}.csv"); df.to_csv(p, index=False); return p

def pick_features(df, target, id_col, date_col):
    drop = {target, id_col, date_col}
    feats = []
    for c in df.columns:
        if c in drop: continue
        if "_id" in c.lower(): continue
        if pd.api.types.is_numeric_dtype(df[c]): feats.append(c)
    return feats

def time_split(df, frac_train=0.75, frac_valid=0.15):
    n = len(df); s1 = int(n*frac_train); s2 = int(n*(frac_train+frac_valid))
    return (0, s1), (s1, s2), (s2, n)

def eval_probs(y_true, p, thr=0.5):
    p_safe = np.clip(p, 1e-15, 1 - 1e-15)
    yhat = (p_safe >= thr).astype(int)
    return dict(
        auc=float(roc_auc_score(y_true, p_safe)),
        acc=float(accuracy_score(y_true, yhat)),
        logloss=float(log_loss(y_true, p_safe)),
        brier=float(brier_score_loss(y_true, p_safe)),
        f1=float(f1_score(y_true, yhat)),
    )

def choose_threshold(y_true, p, strategy="f1"):
    ps = np.linspace(0.2, 0.8, 121)
    if strategy == "f1":
        scores = [f1_score(y_true, (p>=t).astype(int)) for t in ps]
        return float(ps[int(np.argmax(scores))]), float(np.max(scores))
    elif strategy == "youden":
        best_t, best_j = 0.5, -1
        for t in ps:
            yhat = (p>=t).astype(int)
            tp = np.sum((y_true==1)&(yhat==1))
            fn = np.sum((y_true==1)&(yhat==0))
            fp = np.sum((y_true==0)&(yhat==1))
            tn = np.sum((y_true==0)&(yhat==0))
            tpr = tp / max(1, tp+fn); fpr = fp / max(1, fp+tn)
            j = tpr - fpr
            if j > best_j: best_j, best_t = j, t
        return float(best_t), float(best_j)
    elif strategy == "balanced":
        prev = float(np.mean(y_true))
        diffs = [abs(np.mean(p>=t) - prev) for t in ps]
        return float(ps[int(np.argmin(diffs))]), np.nan
    else:
        return 0.5, np.nan

def plot_importance(values, names, title, base_path, top_k=20, show_tail=True):
    order = np.argsort(np.abs(values))
    names = np.array(names)[order]; vals = np.array(values)[order]
    # top
    top_names, top_vals = names[-top_k:], vals[-top_k:]
    plt.figure(figsize=(8, max(3, 0.35*len(top_names))))
    plt.barh(top_names, top_vals); plt.title(f"{title} — Top {len(top_names)}")
    plt.tight_layout(); plt.savefig(base_path.replace(".png","__top.png")); plt.close()
    # tail
    if show_tail:
        tail_n = min(top_k, len(names))
        tail_names, tail_vals = names[:tail_n], vals[:tail_n]
        plt.figure(figsize=(8, max(3, 0.35*tail_n)))
        plt.barh(tail_names, tail_vals); plt.title(f"{title} — Smallest {tail_n} (near 0)")
        plt.tight_layout(); plt.savefig(base_path.replace(".png","__tail.png")); plt.close()

def plot_roc_pr(y_true, p_dict, out_dir, prefix):
    # ROC
    plt.figure()
    for name, proba in p_dict.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr,tpr):.3f})")
    plt.plot([0,1],[0,1],"--",alpha=.5); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"ROC — {prefix}"); plt.legend(); plt.tight_layout()
    plt.savefig(fig_path(out_dir, f"roc__{prefix}")); plt.close()
    # PR
    plt.figure()
    for name, proba in p_dict.items():
        pr, rc, _ = precision_recall_curve(y_true, proba)
        plt.plot(rc, pr, label=name)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR — {prefix}")
    plt.legend(); plt.tight_layout()
    plt.savefig(fig_path(out_dir, f"pr__{prefix}")); plt.close()

def plot_calibration(y_true, p_dict, out_dir, prefix, bins=7):
    from sklearn.calibration import calibration_curve
    plt.figure()
    for name, proba in p_dict.items():
        frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=bins, strategy="quantile")
        plt.plot(mean_pred, frac_pos, marker="o", label=name)
    plt.plot([0,1],[0,1],"--",alpha=.5); plt.xlabel("predicted prob"); plt.ylabel("observed win rate")
    plt.title(f"Reliability — {prefix}"); plt.legend(); plt.tight_layout()
    plt.savefig(fig_path(out_dir, f"calibration__{prefix}")); plt.close()

def confusion_plot(cm, title, out_path):
    plt.figure(); plt.imshow(cm, cmap="Blues")
    for (i,j),v in np.ndenumerate(cm): plt.text(j,i,str(v),ha="center",va="center")
    plt.xticks([0,1],["Pred 0","Pred 1"]); plt.yticks([0,1],["True 0","True 1"])
    plt.title(title); plt.tight_layout(); plt.savefig(out_path); plt.close()


# --------------------------
# Leakage-safe rolling feature augmentation (optional)
# --------------------------
ROLLING_CANDIDATES = [
    ("home_SP_xFIP", "home_sp_id", (3,5)),
    ("away_SP_xFIP", "away_sp_id", (3,5)),
    ("home_BP_xFIP", "home_team_id", (3,5)),
    ("away_BP_xFIP", "away_team_id", (3,5)),
    ("home_lineup_woba_vsHand", "home_team_id", (3,5)),
    ("away_lineup_woba_vsHand", "away_team_id", (3,5)),
]

def add_rolling(df, date_col):
    df = df.copy().sort_values(date_col)
    for col, key, wins in ROLLING_CANDIDATES:
        if col not in df.columns or key not in df.columns: 
            continue
        df[f"{col}_r3"] = (
            df.groupby(key)[col]
              .apply(lambda s: s.shift(1).rolling(wins[0], min_periods=1).mean())
              .values
        )
        df[f"{col}_r5"] = (
            df.groupby(key)[col]
              .apply(lambda s: s.shift(1).rolling(wins[1], min_periods=1).mean())
              .values
        )
    return df


# --------------------------
# Rolling-origin CV
# --------------------------
def rolling_origin_cv(df, date_col, target, features, cv_folds, build_pipelines):
    cuts = np.linspace(0, len(df), cv_folds+1, dtype=int)
    rows = []
    for i in range(1, len(cuts)):
        start = cuts[i-1]; end = cuts[i]
        if start == 0 or end-start < 10:    # need history + test rows
            continue
        tr_idx = np.arange(0, start); te_idx = np.arange(start, end)
        Xtr, ytr = df.iloc[tr_idx][features], df.iloc[tr_idx][target].astype(int).values
        Xte, yte = df.iloc[te_idx][features], df.iloc[te_idx][target].astype(int).values
        logit, gb_pipe = build_pipelines()
        logit.fit(Xtr, ytr); pte_l = logit.predict_proba(Xte)[:,1]
        gb_pipe.fit(Xtr, ytr); pte_g = gb_pipe.predict_proba(Xte)[:,1]
        rows.append([i, end-start, float(roc_auc_score(yte, pte_l)), float(roc_auc_score(yte, pte_g))])
    return pd.DataFrame(rows, columns=["fold","test_rows","auc_logit","auc_boost"])


# --------------------------
# Monthly walk-forward evaluation
# --------------------------
def monthly_walkforward(df, date_col, target, features, build_pipelines, out_dir):
    per = df[date_col].dt.to_period("M")
    months = sorted(per.unique())
    rows = []
    for k in range(1, len(months)):
        train_mask = per < months[k]
        test_mask  = per == months[k]
        if train_mask.sum() < 200 or test_mask.sum() < 20:  # safety
            continue
        Xtr, ytr = df.loc[train_mask, features], df.loc[train_mask, target].astype(int).values
        Xte, yte = df.loc[test_mask, features], df.loc[test_mask, target].astype(int).values
        logit, gb_pipe = build_pipelines()
        logit.fit(Xtr, ytr); pte_l = logit.predict_proba(Xte)[:,1]
        gb_pipe.fit(Xtr, ytr); pte_g = gb_pipe.predict_proba(Xte)[:,1]
        rows.append([str(months[k]), float(roc_auc_score(yte, pte_l)), float(roc_auc_score(yte, pte_g)), int(test_mask.sum())])
    mdf = pd.DataFrame(rows, columns=["month","auc_logit","auc_boost","rows"])
    if not mdf.empty:
        save_csv(mdf, out_dir, "monthly_walkforward")
        plt.figure()
        plt.plot(mdf["month"], mdf["auc_logit"], marker="o", label="LogReg"); 
        plt.plot(mdf["month"], mdf["auc_boost"], marker="o", label="Boost")
        plt.xticks(rotation=45); plt.ylabel("AUC"); plt.title("Monthly Walk-Forward AUC")
        plt.legend(); plt.tight_layout(); plt.savefig(fig_path(out_dir,"monthly_walkforward_auc")); plt.close()
    return mdf


# --------------------------
# Betting helpers (American moneyline)
# --------------------------
def american_to_decimal(ml):
    ml = float(ml)
    if ml > 0: return 1.0 + ml/100.0
    return 1.0 + 100.0/abs(ml)

def american_to_implied_prob(ml):
    ml = float(ml)
    if ml > 0: return 100.0 / (ml + 100.0)
    return abs(ml) / (abs(ml) + 100.0)

def devig_two_way(p_home_raw, p_away_raw):
    s = p_home_raw + p_away_raw
    if s <= 0: return np.nan, np.nan
    return p_home_raw/s, p_away_raw/s

def simulate_bets(df, p_home_model, y_true, home_ml, away_ml, min_edge=0.0, kelly_cap=0.05):
    """
    df-length vectors. Edge = model_p_home - devig_implied_home.
    Bet rule:
      - If edge >= min_edge: bet home at home_ml
      - Else if (1 - model_p_home) - implied_away >= min_edge: bet away at away_ml
      - Otherwise skip
    Returns dataframe with per-game profits for flat 1-unit and Kelly (capped).
    """
    out = []
    for i in range(len(df)):
        ml_h, ml_a = home_ml[i], away_ml[i]
        if np.isnan(ml_h) or np.isnan(ml_a): 
            out.append((0,0,0,0,0,0)); continue
        qh_raw, qa_raw = american_to_implied_prob(ml_h), american_to_implied_prob(ml_a)
        qh, qa = devig_two_way(qh_raw, qa_raw)
        if np.isnan(qh): out.append((0,0,0,0,0,0)); continue

        ph = float(np.clip(p_home_model[i], 1e-9, 1-1e-9))
        edge_home = ph - qh
        edge_away = (1 - ph) - qa

        flat = 0.0; kelly = 0.0; bet_dir = 0; edge_used = 0.0
        # choose side with positive edge
        if edge_home >= min_edge or edge_away >= min_edge:
            if edge_home >= edge_away:
                # bet home
                b = american_to_decimal(ml_h) - 1.0
                # Kelly fraction
                f = max(0.0, min(kelly_cap, (b*ph - (1-ph)) / b))
                # outcome
                win = 1 if y_true[i]==1 else 0
                flat = (b if win else -1)
                kelly = f * (b if win else -1)
                bet_dir = 1; edge_used = edge_home
            else:
                # bet away
                b = american_to_decimal(ml_a) - 1.0
                pa = 1 - ph
                f = max(0.0, min(kelly_cap, (b*pa - (1-pa)) / b))
                win = 1 if y_true[i]==0 else 0
                flat = (b if win else -1)
                kelly = f * (b if win else -1)
                bet_dir = -1; edge_used = edge_away
        out.append((flat, kelly, bet_dir, edge_used, qh, ph))
    return pd.DataFrame(out, columns=["flat_profit","kelly_profit","bet_dir","edge","implied_home","model_home"])


# --------------------------
# Main
# --------------------------
def main(args):
    outdir = ensure_dir(args.outdir)
    df = pd.read_csv(args.data, parse_dates=[args.date_col]).sort_values(args.date_col).reset_index(drop=True)

    # Optional rolling features
    if args.add_rolling:
        df = add_rolling(df, args.date_col)

    # Splits
    (a,b),(b,c),(c,d) = time_split(df, 0.75, 0.15)
    splits_msg = [
        f"Train: {b-a} rows (index {a}:{b})  target_mean={df.iloc[a:b][args.target].mean():.3f}",
        f"Valid: {c-b} rows (index {b}:{c})  target_mean={df.iloc[b:c][args.target].mean():.3f}",
        f"Test : {d-c} rows (index {c}:{d})  target_mean={df.iloc[c:d][args.target].mean():.3f}",
    ]
    with open(os.path.join(outdir, "splits.txt"), "w") as f: f.write("\n".join(splits_msg))
    print("\n".join(splits_msg))

    # Features
    features_all = pick_features(df, args.target, args.id_col, args.date_col)
    X = df[features_all].copy(); y = df[args.target].astype(int).values
    Xtr, ytr = X.iloc[a:b], y[a:b]
    Xva, yva = X.iloc[b:c], y[b:c]
    Xte, yte = X.iloc[c:d], y[c:d]

    # ---------- Feature pruning ----------
    base_logit = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000))
    ])
    base_logit.fit(Xtr, ytr)
    pi = permutation_importance(base_logit, Xva, yva, n_repeats=8, random_state=42)
    rank_df = pd.DataFrame({"feature": features_all, "perm_importance": pi.importances_mean}).sort_values("perm_importance", ascending=False)
    save_csv(rank_df, outdir, "perm_importance_valid")
    keep = rank_df.head(args.topk)["feature"].tolist()
    keep = [c for c in keep if Xtr[c].notna().any()]  # drop all-NaN-on-train

    with open(os.path.join(outdir,"kept_features.txt"),"w") as f: f.write("\n".join(keep))
    print(f"[prune] keeping top {len(keep)} features for final models.")

    Xtr, Xva, Xte = Xtr[keep], Xva[keep], Xte[keep]

    # ---------- Pipelines ----------
    def build_pipelines():
        logit = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000))
        ])
        if HAS_LGB:
            gbm = lgb.LGBMClassifier(
                n_estimators=1200, learning_rate=0.02, num_leaves=31,
                colsample_bytree=0.9, subsample=0.8, reg_lambda=0.5,
                random_state=42
            )
            gb_pipe = make_pipeline(SimpleImputer(strategy="median"), gbm)
        else:
            gbm = GradientBoostingClassifier(
                n_estimators=1000, learning_rate=0.01, max_depth=3,
                subsample=0.9, random_state=42
            )
            gb_pipe = make_pipeline(SimpleImputer(strategy="median"), gbm)
        return logit, gb_pipe

    logit, gb_pipe = build_pipelines()

    # Fit & calibrate logistic
    logit.fit(Xtr, ytr)
    pva_raw = logit.predict_proba(Xva)[:,1]
    pte_raw = logit.predict_proba(Xte)[:,1]
    iso_logit = IsotonicRegression(out_of_bounds="clip").fit(pva_raw, yva)
    pva_cal = iso_logit.predict(pva_raw)
    pte_cal = iso_logit.predict(pte_raw)

    # Fit & calibrate boosting
    gb_pipe.fit(Xtr, ytr)
    pva_raw_gb = gb_pipe.predict_proba(Xva)[:,1]
    pte_raw_gb = gb_pipe.predict_proba(Xte)[:,1]
    iso_gb = IsotonicRegression(out_of_bounds="clip").fit(pva_raw_gb, yva)
    pva_cal_gb = iso_gb.predict(pva_raw_gb)
    pte_cal_gb = iso_gb.predict(pte_raw_gb)

    # Thresholds
    thr_logit, aux_log = choose_threshold(yva, pva_cal, strategy=args.threshold_strategy)
    thr_gb,    aux_gb  = choose_threshold(yva, pva_cal_gb, strategy=args.threshold_strategy)

    m_log_v = eval_probs(yva, pva_cal, thr_logit)
    m_log_t = eval_probs(yte, pte_cal, thr_logit)
    m_gb_v  = eval_probs(yva, pva_cal_gb, thr_gb)
    m_gb_t  = eval_probs(yte, pte_cal_gb, thr_gb)

    # Logistic importances
    coef = logit.named_steps["clf"].coef_.ravel()
    save_csv(pd.DataFrame({"feature": keep, "coef": coef, "abs_coef": np.abs(coef)}).sort_values("abs_coef", ascending=False),
             outdir, "logit_coeffs_kept")
    plot_importance(coef, keep, "Logistic — |standardized coef|", fig_path(outdir, "logit_importance_kept"))

    # Boosting importances
    if HAS_LGB:
        imp = gb_pipe.named_steps['lgbmclassifier'].feature_importances_
        imp_title = "LightGBM — Gain"; imp_base = "lgbm_importance_kept"
    else:
        imp = gb_pipe.named_steps['gradientboostingclassifier'].feature_importances_
        imp_title = "Gradient Boosting — Gain"; imp_base = "gb_importance_kept"
    save_csv(pd.DataFrame({"feature": keep, "importance": imp}).sort_values("importance", ascending=False),
             outdir, imp_base)
    plot_importance(imp, keep, imp_title, fig_path(outdir, imp_base))

    # Permutation importance on test
    pi_gb = permutation_importance(gb_pipe, Xte, yte, n_repeats=8, random_state=42)
    save_csv(pd.DataFrame({"feature": keep, "perm_importance_test": pi_gb.importances_mean})
             .sort_values("perm_importance_test", ascending=False), outdir, "gb_perm_importance_test")

    # Curves & confusion
    plot_roc_pr(yva, {"LogReg+Cal": pva_cal, "Boost+Cal": pva_cal_gb}, outdir, "valid")
    plot_roc_pr(yte, {"LogReg+Cal": pte_cal, "Boost+Cal": pte_cal_gb}, outdir, "test")
    plot_calibration(yte, {"LogReg+Cal": pte_cal, "Boost+Cal": pte_cal_gb}, outdir, "test")
    confusion_plot(confusion_matrix(yte, (pte_cal>=thr_logit).astype(int)),
                   f"Confusion — Test — Logistic (thr={thr_logit:.2f}, {args.threshold_strategy})",
                   fig_path(outdir,"cm_test_logit"))
    confusion_plot(confusion_matrix(yte, (pte_cal_gb>=thr_gb).astype(int)),
                   f"Confusion — Test — Boost (thr={thr_gb:.2f}, {args.threshold_strategy})",
                   fig_path(outdir,"cm_test_boost"))

    # Rolling-origin CV
    if args.cv_folds and args.cv_folds >= 3:
        df_cv = df[[args.date_col, args.target] + keep].copy()
        cv_df = rolling_origin_cv(df_cv, args.date_col, args.target, keep, args.cv_folds, build_pipelines)
        save_csv(cv_df, outdir, "timeseries_cv")
        plt.figure()
        plt.plot(cv_df["fold"], cv_df["auc_logit"], marker="o", label="LogReg")
        plt.plot(cv_df["fold"], cv_df["auc_boost"], marker="o", label="Boost")
        plt.xlabel("Fold (chronological)"); plt.ylabel("AUC"); plt.title("Rolling-Origin CV AUC")
        plt.legend(); plt.tight_layout(); plt.savefig(fig_path(outdir, "timeseries_cv_auc")); plt.close()

    # Monthly walk-forward retrain
    monthly_walkforward(df[[args.date_col, args.target] + keep].copy(), args.date_col, args.target, keep, build_pipelines, outdir)

    # ---------- Betting / EV module ----------
    if args.home_odds_col and args.away_odds_col and args.home_odds_col in df.columns and args.away_odds_col in df.columns:
        test_df = df.iloc[c:d].copy()
        # Simulate using LOGIT probabilities (calibrated)
        sim = simulate_bets(
            test_df, pte_cal, yte,
            test_df[args.home_odds_col].astype(float).values,
            test_df[args.away_odds_col].astype(float).values,
            min_edge=args.min_edge, kelly_cap=args.kelly_cap
        )
        # Cumulative curves
        sim["flat_cum"] = sim["flat_profit"].cumsum()
        sim["kelly_cum"] = sim["kelly_profit"].cumsum()
        save_csv(pd.concat([test_df[[args.date_col]], sim], axis=1), outdir, "betting_sim_logit_test")

        plt.figure()
        plt.plot(sim["flat_cum"], label="Flat (1u)")
        plt.plot(sim["kelly_cum"], label=f"Kelly (cap {args.kelly_cap:.2f})")
        plt.title(f"Betting Cumulative Profit — Test (min_edge={args.min_edge})"); plt.xlabel("Bets"); plt.ylabel("Units")
        plt.legend(); plt.tight_layout(); plt.savefig(fig_path(outdir,"betting_cumulative")); plt.close()

        # Profit vs edge threshold scan
        edges = np.linspace(0.0, 0.08, 17)
        pts = []
        for e in edges:
            s2 = simulate_bets(
                test_df, pte_cal, yte,
                test_df[args.home_odds_col].astype(float).values,
                test_df[args.away_odds_col].astype(float).values,
                min_edge=float(e), kelly_cap=args.kelly_cap
            )
            pts.append([e, s2["flat_profit"].sum(), s2["kelly_profit"].sum(), int((s2["flat_profit"]!=0).sum())])
        pr_curve = pd.DataFrame(pts, columns=["min_edge","flat_units","kelly_units","num_bets"])
        save_csv(pr_curve, outdir, "betting_edge_sweep")
        plt.figure()
        plt.plot(pr_curve["min_edge"], pr_curve["flat_units"], marker="o", label="Flat units")
        plt.plot(pr_curve["min_edge"], pr_curve["kelly_units"], marker="o", label="Kelly units")
        plt.xlabel("Min Edge"); plt.ylabel("Units won on Test"); plt.title("Profit vs Edge Threshold")
        plt.legend(); plt.tight_layout(); plt.savefig(fig_path(outdir,"betting_profit_vs_edge")); plt.close()

    # Metrics & report
    metrics_df = pd.DataFrame([
        ["logistic+cal", "valid", args.threshold_strategy, thr_logit, m_log_v["f1"], m_log_v["auc"], m_log_v["acc"], m_log_v["logloss"], m_log_v["brier"]],
        ["logistic+cal", "test" , args.threshold_strategy, thr_logit, np.nan   , m_log_t["auc"], m_log_t["acc"], m_log_t["logloss"], m_log_t["brier"]],
        ["boost+cal"   , "valid", args.threshold_strategy, thr_gb   , m_gb_v["f1"], m_gb_v["auc"] , m_gb_v["acc"] , m_gb_v["logloss"] , m_gb_v["brier"]],
        ["boost+cal"   , "test" , args.threshold_strategy, thr_gb   , np.nan   , m_gb_t["auc"] , m_gb_t["acc"] , m_gb_t["logloss"] , m_gb_t["brier"]],
    ], columns=["model","split","thr_strategy","threshold","f1_at_thr","auc","acc","logloss","brier"])
    save_csv(metrics_df, outdir, "metrics_refined")

    imgs = [f for f in os.listdir(outdir) if f.endswith(".png")]
    html = f"""
    <html><head><meta charset="utf-8"/>
    <title>MLB Refined Model Report (Timeseries + Betting)</title>
    <style>
      body{{font-family:Arial;margin:24px}}
      img{{max-width:100%;height:auto;margin:6px 0}}
      pre{{background:#f5f5f5;padding:8px}}
      table{{border-collapse:collapse}}
      th,td{{border:1px solid #ddd;padding:4px 8px}}
    </style></head><body>
      <h1>MLB Refined Model Report (Timeseries + Betting)</h1>
      <p><b>Generated:</b> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
      <h2>Splits</h2>
      <pre>{os.linesep.join(splits_msg)}</pre>
      <h2>Kept Features (top {args.topk} by valid permutation importance)</h2>
      <pre>{"\\n".join(keep)}</pre>
      <h2>Metrics</h2>
      {metrics_df.to_html(index=False)}
      {"<h2>Rolling-Origin CV</h2>"+pd.read_csv(os.path.join(outdir,'timeseries_cv.csv')).to_html(index=False) if os.path.exists(os.path.join(outdir,'timeseries_cv.csv')) else ""}
      {"<h2>Monthly Walk-Forward</h2>"+pd.read_csv(os.path.join(outdir,'monthly_walkforward.csv')).to_html(index=False) if os.path.exists(os.path.join(outdir,'monthly_walkforward.csv')) else ""}
      <h2>Figures</h2>
      {''.join(f'<img src="{f}"/>' for f in sorted(imgs))}
    </body></html>
    """
    with open(os.path.join(outdir,"report_refined.html"),"w",encoding="utf-8") as f:
        f.write(html)

    print("\n✅ Refined training complete.")
    print("Artifacts:", outdir)
    print("Open report:", os.path.join(outdir, "report_refined.html"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--date-col", default="game_date")
    ap.add_argument("--id-col", default="game_pk")
    ap.add_argument("--target", default="home_win")
    ap.add_argument("--outdir", default="model_out_refined")
    ap.add_argument("--topk", type=int, default=25)
    ap.add_argument("--cv-folds", type=int, default=6)
    ap.add_argument("--threshold-strategy", choices=["f1","youden","balanced"], default="f1")
    ap.add_argument("--add-rolling", action="store_true")

    # Betting args (American moneylines)
    ap.add_argument("--home-odds-col", type=str, default=None, help="column with American moneyline for HOME team")
    ap.add_argument("--away-odds-col", type=str, default=None, help="column with American moneyline for AWAY team")
    ap.add_argument("--min-edge", type=float, default=0.02, help="minimum model edge over devig implied prob to bet")
    ap.add_argument("--kelly-cap", type=float, default=0.05, help="cap Kelly fraction (0..1)")

    args = ap.parse_args()
    main(args)




