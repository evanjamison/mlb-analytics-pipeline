"""
model_train_allinone.py — leaderboard + walk-forward + betting (ONE SCRIPT)

Example (PowerShell one-line):
  py model_train_allinone.py --data "out\\mlb_features_prepared.csv" --date-col game_date --id-col game_pk --target home_win --outdir "model_all_ts" --topk 25 --threshold-strategy f1 --add-rolling --add-interactions --calibration isotonic --walkforward ts --n-splits 6
"""

# ---- BACKEND FIX (must be before pyplot import) ----
import matplotlib
matplotlib.use("Agg")  # headless, avoids Tk/Tkinter issues on save/exit

import argparse, os, warnings, json, inspect
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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import calibration_curve
from sklearn import set_config

# ---- Ensure sklearn transformers emit pandas DataFrames (feature names preserved) ----
set_config(transform_output="pandas")

warnings.filterwarnings("ignore", category=FutureWarning)
plt.rcParams.update({"figure.dpi": 120})

# Optional libs
try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False
try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except Exception:
    HAS_CAT = False
try:
    from xgboost import XGBClassifier
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False


# ----------------------- utils -----------------------
def ensure_dir(p): os.makedirs(p, exist_ok=True); return p
def fig_path(out_dir, name): return os.path.join(out_dir, f"{name}.png")
def save_csv(df, out_dir, name):
    p = os.path.join(out_dir, f"{name}.csv"); df.to_csv(p, index=False); return p

def pick_features(df, target, id_col, date_col):
    """numeric features only, skip IDs/dates/targets and all-NaN numeric cols"""
    drop = {target, id_col, date_col}
    feats = []
    for c in df.columns:
        if c in drop:
            continue
        if "_id" in c.lower():
            continue
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any():
            feats.append(c)
    return feats

def time_split(df, frac_train=0.75, frac_valid=0.15):
    n = len(df); s1 = int(n*frac_train); s2 = int(n*(frac_train+frac_valid))
    return (0, s1), (s1, s2), (s2, n)

def eval_probs(y_true, p, thr=0.5):
    p = np.clip(p, 1e-15, 1-1e-15); yhat = (p>=thr).astype(int)
    return dict(
        auc=float(roc_auc_score(y_true, p)),
        acc=float(accuracy_score(y_true, yhat)),
        logloss=float(log_loss(y_true, p)),
        brier=float(brier_score_loss(y_true, p)),
        f1=float(f1_score(y_true, yhat))
    )

def choose_threshold(y_true, p, strategy="f1"):
    grid = np.linspace(0.2, 0.8, 121)
    if strategy=="f1":
        s=[f1_score(y_true,(p>=t).astype(int)) for t in grid]; i=int(np.argmax(s))
        return float(grid[i]), float(s[i])
    elif strategy=="youden":
        best,bt=-9,0.5
        for t in grid:
            yhat=(p>=t).astype(int)
            tp=((y_true==1)&(yhat==1)).sum(); fn=((y_true==1)&(yhat==0)).sum()
            fp=((y_true==0)&(yhat==1)).sum(); tn=((y_true==0)&(yhat==0)).sum()
            tpr=tp/max(1,tp+fn); fpr=fp/max(1,fp+tn); j=tpr-fpr
            if j>best: best=j; bt=t
        return float(bt), float(best)
    else: # balanced prevalence
        prev=float(np.mean(y_true)); diffs=[abs(np.mean(p>=t)-prev) for t in grid]
        return float(grid[int(np.argmin(diffs))]), np.nan

def plot_roc_pr(y_true, pdict, out_dir, prefix):
    # ROC
    plt.figure()
    for k,v in pdict.items():
        fpr,tpr,_=roc_curve(y_true,v); plt.plot(fpr,tpr,label=f"{k} (AUC={auc(fpr,tpr):.3f})")
    plt.plot([0,1],[0,1],"--",alpha=.4); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC — {prefix}")
    plt.legend(); plt.tight_layout(); plt.savefig(fig_path(out_dir,f"roc__{prefix}")); plt.close()
    # PR
    plt.figure()
    for k,v in pdict.items():
        pr,rc,_=precision_recall_curve(y_true,v); plt.plot(rc,pr,label=k)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR — {prefix}")
    plt.legend(); plt.tight_layout(); plt.savefig(fig_path(out_dir,f"pr__{prefix}")); plt.close()


def plot_calibration(y_true, pdict, out_dir, prefix, bins=7):
    plt.figure()
    for k,v in pdict.items():
        frac,mean=calibration_curve(y_true,v,n_bins=bins,strategy="quantile")
        plt.plot(mean,frac,marker="o",label=k)
    plt.plot([0,1],[0,1],'--',alpha=.4); plt.xlabel("pred prob"); plt.ylabel("observed"); plt.title(f"Reliability — {prefix}")
    plt.legend(); plt.tight_layout(); plt.savefig(fig_path(out_dir,f"calibration__{prefix}")); plt.close()

def confusion_plot(cm, title, out_path):
    plt.figure(); plt.imshow(cm,cmap="Blues")
    for (i,j),v in np.ndenumerate(cm): plt.text(j,i,str(v),ha="center",va="center")
    plt.xticks([0,1],["Pred 0","Pred 1"]); plt.yticks([0,1],["True 0","True 1"])
    plt.title(title); plt.tight_layout(); plt.savefig(out_path); plt.close()

# ---- Pandas-preserving imputer ----
def pd_imputer(strategy="median"):
    return SimpleImputer(strategy=strategy).set_output(transform="pandas")

# rolling features (safe, shifted)
ROLLING_CANDIDATES=[
    ("home_SP_xFIP","home_sp_id",(3,5)),
    ("away_SP_xFIP","away_sp_id",(3,5)),
    ("home_BP_xFIP","home_team_id",(3,5)),
    ("away_BP_xFIP","away_team_id",(3,5)),
    ("home_lineup_woba_vsHand","home_team_id",(3,5)),
    ("away_lineup_woba_vsHand","away_team_id",(3,5)),
]
def add_rolling(df, date_col):
    df=df.copy().sort_values(date_col)
    for col,key,w in ROLLING_CANDIDATES:
        if col not in df.columns or key not in df.columns: continue
        df[f"{col}_r3"]=df.groupby(key)[col].apply(lambda s:s.shift(1).rolling(w[0],min_periods=1).mean()).values
        df[f"{col}_r5"]=df.groupby(key)[col].apply(lambda s:s.shift(1).rolling(w[1],min_periods=1).mean()).values
    return df

# simple interaction builder (stat-only)
def add_interactions(df):
    df = df.copy()
    def safe_mul(a,b,name):
        if a in df.columns and b in df.columns:
            df[name]=df[a]*df[b]
    safe_mul("home_SP_FB_rate","INT_WindIn__ParkHR","INT_SPFBxParkHR_home")
    safe_mul("away_SP_FB_rate","INT_WindIn__ParkHR","INT_SPFBxParkHR_away")
    safe_mul("weather_temp_f","Team_Elo_diff__z_m","INT_TempxEloZ")
    safe_mul("Wind_In_flag","home_SP_FB_rate","INT_WindInxSPFB_home")
    safe_mul("Wind_In_flag","away_SP_FB_rate","INT_WindInxSPFB_away")
    safe_mul("away_BP_IP_d1","away_BP_xFIP_season","INT_BPloadxSkill_away")
    safe_mul("home_BP_IP_d1","home_BP_xFIP_season","INT_BPloadxSkill_home")
    return df

# ------------------- betting helpers -------------------
def american_to_decimal(ml):
    ml=float(ml);  return 1.0 + (ml/100.0 if ml>0 else 100.0/abs(ml))
def american_to_implied_prob(ml):
    ml=float(ml);  return 100.0/(ml+100.0) if ml>0 else abs(ml)/(abs(ml)+100.0)
def devig_two_way(p_home_raw, p_away_raw):
    s=p_home_raw+p_away_raw; 
    return (p_home_raw/s, p_away_raw/s) if s>0 else (np.nan,np.nan)

def simulate_bets(df, p_home_model, y_true, home_ml, away_ml, min_edge=0.0, kelly_cap=0.05):
    out=[]
    for i in range(len(df)):
        ml_h, ml_a = home_ml[i], away_ml[i]
        if np.isnan(ml_h) or np.isnan(ml_a): 
            out.append((0,0,0,0,0,0)); continue
        qh_raw, qa_raw = american_to_implied_prob(ml_h), american_to_implied_prob(ml_a)
        qh, qa = devig_two_way(qh_raw, qa_raw)
        if np.isnan(qh): out.append((0,0,0,0,0,0)); continue
        ph=float(np.clip(p_home_model[i],1e-9,1-1e-9))
        edge_home, edge_away = ph - qh, (1-ph) - qa

        flat=0.0; kelly=0.0; bet_dir=0; edge_used=0.0
        if edge_home >= min_edge or edge_away >= min_edge:
            if edge_home >= edge_away:
                b = american_to_decimal(ml_h)-1.0
                f = max(0.0, min(kelly_cap, (b*ph - (1-ph))/b))
                win = 1 if y_true[i]==1 else 0
                flat = (b if win else -1); kelly = f*(b if win else -1)
                bet_dir=1; edge_used=edge_home
            else:
                b = american_to_decimal(ml_a)-1.0
                pa = 1 - ph
                f = max(0.0, min(kelly_cap, (b*pa - (1-pa))/b))
                win = 1 if y_true[i]==0 else 0
                flat = (b if win else -1); kelly = f*(b if win else -1)
                bet_dir=-1; edge_used=edge_away
        out.append((flat,kelly,bet_dir,edge_used,qh,ph))
    return pd.DataFrame(out, columns=["flat_profit","kelly_profit","bet_dir","edge","implied_home","model_home"])


# ----------- calibration (fit once, apply many) ----------
def make_calibrator(p_ref: np.ndarray, y_ref: np.ndarray, method: str):
    """
    Fit on reference (e.g., validation) and return a callable f(x) -> calibrated probs.
    """
    if method == "raw":
        return lambda x: np.asarray(x)
    elif method == "platt":
        lr = LogisticRegression(max_iter=2000)
        lr.fit(np.asarray(p_ref).reshape(-1,1), y_ref)
        return lambda x: lr.predict_proba(np.asarray(x).reshape(-1,1))[:,1]
    else:  # isotonic
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(np.asarray(p_ref), y_ref)
        return lambda x: iso.predict(np.asarray(x))

# ------------- LightGBM diagnostics (NEW) --------------
def lgbm_diagnostics(model, Xtr_b, ytr, Xva_b, yva, feat_names, outdir, tag="lgbm"):
    """
    Writes a diagnostics report and importances to disk.
    Flags 'stuck' states: all-zero importances, tiny best_iteration, degenerate class balance.
    """
    path_txt = os.path.join(outdir, f"{tag}_diagnostics.txt")
    diag = {}

    # importances
    try:
        imps = np.asarray(model.feature_importances_)
        diag["all_zero_importances"] = bool(np.all(imps == 0))
        top = pd.DataFrame({"feature": feat_names, "importance": imps}).sort_values("importance", ascending=False)
        save_csv(top, outdir, f"{tag}_feature_importances")
    except Exception as e:
        diag["importance_error"] = str(e)

    # early stop / best iteration
    best_iter = getattr(model, "best_iteration_", None)
    diag["best_iteration"] = int(best_iter) if best_iter is not None else None
    diag["n_estimators"] = int(getattr(model, "n_estimators", -1))
    diag["early_stop_too_early"] = bool(best_iter is not None and best_iter < 80)

    # quick sanity AUC on valid (raw)
    try:
        pva_raw = model.predict_proba(Xva_b)[:,1]
        diag["valid_auc_raw"] = float(roc_auc_score(yva, pva_raw))
        diag["valid_logloss_raw"] = float(log_loss(yva, np.clip(pva_raw,1e-9,1-1e-9)))
    except Exception as e:
        diag["valid_predict_error"] = str(e)

    # class balance (train/valid)
    diag["train_pos_rate"] = float(np.mean(ytr))
    diag["valid_pos_rate"] = float(np.mean(yva))
    diag["imbalanced_valid"] = bool(np.mean(yva) < 0.05 or np.mean(yva) > 0.95)

    # write
    with open(path_txt, "w", encoding="utf-8") as f:
        f.write(json.dumps(diag, indent=2))
    print(f"[LGBM] diagnostics written -> {path_txt}")
    return diag

# ------------- Portable XGBoost early-stopping ----------
def fit_xgb_portable(model, X_tr, y_tr, X_va, y_va, rounds=200, metric="logloss", verbose=False):
    """
    Works across xgboost versions/wrappers:
      * uses early_stopping_rounds if supported
      * otherwise falls back to callbacks.EarlyStopping
      * otherwise fits without early stopping
    """
    kwargs = dict(eval_set=[(X_va, y_va)], verbose=verbose)
    sig = inspect.signature(model.fit).parameters

    if "early_stopping_rounds" in sig:
        return model.fit(X_tr, y_tr, early_stopping_rounds=rounds, **kwargs)

    if "callbacks" in sig:
        maximize = metric.lower() in {"auc", "aucpr"}
        cb = [xgb.callback.EarlyStopping(rounds, save_best=True, maximize=maximize)]
        try:
            if getattr(model, "eval_metric", None) is None:
                model.set_params(eval_metric=metric)
        except Exception:
            pass
        return model.fit(X_tr, y_tr, callbacks=cb, **kwargs)

    return model.fit(X_tr, y_tr, **kwargs)


# ------------------------- main -------------------------
def main(args):
    outdir=ensure_dir(args.outdir)
    df=pd.read_csv(args.data,parse_dates=[args.date_col]).sort_values(args.date_col).reset_index(drop=True)

    if args.add_rolling: df=add_rolling(df,args.date_col)
    if args.add_interactions: df=add_interactions(df)

    # splits
    (a,b),(b,c),(c,d)=time_split(df,0.75,0.15)
    splits=[
        f"Train: {b-a} rows (index {a}:{b})  target_mean={df.iloc[a:b][args.target].mean():.3f}",
        f"Valid: {c-b} rows (index {b}:{c})  target_mean={df.iloc[b:c][args.target].mean():.3f}",
        f"Test : {d-c} rows (index {c}:{d})  target_mean={df.iloc[c:d][args.target].mean():.3f}",
    ]
    with open(os.path.join(outdir,"splits.txt"),"w") as f: f.write("\n".join(splits))
    print("\n".join(splits))

    feats_all=pick_features(df,args.target,args.id_col,args.date_col)
    X=df[feats_all]; y=df[args.target].astype(int).values
    Xtr,ytr=X.iloc[a:b],y[a:b]; Xva,yva=X.iloc[b:c],y[b:c]; Xte,yte=X.iloc[c:d],y[c:d]

    # prune by permutation importance on valid using baseline logistic
    base = Pipeline([
        ("imp", pd_imputer("median")),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000))
    ])
    base.fit(Xtr,ytr)
    pi=permutation_importance(base,Xva,yva,n_repeats=8,random_state=42)
    rank=pd.DataFrame({"feature":feats_all,"perm_importance":pi.importances_mean}).sort_values("perm_importance",ascending=False)
    save_csv(rank,outdir,"perm_importance_valid")
    keep = [c for c in rank.head(args.topk)["feature"].tolist() if Xtr[c].notna().any()]
    Xtr,Xva,Xte = Xtr[keep].copy(), Xva[keep].copy(), Xte[keep].copy()
    with open(os.path.join(outdir,"kept_features.txt"),"w") as f: f.write("\n".join(keep))
    print(f"[prune] keeping {len(keep)} features.")

    # ---- shared imputer for boosters (so we can use early stopping cleanly) ----
    imp_boost = pd_imputer("median")
    imp_boost.fit(Xtr)
    Xtr_b = imp_boost.transform(Xtr).to_numpy()
    Xva_b = imp_boost.transform(Xva).to_numpy()
    Xte_b = imp_boost.transform(Xte).to_numpy()

    # models
    models={}
    # Linear baselines
    models["logreg"]=Pipeline([
        ("imp",pd_imputer("median")),
        ("sc",StandardScaler()),
        ("clf",LogisticRegression(max_iter=2000))
    ])
    models["elastic"]=Pipeline([
        ("imp",pd_imputer("median")),
        ("sc",StandardScaler()),
        ("clf",LogisticRegression(penalty="elasticnet",solver="saga",l1_ratio=0.5,max_iter=4000))
    ])
    # Bagging style
    models["rf"]=make_pipeline(pd_imputer("median"),
                               RandomForestClassifier(n_estimators=800,min_samples_leaf=3,
                                                      max_features="sqrt",random_state=42,n_jobs=-1))
    models["et"]=make_pipeline(pd_imputer("median"),
                               ExtraTreesClassifier(n_estimators=800,min_samples_leaf=3,
                                                    max_features="sqrt",random_state=42,n_jobs=-1))
    # Gradient Boosting (sklearn)
    models["gb"]=make_pipeline(pd_imputer("median"),
                               GradientBoostingClassifier(
                                   n_estimators=2000, learning_rate=0.01,
                                   max_depth=3, subsample=0.8, random_state=42))

    # LightGBM (hardened params + diagnostics)
    if HAS_LGB:
        models["lgbm"] = lgb.LGBMClassifier(
            n_estimators=3000,           # plenty, we rely on early stopping
            learning_rate=0.02,          # modest LR to help positive gains
            num_leaves=63,
            min_data_in_leaf=20,
            min_gain_to_split=0.0,       # allow tiny gains (prevents stall)
            subsample=0.8, subsample_freq=1,
            colsample_bytree=0.9,
            reg_lambda=1.0, reg_alpha=0.0,
            random_state=42, objective="binary",
            verbose=-1
        )

    # XGBoost
    if HAS_XGB:
        models["xgb"] = XGBClassifier(
            n_estimators=5000, learning_rate=0.01, max_depth=6,
            min_child_weight=2, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.5, reg_alpha=0.0, n_jobs=-1, tree_method="hist",
            objective="binary:logistic", eval_metric="logloss", random_state=42
        )

    # CatBoost
    if HAS_CAT:
        models["cat"]=CatBoostClassifier(
            depth=6, learning_rate=0.03, iterations=5000,
            l2_leaf_reg=3.0, loss_function="Logloss", eval_metric="AUC",
            od_type="Iter", od_wait=200, random_seed=42, verbose=False,
            allow_writing_files=False
        )

    # train + calibrate (fit on VALID once) + evaluate
    rows=[]; preds_valid={}; preds_test={}; thresholds={}
    for name, mdl in models.items():
        if name in {"lgbm","xgb","cat"}:
            if name=="lgbm":
                mdl.fit(Xtr_b, ytr,
                        eval_set=[(Xva_b, yva)],
                        eval_metric="auc",
                        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
                _ = lgbm_diagnostics(mdl, Xtr_b, ytr, Xva_b, yva, keep, outdir, tag="lgbm")
                pva_raw = mdl.predict_proba(Xva_b)[:,1]
                pte_raw = mdl.predict_proba(Xte_b)[:,1]
            elif name=="xgb":
                fit_xgb_portable(mdl, Xtr_b, ytr, Xva_b, yva, rounds=200, metric="logloss", verbose=False)
                pva_raw = mdl.predict_proba(Xva_b)[:,1]
                pte_raw = mdl.predict_proba(Xte_b)[:,1]
            else:  # cat
                mdl.fit(Xtr_b, ytr, eval_set=(Xva_b, yva), verbose=False)
                pva_raw=mdl.predict_proba(Xva_b)[:,1]; pte_raw=mdl.predict_proba(Xte_b)[:,1]
        else:
            mdl.fit(Xtr,ytr)
            pva_raw=mdl.predict_proba(Xva)[:,1]; pte_raw=mdl.predict_proba(Xte)[:,1]

        calibrate = make_calibrator(pva_raw, yva, args.calibration)
        pva = calibrate(pva_raw)
        pte = calibrate(pte_raw)

        preds_valid[name]=pva; preds_test[name]=pte

        thr,_=choose_threshold(yva,pva,args.threshold_strategy); thresholds[name]=thr
        m_v=eval_probs(yva,pva,thr); m_t=eval_probs(yte,pte,thr)
        rows.append([name,"valid",args.threshold_strategy,thr,m_v["auc"],m_v["acc"],m_v["logloss"],m_v["brier"]])
        rows.append([name,"test" ,args.threshold_strategy,thr,m_t["auc"],m_t["acc"],m_t["logloss"],m_t["brier"]])

        cm=confusion_matrix(yte,(pte>=thr).astype(int))
        confusion_plot(cm,f"Confusion — Test — {name} (thr={thr:.2f})",fig_path(outdir,f"cm__{name}"))

    leaderboard=pd.DataFrame(rows,columns=["model","split","thr_strategy","threshold","auc","acc","logloss","brier"])
    save_csv(leaderboard,outdir,"leaderboard")

    # plots comparing all
    plot_roc_pr(yva,preds_valid,outdir,"valid_all")
    plot_roc_pr(yte,preds_test,outdir,"test_all")
    plot_calibration(yte,preds_test,outdir,"test_all")

    # soft voting of top3 (valid AUC)
    top3=leaderboard[leaderboard.split=="valid"].sort_values("auc",ascending=False).head(3)["model"].tolist()
    pva_ens=pte_ens=None; thr_ens=None
    if len(top3)>=2:
        pva_ens=np.mean([preds_valid[m] for m in top3],axis=0)
        pte_ens=np.mean([preds_test[m] for m in top3],axis=0)
        thr_ens,_=choose_threshold(yva,pva_ens,args.threshold_strategy)
        mv=eval_probs(yva,pva_ens,thr_ens); mt=eval_probs(yte,pte_ens,thr_ens)
        leaderboard=pd.concat([leaderboard,
                               pd.DataFrame([["vote3","valid",args.threshold_strategy,thr_ens,mv["auc"],mv["acc"],mv["logloss"],mv["brier"]],
                                             ["vote3","test" ,args.threshold_strategy,thr_ens,mt["auc"],mt["acc"],mt["logloss"],mt["brier"]]], 
                                            columns=leaderboard.columns)],ignore_index=True)
        save_csv(leaderboard,outdir,"leaderboard")
        cm=confusion_matrix(yte,(pte_ens>=thr_ens).astype(int))
        confusion_plot(cm,f"Confusion — Test — vote3 (thr={thr_ens:.2f})",fig_path(outdir,"cm__vote3"))
        plot_roc_pr(yte,{"vote3":pte_ens},outdir,"test_vote3")
        plot_calibration(yte,{"vote3":pte_ens},outdir,"test_vote3")

    # ---------------- monthly walk-forward ----------------
    per=df[args.date_col].dt.to_period("M"); months=sorted(per.unique())
    wf_rows=[]
    for k in range(1,len(months)):
        tr=(per<months[k]); te=(per==months[k])
        if tr.sum()<200 or te.sum()<20: continue
        Xtr_m, ytr_m = df.loc[tr,keep], df.loc[tr,args.target].astype(int).values
        Xte_m, yte_m = df.loc[te,keep], df.loc[te,args.target].astype(int).values

        imp_m = pd_imputer("median"); imp_m.fit(Xtr_m)
        Xtr_m_b = imp_m.transform(Xtr_m).to_numpy()
        Xte_m_b = imp_m.transform(Xte_m).to_numpy()

        month=str(months[k])
        for name, mdl in models.items():
            if name=="lgbm":
                mdl.fit(Xtr_m_b,ytr_m,
                        eval_set=[(Xte_m_b,yte_m)],
                        eval_metric="auc",
                        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
                _ = lgbm_diagnostics(mdl, Xtr_m_b, ytr_m, Xte_m_b, yte_m, keep, outdir, tag=f"lgbm_{month}")
                pte_raw = mdl.predict_proba(Xte_m_b)[:,1]
                split=int(len(Xtr_m_b)*0.8)
                pcal_raw = mdl.predict_proba(Xtr_m_b[split:])[:,1] if split>=50 else mdl.predict_proba(Xtr_m_b)[:,1]
                yref = ytr_m[split:] if split>=50 else ytr_m
                pte = make_calibrator(pcal_raw, yref, args.calibration)(pte_raw)
            elif name=="xgb":
                fit_xgb_portable(mdl, Xtr_m_b, ytr_m, Xte_m_b, yte_m, rounds=200, metric="logloss", verbose=False)
                split=int(len(Xtr_m_b)*0.8)
                pcal_raw = mdl.predict_proba(Xtr_m_b[split:])[:,1] if split>=50 else mdl.predict_proba(Xtr_m_b)[:,1]
                yref = ytr_m[split:] if split>=50 else ytr_m
                pte = make_calibrator(pcal_raw, yref, args.calibration)(mdl.predict_proba(Xte_m_b)[:,1])
            elif name=="cat":
                mdl.fit(Xtr_m_b,ytr_m,eval_set=(Xte_m_b,yte_m),verbose=False)
                split=int(len(Xtr_m_b)*0.8)
                pcal_raw = mdl.predict_proba(Xtr_m_b[split:])[:,1] if split>=50 else mdl.predict_proba(Xtr_m_b)[:,1]
                yref = ytr_m[split:] if split>=50 else ytr_m
                pte = make_calibrator(pcal_raw, yref, args.calibration)(mdl.predict_proba(Xte_m_b)[:,1])
            else:
                mdl.fit(Xtr_m,ytr_m)
                split=int(len(Xtr_m)*0.8)
                pcal_raw = mdl.predict_proba(Xtr_m.iloc[split:])[:,1] if len(Xtr_m)>=50 else mdl.predict_proba(Xtr_m)[:,1]
                yref = ytr_m[split:] if len(Xtr_m)>=50 else ytr_m
                pte = make_calibrator(pcal_raw, yref, args.calibration)(mdl.predict_proba(Xte_m)[:,1])

            wf_rows.append([month,name,float(roc_auc_score(yte_m,pte)),int(te.sum()),args.calibration])

        if 'pva_ens' in locals() and pva_ens is not None:
            preds=[]
            for mname in top3:
                mdl=models[mname]
                if mname=="lgbm":
                    mdl.fit(Xtr_m_b,ytr_m,eval_set=[(Xte_m_b,yte_m)],
                            eval_metric="auc",
                            callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
                    preds.append(mdl.predict_proba(Xte_m_b)[:,1])
                elif mname=="xgb":
                    fit_xgb_portable(mdl, Xtr_m_b, ytr_m, Xte_m_b, yte_m, rounds=200, metric="logloss", verbose=False)
                    preds.append(mdl.predict_proba(Xte_m_b)[:,1])
                elif mname=="cat":
                    mdl.fit(Xtr_m_b,ytr_m,eval_set=(Xte_m_b,yte_m),verbose=False)
                    preds.append(mdl.predict_proba(Xte_m_b)[:,1])
                else:
                    mdl.fit(Xtr_m,ytr_m); preds.append(mdl.predict_proba(Xte_m)[:,1])
            pte_avg=np.mean(preds,axis=0)
            wf_rows.append([month,"vote3",float(roc_auc_score(yte_m,pte_avg)),int(te.sum()),args.calibration])

    if wf_rows:
        wf=pd.DataFrame(wf_rows,columns=["month","model","auc","rows","calibration"])
        save_csv(wf,outdir,"monthly_walkforward")
        for model_name, g in wf.groupby("model"):
            plt.figure(); plt.plot(g["month"],g["auc"],marker="o"); plt.xticks(rotation=45)
            plt.ylabel("AUC"); plt.title(f"Monthly Walk-Forward AUC — {model_name} ({args.calibration})")
            plt.tight_layout(); plt.savefig(fig_path(outdir,f"monthly_wf__{model_name}")); plt.close()

    # ---------------- Expanding TimeSeriesSplit walk-forward (optional) ----------------
    if args.walkforward == "ts":
        tsv_rows=[]
        dates=df[args.date_col]
        splitter=TimeSeriesSplit(n_splits=args.n_splits)
        idx_df = pd.DataFrame({"date": dates}).reset_index(drop=True)
        for fold,(tr_idx, te_idx) in enumerate(splitter.split(idx_df)):
            Xtr_w, ytr_w = df.loc[tr_idx, keep], df.loc[tr_idx, args.target].astype(int).values
            Xte_w, yte_w = df.loc[te_idx, keep], df.loc[te_idx, args.target].astype(int).values
            start, end = dates.iloc[te_idx].min(), dates.iloc[te_idx].max()

            imp_w = pd_imputer("median"); imp_w.fit(Xtr_w)
            Xtr_w_b = imp_w.transform(Xtr_w).to_numpy()
            Xte_w_b = imp_w.transform(Xte_w).to_numpy()

            for name, mdl in models.items():
                if name=="lgbm":
                    mdl.fit(Xtr_w_b,ytr_w,eval_set=[(Xte_w_b,yte_w)],
                            eval_metric="auc",
                            callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
                    p_raw = mdl.predict_proba(Xte_w_b)[:,1]
                    pcal_raw = mdl.predict_proba(Xtr_w_b[int(len(Xtr_w_b)*0.8):])[:,1] if len(Xtr_w_b)>=50 else mdl.predict_proba(Xtr_w_b)[:,1]
                elif name=="xgb":
                    fit_xgb_portable(mdl, Xtr_w_b, ytr_w, Xte_w_b, yte_w, rounds=200, metric="logloss", verbose=False)
                    p_raw = mdl.predict_proba(Xte_w_b)[:,1]
                    pcal_raw = mdl.predict_proba(Xtr_w_b[int(len(Xtr_w_b)*0.8):])[:,1] if len(Xtr_w_b)>=50 else mdl.predict_proba(Xtr_w_b)[:,1]
                elif name=="cat":
                    mdl.fit(Xtr_w_b,ytr_w,eval_set=(Xte_w_b,yte_w),verbose=False)
                    p_raw = mdl.predict_proba(Xte_w_b)[:,1]
                    pcal_raw = mdl.predict_proba(Xtr_w_b[int(len(Xtr_w_b)*0.8):])[:,1] if len(Xtr_w_b)>=50 else mdl.predict_proba(Xtr_w_b)[:,1]
                else:
                    mdl.fit(Xtr_w,ytr_w)
                    p_raw = mdl.predict_proba(Xte_w)[:,1]
                    pcal_raw = mdl.predict_proba(Xtr_w.iloc[int(len(Xtr_w)*0.8):])[:,1] if len(Xtr_w)>=50 else mdl.predict_proba(Xtr_w)[:,1]

                yref = ytr_w[int(len(Xtr_w)*0.8):] if len(Xtr_w)>=50 else ytr_w
                calibrate_w = make_calibrator(pcal_raw, yref, args.calibration)
                p = calibrate_w(p_raw)

                tsv_rows.append([fold+1, str(start.date()), str(end.date()), name,
                                 float(roc_auc_score(yte_w,p)), float(log_loss(yte_w,np.clip(p,1e-9,1-1e-9))),
                                 float(brier_score_loss(yte_w,p)), int(len(te_idx))])

        if tsv_rows:
            wf2=pd.DataFrame(tsv_rows,columns=["fold","start","end","model","auc","logloss","brier","rows"])
            save_csv(wf2,outdir,"ts_walkforward")
            for model_name, g in wf2.groupby("model"):
                plt.figure(); plt.plot(g["fold"],g["auc"],marker="o"); plt.xticks(g["fold"])
                plt.ylabel("AUC"); plt.xlabel("Fold"); plt.title(f"TS Walk-Forward AUC — {model_name} ({args.calibration})")
                plt.tight_layout(); plt.savefig(fig_path(outdir,f"ts_wf__{model_name}")); plt.close()

    # ---------------- betting on TEST (optional) ----------------
    if args.home_odds_col and args.away_odds_col and args.home_odds_col in df.columns and args.away_odds_col in df.columns:
        test_df=df.iloc[c:d].copy()
        if 'vote3' in leaderboard['model'].unique():
            name_for_bets = "vote3"; ptest = preds_test.get("vote3")
            if ptest is None:
                top3=leaderboard[leaderboard.split=="valid"].sort_values("auc",ascending=False).head(3)["model"].tolist()
                ptest=np.mean([preds_test[m] for m in top3],axis=0)
        else:
            best = leaderboard[leaderboard.split=="valid"].sort_values("auc",ascending=False).iloc[0]["model"]
            name_for_bets = best; ptest = preds_test[best]

        sim = simulate_bets(
            test_df, ptest, yte,
            test_df[args.home_odds_col].astype(float).values,
            test_df[args.away_odds_col].astype(float).values,
            min_edge=args.min_edge, kelly_cap=args.kelly_cap
        )
        sim["flat_cum"]=sim["flat_profit"].cumsum(); sim["kelly_cum"]=sim["kelly_profit"].cumsum()
        save_csv(pd.concat([test_df[[args.date_col]],sim],axis=1),outdir,f"betting_sim__{name_for_bets}")

        plt.figure(); plt.plot(sim["flat_cum"],label="Flat (1u)"); plt.plot(sim["kelly_cum"],label=f"Kelly (cap {args.kelly_cap:.2f})")
        plt.title(f"Betting Cumulative Profit — Test ({name_for_bets}, min_edge={args.min_edge})"); plt.xlabel("Bets"); plt.ylabel("Units")
        plt.legend(); plt.tight_layout(); plt.savefig(fig_path(outdir,f"betting_cumulative__{name_for_bets}")); plt.close()

        edges=np.linspace(0.0,0.08,17); pts=[]
        for e in edges:
            s2=simulate_bets(
                test_df, ptest, yte,
                test_df[args.home_odds_col].astype(float).values,
                test_df[args.away_odds_col].astype(float).values,
                min_edge=float(e), kelly_cap=args.kelly_cap
            )
            pts.append([e,s2["flat_profit"].sum(),s2["kelly_profit"].sum(),int((s2["flat_profit"]!=0).sum())])
        pr_curve=pd.DataFrame(pts,columns=["min_edge","flat_units","kelly_units","num_bets"])
        save_csv(pr_curve,outdir,f"betting_edge_sweep__{name_for_bets}")
        plt.figure(); plt.plot(pr_curve["min_edge"],pr_curve["flat_units"],marker="o",label="Flat units")
        plt.plot(pr_curve["min_edge"],pr_curve["kelly_units"],marker="o",label="Kelly units")
        plt.xlabel("Min Edge"); plt.ylabel("Units won (Test)"); plt.title(f"Profit vs Edge — {name_for_bets}")
        plt.legend(); plt.tight_layout(); plt.savefig(fig_path(outdir,f"betting_profit_vs_edge__{name_for_bets}")); plt.close()

    # ---------------- HTML report ----------------
    imgs=[f for f in os.listdir(outdir) if f.endswith(".png")]
    html=f"""
    <html><head><meta charset="utf-8"/>
    <title>MLB All-in-One Model Report</title>
    <style>
      body{{font-family:Arial;margin:24px}} img{{max-width:100%;height:auto;margin:6px 0}}
      table{{border-collapse:collapse}} th,td{{border:1px solid #ddd;padding:4px 8px}}
      pre{{background:#f5f5f5;padding:8px}}
    </style></head><body>
      <h1>MLB All-in-One Model Report</h1>
      <p><b>Generated:</b> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
      <h2>Args</h2><pre>calibration={args.calibration} | walkforward={args.walkforward} | n_splits={args.n_splits}</pre>
      <h2>Splits</h2><pre>{os.linesep.join(splits)}</pre>
      <h2>Kept Features (top {args.topk})</h2><pre>{"\\n".join(keep)}</pre>
      <h2>Leaderboard</h2>{pd.read_csv(os.path.join(outdir,"leaderboard.csv")).to_html(index=False)}
      {"<h2>Monthly Walk-Forward</h2>"+pd.read_csv(os.path.join(outdir,'monthly_walkforward.csv')).to_html(index=False) if os.path.exists(os.path.join(outdir,'monthly_walkforward.csv')) else ""}
      {"<h2>TS Walk-Forward</h2>"+pd.read_csv(os.path.join(outdir,'ts_walkforward.csv')).to_html(index=False) if os.path.exists(os.path.join(outdir,'ts_walkforward.csv')) else ""}
      <h2>Figures</h2>
      {''.join(f'<img src="{f}"/>' for f in sorted(imgs))}
    </body></html>
    """
    with open(os.path.join(outdir,"report_allinone.html"),"w",encoding="utf-8") as f: f.write(html)
    print("\n✅ All-in-one pipeline complete.")
    print("Artifacts:", outdir)
    print("Open report:", os.path.join(outdir,"report_allinone.html"))


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--date-col", default="game_date")
    ap.add_argument("--id-col", default="game_pk")
    ap.add_argument("--target", default="home_win")
    ap.add_argument("--outdir", default="model_all")
    ap.add_argument("--topk", type=int, default=25)
    ap.add_argument("--threshold-strategy", choices=["f1","youden","balanced"], default="f1")
    ap.add_argument("--add-rolling", action="store_true")
    ap.add_argument("--add-interactions", action="store_true")

    # calibration + walk-forward
    ap.add_argument("--calibration", choices=["raw","platt","isotonic"], default="isotonic")
    ap.add_argument("--walkforward", choices=["monthly","ts","none"], default="monthly")
    ap.add_argument("--n-splits", type=int, default=6)

    # betting args (optional)
    ap.add_argument("--home-odds-col", type=str, default=None)
    ap.add_argument("--away-odds-col", type=str, default=None)
    ap.add_argument("--min-edge", type=float, default=0.02)
    ap.add_argument("--kelly-cap", type=float, default=0.05)

    args=ap.parse_args(); main(args)
