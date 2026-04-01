#!/usr/bin/env python
"""
predict_api.py — FastAPI server for MLB game win-probability predictions.

Endpoints
---------
GET  /health                      → liveness check + model metadata
GET  /predictions?date=YYYY-MM-DD → predictions for all scheduled games on that date
GET  /predictions/today           → shortcut for today's date (UTC)
POST /predict                     → single game prediction from raw feature dict

Setup
-----
    pip install fastapi uvicorn joblib pandas numpy requests scikit-learn lightgbm pybaseball

Run locally
-----------
    uvicorn predict_api:app --host 0.0.0.0 --port 8000 --reload

Deploy (systemd example)
------------------------
    [Unit]
    Description=MLB Prediction API
    After=network.target

    [Service]
    WorkingDirectory=/opt/mlb-pipeline
    ExecStart=/opt/mlb-pipeline/.venv/bin/uvicorn predict_api:app --host 0.0.0.0 --port 8000
    Restart=always
    RestartSec=5

    [Install]
    WantedBy=multi-user.target

Environment variables
---------------------
MODEL_PATH   Path to best_model.pkl  (default: model_all_ts/best_model.pkl)
DATA_PATH    Path to mlb_features_prepared.csv for live feature lookup
             (default: out/mlb_features_prepared.csv)
"""

from __future__ import annotations

import os
import logging
from datetime import datetime, timezone, date
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("mlb_api")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "model_all_ts/best_model.pkl")
DATA_PATH  = os.environ.get("DATA_PATH",  "out/mlb_features_prepared.csv")

MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="MLB Prediction API",
    description="Win-probability predictions powered by a LightGBM walk-forward model.",
    version="1.0.0",
)

# Allow the portfolio dashboard (and localhost dev) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten to your domain in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model loading (singleton on startup)
# ---------------------------------------------------------------------------
_artifact: Dict[str, Any] = {}

@app.on_event("startup")
def load_model():
    global _artifact
    if not os.path.exists(MODEL_PATH):
        log.warning(f"Model file not found at {MODEL_PATH}. /predictions will return 503.")
        return
    log.info(f"Loading model from {MODEL_PATH} ...")
    _artifact = joblib.load(MODEL_PATH)
    log.info(
        f"Model loaded: {_artifact.get('model_name')}  "
        f"AUC={_artifact.get('auc_test', '?'):.4f}  "
        f"features={len(_artifact.get('feature_cols', []))}"
    )

def _require_model():
    if not _artifact:
        raise HTTPException(status_code=503, detail="Model not loaded. Run model_train_allinone.py first.")

# ---------------------------------------------------------------------------
# Feature lookup helpers
# ---------------------------------------------------------------------------

def _load_feature_store() -> pd.DataFrame:
    """Load the prepared feature CSV as a feature store keyed by game_pk."""
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame()
    return pd.read_csv(DATA_PATH, parse_dates=["game_date"])


def _features_for_game(game_pk: int, feat_store: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Return a single-row DataFrame of features for game_pk, or None if not found.
    The row will have exactly the columns the model expects (feature_cols), with
    any missing columns filled with NaN so the preprocessor's imputer handles them.
    """
    if feat_store.empty or "game_pk" not in feat_store.columns:
        return None

    row = feat_store[feat_store["game_pk"] == game_pk]
    if row.empty:
        return None

    feat_cols: List[str] = _artifact["feature_cols"]
    # Reindex to model's expected feature set; missing cols → NaN (handled by imputer)
    return row[feat_cols] if all(c in row.columns for c in feat_cols) \
           else row.reindex(columns=feat_cols)


def _predict_row(row_df: pd.DataFrame) -> Dict[str, float]:
    """Run a single-row DataFrame through the saved pipeline."""
    pipeline = _artifact["pipeline"]
    threshold = _artifact.get("threshold", 0.5)

    proba = pipeline.predict_proba(row_df)[0]
    home_prob = float(proba[1])
    away_prob = float(proba[0])
    edge = abs(home_prob - 0.5)
    predicted_winner = "home" if home_prob >= threshold else "away"

    return {
        "home_win_prob":   round(home_prob, 4),
        "away_win_prob":   round(away_prob, 4),
        "predicted_winner": predicted_winner,
        "edge":            round(edge, 4),
        "threshold":       round(threshold, 4),
        "high_confidence": edge > 0.07,
    }

# ---------------------------------------------------------------------------
# MLB Stats API helpers
# ---------------------------------------------------------------------------

def _fetch_schedule(date_str: str) -> List[Dict]:
    """Fetch scheduled games from the MLB Stats API for a given date."""
    params = {
        "sportId": 1,
        "date": date_str,
        "hydrate": "probablePitcher,team,venue",
    }
    try:
        resp = requests.get(MLB_SCHEDULE_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("dates", [{}])[0].get("games", []) if data.get("dates") else []
    except Exception as e:
        log.error(f"MLB schedule fetch failed for {date_str}: {e}")
        return []


def _game_to_response(game: Dict, prediction: Optional[Dict], source: str) -> Dict:
    """Reshape an MLB Stats API game object into our response schema."""
    home = game.get("teams", {}).get("home", {})
    away = game.get("teams", {}).get("away", {})
    home_team = home.get("team", {})
    away_team = away.get("team", {})

    game_time_utc = game.get("gameDate", "")
    try:
        game_time_local = datetime.fromisoformat(game_time_utc.replace("Z", "+00:00")) \
                          .astimezone().strftime("%I:%M %p %Z")
    except Exception:
        game_time_local = game_time_utc

    return {
        "game_pk":   game.get("gamePk"),
        "game_time": game_time_local,
        "status":    game.get("status", {}).get("abstractGameState", "Preview"),
        "venue":     game.get("venue", {}).get("name", ""),
        "home": {
            "team_id":   home_team.get("id"),
            "team_name": home_team.get("name", ""),
            "abbrev":    home_team.get("abbreviation", ""),
            "sp_name":   home.get("probablePitcher", {}).get("fullName", "TBD"),
            "sp_id":     home.get("probablePitcher", {}).get("id"),
        },
        "away": {
            "team_id":   away_team.get("id"),
            "team_name": away_team.get("name", ""),
            "abbrev":    away_team.get("abbreviation", ""),
            "sp_name":   away.get("probablePitcher", {}).get("fullName", "TBD"),
            "sp_id":     away.get("probablePitcher", {}).get("id"),
        },
        "prediction":    prediction,
        "prediction_source": source,   # "model" | "unavailable"
    }

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class SingleGameRequest(BaseModel):
    """POST /predict — supply feature values directly as a flat dict."""
    features: Dict[str, float]
    game_meta: Optional[Dict[str, Any]] = None   # optional context (team names etc.)


class PredictionResponse(BaseModel):
    date: str
    games_scheduled: int
    games_predicted: int
    model_name: str
    model_auc: float
    predictions: List[Dict]

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Liveness + model metadata."""
    if not _artifact:
        return {"status": "degraded", "reason": "model not loaded", "model_path": MODEL_PATH}
    return {
        "status":       "ok",
        "model_name":   _artifact.get("model_name"),
        "model_auc":    round(_artifact.get("auc_test", 0), 4),
        "threshold":    round(_artifact.get("threshold", 0.5), 4),
        "feature_count": len(_artifact.get("feature_cols", [])),
        "trained_on_rows": _artifact.get("trained_on_rows"),
        "date_range":   _artifact.get("date_range"),
        "calibration":  _artifact.get("calibration"),
    }


@app.get("/predictions/today", response_model=PredictionResponse)
def predictions_today():
    """Predictions for today (UTC)."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return _predictions_for_date(today)


@app.get("/predictions", response_model=PredictionResponse)
def predictions_by_date(
    date: str = Query(..., description="Date in YYYY-MM-DD format", example="2025-07-04")
):
    """Predictions for a specific date."""
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=422, detail="date must be YYYY-MM-DD")
    return _predictions_for_date(date)


def _predictions_for_date(date_str: str) -> Dict:
    _require_model()

    games = _fetch_schedule(date_str)
    if not games:
        return {
            "date": date_str,
            "games_scheduled": 0,
            "games_predicted": 0,
            "model_name": _artifact.get("model_name", ""),
            "model_auc": round(_artifact.get("auc_test", 0), 4),
            "predictions": [],
        }

    feat_store = _load_feature_store()
    results = []
    predicted_count = 0

    for game in games:
        game_pk = game.get("gamePk")
        row_df = _features_for_game(game_pk, feat_store) if game_pk else None

        if row_df is not None and not row_df.empty:
            try:
                pred = _predict_row(row_df)
                source = "model"
                predicted_count += 1
            except Exception as e:
                log.warning(f"Prediction failed for game_pk={game_pk}: {e}")
                pred = None
                source = "unavailable"
        else:
            # Game not yet in feature store (future game / ingest not run yet)
            pred = None
            source = "unavailable"

        results.append(_game_to_response(game, pred, source))

    return {
        "date":            date_str,
        "games_scheduled": len(games),
        "games_predicted": predicted_count,
        "model_name":      _artifact.get("model_name", ""),
        "model_auc":       round(_artifact.get("auc_test", 0), 4),
        "predictions":     results,
    }


@app.post("/predict")
def predict_single(body: SingleGameRequest):
    """
    Predict from a raw feature dict.  Useful for testing or custom integrations.

    Example body:
    {
        "features": {
            "Team_WinPct_diff": 0.12,
            "SP_xFIP_diff": -0.4,
            ...
        },
        "game_meta": {"home_team": "LAD", "away_team": "SF"}
    }
    """
    _require_model()
    feat_cols: List[str] = _artifact["feature_cols"]

    # Build a DataFrame with model's expected columns; fill unknowns with NaN
    row = {c: body.features.get(c, np.nan) for c in feat_cols}
    row_df = pd.DataFrame([row])

    try:
        pred = _predict_row(row_df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    return {
        "prediction": pred,
        "game_meta": body.game_meta,
        "model_name": _artifact.get("model_name"),
        "model_auc":  round(_artifact.get("auc_test", 0), 4),
        "features_supplied": len([v for v in body.features.values() if not np.isnan(v)]),
        "features_expected": len(feat_cols),
    }
