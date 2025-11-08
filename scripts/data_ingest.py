# =============================================================================
# MLB Features Pipeline (fixed end-to-end)
# - Scores captured from schedule
# - Team rolling features computed & merged
# - One final DataFrame with all columns for your report
# =============================================================================
from __future__ import annotations

import time
from functools import lru_cache
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse, os
import numpy as np
import pandas as pd
import statsapi
from pybaseball import (
    statcast, statcast_pitcher, cache as _pb_cache
)

_pb_cache.enable()

# ---------------------------
# Helpers
# ---------------------------
def _retry(fn, *args, tries=3, delay=0.5, **kwargs):
    for i in range(tries):
        try:
            return fn(*args, **kwargs)
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(delay * (2 ** i))

_FINISHED_STATUSES = {'Final', 'Game Over', 'Completed', 'Completed Early', 'Postponed'}

def build_team_abbrev_map() -> dict[int, str]:
    data = statsapi.get('teams', {'sportId': 1, 'activeStatus': 'Yes'})
    m = {}
    for t in data.get('teams', []):
        abbr = (t.get('abbreviation') or t.get('teamCode') or t.get('fileCode'))
        if abbr:
            m[int(t['id'])] = abbr.upper()
    return m

_TEAM_ABBR_MAP = build_team_abbrev_map()

def _team_abbrev_map_for_ids(team_ids: list[int]) -> dict[int, str]:
    tids = [int(x) for x in pd.Series(team_ids).dropna().unique().tolist()]
    if not tids:
        return {}
    try:
        data = _retry(statsapi.get, 'teams', {'teamIds': ','.join(map(str, tids))})
        teams = data.get('teams', []) or []
        m = {}
        for t in teams:
            abbr = (
                t.get('abbreviation')
                or t.get('teamCode')
                or t.get('fileCode')
                or t.get('clubName')
            )
            if abbr:
                m[int(t['id'])] = str(abbr).upper()
        return m
    except Exception:
        return {}

def _normalize_abbr_col(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.upper()
    s = s.mask(s.isin(['', 'NONE', 'NAN', 'NULL']))
    return s

# ---------------------------
# Probables / Lineups
# ---------------------------
def _probables_one_game(game_pk: int):
    try:
        g = _retry(statsapi.get, 'game', {'gamePk': int(game_pk)})
        pp = (g.get('gameData') or {}).get('probablePitchers') or {}
        h = pp.get('home') or {}
        a = pp.get('away') or {}
        return h.get('id'), h.get('fullName'), a.get('id'), a.get('fullName')
    except Exception:
        return None, None, None, None

def fetch_lineup_ids(game_pk: int) -> dict:
    try:
        g = statsapi.get('game', {'gamePk': int(game_pk)})
        box = g.get('liveData', {}).get('boxscore', {}).get('teams', {})
        out = {}
        for side in ('home', 'away'):
            lineup = []
            players = (box.get(side, {}) or {}).get('players', {}) or {}
            for p in players.values():
                bo = p.get('battingOrder')
                pos = (p.get('position') or {}).get('code')  # '1' = P
                pid = (p.get('person') or {}).get('id')
                if bo and pid and pos != '1':
                    try:
                        lineup.append((int(bo), int(pid)))
                    except Exception:
                        pass
            lineup = [pid for bo, pid in sorted(lineup)]
            out[side] = lineup
        return out
    except Exception:
        return {'home': [], 'away': []}

@lru_cache(maxsize=256)
def statcast_window_agg(end_date_str: str, window_days: int = 60) -> pd.DataFrame:
    end = pd.to_datetime(end_date_str).date()
    start = (end - timedelta(days=window_days - 1)).strftime("%Y-%m-%d")
    stop = (end - timedelta(days=1)).strftime("%Y-%m-%d")
    sc = statcast(start_dt=start, end_dt=stop)
    if sc is None or sc.empty:
        return pd.DataFrame(columns=['batter', 'p_throws', 'woba_value', 'woba_denom'])
    agg = (sc.groupby(['batter', 'p_throws'], as_index=False)[['woba_value', 'woba_denom']]
           .sum())
    return agg

def lineup_woba_from_agg(lineup_ids, vs_hand, agg_df) -> float | np.nan:
    if not lineup_ids or agg_df.empty or vs_hand not in ('L', 'R'):
        return np.nan
    sub = agg_df[(agg_df['batter'].isin(lineup_ids)) & (agg_df['p_throws'] == vs_hand)]
    num = float(sub['woba_value'].sum()); den = float(sub['woba_denom'].sum())
    return (num / den) if den > 0 else np.nan

def add_lineup_woba_features(games: pd.DataFrame, window_days: int = 60, max_workers: int = 8) -> pd.DataFrame:
    df = games.copy()
    df['game_date'] = pd.to_datetime(df['game_date'])
    df = df.sort_values('game_date')

    lineup_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(fetch_lineup_ids, int(pk)): int(pk) for pk in df['game_pk'].unique()}
        for f in as_completed(futs):
            lineup_map[futs[f]] = f.result()

    home_vals, away_vals = [], []
    for dt, idx in df.groupby(df['game_date'].dt.date).groups.items():
        agg = statcast_window_agg(str(dt), window_days)
        for i in idx:
            pk = int(df.at[i, 'game_pk'])
            lu = lineup_map.get(pk, {'home': [], 'away': []})

            # Use OPPOSING starter's hand
            home_vs_hand = df.at[i, 'away_sp_throws']
            away_vs_hand = df.at[i, 'home_sp_throws']

            home_vals.append(lineup_woba_from_agg(lu['home'], home_vs_hand, agg))
            away_vals.append(lineup_woba_from_agg(lu['away'], away_vs_hand, agg))

    df['home_lineup_woba_vsHand'] = pd.Series(home_vals, index=df.index)
    df['away_lineup_woba_vsHand'] = pd.Series(away_vals, index=df.index)
    df['Lineup_woba_vsHand_diff'] = df['home_lineup_woba_vsHand'] - df['away_lineup_woba_vsHand']
    return df




def add_lineup_woba_season_features(
    games: pd.DataFrame,
    season_start: str | None = None,   # defaults to March 1 of the earliest season in `games`
    max_workers: int = 8
) -> pd.DataFrame:
    """
    Adds season-to-date (pre-game) lineup wOBA vs opposing starter hand:
      - home_lineup_woba_vsHand_season
      - away_lineup_woba_vsHand_season
      - Lineup_woba_vsHand_season_diff  (home - away)

    Requirements in `games`: ['game_pk','game_date','home_sp_throws','away_sp_throws'].
    Uses the same lineup detection as add_lineup_woba_features().
    """
    if games.empty:
        return games

    df = games.copy()
    df['game_date'] = pd.to_datetime(df['game_date'])
    df = df.sort_values('game_date').reset_index(drop=True)

    # ----- Build or reuse lineups per game
    lineup_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(fetch_lineup_ids, int(pk)): int(pk) for pk in df['game_pk'].dropna().astype(int).unique()}
        for f in as_completed(futs):
            lineup_map[futs[f]] = f.result()

    # ----- Season window
    min_year = int(df['game_date'].dt.year.min())
    max_dt   = df['game_date'].max().date()
    season_start = season_start or f"{min_year}-03-01"
    end_for_pull = (max_dt - timedelta(days=1)).strftime("%Y-%m-%d")
    if pd.to_datetime(season_start) > pd.to_datetime(end_for_pull):
        # nothing to compute yet
        df['home_lineup_woba_vsHand_season'] = np.nan
        df['away_lineup_woba_vsHand_season'] = np.nan
        df['Lineup_woba_vsHand_season_diff'] = np.nan
        return df

    # ----- One Statcast pull for the season-to-date window
    sc = statcast(start_dt=season_start, end_dt=end_for_pull)
    if sc is None or sc.empty:
        df['home_lineup_woba_vsHand_season'] = np.nan
        df['away_lineup_woba_vsHand_season'] = np.nan
        df['Lineup_woba_vsHand_season_diff'] = np.nan
        return df

    # We only need these columns
    keep_cols = ['batter','p_throws','woba_value','woba_denom','game_date']
    sc = sc[[c for c in keep_cols if c in sc.columns]].copy()
    if sc.empty:
        df['home_lineup_woba_vsHand_season'] = np.nan
        df['away_lineup_woba_vsHand_season'] = np.nan
        df['Lineup_woba_vsHand_season_diff'] = np.nan
        return df

    # Aggregate by batter, pitcher hand, and day → then build pre-game cumulative sums
    sc['game_date'] = pd.to_datetime(sc['game_date']).dt.date
    daily = (sc.groupby(['batter','p_throws','game_date'], as_index=False)[['woba_value','woba_denom']].sum()
               .sort_values(['batter','p_throws','game_date']))
    daily['cum_woba_value'] = daily.groupby(['batter','p_throws'])['woba_value'].cumsum()
    daily['cum_woba_denom'] = daily.groupby(['batter','p_throws'])['woba_denom'].cumsum()
    # pre-game shift: use totals up to the previous day
    daily['pre_woba_value'] = daily.groupby(['batter','p_throws'])['cum_woba_value'].shift(1)
    daily['pre_woba_denom'] = daily.groupby(['batter','p_throws'])['cum_woba_denom'].shift(1)

    # Lookup maps keyed by (batter, hand, date)
    idx = pd.MultiIndex.from_frame(daily[['batter','p_throws','game_date']])
    val_map = pd.Series(daily['pre_woba_value'].values, index=idx)
    den_map = pd.Series(daily['pre_woba_denom'].values, index=idx)

    def _lineup_woba_season(lineup_ids: list[int], vs_hand: str, on_date: datetime.date) -> float | np.nan:
        if not lineup_ids or vs_hand not in ('L', 'R'):
            return np.nan

        num_sum = 0.0
        den_sum = 0.0

        for pid in lineup_ids:
            try:
                # look up pre-game cumulative numerator/denominator for this batter/hand/date
                v = val_map.loc[(int(pid), vs_hand, on_date)]
                d = den_map.loc[(int(pid), vs_hand, on_date)]
            except KeyError:
                # this batter/hand/date never occurred in the window → skip
                continue
            except Exception:
                # any other lookup/type issue → skip this batter
                continue

            # skip if either is NA or denominator is zero
            if pd.isna(v) or pd.isna(d) or float(d) == 0.0:
                continue

            num_sum += float(v)
            den_sum += float(d)

        return (num_sum / den_sum) if den_sum > 0 else np.nan


    # Compute pre-game lineup wOBA vs opponent SP hand for each game/date
    home_vals, away_vals = [], []
    for i, row in df.iterrows():
        gd = row['game_date'].date()
        pk = int(row['game_pk'])
        lu = lineup_map.get(pk, {'home': [], 'away': []})

        # IMPORTANT: lineup faces the OPPOSING starter's hand
        vs_home = row.get('away_sp_throws')  # away SP hand faced by HOME lineup
        vs_away = row.get('home_sp_throws')  # home SP hand faced by AWAY lineup

        home_vals.append(_lineup_woba_season(lu.get('home', []), vs_home, gd))
        away_vals.append(_lineup_woba_season(lu.get('away', []), vs_away, gd))

    df['home_lineup_woba_vsHand_season'] = home_vals
    df['away_lineup_woba_vsHand_season'] = away_vals
    df['Lineup_woba_vsHand_season_diff'] = (
        df['home_lineup_woba_vsHand_season'] - df['away_lineup_woba_vsHand_season']
    )
    return df

# ---------------------------
# SP rolling via Statcast
# ---------------------------
def _to_date(s): 
    return pd.to_datetime(s).date()

def _fmt(d): 
    return pd.to_datetime(d).strftime("%Y-%m-%d")

@lru_cache(maxsize=5000)
def _pitcher_window_stats(pid: int, end_date_str: str, window_days: int, lg_hr_per_fb: float = 0.105):
    # window is [end - (window_days-1), end-1]
    end = _to_date(end_date_str)
    start = end - timedelta(days=window_days - 1)
    stop  = end - timedelta(days=1)
    if start > stop:
        return {'KminusBB_pct': np.nan, 'FIP_nc': np.nan, 'xFIP_nc': np.nan,
                'IP': np.nan, 'BF': 0, 'FB': 0, 'FB_rate': np.nan}

    # ---- ensure df exists before checking emptiness
    df = None
    for attempt in range(3):
        try:
            df = statcast_pitcher(_fmt(start), _fmt(stop), int(pid))
            if df is not None and not df.empty:
                break
        except Exception:
            if attempt == 2:
                # optional: print/log
                pass
            time.sleep(1 * (attempt + 1))

    if df is None or df.empty:
        return {'KminusBB_pct': np.nan, 'FIP_nc': np.nan, 'xFIP_nc': np.nan,
                'IP': np.nan, 'BF': 0, 'FB': 0, 'FB_rate': np.nan}

    # ---- columns (robust to missing)
    ev = df['events'].astype(str) if 'events' in df.columns else pd.Series('', index=df.index, dtype=str)
    bb_type = df['bb_type'].astype(str) if 'bb_type' in df.columns else pd.Series('', index=df.index, dtype=str)

    # ---- tallies
    BF  = int((ev != '').count())
    K   = int(ev.isin(['strikeout', 'strikeout_double_play']).sum())
    BB  = int((ev == 'walk').sum())
    HBP = int((ev == 'hit_by_pitch').sum())
    HR  = int((ev == 'home_run').sum())
    FB  = int((bb_type == 'fly_ball').sum())

    # outs -> IP
    one_out = [
        'strikeout','field_out','force_out','sac_fly','sac_bunt','fielders_choice_out','other_out',
        'caught_stealing_2b','caught_stealing_3b','caught_stealing_home','pickoff_1b','pickoff_2b','pickoff_3b'
    ]
    two_out = ['double_play','grounded_into_double_play']
    three_out = ['triple_play']
    outs = (
        int(ev.isin(one_out).sum())
        + 2 * int(ev.isin(two_out).sum())
        + 3 * int(ev.isin(three_out).sum())
        + int((ev == 'strikeout_double_play').sum())
    )
    IP = outs / 3.0

    # rates/estimators
    KminusBB = ((K / BF) - (BB / BF)) * 100.0 if BF > 0 else np.nan
    if IP > 0:
        FIP_nc = (13 * HR + 3 * (BB + HBP) - 2 * K) / IP
        HR_hat = FB * lg_hr_per_fb
        xFIP_nc = (13 * HR_hat + 3 * (BB + HBP) - 2 * K) / IP
    else:
        FIP_nc = np.nan
        xFIP_nc = np.nan

    FB_rate = (FB / BF) if BF > 0 else np.nan

    return {
        'KminusBB_pct': KminusBB,
        'FIP_nc': FIP_nc,
        'xFIP_nc': xFIP_nc,
        'IP': IP,
        'BF': BF,
        'FB': FB,
        'FB_rate': FB_rate,
    }


def add_sp_rolling_features_savant(
    games: pd.DataFrame,
    window_days: int = 60,
    lg_hr_per_fb: float = 0.105,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Adds pre-game Statcast rolling features for the listed window:
      - home_SP_KminusBB, away_SP_KminusBB
      - home_SP_FIP,      away_SP_FIP
      - home_SP_xFIP,     away_SP_xFIP
      - home_SP_FB,       away_SP_FB
      - home_SP_FB_rate,  away_SP_FB_rate
      - SP_KminusBB_diff, SP_FIP_diff, SP_xFIP_diff, SP_FB_rate_diff
    """
    df = games.copy()
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df['game_date'] = pd.to_datetime(df['game_date'])
    df = df.sort_values('game_date').reset_index(drop=True)

    n = len(df)
    home_kbb = np.full(n, np.nan); away_kbb = np.full(n, np.nan)
    home_fip = np.full(n, np.nan); away_fip = np.full(n, np.nan)
    home_xfp = np.full(n, np.nan); away_xfp = np.full(n, np.nan)
    home_fb  = np.full(n, np.nan); away_fb  = np.full(n, np.nan)
    home_fbr = np.full(n, np.nan); away_fbr = np.full(n, np.nan)

    total_pitchers = 0; successful_fetches = 0

    # process one calendar day at a time (minimizes redundant pulls per pitcher)
    for dt, idx in df.groupby(df['game_date'].dt.date).groups.items():
        rows = df.iloc[list(idx)]
        pids = pd.unique(pd.concat([rows['home_sp_id'], rows['away_sp_id']], ignore_index=True))
        pids = [pid for pid in pids if pd.notna(pid)]
        if verbose:
            print(f"📅 Processing {dt}: {len(pids)} pitchers for {len(idx)} games...")
        total_pitchers += len(pids)

        cache = {}
        for pid in pids:
            result = _pitcher_window_stats(int(pid), str(dt), window_days, lg_hr_per_fb)
            cache[int(pid)] = result
            if result['BF'] > 0:
                successful_fetches += 1

        for i in idx:
            hid = df.iloc[i]['home_sp_id']; aid = df.iloc[i]['away_sp_id']
            if pd.notna(hid) and int(hid) in cache:
                s = cache[int(hid)]
                home_kbb[i], home_fip[i], home_xfp[i] = s['KminusBB_pct'], s['FIP_nc'], s['xFIP_nc']
                home_fb[i],  home_fbr[i]              = s['FB'], s['FB_rate']
            if pd.notna(aid) and int(aid) in cache:
                s = cache[int(aid)]
                away_kbb[i], away_fip[i], away_xfp[i] = s['KminusBB_pct'], s['FIP_nc'], s['xFIP_nc']
                away_fb[i],  away_fbr[i]              = s['FB'], s['FB_rate']

    if verbose and total_pitchers > 0:
        ok_pct = 100 * successful_fetches / total_pitchers
        print(f"\n✅ SP Rolling Features Complete:")
        print(f"   Total pitchers processed: {total_pitchers}")
        print(f"   Successful Statcast fetches: {successful_fetches} ({ok_pct:.1f}%)")
        print(f"   Failed/Empty: {total_pitchers - successful_fetches} ({100-ok_pct:.1f}%)")

    # attach columns
    df['home_SP_KminusBB'] = home_kbb
    df['away_SP_KminusBB'] = away_kbb
    df['home_SP_FIP']      = home_fip
    df['away_SP_FIP']      = away_fip
    df['home_SP_xFIP']     = home_xfp
    df['away_SP_xFIP']     = away_xfp

    df['home_SP_FB']       = home_fb
    df['away_SP_FB']       = away_fb
    df['home_SP_FB_rate']  = home_fbr
    df['away_SP_FB_rate']  = away_fbr

    # diffs
    df['SP_KminusBB_diff'] = df['home_SP_KminusBB'] - df['away_SP_KminusBB']
    df['SP_FIP_diff']      = df['home_SP_FIP']      - df['away_SP_FIP']
    df['SP_xFIP_diff']     = df['home_SP_xFIP']     - df['away_SP_xFIP']
    df['SP_FB_rate_diff']  = df['home_SP_FB_rate']  - df['away_SP_FB_rate']

    return df

# ---------------------------
# Games fetch (WITH SCORES)
# ---------------------------

def _check_stage(df: pd.DataFrame, name: str):
    try:
        uniq = df['game_pk'].nunique()
    except Exception:
        uniq = 'n/a'
    print(f"{name:>28}: rows={len(df):5d}  unique game_pks={uniq}")

def safe_merge_one_to_one(left: pd.DataFrame, right: pd.DataFrame, **kwargs) -> pd.DataFrame:
    # force one-to-one to prevent row explosion; raises if violated
    return left.merge(right, validate='one_to_one', **kwargs)

def fetch_games_with_probables_statsapi_fast(
    start_date: str,
    end_date  : str,
    include_finished: bool = False,
    fetch_handedness: bool = False,
    max_workers: int = 8,
    debug: bool = False,
) -> pd.DataFrame:

    sched = _retry(statsapi.get, 'schedule',
                   {'startDate': start_date, 'endDate': end_date, 'sportId': 1})

    rows = []
    for d in sched.get('dates', []):
        for g in d.get('games', []):
            status = (g.get('status') or {}).get('detailedState')
            gtype  = g.get('gameType')

            # keep only regular season
            if gtype != 'R':
                continue
            # option: skip finished (for live runs)
            if not include_finished and status in _FINISHED_STATUSES:
                continue
            # drop obviously unusable rows
            if status in {'Postponed', 'Cancelled'}:
                continue

            teams_data = g.get('teams', {}) or {}
            home_data = teams_data.get('home', {}) or {}
            away_data = teams_data.get('away', {}) or {}

            home = home_data.get('team', {}) or {}
            away = away_data.get('team', {}) or {}

            rows.append({
                'game_date'     : d.get('date'),
                'game_datetime' : g.get('gameDate'),
                'game_pk'       : g.get('gamePk'),
                'game_type'     : gtype,
                'status'        : status,
                'venue_id'      : (g.get('venue') or {}).get('id'),
                'venue_name'    : (g.get('venue') or {}).get('name'),
                'home_team_id'  : home.get('id'),
                'home_team_name': home.get('name'),
                'home_abbrev'   : home.get('abbreviation') or _TEAM_ABBR_MAP.get(home.get('id')),
                'away_team_id'  : away.get('id'),
                'away_team_name': away.get('name'),
                'away_abbrev'   : away.get('abbreviation') or _TEAM_ABBR_MAP.get(away.get('id')),
                # scores present for completed games
                'home_score'    : home_data.get('score'),
                'away_score'    : away_data.get('score'),
                'home_win'      : 1 if (home_data.get('isWinner') or False) else 0,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        if debug: print("[DEBUG] schedule returned no rows after filters")
        return df

    # backfill abbreviations & normalize
    if '_TEAM_ABBR_MAP' in globals():
        df['home_abbrev'] = df['home_abbrev'].fillna(df['home_team_id'].map(_TEAM_ABBR_MAP))
        df['away_abbrev'] = df['away_abbrev'].fillna(df['away_team_id'].map(_TEAM_ABBR_MAP))
    need_ids = pd.concat([df['home_team_id'], df['away_team_id']]).dropna().astype(int).unique().tolist()
    abbr_map = _team_abbrev_map_for_ids(need_ids)
    df['home_abbrev'] = _normalize_abbr_col(df['home_abbrev'].fillna(df['home_team_id'].map(abbr_map)))
    df['away_abbrev'] = _normalize_abbr_col(df['away_abbrev'].fillna(df['away_team_id'].map(abbr_map)))

    # dedupe per game (pick latest record by game_datetime if MLB pushes updates)
    df['game_datetime'] = pd.to_datetime(df['game_datetime'], errors='coerce')
    df = (df.sort_values(['game_pk','game_datetime'])
            .drop_duplicates('game_pk', keep='last')
            .reset_index(drop=True))

    # parallel probable starters
    pk_list = df['game_pk'].dropna().astype(int).unique().tolist()
    prob_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut = {ex.submit(_probables_one_game, pk): pk for pk in pk_list}
        for f in as_completed(fut):
            pk = fut[f]
            try:
                prob_map[pk] = f.result()
            except Exception:
                prob_map[pk] = (None, None, None, None)

    probs_df = pd.DataFrame(
        df['game_pk'].astype(int).map(lambda pk: prob_map.get(pk, (None, None, None, None))).tolist(),
        columns=['home_sp_id', 'home_sp_name', 'away_sp_id', 'away_sp_name']
    )
    df = pd.concat([df.reset_index(drop=True), probs_df.reset_index(drop=True)], axis=1)

    if fetch_handedness:
        def _hand(pid):
            if pd.isna(pid): return None
            try:
                p = _retry(statsapi.get, 'people', {'personIds': int(pid)})
                return p['people'][0]['pitchHand']['code']
            except Exception:
                return None
        df['home_sp_throws'] = df['home_sp_id'].apply(_hand)
        df['away_sp_throws'] = df['away_sp_id'].apply(_hand)

    if debug:
        print(f"[DEBUG] schedule rows={len(df)}, unique game_pks={df['game_pk'].nunique()}")

    return df

# ---------------------------
# Team rolling + Elo
# ---------------------------
COL_GAME_ID   = 'game_id'
COL_DATE      = 'game_date'
COL_SEASON    = 'season'
COL_HOME      = 'home_team'
COL_AWAY      = 'away_team'
COL_HS        = 'home_score'
COL_AS        = 'away_score'

def _to_team_games_long(games: pd.DataFrame) -> pd.DataFrame:
    g = games.copy()
    g[COL_DATE] = pd.to_datetime(g[COL_DATE])
    if COL_SEASON not in g.columns:
        g[COL_SEASON] = g[COL_DATE].dt.year

    home = g[[COL_GAME_ID, COL_DATE, COL_SEASON, COL_HOME, COL_AWAY, COL_HS, COL_AS]].copy()
    home.rename(columns={COL_HOME: 'team', COL_AWAY: 'opp', COL_HS: 'runs_for', COL_AS: 'runs_against'}, inplace=True)
    home['is_home'] = 1

    away = g[[COL_GAME_ID, COL_DATE, COL_SEASON, COL_HOME, COL_AWAY, COL_HS, COL_AS]].copy()
    away.rename(columns={COL_AWAY: 'team', COL_HOME: 'opp', COL_AS: 'runs_for', COL_HS: 'runs_against'}, inplace=True)
    away['is_home'] = 0

    long = pd.concat([home, away], ignore_index=True)
    long['run_diff'] = long['runs_for'] - long['runs_against']
    long['win'] = (long['runs_for'] > long['runs_against']).astype(int)
    return long.sort_values([COL_DATE, COL_GAME_ID, 'team']).reset_index(drop=True)

def _add_team_cums_and_rolling(long_df: pd.DataFrame, last_n: int) -> pd.DataFrame:
    df = long_df.copy()
    df.sort_values(['team', COL_SEASON, COL_DATE, COL_GAME_ID], inplace=True)

    group_keys = ['team', COL_SEASON]
    df['stg_runs_for']     = df.groupby(group_keys)['runs_for'].transform(lambda s: s.shift().cumsum())
    df['stg_runs_against'] = df.groupby(group_keys)['runs_against'].transform(lambda s: s.shift().cumsum())
    df['stg_games']        = df.groupby(group_keys).cumcount()

    df['stg_run_diff_per_game'] = np.where(
        df['stg_games'] > 0,
        (df['stg_runs_for'] - df['stg_runs_against']) / df['stg_games'],
        np.nan
    )

    df['stg_wins']    = df.groupby(group_keys)['win'].transform(lambda s: s.shift().cumsum())
    df['stg_win_pct'] = np.where(df['stg_games'] > 0, df['stg_wins'] / df['stg_games'], np.nan)

    def _pre_game_roll(s: pd.Series) -> pd.Series:
        return s.shift().rolling(window=last_n, min_periods=1)

    df['lastN_wins']    = df.groupby('team')['win'].transform(lambda s: _pre_game_roll(s).sum())
    df['lastN_games']   = df.groupby('team')['win'].transform(lambda s: _pre_game_roll(s).count())
    df['lastN_win_pct'] = np.where(df['lastN_games'] > 0, df['lastN_wins'] / df['lastN_games'], np.nan)
    return df

def _compute_simple_elo(long_df: pd.DataFrame, K: float = 20.0, HFA: float = 55.0, base: float = 1500.0) -> pd.DataFrame:
    df = long_df.copy().sort_values([COL_DATE, COL_GAME_ID]).reset_index(drop=True)
    elo = {}
    pre_elo = []

    for _, row in df.iterrows():
        t = row['team']; o = row['opp']; is_home = row['is_home'] == 1
        elo_t = elo.get(t, base); elo_o = elo.get(o, base)
        pre_elo.append(elo_t)

        diff = (elo_t + (HFA if is_home else 0)) - elo_o
        exp_t = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
        score_t = 1.0 if row['win'] == 1 else 0.0

        elo[t] = elo_t + K * (score_t - exp_t)
        elo[o] = elo_o + K * ((1.0 - score_t) - (1.0 - exp_t))

    df['elo_pre'] = pre_elo
    return df

def add_team_rolling_features(
    games_df: pd.DataFrame,
    last_n: int = 10,
    include_elo: bool = False,
    elo_K: float = 20.0,
    elo_HFA: float = 55.0,
    elo_base: float = 1500.0,
) -> pd.DataFrame:

    required = {COL_GAME_ID, COL_DATE, COL_HOME, COL_AWAY, COL_HS, COL_AS}
    missing = required - set(games_df.columns)
    if missing:
        raise ValueError(f"games_df is missing required columns: {missing}")

    long = _to_team_games_long(games_df)
    long = _add_team_cums_and_rolling(long, last_n=last_n)

    if include_elo:
        long = _compute_simple_elo(long, K=elo_K, HFA=elo_HFA, base=elo_base)

    home_feats = long[long['is_home'] == 1][
        [COL_GAME_ID, 'team', 'stg_run_diff_per_game', 'stg_win_pct', 'lastN_wins', 'lastN_games', 'lastN_win_pct']
        + (['elo_pre'] if include_elo else [])
    ].copy()
    away_feats = long[long['is_home'] == 0][
        [COL_GAME_ID, 'team', 'stg_run_diff_per_game', 'stg_win_pct', 'lastN_wins', 'lastN_games', 'lastN_win_pct']
        + (['elo_pre'] if include_elo else [])
    ].copy()

    home_feats.rename(columns={
        'team': 'home_team',
        'stg_run_diff_per_game': 'Home_STG_RunDiffpg',
        'stg_win_pct': 'Home_STG_WinPct',
        'lastN_wins': 'Home_LastN_Wins',
        'lastN_games': 'Home_LastN_Games',
        'lastN_win_pct': 'Home_LastN_WinPct',
        **({'elo_pre': 'Home_Elo'} if include_elo else {})
    }, inplace=True)

    away_feats.rename(columns={
        'team': 'away_team',
        'stg_run_diff_per_game': 'Away_STG_RunDiffpg',
        'stg_win_pct': 'Away_STG_WinPct',
        'lastN_wins': 'Away_LastN_Wins',
        'lastN_games': 'Away_LastN_Games',
        'lastN_win_pct': 'Away_LastN_WinPct',
        **({'elo_pre': 'Away_Elo'} if include_elo else {})
    }, inplace=True)

    out = games_df.copy()
    out = out.merge(home_feats, on=[COL_GAME_ID, 'home_team'], how='left')
    out = out.merge(away_feats, on=[COL_GAME_ID, 'away_team'], how='left')

    out['Team_RunDiffpg_diff']    = out['Home_STG_RunDiffpg'] - out['Away_STG_RunDiffpg']
    out['Team_WinPct_diff']       = out['Home_STG_WinPct']    - out['Away_STG_WinPct']
    out['Team_LastN_WinPct_diff'] = out['Home_LastN_WinPct']  - out['Away_LastN_WinPct']
    if include_elo:
        out['Team_Elo_diff'] = out['Home_Elo'] - out['Away_Elo']
    return out

# ---------------------------
# Missingness utility
# ---------------------------
def missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        print("⚠️ The DataFrame is empty — no missingness to compute.")
        return pd.DataFrame(columns=["column", "missing_pct"])
    missing_pct = df.isna().mean() * 100
    return (missing_pct.reset_index().rename(columns={"index": "column", 0: "missing_pct"})
            .sort_values(by="missing_pct", ascending=False).reset_index(drop=True))



# =======================
# Bullpen rolling features
# =======================
def _fetch_game_starters_bulk(game_pks: list[int], max_workers: int = 12) -> dict[int, tuple[int|None, int|None]]:
    """
    Return {game_pk: (home_starter_id, away_starter_id)} using boxscore.
    """
    def _one(pk: int):
        try:
            j = _retry(statsapi.get, 'game', {'gamePk': int(pk)})
            teams = (j.get('liveData') or {}).get('boxscore', {}).get('teams', {})
            def _starter(side):
                players = (teams.get(side, {}) or {}).get('players', {}) or {}
                for p in players.values():
                    st = (p.get('stats') or {}).get('pitching') or {}
                    gs = st.get('gamesStarted', 0) or st.get('gamesStartedPitching', 0)
                    pid = (p.get('person') or {}).get('id')
                    if gs and pid:
                        return int(pid)
                return None
            return (pk, (_starter('home'), _starter('away')))
        except Exception:
            return (pk, (None, None))
    out = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut = {ex.submit(_one, int(pk)): pk for pk in game_pks}
        for f in as_completed(fut):
            pk, tup = f.result()
            out[int(pk)] = tup
    return out

def _outs_from_events(ev: pd.Series) -> int:
    one_out = {
        'strikeout','field_out','force_out','sac_fly','sac_bunt','fielders_choice_out','other_out',
        'caught_stealing_2b','caught_stealing_3b','caught_stealing_home','pickoff_1b','pickoff_2b','pickoff_3b'
    }
    two_out = {'double_play','grounded_into_double_play'}
    three_out = {'triple_play'}
    return (
        int(ev.isin(one_out).sum())
        + 2 * int(ev.isin(two_out).sum())
        + 3 * int(ev.isin(three_out).sum())
        + int((ev == 'strikeout_double_play').sum())  # add the extra out
    )

def _bp_aggregate_xfip(df: pd.DataFrame, lg_hr_per_fb: float = 0.105) -> tuple[float, float]:
    """
    Aggregate over a set of bullpen rows (already filtered) and return (IP, xFIP).
    """
    if df.empty:
        return 0.0, np.nan
    ev = df['events'].astype(str)
    bb_type = df.get('bb_type', pd.Series('', index=df.index)).astype(str)
    BF  = int((ev != '').sum())
    K   = int(ev.isin(['strikeout','strikeout_double_play']).sum())
    BB  = int((ev == 'walk').sum())
    HBP = int((ev == 'hit_by_pitch').sum())
    HR  = int((ev == 'home_run').sum())
    FB  = int((bb_type == 'fly_ball').sum())
    IP  = _outs_from_events(ev) / 3.0
    if IP <= 0:
        return 0.0, np.nan
    HR_hat = FB * lg_hr_per_fb
    xFIP = (13*HR_hat + 3*(BB + HBP) - 2*K) / IP
    return float(IP), float(xFIP)

def add_bullpen_rolling_features(
    games: pd.DataFrame,
    window_days: int = 10,
    lg_hr_per_fb: float = 0.105,
    max_workers: int = 12,
) -> pd.DataFrame:
    """
    Adds rolling bullpen workload/quality features computed PRE-GAME for each date:
      - home_BP_IP_lastN, away_BP_IP_lastN, BP_IP_lastN_diff
      - home_BP_xFIP_lastN, away_BP_xFIP_lastN, BP_xFIP_lastN_diff

    Requires columns: ['game_pk','game_date','home_team_name','away_team_name','home_abbrev','away_abbrev']
    """
    df = games.copy()
    df['game_date'] = pd.to_datetime(df['game_date'])
    df = df.sort_values('game_date').reset_index(drop=True)

    # Map full name -> abbrev (for consistent team keys)
    name_to_abbr = {}
    if 'home_abbrev' in df.columns and 'home_team_name' in df.columns:
        name_to_abbr.update(dict(df[['home_team_name','home_abbrev']].dropna().values))
    if 'away_abbrev' in df.columns and 'away_team_name' in df.columns:
        name_to_abbr.update(dict(df[['away_team_name','away_abbrev']].dropna().values))

    # Pre-fetch starters for ALL games in the whole range (so window filtering can exclude them)
    all_pks = df['game_pk'].dropna().astype(int).unique().tolist()
    starters_map = _fetch_game_starters_bulk(all_pks, max_workers=max_workers)  # {pk: (homeSP, awaySP)}

    # Prepare outputs
    n = len(df)
    h_ip = np.full(n, np.nan); a_ip = np.full(n, np.nan)
    h_xf = np.full(n, np.nan); a_xf = np.full(n, np.nan)

    # Group by calendar date; do one Statcast pull per date (window)
    by_date = df.groupby(df['game_date'].dt.date).groups
    for dt, idx in by_date.items():
        idx = list(idx)

        # build rolling window [dt - window_days, dt - 1]
        end = pd.to_datetime(dt).date()
        start = (end - timedelta(days=window_days)).strftime('%Y-%m-%d')
        stop  = (end - timedelta(days=1)).strftime('%Y-%m-%d')
        if pd.to_datetime(start) > pd.to_datetime(stop):
            # nothing to compute yet (first day)
            for i in idx:
                h_ip[i]=0.0; a_ip[i]=0.0; h_xf[i]=np.nan; a_xf[i]=np.nan
            continue

        # Pull once for the whole window
        sc = statcast(start_dt=start, end_dt=stop)
        if sc is None or sc.empty:
            for i in idx:
                h_ip[i]=0.0; a_ip[i]=0.0; h_xf[i]=np.nan; a_xf[i]=np.nan
            continue

        # Identify pitcher team (home or away) for each pitch
        # If inning_topbot == 'Top' -> fielding team is home; else away
        sc = sc[['game_pk','home_team','away_team','inning_topbot','pitcher','events','bb_type']].copy()
        sc['pitcher_team_name'] = np.where(sc['inning_topbot'].str.upper().eq('TOP'),
                                           sc['home_team'], sc['away_team'])
        # Map to abbreviation keys (fall back to name if unknown)
        sc['pitcher_team'] = sc['pitcher_team_name'].map(name_to_abbr).fillna(sc['pitcher_team_name'])

        # Build a dict of starters per (game_pk, team_key) to exclude from bullpen
        to_exclude = {}
        for pk, (hsp, asp) in starters_map.items():
            # get team keys for this pk from any row in statcast (if present)
            subset = sc[sc['game_pk']==pk]
            if subset.empty:
                continue
            # team keys
            home_name = subset['home_team'].iloc[0]
            away_name = subset['away_team'].iloc[0]
            home_key = name_to_abbr.get(home_name, home_name)
            away_key = name_to_abbr.get(away_name, away_name)
            if hsp:
                to_exclude[(pk, home_key)] = int(hsp)
            if asp:
                to_exclude[(pk, away_key)] = int(asp)

        # Mark bullpen rows (exclude starters for that (game_pk, team))
        def _is_bp_row(row):
            key = (int(row['game_pk']), row['pitcher_team'])
            starter = to_exclude.get(key)
            return (starter is None) or (int(row['pitcher']) != int(starter))
        sc_bp = sc[sc.apply(_is_bp_row, axis=1)]
        if sc_bp.empty:
            for i in idx:
                h_ip[i]=0.0; a_ip[i]=0.0; h_xf[i]=np.nan; a_xf[i]=np.nan
            continue

        # Aggregate per team over the window
        team_groups = sc_bp.groupby('pitcher_team')

        agg_cache = {}
        for team_key, g in team_groups:
            agg_cache[team_key] = _bp_aggregate_xfip(g, lg_hr_per_fb=lg_hr_per_fb)

        # Fill outputs for each game on date dt
        for i in idx:
            h_key = df.iloc[i].get('home_abbrev') or df.iloc[i]['home_team_name']
            a_key = df.iloc[i].get('away_abbrev') or df.iloc[i]['away_team_name']
            # prefer abbrev mapping if available
            h_key = name_to_abbr.get(df.iloc[i]['home_team_name'], h_key)
            a_key = name_to_abbr.get(df.iloc[i]['away_team_name'], a_key)

            h_ip[i], h_xf[i] = agg_cache.get(h_key, (0.0, np.nan))
            a_ip[i], a_xf[i] = agg_cache.get(a_key, (0.0, np.nan))

    # Attach to frame
    df['home_BP_IP_lastN']   = h_ip
    df['away_BP_IP_lastN']   = a_ip
    df['BP_IP_lastN_diff']   = df['home_BP_IP_lastN'] - df['away_BP_IP_lastN']

    df['home_BP_xFIP_lastN'] = h_xf
    df['away_BP_xFIP_lastN'] = a_xf
    df['BP_xFIP_lastN_diff'] = df['home_BP_xFIP_lastN'] - df['away_BP_xFIP_lastN']

    return df


def add_bullpen_season_xfip_features(
    games: pd.DataFrame,
    season_start: str | None = None,   # e.g. "2024-03-01" (defaults to March 1 of games' year)
    lg_hr_per_fb: float = 0.105,
) -> pd.DataFrame:
    """
    Adds season-to-date bullpen xFIP (pre-game):
      - home_BP_xFIP_season, away_BP_xFIP_season, BP_xFIP_season_diff

    Implementation:
      * Single Statcast pull for [season_start, max_game_date - 1]
      * For each (game_pk, team), infer starter as the FIRST pitcher to appear
        in Statcast; exclude starter rows => bullpen-only events
      * Aggregate bullpen events per (team_key, game_date)
      * Cumulative sums per team up to the *previous* day yield pre-game season xFIP
    """
    if games.empty:
        return games

    df = games.copy()
    df['game_date'] = pd.to_datetime(df['game_date'])
    df = df.sort_values('game_date').reset_index(drop=True)

    # Team key mapping (prefer abbreviations)
    name_to_abbr = {}
    if {'home_team_name','home_abbrev'}.issubset(df.columns):
        name_to_abbr.update(dict(df[['home_team_name','home_abbrev']].dropna().values))
    if {'away_team_name','away_abbrev'}.issubset(df.columns):
        name_to_abbr.update(dict(df[['away_team_name','away_abbrev']].dropna().values))

    # Season range
    max_dt = df['game_date'].max().date()
    end_for_pull = (max_dt - timedelta(days=1)).strftime('%Y-%m-%d')
    if season_start is None:
        yr = df['game_date'].dt.year.min()
        season_start = f"{yr}-03-01"  # safe early-season default

    if pd.to_datetime(season_start) > pd.to_datetime(end_for_pull):
        # nothing to compute yet
        df['home_BP_xFIP_season'] = np.nan
        df['away_BP_xFIP_season'] = np.nan
        df['BP_xFIP_season_diff'] = np.nan
        return df

    # One Statcast pull for the season-to-date window
    sc = statcast(start_dt=season_start, end_dt=end_for_pull)
    if sc is None or sc.empty:
        df['home_BP_xFIP_season'] = np.nan
        df['away_BP_xFIP_season'] = np.nan
        df['BP_xFIP_season_diff'] = np.nan
        return df

    # Keep only the columns we need
    keep = ['game_pk','game_date','home_team','away_team','inning_topbot','pitcher','events','bb_type']
    sc = sc[[c for c in keep if c in sc.columns]].copy()

    # Team that is pitching this play: if TOP → home fields; else away fields
    sc['pitcher_team_name'] = np.where(sc['inning_topbot'].str.upper().eq('TOP'),
                                       sc['home_team'], sc['away_team'])
    sc['team_key'] = sc['pitcher_team_name'].map(name_to_abbr).fillna(sc['pitcher_team_name'])

    # ---- Infer starters per (game_pk, team_key): first pitcher to appear for that team
    # Use the existing row order (Statcast returns chronological within game)
    first_pitcher = (sc.groupby(['game_pk','team_key'])['pitcher']
                       .first()
                       .rename('starter_pid')
                       .reset_index())
    sc = sc.merge(first_pitcher, on=['game_pk','team_key'], how='left')

    # Bullpen-only rows
    sc_bp = sc[sc['pitcher'] != sc['starter_pid']].copy()
    if sc_bp.empty:
        df['home_BP_xFIP_season'] = np.nan
        df['away_BP_xFIP_season'] = np.nan
        df['BP_xFIP_season_diff'] = np.nan
        return df

    # ---- Aggregate bullpen events per team × day
    ev = sc_bp['events'].astype(str)
    sc_bp['K']   = ev.isin(['strikeout','strikeout_double_play']).astype(int)
    sc_bp['BB']  = (ev == 'walk').astype(int)
    sc_bp['HBP'] = (ev == 'hit_by_pitch').astype(int)
    sc_bp['HR']  = (ev == 'home_run').astype(int)
    bb_type = sc_bp.get('bb_type', pd.Series('', index=sc_bp.index)).astype(str)
    sc_bp['FB']  = (bb_type == 'fly_ball').astype(int)

    # Outs from events
    one_out = {
        'strikeout','field_out','force_out','sac_fly','sac_bunt','fielders_choice_out','other_out',
        'caught_stealing_2b','caught_stealing_3b','caught_stealing_home','pickoff_1b','pickoff_2b','pickoff_3b'
    }
    two_out = {'double_play','grounded_into_double_play'}
    three_out = {'triple_play'}
    sc_bp['OUTS'] = (
        ev.isin(one_out).astype(int)
        + 2 * ev.isin(two_out).astype(int)
        + 3 * ev.isin(three_out).astype(int)
        + (ev == 'strikeout_double_play').astype(int)  # add the extra out on K-DP
    )

    sc_bp['game_date'] = pd.to_datetime(sc_bp['game_date']).dt.date
    daily = (sc_bp
             .groupby(['team_key','game_date'], as_index=False)
             [['K','BB','HBP','HR','FB','OUTS']].sum())

    # ---- Cumulative to *previous* day (pre-game)
    daily = daily.sort_values(['team_key','game_date'])
    for col in ['K','BB','HBP','HR','FB','OUTS']:
        daily[f'cum_{col}'] = daily.groupby('team_key')[col].cumsum()

    # Pre-game means shift by one day within each team
    for col in ['cum_K','cum_BB','cum_HBP','cum_HR','cum_FB','cum_OUTS']:
        daily[col] = daily.groupby('team_key')[col].shift(1)

    # Compute xFIP from cumulatives
    daily['IP'] = daily['cum_OUTS'] / 3.0
    daily['xFIP_season'] = np.where(
        daily['IP'] > 0,
        (13 * daily['cum_FB'] * lg_hr_per_fb + 3 * (daily['cum_BB'] + daily['cum_HBP']) - 2 * daily['cum_K']) / daily['IP'],
        np.nan
    )

    # Map (team_key, date) → season xFIP up to that day
    # Convert df dates to pure date for joining
    df_dates = df['game_date'].dt.date
    df_home_key = df.get('home_abbrev', df['home_team_name'])
    df_away_key = df.get('away_abbrev', df['away_team_name'])

    # Build lookup dicts
    # For speed, pivot to a MultiIndex Series
    idx = pd.MultiIndex.from_frame(daily[['team_key','game_date']])
    xmap = pd.Series(daily['xFIP_season'].values, index=idx)

    def _lookup(team_key, d):
        try:
            return float(xmap.loc[(team_key, d)])
        except KeyError:
            return np.nan

    home_vals = []
    away_vals = []
    for tkey_h, tkey_a, d in zip(df_home_key, df_away_key, df_dates):
        home_vals.append(_lookup(tkey_h, d))
        away_vals.append(_lookup(tkey_a, d))

    df['home_BP_xFIP_season'] = home_vals
    df['away_BP_xFIP_season'] = away_vals
    df['BP_xFIP_season_diff'] = df['home_BP_xFIP_season'] - df['away_BP_xFIP_season']

    return df

def add_season_park_factor_features(
    games: pd.DataFrame,
    prior_games: int = 300,   # strength of the prior toward neutral (1.00)
    friendly_hi: float = 1.02,
    friendly_lo: float = 0.98,
) -> pd.DataFrame:
    """
    Adds pre-game, season-to-date park factor features computed from your own scores:
      - Park_RunFactor_season  (continuous, neutral=1.00)
      - Park_Friendly_flag     (1 hitter, 0 neutral, -1 pitcher)
      - Park_RunFactor_diff    (centered at 0 = neutral)

    Assumes columns exist: ['game_pk','game_date','venue_id','home_score','away_score'].
    Uses within-season league avg runs/game as baseline. Shrinks small samples toward 1.00.
    Leakage-safe: for each game, only games at that venue *before* that date are used.
    """
    df = games.copy()
    if df.empty:
        return df

    # Ensure types and helper columns
    df['game_date'] = pd.to_datetime(df['game_date'])
    df['_season']   = df['game_date'].dt.year
    # normalize to (date at midnight) so we can join reliably
    df['gdate']     = df['game_date'].dt.normalize()
    df['_runs']     = pd.to_numeric(df['home_score'], errors='coerce').fillna(0) + \
                      pd.to_numeric(df['away_score'], errors='coerce').fillna(0)

    # -------------------------------
    # League baseline per season/day
    # -------------------------------
    league_daily = (df.groupby(['_season','gdate'], as_index=False)
                      .agg(league_runs=('_runs','sum'),
                           league_games=('game_pk','nunique')))
    league_daily['lg_rpg_day'] = league_daily['league_runs'] / league_daily['league_games']

    # Pre-game cumulative league runs/game per season (shift by 1 day)
    league_daily = league_daily.sort_values(['_season','gdate'])
    league_daily['cum_runs']  = league_daily.groupby('_season')['league_runs'].cumsum().shift(1)
    league_daily['cum_games'] = league_daily.groupby('_season')['league_games'].cumsum().shift(1)
    league_daily['lg_rpg_pre'] = np.where(
        league_daily['cum_games'] > 0,
        league_daily['cum_runs'] / league_daily['cum_games'],
        np.nan
    )

    # Map for quick lookups
    lg_map = pd.Series(
        league_daily['lg_rpg_pre'].values,
        index=pd.MultiIndex.from_frame(league_daily[['_season','gdate']])
    )

    # --------------------------------------
    # Venue totals and pre-game cumulatives
    # --------------------------------------
    venue_daily = (df.groupby(['_season','gdate','venue_id'], as_index=False)
                     .agg(venue_runs=('_runs','sum'),
                          venue_games=('game_pk','nunique')))

    venue_daily = venue_daily.sort_values(['venue_id','_season','gdate'])
    # cumulative pre-game
    venue_daily['cum_runs']  = venue_daily.groupby(['venue_id','_season'])['venue_runs'].cumsum().shift(1)
    venue_daily['cum_games'] = venue_daily.groupby(['venue_id','_season']).cumcount()

    # attach league pre-game rpg for the same season/date
    key = pd.MultiIndex.from_frame(venue_daily[['_season','gdate']])
    venue_daily['lg_rpg_pre'] = lg_map.reindex(key).values

    # venue pre-game runs/game (with shrinkage toward league)
    venue_daily['venue_rpg_pre'] = np.where(
        venue_daily['cum_games'] > 0,
        venue_daily['cum_runs'] / venue_daily['cum_games'],
        np.nan
    )
    venue_daily['venue_rpg_shrunk'] = np.where(
        (venue_daily['cum_games'] > 0) & venue_daily['lg_rpg_pre'].notna(),
        (venue_daily['cum_runs'] + prior_games * venue_daily['lg_rpg_pre']) /
        (venue_daily['cum_games'] + prior_games),
        np.nan
    )

    # Park factor = venue_rpg / league_rpg (both pre-game, shrunk)
    venue_daily['Park_RunFactor_season'] = np.where(
        venue_daily['lg_rpg_pre'] > 0,
        venue_daily['venue_rpg_shrunk'] / venue_daily['lg_rpg_pre'],
        np.nan
    )

    # Build lookup: (venue_id, season, gdate) -> PF
    pf_map = pd.Series(
        venue_daily['Park_RunFactor_season'].values,
        index=pd.MultiIndex.from_frame(venue_daily[['venue_id','_season','gdate']])
    )

    # Attach to games frame using the same (venue, season, date)
    out_vals = []
    for vid, yr, gd in zip(df['venue_id'], df['_season'], df['gdate']):
        try:
            out_vals.append(float(pf_map.loc[(vid, yr, gd)]))
        except KeyError:
            out_vals.append(np.nan)

    df['Park_RunFactor_season'] = out_vals
    df['Park_RunFactor_diff']   = df['Park_RunFactor_season'] - 1.00
    df['Park_Friendly_flag']    = np.where(df['Park_RunFactor_season'] >= friendly_hi,  1,
                                   np.where(df['Park_RunFactor_season'] <= friendly_lo, -1, 0))

    return df

# =============================================================================
# Extra Context Features (SP rest, bullpen rest, weather, umpire, fatigue)
# =============================================================================

def add_sp_rest_features(games: pd.DataFrame) -> pd.DataFrame:
    """
    Adds starting-pitcher rest/last-start features:
      - home_SP_days_rest, away_SP_days_rest
      - home_SP_laststart_IP, away_SP_laststart_IP
    Looks up the most recent *start* strictly before the game's date.
    """
    if games.empty:
        return games

    def _ip_to_float(ip):
        try:
            s = str(ip)
            if '.' not in s:
                return float(int(s))
            whole, frac = s.split('.', 1)
            w = int(whole)
            f = 1/3 if frac[:1] == '1' else (2/3 if frac[:1] == '2' else 0.0)
            return float(w + f)
        except Exception:
            return np.nan

    df = games.copy()
    df['game_date'] = pd.to_datetime(df['game_date']).dt.date

    from functools import lru_cache

    @lru_cache(maxsize=50000)
    def _starts_for(pid: int, year: int):
        """
        Return a sorted list of (date, IP_float) for *starts* in the given season.
        Uses the 'hydrate=stats(type=gameLog,...)' shape from StatsAPI.
        """
        try:
            hydro = f"stats(group=pitching,type=gameLog,season={int(year)},gameType=R)"
            j = _retry(statsapi.get, 'people', {'personIds': int(pid), 'hydrate': hydro})
            people = j.get('people') or []
            if not people:
                return []
            stats = people[0].get('stats') or []
            if not stats:
                return []
            splits = stats[0].get('splits') or []
            out = []
            for s in splits:
                st = s.get('stat', {}) or {}
                # only count starts
                gs = st.get('gamesStarted') or st.get('gamesStartedPitching') or 0
                if not gs:
                    continue
                dstr = (s.get('date') or '')[:10]
                if not dstr:
                    continue
                ip = _ip_to_float(st.get('inningsPitched'))
                out.append((pd.to_datetime(dstr).date(), ip))
            return sorted(out)
        except Exception:
            return []

    home_days, away_days, home_ip, away_ip = [], [], [], []
    for _, r in df.iterrows():
        gd = r['game_date']
        yr = int(gd.year)

        # helper to look up prior start; tries current season then previous
        def _prior_start(pid):
            if pd.isna(pid):
                return (np.nan, np.nan)
            pid = int(pid)
            starts = _starts_for(pid, yr)
            if not starts:
                starts = _starts_for(pid, yr - 1)
            prev = [t for t in starts if t[0] < gd]
            if not prev:
                return (np.nan, np.nan)
            ps, lip = prev[-1]
            return ((gd - ps).days, lip)

        d, ip = _prior_start(r.get('home_sp_id'))
        home_days.append(d); home_ip.append(ip)
        d, ip = _prior_start(r.get('away_sp_id'))
        away_days.append(d); away_ip.append(ip)

    df['home_SP_days_rest']    = home_days
    df['away_SP_days_rest']    = away_days
    df['home_SP_laststart_IP'] = home_ip
    df['away_SP_laststart_IP'] = away_ip
    return df


def add_bullpen_rest_snapshots(
    games: pd.DataFrame,
    windows: tuple[int, ...] = (1, 3),
    lg_hr_per_fb: float = 0.105,
    max_workers: int = 12
) -> pd.DataFrame:
    """
    Adds bullpen workload snapshots (IP only, and xFIP optional) for short lookbacks:
      - home_BP_IP_d{w}, away_BP_IP_d{w}          (for each w in windows, e.g., d1, d3)
      - BP_IP_d{w}_diff
    Uses Statcast windows that end the day BEFORE the game. Excludes starters.
    """
    if games.empty:
        return games

    df = games.copy()
    df['game_date'] = pd.to_datetime(df['game_date'])
    df = df.sort_values('game_date').reset_index(drop=True)

    # map team name -> abbrev (consistent key)
    name_to_abbr = {}
    if {'home_team_name', 'home_abbrev'}.issubset(df.columns):
        name_to_abbr.update(dict(df[['home_team_name', 'home_abbrev']].dropna().values))
    if {'away_team_name', 'away_abbrev'}.issubset(df.columns):
        name_to_abbr.update(dict(df[['away_team_name', 'away_abbrev']].dropna().values))

    # starters per game (to exclude from bullpen)
    all_pks = df['game_pk'].dropna().astype(int).unique().tolist()
    starters_map = _fetch_game_starters_bulk(all_pks, max_workers=max_workers)  # {pk: (homeSP, awaySP)}

    # Helper to compute team bullpen IP in an arbitrary [start, stop] date range
    def _bp_ip_range(start: str, stop: str) -> pd.DataFrame:
        sc = statcast(start_dt=start, end_dt=stop)
        if sc is None or sc.empty:
            return pd.DataFrame(columns=['pitcher_team', 'IP'])
        keep = ['game_pk', 'home_team', 'away_team', 'inning_topbot', 'pitcher', 'events', 'bb_type']
        sc = sc[[c for c in keep if c in sc.columns]].copy()
        # which team is in field
        sc['pitcher_team_name'] = np.where(sc['inning_topbot'].str.upper().eq('TOP'),
                                           sc['home_team'], sc['away_team'])
        sc['pitcher_team'] = sc['pitcher_team_name'].map(name_to_abbr).fillna(sc['pitcher_team_name'])
        # exclude starters for that team in that game
        def _is_bp_row(row):
            pk = int(row['game_pk'])
            team = row['pitcher_team']
            # map team name back to key to find starter by home/away
            subset = sc[sc['game_pk'] == pk]
            if subset.empty:
                return True
            home_name = subset['home_team'].iloc[0]
            away_name = subset['away_team'].iloc[0]
            home_key  = name_to_abbr.get(home_name, home_name)
            away_key  = name_to_abbr.get(away_name, away_name)
            hsp, asp = starters_map.get(pk, (None, None))
            if team == home_key and hsp is not None:
                return int(row['pitcher']) != int(hsp)
            if team == away_key and asp is not None:
                return int(row['pitcher']) != int(asp)
            return True

        sc = sc[sc.apply(_is_bp_row, axis=1)]
        if sc.empty:
            return pd.DataFrame(columns=['pitcher_team', 'IP'])

        # convert events → outs → IP
        ev = sc['events'].astype(str)
        outs = _outs_from_events(ev)
        sc['_OUTS'] = outs
        ip_by_team = (sc.groupby('pitcher_team')['_OUTS'].sum() / 3.0).rename('IP').reset_index()
        return ip_by_team

    # Pre-allocate result columns
    for w in windows:
        df[f'home_BP_IP_d{w}'] = np.nan
        df[f'away_BP_IP_d{w}'] = np.nan
        df[f'BP_IP_d{w}_diff'] = np.nan

    # Do per calendar date to minimize pulls
    for dt, idx in df.groupby(df['game_date'].dt.date).groups.items():
        end = pd.to_datetime(dt).date()
        for w in windows:
            start = (end - timedelta(days=w)).strftime('%Y-%m-%d')
            stop  = (end - timedelta(days=1)).strftime('%Y-%m-%d')
            if pd.to_datetime(start) > pd.to_datetime(stop):
                # nothing yet (first day)
                for i in idx:
                    df.at[i, f'home_BP_IP_d{w}'] = 0.0
                    df.at[i, f'away_BP_IP_d{w}'] = 0.0
                    df.at[i, f'BP_IP_d{w}_diff'] = 0.0
                continue

            snap = _bp_ip_range(start, stop)
            for i in idx:
                h_key = name_to_abbr.get(df.at[i, 'home_team_name'], df.at[i, 'home_team_name'])
                a_key = name_to_abbr.get(df.at[i, 'away_team_name'], df.at[i, 'away_team_name'])

                h_ip = float(snap[snap['pitcher_team'] == h_key]['IP'].sum()) if not snap.empty else 0.0
                a_ip = float(snap[snap['pitcher_team'] == a_key]['IP'].sum()) if not snap.empty else 0.0
                df.at[i, f'home_BP_IP_d{w}'] = h_ip
                df.at[i, f'away_BP_IP_d{w}'] = a_ip
                df.at[i, f'BP_IP_d{w}_diff'] = h_ip - a_ip

    return df


def add_basic_weather(games: pd.DataFrame, max_workers: int = 16) -> pd.DataFrame:
    """
    Adds simple weather fields straight from the StatsAPI game object:
      - weather_temp_f (float), weather_wind_mph (float), weather_wind_dir (str), weather_condition (str)
    """
    if games.empty:
        return games

    df = games.copy()
    df['game_pk'] = pd.to_numeric(df['game_pk'], errors='coerce')

    def _parse_temp(t):
        # examples: "80 F, Sunny"
        try:
            if t is None:
                return np.nan
            s = str(t)
            num = ''.join(ch for ch in s if (ch.isdigit() or ch == '.' or ch == '-'))
            return float(num) if num not in ('', '-') else np.nan
        except Exception:
            return np.nan

    def _parse_wind(w):
        # examples: "8 mph, In from LF" or "10 mph, Out to RF"
        try:
            if w is None:
                return (np.nan, None)
            s = str(w)
            parts = s.split(',', 1)
            mph_part = parts[0]
            dir_part = parts[1].strip() if len(parts) > 1 else None
            num = ''.join(ch for ch in mph_part if (ch.isdigit() or ch == '.' or ch == '-'))
            return (float(num) if num not in ('', '-') else np.nan, dir_part)
        except Exception:
            return (np.nan, None)

    weather_map = {}

    def _one(pk):
        try:
            j = _retry(statsapi.get, 'game', {'gamePk': int(pk)})
            w = ((j.get('gameData') or {}).get('weather') or {})
            temp = _parse_temp(w.get('temp') or w.get('temperature'))
            speed, wdir = _parse_wind(w.get('wind'))
            cond = w.get('condition')
            return (pk, (temp, speed, wdir, cond))
        except Exception:
            return (pk, (np.nan, np.nan, None, None))

    pks = [int(x) for x in df['game_pk'].dropna().unique().tolist()]
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_one, pk): pk for pk in pks}
        for f in as_completed(futs):
            k, tup = f.result()
            weather_map[int(k)] = tup

    df['weather_temp_f']    = df['game_pk'].map(lambda pk: weather_map.get(int(pk), (np.nan, np.nan, None, None))[0])
    df['weather_wind_mph']  = df['game_pk'].map(lambda pk: weather_map.get(int(pk), (np.nan, np.nan, None, None))[1])
    df['weather_wind_dir']  = df['game_pk'].map(lambda pk: weather_map.get(int(pk), (np.nan, np.nan, None, None))[2])
    df['weather_condition'] = df['game_pk'].map(lambda pk: weather_map.get(int(pk), (np.nan, np.nan, None, None))[3])
    return df


def add_plate_umpire(games: pd.DataFrame, max_workers: int = 16) -> pd.DataFrame:
    """
    Adds plate umpire info from StatsAPI:
      - plate_umpire_id (int), plate_umpire_name (str)
    """
    if games.empty:
        return games

    df = games.copy()
    df['game_pk'] = pd.to_numeric(df['game_pk'], errors='coerce')

    def _one(pk):
        try:
            j = _retry(statsapi.get, 'game', {'gamePk': int(pk)})
            officials = ((j.get('liveData') or {}).get('boxscore') or {}).get('officials') or []
            # look for 'Home Plate' or similar
            for off in officials:
                title = (off.get('officialType') or '').lower()
                if 'home' in title and 'plate' in title:
                    person = off.get('official') or {}
                    return (pk, (person.get('id'), person.get('fullName')))
            return (pk, (None, None))
        except Exception:
            return (pk, (None, None))

    pk_list = [int(x) for x in df['game_pk'].dropna().unique().tolist()]
    ump_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_one, pk): pk for pk in pk_list}
        for f in as_completed(futs):
            k, v = f.result()
            ump_map[int(k)] = v

    df['plate_umpire_id']   = df['game_pk'].map(lambda pk: ump_map.get(int(pk), (None, None))[0])
    df['plate_umpire_name'] = df['game_pk'].map(lambda pk: ump_map.get(int(pk), (None, None))[1])
    return df


def add_schedule_fatigue(games: pd.DataFrame) -> pd.DataFrame:
    """
    Adds simple schedule fatigue features based only on your games frame:
      - Home_DaysSinceLast, Away_DaysSinceLast
      - Home_B2B, Away_B2B  (1 if played yesterday)
    Note: if your date range starts mid-season, first appearance for each team
    will return NaN for DaysSinceLast (expected).
    """
    if games.empty:
        return games

    df = games.copy()
    df['game_date'] = pd.to_datetime(df['game_date']).dt.date

    # Build per-team series of dates from both home/away appearances
    team_dates = {}
    for _, r in df.iterrows():
        for side, col in (('home', 'home_team_name'), ('away', 'away_team_name')):
            tm = r[col]
            if pd.isna(tm):
                continue
            team_dates.setdefault(tm, []).append(r['game_date'])

    # For each team, compute gap to previous appearance
    prev_map = {}  # (team, date) -> days since last
    for tm, dts in team_dates.items():
        dts = sorted(set(dts))
        prev = {}
        last = None
        for d in dts:
            prev[d] = (d - last).days if last else np.nan
            last = d
        for d, gap in prev.items():
            prev_map[(tm, d)] = gap

    # Attach to rows
    H_days, A_days = [], []
    for _, r in df.iterrows():
        H_days.append(prev_map.get((r['home_team_name'], r['game_date']), np.nan))
        A_days.append(prev_map.get((r['away_team_name'], r['game_date']), np.nan))

    df['Home_DaysSinceLast'] = H_days
    df['Away_DaysSinceLast'] = A_days
    df['Home_B2B'] = (df['Home_DaysSinceLast'] == 1).astype(float)
    df['Away_B2B'] = (df['Away_DaysSinceLast'] == 1).astype(float)
    return df

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create curated interaction terms. Everything uses only pre-game inputs.
    NaN-safe (products become NaN if either side is NaN).
    """
    out = df.copy()

    # ---- Pitching × Park context
    if {'SP_xFIP_diff','Park_RunFactor_season'}.issubset(out.columns):
        out['INT_SPxFIPdiff__Park'] = out['SP_xFIP_diff'] * out['Park_RunFactor_season']
    if {'SP_KminusBB_diff','Park_RunFactor_season'}.issubset(out.columns):
        out['INT_SPKBBdiff__Park'] = out['SP_KminusBB_diff'] * out['Park_RunFactor_season']

    # ---- Lineup quality × Pitcher quality
    if {'Lineup_woba_vsHand_diff','SP_KminusBB_diff'}.issubset(out.columns):
        out['INT_LineupVsHand__SPKBB'] = out['Lineup_woba_vsHand_diff'] * out['SP_KminusBB_diff']
    if {'Lineup_woba_vsHand_season_diff','SP_xFIP_diff'}.issubset(out.columns):
        out['INT_LineupSeason__SPxFIP'] = out['Lineup_woba_vsHand_season_diff'] * out['SP_xFIP_diff']

    # ---- Team form × Elo (recent form amplifies/attenuates ratings)
    if {'Team_LastN_WinPct_diff','Team_Elo_diff'}.issubset(out.columns):
        out['INT_LastN__Elo'] = out['Team_LastN_WinPct_diff'] * out['Team_Elo_diff']

    # ---- Bullpen fatigue × quality
    if {'BP_IP_lastN_diff','BP_xFIP_lastN_diff'}.issubset(out.columns):
        out['INT_BP_IPxQual_lastN'] = out['BP_IP_lastN_diff'] * out['BP_xFIP_lastN_diff']
    if {'BP_IP_d1_diff','BP_xFIP_lastN_diff'}.issubset(out.columns):
        out['INT_BP_YdayIPxQual'] = out['BP_IP_d1_diff'] * out['BP_xFIP_lastN_diff']

    # ---- Weather / park interactions
    if {'weather_wind_mph','Park_RunFactor_season'}.issubset(out.columns):
        out['INT_Wind__Park'] = out['weather_wind_mph'] * out['Park_RunFactor_season']
    if {'weather_temp_f','Park_RunFactor_season'}.issubset(out.columns):
        out['INT_Temp__Park'] = out['weather_temp_f'] * out['Park_RunFactor_season']

    # ---- Rest / travel fatigue interactions
    if {'Home_DaysSinceLast','Away_DaysSinceLast','Team_Elo_diff'}.issubset(out.columns):
        out['INT_RestDiff__Elo'] = (out['Home_DaysSinceLast'] - out['Away_DaysSinceLast']) * out['Team_Elo_diff']

    # ---- Simple nonlinear transforms that often help trees + logreg
    for c in ['SP_xFIP_diff','SP_FIP_diff','SP_KminusBB_diff',
              'Lineup_woba_vsHand_diff','Team_RunDiffpg_diff','Team_WinPct_diff',
              'Team_Elo_diff','BP_xFIP_lastN_diff','BP_xFIP_season_diff']:
        if c in out.columns:
            out[f'ABS_{c}'] = out[c].abs()          # magnitude of edge
            out[f'SQ_{c}']  = out[c] * out[c]       # gentle nonlinearity

    return out


def add_lineup_quality_extras(games: pd.DataFrame, pa_window_days: int = 7, top_k: int = 4) -> pd.DataFrame:
    df = games.copy()
    df['game_date'] = pd.to_datetime(df['game_date']).dt.date

    # 1) Count L/R batters in CONFIRMED lineup (you already collect lineup IDs)
    def _bat_side(pid):
        if pd.isna(pid): return None
        try:
            j = statsapi.get('people', {'personIds': int(pid)})
            return ((j.get('people') or [{}])[0].get('batSide') or {}).get('code')  # 'L','R','S'
        except Exception:
            return None

    # cache batter hand for speed
    _bat_side_cache = {}
    def bat_side_cached(pid):
        pid = int(pid)
        if pid not in _bat_side_cache:
            _bat_side_cache[pid] = _bat_side(pid)
        return _bat_side_cache[pid]

    # reuse your lineup detector
    lineup_map = {}
    for pk in df['game_pk'].dropna().astype(int).unique():
        lineup_map[int(pk)] = fetch_lineup_ids(int(pk))  # {'home':[pids], 'away':[pids]}

    home_L, home_R, away_L, away_R = [], [], [], []
    for _, r in df.iterrows():
        pk = int(r['game_pk'])
        lu = lineup_map.get(pk, {'home':[], 'away':[]})
        def _count_lr(pids):
            b = [bat_side_cached(pid) for pid in pids if pd.notna(pid)]
            return b.count('L') + b.count('S'), b.count('R')  # treat 'S' as L-advantage vs RHP
        # counts for facing OPPOSING SP hand (your lineup columns already use this logic)
        Lh, Rh = _count_lr(lu.get('home', []))
        La, Ra = _count_lr(lu.get('away', []))
        home_L.append(Lh); home_R.append(Rh)
        away_L.append(La); away_R.append(Ra)

    df['home_lineup_L_cnt'] = home_L
    df['home_lineup_R_cnt'] = home_R
    df['away_lineup_L_cnt'] = away_L
    df['away_lineup_R_cnt'] = away_R
    df['Lineup_L_cnt_diff']  = df['home_lineup_L_cnt'] - df['away_lineup_L_cnt']
    df['Lineup_R_cnt_diff']  = df['home_lineup_R_cnt'] - df['away_lineup_R_cnt']

    # 2) “Top-4 hitters present” (rank by last 60d wOBA across MLB, then count overlap)
    agg60 = statcast_window_agg(df['game_date'].max().strftime('%Y-%m-%d'), 60)
    # combine vs L/R into overall
    top_overall = (agg60.groupby('batter', as_index=False)[['woba_value','woba_denom']].sum())
    top_overall['woba'] = np.where(top_overall['woba_denom']>0,
                                   top_overall['woba_value']/top_overall['woba_denom'], np.nan)
    top_overall = top_overall.dropna(subset=['woba']).sort_values('woba', ascending=False)
    top_ids = set(top_overall.head(500)['batter'].astype(int))  # generous pool

    home_topk, away_topk = [], []
    for _, r in df.iterrows():
        pk = int(r['game_pk']); lu = lineup_map.get(pk, {'home':[], 'away':[]})
        home_topk.append(sum(int(pid) in top_ids for pid in lu.get('home', [])[:top_k]))
        away_topk.append(sum(int(pid) in top_ids for pid in lu.get('away', [])[:top_k]))
    df['home_top4_hitters'] = home_topk
    df['away_top4_hitters'] = away_topk
    df['Top4_hitters_diff'] = df['home_top4_hitters'] - df['away_top4_hitters']

    # 3) Injury/absence proxy: lineup PA in last 7d (sum of wOBA denominator)
    def _pa_window(end_date):
        start = (end_date - timedelta(days=pa_window_days)).strftime('%Y-%m-%d')
        stop  = (end_date - timedelta(days=1)).strftime('%Y-%m-%d')
        sc = statcast(start_dt=start, end_dt=stop)
        if sc is None or sc.empty:
            return pd.DataFrame(columns=['batter','PA'])
        g = sc.groupby('batter', as_index=False)['woba_denom'].sum().rename(columns={'woba_denom':'PA'})
        return g

    home_pa7, away_pa7 = [], []
    for d, pk in zip(df['game_date'], df['game_pk'].astype(int)):
        lu = lineup_map.get(pk, {'home':[], 'away':[]})
        pa7 = _pa_window(d)
        pa_map = dict(zip(pa7['batter'].astype(int), pa7['PA']))
        home_pa7.append(sum(pa_map.get(int(pid), 0.0) for pid in lu.get('home', [])))
        away_pa7.append(sum(pa_map.get(int(pid), 0.0) for pid in lu.get('away', [])))
    df['home_lineup_PA7'] = home_pa7
    df['away_lineup_PA7'] = away_pa7
    df['Lineup_PA7_diff'] = df['home_lineup_PA7'] - df['away_lineup_PA7']

    return df


def add_bullpen_highlev_and_closer(games: pd.DataFrame) -> pd.DataFrame:
    df = games.copy()
    df['game_date'] = pd.to_datetime(df['game_date'])

    # reuse starter exclusion map from your bullpen functions
    starters_map = _fetch_game_starters_bulk(df['game_pk'].dropna().astype(int).unique().tolist(), max_workers=12)

    # handy mapping for team key
    name_to_abbr = {}
    name_to_abbr.update(dict(df[['home_team_name','home_abbrev']].dropna().values)) if {'home_team_name','home_abbrev'}.issubset(df.columns) else None
    name_to_abbr.update(dict(df[['away_team_name','away_abbrev']].dropna().values)) if {'away_team_name','away_abbrev'}.issubset(df.columns) else None

    def _bp_window(start, stop):
        sc = statcast(start_dt=start, end_dt=stop)
        if sc is None or sc.empty:
            return pd.DataFrame(columns=['game_pk','team_key','inning','pitch_count'])
        keep = ['game_pk','home_team','away_team','inning_topbot','pitcher','events','bb_type','inning','pitch_number']
        sc = sc[[c for c in keep if c in sc.columns]].copy()
        sc['pitcher_team_name'] = np.where(sc['inning_topbot'].str.upper().eq('TOP'), sc['home_team'], sc['away_team'])
        sc['team_key'] = sc['pitcher_team_name'].map(name_to_abbr).fillna(sc['pitcher_team_name'])

        # exclude starters
        def _is_bp(row):
            pk = int(row['game_pk'])
            hsp, asp = starters_map.get(pk, (None, None))
            home_name = row['home_team'] if 'home_team' in row else None
            away_name = row['away_team'] if 'away_team' in row else None
            home_key  = name_to_abbr.get(home_name, home_name)
            away_key  = name_to_abbr.get(away_name, away_name)
            if row['team_key'] == home_key and hsp is not None and pd.notna(row['pitcher']):
                return int(row['pitcher']) != int(hsp)
            if row['team_key'] == away_key and asp is not None and pd.notna(row['pitcher']):
                return int(row['pitcher']) != int(asp)
            return True

        sc = sc[sc.apply(_is_bp, axis=1)]
        if sc.empty:
            return sc

        # pitch_count by pitcher per (team, day)
        sc['pitch_count'] = 1
        return sc

    # allocate outputs
    df['home_BP_HiLevIP_d3'] = np.nan
    df['away_BP_HiLevIP_d3'] = np.nan
    df['BP_HiLevIP_d3_diff'] = np.nan
    df['home_CloserAvail'] = np.nan
    df['away_CloserAvail'] = np.nan

    grouped = df.groupby(df['game_date'].dt.date).groups
    for dt, idx in grouped.items():
        end = pd.to_datetime(dt)
        start3 = (end - timedelta(days=3)).strftime('%Y-%m-%d'); stop3 = (end - timedelta(days=1)).strftime('%Y-%m-%d')
        start1 = (end - timedelta(days=1)).strftime('%Y-%m-%d');  stop1 = (end - timedelta(days=1)).strftime('%Y-%m-%d')

        sc3 = _bp_window(start3, stop3)
        sc1 = _bp_window(start1, stop1)

        # high-leverage proxy: 8th–9th inning IP for bullpen
        hi = sc3[sc3.get('inning', 0).astype(float) >= 8]
        if not hi.empty:
            outs = _outs_from_events(hi['events'].astype(str))
            hi['_OUTS'] = outs
            ip_by_team = (hi.groupby('team_key')['_OUTS'].sum() / 3.0).rename('IP').reset_index()
        else:
            ip_by_team = pd.DataFrame(columns=['team_key','IP'])

        # “Closer availability” proxy: did ANY reliever throw ≥25 pitches yesterday?
        close_flag = {}
        if not sc1.empty:
            pc = (sc1.groupby(['team_key','pitcher'])['pitch_count'].sum().reset_index())
            for (t, sub) in pc.groupby('team_key'):
                close_flag[t] = int(sub['pitch_count'].max() >= 25)

        for i in idx:
            hkey = name_to_abbr.get(df.at[i,'home_team_name'], df.at[i,'home_team_name'])
            akey = name_to_abbr.get(df.at[i,'away_team_name'], df.at[i,'away_team_name'])
            hIP = float(ip_by_team[ip_by_team['team_key']==hkey]['IP'].sum()) if not ip_by_team.empty else 0.0
            aIP = float(ip_by_team[ip_by_team['team_key']==akey]['IP'].sum()) if not ip_by_team.empty else 0.0
            df.at[i,'home_BP_HiLevIP_d3'] = hIP
            df.at[i,'away_BP_HiLevIP_d3'] = aIP
            df.at[i,'BP_HiLevIP_d3_diff'] = hIP - aIP
            df.at[i,'home_CloserAvail'] = 1 - close_flag.get(hkey, 0)  # 1 if available
            df.at[i,'away_CloserAvail'] = 1 - close_flag.get(akey, 0)

    return df

def add_park_hr_factor(games: pd.DataFrame, season_start: str | None = None, prior_games: int = 300) -> pd.DataFrame:
    """
    Compute pre-game, season-to-date HR park factor using Statcast HR events.
    - No need for 'home_HR' / 'away_HR' columns.
    - Shrinks venue HR/game toward league HR/game with `prior_games`.
    - Returns a copy of `games` with 'Park_HRFactor_season' added.
    """
    df = games.copy()
    if df.empty:
        df['Park_HRFactor_season'] = np.nan
        return df

    # Normalize dates
    df['game_date'] = pd.to_datetime(df['game_date'])
    df = df.sort_values('game_date').reset_index(drop=True)
    df['_season'] = df['game_date'].dt.year
    df['gdate']   = df['game_date'].dt.normalize()

    # Determine Statcast window: from season start to day BEFORE max game date
    max_dt = df['game_date'].max().date()
    end_for_pull = (max_dt - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    if season_start is None:
        season_start = f"{int(df['_season'].min())}-03-01"

    if pd.to_datetime(season_start) > pd.to_datetime(end_for_pull):
        df['Park_HRFactor_season'] = np.nan
        return df

    sc = statcast(start_dt=season_start, end_dt=end_for_pull)
    if sc is None or sc.empty:
        df['Park_HRFactor_season'] = np.nan
        return df

    # Keep only what we need
    keep = ['game_pk','game_date','events']
    if 'venue_id' in sc.columns:
        keep.append('venue_id')
    sc = sc[[c for c in keep if c in sc.columns]].copy()

    # If Statcast didn’t include venue_id, merge it from your games map (game_pk → venue_id)
    if 'venue_id' not in sc.columns:
        vid_map = (df[['game_pk','venue_id']]
                   .dropna()
                   .drop_duplicates()
                   .astype({'game_pk': int, 'venue_id': int}))
        sc = sc.merge(vid_map, on='game_pk', how='left')

    if 'venue_id' not in sc.columns or sc['venue_id'].isna().all():
        # Can't compute without venues; bail gracefully
        df['Park_HRFactor_season'] = np.nan
        return df

    # Build daily HR counts by venue and league (pre-game cumulatives)
    sc['game_date'] = pd.to_datetime(sc['game_date']).dt.normalize()
    sc['HR'] = (sc['events'].astype(str) == 'home_run').astype(int)
    sc = sc.dropna(subset=['venue_id'])
    sc['venue_id'] = sc['venue_id'].astype(int)

    # Venue-by-day totals
    vday = (sc.groupby(['venue_id','game_date'], as_index=False)['HR']
              .sum()
              .rename(columns={'game_date':'gdate','HR':'venue_HR'}))
    vday['_season'] = vday['gdate'].dt.year

    # League-by-day totals (sum across venues)
    lg = (vday.groupby(['_season','gdate'], as_index=False)['venue_HR']
                .sum()
                .rename(columns={'venue_HR':'lg_HR'}))

    # How many MLB games that day? Use your games frame
    lg_games = (df.groupby(['_season','gdate'], as_index=False)['game_pk']
                  .nunique()
                  .rename(columns={'game_pk':'lg_games'}))

    lg = lg.merge(lg_games, on=['_season','gdate'], how='left').sort_values(['_season','gdate'])
    lg['lg_HR_cum']   = lg.groupby('_season')['lg_HR'].cumsum().shift(1)
    lg['lg_g_cum']    = lg.groupby('_season')['lg_games'].cumsum().shift(1)
    lg['lg_HRpg_pre'] = np.where(lg['lg_g_cum'] > 0, lg['lg_HR_cum'] / lg['lg_g_cum'], np.nan)

    # Venue pre-game HR/game with shrinkage toward league
    v = vday.sort_values(['venue_id','_season','gdate']).copy()
    v['venue_HR_cum'] = v.groupby(['venue_id','_season'])['venue_HR'].cumsum().shift(1)
    v['venue_g_cum']  = v.groupby(['venue_id','_season']).cumcount()
    v = v.merge(lg[['_season','gdate','lg_HRpg_pre']], on=['_season','gdate'], how='left')

    v['venue_HRpg_shrunk'] = np.where(
        (v['venue_g_cum'] > 0) & v['lg_HRpg_pre'].notna(),
        (v['venue_HR_cum'] + prior_games * v['lg_HRpg_pre']) / (v['venue_g_cum'] + prior_games),
        np.nan
    )
    v['Park_HRFactor_season'] = np.where(
        v['lg_HRpg_pre'] > 0,
        v['venue_HRpg_shrunk'] / v['lg_HRpg_pre'],
        np.nan
    )

    # Attach back to games by (venue_id, season, date)
    pf_map = pd.Series(
        v['Park_HRFactor_season'].values,
        index=pd.MultiIndex.from_frame(v[['venue_id','_season','gdate']])
    )

    df['Park_HRFactor_season'] = [
        pf_map.get((vid, yr, gd), np.nan)
        for vid, yr, gd in zip(df['venue_id'], df['_season'], df['gdate'])
    ]
    return df




def add_wind_interactions_and_sp_fb(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # wind direction to simple flags
    d = out.get('weather_wind_dir').astype(str).str.lower()
    out['Wind_Out_flag'] = d.str.contains('out').astype(float)
    out['Wind_In_flag']  = d.str.contains('in').astype(float)

    # SP fly-ball rate (you already count FB & BF inside _pitcher_window_stats; expose it)
    # If not present, fill basic proxies as NA to keep pipeline going
    for col in ['home_SP_FB_rate','away_SP_FB_rate']:
        if col not in out.columns:
            out[col] = np.nan
    out['SP_FB_rate_diff'] = out['home_SP_FB_rate'] - out['away_SP_FB_rate']

    # interactions
    if 'Park_HRFactor_season' in out.columns:
        out['INT_WindOut__ParkHR'] = out['Wind_Out_flag'] * out['Park_HRFactor_season']
        out['INT_WindIn__ParkHR']  = out['Wind_In_flag']  * out['Park_HRFactor_season']
    out['INT_WindOut__SPFB'] = out['Wind_Out_flag'] * out['SP_FB_rate_diff']
    return out

def add_travel_schedule_load(games: pd.DataFrame) -> pd.DataFrame:
    df = games.copy()
    df['game_date'] = pd.to_datetime(df['game_date']).dt.date

    # venue -> timezone
    tz_map = {}
    for vid in pd.Series(df['venue_id']).dropna().astype(int).unique():
        try:
            v = statsapi.get('venues', {'venueIds': int(vid)})
            tz = ((v.get('venues') or [{}])[0].get('timeZone') or {}).get('tz')
            tz_map[int(vid)] = tz
        except Exception:
            tz_map[int(vid)] = None

    df['venue_tz'] = df['venue_id'].map(tz_map)

    # build chronological per team schedule with tz
    def _per_team_flags(team_col):
        rows = []
        for tm, g in df.groupby(team_col):
            g = g.sort_values('game_date').copy()
            prev_date = None
            prev_tz   = None
            rolling3  = []  # last two gaps (in days)
            first_after_off = []
            tz_change = []
            g_in3 = []
            for _, r in g.iterrows():
                d = r['game_date']; tz = r['venue_tz']
                gap = (d - prev_date).days if prev_date else np.nan
                first_after_off.append(int(gap == 2))  # exactly one off-day
                tz_change.append(int(prev_tz is not None and tz is not None and tz != prev_tz))
                # 3in3: have we played on each of prior 2 days?
                rolling3.append(0 if pd.isna(gap) else int(gap == 1))
                if len(rolling3) >= 3:
                    g_in3.append(int(rolling3[-1] == 1 and rolling3[-2] == 1))
                else:
                    g_in3.append(0)
                prev_date, prev_tz = d, tz
            g[f'{team_col}_FirstAfterOff'] = first_after_off
            g[f'{team_col}_TZChange'] = tz_change
            g[f'{team_col}_3in3'] = g_in3
            rows.append(g[[team_col,'game_pk',f'{team_col}_FirstAfterOff',f'{team_col}_TZChange',f'{team_col}_3in3']])
        return pd.concat(rows, ignore_index=True)

    home_flags = _per_team_flags('home_team_name')
    away_flags = _per_team_flags('away_team_name')

    df = df.merge(home_flags, on=['game_pk','home_team_name'], how='left')
    df = df.merge(away_flags, on=['game_pk','away_team_name'], how='left')

    # centered diffs
    df['TZChange_diff']       = df['home_team_name_TZChange'] - df['away_team_name_TZChange']
    df['FirstAfterOff_diff']  = df['home_team_name_FirstAfterOff'] - df['away_team_name_FirstAfterOff']
    df['ThreeInThree_diff']   = df['home_team_name_3in3'] - df['away_team_name_3in3']
    return df

def _ingest_year_to_raw(
    year: int,
    backfill_days: int = 35,
    out_path: str = "out/raw_games.csv",
    include_finished: bool = True,
    fetch_handedness: bool = True,
    max_workers: int = 20,
) -> tuple[str, str]:
    """
    Fetch schedule+probables for the given YEAR, including a warmup window that
    starts `backfill_days` before Jan 1. Merge & dedupe into `out/raw_games.csv`.
    Keeps only rows within [year_start - backfill_days, Dec 31 year].
    Returns (year_start_str, year_end_str) for downstream feature building.

    Requires your file to define:
      fetch_games_with_probables_statsapi_fast(start_date, end_date, include_finished, fetch_handedness, max_workers, debug)
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    year_start = datetime(year, 1, 1).date()
    year_end   = datetime(year, 12, 31).date()
    warmup_start = (year_start - timedelta(days=backfill_days))

    start_date = warmup_start.strftime("%Y-%m-%d")
    end_date   = year_end.strftime("%Y-%m-%d")

    print(f"📥 Ingesting YEAR {year} with warmup → {start_date} to {end_date}")

    # Pull once for the whole span (warmup → Dec 31)
    new_df = fetch_games_with_probables_statsapi_fast(
        start_date=start_date,
        end_date=end_date,
        include_finished=include_finished,
        fetch_handedness=fetch_handedness,
        max_workers=max_workers,
        debug=False,
    )

    # Load existing (if any)
    if os.path.exists(out_path):
        existing = pd.read_csv(out_path)
        # Normalize date columns for consistent filtering/merge
        for c in ("game_date", "game_datetime"):
            if c in existing.columns:
                existing[c] = pd.to_datetime(existing[c], errors="coerce")
    else:
        existing = pd.DataFrame()

    before = len(existing)

    # Merge & dedupe by game_pk
    if not existing.empty:
        all_df = pd.concat([existing, new_df], ignore_index=True)
    else:
        all_df = new_df.copy()

    if "game_pk" in all_df.columns:
        # Sort so the latest API record per game wins
        sort_cols = [c for c in ["game_pk", "game_datetime"] if c in all_df.columns]
        if sort_cols:
            all_df = all_df.sort_values(sort_cols, na_position="last")
        all_df = all_df.drop_duplicates(subset=["game_pk"], keep="last")
    else:
        all_df = all_df.drop_duplicates()

    # Keep only warmup + target year rows
    if "game_date" in all_df.columns:
        all_df["game_date"] = pd.to_datetime(all_df["game_date"], errors="coerce").dt.date
        mask_keep = (all_df["game_date"] >= warmup_start) & (all_df["game_date"] <= year_end)
        all_df = all_df.loc[mask_keep].copy()

    # Progress message
    added = len(all_df) - before
    if added <= 0:
        print(f"[ingest] No new games to add. Total still {len(all_df):,} rows.")
    else:
        print(f"[ingest] {added:,} new games added (total {len(all_df):,} rows).")

    # Save
    all_df.to_csv(out_path, index=False)
    print(f"[ingest] Saved → {out_path}")

    return (year_start.strftime("%Y-%m-%d"), year_end.strftime("%Y-%m-%d"))


# ========= EXPORTS for pipeline (append to bottom of data_ingest.py) =========

def build_raw_dataset_year(
    year: int,
    backfill_days: int = 35,
    out_csv: str = "out/raw_games.csv",
    include_finished: bool = True,
    fetch_handedness: bool = True,
    max_workers: int = 20,
):
    """
    Thin wrapper around _ingest_year_to_raw that returns (start, end, out_csv).
    Keeps your warmup window logic intact.
    """
    start, end = _ingest_year_to_raw(
        year=year,
        backfill_days=backfill_days,
        out_path=out_csv,
        include_finished=include_finished,
        fetch_handedness=fetch_handedness,
        max_workers=max_workers,
    )
    return start, end, out_csv


# ========= PUBLIC FEATURE BUILDER (called by run_pipeline.py) =========
def build_feature_table(
    start: str,
    end: str,
    out_csv: str = "out/_tmp_features.csv",
    verbose: bool = False,
    max_workers: int = 12,
) -> str:
    """
    Build a leak-safe feature table for the inclusive date window [start, end].
    Writes CSV to `out_csv` and returns that path.

    The function reuses your existing context and rolling/season blocks.
    """
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    if verbose:
        print(f"[build_feature_table] window: {start} → {end}")
        print("[build_feature_table] fetching base games from StatsAPI …")

    # --- Base games for window (includes scores for finished games)
    games = fetch_games_with_probables_statsapi_fast(
        start_date=start,
        end_date=end,
        include_finished=True,
        fetch_handedness=True,
        max_workers=max_workers,
        debug=False,
    )
    if games is None or games.empty:
        # Write empty but valid CSV to keep the pipeline flowing
        pd.DataFrame(columns=["game_pk", "game_date"]).to_csv(out_csv, index=False)
        if verbose: print("[build_feature_table] no games in window; wrote empty CSV")
        return out_csv

    # =========================
    # CONTEXT / FEATURE BLOCKS
    # =========================
    if verbose: print("[build_feature_table] adding context features …")
    games = add_sp_rest_features(games)
    games = add_bullpen_rest_snapshots(games)
    games = add_basic_weather(games)
    games = add_plate_umpire(games)
    games = add_schedule_fatigue(games)
    games = add_bullpen_highlev_and_closer(games)
    games = add_lineup_quality_extras(games, pa_window_days=7, top_k=4)
    games = add_park_hr_factor(games)
    games = add_travel_schedule_load(games)

    if verbose: print("[build_feature_table] lineup wOBA 15d window …")
    games = add_lineup_woba_features(games, window_days=15)

    if verbose: print("[build_feature_table] SP Statcast rolling 30d …")
    games = add_sp_rolling_features_savant(games, window_days=30, verbose=verbose)
    games = add_wind_interactions_and_sp_fb(games)

    if verbose: print("[build_feature_table] bullpen rolling (IP & xFIP) …")
    games = add_bullpen_rolling_features(games, window_days=30, lg_hr_per_fb=0.105, max_workers=max_workers)

    if verbose: print("[build_feature_table] bullpen season-to-date xFIP …")
    games = add_bullpen_season_xfip_features(games, season_start=None, lg_hr_per_fb=0.105)

    if verbose: print("[build_feature_table] park run factor (season shrunk) …")
    games = add_season_park_factor_features(games, prior_games=300, friendly_hi=1.005, friendly_lo=0.995)

    if verbose: print("[build_feature_table] lineup wOBA season-to-date vs opp hand …")
    games = add_lineup_woba_season_features(games, season_start=None)

    if verbose: print("[build_feature_table] team rolling/Elo …")
    games_for_rolling = games.copy()
    games_for_rolling['game_id']   = games_for_rolling['game_pk']
    games_for_rolling['home_team'] = games_for_rolling.get('home_abbrev', games_for_rolling['home_team_name'])
    games_for_rolling['away_team'] = games_for_rolling.get('away_abbrev', games_for_rolling['away_team_name'])
    games_for_rolling['game_date']  = pd.to_datetime(games_for_rolling['game_date'])
    games_for_rolling['home_score'] = pd.to_numeric(games_for_rolling['home_score'], errors='coerce')
    games_for_rolling['away_score'] = pd.to_numeric(games_for_rolling['away_score'], errors='coerce')

    team_feats = add_team_rolling_features(
        games_for_rolling[['game_id','game_date','home_team','away_team','home_score','away_score']].copy(),
        last_n=10, include_elo=True, elo_K=20, elo_HFA=55
    )
    cols_to_keep = [
        'game_id',
        'Team_RunDiffpg_diff','Team_WinPct_diff','Team_LastN_WinPct_diff','Team_Elo_diff',
        'Home_STG_RunDiffpg','Away_STG_RunDiffpg','Home_STG_WinPct','Away_STG_WinPct',
        'Home_LastN_WinPct','Away_LastN_WinPct'
    ]
    games_all = games.merge(team_feats[cols_to_keep], left_on='game_pk', right_on='game_id', how='left')

    # Final curated interactions (uses only pre-game inputs)
    final_df = add_interaction_features(games_all)

    # Persist
    # Normalize datetimes for stable CSV
    for col in ['game_date', 'game_datetime']:
        if col in final_df.columns:
            final_df[col] = pd.to_datetime(final_df[col], errors='coerce')
    final_df.to_csv(out_csv, index=False)
    if verbose:
        print(f"[build_feature_table] wrote {len(final_df):,} rows × {final_df.shape[1]} cols → {out_csv}")
    return out_csv

# Optional alias so other code names still work
build_features = build_feature_table

# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2025,
                        help="Target MLB season (default 2025).")
    parser.add_argument("--backfill-days", type=int, default=35,
                        help="Warmup days before Jan 1 to stabilize rolling features.")
    parser.add_argument("--only-ingest", action="store_true",
                        help="Only update out/raw_games.csv and exit.")
    parser.add_argument("--raw-path", default="out/raw_games.csv",
                        help="Where to store merged raw games CSV.")
    args = parser.parse_args()

    # ---- Step 1: ingest year (with warmup) into out/raw_games.csv
    y_start, y_end = _ingest_year_to_raw(
        year=args.year,
        backfill_days=args.backfill_days,
        out_path=args.raw_path,
        include_finished=True,         # include completed games
        fetch_handedness=True
    )

    if args.only_ingest:
        print("✅ Ingestion complete. Exiting (--only-ingest).")
        raise SystemExit(0)

    # ---- Step 2: build features for *that year only* (the warmup rows are for lookbacks)
    start_date = y_start
    end_date   = y_end
    print(f"📅 Feature build range (year {args.year}): {start_date} → {end_date}")

    # You already have a rich fetch function; use it to create the base 'games' frame
    games = fetch_games_with_probables_statsapi_fast(
        start_date=start_date,
        end_date=end_date,
        include_finished=True,
        fetch_handedness=True,
        max_workers=20,
        debug=False
    )
    print(f"✅ Fetched {len(games)} games for features")

    # =========================
    # CONTEXT / FEATURE BLOCKS
    # (unchanged from your code)
    # =========================
    print("\n" + "="*60)
    print("ADDING CONTEXTUAL FEATURES (Rest, Weather, Umpire, Fatigue)...")
    print("="*60)
    games = add_sp_rest_features(games)
    games = add_bullpen_rest_snapshots(games)
    games = add_basic_weather(games)
    games = add_plate_umpire(games)
    games = add_schedule_fatigue(games)
    games = add_bullpen_highlev_and_closer(games)
    games = add_lineup_quality_extras(games, pa_window_days=7, top_k=4)
    games = add_park_hr_factor(games)
    games = add_travel_schedule_load(games)

    print("\n" + "="*60)
    print("ADDING LINEUP WOBA FEATURES...")
    print("="*60)
    games = add_lineup_woba_features(games, window_days=15)

    print("\n" + "="*60)
    print("ADDING SP ROLLING FEATURES FROM STATCAST...")
    print("="*60)
    games_with_sp = add_sp_rolling_features_savant(games, window_days=30, verbose=True)
    games_with_sp = add_wind_interactions_and_sp_fb(games_with_sp)

    print("\n" + "="*60)
    print("ADDING BULLPEN ROLLING FEATURES (IP & xFIP)...")
    print("="*60)
    games_with_sp = add_bullpen_rolling_features(
        games_with_sp, window_days=30, lg_hr_per_fb=0.105, max_workers=12
    )

    print("\n" + "="*60)
    print("ADDING BULLPEN SEASON-TO-DATE xFIP...")
    print("="*60)
    games_with_sp = add_bullpen_season_xfip_features(
        games_with_sp, season_start=None, lg_hr_per_fb=0.105
    )

    print("\n" + "="*60)
    print("ADDING PARK FACTOR (SEASON-TO-DATE, SHRUNK)...")
    print("="*60)
    games_with_sp = add_season_park_factor_features(
        games_with_sp, prior_games=300, friendly_hi=1.005, friendly_lo=0.995
    )

    print("\n" + "="*60)
    print("ADDING LINEUP WOBA SEASON-TO-DATE (vs Opposing Hand)...")
    print("="*60)
    games_with_sp = add_lineup_woba_season_features(games_with_sp, season_start=None)

    print("\n" + "="*60)
    print("ADDING TEAM ROLLING FEATURES (ELO, WIN%, RUN DIFF)...")
    print("="*60)
    games_for_rolling = games_with_sp.copy()
    games_for_rolling['game_id']   = games_for_rolling['game_pk']
    games_for_rolling['home_team'] = games_for_rolling.get('home_abbrev', games_for_rolling['home_team_name'])
    games_for_rolling['away_team'] = games_for_rolling.get('away_abbrev', games_for_rolling['away_team_name'])
    games_for_rolling['game_date']  = pd.to_datetime(games_for_rolling['game_date'])
    games_for_rolling['home_score'] = pd.to_numeric(games_for_rolling['home_score'], errors='coerce')
    games_for_rolling['away_score'] = pd.to_numeric(games_for_rolling['away_score'], errors='coerce')

    team_feats = add_team_rolling_features(
        games_for_rolling[['game_id','game_date','home_team','away_team','home_score','away_score']].copy(),
        last_n=10, include_elo=True, elo_K=20, elo_HFA=55
    )
    cols_to_keep = [
        'game_id',
        'Team_RunDiffpg_diff','Team_WinPct_diff','Team_LastN_WinPct_diff','Team_Elo_diff',
        'Home_STG_RunDiffpg','Away_STG_RunDiffpg','Home_STG_WinPct','Away_STG_WinPct',
        'Home_LastN_WinPct','Away_LastN_WinPct'
    ]
    games_all = games_with_sp.merge(team_feats[cols_to_keep], left_on='game_pk', right_on='game_id', how='left')

    # Final interactions
    final_df = add_interaction_features(games_all)

    # ===== Save outputs (year-stamped) =====
    os.makedirs("out", exist_ok=True)
    rng = f"{args.year}_season"
    full_csv = f"out/mlb_features_full_{rng}.csv"
    mini_csv = f"out/mlb_features_model_{rng}.csv"

    final_df_to_save = final_df.copy()
    for col in ['game_date','game_datetime']:
        if col in final_df_to_save.columns:
            final_df_to_save[col] = pd.to_datetime(final_df_to_save[col], errors='coerce')
    final_df_to_save.to_csv(full_csv, index=False)
    print(f"\n💾 Saved full dataset to: {full_csv}")

    drop_cols = [
        'game_pk','game_datetime','venue_name',
        'home_team_name','away_team_name',
        'home_sp_name','away_sp_name',
        'home_score','away_score','status'
    ]
    model_df = final_df_to_save.drop(columns=[c for c in drop_cols if c in final_df_to_save.columns], errors='ignore')
    model_df.to_csv(mini_csv, index=False)
    print(f"💾 Saved modeling subset to: {mini_csv}")
