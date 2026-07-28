#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
international_predictions.py

Extends the Dixon-Coles prediction pipeline to international/cup
competitions that football-data.co.uk doesn't cover (it only has domestic
leagues): UEFA Champions League, FIFA World Cup, UEFA European Championship.

Data source: football-data.org v4 API. Free tier covers 12 competitions
including these 3, gives fixtures/results/standings, 10 requests/minute.
Requires a free API token (sign up at https://www.football-data.org/client/register)
set as the FOOTBALL_DATA_ORG_TOKEN environment variable / GitHub secret.
NOTE: the free tier does NOT include odds. Odds for these 3 competitions
come from odds_client.py (The Odds API) instead, merged in by team name via
team_utils.py in predictions_merger.py's odd_addition() -- see that file.

Reuses the core Dixon-Coles modeling/output logic from
majorleague_predictions.py (dixon_coles_simulate_match, solve_parameters_decay,
resultdef, calculate_win_and_goal_form, calc_standings, load/save_cached_params)
instead of duplicating ~500 lines of proven, tested code (including the bet
builder combo math). resultdef() internally references a couple of
module-level globals (path, historyfunc) that only get set when
majorleague_predictions.py runs as __main__ -- since we're importing it
instead, those are patched below to route through this script's own
history logic. This is intentional, not a workaround for a bug --
see the "Patch mlp's internals" section below for exactly what and why.

The football-data.org API access itself (auth, retries, rate limiting,
match-object parsing) lives in football_data_org_client.py, shared with
check_fixtures.py and update_results.py so all three scripts fetch/parse
matches identically.
"""

import os
import datetime
import pandas as pd
import numpy as np

import majorleague_predictions as mlp
import football_data_org_client as fdo
import date_utils
import market_odds
import model_config
import odds_client
import team_utils

COMPETITIONS = fdo.COMPETITIONS

DATAPATH = 'predictions_data/'
# One file per run, named for the single betting day it covers -- see
# majorleague_predictions.py.
DATANAME = 'international_prediction_{date}'

HISTORY_SEASONS_BACK = 4  # how many prior seasons to try pulling for model fitting
MIN_HISTORY_MATCHES = 20  # below this, Dixon-Coles fitting isn't meaningful


def fetch_historical_matches(code, seasons_back=HISTORY_SEASONS_BACK):
    return fdo.fetch_historical_matches(code, seasons_back=seasons_back)


def fetch_today_window_matches(code):
    """Matches within today's fetch window (today from the 09:00 cutoff
    onward, plus tomorrow up to 09:00 -- see date_utils.py). Previously
    fetched only tomorrow's exact calendar date, which excluded today's
    own matches entirely and included all of tomorrow's regardless of
    kickoff time. One ranged request covers both calendar days --
    football-data.org's dateFrom/dateTo filter is whole-day only, so
    date_utils narrows the result down to the actual time-of-day cutoff
    afterward."""
    today = datetime.date.today()
    tomorrow = today + datetime.timedelta(days=1)
    matches = fdo.fetch_matches(code, date_from=today.isoformat(), date_to=tomorrow.isoformat(),
                                 status="SCHEDULED")
    df = fdo.matches_to_df(matches)
    if df.empty:
        return df
    return df[date_utils.in_fetch_window(df['Date'], df['Time'])]


def historyfunc_international(hist_df, hw, aw):
    """Self-contained head-to-head stats, replacing majorleague_predictions'
    historyfunc() (which is tied to football-data.co.uk's URL/season-swap
    pattern and doesn't apply to the football-data.org API). Matches the
    dict shape resultdef() expects to look up by prediction code.

    Simplification vs. the original historyfunc: this looks at H2H meetings
    regardless of historical venue (since international fixtures are far
    less frequent than domestic ones -- venue-split H2H would mostly be
    empty). Good enough for a supplementary display stat; the actual
    prediction probability still comes from Dixon-Coles, not this.
    """
    h2h = hist_df[
        ((hist_df['HomeTeam'] == hw) & (hist_df['AwayTeam'] == aw)) |
        ((hist_df['HomeTeam'] == aw) & (hist_df['AwayTeam'] == hw))
    ]
    n = len(h2h)
    if n == 0:
        return {}

    def pct(cond):
        return round(100 * cond.sum() / n, 1)

    is_hw_home = h2h['HomeTeam'] == hw
    hw_goals = h2h['HomeGoals'].where(is_hw_home, h2h['AwayGoals'])
    aw_goals = h2h['AwayGoals'].where(is_hw_home, h2h['HomeGoals'])
    total_goals = h2h['HomeGoals'] + h2h['AwayGoals']

    return {
        '1': pct(hw_goals > aw_goals),
        'X': pct(hw_goals == aw_goals),
        '2': pct(hw_goals < aw_goals),
        'O1_5': pct(total_goals > 1), 'O2_5': pct(total_goals > 2), 'O3_5': pct(total_goals > 3),
        'GG': pct((h2h['HomeGoals'] > 0) & (h2h['AwayGoals'] > 0)),
        'hO1_5': pct(hw_goals > 1), 'hO2_5': pct(hw_goals > 2),
        'aO1_5': pct(aw_goals > 1), 'aO2_5': pct(aw_goals > 2),
    }


def save_results_(df):
    if df.empty:
        return
    if not os.path.exists(DATAPATH):
        os.makedirs(DATAPATH)
    filename = DATAPATH + '/' + DATANAME

    if os.path.exists(filename):
        temp = pd.read_csv(filename)
        towrite = pd.concat([temp, df])
    else:
        towrite = df

    towrite.to_csv(filename, index=False)


def patch_mlp_internals(hist_df, competition_code):
    """resultdef() reaches for `path` and `historyfunc` as module-level
    globals in majorleague_predictions.py's own namespace -- those only get
    set there when that file runs as __main__. Since we're importing it
    instead, point them at our own equivalents so resultdef's internal
    history calls work correctly without duplicating its ~130 lines
    (including the bet-builder combo math) here.
    """
    mlp.path = competition_code  # placeholder value; our historyfunc ignores it

    def _bound_historyfunc(path, hw, aw):
        return historyfunc_international(hist_df, hw, aw)

    mlp.historyfunc = _bound_historyfunc


def run_competition(name, code, odds_lookup=None, calibration=None):
    print(f"Checking upcoming fixtures for {name} ({code})..")
    next_match = fetch_today_window_matches(code)
    if next_match.empty:
        print(f"No fixtures in today's window for {name}.. skipping")
        return pd.DataFrame()

    print(f"Fetching historical results for {name}..")
    hist_df = fetch_historical_matches(code)
    if len(hist_df) < MIN_HISTORY_MATCHES:
        print(f"WARNING: Not enough history for {name} ({len(hist_df)} matches).. skipping")
        return pd.DataFrame()

    try:
        standings_df = mlp.calc_standings(hist_df)
    except Exception as e:
        print(f"ERROR: Error calculating standings for {name}..", e)
        return pd.DataFrame()

    try:
        teams_sorted = np.sort(hist_df['HomeTeam'].unique())
        warm_start = mlp.load_cached_params(code, teams_sorted)
        params = mlp.solve_parameters_decay(hist_df, init_vals=warm_start)
        mlp.save_cached_params(code, params)
    except Exception as e:
        print(f"ERROR: Error fitting parameters for {name}..", e)
        return pd.DataFrame()

    patch_mlp_internals(hist_df, code)

    # Whole-competition computation -- once, not once per fixture.
    try:
        form_df = mlp.calculate_win_and_goal_form(hist_df)
    except Exception as e:
        print(f"WARNING: Could not compute form for {name}..", e)
        form_df = None

    frames = []
    for _, row in next_match.iterrows():
        ht, at = row['HomeTeam'], row['AwayTeam']
        try:
            result = mlp.dixon_coles_simulate_match(params, ht, at)
        except Exception as e:
            print(f"ERROR: Issue simulating {ht} vs {at} ({name})", e)
            continue

        odds_row = market_odds.lookup_odds(odds_lookup, ht, at, team_utils.normalize)
        res = mlp.resultdef(result, ht, at, code, row['Date'], row['Time'], standings_df, hist_df,
                            odds_row=odds_row, form_df=form_df, calibration=calibration)
        if not res.empty:
            frames.append(res)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    # football-data.co.uk has no international odds at all, so unlike the
    # domestic scripts (where prices ride along with the fixture file) these
    # come from The Odds API. A failure here is non-fatal: predictions still
    # export, just unpriced.
    try:
        odds_df = odds_client.fetch_all_international_odds()
    except Exception as e:
        print("WARNING: Could not fetch international odds.. exporting unpriced.", e)
        odds_df = market_odds.empty_odds_frame()
    odds_lookup = market_odds.build_lookup(odds_df, team_utils.normalize)
    print(f"International odds available for {len(odds_lookup)} fixtures.")

    calibration = model_config.load_calibration()

    frames = []
    for name, code in COMPETITIONS.items():
        try:
            res = run_competition(name, code, odds_lookup=odds_lookup, calibration=calibration)
            if not res.empty:
                frames.append(res)
        except Exception as e:
            print(f"ERROR: Unhandled error processing {name} ({code})..", e)
            continue

    if not frames:
        print("No international predictions generated today.")
        return

    results_df = pd.concat(frames, ignore_index=True)
    print(f"Saving {len(results_df)} international prediction rows..")
    save_results_(results_df)


if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__) or '.')

    # The fetch window is anchored on today, so the betting day it covers
    # is today by construction -- no need to inspect the fixtures first the
    # way the domestic scripts do.
    DATANAME = DATANAME.replace('{date}', datetime.date.today().strftime('%Y-%m-%d')) + '.csv'

    try:
        main()
    except Exception as e:
        print("CRITICAL: Unhandled exception in international_predictions.py", e)
        raise
