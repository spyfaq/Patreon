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
builder combo math). resultdef() internally references a few module-level
globals (logger, path, historyfunc) that only get set when
majorleague_predictions.py runs as __main__ -- since we're importing it
instead, those are patched below to route through this script's own
logger/history logic. This is intentional, not a workaround for a bug --
see the "Patch mlp's internals" section below for exactly what and why.

The football-data.org API access itself (auth, retries, rate limiting,
match-object parsing) lives in football_data_org_client.py, shared with
check_fixtures.py and update_results.py so all three scripts fetch/parse
matches identically.
"""

import os
import sys
import datetime
import pandas as pd
import numpy as np

import majorleague_predictions as mlp
import football_data_org_client as fdo
from jsonlogger_class import JSONLogger

COMPETITIONS = fdo.COMPETITIONS

DATAPATH = 'predictions_data/'
DATANAME = 'my_prediction_international_data_{date1}_{date2}'
LOGPATH = 'logs/simu/'
LOGNAME = '{date}_international_logs'

HISTORY_SEASONS_BACK = 4  # how many prior seasons to try pulling for model fitting
MIN_HISTORY_MATCHES = 20  # below this, Dixon-Coles fitting isn't meaningful


def fetch_historical_matches(code, seasons_back=HISTORY_SEASONS_BACK):
    return fdo.fetch_historical_matches(code, seasons_back=seasons_back, logger=logger)


def fetch_tomorrow_matches(code):
    """Tomorrow's scheduled matches only -- matches the 1-day window cap
    the rest of the pipeline now uses (see majorleague_predictions.py)."""
    tomorrow = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()
    return fdo.fetch_matches_on_date(code, tomorrow, status="SCHEDULED", logger=logger)


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
    """resultdef() reaches for `logger`, `path`, and `historyfunc` as
    module-level globals in majorleague_predictions.py's own namespace --
    those only get set there when that file runs as __main__. Since we're
    importing it instead, point them at our own equivalents so resultdef's
    internal logging/history calls work correctly without duplicating its
    ~130 lines (including the bet-builder combo math) here.
    """
    mlp.logger = logger
    mlp.path = competition_code  # placeholder value; our historyfunc ignores it

    def _bound_historyfunc(path, hw, aw):
        return historyfunc_international(hist_df, hw, aw)

    mlp.historyfunc = _bound_historyfunc


def run_competition(name, code):
    logger.log('info', f"Checking upcoming fixtures for {name} ({code})..")
    next_match = fetch_tomorrow_matches(code)
    if next_match.empty:
        logger.log('info', f"No fixtures tomorrow for {name}.. skipping")
        return pd.DataFrame()

    logger.log('info', f"Fetching historical results for {name}..")
    hist_df = fetch_historical_matches(code)
    if len(hist_df) < MIN_HISTORY_MATCHES:
        logger.log('warning', f"Not enough history for {name} ({len(hist_df)} matches).. skipping")
        return pd.DataFrame()

    try:
        standings_df = mlp.calc_standings(hist_df)
    except Exception as e:
        logger.log('error', f"Error calculating standings for {name}..", info=str(e))
        return pd.DataFrame()

    try:
        teams_sorted = np.sort(hist_df['HomeTeam'].unique())
        warm_start = mlp.load_cached_params(code, teams_sorted)
        params = mlp.solve_parameters_decay(hist_df, init_vals=warm_start)
        mlp.save_cached_params(code, params)
    except Exception as e:
        logger.log('error', f"Error fitting parameters for {name}..", info=str(e))
        return pd.DataFrame()

    patch_mlp_internals(hist_df, code)

    div_df = pd.DataFrame()
    for _, row in next_match.iterrows():
        ht, at = row['HomeTeam'], row['AwayTeam']
        try:
            result = mlp.dixon_coles_simulate_match(params, ht, at)
        except Exception as e:
            logger.log('error', f"Issue simulating {ht} vs {at} ({name})", info=str(e))
            continue

        res = mlp.resultdef(result, ht, at, code, row['Date'], row['Time'], standings_df, hist_df)
        div_df = pd.concat([div_df, res])

    return div_df


def main():
    results_df = pd.DataFrame()
    for name, code in COMPETITIONS.items():
        try:
            res = run_competition(name, code)
            results_df = pd.concat([results_df, res])
        except Exception as e:
            logger.log('error', f"Unhandled error processing {name} ({code})..", info=str(e))
            continue

    if results_df.empty:
        logger.log('info', "No international predictions generated today.")
        return

    logger.log('info', f"Saving {len(results_df)} international prediction rows..")
    save_results_(results_df)


if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__) or '.')

    datesave = datetime.date.today().strftime('%Y%m%d')
    LOGNAME = LOGNAME.replace('{date}', datesave) + '.json'

    if os.path.exists(LOGPATH + '/' + LOGNAME):
        logger = JSONLogger(log_file=LOGNAME, log_dir=LOGPATH)
        logger.log('critical', "Tried to rerun! Forced exit app!")
        sys.exit()
    else:
        logger = JSONLogger(log_file=LOGNAME, log_dir=LOGPATH)

    tomorrow_str = (datetime.date.today() + datetime.timedelta(days=1)).strftime('%d%m%Y')
    DATANAME = DATANAME.replace('{date1}', tomorrow_str).replace('{date2}', tomorrow_str) + '.csv'

    try:
        main()
    except Exception as e:
        logger.log('critical', "Unhandled exception in international_predictions.py", info=str(e))
        raise
