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
come from odds_client.py (The Odds API) instead, matched to each fixture by
team name via team_utils.py so that every prediction row this script emits
already carries the price for its own market -- same as the domestic
scripts do straight from the football-data.co.uk fixtures feed.

Reuses the core Dixon-Coles modeling/output logic from
majorleague_predictions.py (dixon_coles_simulate_match, solve_parameters_decay,
resultdef, calculate_win_and_goal_form, calc_standings, load/save_cached_params)
<<<<<<< HEAD
instead of duplicating ~500 lines of proven, tested code. resultdef()
internally references a couple of module-level globals (path, historyfunc)
that only get set when majorleague_predictions.py runs as __main__ -- since
we're importing it instead, those are patched below to route through this
script's own history logic. This is intentional, not a workaround for a
bug -- see the "Patch mlp's internals" section below for exactly what and why.
=======
instead of duplicating ~500 lines of proven, tested code (including the bet
builder combo math). resultdef() internally references a couple of
module-level globals (path, historyfunc) that only get set when
majorleague_predictions.py runs as __main__ -- since we're importing it
instead, those are patched below to route through this script's own
history logic. This is intentional, not a workaround for a bug --
see the "Patch mlp's internals" section below for exactly what and why.
>>>>>>> a25742a0b0cc1fb64d6c5769f13750c09ceb2857

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
<<<<<<< HEAD
import odds_client
import odds_utils
import team_utils
=======
>>>>>>> a25742a0b0cc1fb64d6c5769f13750c09ceb2857

COMPETITIONS = fdo.COMPETITIONS

DATAPATH = 'predictions_data/'
DATANAME = 'my_prediction_international_data_{date1}_{date2}'

HISTORY_SEASONS_BACK = 4  # how many prior seasons to try pulling for model fitting
MIN_HISTORY_MATCHES = 20  # below this, Dixon-Coles fitting isn't meaningful


def fetch_historical_matches(code, seasons_back=HISTORY_SEASONS_BACK):
    return fdo.fetch_historical_matches(code, seasons_back=seasons_back)


def fetch_today_window_matches(code):
    """Matches within today's fetch window (today from the 08:00 cutoff
    onward, plus tomorrow up to 08:00 -- see date_utils.py). Previously
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


def fetch_competition_odds(code):
    """1X2 + Over/Under prices for one competition from The Odds API.
    football-data.org's free tier carries no odds at all, so this is the
    only source of prices for CL/WC/EC.

    Returns an empty frame rather than raising if the key is missing, the
    API is down, or the tournament simply isn't running -- predictions
    still get produced, just without prices attached.
    """
    try:
        odds_df = odds_client.fetch_odds_for_competition(code)
    except Exception as e:
        print(f"WARNING: Could not fetch odds for {code}.. ({e})")
        return pd.DataFrame()

    if odds_df.empty:
        print(f"No odds available for {code} right now.")
        return odds_df

    # Over 1.5 / Over 3.5 have no market of their own; derive them from
    # the real Over 2.5 price, same as the domestic scripts do.
    odds_df['AvgOver25'] = pd.to_numeric(odds_df['AvgOver25'], errors='coerce')
    odds_df['AvgOver15'], odds_df['AvgOver35'] = odds_utils.derive_over_under_odds(odds_df['AvgOver25'])
    return odds_df


def odds_for_fixture(odds_df, ht, at):
    """Pick this fixture's row out of the odds frame, tolerating the team
    naming differences between football-data.org ("Real Madrid CF") and
    The Odds API ("Real Madrid") -- an exact string match would silently
    leave every international row unpriced."""
    if odds_df is None or odds_df.empty:
        return {}

    mh = team_utils.best_match(ht, odds_df['HomeTeam'].dropna().unique().tolist())
    ma = team_utils.best_match(at, odds_df['AwayTeam'].dropna().unique().tolist())
    if mh is None or ma is None:
        return {}

    row = odds_df[(odds_df['HomeTeam'] == mh) & (odds_df['AwayTeam'] == ma)]
    if row.empty:
        return {}
    return {col: row.iloc[0].get(col) for col in odds_utils.ODD_COLUMNS}


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
<<<<<<< HEAD
    globals in majorleague_predictions.py's own namespace -- those only
    get set there when that file runs as __main__. Since we're importing
    it instead, point them at our own equivalents so resultdef's internal
    history call works correctly without duplicating its ~100 lines here.
=======
    globals in majorleague_predictions.py's own namespace -- those only get
    set there when that file runs as __main__. Since we're importing it
    instead, point them at our own equivalents so resultdef's internal
    history calls work correctly without duplicating its ~130 lines
    (including the bet-builder combo math) here.
>>>>>>> a25742a0b0cc1fb64d6c5769f13750c09ceb2857
    """
    mlp.path = competition_code  # placeholder value; our historyfunc ignores it

    def _bound_historyfunc(path, hw, aw):
        return historyfunc_international(hist_df, hw, aw)

    mlp.historyfunc = _bound_historyfunc


def run_competition(name, code):
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
<<<<<<< HEAD
        print(f"ERROR: Error calculating standings for {name}.. ({e})")
=======
        print(f"ERROR: Error calculating standings for {name}..", e)
>>>>>>> a25742a0b0cc1fb64d6c5769f13750c09ceb2857
        return pd.DataFrame()

    try:
        teams_sorted = np.sort(hist_df['HomeTeam'].unique())
        warm_start = mlp.load_cached_params(code, teams_sorted)
        params = mlp.solve_parameters_decay(hist_df, init_vals=warm_start)
        mlp.save_cached_params(code, params)
    except Exception as e:
<<<<<<< HEAD
        print(f"ERROR: Error fitting parameters for {name}.. ({e})")
=======
        print(f"ERROR: Error fitting parameters for {name}..", e)
>>>>>>> a25742a0b0cc1fb64d6c5769f13750c09ceb2857
        return pd.DataFrame()

    patch_mlp_internals(hist_df, code)

    print(f"Fetching odds for {name}..")
    odds_df = fetch_competition_odds(code)

    div_df = pd.DataFrame()
    for _, row in next_match.iterrows():
        ht, at = row['HomeTeam'], row['AwayTeam']
        try:
            result = mlp.dixon_coles_simulate_match(params, ht, at)
        except Exception as e:
<<<<<<< HEAD
            print(f"ERROR: Issue simulating {ht} vs {at} ({name}) ({e})")
=======
            print(f"ERROR: Issue simulating {ht} vs {at} ({name})", e)
>>>>>>> a25742a0b0cc1fb64d6c5769f13750c09ceb2857
            continue

        match_odds = odds_for_fixture(odds_df, ht, at)
        if not match_odds:
            print(f"No odds matched for {ht} vs {at} ({name}).. rows go out unpriced")

        res = mlp.resultdef(result, ht, at, code, row['Date'], row['Time'], standings_df, hist_df,
                            odds=match_odds)
        div_df = pd.concat([div_df, res])

    return div_df


def main():
    results_df = pd.DataFrame()
    for name, code in COMPETITIONS.items():
        try:
            res = run_competition(name, code)
            results_df = pd.concat([results_df, res])
        except Exception as e:
<<<<<<< HEAD
            print(f"ERROR: Unhandled error processing {name} ({code}).. ({e})")
=======
            print(f"ERROR: Unhandled error processing {name} ({code})..", e)
>>>>>>> a25742a0b0cc1fb64d6c5769f13750c09ceb2857
            continue

    if results_df.empty:
        print("No international predictions generated today.")
        return

    print(f"Saving {len(results_df)} international prediction rows..")
    save_results_(results_df)


if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__) or '.')

    today_str = datetime.date.today().strftime('%d%m%Y')
    DATANAME = DATANAME.replace('{date1}', today_str).replace('{date2}', today_str) + '.csv'

    try:
        main()
    except Exception as e:
<<<<<<< HEAD
        print(f"CRITICAL: Unhandled exception in international_predictions.py ({e})")
=======
        print("CRITICAL: Unhandled exception in international_predictions.py", e)
>>>>>>> a25742a0b0cc1fb64d6c5769f13750c09ceb2857
        raise
