#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import  sys, os, requests, warnings, json
from scipy.stats import poisson
from scipy.optimize import minimize
import model_config
import date_utils
import odds_utils
import team_utils
from collections import defaultdict

"""
Running year and leagues
"""
YEAR = '2526'

LEAGUES = {'En PremierLeague': 'E0',
               'En Championship': 'E1',
               'De Bundesliga': 'D1',
               'It Serie A': 'I1',
               'Sp LaLiga': 'SP1',
               'Fr Championnat': 'F1',
               'Nh Eredivisie': 'N1',
               'Bg JupilerLeague': 'B1',
               'Pr Liga I': 'P1',
               'Gr SuperLeague': 'G1',
               'De Bundesliga 2': 'D2',
               'It Serie B': 'I2',
               'Sp Segunda': 'SP2',
               'Fr Division 2': 'F2',
               'En League 1': 'E2',
               'SC PremierLeague': 'SC0',
               'Tr Futbol Ligi 1': 'T1'
               }
DIVISIONS = {
    "E0": "39",
    "E1": "40",
    "E2": "41",
    "SC0": "179",
    "D1": "78",
    "D2": "79",
    "I1": "135",
    "I2": "136",
    "SP1": "140",
    "SP2": "141",
    "F1": "61",
    "F2": "62",
    "B1": "144",
    "N1": "88",
    "P1": "94",
    "T1": "204",
    "G1": "197",
}


"""
Path to save  data
"""
DATAPATH = 'predictions_data/'
DATANAME = 'my_prediction_major_data_{date1}_{date2}'
PARAMSPATH = 'model_params/'

# One row per (match, prediction), each carrying the bookmaker odd for
# that specific prediction -- see resultdef().
OUTPUT_COLUMNS = ["Division", "Date", "Time", "HomeTeam", "AwayTeam", "Prediction", "Odd",
                  "Prediction %", "History %", "HomeTeam Stats", "AwayTeam Stats",
                  "HomeForm", "AwayForm"]

warnings.filterwarnings('ignore')

def resolve_team_params(params_dict, team, teams_in_model=None):
    """Return (attack, defence) for `team`, falling back to a league-average
    placeholder when the team has no fitted parameters.

    A newly promoted team has no matches in the training window, so
    calc_means() raised KeyError -> the caller logged it and `continue`d,
    silently dropping that fixture from the day's card entirely. Early in a
    season that can be several fixtures a day disappearing with no visible
    reason beyond a log line.

    The fallback treats an unknown team as exactly league-average: attack
    at the fitted mean (~0 by the identifiability penalty) and defence at
    the fitted mean. That is deliberately a weak, unopinionated estimate --
    a promoted side is usually weaker than average, so if anything this
    flatters them. It exists so the fixture still gets a prediction that
    downstream confidence gates can then judge on its merits, rather than
    vanishing. Callers can check `is_fallback` to decide whether to trust
    it.
    """
    a_key, d_key = 'attack_' + team, 'defence_' + team
    if a_key in params_dict and d_key in params_dict:
        return params_dict[a_key], params_dict[d_key], False

    atts = [v for k, v in params_dict.items() if k.startswith('attack_')]
    defs = [v for k, v in params_dict.items() if k.startswith('defence_')]
    if not atts or not defs:
        raise ValueError(f"No fitted parameters at all; cannot fall back for {team}")
    return float(np.mean(atts)), float(np.mean(defs)), True


def calc_means(param_dict, homeTeam, awayTeam):
    """Calculate expected goals for home and away teams with safety checks.

    Interface is identical to original: returns [lambda_home, lambda_away].
    Unknown teams (e.g. newly promoted, no matches in the training window)
    resolve to league-average parameters rather than raising -- see
    resolve_team_params().
    """
    h_att, h_def, _ = resolve_team_params(param_dict, homeTeam)
    a_att, a_def, _ = resolve_team_params(param_dict, awayTeam)

    lambda_home = np.exp(h_att + a_def + param_dict['home_adv'])
    lambda_away = np.exp(h_def + a_att)

    # Numerical safety: clamp to avoid extreme Poisson means that break simulation
    # Allow wide range but prevent absurd values due to optimizer instability
    lambda_home = float(np.clip(lambda_home, 1e-6, 20.0))
    lambda_away = float(np.clip(lambda_away, 1e-6, 20.0))
    return [lambda_home, lambda_away]

def rho_correction(x, y, lambda_x, mu_y, rho):
    """Dixon-Coles short-score correction (same formulas as original).

    Kept exactly but robust to small numerical issues by clipping rho into (-1,1).
    """
    # ensure rho within sensible bounds
    rho = float(np.clip(rho, -0.9999, 0.9999))
    if x == 0 and y == 0:
        return 1 - (lambda_x * mu_y * rho)
    elif x == 0 and y == 1:
        return 1 + (lambda_x * rho)
    elif x == 1 and y == 0:
        return 1 + (mu_y * rho)
    elif x == 1 and y == 1:
        return 1 - rho
    else:
        return 1.0

def dixon_coles_simulate_match(params_dict, homeTeam, awayTeam, max_goals=5):
    """Simulate matrix of score probabilities.

    Changes (low-level, backward compatible):
    - Uses calc_means (which includes clamping of lambda/mu) so simulation is stable.
    - Extends goal range if needed to capture tail mass, but always returns a matrix where indices 0..5 are present.
    - Normalizes matrix so probabilities sum to 1 after rho-correction.
    """
    team_avgs = calc_means(params_dict, homeTeam, awayTeam)

    # ensure at least 5 for downstream compatibility
    limit = max(max_goals, 5)

    # extend if tail mass beyond 'limit' is non-negligible (but cap to avoid huge matrices)
    tail_threshold = 1e-6
    cap_limit = 12
    while limit < cap_limit:
        tails = [1 - poisson.cdf(limit, a) for a in team_avgs]
        if max(tails) > tail_threshold:
            limit += 1
        else:
            break

    p_home = np.array([poisson.pmf(i, team_avgs[0]) for i in range(limit + 1)])
    p_away = np.array([poisson.pmf(i, team_avgs[1]) for i in range(limit + 1)])

    output_matrix = np.outer(p_home, p_away)

    # apply Dixon-Coles correction on the top-left 2x2 block
    for i in range(min(2, output_matrix.shape[0])):
        for j in range(min(2, output_matrix.shape[1])):
            corr = rho_correction(i, j, team_avgs[0], team_avgs[1], params_dict.get('rho', 0))
            # guard against non-finite corrections
            if not np.isfinite(corr):
                corr = 1.0
            output_matrix[i, j] *= corr

    # The tau correction is only valid while it keeps every cell
    # non-negative -- e.g. the 0-0 cell is (1 - lambda*mu*rho), which goes
    # NEGATIVE once rho > 1/(lambda*mu). _dc_log_like_single already
    # rejects that region during fitting (corr <= 0 -> 1e9 penalty), but
    # only for scorelines actually observed in the training data, so a
    # fitted rho can still produce a negative cell for some unobserved
    # simulated matchup. Left unclipped, that negative mass silently
    # subtracts from the sum used to normalize, inflating every other
    # market's probability. Floor at 0 and let the normalization below
    # redistribute.
    np.clip(output_matrix, 0.0, None, out=output_matrix)

    total = output_matrix.sum()
    if total <= 0 or not np.isfinite(total):
        print(f"ERROR: Non-positive total probability for {homeTeam}-{awayTeam} ({total})")
        sz = output_matrix.shape[0]
        return np.ones((sz, sz)) / (sz * sz)

    output_matrix = output_matrix / total
    return output_matrix

def _dc_log_like_single(params, data, teams, xi=0.0, reg=0.05, ident_pen=1e3):
    """Negative log-likelihood for Dixon-Coles with exponential decay and L2 regularization.

    Implementation notes:
    - Adds a strong quadratic penalty on the sum of attack coefficients to enforce identifiability
      without equality constraints that can mislead some optimizers.
    - Returns a large penalty if impossible pmf/corr or if lam/mu become numerically extreme.
    """
    n = len(teams)
    attack = params[:n]
    defence = params[n:2*n]
    rho = params[-2]
    gamma = params[-1]

    atk = dict(zip(teams, attack))
    dfs = dict(zip(teams, defence))

    ll = 0.0
    for row in data.itertuples(index=False):
        # compute raw lambda/mu (no clipping here) to keep gradients meaningful
        lambda_raw = np.exp(atk[row.HomeTeam] + dfs[row.AwayTeam] + gamma)
        mu_raw = np.exp(atk[row.AwayTeam] + dfs[row.HomeTeam])
        # if raw means are absurd, return big penalty so optimiser avoids these regions
        if not np.isfinite(lambda_raw) or not np.isfinite(mu_raw) or lambda_raw > 100 or mu_raw > 100:
            return 1e9
        corr = rho_correction(row.HomeGoals, row.AwayGoals, lambda_raw, mu_raw, rho)
        pmf_x = poisson.pmf(row.HomeGoals, lambda_raw)
        pmf_y = poisson.pmf(row.AwayGoals, mu_raw)
        if pmf_x <= 0 or pmf_y <= 0 or corr <= 0:
            return 1e9
        contrib = np.log(corr) + np.log(pmf_x) + np.log(pmf_y)
        weight = np.exp(-xi * row.time_diff)
        ll += weight * contrib

    # L2 regularization on attack & defence to avoid overfitting.
    # Defence is penalized around its own mean rather than around zero.
    # Rationale: with the identifiability penalty below pinning
    # sum(attack) = 0, it is mean(defence) that carries the league's
    # overall scoring level, so shrinking defence toward zero technically
    # also shrinks expected goals toward exp(0) = 1.0 per team. Penalizing
    # spread-around-the-mean targets the genuine overfitting risk
    # (team-to-team differences) while leaving the level free.
    # MEASURED IMPACT: negligible. Tested on synthetic leagues with a
    # known true scoring rate across 150/80-match samples and reg up to
    # 0.5 -- the estimated mean xG bias was identical to 3 decimal places
    # either way, because the log-likelihood term dominates the penalty by
    # orders of magnitude at any sane reg. Kept as the more principled
    # form, NOT as a fix for an observed problem.
    reg_pen = reg * (np.sum(attack ** 2) + np.sum((defence - np.mean(defence)) ** 2))
    # identifiability penalty: encourage mean(attack) ~ 0
    ident_penalty = ident_pen * (np.sum(attack) ** 2)

    return -ll + reg_pen + ident_penalty

def params_vector_from_dict(params_dict, teams):
    """Flatten a fitted-params dict back into the raw vector solve_parameters_decay
    optimizes over, in the same [attack..., defence..., rho, home_adv] order,
    for the given (sorted) team list. Returns None if any team is missing
    from the cached dict (e.g. promoted/relegated team not seen last run) --
    a fresh random init is safer than partially-wrong warm start in that case.
    """
    try:
        attack = [params_dict['attack_' + t] for t in teams]
        defence = [params_dict['defence_' + t] for t in teams]
        return np.array(attack + defence + [params_dict['rho'], params_dict['home_adv']])
    except KeyError:
        return None


def load_cached_params(divis, teams):
    """Load last run's fitted Dixon-Coles params for this league, to use as
    the optimizer's starting point (fix #1: warm-start instead of always
    starting from a random init, which is the main compute cost per league
    per run on a GitHub Actions runner).
    """
    path = os.path.join(PARAMSPATH, f'{divis}_params.json')
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            cached = json.load(f)
    except Exception:
        return None
    return params_vector_from_dict(cached, teams)


def save_cached_params(divis, params_dict):
    if not os.path.exists(PARAMSPATH):
        os.makedirs(PARAMSPATH)
    path = os.path.join(PARAMSPATH, f'{divis}_params.json')
    with open(path, 'w') as f:
        json.dump(params_dict, f)


def solve_parameters_decay(dataset, xi=None, debug=False, init_vals=None, options={'disp': False, 'maxiter': 200},
                           constraints=None, reg=0.05, restarts=3, bounds_scale=3.0, seed=42, **kwargs):
    """Estimate Dixon-Coles parameters with L2 regularization, bounds and multiple restarts.

    This function preserves the original return format (a dict mapping names to values). It
    replaces the equality constraint approach by bounded optimization + identifiability penalty
    for more robust fits.
    """
    # xi (time-decay rate) now comes from model_config so it can be
    # tuned by backtest_calibration.py rather than being a magic number
    # duplicated across both fitters. None -> configured/default value.
    if xi is None:
        xi = model_config.get_xi()
    teams = np.sort(dataset['HomeTeam'].unique())
    away_teams = np.sort(dataset['AwayTeam'].unique())
    if not np.array_equal(teams, away_teams):
        raise ValueError("something not right")
    n_teams = len(teams)

    # bounds: keep attack/defence in [-bounds_scale, bounds_scale], rho in (-0.999,0.999), home_adv reasonable
    b_att = [(-bounds_scale, bounds_scale)] * n_teams
    b_def = [(-bounds_scale, bounds_scale)] * n_teams
    b_rho = [(-0.9999, 0.9999), (-2.5, 2.5)]
    bounds = b_att + b_def + b_rho

    # Seeded RNG so repeated runs on the same data produce the same fitted
    # parameters (previously used the unseeded global np.random state).
    rng = np.random.RandomState(seed)

    def make_init():
        return np.concatenate((rng.normal(0, 0.2, n_teams),
                               rng.normal(0, 0.2, n_teams),
                               np.array([0.0, 0.1])
                               ))

    best = None
    best_val = np.inf
    best_x = None

    for r in range(max(1, restarts)):
        if init_vals is not None and r == 0:
            init = init_vals
        else:
            init = make_init()
        try:
            # try L-BFGS-B with bounds (robust and fast)
            res = minimize(lambda x: _dc_log_like_single(x, dataset, list(teams), xi=xi, reg=reg), init,
                           method='L-BFGS-B', bounds=bounds, options={'maxiter': options.get('maxiter', 200)})
            # fallback to SLSQP without bounds if needed
            if (not res.success) and constraints is not None:
                res = minimize(lambda x: _dc_log_like_single(x, dataset, list(teams), xi=xi, reg=reg), init,
                               method='SLSQP', bounds=bounds, constraints=constraints, options=options)

            if res.success and res.fun < best_val:
                best_val = res.fun
                best = res
                best_x = res.x
        except Exception:
            continue

    if best is None:
        raise RuntimeError('Optimization failed for all restarts')

    x = best_x
    param_names = ["attack_" + team for team in teams] + ["defence_" + team for team in teams] + ['rho', 'home_adv']
    return dict(zip(param_names, x))

def resultdef(result, ht, at, divis, mdata, mtime, standings, lgdata, odds=None, THRESH = None):
    # `odds` is that fixture's bookmaker average prices (a dict/Series of
    # the Avg* columns, see upcoming()). Every qualifying market becomes
    # its own row carrying the odd for that specific market, so the output
    # is "one row per prediction, priced", not one row per match.
    # THRESH is now per-market rather than one flat number (pass an explicit
    # value to override for every market). Rationale: a flat 0.5 meant
    # something completely different depending on the market's natural
    # frequency -- Over 1.5 lands ~75% of the time so it cleared 0.5 on
    # virtually every fixture, while Over 2.5 (~52% base) was cut on roughly
    # half of all fixtures before it was ever judged. The floor is now each
    # market's own base rate (see model_config.get_record_floor), so
    # "recorded" consistently means "the model rates this at least as likely
    # as typical for this market".
    max_g = result.shape[0] - 1
    max_g_away = result.shape[1] - 1

    # Vectorized market sums (previously Python-loop generators wrapped in
    # np.sum, re-evaluated for every match -- this is the same result via
    # pure numpy slicing/masking, meaningfully cheaper multiplied across
    # every fixture in every league on every run).
    gi, gj = np.indices(result.shape)
    total_goals = gi + gj

    over1_5 = result[total_goals > 1].sum()
    over2_5 = result[total_goals > 2].sum()
    over3_5 = result[total_goals > 3].sum()

    home = np.sum(np.tril(result, -1))
    away = np.sum(np.triu(result, 1))
    draw = np.sum(np.diag(result))

    gg = result[1:, 1:].sum()

    hO1_5 = result[2:, :].sum()
    hO2_5 = result[3:, :].sum()

    aO1_5 = result[:, 2:].sum()
    aO2_5 = result[:, 3:].sum()


    dict = {'O1_5': over1_5,
            'O2_5': over2_5,
            'O3_5': over3_5,
            '1':home,
            '2':away,
            'X': draw,
            'GG': gg,
            'hO1_5': hO1_5,
            'hO2_5': hO2_5,
            'aO1_5': aO1_5,
            'aO2_5': aO2_5,
            }

    outcome = pd.DataFrame(columns=OUTPUT_COLUMNS)
    rows = []
    hist_dict = None

    for res in dict.keys():
        val = dict[res]
        if THRESH is not None:
            this_thresh = THRESH          # explicit caller override
        else:
            this_thresh = model_config.get_record_floor(res)
        if val > this_thresh:
            if hist_dict is None:
                print(f"Calculating class history for {ht}-{at}")
                hist_dict = historyfunc(path, ht, at)
            try:
                hist_perc = hist_dict[res]
            except KeyError:
                print(f"WARNING: No history data for {ht}-{at} ({res})")
                hist_perc = '-'

            # calc_standings() only creates a row for a team that has
            # appeared in an actual PLAYED result -- a promoted team, even
            # though it now reaches this point via resolve_team_params()'s
            # fallback, still has no standings row at all. .squeeze() on
            # that empty lookup returns an empty Series (not NaN, not a
            # string), which got embedded in the outcome frame, garbled
            # through the CSV round-trip, and then failed to match
            # parse_summary()'s regex downstream -- 'NoneType' object has
            # no attribute 'groups'. Use .iloc[0] after an explicit
            # emptiness check instead, with a well-formed zero-record
            # placeholder string (still valid input to parse_summary())
            # so a promoted team's fixture reads honestly as "no history"
            # rather than crashing the whole league's run one script later.
            NO_STATS = "0M 0W 0D 0L 0-0 (0.0-0.0)"
            home_rows = standings.loc[standings['team'] == ht, 'summary_home']
            away_rows = standings.loc[standings['team'] == at, 'summary_away']
            homestats = home_rows.iloc[0] if not home_rows.empty else NO_STATS
            awaystats = away_rows.iloc[0] if not away_rows.empty else NO_STATS

            # Odd for THIS market specifically. None when the market has
            # no published price (GG, home/away-specific overs) or the
            # fixture feed carried no odds for this match.
            odd = odds_utils.odd_for_prediction(res, odds)

            rows.append([divis, mdata, mtime, ht, at, res, odd, val.round(2), hist_perc, homestats, awaystats, '', ''])

    if rows:
        outcome = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

        # Form data + merge computed once for the whole match, not once per
        # qualifying market (previously recomputed calculate_win_and_goal_form
        # and re-merged the entire growing outcome frame on every iteration).
        form_df = calculate_win_and_goal_form(lgdata)
        # LEFT joins, not inner. An inner join silently DROPS any fixture
        # whose team has no recent-form row -- which is exactly what a
        # newly promoted team looks like, since form is computed from this
        # league's own match history and a promoted side has none yet.
        # With the promoted-team parameter fallback in resolve_team_params()
        # those fixtures now reach this point instead of being skipped
        # earlier, so an inner join here could drop every row and leave
        # `merged` empty -- at which point merged.apply(...) returns a
        # 0-column frame and assigning it to a 2-column key raised
        # "ValueError: Columns must be same length as key" and killed the
        # whole league's run. A left join keeps the fixture with empty form
        # instead, which is the honest representation: we have a prediction
        # but no form history to show alongside it.
        merged = outcome.merge(form_df, left_on='HomeTeam', right_on='team',
                                suffixes=('', '_home'), how='left')
        merged = merged.merge(form_df, left_on='AwayTeam', right_on='team',
                               suffixes=('_home', '_away'), how='left')

        # Function to select correct form based on prediction type
        def pick_form(row):
            pred = row['Prediction']
            if pred in ['1', '2', 'X']:
                return pd.Series([row['HomeWinForm_home'], row['AwayWinForm_away']])
            elif pred in ['O1_5', 'O2_5', 'O3_5', 'GG', 'hO1_5', 'hO2_5', 'aO1_5', 'aO2_5']:
                return pd.Series([row['HomeGoalsForm_home'], row['AwayGoalsForm_away']])
            else:
                return pd.Series([None, None])

        # Guard the empty case explicitly: DataFrame.apply on a 0-row frame
        # returns a 0-COLUMN result, which cannot be assigned to a 2-column
        # key. The left joins above make this unlikely, but `rows` could
        # still be empty-after-filtering in principle, and a crash here
        # takes down the entire league.
        if merged.empty:
            merged['HomeForm'] = pd.Series(dtype=object)
            merged['AwayForm'] = pd.Series(dtype=object)
        else:
            merged[['HomeForm', 'AwayForm']] = merged.apply(pick_form, axis=1)

        # Final result
        outcome = merged[OUTPUT_COLUMNS]

    return(outcome)

def download_league_data(url):
    league_data = pd.read_csv(url, encoding='utf-8-sig')
    league_data['Date'] = pd.to_datetime(league_data['Date'], format='%d/%m/%Y')
    league_data['time_diff'] = (league_data['Date'].max() - league_data['Date']).dt.days
    league_data = league_data[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'time_diff', 'Date']]
    league_data = league_data.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals'})

    return (league_data)

def attach_fixture_odds(next_match):
    """Normalize whichever bookmaker-average columns a fixtures feed
    happens to publish into the fixed Avg* set every prediction row is
    priced from (see odds_utils.MARKET_ODD_COLUMNS).

    football-data.co.uk publishes 1X2 (AvgH/AvgD/AvgA) and, not always,
    Over/Under 2.5 as 'Avg>2.5' -- so it's looked up tolerantly rather
    than indexed directly, and anything missing becomes NaN instead of a
    KeyError that would kill the whole run. Over 1.5 / Over 3.5 have no
    published market at all and are derived from the real Over 2.5 price
    (see odds_utils.derive_over_under_odds).
    """
    found = team_utils.find_columns(next_match.columns, ['AvgH', 'AvgD', 'AvgA', 'Avg>2.5'])
    rename = {c: 'AvgOver25' for c in found if c.strip().lower() == 'avg>2.5'}
    next_match = next_match.rename(columns=rename)

    for col in ['AvgH', 'AvgD', 'AvgA', 'AvgOver25']:
        if col in next_match.columns:
            next_match[col] = pd.to_numeric(next_match[col], errors='coerce')
        else:
            next_match[col] = np.nan

    next_match['AvgOver15'], next_match['AvgOver35'] = odds_utils.derive_over_under_odds(next_match['AvgOver25'])
    return next_match


def upcoming(uri):
    next_match = pd.read_csv(uri, encoding='utf-8-sig')
    keep = ['Date', 'Time', 'Div', 'HomeTeam', 'AwayTeam'] + team_utils.find_columns(
        next_match.columns, ['AvgH', 'AvgD', 'AvgA', 'Avg>2.5'])
    next_match = next_match[keep]
    next_match = attach_fixture_odds(next_match)
    next_match['Date'] = pd.to_datetime(next_match['Date'], format='%d/%m/%Y')
    return next_match

def load_fixtures_rapidapi():
    url = "https://api-football-v1.p.rapidapi.com/v3/fixtures"
    rapidapi_key = os.environ.get("RAPIDAPI_KEY")
    if not rapidapi_key:
        print("ERROR: RAPIDAPI_KEY environment variable not set; cannot call RapidAPI fixtures endpoint.")
        raise RuntimeError("RAPIDAPI_KEY environment variable not set")

    headers = {
        "x-rapidapi-key": rapidapi_key,
        "x-rapidapi-host": "api-football-v1.p.rapidapi.com"
    }

    df = pd.DataFrame()
    for k,league in DIVISIONS.items():

        querystring = {"league":league, "season":"2025", "from":"2025-08-14", "to":"2025-08-18"}


        response = requests.get(url, headers=headers, params=querystring)

        if response.status_code == 200:
            matches = response.json().get("response", [])
            
            data = pd.DataFrame([{
                "Date": m["fixture"]["date"][:10],
                "Time": m["fixture"]["date"][11:16],
                "Div": k,
                "HomeTeam": m["teams"]["home"]["name"],
                "AwayTeam": m["teams"]["away"]["name"],
            } for m in matches])
            

            df = pd.concat([df, data])
    # This endpoint carries no prices, but the downstream code always
    # reads the Avg* columns -- attach them (as NaN) so a fixture from
    # here simply goes out unpriced instead of raising a KeyError.
    df = attach_fixture_odds(df)
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    return df

def save_results_(df):
    if not os.path.exists(DATAPATH):
        os.makedirs(DATAPATH)
    filename = DATAPATH + '/' + DATANAME

    if os.path.exists(filename):
        temp = pd.read_csv(filename)
        towrite = pd.concat([temp,df])
    else:
        towrite = df

    try:
        # Combine Date and Time into a single datetime in UTC
        dt_utc = pd.to_datetime(
            towrite['Date'].astype(str) + ' ' + towrite['Time'].astype(str),
            utc=True
        )

        # Convert from UTC to Greece time (Athens)
        dt_gr = dt_utc.dt.tz_convert('Europe/Athens')

        # Update your DataFrame
        towrite['Date'] = dt_gr.dt.strftime('%Y-%m-%d') + ', ' + dt_gr.dt.day_name(locale='en_US')
        towrite['Time'] = dt_gr.dt.strftime('%H:%M')  

        towrite['Date_temp'] = pd.to_datetime(towrite['Date'], format="%Y-%m-%d, %A")
        towrite['Time_temp'] = pd.to_datetime(towrite['Time'], format="%H:%M").dt.time
        towrite['Datetime_temp'] = towrite.apply(lambda x: pd.Timestamp.combine(x['Date_temp'], x['Time_temp']), axis=1)
        towrite.sort_values(by=['Datetime_temp', 'HomeTeam'], inplace=True)
        towrite.drop(columns=['Date_temp', 'Time_temp', 'Datetime_temp'],inplace=True)
    except Exception as e:
        print(f"ERROR: Issue converting date.. Saving without sorting.. ({e})")
    towrite.to_csv(filename, index=False)

def calculate_win_and_goal_form(df):
    # Ensure date is datetime
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True) 
    df = df.sort_values('Date')

    # ----- HOME perspective -----
    home_df = df[['HomeTeam', 'Date', 'HomeGoals', 'AwayGoals']].copy()
    home_df['team'] = home_df['HomeTeam']
    home_df['home_away'] = 'home'
    home_df['win_form'] = home_df.apply(lambda x: 'W' if x['HomeGoals'] > x['AwayGoals'] 
                                        else 'D' if x['HomeGoals'] == x['AwayGoals'] 
                                        else 'L', axis=1)
    home_df['goal_form'] = ((home_df['HomeGoals'] + home_df['AwayGoals']) > 2.5).map({True: 'O', False: 'U'})

    # ----- AWAY perspective -----
    away_df = df[['AwayTeam', 'Date', 'HomeGoals', 'AwayGoals']].copy()
    away_df['team'] = away_df['AwayTeam']
    away_df['home_away'] = 'away'
    away_df['win_form'] = away_df.apply(lambda x: 'W' if x['AwayGoals'] > x['HomeGoals'] 
                                        else 'D' if x['AwayGoals'] == x['HomeGoals'] 
                                        else 'L', axis=1)
    away_df['goal_form'] = ((away_df['HomeGoals'] + away_df['AwayGoals']) > 2.5).map({True: 'O', False: 'U'})

    # Combine home & away
    all_games = pd.concat([
        home_df[['team', 'Date', 'home_away', 'win_form', 'goal_form']],
        away_df[['team', 'Date', 'home_away', 'win_form', 'goal_form']]
    ])

    # Helper to pad form to 6 characters
    def pad_form(form_list):
        form_str = ''.join(form_list[-6:])
        return form_str.rjust(6, '-')  # pad on the left so most recent stays at right

    # Compute last 6 for each type of form
    latest_forms = []
    for (team, ha), group in all_games.groupby(['team', 'home_away']):
        group = group.sort_values('Date')
        win_last6 = pad_form(group['win_form'].tolist())
        goal_last6 = pad_form(group['goal_form'].tolist())
        latest_forms.append({
            'team': team,
            'home_away': ha,
            'win_last6': win_last6,
            'goal_last6': goal_last6
        })

    # Create DataFrame and pivot
    form_df = pd.DataFrame(latest_forms)
    form_df = form_df.pivot(index='team', columns='home_away', values=['win_last6', 'goal_last6']).reset_index()

    # Flatten MultiIndex column names
    form_df.columns = ['team', 'AwayWinForm', 'HomeWinForm', 'AwayGoalsForm', 'HomeGoalsForm']
    # Reorder columns
    form_df = form_df[['team', 'HomeWinForm', 'AwayWinForm', 'HomeGoalsForm', 'AwayGoalsForm']]

    return form_df

def historyfunc(path, hw, aw):
    """
    :return: history percentage of home win, away win, over, under
    """
    history = list()
    win = 0
    lose = 0
    draw = 0
    ov = 0
    und = 0
    ov3_5 = 0
    ov1_5 = 0
    gg = 0
    hO0_5 = 0
    hO1_5 = 0
    hO2_5 = 0
    aO0_5 = 0
    aO1_5 = 0
    aO2_5 = 0

    for year in range(1, 5):

        now = str(int(YEAR[0:2]) - year)
        nows = str(int(YEAR[2:4]) - year)
        bf = now + nows
        ncsv = path.replace(f"{YEAR}", bf)
        old_data = pd.read_csv(ncsv, encoding='utf-8-sig')
        old_data = old_data[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']]
        ht_found = old_data.loc[(old_data["HomeTeam"] == hw)]
        try:
            if (ht_found.loc[ht_found["AwayTeam"] == aw]['FTHG'].iloc[0]) > (
            ht_found.loc[ht_found["AwayTeam"] == aw]['FTAG'].iloc[0]):
                win = win + 1
            elif (ht_found.loc[ht_found["AwayTeam"] == aw]['FTHG'].iloc[0]) < (
            ht_found.loc[ht_found["AwayTeam"] == aw]['FTAG'].iloc[0]):
                lose = lose + 1
            elif (ht_found.loc[ht_found["AwayTeam"] == aw]['FTHG'].iloc[0]) == (
            ht_found.loc[ht_found["AwayTeam"] == aw]['FTAG'].iloc[0]):
                draw = draw + 1

            if (ht_found.loc[ht_found["AwayTeam"] == aw]['FTHG'].iloc[0]) + (
            ht_found.loc[ht_found["AwayTeam"] == aw]['FTAG'].iloc[0]) > 2:
                ov = ov + 1
            else:
                und = und + 1

            if (ht_found.loc[ht_found["AwayTeam"] == aw]['FTHG'].iloc[0]) + (
            ht_found.loc[ht_found["AwayTeam"] == aw]['FTAG'].iloc[0]) > 2:
                ov1_5 = ov1_5 + 1

            if (ht_found.loc[ht_found["AwayTeam"] == aw]['FTHG'].iloc[0]) > 0 and (
            ht_found.loc[ht_found["AwayTeam"] == aw]['FTAG'].iloc[0]) > 0:
                gg = gg + 1

            if (ht_found.loc[ht_found["AwayTeam"] == aw]['FTHG'].iloc[0]) + (
            ht_found.loc[ht_found["AwayTeam"] == aw]['FTAG'].iloc[0]) > 3:
                ov3_5 = ov3_5 + 1

            if (ht_found.loc[ht_found["AwayTeam"] == aw]['FTHG'].iloc[0]) > 0.5:
                hO0_5 += 1
            if (ht_found.loc[ht_found["AwayTeam"] == aw]['FTHG'].iloc[0]) > 1.5:
                hO1_5 += 1
            if (ht_found.loc[ht_found["AwayTeam"] == aw]['FTHG'].iloc[0]) > 2.5:
                hO2_5 += 1

            if (ht_found.loc[ht_found["AwayTeam"] == aw]['FTAG'].iloc[0]) > 0.5:
                aO0_5 += 1
            if (ht_found.loc[ht_found["AwayTeam"] == aw]['FTAG'].iloc[0]) > 1.5:
                aO1_5 += 1
            if (ht_found.loc[ht_found["AwayTeam"] == aw]['FTAG'].iloc[0]) > 2.5:
                aO2_5 += 1

        except:
            pass

    if win + draw + lose > 0:
        totalm = win + draw + lose
        perc_h = f'{win}/{totalm}'
        perc_d = f'{draw}/{totalm}'
        perc_a = f'{lose}/{totalm}'
        perc_o = f'{ov}/{totalm}'
        perc_o3 = f'{ov3_5}/{totalm}'
        perc_o1 = f'{ov1_5}/{totalm}'
        perc_gg = f'{gg}/{totalm}'
        perc_hO0_5 = f'{hO0_5}/{totalm}'
        perc_hO1_5 = f'{hO1_5}/{totalm}'
        perc_hO2_5 = f'{hO2_5}/{totalm}'
        perc_aO0_5 = f'{aO0_5}/{totalm}'
        perc_aO1_5 = f'{aO1_5}/{totalm}'
        perc_aO2_5 = f'{aO2_5}/{totalm}'

    else:
        perc_h = '-'
        perc_d = '-'
        perc_a = '-'
        perc_o = '-'
        perc_o3 = '-'
        perc_o1 = '-'
        perc_gg = '-'
        perc_hO0_5 = '-'
        perc_hO1_5 = '-'
        perc_hO2_5 = '-'
        perc_aO0_5 = '-'
        perc_aO1_5 = '-'
        perc_aO2_5 = '-'

    dict = {'O1_5': perc_o1,
            'O2_5': perc_o,
            'O3_5': perc_o3,
            '1': perc_h,
            '2': perc_a,
            'X': perc_d,
            'GG': perc_gg,
            'hO0_5': perc_hO0_5,
            'hO1_5': perc_hO1_5,
            'hO2_5': perc_hO2_5,
            'aO0_5': perc_aO0_5,
            'aO1_5': perc_aO1_5,
            'aO2_5': perc_aO2_5,
            }

    return (dict)

def calc_standings(results, season=None):
    table = defaultdict(lambda: {
        "matches":0,"wins":0,"draws":0,"losses":0,"gf":0,"ga":0,
        "home":{"matches":0,"wins":0,"draws":0,"losses":0,"gf":0,"ga":0},
        "away":{"matches":0,"wins":0,"draws":0,"losses":0,"gf":0,"ga":0}
    })

    # Optional filter for season
    if season is not None and "Season" in results.columns:
        df_season = results[results["Season"] == season]
    else:
        df_season = results

    for r in df_season.itertuples(index=False):
        hg, ag = r.HomeGoals, r.AwayGoals
        home, away = r.HomeTeam, r.AwayTeam

        # home
        tab = table[home]
        tab["matches"] += 1; tab["gf"] += hg; tab["ga"] += ag
        tab["home"]["matches"] += 1; tab["home"]["gf"] += hg; tab["home"]["ga"] += ag
        if hg > ag:
            tab["wins"] += 1; tab["home"]["wins"] += 1
        elif hg == ag:
            tab["draws"] += 1; tab["home"]["draws"] += 1
        else:
            tab["losses"] += 1; tab["home"]["losses"] += 1

        # away
        tab = table[away]
        tab["matches"] += 1; tab["gf"] += ag; tab["ga"] += hg
        tab["away"]["matches"] += 1; tab["away"]["gf"] += ag; tab["away"]["ga"] += hg
        if ag > hg:
            tab["wins"] += 1; tab["away"]["wins"] += 1
        elif ag == hg:
            tab["draws"] += 1; tab["away"]["draws"] += 1
        else:
            tab["losses"] += 1; tab["away"]["losses"] += 1

    # Build DataFrame with both detailed and summary columns
    rows = []
    for team_name, stats in table.items():
        matches = stats["matches"]
        gf, ga = stats["gf"], stats["ga"]
        points = stats["wins"]*3 + stats["draws"]

        home, away = stats["home"], stats["away"]

        # averages
        avg_gf = gf / matches if matches else 0
        avg_ga = ga / matches if matches else 0
        home_avg_gf = home["gf"] / home["matches"] if home["matches"] else 0
        home_avg_ga = home["ga"] / home["matches"] if home["matches"] else 0
        away_avg_gf = away["gf"] / away["matches"] if away["matches"] else 0
        away_avg_ga = away["ga"] / away["matches"] if away["matches"] else 0

        rows.append({
            "team": team_name,
            "Points": points,
            "Matches": matches,
            # home
            "athome_goal_scored": home["gf"],
            "athome_goal_against": home["ga"],
            "athome_points": home["wins"]*3 + home["draws"],
            "athome_wins": home["wins"],
            "athome_draws": home["draws"],
            "athome_loses": home["losses"],
            # away
            "away_goal_scored": away["gf"],
            "away_goal_against": away["ga"],
            "away_points": away["wins"]*3 + away["draws"],
            "away_wins": away["wins"],
            "away_draws": away["draws"],
            "away_loses": away["losses"],
            # summaries
            "summary_home": f'{matches}M {stats["wins"]}W {stats["draws"]}D {stats["losses"]}L '
                            f'{gf}-{ga} ({avg_gf:.1f}-{avg_ga:.1f}) | '
                            f'Home: {home["matches"]}M {home["wins"]}W {home["draws"]}D {home["losses"]}L '
                            f'{home["gf"]}-{home["ga"]} ({home_avg_gf:.1f}-{home_avg_ga:.1f})',
            "summary_away": f'{matches}M {stats["wins"]}W {stats["draws"]}D {stats["losses"]}L '
                            f'{gf}-{ga} ({avg_gf:.1f}-{avg_ga:.1f}) | '
                            f'Away: {away["matches"]}M {away["wins"]}W {away["draws"]}D {away["losses"]}L '
                            f'{away["gf"]}-{away["ga"]} ({away_avg_gf:.1f}-{away_avg_ga:.1f})'
        })

    standings = pd.DataFrame(rows)
    standings.sort_values(["Points","athome_goal_scored","away_goal_scored"], ascending=[False,False,False], inplace=True)
    standings.reset_index(drop=True, inplace=True)
    return standings


if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))

    print("Downloading schedule..")
    next_match = upcoming('https://www.football-data.co.uk/fixtures.csv')
    #next_match = load_fixtures_rapidapi()

    # Fetch window: today from the 08:00 cutoff onward, plus tomorrow up
    # to 08:00 (see date_utils.py). Previously this filtered to `Date ==
    # tomorrow` exactly -- a run on day X only ever fetched day X+1's
    # fixtures, never day X's own daytime/evening matches, and everything
    # in day X+1 (even matches well after the cutoff) got swept into that
    # batch instead of being left for day X+1's own run.
    next_match = next_match[date_utils.in_fetch_window(next_match['Date'], next_match['Time'])]

    if next_match.empty:
        print("No fixtures in today's window.. Bye")
        sys.exit()

    fromdate = min(next_match['Date']).strftime('%d%m%Y')
    todate = max(next_match['Date']).strftime('%d%m%Y')
    DATANAME = DATANAME.replace('{date1}', fromdate).replace('{date2}', todate) + '.csv'

    print(f"Running for each league.. ({len(LEAGUES)})")
    results_df = pd.DataFrame()
    for key in LEAGUES:

        div_df = pd.DataFrame()
        divis = LEAGUES[key]

        if (divis in next_match['Div'].unique()) == False:
            print(f"WARNING: No match to simulate for {divis}..")
            continue

        prefix = "https://www.football-data.co.uk/"
        pre = F"mmz4281/{YEAR}/{divis}.csv"
        path = prefix + pre
        print(f"Downloading {divis} data.. ({path})")
        try:
            league_data = download_league_data(path)
        except Exception as e:
            print(f"ERROR: Error during downloading {divis} data.. ({e})")
            continue

        print(f"Calculating standings for {divis}..")
        Standings = {}
        try:
            standings_df = calc_standings(league_data)
        except Exception as e:
            print(f"ERROR: Error during calculating standings for {divis}.. ({e})")
            continue

        print(f"Calculating parameters for {divis}..")
        try:
            teams_sorted = np.sort(league_data['HomeTeam'].unique())
            warm_start = load_cached_params(divis, teams_sorted)
            params = solve_parameters_decay(league_data, init_vals=warm_start)
            save_cached_params(divis, params)
        except Exception as e:
            print(f"ERROR: Error during calculating parameters for {divis}.. ({e})")
            continue

        print(f"Simulating matches for {divis}..")
        for match in next_match.loc[next_match['Div']==divis].index:
            ht = next_match['HomeTeam'][match]
            at = next_match['AwayTeam'][match]
            mdate = next_match['Date'][match]
            mtime = next_match['Time'][match]
            # This fixture's own prices, carried through from the
            # fixtures feed so each prediction row can be priced.
            match_odds = {col: next_match[col][match] for col in odds_utils.ODD_COLUMNS}

            try:
                result = dixon_coles_simulate_match(params, ht, at)
            except Exception as e:
                print(f"ERROR: Issue encountered during simulation of {ht, at} ({e})")
                continue

            res = resultdef(result, ht, at, divis, mdate, mtime, standings_df, league_data,
                            odds=match_odds)
            results_df = pd.concat([results_df, res])
            div_df = pd.concat([div_df, res])


        try:
            print(f"{divis} completed. Appending data to csv..")
            save_results_(div_df)
        except Exception as e:
            print(f"CRITICAL: Issue during saving of {divis}.. ({e})")

    print('Simulation completed..')