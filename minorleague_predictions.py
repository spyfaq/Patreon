#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import  sys, os, datetime, warnings, json
from scipy.stats import poisson
from scipy.optimize import minimize
from jsonlogger_class import JSONLogger
import date_utils
from collections import defaultdict

"""
Running year and leagues
"""

LEAGUES = {
    'Austria' : 'AUT',
    'Argentina': 'ARG',
    'Brazil' : 'BRA',
    'Denmark' : 'DNK',
    'Finland' : 'FIN',
    'Ireland' : 'IRL',
    'Mexico' : 'MEX',
    'Norway' : 'NOR',
    'Poland' : 'POL',
    'Romania' : 'ROU',
    'Sweden' : 'SWE',
    'Switzerland' : 'SWZ',
    'USA' : 'USA'
}

DIVISIONS = {
    "AUT": "218",
    "ARG": "129",
    "BRA": "71",
    "DNK": "120",
    "FIN": "244",
    "IRL": "357",
    "MEX": "262",
    "NOR": "103",
    "POL": "106",
    "ROU": "283",
    "SWE": "113",
    "SWZ": "207",
    "USA": "253",
}

"""
Path to save  data
"""
DATAPATH = 'predictions_data/'
DATANAME = 'my_prediction_minor_data_{date1}_{date2}'
PARAMSPATH = 'model_params/'
LOGPATH = 'logs/simu/'
LOGNAME = '{date}_my_prediction_minor_logs'

warnings.filterwarnings('ignore')

def calc_means(param_dict, homeTeam, awayTeam):
    """Calculate expected goals for home and away teams with safety checks.

    Interface is identical to original: returns [lambda_home, lambda_away]
    """
    try:
        lambda_home = np.exp(param_dict['attack_' + homeTeam] + param_dict['defence_' + awayTeam] + param_dict['home_adv'])
        lambda_away = np.exp(param_dict['defence_' + homeTeam] + param_dict['attack_' + awayTeam])
    except KeyError as e:
        raise ValueError(f"Missing team parameter: {e}")

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

    total = output_matrix.sum()
    if total <= 0 or not np.isfinite(total):
        try:
            logger.log('error', f"Non-positive total probability for {homeTeam}-{awayTeam}", info=str(total))
        except Exception:
            pass
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

    # L2 regularization on attack & defence to avoid overfitting
    reg_pen = reg * (np.sum(attack ** 2) + np.sum(defence ** 2))
    # identifiability penalty: encourage mean(attack) ~ 0
    ident_penalty = ident_pen * (np.sum(attack) ** 2)

    return -ll + reg_pen + ident_penalty

def params_vector_from_dict(params_dict, teams):
    """Flatten a fitted-params dict back into the raw vector solve_parameters_decay
    optimizes over, in the same [attack..., defence..., rho, home_adv] order,
    for the given (sorted) team list. Returns None if any team is missing
    from the cached dict (e.g. a team not seen last run) -- a fresh random
    init is safer than a partially-wrong warm start in that case.
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


def solve_parameters_decay(dataset, xi=0.0018, debug=False, init_vals=None, options={'disp': False, 'maxiter': 200},
                           constraints=None, reg=0.05, restarts=3, bounds_scale=3.0, seed=42, **kwargs):
    """Estimate Dixon-Coles parameters with L2 regularization, bounds and multiple restarts.

    This function preserves the original return format (a dict mapping names to values). It
    replaces the equality constraint approach by bounded optimization + identifiability penalty
    for more robust fits.
    """
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

def resultdef(result, ht, at, divis, mdata, mtime, standings, old_df, lgdata, THRESH = 0.5):
    # THRESH raised from 0.4 -> 0.5: 40% let markets close to a coin-flip
    # through as "predictions". 0.5 is still permissive but avoids flagging
    # outcomes the model itself thinks are less likely than not.
    max_g = result.shape[0] - 1
    max_g_away = result.shape[1] - 1

    # Vectorized market sums (previously Python-loop generators wrapped in
    # np.sum -- same result via pure numpy slicing/masking, cheaper across
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

    # Bet builder: exact joint probabilities for every {1,X,2} x goal-market
    # combo, computed directly from the score grid rather than assuming
    # independence (see majorleague_predictions.py for the full rationale).
    side_masks = {'1': gi > gj, 'X': gi == gj, '2': gi < gj}
    goal_masks = {
        'O1_5': total_goals > 1, 'O2_5': total_goals > 2, 'O3_5': total_goals > 3,
        'GG': (gi >= 1) & (gj >= 1),
        'hO1_5': gi >= 2, 'hO2_5': gi >= 3,
        'aO1_5': gj >= 2, 'aO2_5': gj >= 3,
    }
    combo_dict = {}
    for side_name, side_mask in side_masks.items():
        for goal_name, goal_mask in goal_masks.items():
            combo_dict[f'{side_name}+{goal_name}'] = result[side_mask & goal_mask].sum()

    outcome = pd.DataFrame(columns=["Division", "Date", "Time", "HomeTeam", "AwayTeam", "Prediction", "Prediction %", 
                             "History %", "HomeTeam Stats", "AwayTeam Stats", "HomeForm", "AwayForm"])
    
    logger.log('info', "Calculating class history", info=str(f'{ht}-{at}'))
    hist_dict = historyfunc(path, ht, at, old_df)
    rows = []

    # Same rationale as majorleague_predictions.py: combos are intersections
    # so they're inherently lower-probability than either leg alone; this is
    # just a sanity floor, real ranking happens in best_bets_selector.py.
    COMBO_THRESH = 0.10

    for res in list(dict.keys()) + list(combo_dict.keys()):
        is_combo = res in combo_dict
        val = combo_dict[res] if is_combo else dict[res]
        this_thresh = COMBO_THRESH if is_combo else THRESH
        if val > this_thresh:
            try:
                hist_perc = hist_dict[res]
            except:
                if not is_combo:
                    logger.log('warning', f"No history data for {ht}-{at}",)
                hist_perc = '-'

            homestats = standings.loc[standings['team'] == ht, 'summary_home'].squeeze()
            awaystats = standings.loc[standings['team'] == at, 'summary_away'].squeeze()

            rows.append([divis, mdata, mtime, ht, at, res, val.round(2), hist_perc, homestats, awaystats, '', ''])

    if rows:
        outcome = pd.DataFrame(rows, columns=["Division", "Date", "Time", "HomeTeam", "AwayTeam", "Prediction", "Prediction %",
                                 "History %", "HomeTeam Stats", "AwayTeam Stats", "HomeForm", "AwayForm"])

        form_df = calculate_win_and_goal_form(lgdata)
        merged = outcome.merge(form_df, left_on='HomeTeam', right_on='team', suffixes=('', '_home'))
        merged = merged.merge(form_df, left_on='AwayTeam', right_on='team', suffixes=('_home', '_away'))

        # Function to select correct form based on prediction type
        def pick_form(row):
            pred = row['Prediction']
            goal_leg = pred.split('+')[1] if '+' in pred else pred
            if pred in ['1', '2', 'X']:
                return pd.Series([row['HomeWinForm_home'], row['AwayWinForm_away']])
            elif goal_leg in ['O1_5', 'O2_5', 'O3_5', 'GG', 'hO1_5', 'hO2_5', 'aO1_5', 'aO2_5']:
                return pd.Series([row['HomeGoalsForm_home'], row['AwayGoalsForm_away']])
            else:
                return pd.Series([None, None])

        merged[['HomeForm', 'AwayForm']] = merged.apply(pick_form, axis=1)

        # Final result
        outcome = merged[["Division", "Date", "Time", "HomeTeam", "AwayTeam", "Prediction", "Prediction %", 
                         "History %", "HomeTeam Stats", "AwayTeam Stats", "HomeForm", "AwayForm"]]

    return(outcome)

def download_league_data(url):
    league_data = pd.read_csv(url)
    league_data['Date'] = pd.to_datetime(league_data['Date'], format='%d/%m/%Y')
    league_data['time_diff'] = (league_data['Date'].max() - league_data['Date']).dt.days

    def extract_season(season):
        try:
            if '/' in season:
                start_year, end_year = season.split('/')
                return pd.Series([int(start_year), int(end_year)])
            else:
                return pd.Series([int(season), int(season)])
        except:
            return pd.Series([int(season), int(season)])
        
    league_data[['season_start', 'season_end']] = league_data['Season'].apply(extract_season)

    # Current season is the maximum season_end
    current_season = league_data['season_end'].max()

    # Find unique seasons and sort by season_end
    unique_seasons = league_data[['Season', 'season_end']].drop_duplicates().sort_values(by='season_end', ascending=False)

    # Identify the last 5 seasons, excluding the running season
    last_5_seasons = unique_seasons['season_end'].unique()[1:6]

    # Filter the original DataFrame to include only the last 5 seasons
    old_league = league_data[league_data['season_end'].isin(last_5_seasons)]
    old_league = old_league.rename(columns={'HG': 'FTHG', 'AG': 'FTAG', 'Home': 'HomeTeam', 'Away': 'AwayTeam'})
    old_league = old_league[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'time_diff']]

    league_data = league_data[league_data['season_end'] == current_season]
    league_data = league_data[['Home', 'Away', 'HG', 'AG', 'Res', 'time_diff', 'Date']]
    league_data = league_data.rename(columns={'HG': 'HomeGoals', 'AG': 'AwayGoals', 'Home': 'HomeTeam', 'Away': 'AwayTeam', 'Res': 'FTR'})
    

    return (league_data, old_league)

def upcoming(uri):
    next_match = pd.read_csv(uri, encoding='utf-8-sig')
    next_match = next_match[['Date','Time', 'Country', 'Home','Away']]
    next_match = next_match.rename(columns={'Country': 'Div', 'Home': 'HomeTeam', 'Away': 'AwayTeam'})    
    next_match['Date'] = pd.to_datetime(next_match['Date'], format='%d/%m/%Y')
    return next_match

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
        towrite['Date'] = dt_gr.dt.strftime('%d-%m-%Y') + ', ' + dt_gr.dt.day_name(locale='en_US')
        towrite['Time'] = dt_gr.dt.strftime('%H:%M')  

        towrite['Date_temp'] = pd.to_datetime(towrite['Date'], dayfirst=True)
        towrite['Time_temp'] = pd.to_datetime(towrite['Time'], format="%H:%M").dt.time
        towrite['Datetime_temp'] = towrite.apply(lambda x: pd.Timestamp.combine(x['Date_temp'], x['Time_temp']), axis=1)
        towrite.sort_values(by=['Datetime_temp', 'HomeTeam'], inplace=True)
        towrite.drop(columns=['Date_temp', 'Time_temp', 'Datetime_temp'],inplace=True)
    except:
        logger.log('error', f"Issue converting date.. Saving without sorting..")
        
    towrite.to_csv(filename, index=False)

def historyfunc(path, hw, aw, old_df):
    """
    :return: history percentage of home win, away win, over, under
    """
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


    old_data = old_df.loc[(old_df["HomeTeam"] == hw) & (old_df["AwayTeam"] == aw)]

    for ind in old_data.index:
        try:
            # Extract relevant values
            fthg = old_data.at[ind, 'FTHG']  # Full Time Home Goals
            ftag = old_data.at[ind, 'FTAG']  # Full Time Away Goals
            
            # Win, lose, draw calculation
            if fthg > ftag:
                win += 1
            elif fthg < ftag:
                lose += 1
            else:
                draw += 1

            # Over/Under calculations
            total_goals = fthg + ftag

            if total_goals > 2:
                ov += 1
            else:
                und += 1

            if total_goals > 1.5:
                ov1_5 += 1

            if fthg > 0 and ftag > 0:
                gg += 1

            if total_goals > 3.5:
                ov3_5 += 1

            # Home goals over calculations
            if fthg > 0.5:
                hO0_5 += 1
            if fthg > 1.5:
                hO1_5 += 1
            if fthg > 2.5:
                hO2_5 += 1

            # Away goals over calculations
            if ftag > 0.5:
                aO0_5 += 1
            if ftag > 1.5:
                aO1_5 += 1
            if ftag > 2.5:
                aO2_5 += 1

        except :
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
    datesave = datetime.date.today().strftime('%Y%m%d')
    LOGNAME = LOGNAME.replace('{date}', datesave) + '.json'

    if os.path.exists(LOGPATH + '/' +LOGNAME):
        logger = JSONLogger(log_file=LOGNAME, log_dir=LOGPATH)
        logger.log('critical', "Tried to rerun! Forced exit app!")
        exit()
    else:
        logger = JSONLogger(log_file=LOGNAME, log_dir=LOGPATH)


    logger.log('info', "Downloading schedule..")
    next_match = upcoming('https://www.football-data.co.uk/new_league_fixtures.csv')

    # Fetch window: see majorleague_predictions.py for the full rationale
    # (today from the 08:00 cutoff onward, plus tomorrow up to 08:00).
    next_match = next_match[date_utils.in_fetch_window(next_match['Date'], next_match['Time'])]

    if next_match.empty:
        logger.log('info', "No fixtures in today's window.. Bye")
        sys.exit()

    fromdate = min(next_match['Date']).strftime('%d%m%Y')
    todate = max(next_match['Date']).strftime('%d%m%Y')
    DATANAME = DATANAME.replace('{date1}', fromdate).replace('{date2}', todate) + '.csv'
    if os.path.exists(DATAPATH + '/' +DATANAME):
        logger = JSONLogger(log_file=LOGNAME, log_dir=LOGPATH)
        logger.log('critical', "Data exists already! Forced exit app!")
        exit()

    logger.log('info', "Running for each league..", info=str(len(LEAGUES)))
    results_df = pd.DataFrame()
    for key in LEAGUES:
        
        div_df = pd.DataFrame()
        divi = LEAGUES[key]
        divis = key

        if (key in next_match['Div'].unique()) == False:
            logger.log('warning', f"No match to simulate for {divis}..")
            continue

        prefix = "https://www.football-data.co.uk/"
        pre = F"new/{divi}.csv"
        path = prefix + pre
        logger.log('info', f"Downloading {divis} data..", info=str(path))
        league_data, old_data = download_league_data(path)

        logger.log('info', f"Calculating standings for {divis}..")
        Standings = {}
        standings_df = calc_standings(league_data)

        logger.log('info', f"Calculating parameters for {divis}..")
        try:
            teams_sorted = np.sort(league_data['HomeTeam'].unique())
            warm_start = load_cached_params(divis, teams_sorted)
            params = solve_parameters_decay(league_data, init_vals=warm_start)
            save_cached_params(divis, params)
        except Exception as e:
            logger.log('error', f"Simulating problem.. skipping {divis}.. ", info=str(e))
            continue

        logger.log('info', f"Simulating matches for {divis}..")
        for match in next_match.loc[next_match['Div']==divis].index:
            ht = next_match['HomeTeam'][match]
            at = next_match['AwayTeam'][match]
            mdate = next_match['Date'][match]
            mtime = next_match['Time'][match]

            try:
                result = dixon_coles_simulate_match(params, ht, at)
            except Exception as e:
                logger.log('error', f"Issue encountered during simulation of {ht, at}", info=str(e))
                continue

            res = resultdef(result, ht, at, divis, mdate, mtime, standings_df, old_data, league_data)
            results_df = pd.concat([results_df, res])
            div_df = pd.concat([div_df, res])
        
        try:
            logger.log('info', f"{divis} completed. Appending data to csv..")
            save_results_(div_df)
        except Exception as e:
            logger.log('critical', f"Issue during saving of {divis}..", info=str(e))

    logger.log('info', 'Simulation completed..')