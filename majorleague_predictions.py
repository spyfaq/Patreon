#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import  sys, os, datetime, requests
from scipy.stats import poisson
from scipy.optimize import minimize
from jsonlogger_class import JSONLogger

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
LOGPATH = 'logs/simu/'
LOGNAME = '{date}_my_prediction_major_logs'


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

def solve_parameters_decay(dataset, xi=0.0, debug=False, init_vals=None, options={'disp': False, 'maxiter': 200},
                           constraints=None, reg=0.05, restarts=3, bounds_scale=3.0, **kwargs):
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

    def make_init():
        return np.concatenate((np.random.normal(0, 0.2, n_teams),
                               np.random.normal(0, 0.2, n_teams),
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

def resultdef(result, ht, at, divis, mdata, mtime, standings, lgdata, THRESH = 0.4):
    max_g = result.shape[0] - 1
    max_g_away = result.shape[1] - 1
    
    over1_5 = np.sum(result[i, j] for i in range(max_g + 1) for j in range(max_g_away + 1) if i + j > 1)
    over2_5 = np.sum(result[i, j] for i in range(max_g + 1) for j in range(max_g_away + 1) if i + j > 2)
    over3_5 = np.sum(result[i, j] for i in range(max_g + 1) for j in range(max_g_away + 1) if i + j > 3)

    home = np.sum(np.tril(result, -1))
    away = np.sum(np.triu(result, 1))
    draw = np.sum(np.diag(result))

    gg = np.sum(result[i, j] for i in range(1, max_g + 1) for j in range(1, max_g_away + 1))

    hO1_5 = np.sum(result[i, j] for i in range(2, max_g + 1) for j in range(max_g_away + 1))
    hO2_5 = np.sum(result[i, j] for i in range(3, max_g + 1) for j in range(max_g_away + 1))

    aO1_5 = np.sum(result[i, j] for i in range(max_g + 1) for j in range(2, max_g_away + 1))
    aO2_5 = np.sum(result[i, j] for i in range(max_g + 1) for j in range(3, max_g_away + 1))


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

    outcome = pd.DataFrame(columns=['Division', 'Date', 'Time', 'HomeTeam', 'AwayTeam', 'Prediction', 'Prediction %', 'History %',
                                    'Outcome', 'HG', 'AG',
                                    'HT_Points', 'HT_Matches', 'HT_athome_goal_scored', 'HT_athome_goal_against',
                                    'HT_athome_points', 'HT_athome_wins', 'HT_athome_draws', 'HT_athome_loses',
                                    'AT_Points', 'AT_Matches', 'AT_away_goal_scored', 'AT_away_goal_against',
                                    'AT_away_points', 'AT_away_wins', 'AT_away_draws', 'AT_away_loses'])
    for res in dict.keys():
        if dict[res] > THRESH:
            logger.log('info', "Calculating class history", info=str(f'{ht}-{at}'))
            hist_dict = historyfunc(path, ht, at)
            form_df = calculate_win_and_goal_form(lgdata)
            try:
                hist_perc = hist_dict[res]
            except:
                logger.log('warning', f"No history data for {ht}-{at}",)
                hist_perc = '-'

            hmcol = ['Points', 'Matches', 'athome_goal_scored', 'athome_goal_against', 'athome_points', 'athome_wins',
                     'athome_draws', 'athome_loses', 'team']
            homestats = standings[hmcol].loc[standings['team']==ht]
            homestats = homestats.drop('team', axis=1)
            homestats = homestats.add_prefix('HT_')
            homestats=homestats.squeeze()


            awcol = ['Points', 'Matches','away_goal_scored', 'away_goal_against', 'away_points', 'away_wins',
                        'away_draws', 'away_loses', 'team']
            awaystats = standings[awcol].loc[standings['team'] == at]
            awaystats = awaystats.drop('team', axis=1)
            awaystats = awaystats.add_prefix('AT_')
            awaystats = awaystats.squeeze()

            tempser = pd.Series([divis, mdata, mtime, ht, at, res, dict[res].round(2), hist_perc, '','',''])
            tempser = pd.concat([tempser, homestats, awaystats])
            tempser = tempser.tolist()

            outcome.loc[len(outcome)] = tempser
            
            # Merge form data for home and away teams
            merged = outcome.merge(form_df, left_on='HomeTeam', right_on='team', suffixes=('', '_home'))
            merged = merged.merge(form_df, left_on='AwayTeam', right_on='team', suffixes=('_home', '_away'))

            # Function to select correct form based on prediction type
            def pick_form(row):
                if row['Prediction'] in ['1', '2', 'X']:
                    return pd.Series([row['HomeWinForm_home'], row['AwayWinForm_away']])
                elif row['Prediction'] in ['O1_5', 'O2_5', 'O3_5', 'hO1_5', 'hO2_5', 'aO1_5', 'aO2_5']:
                    return pd.Series([row['HomeGoalsForm_home'], row['AwayGoalsForm_away']])
                else:
                    return pd.Series([None, None])

            merged[['HomeForm', 'AwayForm']] = merged.apply(pick_form, axis=1)

            # Final result
            result = merged[["Division", "Date", "Time", "HomeTeam", "AwayTeam", "Prediction", "Prediction %", 
                             "History %", "Outcome", "HG", "AG", "HT_Points", "HT_Matches", "HT_athome_goal_scored", 
                             "HT_athome_goal_against", "HT_athome_points", "HT_athome_wins", "HT_athome_draws", "HT_athome_loses", 
                             "AT_Points", "AT_Matches", "AT_away_goal_scored", "AT_away_goal_against", "AT_away_points", "AT_away_wins", 
                             "AT_away_draws", "AT_away_loses", "HomeForm", "AwayForm"]]

    return(result)

def download_league_data(url):
    league_data = pd.read_csv(url, encoding='utf-8-sig')
    league_data['Date'] = pd.to_datetime(league_data['Date'], format='%d/%m/%Y')
    league_data['time_diff'] = (league_data['Date'].max() - league_data['Date']).dt.days
    league_data = league_data[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'time_diff', 'Date']]
    league_data = league_data.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals'})

    return (league_data)

def upcoming(uri):
    next_match = pd.read_csv(uri, encoding='utf-8-sig')
    next_match = next_match[['Date','Time','Div','HomeTeam','AwayTeam']]
    next_match['Date'] = pd.to_datetime(next_match['Date'], format='%d/%m/%Y')
    return next_match

def load_fixtures_rapidapi():
    url = "https://api-football-v1.p.rapidapi.com/v3/fixtures"
    headers = {
        "x-rapidapi-key": "1bf4766257mshe9c8904f8a1cd83p10743cjsnd805bdb2ddc1",
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
    except:
        logger.log('error', f"Issue converting date.. Saving without sorting..")
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

def calc_standings(league_data):
    standings = dict()

    for team in league_data['HomeTeam']:
        temp = league_data.loc[league_data['HomeTeam'] == team]['FTR'].value_counts()
        try:
            lose = temp['A']
        except:
            lose = 0

        try:
            win = temp['H']
        except:
            win = 0

        try:
            draw = temp['D']
        except:
            draw = 0

        try:
            Standings[team].update({'Home':
                                {'Win': win,
                                    'Draw': draw,
                                    'Lose': lose,
                                    'Scored':(league_data.loc[league_data['HomeTeam'] == team]['HomeGoals']).sum(),
                                    'Eaten':(league_data.loc[league_data['HomeTeam'] == team]['AwayGoals']).sum(),
                                    'Points': (win * 3) + draw,
                                    }
                            })
        except KeyError:
            Standings[team] = {'Home':
                                {'Win': win,
                                    'Draw': draw,
                                    'Lose': lose,
                                    'Scored':(league_data.loc[league_data['HomeTeam'] == team]['HomeGoals']).sum(),
                                    'Eaten':(league_data.loc[league_data['HomeTeam'] == team]['AwayGoals']).sum(),
                                    'Points': (win * 3) + draw,
                                    'Matches': win + draw + lose
                                    }
                            }            

    for team in league_data['AwayTeam'] :
        temp = league_data.loc[league_data['AwayTeam'] == team]['FTR'].value_counts()
        try:
            lose = temp['H']
        except:
            lose = 0

        try:
            win = temp['A']
        except:
            win = 0

        try:
            draw = temp['D']
        except:
            draw = 0

        try:
            Standings[team].update({'Away':
                                {'Win': win,
                                    'Draw': draw,
                                    'Lose': lose,
                                    'Scored':(league_data.loc[league_data['AwayTeam'] == team]['AwayGoals']).sum(),
                                    'Eaten': (league_data.loc[league_data['AwayTeam'] == team]['HomeGoals']).sum(),
                                    'Points': (win * 3) + draw
                                    }
                            })
        except KeyError:
            Standings[team] = {'Away':
                                {'Win': win,
                                    'Draw': draw,
                                    'Lose': lose,
                                    'Scored':(league_data.loc[league_data['AwayTeam'] == team]['AwayGoals']).sum(),
                                    'Eaten': (league_data.loc[league_data['AwayTeam'] == team]['HomeGoals']).sum(),
                                    'Points': (win * 3) + draw,
                                    'Matches': win + draw + lose
                                    }
                            }   

    for team in Standings.keys():
        Standings[team].update({'Sum':
                                    {
                                    'Win': Standings[team].get('Home',{}).get('Win', 0) + Standings[team].get('Away',{}).get('Win', 0),
                                    'Draw': Standings[team].get('Home',{}).get('Draw', 0) + Standings[team].get('Away',{}).get('Draw', 0),
                                    'Lose': Standings[team].get('Home',{}).get('Lose', 0) + Standings[team].get('Away',{}).get('Lose', 0),
                                    'Scored': Standings[team].get('Home',{}).get('Scored', 0) + Standings[team].get('Away',{}).get('Scored', 0),
                                    'Eaten': Standings[team].get('Home',{}).get('Eaten', 0) + Standings[team].get('Away',{}).get('Eaten', 0),
                                    'Points': Standings[team].get('Home',{}).get('Points', 0) + Standings[team].get('Away',{}).get('Points', 0)
                                    }
                                })

        standings[team] = { 'Points': Standings[team]['Sum']['Points'],
                            'Matches': (Standings[team].get('Home',{}).get('Win', 0) + Standings[team].get('Home',{}).get('Draw', 0) + Standings[team].get('Home',{}).get('Lose', 0) +
                                        Standings[team].get('Away',{}).get('Win', 0) + Standings[team].get('Away',{}).get('Draw', 0) + Standings[team].get('Away',{}).get('Lose', 0)),
                            'athome_goal_scored': Standings[team].get('Home',{}).get('Scored', 0),
                            'athome_goal_against': Standings[team].get('Home',{}).get('Eaten', 0),
                            'athome_points': Standings[team].get('Home',{}).get('Points', 0),
                            'athome_wins': Standings[team].get('Home',{}).get('Win', 0),
                            'athome_draws': Standings[team].get('Home',{}).get('Draw', 0),
                            'athome_loses': Standings[team].get('Home',{}).get('Lose', 0),
                            'away_goal_scored': Standings[team].get('Away',{}).get('Scored', 0),
                            'away_goal_against': Standings[team].get('Away',{}).get('Eaten', 0),
                            'away_points': Standings[team].get('Away',{}).get('Points', 0),
                            'away_wins': Standings[team].get('Away',{}).get('Win', 0),
                            'away_draws': Standings[team].get('Away',{}).get('Draw', 0),
                            'away_loses': Standings[team].get('Away',{}).get('Lose', 0)
                        }

    temp = pd.DataFrame(standings)
    standings_df = temp.transpose()
    standings_df.sort_values(['Points'], inplace=True, ascending=False)
    standings_df['team']=standings_df.index
    return(standings_df)

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
    next_match = upcoming('https://www.football-data.co.uk/fixtures.csv')
    #next_match = load_fixtures_rapidapi()
    fromdate = min(next_match['Date']).strftime('%d%m%Y')
    todate = max(next_match['Date']).strftime('%d%m%Y')
    DATANAME = DATANAME.replace('{date1}', fromdate).replace('{date2}', todate) + '.csv'
    if os.path.exists(DATAPATH + '/' +DATANAME):
        logger = JSONLogger(log_file=LOGNAME, log_dir=LOGPATH)
        logger.log('critical', "Data exists already! Forced exit app!")
        exit()
    
    if next_match['Date'].max() <= pd.Timestamp(datetime.date.today() - datetime.timedelta(days=2)):
        logger.log('info', "Nothing new.. Bye")
        sys.exit()

    logger.log('info', "Running for each league..", info=str(len(LEAGUES)))
    results_df = pd.DataFrame()
    for key in LEAGUES:
        
        div_df = pd.DataFrame()
        divis = LEAGUES[key]

        if (divis in next_match['Div'].unique()) == False:
            logger.log('warning', f"No match to simulate for {divis}..")
            continue

        prefix = "https://www.football-data.co.uk/"
        pre = F"mmz4281/{YEAR}/{divis}.csv"
        path = prefix + pre
        logger.log('info', f"Downloading {divis} data..", info=str(path))
        try:
            league_data = download_league_data(path)
        except Exception as e:
            logger.log('error', f"Error during downloading {divis} data..", info=str(e))
            continue

        logger.log('info', f"Calculating standings for {divis}..")
        Standings = {}
        try:
            standings_df = calc_standings(league_data)
        except Exception as e:
            logger.log('error', f"Error during calculating standings for {divis}..", info=str(e))   
            continue         

        logger.log('info', f"Calculating parameters for {divis}..")
        try:
            params = solve_parameters_decay(league_data)
        except Exception as e:
            logger.log('error', f"Error during calculating parameters for {divis}..", info=str(e))   
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
            
            res = resultdef(result, ht, at, divis, mdate, mtime, standings_df, league_data)
            results_df = pd.concat([results_df, res])
            div_df = pd.concat([div_df, res])

        
        try:
            logger.log('info', f"{divis} completed. Appending data to csv..")
            save_results_(div_df)
        except Exception as e:
            logger.log('critical', f"Issue during saving of {divis}..", info=str(e))

    logger.log('info', 'Simulation completed..')