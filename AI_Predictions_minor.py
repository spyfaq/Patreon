#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import  sys, os, datetime
from scipy.stats import poisson
from scipy.optimize import minimize
from jsonlogger_class import JSONLogger


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
    'Russia' : 'RUS',
    'Sweden' : 'SWE',
    'Switzerland' : 'SWZ',
    'USA' : 'USA'
}

"""
Path to save  data
"""
DATAPATH = 'predictions_data/'
DATANAME = 'my_prediction_minor_data_{date1}_{date2}'
LOGPATH = 'logs/mine/'
LOGNAME = '{date}_my_prediction_minor_logs'


def calc_means(param_dict, homeTeam, awayTeam):
    return [np.exp(param_dict['attack_' + homeTeam] + param_dict['defence_' + awayTeam] + param_dict['home_adv']),
            np.exp(param_dict['defence_' + homeTeam] + param_dict['attack_' + awayTeam])]

def rho_correction(x, y, lambda_x, mu_y, rho):
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
    team_avgs = calc_means(params_dict, homeTeam, awayTeam)
    team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals + 1)] for team_avg in team_avgs]
    output_matrix = np.outer(np.array(team_pred[0]), np.array(team_pred[1]))
    correction_matrix = np.array([[rho_correction(home_goals, away_goals, team_avgs[0],
                                                  team_avgs[1], params_dict['rho']) for away_goals in range(2)]
                                  for home_goals in range(2)])
    output_matrix[:2, :2] = output_matrix[:2, :2] * correction_matrix
    if np.sum(output_matrix) > 1:
        logger.log('error', f"Data for {homeTeam} - {awayTeam} are not accurate. Probability > 1", info=str(np.sum(output_matrix)))
    return output_matrix

def solve_parameters_decay(dataset, xi=0, debug=False, init_vals=None, options={'disp': True, 'maxiter': 100},
                           constraints=[{'type': 'eq', 'fun': lambda x: sum(x[:20]) - 20}], **kwargs):

    teams = np.sort(dataset['HomeTeam'].unique())
    # check for no weirdness in dataset
    away_teams = np.sort(dataset['AwayTeam'].unique())
    if not np.array_equal(teams, away_teams):
        raise ValueError("something not right")
    n_teams = len(teams)
    if init_vals is None:
        # random initialisation of model parameters
        init_vals = np.concatenate((np.random.uniform(0, 1, (n_teams)),  # attack strength
                                    np.random.uniform(0, -1, (n_teams)),  # defence strength
                                    np.array([0, 1.0])  # rho (score correction), gamma (home advantage)
                                    ))

    def dc_log_like_decay(x, y, alpha_x, beta_x, alpha_y, beta_y, rho, gamma, t, xi=xi):
        lambda_x, mu_y = np.exp(alpha_x + beta_y + gamma), np.exp(alpha_y + beta_x)
        value = np.exp(-xi * t) * (np.log(rho_correction(x, y, lambda_x, mu_y, rho)) +
                                  np.log(poisson.pmf(x, lambda_x)) + np.log(poisson.pmf(y, mu_y)))
        return value

    def estimate_paramters(params):
        score_coefs = dict(zip(teams, params[:n_teams]))
        defend_coefs = dict(zip(teams, params[n_teams:(2 * n_teams)]))
        rho, gamma = params[-2:]
        log_like = [
            dc_log_like_decay(row.HomeGoals, row.AwayGoals, score_coefs[row.HomeTeam], defend_coefs[row.HomeTeam],
                              score_coefs[row.AwayTeam], defend_coefs[row.AwayTeam],
                              rho, gamma, row.time_diff, xi=xi) for row in dataset.itertuples()]
        return -sum(log_like)

    sys.stdout = open(os.devnull, 'w')
    opt_output = minimize(estimate_paramters, init_vals, options=options, constraints=constraints)
    sys.stdout = sys.__stdout__
    if debug:
        # sort of hacky way to investigate the output of the optimisation process
        return opt_output
    else:
        return dict(zip(["attack_" + team for team in teams] +
                        ["defence_" + team for team in teams] +
                        ['rho', 'home_adv'],
                        opt_output.x))

def resultdef(result, ht, at, divis, mdata, mtime, standings, old_df, THRESH = 0.3):
    under3_5 = result[0][0] + result[0][1] + result[0][2] + result[1][2] + result[0][3] + result[1][0] + result[1][1] + \
               result[2][0] + result[2][1] + result[3][0]
    under2_5 = result[0][0] + result[0][1] + result[0][2] + result[1][0] + result[1][1] + result[2][0]
    under1_5 = result[0][0] + result[0][1] + result[1][0]
    over3_5 = 1 - under3_5
    over2_5 = 1 - under2_5
    over1_5 = 1 - under1_5

    home = np.sum(np.tril(result, -1))
    away = np.sum(np.triu(result, 1))
    draw = np.sum(np.diag(result))
    hO0_5 = result.sum(axis=1)[1] + result.sum(axis=1)[2] + result.sum(axis=1)[3] + result.sum(axis=1)[4] + result.sum(axis=1)[5] 
    hO1_5 = result.sum(axis=1)[2] + result.sum(axis=1)[3] + result.sum(axis=1)[4] + result.sum(axis=1)[5] 
    hO2_5 = result.sum(axis=1)[3] + result.sum(axis=1)[4] + result.sum(axis=1)[5]

    aO0_5 = result.sum(axis=0)[1] + result.sum(axis=0)[2] + result.sum(axis=0)[3] + result.sum(axis=0)[4] + result.sum(axis=0)[5]
    aO1_5 = result.sum(axis=0)[2] + result.sum(axis=0)[3] + result.sum(axis=0)[4] + result.sum(axis=0)[5]
    aO2_5 = result.sum(axis=0)[3] + result.sum(axis=0)[4] + result.sum(axis=0)[5]

    temp = np.delete(result, 0, 1)
    goalgoal = np.delete(temp, 0, 0)
    gg = np.sum(goalgoal)

    dict = {'O1_5': over1_5,
            'O2_5': over2_5,
            'O3_5': over3_5,
            '1':home,
            '2':away,
            'X': draw,
            'GG': gg,
            'hO0_5': hO0_5,
            'hO1_5': hO1_5,
            'hO2_5': hO2_5,
            'aO0_5': aO0_5,
            'aO1_5': aO1_5,
            'aO2_5': aO2_5,
            }

    outcome = pd.DataFrame(columns=['Division', 'Date', 'Time', 'HomeTeam', 'AwayTeam', 'Prediction', 'Prediction %', 'History %',
                                    'Outcome', 'HG', 'AG',
                                    'HT_Points', 'HT_Matches', 'HT_athome_goal_scored', 'HT_athome_goal_against',
                                    'HT_athome_points', 'HT_athome_wins', 'HT_athome_draws', 'HT_athome_loses',
                                    'AT_Points', 'AT_Matches', 'AT_away_goal_scored', 'AT_away_goal_against',
                                    'AT_away_points', 'AT_away_wins', 'AT_away_draws', 'AT_away_loses'])
    
    logger.log('info', "Calculating class history", info=str(f'{ht}-{at}'))
    hist_dict = historyfunc(path, ht, at, old_df)
    
    for res in dict.keys():
        if dict[res] > THRESH:
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
    league_data = league_data[['Home', 'Away', 'HG', 'AG', 'Res', 'time_diff']]
    league_data = league_data.rename(columns={'HG': 'HomeGoals', 'AG': 'AwayGoals', 'Home': 'HomeTeam', 'Away': 'AwayTeam', 'Res': 'FTR'})
    

    return (league_data, old_league)

def upcoming(uri):
    next_match = pd.read_csv(uri, encoding='cp1252')
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

    towrite['Date'] = pd.to_datetime(towrite['Date'], dayfirst=True)
    towrite['Date'] = towrite['Date'].dt.strftime('%d-%m-%Y, %A')
    towrite['Date_temp'] = pd.to_datetime(towrite['Date'], dayfirst=True)
    towrite['Time_temp'] = pd.to_datetime(towrite['Time']).dt.time
    towrite['Datetime_temp'] = towrite.apply(lambda x: pd.Timestamp.combine(x['Date_temp'], x['Time_temp']), axis=1)
    towrite.sort_values(by=['Datetime_temp', 'HomeTeam'], inplace=True)
    towrite.drop(columns=['Date_temp', 'Time_temp', 'Datetime_temp'],inplace=True)

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
    next_match = upcoming('https://www.football-data.co.uk/new_league_fixtures.csv')
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
            params = solve_parameters_decay(league_data)
        except Exception as e:
            logger.log('error', f"Simulating problem.. skipping {divis}.. ")
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

            res = resultdef(result, ht, at, divis, mdate, mtime, standings_df, old_data)
            results_df = pd.concat([results_df, res])
            div_df = pd.concat([div_df, res])
        
        try:
            logger.log('info', f"{divis} completed. Appending data to csv..")
            save_results_(div_df)
        except Exception as e:
            logger.log('critical', f"Issue during saving of {divis}..", info=str(e))

    logger.log('info', 'Simulation completed..')