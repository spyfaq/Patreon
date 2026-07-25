#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from jsonlogger_class import JSONLogger
import os, datetime, glob

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

LEAGUES2 = {
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

"""
Path to save  data
"""
DATAPATH = 'history/'
LOGPATH = 'logs/mine/'
LOGNAME = '{date}_my_results_logs'

def find_VIP_files():
    print(f'Looking for VIP files to update ..')
    # Look recursively for files named "VIP_date.xlsx" in any dated
    # subfolder under history/ (e.g. history/2026-08-01_2026-08-04/VIP_*.xlsx),
    # since predictions now accumulate across many separate pipeline runs.
    search_pattern = os.path.join(DATAPATH, "**", "VIP_*.xlsx")
    return glob.glob(search_pattern, recursive=True)

def download_league_data(url):
    try:
        league_data = pd.read_csv(url, encoding='utf-8-sig')
        league_data = league_data[['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']]
        league_data['Date'] = pd.to_datetime(league_data["Date"], format="%d/%m/%Y").dt.strftime("%d-%m-%Y")

    except:
        league_data = pd.DataFrame()
        
    return (league_data)

def download_minor_league_data(url):
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

    league_data = league_data[league_data['season_end'] == current_season]
    league_data = league_data[['Country', 'Date', 'Home', 'Away', 'HG', 'AG']]
    league_data = league_data.rename(columns={'Country': 'Div', 'HG': 'FTHG', 'AG': 'FTAG', 'Home': 'HomeTeam', 'Away': 'AwayTeam'})
    league_data['Date'] = pd.to_datetime(league_data["Date"], format="%d/%m/%Y").dt.strftime("%d-%m-%Y")

    return (league_data)    

def match_matched(results):
    def define(row):
        try:
            if row['Prediction'] == 'Over 3.5 Goals' and int(row['HG'] + row['AG']) > 3.5:
                return('TRUE')
            elif row['Prediction'] == 'Over 2.5 Goals' and int(row['HG'] + row['AG']) > 2.5:
                return('TRUE')
            elif row['Prediction'] == 'Over 1.5 Goals' and int(row['HG'] + row['AG']) > 1.5:
                return('TRUE')
            elif row['Prediction'] == 'Both Teams to Score' and int(row['HG']) > 0.5 and int(row['AG']) > 0.5 :
                return('TRUE')
            elif row['Prediction'] == 'Home Win' and int(row['HG']) > int(row['AG']):
                return('TRUE')
            elif row['Prediction'] == 'Away Win' and int(row['HG']) < int(row['AG']):
                return('TRUE')
            elif row['Prediction'] == 'Draw' and int(row['HG']) == int(row['AG']):
                return('TRUE')


            elif row['Prediction'] == 'Home team Over 1.5 Goals' and int(row['HG']) > 1:
                return('TRUE')
            elif row['Prediction'] == 'Home team Over 2.5 Goals' and int(row['HG']) > 2:
                return('TRUE')

            elif row['Prediction'] == 'Away team Over 1.5 Goals' and int(row['AG']) > 1:
                return('TRUE')
            elif row['Prediction'] == 'Away team Over 2.5 Goals' and int(row['AG']) > 2:
                return('TRUE')

            else:
                return('FALSE')
        except:
            return('')

    results['unique'] = results['Date'].astype(str) + results['HomeTeam'].astype(str) + results['AwayTeam'].astype(str)
    results.rename(columns={'FTHG': 'HG', 'FTAG': 'AG'}, inplace=True)

    filesfound = find_VIP_files()
    if not filesfound:
        print("No files to update..")
        
    for file in filesfound:
        print(f'Updating {file} ..')
        prediction_Full = pd.read_excel(file)
        prediction_Full['unique'] = prediction_Full['Date'].astype(str) + \
                                prediction_Full['HomeTeam'].astype(str) + prediction_Full['AwayTeam'].astype(str)

        final_Full = pd.merge(prediction_Full, results[['unique', 'HG', 'AG']], on=['unique'], how="left")
        final_Full.drop('unique', axis=1, inplace=True)


        final_Full['Outcome'] = final_Full.apply(lambda x: define(x), axis=1)
        final_Full = final_Full[['Division', 'Date', 'HomeTeam', 'AwayTeam', 'Prediction', 'HG', 'AG', 'Outcome', 'History H2H', 'HomeForm', 'AwayForm', 'Reasoning', 'PickedforFree']]
        
        # Check coverage
        true_false_count = final_Full['Outcome'].isin(['TRUE', 'FALSE']).sum()
        total_count = len(final_Full)
        percent_dash = (true_false_count / total_count)*100

        # Check and print result
        if percent_dash < 90:
            print(f"Too many '-' ({percent_dash:.2f}%).. Skipping {file}")
            continue
        
        folder = os.path.dirname(file)
        filename = os.path.basename(file)

        # Prepend "update_" to the original filename
        new_filename = "update_" + filename
        new_path = os.path.join(folder, new_filename)
        print(f'New file created {new_filename} ..')
        final_Full.to_excel(new_path, index=False, sheet_name="Sheet1")

        print(f'Removing {file} ..')
        os.remove(file)

if __name__ == '__main__':
    datesave = datetime.date.today().strftime('%Y%m%d')
    LOGNAME = LOGNAME.replace('{date}', datesave) + '.json'
    logger = JSONLogger(log_file=LOGNAME, log_dir=LOGPATH)

    league_data_full = pd.DataFrame()
    for key in LEAGUES:
        divis = LEAGUES[key]

        print(f'Downloading league ({divis}) data..')
        pre = F"mmz4281/{YEAR}/{divis}.csv"
        prefix = "https://www.football-data.co.uk/"
        path = prefix + pre
        league_data = download_league_data(path)
        league_data_full =pd.concat([league_data_full, league_data])

    for key in LEAGUES2:
        divis = LEAGUES2[key]

        print(f'Downloading league ({divis}) data..')
        prefix = "https://www.football-data.co.uk/"
        pre = F"new/{divis}.csv"
        path = prefix + pre
        league_data = download_minor_league_data(path)
        league_data_full =pd.concat([league_data_full, league_data])

    print(f'Evaluating Predictions ..')
    match_matched(league_data_full)
