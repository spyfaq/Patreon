#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import datetime, os, time
from jsonlogger_class import JSONLogger

LOGPATH = 'logs/data/'
LOGNAME = '{date}_tipsselection_logs'
DATANAME = 'my_prediction_data_{date1}_{date2}'
DATAPATH = 'predictions_data/'
PUBLISHPATH = 'publish/'

def newest_predictions() -> str:
    logger.log('info', 'Searching latest prediction file..')
    files = os.listdir(DATAPATH)

    paths = []
    for basename in files:
       if basename[0:8] == 'my_predi':
        paths.append(os.path.join(DATAPATH, basename))

    try:
        file = max(paths, key=os.path.getctime)
        logger.log('info', f'File found..', file)
        return file
    except:
        logger.log('error', f'File not found..')
        return('\\99999999')

def fetchdata(filename):
    logger.log('info', f'Loading file..', filename)
    df = pd.read_csv(filename)
    df = df[['Division', 'Date', 'Time', 'HomeTeam', 'AwayTeam', 'Prediction', 'Prediction %', 'History %', 
                   'HT_athome_goal_scored', 'HT_athome_goal_against', 'HT_athome_points', 'HT_athome_wins', 'HT_athome_draws', 'HT_athome_loses',
                   'AT_away_goal_scored', 'AT_away_goal_against', 'AT_away_points', 'AT_away_wins', 'AT_away_draws', 'AT_away_loses']]
      
    df['Hometeam GpG'] = (df['HT_athome_goal_scored'] / (df['HT_athome_wins'] + df['HT_athome_draws'] + df['HT_athome_loses'])).round(2)
    df['Hometeam WpG'] = (df['HT_athome_wins'] / (df['HT_athome_wins'] + df['HT_athome_draws'] + df['HT_athome_loses'])).round(2)
    df['Hometeam DpG'] = (df['HT_athome_draws'] / (df['HT_athome_wins'] + df['HT_athome_draws'] + df['HT_athome_loses'])).round(2)
    df['Hometeam LpG'] = (df['HT_athome_loses'] / (df['HT_athome_wins'] + df['HT_athome_draws'] + df['HT_athome_loses'])).round(2)

    df['Awayteam GpG'] = (df['AT_away_goal_scored'] / (df['AT_away_wins'] + df['AT_away_draws'] + df['AT_away_loses'])).round(2)
    df['Awayteam WpG'] = (df['AT_away_wins'] / (df['AT_away_wins'] + df['AT_away_draws'] + df['AT_away_loses'])).round(2)
    df['Awayteam DpG'] = (df['AT_away_draws'] / (df['AT_away_wins'] + df['AT_away_draws'] + df['AT_away_loses'])).round(2)
    df['Awayteam LpG'] = (df['AT_away_loses'] / (df['AT_away_wins'] + df['AT_away_draws'] + df['AT_away_loses'])).round(2)
 
    logger.log('info', f'Preparing Tier1 file..')
    tier1_df = df[df['Prediction'].str.contains('O')][['Division', 'Date', 'HomeTeam', 'AwayTeam', 'Prediction', 'Prediction %', 'History %', 'Hometeam GpG', 'Awayteam GpG']]
    excluded_values = ['hO0_5', 'aO0_5']
    mask = ~tier1_df['Prediction'].isin(excluded_values)
    tier1_df = tier1_df[mask]
    tier1_df['History %'] = tier1_df['History %'].replace('-', np.nan)
    tier1_df = tier1_df.sort_values(by=['History %', 'Hometeam GpG', 'Awayteam GpG', 'Prediction %'], ascending=False, na_position='last').groupby('Date')
    
    def filter_top_pred(group):
        top_matches = group['HomeTeam'].unique()[:10]
        return group[group['HomeTeam'].isin(top_matches)]

    filtered_df = tier1_df.apply(filter_top_pred)
    tier1_df = filtered_df.reset_index(drop=True)
    
    tier1_df.drop(columns='Prediction %', inplace=True)
    tier1_df = tier1_df.sort_values(by=['HomeTeam', 'Date'])
    tier1_df['History %'] = "'"+tier1_df['History %']
    tier1_df.to_csv(PUBLISHPATH + f'Tier1_{datesave}.csv', index=False)

    logger.log('info', f'Preparing Tier2 file..')
    tier2_df = df[(df['Prediction'] == '1') | (df['Prediction'] == '2') | (df['Prediction'] == 'X')][['Division', 'Date', 'HomeTeam', 'AwayTeam', 'Prediction', 'Prediction %', 'History %', 'Hometeam WpG', 'Hometeam DpG', 'Hometeam LpG', 'Awayteam WpG', 'Awayteam DpG', 'Awayteam LpG']]
    tier2_df['History %'] = tier2_df['History %'].replace('-', np.nan)
    
    # Define custom sorting function
    def custom_sort(group):
        if group['Prediction'].iloc[0] in ['1', '2']:
            filtered_group = group[group['Prediction %'] >= 0.5]  
            if filtered_group.empty:
                return pd.DataFrame()
            
            if group['Prediction'].iloc[0] == '1':
                sorted_group = filtered_group.sort_values(by=['History %', 'Hometeam WpG', 'Awayteam LpG', 'Prediction %'], ascending=False, na_position='last')
            elif group['Prediction'].iloc[0] == '2':
                sorted_group = filtered_group.sort_values(by=['History %', 'Hometeam LpG', 'Awayteam WpG', 'Prediction %'], ascending=False, na_position='last')

        elif group['Prediction'].iloc[0] == 'X':
            sorted_group = group.sort_values(by=['History %', 'Hometeam DpG', 'Awayteam DpG', 'Prediction %'], ascending=False)
            
        return sorted_group.head(7)

    result = tier2_df.groupby(['Date', 'Prediction']).apply(custom_sort)
    result.drop(columns='Prediction %', inplace=True)
    result['History %'] = "'"+result['History %']
    result.to_csv(PUBLISHPATH + f'Tier2_{datesave}.csv', index=False)

    logger.log('info', f'Process completed.. Files are available..', PUBLISHPATH)
    return 


def main():
    filename = newest_predictions()
    fetchdata(filename)
    return


if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    datesave = datetime.date.today().strftime('%Y%m%d')
    LOGNAME = LOGNAME.replace('{date}', datesave) + '.json'
    logger = JSONLogger(log_file=LOGNAME, log_dir=LOGPATH)

    try:
        main()
    except Exception as e:
        logger.log('critical', "Exception occured whie running", info=str(e))