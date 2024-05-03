#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import datetime, os, time
from jsonlogger_class import JSONLogger
import plotly.graph_objs as go
from plotly.subplots import make_subplots

LOGPATH = 'logs/data/'
LOGNAME = '{date}_accuracy_logs'
PUBLISHPATH = 'publish/'

def newest_predictions() -> str:
    logger.log('info', 'Searching latest prediction file..')
    files = os.listdir(PUBLISHPATH)

    paths = []
    for basename in files:
       if basename[0:4] == 'Tier':
        paths.append(os.path.join(PUBLISHPATH, basename))

    try:
        file = max(paths, key=os.path.getctime)
        logger.log('info', f'File found..', file)
        return file.split('_')[-1]
    except:
        logger.log('error', f'File not found..')
        return('\\99999999')

def download_league_data():
    logger.log('info', 'Creating the urls needed..')
    current_month = datetime.datetime.now().month
    current_year = datetime.datetime.now().year

    previous_year = current_year - 1
    next_year = current_year + 1

    if current_month >= 9:  # September to December
        checkyear = int(str(current_year)[2:] + str(next_year)[2:])
    else:  # January to August
        checkyear = int(str(previous_year)[2:] + str(current_year)[2:])

    logger.log('info', 'Downloading results..')
    league_list = ['E0', 'E1', 'D1', 'I1', 'SP1', 'F1', 'N1', 'B1', 'P1', 'G1', 'D2', 'I2', 'SP2', 'F2', 'E2', 'SC0']
    prefix = "https://www.football-data.co.uk/"
    dfs = []
    for lg in league_list:
        pre = F"mmz4281/{checkyear}/{lg}.csv"
        path = prefix + pre
        df = pd.read_csv(path, encoding='latin1')
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    league_data = combined_df[['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']]
    league_data['Date'] = pd.to_datetime(league_data['Date'], format='%d/%m/%Y').dt.strftime('%d-%m-%Y, %A')
    return (league_data) 

def fetchaccuracy(date, results):
   logger.log('info', 'Latest published predictions loaded..')
   tier1_df = pd.read_csv(PUBLISHPATH + f'Tier1_{date}')
   tier2_df = pd.read_csv(PUBLISHPATH + f'Tier2_{date}')
   
   logger.log('info', 'Checking accuracy predictions..')
   def check_accuracy(row):
        if (np.isnan(row['FTHG'])):
            return np.NaN
        
        if row['Prediction'] == '1' and row['FTHG'] > row['FTAG']:
            return True
        elif row['Prediction'] == '2' and row['FTHG'] < row['FTAG']:
            return True
        elif row['Prediction'] == 'X' and row['FTHG'] == row['FTAG']:
            return True
        
        elif row['Prediction'] == 'O1_5' and row['FTHG'] + row['FTAG'] > 1.5:
            return True
        elif row['Prediction'] == 'O2_5' and row['FTHG'] + row['FTAG'] > 2.5:
            return True       
        elif row['Prediction'] == 'O3_5' and row['FTHG'] + row['FTAG'] > 3.5:
            return True   

        elif row['Prediction'] == 'hO1_5' and row['FTHG'] > 1.5:
            return True               
        elif row['Prediction'] == 'hO2_5' and row['FTHG'] > 2.5:
            return True          
        
        elif row['Prediction'] == 'aO1_5' and row['FTAG'] > 1.5:
            return True               
        elif row['Prediction'] == 'aO2_5' and row['FTAG'] > 2.5:
            return True    
        else:
            return False

   tier1 = pd.merge(tier1_df, results, on=['HomeTeam', 'AwayTeam', 'Date'], how='left')
   tier1.drop(['Div'], axis=1, inplace=True)
   tier1['Prediction_Accuracy'] = tier1.apply(check_accuracy, axis=1)

   tier2 = pd.merge(tier2_df, results, on=['HomeTeam', 'AwayTeam', 'Date'], how='left')
   tier2.drop(['Div'], axis=1, inplace=True)
   tier2['Prediction_Accuracy'] = tier2.apply(check_accuracy, axis=1)

   tier1.to_csv(PUBLISHPATH + f'Tier1_updated_{date}', index=False)
   tier2.to_csv(PUBLISHPATH + f'Tier2_updated_{date}', index=False)
   return tier1, tier2

def niceplots(tier1, tier2):
    logger.log('info', 'Creating plots per Tier..')
    tier1_counts = tier1['Prediction_Accuracy'].value_counts().sort_index()
    tier2_counts = tier2['Prediction_Accuracy'].value_counts().sort_index()

    color_map = {True: '#3CB371', False: '#CCCCCC'}
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Tier 1 Prediction Accuracy', 'Tier 2 Prediction Accuracy'), specs=[[{'type':'domain'}, {'type':'domain'}]])
    fig.add_trace(go.Pie(
        labels=tier1_counts.index, values=tier1_counts.values, 
        name='Tier 1', sort=False, 
        marker=dict(colors=[color_map[label] for label in tier1_counts.keys()])), 
        row=1, col=1)
    
    fig.add_trace(go.Pie(
        labels=tier2_counts.index, values=tier2_counts.values, 
        name='Tier 2', sort=False, 
        marker=dict(colors=[color_map[label] for label in tier2_counts.keys()])), 
        row=1, col=2)
    
    fig.update_traces(hole=0.4, textinfo='percent+label')
    fig.update_layout(showlegend=False)
    fig.write_image(PUBLISHPATH+f"tiers_accuracy_plot_{datesave}.png")

    logger.log('info', 'Creating plots per prediction..')
    tier1_accuracy = tier1.groupby('Prediction')['Prediction_Accuracy'].mean().reset_index()
    tier2_accuracy = tier2.groupby('Prediction')['Prediction_Accuracy'].mean().reset_index()
    tier1_accuracy['Prediction_Accuracy'] *= 100
    tier2_accuracy['Prediction_Accuracy'] *= 100

    fig = make_subplots(rows=1, cols=2, subplot_titles=('Tier 1 Accuracy per Prediction', 'Tier 2 Accuracy per Prediction'))
    fig.add_trace(go.Bar(
        x=tier1_accuracy['Prediction'], y=tier1_accuracy['Prediction_Accuracy'], 
        text=tier1_accuracy['Prediction_Accuracy'], textposition='inside', texttemplate='%{text:.0s}', marker_color='#3CB371',
        name='Tier 1'), row=1, col=1)
    
    fig.add_trace(go.Bar(
        x=tier2_accuracy['Prediction'], y=tier2_accuracy['Prediction_Accuracy'], 
        text=tier2_accuracy['Prediction_Accuracy'], textposition='inside', texttemplate='%{text:.0s}', marker_color='#3CB371', 
        name='Tier 2'), row=1, col=2)
    
    fig.update_layout(title='Accuracy per Prediction Category', showlegend=False)
    fig.update_xaxes(title_text='Prediction', row=1, col=1)
    fig.update_xaxes(title_text='Prediction', row=1, col=2)
    fig.update_yaxes(title_text='Accuracy %', row=1, col=1)
    fig.write_image(PUBLISHPATH+f"predictions_accuracy_plot_{datesave}.png")

    logger.log('info', 'Creating plots per league..')
    tier1_accuracy = tier1.groupby('Division')['Prediction_Accuracy'].mean().reset_index()
    tier2_accuracy = tier2.groupby('Division')['Prediction_Accuracy'].mean().reset_index()
    tier1_accuracy['Prediction_Accuracy'] *= 100
    tier2_accuracy['Prediction_Accuracy'] *= 100

    fig = make_subplots(rows=1, cols=2, subplot_titles=('Tier 1 Accuracy per League', 'Tier 2 Accuracy per League'))
    fig.add_trace(go.Bar(
        x=tier1_accuracy['Division'], y=tier1_accuracy['Prediction_Accuracy'], 
        text=tier1_accuracy['Prediction_Accuracy'], textposition='inside', texttemplate='%{text:.0s}', marker_color='#3CB371',
        name='Tier 1'), row=1, col=1)
    
    fig.add_trace(go.Bar(
        x=tier2_accuracy['Division'], y=tier2_accuracy['Prediction_Accuracy'], 
        text=tier2_accuracy['Prediction_Accuracy'], textposition='inside', texttemplate='%{text:.0s}', marker_color='#3CB371', 
        name='Tier 2'), row=1, col=2)
    
    fig.update_layout(title='Accuracy per League', showlegend=False)
    fig.update_xaxes(title_text='League', row=1, col=1)
    fig.update_xaxes(title_text='League', row=1, col=2)
    fig.update_yaxes(title_text='Accuracy %', row=1, col=1)
    fig.write_image(PUBLISHPATH+f"division_accuracy_plot_{datesave}.png")

def main():
    filename = newest_predictions()
    results = download_league_data()
    t1df, t2df = fetchaccuracy(filename, results)
    niceplots(t1df, t2df)
    logger.log('info', f'Process completed.. Files are available..', PUBLISHPATH)
    return

if __name__ == '__main__':
    os.chdir('D:\\Python Apps\\Patreon')
    datesave = datetime.date.today().strftime('%Y%m%d')
    LOGNAME = LOGNAME.replace('{date}', datesave) + '.json'
    logger = JSONLogger(log_file=LOGNAME, log_dir=LOGPATH)

    try:
        main()
    except Exception as e:
        logger.log('critical', "Exception occured whie running", info=str(e))