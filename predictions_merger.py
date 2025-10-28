#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, datetime, re
import pandas as pd
from jsonlogger_class import JSONLogger


LOGPATH = 'logs/merger/'
LOGNAME = '{date}_merger_logs'

DATAPATH = 'predictions_data/'
MAJORDATANAME = 'my_prediction_major_data_{date1}_{date2}'
MINORDATANAME = 'my_prediction_minor_data_{date1}_{date2}'
DATANAME = 'my_prediction_data_{date1}_{date2}'


def newest_predictions(sever) -> str:
    logger.log('info', f'Searching latest prediction file for {sever} leagues..')
    files = os.listdir(DATAPATH)

    paths = []
    for basename in files:
       if f'my_prediction_{sever}_data_' in basename:
        paths.append(os.path.join(DATAPATH, basename))

    try:
        file = max(paths, key=os.path.getctime)
        logger.log('info', f'File found..', file)
        return file
    except:
        logger.log('warning', f'File for {sever} not found..')
        return('\\99999999')
    
def accumulate_data(majorfile, minorfile):
    logger.log('info', f'Trying to match major n minor league files..')
    
    def str_to_date(date_str):
        print(date_str)
        return datetime.datetime.strptime(date_str, '%d%m%Y')
    

    # Extract dates from the filenames
    if '99999999' not in majorfile and '99999999' not in minorfile:

        major_dates = re.findall(r'\d{8}', majorfile)
        minor_dates = re.findall(r'\d{8}', minorfile)

        # Convert dates to datetime objects
        major_start_date = str_to_date(major_dates[0])
        major_end_date = str_to_date(major_dates[1])
        minor_start_date = str_to_date(minor_dates[0])
        minor_end_date = str_to_date(minor_dates[1])

        # Check if dates are within 1 day
        if (abs((major_start_date - minor_start_date).days) <= 1 or abs((major_end_date - minor_end_date).days) <= 1):
            major_df = pd.read_csv(majorfile)
            minor_df = pd.read_csv(minorfile)
        
            concatenated_df = pd.concat([major_df, minor_df], ignore_index=True)

            logger.log('info', f'Matched major n minor league files..')
            return concatenated_df
        else:
            # Determine which file has the latest data
            if major_end_date > minor_end_date:
                latest_df = pd.read_csv(majorfile)
            else:
                latest_df = pd.read_csv(minorfile)

            logger.log('warning', f'Didnt match major n minor league files.. Keeping last file', info=latest_df)
            return latest_df
    else:
        if '99999999' in majorfile: 
            logger.log('warning', f'Only minor leagues found')
            latest_df = pd.read_csv(minorfile)
            return latest_df
        else:
            logger.log('warning', f'Only major leagues found')
            latest_df = pd.read_csv(majorfile)
            return latest_df
        
def saveto_csv(towrite):
    logger.log('info', f'Saving results..')

    core = towrite['Date'].astype(str).str.split(",", n=1).str[0].str.strip()

    # Remove the weekday name after the comma
    d_ymd = pd.to_datetime(core, format="%Y-%m-%d", errors="coerce")  # 2025-08-17
    d_dmy = pd.to_datetime(core, format="%d-%m-%Y", errors="coerce")  # 15-08-2025
    d_full = pd.to_datetime(core, format="%Y-%m-%d %H:%M:%S", errors="coerce")  # YYYY-MM-DD HH:MM:SS

    # 3) Merge results (priority: full datetime > YYYY-MM-DD > DD-MM-YYYY)
    towrite["Date"] = d_full.fillna(d_ymd).fillna(d_dmy)

    towrite['Date'] = towrite['Date'].dt.strftime('%d-%m-%Y')
    towrite['Date_temp'] = pd.to_datetime(towrite['Date'], dayfirst=True)
    towrite['Time_temp'] = (
    pd.to_datetime(towrite['Time'], errors='coerce')
    .dt.time
    .fillna(datetime.time(0, 0))  # replace NaT with 00:00
)
    towrite['Datetime_temp'] = towrite.apply(lambda x: pd.Timestamp.combine(x['Date_temp'], x['Time_temp']), axis=1)
    towrite.sort_values(by=['Datetime_temp', 'HomeTeam'], inplace=True)

    fromdate = min(towrite['Date_temp']).strftime('%d%m%Y')
    todate = max(towrite['Date_temp']).strftime('%d%m%Y')

    towrite.drop(columns=['Date_temp', 'Time_temp', 'Datetime_temp'],inplace=True)
    datename = DATANAME.replace('{date1}', fromdate).replace('{date2}', todate) + '.csv'

    filename = DATAPATH + '/' + datename
    towrite.to_csv(filename, index=False)
    logger.log('info', f'Results saved to csv..', info=filename)

def odd_addition(df):
    logger.log('info', f'Getting odds..')
    next_match1 = pd.read_csv('https://www.football-data.co.uk/fixtures.csv', encoding='utf-8-sig')
    next_match1 = next_match1[['Date','Time','Div','HomeTeam','AwayTeam', 'AvgH', 'AvgD', 'AvgA']]

    next_match2 = pd.read_csv('https://www.football-data.co.uk/new_league_fixtures.csv', encoding='utf-8-sig')
    next_match2 = next_match2[['Date','Time', 'Country', 'Home','Away', 'AvgH', 'AvgD', 'AvgA']]
    next_match2 = next_match2.rename(columns={'Country': 'Div', 'Home': 'HomeTeam', 'Away': 'AwayTeam'})    
    
    next_match = pd.concat([next_match1, next_match2])
    next_match['Date'] = pd.to_datetime(next_match['Date'], format='%d/%m/%Y')

    # Merge predictions with fixtures
    merged = df.merge(next_match[['HomeTeam', 'AwayTeam', 'AvgH', 'AvgA']], on=['HomeTeam', 'AwayTeam'], how='left')
    
    # Map based on prediction type
    def map_avg(row):
        if row['Prediction'] == '1':
            return row['AvgH']
        elif row['Prediction'] == '2':
            return row['AvgA']
        return None

    merged['AVGOdd'] = merged.apply(map_avg, axis=1)

    # Keep only original prediction columns + new mapped value
    df_result = merged[df.columns.tolist() + ['AVGOdd']]
    logger.log('info', f'Odds Captured..')
    return(df_result)

def merging_func():
    last_majorfile = newest_predictions('major')
    last_minorfile = newest_predictions('minor')
    concdata = accumulate_data(last_majorfile, last_minorfile)
    finaldf = odd_addition(concdata)
    saveto_csv(finaldf)

    logger.log('info', f'Deleting interm files..', info=f'{last_majorfile}, {last_minorfile}')

    if last_majorfile != '\\99999999':
        os.remove(last_majorfile)
    
    if last_minorfile != '\\99999999':
        os.remove(last_minorfile)
    
    logger.log('info', f'Process completed..')
    return


if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    datesave = datetime.date.today().strftime('%Y%m%d')
    LOGNAME = LOGNAME.replace('{date}', datesave) + '.json'
    
    if os.path.exists(LOGPATH + '/' +LOGNAME):
        logger = JSONLogger(log_file=LOGNAME, log_dir=LOGPATH)
    else:
        logger = JSONLogger(log_file=LOGNAME, log_dir=LOGPATH)

    try:
        merging_func()

    except Exception as e:
        logger.log('critical', "Exception occured whie running", info=str(e))
