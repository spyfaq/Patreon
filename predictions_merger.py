#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, datetime, re
import pandas as pd
import team_utils
import odds_client
import odds_utils



DATAPATH = 'predictions_data/'
MAJORDATANAME = 'my_prediction_major_data_{date1}_{date2}'
MINORDATANAME = 'my_prediction_minor_data_{date1}_{date2}'
DATANAME = 'my_prediction_data_{date1}_{date2}'


def newest_predictions(sever) -> str:
    print(f'Searching latest prediction file for {sever} leagues..')
    files = os.listdir(DATAPATH)

    paths = []
    for basename in files:
       if f'my_prediction_{sever}_data_' in basename:
        paths.append(os.path.join(DATAPATH, basename))

    try:
        file = max(paths, key=os.path.getctime)
        print(f'File found..', file)
        return file
    except:
        print(f'WARNING: File for {sever} not found..')
        return('\\99999999')
    
def accumulate_data(files: dict) -> pd.DataFrame:
    """Merge prediction files from multiple sources (major/minor/international).
    `files` maps a source name to its file path, or the '\\99999999' sentinel
    if that source produced nothing today (e.g. no international fixtures
    outside a tournament window -- this is expected, not an error).
    """
    print(f'Trying to match prediction files from: {list(files.keys())}..')

    def str_to_date(date_str):
        return datetime.datetime.strptime(date_str, '%d%m%Y')

    available = {name: path for name, path in files.items() if '99999999' not in path}

    if not available:
        print('WARNING: No prediction files found from any source today..')
        return pd.DataFrame()

    if len(available) == 1:
        name, path = next(iter(available.items()))
        print(f'WARNING: Only {name} predictions found today')
        return pd.read_csv(path)

    date_ranges = {}
    for name, path in available.items():
        dates = re.findall(r'\d{8}', path)
        date_ranges[name] = (str_to_date(dates[0]), str_to_date(dates[1]))

    starts = [r[0] for r in date_ranges.values()]
    ends = [r[1] for r in date_ranges.values()]
    max_spread = max(max(starts) - min(starts), max(ends) - min(ends))

    if max_spread <= datetime.timedelta(days=1):
        dfs = [pd.read_csv(path) for path in available.values()]
        print(f'Matched {list(available.keys())} prediction files..')
        return pd.concat(dfs, ignore_index=True)
    else:
        # Sources disagree on date range by more than a day -- keep only
        # whichever has the most recent data rather than mixing stale and
        # fresh predictions together.
        latest_name = max(date_ranges, key=lambda n: date_ranges[n][1])
        print(f'WARNING: Prediction files did not align in date.. keeping only {latest_name}')
        return pd.read_csv(available[latest_name])
        
def saveto_csv(towrite):
    print(f'Saving results..')

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
    print(f'Results saved to csv..', filename)

def odd_addition(df):
    print(f'Getting odds..')

    # Over/Under 2.5 columns aren't published for every league/source, so
    # fetch them defensively and fall back to NaN rather than failing the
    # whole merge if a source is missing them.
    ou_candidates = ['Avg>2.5', 'Avg<2.5']
    base_cols = ['Date', 'Time', 'Div', 'HomeTeam', 'AwayTeam', 'AvgH', 'AvgD', 'AvgA']

    next_match1 = pd.read_csv('https://www.football-data.co.uk/fixtures.csv', encoding='utf-8-sig')
    have_ou_1 = team_utils.find_columns(next_match1.columns, ou_candidates)
    print(f'Main fixtures O/U columns found: {have_ou_1 or "NONE"}', list(next_match1.columns))
    next_match1 = next_match1[base_cols + have_ou_1]

    next_match2 = pd.read_csv('https://www.football-data.co.uk/new_league_fixtures.csv', encoding='utf-8-sig')
    have_ou_2 = team_utils.find_columns(next_match2.columns, ou_candidates)
    print(f'New-league fixtures O/U columns found: {have_ou_2 or "NONE"}', list(next_match2.columns))
    next_match2 = next_match2[['Date','Time', 'Country', 'Home','Away', 'AvgH', 'AvgD', 'AvgA'] + have_ou_2]
    next_match2 = next_match2.rename(columns={'Country': 'Div', 'Home': 'HomeTeam', 'Away': 'AwayTeam'})

    next_match = pd.concat([next_match1, next_match2])
    next_match['Date'] = pd.to_datetime(next_match['Date'], format='%d/%m/%Y')
    next_match = next_match.rename(columns={c: 'AvgOver25' for c in have_ou_1 + have_ou_2 if c.strip().lower() == 'avg>2.5'})
    next_match = next_match.rename(columns={c: 'AvgUnder25' for c in have_ou_1 + have_ou_2 if c.strip().lower() == 'avg<2.5'})

    for c in ['AvgD', 'AvgOver25', 'AvgUnder25']:
        if c not in next_match.columns:
            next_match[c] = None

    print(f'Domestic odds: {len(next_match)} fixtures, '
                        f'{next_match["AvgOver25"].notna().sum()} with an Over 2.5 price.')

    # football-data.co.uk (next_match above) only carries domestic-league
    # odds -- nothing for Champions League / World Cup / Euros. Pull those
    # 3 from The Odds API instead (odds_client.py), already shaped to the
    # same Avg*/Date/Time/Div/HomeTeam/AwayTeam columns. A fetch failure
    # here (missing ODDS_API_KEY, API down, outside a tournament window)
    # shouldn't block odds for the domestic leagues that already succeeded
    # above -- international rows just end up with no AVGOdd, same as any
    # unmatched domestic row would.
    odds_cols = ['Date', 'Time', 'Div', 'HomeTeam', 'AwayTeam', 'AvgH', 'AvgD', 'AvgA', 'AvgOver25', 'AvgUnder25']
    try:
        intl_odds = odds_client.fetch_all_international_odds()
    except Exception as e:
        print('WARNING: Could not fetch international odds..', e)
        intl_odds = pd.DataFrame(columns=odds_cols)

    next_match = pd.concat([next_match[odds_cols], intl_odds[odds_cols]], ignore_index=True)

    # football-data.co.uk (and The Odds API's totals market, fetched only
    # at the 2.5 line) never publishes Over 1.5 / Over 3.5 odds at all --
    # approximate them from the real Over 2.5 price (see odds_utils.py).
    next_match['AvgOver15'], next_match['AvgOver35'] = odds_utils.derive_over_under_odds(next_match['AvgOver25'])

    # Merge predictions with fixture odds, tolerating team-naming
    # differences between whichever source produced the prediction
    # (football-data.co.uk for major/minor, football-data.org for
    # international) and whichever source has the odds (football-data.co.uk
    # fixtures for domestic, The Odds API for international) -- an exact
    # string merge previously dropped every international row outright and
    # would silently miss any domestic team name that drifted even
    # slightly between the two feeds.
    odds_lookup = next_match[['HomeTeam', 'AwayTeam', 'AvgH', 'AvgD', 'AvgA',
                               'AvgOver25', 'AvgUnder25', 'AvgOver15', 'AvgOver35']]
    merged = team_utils.fuzzy_merge(df, odds_lookup, left_on=('HomeTeam', 'AwayTeam'),
                                     right_on=('HomeTeam', 'AwayTeam'))

    # Map based on prediction type
    def map_avg(row):
        if row['Prediction'] == '1':
            return row['AvgH']
        elif row['Prediction'] == 'X':
            return row['AvgD']
        elif row['Prediction'] == '2':
            return row['AvgA']
        elif row['Prediction'] == 'O2_5':
            return row['AvgOver25']
        elif row['Prediction'] == 'O1_5':
            return row['AvgOver15']
        elif row['Prediction'] == 'O3_5':
            return row['AvgOver35']
        return None

    merged['AVGOdd'] = merged.apply(map_avg, axis=1)

    # Keep only original prediction columns + new mapped value
    df_result = merged[df.columns.tolist() + ['AVGOdd']]
    print(f'Odds Captured..')
    return(df_result)

def merging_func():
    source_files = {
        'major': newest_predictions('major'),
        'minor': newest_predictions('minor'),
        'international': newest_predictions('international'),
    }
    concdata = accumulate_data(source_files)

    if concdata.empty:
        print('WARNING: Nothing to merge today (no source produced predictions).')
        return

    # Clean up team display names once, right here, so every downstream
    # consumer (odds merge below, predictions_tier.py's posted output,
    # update_results.py's settlement match) works from the same canonical
    # names from this point on -- e.g. international's football-data.org
    # "Real Madrid CF" becomes "Real Madrid", matching the short-form
    # names major/minor already use from football-data.co.uk.
    concdata['HomeTeam'] = concdata['HomeTeam'].apply(team_utils.display_name)
    concdata['AwayTeam'] = concdata['AwayTeam'].apply(team_utils.display_name)

    finaldf = odd_addition(concdata)
    saveto_csv(finaldf)

    print(f'Deleting interm files..', source_files)

    for path in source_files.values():
        if path != '\\99999999':
            os.remove(path)

    print(f'Process completed..')
    return


if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    

    try:
        merging_func()

    except Exception as e:
        print("CRITICAL: Exception occured whie running", e)
