#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from datetime import datetime
import datetime, os
from jsonlogger_class import JSONLogger

LOGPATH = 'logs/data/'
LOGNAME = '{date}_tipsselection_logs'
DATANAME = 'my_prediction_data_{date1}_{date2}'
DATAPATH = 'predictions_data/'
PUBLISHPATH = 'publish/'


today_str = datetime.datetime.today().strftime("%d-%m-%Y")


def newest_predictions() -> str:
    logger.log('info', 'Searching latest prediction file..')
    files = os.listdir(DATAPATH)

    paths = []
    for basename in files:
       if 'my_prediction_data_' in basename:
        paths.append(os.path.join(DATAPATH, basename))

    try:
        file = max(paths, key=os.path.getctime)
        logger.log('info', f'File found..', file)
        return file
    except:
        logger.log('error', f'File not found..')
        return('\\99999999')


def get_color(val):
    if isinstance(val, float):
        if val >= 0.8:
            return "background-color: #c6efce"
        elif val >= 0.6:
            return "background-color: #ffeb9c"
        else:
            return "background-color: #ffc7ce"
    return ""


def generate_reasoning(row):
    pred = row["Prediction"]

    home_form = f"{row['HomeTeam']} has won {row['HT_athome_wins']} of their last {row['HT_athome_wins']+ row['HT_athome_draws'] + row['HT_athome_loses']} home games."
    away_form = f"{row['AwayTeam']} has won {row['AT_away_wins']} of {row['AT_Matches']} away games."
    home_goals = f"{row['HomeTeam']} averages {row['HT_athome_goal_scored']:.1f} goals at home."
    away_goals = f"{row['AwayTeam']} averages {row['AT_away_goal_scored']:.1f} goals away."
    home_concede = f"{row['HomeTeam']} concedes {row['HT_athome_goal_against']:.1f} goals at home."
    away_concede = f"{row['AwayTeam']} concedes {row['AT_away_goal_against']:.1f} goals away."

    if pred == "Home Win":
        return f"{home_form} {away_form} {away_concede}"
    elif pred == "Away Win":
        return f"{away_form} {home_form} {home_concede}"
    elif pred in ["Over 1.5 Goals", "Over 2.5 Goals", "Over 3.5 Goals"]:
        return f"{home_goals} {away_goals} {home_concede} {away_concede}"
    elif pred == "Both Teams to Score":
        return f"{home_goals} {away_goals} Both teams have tendencies to concede regularly."
    elif pred == "Draw":
        return f"Both teams show balanced results: {home_form} {away_form}."
    else:
        return "Prediction based on statistical analysis."

def parse_hist(s):
    try:
        if pd.isna(s):
            return np.nan
        s = str(s).strip()
        if s in ('', '-'):
            return np.nan
        if '/' in s:
            num, den = s.split('/')
            num, den = num.strip(), den.strip()
            if num == '' or den == '':
                return np.nan
            den_i = int(den)
            if den_i == 0:
                return np.nan
            return (int(num) / den_i) * 100
        if s.endswith('%'):
            return float(s.rstrip('%').strip())
        v = float(s)
        return v * 100 if v <= 1 else v
    except Exception:
        return np.nan
    
def main():
    filename = newest_predictions()
    
    logger.log('info', f'Loading file..', filename)
    df_full = pd.read_csv(filename)

    logger.log('info', f'Map predictions to friendly names..')
    prediction_map = {
        "O1_5": "Over 1.5 Goals",
        "O2_5": "Over 2.5 Goals",
        "O3_5": "Over 3.5 Goals",
        "1": "Home Win",
        "2": "Away Win",
        "X": "Draw",
        "GG": "Both Teams to Score",
        "aO1_5": "Away team Over 1.5 Goals",
        "aO2_5": "Away team Over 2.5 Goals",
        "hO1_5": "Home team Over 1.5 Goals",
        "hO2_5": "Home team Over 2.5 Goals"
    }
    
    df_full["Prediction"] = df_full["Prediction"].map(prediction_map).fillna(df_full["Prediction"])
    df_full["Prediction %"] = df_full["Prediction %"].apply(lambda x: f"{x*100:.2f}%")
    df_full["Match"] = df_full["HomeTeam"] + " vs " + df_full["AwayTeam"]

    # Loop through each unique date
    for match_date, df_date in df_full.groupby("Date"):
        date_str = pd.to_datetime(match_date).strftime("%Y-%m-%d")
        logger.log('info', f'Processing date: {date_str}')

        # Convert Prediction % from decimal to real % float (before formatting)
        df_date["PredValue"] = df_date["Prediction %"].apply(lambda s: float(str(s).replace('%', '').strip()))
        # Parse History "x/y" into a fraction
        df_date["HistValue"] = df_date["History %"].apply(parse_hist)

        df_date["ConfScore"] = (0.7 * df_date["PredValue"]) + (0.3 * df_date["HistValue"].fillna(df_date["PredValue"]))
        def conf_emoji(score):
            if score >= 80:
                return "🟢"
            elif score >= 60:
                return "🟠"
            else:
                return "🔴"
        
        df_date["Confidence"] = df_date["ConfScore"].apply(conf_emoji)

        # Add reasoning column (for Tier 3)
        df_date["Reasoning"] = df_date.apply(generate_reasoning, axis=1)

        # Filter: Prediction ≥ 80% and History ≥ 70%
        top_picks = df_date[
            (df_date["PredValue"] >= 64) &
            (df_date["HistValue"] >= 64)
        ].sort_values(by=["PredValue", "HistValue"], ascending=False)
        
        # Fallback if no matches meet criteria
        if top_picks.empty:
            top_picks = df_date.sort_values(by="PredValue", ascending=False).head(5)
            logger.log('warning', f"No matches met criteria for {date_str}, fallback to top 5 by Prediction %")
        
        df_date = df_date.drop(columns=["PredValue", "HistValue", "ConfScore"])

        # Tier 1: Top 5 picks (Division, Match, Prediction)
        logger.log('info', f'Creating Tier 1: Full table styled..')
        tier1 = top_picks[["Division", "Match", "Prediction"]].head(5)
        tier1_md = f"### Supporter Tier Predictions — {date_str}\n\n" + tier1.to_markdown(index=False)
        with open(f"{PUBLISHPATH}/tier1_supporter_{date_str}.md", "w", encoding="utf-8") as f:
            f.write(tier1_md)

        logger.log('info', f'Creating Tier 2: Top 10 picks + reasoning..')
        tier2 = top_picks[["Division", "Match", "Prediction", "Confidence", "Prediction %", "History %"]]
        tier2_md_text = f"### Premium Tier Predictions — {date_str}\n\n" + tier2.to_markdown(index=False)

        top5 = top_picks.head(5)
        tier2_md_text += "\n\n**Reasoning for Top 5 Predictions:**\n"
        for _, row in top5.iterrows():
            tier2_md_text += f"- **{row['Match']}** → {row['Prediction']} ({row['Prediction %']})\n"
            tier2_md_text += f"  _Reasoning: {row['Reasoning']}_\n"

        with open(f"{PUBLISHPATH}/tier2_premium_{date_str}.md", "w", encoding="utf-8") as f:
            f.write(tier2_md_text)

        logger.log('info', f'Creating Tier 3: Full CSV + reasoning..')
        tier3 = df_date[["Division", "Match", "Prediction", "Confidence", "Prediction %", "History %", "Reasoning"]]
        tier3_md_text = f"### VIP Tier Predictions — {date_str}\n\n" + tier3.to_markdown(index=False)

        csv_filename = f"{PUBLISHPATH}/tier3_vip_{date_str}.csv"
        df_date.to_csv(csv_filename, index=False)

        with open(f"{PUBLISHPATH}/tier3_vip_{date_str}.md", "w", encoding="utf-8") as f:
            f.write(tier3_md_text)

    logger.log('info', f'Patreon content generated..')
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