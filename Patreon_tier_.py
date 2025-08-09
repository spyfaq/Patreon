#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import dataframe_image as dfi
from datetime import datetime
import numpy as np
import datetime, os, time
from jsonlogger_class import JSONLogger

LOGPATH = 'logs/data/'
LOGNAME = '{date}_tipsselection_logs'
DATANAME = 'my_prediction_data_{date1}_{date2}'
DATAPATH = 'predictions_data/'
PUBLISHPATH = 'publish/'


today_str = datetime.today().strftime("%d-%m-%Y")


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

    home_form = f"{row['HomeTeam']} has won {row['HT_athome_wins']} of their last {row['HT_athome_matches']} home games."
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
        "GG": "Both Teams to Score"
    }
    df_full["Prediction"] = df_full["Prediction"].map(prediction_map).fillna(df_full["Prediction"])
    df_full["Match"] = df_full["HomeTeam"] + " vs " + df_full["AwayTeam"]

    df = df_full[["Match", "Prediction", "Prediction %", "History %"]]
    df = df.sort_values(by="Prediction %", ascending=False)


    logger.log('info', f'Creating Tier 1: Full table styled..')
    tier1 = df.copy()
    styled_tier1 = tier1.style.format({"Prediction %": "{:.0%}"}).applymap(get_color, subset=["Prediction %"])
    dfi.export(styled_tier1, f"publish/tier1_supporter_{today_str}.png")

    logger.log('info', f'Creating Tier 2: Top 5 picks + reasoning..')
    tier2_top5 = df_full.sort_values(by="Prediction %", ascending=False).head(5).copy()
    tier2_text = f"### Top 5 High-Confidence Picks ({today_str})\n"
    for _, row in tier2_top5.iterrows():
        reason = generate_reasoning(row)
        tier2_text += f"- **{row['Match']}** → {row['Prediction']} ({row['Prediction %']:.0%})\n"
        tier2_text += f"  _Reasoning: {reason}_\n"

    styled_tier2 = tier2_top5[["Match", "Prediction", "Prediction %", "History %"]] \
        .style.format({"Prediction %": "{:.0%}"}).applymap(get_color, subset=["Prediction %"])
    dfi.export(styled_tier2, f"publish/tier2_premium_{today_str}.png")
    with open(f"publish/tier2_premium_{today_str}.md", "w", encoding="utf-8") as f:
        f.write(tier2_text)

    logger.log('info', f'Creating Tier 3: Full CSV + reasoning..')
    tier3_filename = f"publish/tier3_vip_{today_str}.csv"
    df_full.to_csv(tier3_filename, index=False)

    tier3_text = f"### VIP Predictions ({today_str})\nFull prediction data attached as CSV.\n\n" + tier2_text
    with open(f"publish/tier3_vip_{today_str}.md", "w", encoding="utf-8") as f:
        f.write(tier3_text)

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