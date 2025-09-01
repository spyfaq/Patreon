#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import datetime, os, re
from jsonlogger_class import JSONLogger
from openpyxl import load_workbook
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.utils import get_column_letter

LOGPATH = 'logs/data/'
LOGNAME = '{date}_tipsselection_logs'
DATANAME = 'my_prediction_data_{date1}_{date2}'
DATAPATH = 'predictions_data/'
PUBLISHPATH = 'publish/'

LEAGUES = {'EN PremierLeague': 'E0',
               'EN Championship': 'E1',
               'DE Bundesliga': 'D1',
               'IT Serie A': 'I1',
               'SP LaLiga': 'SP1',
               'FR Championnat': 'F1',
               'NH Eredivisie': 'N1',
               'BG JupilerLeague': 'B1',
               'PR Liga I': 'P1',
               'GR SuperLeague': 'G1',
               'DE Bundesliga 2': 'D2',
               'IT Serie B': 'I2',
               'SP Segunda': 'SP2',
               'FR Division 2': 'F2',
               'EN League 1': 'E2',
               'SC PremierLeague': 'SC0',
               'TR Futbol Ligi 1': 'T1',
                'AT Bundesliga': 'Austria',
                'AR Liga Profesional': 'Argentina',
                'BR Serie A': 'Brazil',
                'DK Superliga': 'Denmark',
                'FI Veikkausliiga': 'Finland',
                'IE Premier Division': 'Ireland',
                'MX Liga MX': 'Mexico',
                'NO Eliteserien': 'Norway',
                'PL Ekstraklasa': 'Poland',
                'RO Liga I': 'Romania',
                'SE Allsvenskan': 'Sweden',
                'CH Super League': 'Switzerland',
                'US Major League Soccer': 'USA'
               }

today_str = datetime.datetime.today().strftime("%d-%m-%Y")

def savetoexcel_format(df, excel_file):
    logger.log('info', 'Saving into a nice excel..', info=excel_file)
    df.to_excel(excel_file, index=False, sheet_name="Sheet1")

    # Load workbook with openpyxl
    wb = load_workbook(excel_file)
    ws = wb["Sheet1"]
    
    # Freeze top row
    ws.freeze_panes = "A2"

    # Define table range (A1 through last row/col)
    last_row = ws.max_row
    last_col = ws.max_column
    last_col_letter = get_column_letter(last_col)
    table_range = f"A1:{last_col_letter}{last_row}"

    # Create Excel Table
    table = Table(displayName="PredTable", ref=table_range)

    # Optional: style
    style = TableStyleInfo(
        name="TableStyleLight1",  
        showFirstColumn=False,
        showLastColumn=False,
        showRowStripes=True,  # stripe rows
        showColumnStripes=False
    )
    table.tableStyleInfo = style

    # Add table to sheet
    ws.add_table(table)

    # Auto-adjust column widths
    for col in ws.columns:
        max_length = 0
        col_letter = get_column_letter(col[0].column)
        for cell in col:
            try:
                cell_length = len(str(cell.value))
                if cell_length > max_length:
                    max_length = cell_length
            except:
                pass
        # Add a little extra space
        ws.column_dimensions[col_letter].width = max_length + 2

    # Save workbook
    wb.save(excel_file)

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

def parse_summary(summary: str):
    parts = summary.split("|")
    segment = parts[1].strip() if len(parts) > 1 else parts[0].strip()

    m = re.search(r"(\d+)M (\d+)W (\d+)D (\d+)L", segment)
    matches, wins, draws, losses = map(int, m.groups())

    g = re.search(r"(\d+)-(\d+)", segment)
    gf, ga = map(int, g.groups())

    a = re.search(r"\(([\d.]+)-([\d.]+)\)", segment)
    avg_gf, avg_ga = map(float, a.groups())

    return {
            "matches": matches, "wins": wins, "draws": draws, "losses": losses,
            "gf": gf, "ga": ga, "avg_gf": avg_gf, "avg_ga": avg_ga
        }

def parse_form(form_str: str):
    if type(form_str) == float:
        return {}
    form = form_str.replace("-", "")  # remove unused slots
    total = len(form)
    wins = form.count("W")
    draws = form.count("D")
    losses = form.count("L")
    overs = form.count("O")
    unders = form.count("U")
    return {
        "form": form,
        "wins": wins, "draws": draws, "losses": losses,
        "overs": overs, "unders": unders,
        "total": total
    }


def generate_reasoning(row):
    pred = row["Prediction"]

    home_stats = parse_summary(row["HomeTeam Stats"])
    away_stats = parse_summary(row["AwayTeam Stats"])

    home_form = parse_form(row["HomeForm"])
    away_form = parse_form(row["AwayForm"])

    # Core stats (fixed away goals bug)
    if pred != "Both Teams to Score":
        home_1 = f"{row['HomeTeam']} has won {home_form['wins']} of their last {home_form['total']} home games."
        away_2 = f"{row['AwayTeam']} has won {away_form['wins']} of their last {away_form['total']} home games."

    home_goals = f"{row['HomeTeam']} averages {(home_stats['avg_gf']):.1f} goals at home."
    away_goals = f"{row['AwayTeam']} averages {(away_stats['avg_gf']):.1f} goals at away."
    home_concede = f"{row['HomeTeam']} concedes {(home_stats['avg_ga']):.1f} goals at home."
    away_concede = f"{row['AwayTeam']} concedes {(away_stats['avg_ga']):.1f} goals at away."

    # Reasoning per prediction code
    if pred == "Home Win":
        reasoning = f"{home_1} {away_2} {away_concede} Strong home record supports this."
    elif pred == "Away Win":
        reasoning = f"{away_2} {home_1} {home_concede} Away form makes them favourites."
    elif pred == "Both Teams to Score":
        reasoning = f"{home_goals} {away_goals} Both teams tend to concede ({home_concede}, {away_concede}), making goals at both ends likely."
    elif pred == "Draw":
        reasoning = f"Balanced recent form: {home_1} {away_2} A draw is a strong possibility."
    elif "Home team Over" in pred:
        reasoning = f"{home_goals} {away_concede} The home side’s attack should produce the required goals."
    elif "Away team Over" in pred:
        reasoning = f"{away_goals} {home_concede} The away side’s attack looks likely to score well."
    elif 'Over' in pred:
        reasoning = f"{home_goals} {away_goals} {home_concede} {away_concede} Both sides look capable of goals."
    else:
        reasoning = "Prediction based on statistical analysis."

    return reasoning

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
    df_full.rename(columns={'History %': 'History H2H'}, inplace=True)
    # Reverse the LEAGUES dict
    code_to_name = {v: k for k, v in LEAGUES.items()}

    # Map the 'div' column
    df_full['Division'] = df_full['Division'].map(code_to_name)

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

    df_full['Time'] = (
    pd.to_datetime(df_full['Time'], errors='coerce')
    .dt.time
    .fillna(datetime.time(0, 0))  # replace NaT with 00:00
)
    # Assume df_full['Date'] is already a datetime.date and df_full['Time'] is datetime.time
    df_full['Datetime_temp'] = pd.to_datetime(df_full['Date'].astype(str) + ' ' + df_full['Time'].astype(str), dayfirst=True)

    # Shift early-morning matches (before 06:00) to previous day
    df_full['AdjustedDate'] = df_full['Datetime_temp'].apply(
        lambda dt: (dt - timedelta(days=1)).date() if dt.hour < 8 else dt.date()
    )

    # Loop through each unique date
    for match_date, df_date in df_full.groupby("AdjustedDate"):
        date_str = pd.to_datetime(match_date).strftime("%Y-%m-%d")
        logger.log('info', f'Processing date: {date_str}')

        # Convert Prediction % from decimal to real % float (before formatting)
        df_date["PredValue"] = df_date["Prediction %"].apply(lambda s: float(str(s).replace('%', '').strip()))
        # Parse History "x/y" into a fraction
        df_date["HistValue"] = df_date["History H2H"].apply(parse_hist)

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
        
        df_date = df_date.drop(columns=["PredValue", "HistValue", "ConfScore", "Datetime_temp", "AdjustedDate"])

        # Tier 1: Top 3 picks (Division, Match, Prediction)
        logger.log('info', f'Creating Public: 3 daily picks..')
        n_samples = min(3, len(top_picks))
        public = top_picks[["Division", "Match", "Prediction"]].head(5).sample(n=n_samples, random_state=1)
        # Telegram-friendly public list
        public_tg = f"📊 <b>Free Picks — {date_str}</b>\n\n"
        for _, row in public.iterrows():
            public_tg += f"• <b>{row['Match']}</b> → {row['Prediction']}\n"

        # Add VIP join message
        public_tg += "\n📩 <a href='https://rebrand.ly/betprophet-m'>Join BetProphet.AI VIP now</a> for today’s premium picks before kick-off!\n💎Just €8/month — one winning bet covers your subscription! ✅"

        # Save Telegram text
        with open(f"{PUBLISHPATH}/Public_{date_str}.txt", "w", encoding="utf-8") as f:
            f.write(public_tg)

        logger.log('info', f'Creating VIP: Top 10 picks + reasoning + csv..')
        vip = top_picks[["Division", "Match", "Prediction", "Confidence", "Prediction %", "History H2H"]]
        vip = vip.head(10)
        # Telegram-friendly VIP list
        vip_tg = f"💎 <b>VIP Picks — {date_str}</b>\n\n"
        vip.sort_values(by='Match', inplace=True)
        for _, row in vip.iterrows():
            vip_tg += f"{row['Confidence']} <b>{row['Match']}</b> → {row['Prediction']} ({row['Prediction %']} | {row['History H2H']})\n"

        # Add reasoning for top 5
        vip_tg += "\n<b>Reasoning for Top 5:</b>\n"
        top_picks.head(5).sort_values(by='Match', inplace=True)
        for _, row in top_picks.head(5).iterrows():
            vip_tg += f"• <b>{row['Match']}</b> → {row['Prediction']} ({row['Prediction %']})\n"
            vip_tg += f"  <i>{row['Reasoning']}</i>\n"

        df_date["PickedforFree"] = df_date.apply(
            lambda row: (row["Division"], row["Match"], row["Prediction"]) 
                        in public[["Division", "Match", "Prediction"]].itertuples(index=False, name=None),
            axis=1
        )
        df_save = df_date[["Division", "Date", "Time", "HomeTeam", "AwayTeam", "Prediction", "Prediction %", "History H2H", "HomeForm", "AwayForm", "Reasoning", "HomeTeam Stats", "AwayTeam Stats", "PickedforFree"]]
        # Telegram text, and CSV
        with open(f"{PUBLISHPATH}/VIP_{date_str}.txt", "w", encoding="utf-8") as f:
            f.write(vip_tg)
        filename = f"{PUBLISHPATH}/VIP_{date_str}.xlsx"
        savetoexcel_format(df_save, filename)
        #df_date.to_csv(csv_filename, index=False)

    logger.log('info', f'Telegram content generated..')
    return

if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    datesave = datetime.date.today().strftime('%Y%m%d')
    LOGNAME = LOGNAME.replace('{date}', datesave) + '.json'
    logger = JSONLogger(log_file=LOGNAME, log_dir=LOGPATH)

    main()
