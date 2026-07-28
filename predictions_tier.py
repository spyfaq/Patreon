#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import datetime, os, re
from openpyxl import load_workbook
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.utils import get_column_letter
import team_utils
import date_utils
import model_config

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
                'US Major League Soccer': 'USA',
                'UEFA Champions League': 'CL',
                'FIFA World Cup': 'WC',
                'UEFA European Championship': 'EC',
               }

today_str = datetime.datetime.today().strftime("%d-%m-%Y")

def savetoexcel_format(df, excel_file):
    print('Saving into a nice excel..', excel_file)
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
    print('Searching latest prediction file..')
    files = os.listdir(DATAPATH)

    paths = []
    for basename in files:
       if 'my_prediction_data_' in basename:
        paths.append(os.path.join(DATAPATH, basename))

    try:
        file = max(paths, key=os.path.getctime)
        print(f'File found..', file)
        return file
    except:
        print(f'ERROR: File not found..')
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

EMPTY_SUMMARY = {"matches": 0, "wins": 0, "draws": 0, "losses": 0,
                 "gf": 0, "ga": 0, "avg_gf": 0.0, "avg_ga": 0.0}


def parse_summary(summary: str):
    # Defense in depth, mirroring parse_form()'s fix for the same
    # underlying cause: a team with no match history (e.g. newly promoted)
    # is a legitimate, expected state upstream now produces a well-formed
    # placeholder for (see majorleague/minorleague_predictions.py's
    # NO_STATS), but this parser should never crash the whole run even if
    # some other malformed/missing value reaches it -- a KeyError one
    # script later than parse_form's is exactly what happened here in
    # production.
    if summary is None or isinstance(summary, float) or not str(summary).strip():
        return dict(EMPTY_SUMMARY)

    parts = str(summary).split("|")
    segment = parts[1].strip() if len(parts) > 1 else parts[0].strip()

    m = re.search(r"(\d+)M (\d+)W (\d+)D (\d+)L", segment)
    if not m:
        return dict(EMPTY_SUMMARY)
    matches, wins, draws, losses = map(int, m.groups())

    g = re.search(r"(\d+)-(\d+)", segment)
    gf, ga = map(int, g.groups()) if g else (0, 0)

    a = re.search(r"\(([\d.]+)-([\d.]+)\)", segment)
    avg_gf, avg_ga = map(float, a.groups()) if a else (0.0, 0.0)

    return {
            "matches": matches, "wins": wins, "draws": draws, "losses": losses,
            "gf": gf, "ga": ga, "avg_gf": avg_gf, "avg_ga": avg_ga
        }

EMPTY_FORM = {"form": "", "wins": 0, "draws": 0, "losses": 0,
              "overs": 0, "unders": 0, "total": 0}


def parse_form(form_str: str):
    # Missing form (NaN/None/blank) previously returned {}, so every
    # downstream lookup like home_form['wins'] raised KeyError and took
    # down the whole run. A promoted team legitimately has no form history
    # in this league yet -- and now reaches this point rather than being
    # dropped earlier -- so "no form" has to be a supported state, not a
    # crash. Return a fully-populated zero record; callers check
    # total == 0 to phrase it honestly rather than claiming "won 0 of 0".
    if form_str is None or isinstance(form_str, float) or not str(form_str).strip():
        return dict(EMPTY_FORM)
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
        # total == 0 means no form on record (e.g. a newly promoted side);
        # say so plainly instead of the nonsensical "won 0 of their last 0".
        home_1 = (f"{row['HomeTeam']} has won {home_form['wins']} of their last {home_form['total']} home games."
                  if home_form['total'] else f"{row['HomeTeam']} has no recent home form on record.")
        away_2 = (f"{row['AwayTeam']} has won {away_form['wins']} of their last {away_form['total']} home games."
                  if away_form['total'] else f"{row['AwayTeam']} has no recent form on record.")

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

# The only markets that are genuinely mutually exclusive for a single
# match -- at most one of these can be true, so showing more than one as
# a "tip" for the same fixture is contradictory, not just redundant.
# Everything else (Over/Under thresholds, BTTS, home/away-specific overs)
# can coexist for the same match without contradiction (e.g. Over 2.5
# Goals and Both Teams to Score can both be correct at once), so those
# are left as separate lines rather than collapsed.
MUTUALLY_EXCLUSIVE_OUTCOMES = {"Home Win", "Draw", "Away Win"}


def _rank(top_picks):
    """Shared quality ranking used by both dedup_one_per_match() and
    resolve_match_conflicts(). Priority: (1) a >80% pick with no H2H
    history available, (2) picks with H2H history, by Prediction % then
    History %, (3) everything else."""
    df = top_picks.copy()
    df['priority'] = np.where(
        (df['PredValue'] > 80) & (df['HistValue'].isna()), 1,
        np.where(df['HistValue'].notna(), 2, 3)
    )
    df['Hist_for_sort'] = df['HistValue'].fillna(-1)
    return df.sort_values(
        by=['priority', 'PredValue', 'Hist_for_sort'],
        ascending=[True, False, False]
    )


def dedup_one_per_match(top_picks, limit=None):
    """One prediction per match, keeping only the single best-quality
    market for that fixture. Used where a match can only sensibly occupy
    one slot: the Telegram VIP text (kept short/scannable) and the
    free-tier selection (each of the 3 public matches gets its one best
    pick). `limit=None` keeps every qualifying match; pass a number (e.g.
    10) to cap it, as the Telegram VIP text does to stay readable."""
    df_sorted = _rank(top_picks)

    # keep only one row per match (best by the sorting)
    deduped = df_sorted.drop_duplicates(subset='Match', keep='first').drop(
        columns=['priority', 'Hist_for_sort']
    )
    return deduped.head(limit) if limit is not None else deduped


def resolve_match_conflicts(top_picks):
    """Multiple qualifying markets per match are kept (e.g. a match can
    show both 'Over 2.5 Goals' and 'Both Teams to Score' as separate
    lines -- they're not contradictory, both can be true at once).
    The one thing that IS resolved: 'Home Win' / 'Draw' / 'Away Win' are
    mutually exclusive by definition, so if more than one qualified for
    the same match, only the single best-quality one of those three is
    kept -- every other market for that match is left untouched."""
    df_sorted = _rank(top_picks)

    is_outcome = df_sorted['Prediction'].isin(MUTUALLY_EXCLUSIVE_OUTCOMES)
    outcome_rows = df_sorted[is_outcome].drop_duplicates(subset='Match', keep='first')
    other_rows = df_sorted[~is_outcome]

    # Recombine and re-sort so the output still reads best-first, same as
    # dedup_one_per_match, just without collapsing non-outcome markets.
    resolved = pd.concat([outcome_rows, other_rows]).sort_values(
        by=['priority', 'PredValue', 'Hist_for_sort'],
        ascending=[True, False, False]
    )
    return resolved.drop(columns=['priority', 'Hist_for_sort'])

def main():
    filename = newest_predictions()
    
    print(f'Loading file..', filename)
    df_full = pd.read_csv(filename)
    df_full.rename(columns={'History %': 'History H2H'}, inplace=True)
    if "AVGOdd" not in df_full.columns:
        df_full["AVGOdd"] = None
    # Reverse the LEAGUES dict
    code_to_name = {v: k for k, v in LEAGUES.items()}

    # Map the 'div' column
    df_full['Division'] = df_full['Division'].map(code_to_name)

    # Strip decorative suffixes (FC/CF/AFC/etc.) so team names read
    # consistently regardless of which source produced them --
    # football-data.org's formal names ("Real Madrid CF") would otherwise
    # sit right next to football-data.co.uk's short names ("Man United")
    # in the same VIP list/Telegram post.
    df_full['HomeTeam'] = df_full['HomeTeam'].apply(team_utils.display_name)
    df_full['AwayTeam'] = df_full['AwayTeam'].apply(team_utils.display_name)

    # Drop bet-builder combos (e.g. '1+O2_5', 'X+GG') entirely -- a combo
    # code always contains '+', a single market never does. Combos are a
    # separate feature surfaced by best_bets_selector.py directly from the
    # merged predictions CSV; they were never meant to reach the VIP
    # tipster list or Excel, and letting them through here is what made
    # every match show 20+ near-random "picks" instead of one clear tip.
    df_full = df_full[~df_full['Prediction'].astype(str).str.contains('+', regex=False)].copy()

    print(f'Map predictions to friendly names..')
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
    
    # Keep the raw market code (e.g. 'O2_5') before it becomes a display
    # label ('Over 2.5 Goals') -- the selection gate below is per-market and
    # needs the code to look up that market's threshold and base rate.
    df_full["RawPrediction"] = df_full["Prediction"]
    df_full["Prediction"] = df_full["Prediction"].map(prediction_map).fillna(df_full["Prediction"])
    df_full["Prediction %"] = df_full["Prediction %"].apply(lambda x: f"{x*100:.2f}%")
    df_full["AVGOdd"] = df_full["AVGOdd"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
    df_full["Match"] = df_full["HomeTeam"] + " vs " + df_full["AwayTeam"]

    df_full['Time'] = (
    pd.to_datetime(df_full['Time'], errors='coerce')
    .dt.time
    .fillna(datetime.time(0, 0))  # replace NaT with 00:00
)
    # Shift early-morning matches (before 09:00 UTC) to the previous day's
    # batch -- see date_utils.py. Previously computed inline here only;
    # now shared with best_bets_selector.py and post_from_dropbox.py so
    # all three agree on which day a given match belongs to.
    df_full['AdjustedDate'] = date_utils.adjusted_date_series(df_full['Date'], df_full['Time'])

    # Loop through each unique date
    for match_date, df_date in df_full.groupby("AdjustedDate"):
        date_str = pd.to_datetime(match_date).strftime("%Y-%m-%d")
        print(f'Processing date: {date_str}')

        # Convert Prediction % from decimal to real % float (before formatting)
        df_date["PredValue"] = df_date["Prediction %"].apply(lambda s: float(str(s).replace('%', '').strip()))
        # Parse History "x/y" into a fraction
        df_date["HistValue"] = df_date["History H2H"].apply(parse_hist)

        # Blend weights come from model_config (backtest-tunable) rather
        # than being hardcoded here. Defaults are the historical 0.7/0.3,
        # so behaviour is unchanged until a tuning.json exists.
        W_MODEL, W_HIST = model_config.get_blend_weights()
        df_date["ConfScore"] = (W_MODEL * df_date["PredValue"]) + (W_HIST * df_date["HistValue"].fillna(df_date["PredValue"]))
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

        # Selection gate.
        #
        # Previously H2H acted as a HARD gate: a pick needed
        # PredValue >= 64 AND HistValue >= 50, with the only bypass being
        # PredValue >= 80 *and no H2H record at all*. That had two bad
        # consequences:
        #   1. H2H over a handful of meetings is a very noisy signal (often
        #      largely different squads), yet it could veto a strong model
        #      pick outright.
        #   2. The bypass required History H2H == '-', so a pick at 85%
        #      model confidence that merely had a POOR H2H record was
        #      dropped, while the same 85% pick with no record at all was
        #      kept -- having less information counted in a pick's favour.
        #
        # Now H2H informs ranking (via ConfScore above) rather than
        # vetoing, and a sufficiently strong model probability qualifies on
        # its own regardless of whether an H2H record exists. The
        # H2H-supported route is kept at the same 64/50 level, so nothing
        # that previously qualified stops qualifying -- this only widens
        # the gate, it never narrows it.
        # Thresholds are now PER-MARKET (model_config.MARKET_MIN_PROB)
        # rather than one flat 64/80 for everything. A flat number meant
        # something different in every market: Over 1.5 lands ~75% of the
        # time so it cleared 64 on almost every fixture, while Over 2.5
        # (~52% base) needs ~3.3 total expected goals just to REACH 64,
        # against a real league average of ~2.5-2.8 -- so it was
        # structurally excluded rather than actually judged. 1X2 markets
        # keep the historical 0.64/0.80, so this is neutral for them.
        #
        # A pick must clear its market's absolute bar AND beat that
        # market's own base rate by MIN_LIFT_OVER_BASE, so a probability
        # that merely matches what the market does anyway never reads as
        # a confident tip.
        raw_market = df_date["RawPrediction"]
        prob = df_date["PredValue"] / 100.0

        # Two bars per market, mirroring the original gate's structure:
        # the lower one qualifies WITH H2H support, the higher one with
        # none. 1X2 resolves to exactly the historical 64 / 80, so this is
        # neutral there; only the goal markets move.
        min_bar = pd.Series([model_config.get_market_min_prob(m) * 100 for m in raw_market], index=df_date.index)
        strong_bar = pd.Series([model_config.get_strong_prob(m) * 100 for m in raw_market], index=df_date.index)
        lift_ok = pd.Series([model_config.has_min_lift(m, p) for m, p in zip(raw_market, prob)], index=df_date.index)

        HIST_SUPPORT = 50
        qualifies = lift_ok & (
            (df_date["PredValue"] >= strong_bar)
            | ((df_date["PredValue"] >= min_bar) & (df_date["HistValue"] >= HIST_SUPPORT))
        )
        top_picks = df_date[qualifies].sort_values(by=["PredValue", "HistValue"], ascending=False)
        
        # Fallback if no matches meet criteria
        if top_picks.empty:
            top_picks = df_date.sort_values(by="PredValue", ascending=False).head(5)
            print(f"WARNING: No matches met criteria for {date_str}, fallback to top 5 by Prediction %")
        
        df_date = df_date.drop(columns=["AdjustedDate"])

        # Fully deduped (1 pick/match) -- drives the Telegram VIP text and
        # the free-tier selection, where a match can only sensibly occupy
        # one slot.
        vip_all = dedup_one_per_match(top_picks, limit=None)

        # Tier 1: 3 random matches from the VIP-quality list, using each
        # match's single best pick.
        print(f'Creating Public: 3 daily picks..')
        public = vip_all.sample(n=min(3, len(vip_all)), random_state=42)

        # Telegram-friendly public list
        public_tg = f"📊 <b>Free Picks — {date_str}</b>\n\n"
        for _, row in public.iterrows():
            public_tg += f"• <b>{row['Match']}</b> → {row['Prediction']}\n"

        # Add VIP join message
        public_tg += "\n📩 <a href='https://rebrand.ly/betprophet-m'>Join BetProphet.AI VIP now</a> for today’s premium picks before kick-off!\n💎Just €8/month — one winning bet covers your subscription! ✅"

        # Save Telegram text
        with open(f"{PUBLISHPATH}/Public_{date_str}.txt", "w", encoding="utf-8") as f:
            f.write(public_tg)

        print(f'Creating VIP: Top 10 picks + reasoning + csv..')
        vip = vip_all.head(10)

        # Telegram-friendly VIP list
        vip_tg = f"💎 <b>VIP Picks — {date_str}</b>\n\n"

        for _, row in vip.iterrows():
            vip_tg += f"{row['Confidence']} <b>{row['Match']}</b> → {row['Prediction']} ({row['Prediction %']} | {row['History H2H']})\n"

        # Add reasoning for top 5
        vip_tg += "\n<b>Reasoning for Top 5:</b>\n"
        for _, row in vip.head(5).sort_values(by='Match').iterrows():
            vip_tg += f"• <b>{row['Match']}</b> → {row['Prediction']} ({row['Prediction %']})\n"
            vip_tg += f"  <i>{row['Reasoning']}</i>\n"

        # Excel: every qualifying market per match, EXCEPT Home Win /
        # Draw / Away Win, where only the single best of the three is
        # kept (they're mutually exclusive -- at most one can be true).
        # Everything else (Over/Under thresholds, BTTS, home/away-specific
        # overs) is left as separate lines, since those aren't
        # contradictory and a tipster reasonably shows several angles on
        # the same match. Combos were already dropped on load.
        excel_picks = resolve_match_conflicts(top_picks)

        # Matched on (Match, Prediction), not just Match, since a match
        # can now have more than one Excel row -- only the specific pick
        # that was actually posted publicly should be flagged True.
        public_pairs = set(zip(public["Match"], public["Prediction"]))
        excel_picks = excel_picks.copy()
        excel_picks["PickedforFree"] = excel_picks.apply(
            lambda row: (row["Match"], row["Prediction"]) in public_pairs, axis=1
        )

        excel_picks = excel_picks.rename(columns={"AVGOdd": "Odds"})
        df_save = excel_picks[["Division", "Date", "Time", "HomeTeam", "AwayTeam", "Prediction", "Odds", "Prediction %", "History H2H", "HomeForm", "AwayForm", "Reasoning", "HomeTeam Stats", "AwayTeam Stats", "PickedforFree"]]
        # Telegram text, and CSV
        with open(f"{PUBLISHPATH}/VIP_{date_str}.txt", "w", encoding="utf-8") as f:
            f.write(vip_tg)
        filename = f"{PUBLISHPATH}/VIP_{date_str}.xlsx"
        savetoexcel_format(df_save, filename)
        #df_date.to_csv(csv_filename, index=False)

    print(f'Telegram content generated..')
    return

if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))

    main()
