#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
publish_predictions.py

Single publishing step, replacing predictions_tier.py + best_bets_selector.py.
Reads today's merged-prediction-<date>.csv and writes everything that gets
posted:

  publish/Public_<date>.txt    free tier -- 3 picks priced over 2.00
  publish/VIP_<date>.txt       top 10 picks, each with its own reasoning
  publish/VIP_<date>.xlsx      every qualifying prediction
  publish/BestBets_<date>.txt  markdown, 3-6 matches chosen on edge + form
                               + H2H + standings

Why one script: the two it replaces loaded the same file, re-derived the
same team stats, and re-applied the same day-bucketing independently.
best_bets_selector additionally re-downloaded the football-data.co.uk
fixtures feed and called The Odds API to compute an edge that the merged
file now already carries -- so the whole odds-fetching half of it is gone.

Combos (e.g. '1+O2_5') are included everywhere. Their probability is the
independence product of the legs (see predictions_merger.build_combos),
so their edge is indicative rather than exact; the Best Bets output says
so where one is picked.
"""

import os
import re
import datetime
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.utils import get_column_letter

import team_utils
import date_utils
import model_config

DATAPATH = 'predictions_data/'
PUBLISHPATH = 'publish/'
INPUT_TEMPLATE = 'merged-prediction-{date}.csv'

# ------------------------------------------------------------- free tier
FREE_PICKS = 3

# The free picks are drawn at random from the FREE_POOL highest-EDGE
# qualifying picks.
#
# Why edge defines the pool rather than just ordering it: sampling
# uniformly from a pool ignores that pool's order, so ranking a fixed set
# of 10 by edge and then drawing 3 at random would give the same 3 as
# ranking them any other way. For edge to actually influence the outcome
# it has to decide WHICH picks are in the pool.
#
# Why edge rather than probability (how the VIP list is ranked): the VIP
# ranking is by model probability, and high probability means short odds
# almost by definition. Drawing the free list from it produced a shop
# window full of 1.2-1.4 near-certainties -- unimpressive to read, and
# barely profitable to back. Ranking the pool by edge spreads the odds
# naturally without needing a hard price floor, and showcases the thing
# the product actually sells: finding value the market missed.
#
# If you would rather edge picked the three outright, replace the sample()
# in select_free() with .head(FREE_PICKS) -- that drops the randomness.
FREE_POOL = 10

# Fixed so a re-run on the same day republishes the same free list.
FREE_SEED = 42

# ------------------------------------------------------------- VIP tier
VIP_PICKS = 10

# ------------------------------------------------------------ best bets
BEST_MIN_PICKS = 3
BEST_MAX_PICKS = 6

# Eligibility floors, applied before ranking. A pick has to be genuinely
# priced above the model's disagreement threshold to be considered at all;
# form and standings then decide the ORDER, they never rescue a pick with
# no edge.
MIN_EDGE = 0.03
MIN_MODEL_PROB = 0.45

# Composite weights. Edge dominates deliberately -- it is the only
# component measured against a real market price, so it carries the most
# information about whether a bet is worth making. Form, standings and
# H2H describe whether the model's number is believable, which is a
# ranking question, not an admission one.
W_EDGE, W_FORM, W_STATS, W_H2H = 0.55, 0.20, 0.15, 0.10

# Edge that scores full marks on the edge component. +15pp over the
# market is already exceptional; anything beyond it saturates rather than
# letting one freak price dominate the ranking.
EDGE_SCALE = 0.15

PREDICTION_LABELS = {
    "O1_5": "Over 1.5 Goals", "O2_5": "Over 2.5 Goals", "O3_5": "Over 3.5 Goals",
    "1": "Home Win", "2": "Away Win", "X": "Draw", "GG": "Both Teams to Score",
    "aO1_5": "Away team Over 1.5 Goals", "aO2_5": "Away team Over 2.5 Goals",
    "hO1_5": "Home team Over 1.5 Goals", "hO2_5": "Home team Over 2.5 Goals",
}

RESULT_MARKETS = {'1', 'X', '2'}
GOAL_LINES = {'O1_5': 1.5, 'O2_5': 2.5, 'O3_5': 3.5}

LEAGUES = {'EN PremierLeague': 'E0', 'EN Championship': 'E1', 'DE Bundesliga': 'D1',
           'IT Serie A': 'I1', 'SP LaLiga': 'SP1', 'FR Championnat': 'F1',
           'NH Eredivisie': 'N1', 'BG JupilerLeague': 'B1', 'PR Liga I': 'P1',
           'GR SuperLeague': 'G1', 'DE Bundesliga 2': 'D2', 'IT Serie B': 'I2',
           'SP Segunda': 'SP2', 'FR Division 2': 'F2', 'EN League 1': 'E2',
           'SC PremierLeague': 'SC0', 'TR Futbol Ligi 1': 'T1',
           'AT Bundesliga': 'Austria', 'AR Liga Profesional': 'Argentina',
           'BR Serie A': 'Brazil', 'DK Superliga': 'Denmark',
           'FI Veikkausliiga': 'Finland', 'IE Premier Division': 'Ireland',
           'MX Liga MX': 'Mexico', 'NO Eliteserien': 'Norway',
           'PL Ekstraklasa': 'Poland', 'RO Liga I': 'Romania',
           'SE Allsvenskan': 'Sweden', 'CH Super League': 'Switzerland',
           'US Major League Soccer': 'USA', 'UEFA Champions League': 'CL',
           'FIFA World Cup': 'WC', 'UEFA European Championship': 'EC'}

MUTUALLY_EXCLUSIVE_OUTCOMES = {"Home Win", "Draw", "Away Win"}


# ===================================================== parsing helpers

EMPTY_SUMMARY = {"matches": 0, "wins": 0, "draws": 0, "losses": 0,
                 "gf": 0, "ga": 0, "avg_gf": 0.0, "avg_ga": 0.0}

EMPTY_FORM = {"form": "", "wins": 0, "draws": 0, "losses": 0,
              "overs": 0, "unders": 0, "total": 0}


def parse_summary(summary):
    """Standings summary string -> counts. A team with no history (newly
    promoted) is an expected state, so this returns a zero record rather
    than raising."""
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

    return {"matches": matches, "wins": wins, "draws": draws, "losses": losses,
            "gf": gf, "ga": ga, "avg_gf": avg_gf, "avg_ga": avg_ga}


def parse_form(form_str):
    """Form string -> counts. Result markets carry W/D/L, goal markets
    carry O/U (the prediction scripts pick which one to attach). Missing
    form returns a fully populated zero record; callers check total == 0."""
    if form_str is None or isinstance(form_str, float) or not str(form_str).strip():
        return dict(EMPTY_FORM)
    form = str(form_str).replace("-", "")
    return {"form": form, "wins": form.count("W"), "draws": form.count("D"),
            "losses": form.count("L"), "overs": form.count("O"),
            "unders": form.count("U"), "total": len(form)}


def parse_hist(s):
    """History cell -> percentage float, or NaN. Accepts '60%', '3/5', a
    bare number, or '-'."""
    try:
        if pd.isna(s):
            return np.nan
        s = str(s).strip()
        if s in ('', '-'):
            return np.nan
        if '/' in s:
            num, den = (p.strip() for p in s.split('/'))
            if not num or not den or int(den) == 0:
                return np.nan
            return (int(num) / int(den)) * 100
        if s.endswith('%'):
            return float(s.rstrip('%').strip())
        v = float(s)
        return v * 100 if v <= 1 else v
    except Exception:
        return np.nan


def prediction_label(pred):
    """Display label for any prediction code, combos included."""
    pred = str(pred)
    if '+' in pred:
        side, goal = pred.split('+', 1)
        return f"{PREDICTION_LABELS.get(side, side)} + {PREDICTION_LABELS.get(goal, goal)}"
    return PREDICTION_LABELS.get(pred, pred)


def legs_of(pred):
    """The market codes a prediction is built from -- one for a single,
    two for a combo."""
    return str(pred).split('+') if '+' in str(pred) else [str(pred)]


# ============================================== supporting-signal scores
#
# Each returns 0..1 (higher = the evidence supports the pick) or None when
# the underlying data isn't there.
#
# None is scored as NEUTRAL (0.5), not as zero and not by dropping the
# component. All three choices are defensible and they rank differently,
# so the reasoning matters:
#
#   zero        -- punishes a newly promoted side for history it could not
#                  possibly have. Wrong.
#   drop+renorm -- a pick with NO corroborating data is then scored on edge
#                  alone, and since edge saturates at EDGE_SCALE it lands a
#                  perfect score. A fixture we know nothing about would
#                  outrank one with strong form, standings and H2H behind
#                  it, which inverts the whole point of this ranking.
#   neutral     -- an unknown signal neither helps nor hurts, so a
#                  well-corroborated pick can still outscore a bare one at
#                  the same edge. This is what "edge AND strong form AND
#                  strong statistics" asks for.

NEUTRAL_SUPPORT = 0.5

def _clamp01(x):
    return float(min(max(x, 0.0), 1.0))


def _form_support(code, home_form, away_form):
    """Does recent form point the same way as this pick?"""
    hf, af = home_form, away_form
    if hf['total'] == 0 and af['total'] == 0:
        return None

    def rate(f, key):
        return (f[key] / f['total']) if f['total'] else None

    if code == '1':
        parts = [rate(hf, 'wins'), rate(af, 'losses')]
    elif code == '2':
        parts = [rate(af, 'wins'), rate(hf, 'losses')]
    elif code == 'X':
        parts = [rate(hf, 'draws'), rate(af, 'draws')]
    else:
        # Every goal market: the attached form is the O/U string, so the
        # overs rate is the directly relevant signal.
        parts = [rate(hf, 'overs'), rate(af, 'overs')]

    parts = [p for p in parts if p is not None]
    return _clamp01(sum(parts) / len(parts)) if parts else None


def _stats_support(code, home_stats, away_stats):
    """Do season standings point the same way as this pick?"""
    hs, as_ = home_stats, away_stats
    if hs['matches'] == 0 and as_['matches'] == 0:
        return None

    if code in RESULT_MARKETS:
        h_rate = (hs['wins'] / hs['matches']) if hs['matches'] else None
        a_rate = (as_['wins'] / as_['matches']) if as_['matches'] else None
        if h_rate is None and a_rate is None:
            return None
        h_rate = h_rate if h_rate is not None else 0.0
        a_rate = a_rate if a_rate is not None else 0.0
        if code == '1':
            # Centred on 0.5 so an evenly matched pair scores neutral
            # rather than looking like weak evidence for the home side.
            return _clamp01(0.5 + (h_rate - a_rate) / 2)
        if code == '2':
            return _clamp01(0.5 + (a_rate - h_rate) / 2)
        # A draw is supported by the two sides being close, not by either
        # being good -- so score the ABSENCE of a gap.
        return _clamp01(1.0 - abs(h_rate - a_rate))

    # Goal markets: compare the fixture's expected goal total against the
    # line being bet. Expected total = what the home side scores and the
    # away side concedes, plus the mirror.
    exp_total = (hs['avg_gf'] + as_['avg_ga']) / 2 + (as_['avg_gf'] + hs['avg_ga']) / 2
    if exp_total <= 0:
        return None

    if code in GOAL_LINES:
        line = GOAL_LINES[code]
    elif code in ('hO1_5', 'aO1_5'):
        line = 1.5
    elif code in ('hO2_5', 'aO2_5'):
        line = 2.5
    else:                      # GG -- both teams scoring needs ~2 goals
        line = 2.0

    # Half a goal above the line is decisive support; half below is
    # decisive against. Linear between, which is honest about how coarse
    # a season average is.
    return _clamp01(0.5 + (exp_total - line))


def _h2h_support(hist_value):
    """Head-to-head strike rate as a 0..1 signal. NaN (no meetings on
    record) returns None rather than 0 -- see the block comment above."""
    if hist_value is None or pd.isna(hist_value):
        return None
    return _clamp01(float(hist_value) / 100.0)


def best_bet_score(row):
    """Composite ranking score, edge-dominant.

    Components that are unavailable for this row are dropped and the
    remaining weights renormalized, so a pick is never punished for
    missing data it could not have had.
    """
    edge = row.get('EdgeValue')
    edge_component = _clamp01((edge or 0.0) / EDGE_SCALE)

    home_stats = parse_summary(row.get('HomeTeam Stats'))
    away_stats = parse_summary(row.get('AwayTeam Stats'))
    home_form = parse_form(row.get('HomeForm'))
    away_form = parse_form(row.get('AwayForm'))

    # A combo is judged on the average of its legs. Note the attached form
    # belongs to the combo's result leg (predictions_merger copies the
    # side row), so the goal leg is scored on standings only.
    codes = legs_of(row.get('RawPrediction', row.get('Prediction')))
    form_parts = [v for v in (_form_support(c, home_form, away_form) for c in codes) if v is not None]
    stats_parts = [v for v in (_stats_support(c, home_stats, away_stats) for c in codes) if v is not None]

    form = (sum(form_parts) / len(form_parts)) if form_parts else NEUTRAL_SUPPORT
    stats = (sum(stats_parts) / len(stats_parts)) if stats_parts else NEUTRAL_SUPPORT
    h2h = _h2h_support(row.get('HistValue'))
    h2h = NEUTRAL_SUPPORT if h2h is None else h2h

    return (W_EDGE * edge_component + W_FORM * form
            + W_STATS * stats + W_H2H * h2h)


def support_notes(row):
    """Short human-readable why-this-pick fragments for the markdown."""
    codes = legs_of(row.get('RawPrediction', row.get('Prediction')))
    hf, af = parse_form(row.get('HomeForm')), parse_form(row.get('AwayForm'))
    hs, as_ = parse_summary(row.get('HomeTeam Stats')), parse_summary(row.get('AwayTeam Stats'))

    notes = []
    f = [v for v in (_form_support(c, hf, af) for c in codes) if v is not None]
    if f:
        notes.append(f"form {sum(f)/len(f)*100:.0f}%")
    s = [v for v in (_stats_support(c, hs, as_) for c in codes) if v is not None]
    if s:
        notes.append(f"standings {sum(s)/len(s)*100:.0f}%")
    h = _h2h_support(row.get('HistValue'))
    if h is not None:
        notes.append(f"H2H {h*100:.0f}%")
    return ", ".join(notes) if notes else "no supporting history on record"


# ==================================================== reasoning (VIP)

def generate_reasoning(row):
    """Prose explanation for a pick. Combos explain each leg in turn."""
    codes = legs_of(row.get('RawPrediction', row.get('Prediction')))
    if len(codes) > 1:
        parts = [_reason_for_code(c, row) for c in codes]
        return (" ".join(parts) +
                " Combined probability assumes the two legs are independent, "
                "so treat it as indicative.")
    return _reason_for_code(codes[0], row)


def _reason_for_code(code, row):
    home_stats = parse_summary(row["HomeTeam Stats"])
    away_stats = parse_summary(row["AwayTeam Stats"])
    home_form = parse_form(row["HomeForm"])
    away_form = parse_form(row["AwayForm"])

    home_1 = (f"{row['HomeTeam']} has won {home_form['wins']} of their last {home_form['total']} home games."
              if home_form['total'] else f"{row['HomeTeam']} has no recent home form on record.")
    away_2 = (f"{row['AwayTeam']} has won {away_form['wins']} of their last {away_form['total']} games."
              if away_form['total'] else f"{row['AwayTeam']} has no recent form on record.")

    home_goals = f"{row['HomeTeam']} averages {home_stats['avg_gf']:.1f} goals at home."
    away_goals = f"{row['AwayTeam']} averages {away_stats['avg_gf']:.1f} goals away."
    home_concede = f"{row['HomeTeam']} concedes {home_stats['avg_ga']:.1f} at home."
    away_concede = f"{row['AwayTeam']} concedes {away_stats['avg_ga']:.1f} away."

    if code == '1':
        return f"{home_1} {away_2} {away_concede} Strong home record supports this."
    if code == '2':
        return f"{away_2} {home_1} {home_concede} Away form makes them favourites."
    if code == 'X':
        return f"Balanced recent form: {home_1} {away_2} A draw is a strong possibility."
    if code == 'GG':
        return (f"{home_goals} {away_goals} Both teams tend to concede "
                f"({home_concede} {away_concede}), making goals at both ends likely.")
    if code.startswith('hO'):
        return f"{home_goals} {away_concede} The home side's attack should produce the goals."
    if code.startswith('aO'):
        return f"{away_goals} {home_concede} The away side's attack looks likely to score well."
    if code.startswith('O'):
        return f"{home_goals} {away_goals} {home_concede} {away_concede} Both sides look capable of goals."
    return "Prediction based on statistical analysis."


# ======================================================== ranking utils

def _rank(picks):
    """Shared quality ranking: (1) a >80% pick with no H2H on record,
    (2) picks with H2H, by probability then H2H, (3) everything else."""
    df = picks.copy()
    df['priority'] = np.where(
        (df['PredValue'] > 80) & (df['HistValue'].isna()), 1,
        np.where(df['HistValue'].notna(), 2, 3))
    df['Hist_for_sort'] = df['HistValue'].fillna(-1)
    return df.sort_values(by=['priority', 'PredValue', 'Hist_for_sort'],
                          ascending=[True, False, False])


def dedup_one_per_match(picks, limit=None):
    """One prediction per match -- for the outputs where a fixture can only
    sensibly occupy one slot (VIP list, free picks)."""
    deduped = _rank(picks).drop_duplicates(subset='Match', keep='first').drop(
        columns=['priority', 'Hist_for_sort'])
    return deduped.head(limit) if limit is not None else deduped


def resolve_match_conflicts(picks):
    """Keep several markets per match, but only the best of Home Win /
    Draw / Away Win -- those three are mutually exclusive. Everything else
    can coexist (Over 2.5 and BTTS can both land)."""
    df_sorted = _rank(picks)
    is_outcome = df_sorted['Prediction'].isin(MUTUALLY_EXCLUSIVE_OUTCOMES)
    outcome_rows = df_sorted[is_outcome].drop_duplicates(subset='Match', keep='first')
    other_rows = df_sorted[~is_outcome]
    resolved = pd.concat([outcome_rows, other_rows]).sort_values(
        by=['priority', 'PredValue', 'Hist_for_sort'], ascending=[True, False, False])
    return resolved.drop(columns=['priority', 'Hist_for_sort'])


# ============================================================== outputs

def savetoexcel_format(df, excel_file):
    print('Saving into a nice excel..', excel_file)
    df.to_excel(excel_file, index=False, sheet_name="Sheet1")

    wb = load_workbook(excel_file)
    ws = wb["Sheet1"]
    ws.freeze_panes = "A2"

    last_col_letter = get_column_letter(ws.max_column)
    table = Table(displayName="PredTable", ref=f"A1:{last_col_letter}{ws.max_row}")
    table.tableStyleInfo = TableStyleInfo(
        name="TableStyleLight1", showFirstColumn=False, showLastColumn=False,
        showRowStripes=True, showColumnStripes=False)
    ws.add_table(table)

    for col in ws.columns:
        width = max((len(str(c.value)) for c in col if c.value is not None), default=0)
        ws.column_dimensions[get_column_letter(col[0].column)].width = width + 2

    wb.save(excel_file)


def _write(path, content):
    if not os.path.exists(PUBLISHPATH):
        os.makedirs(PUBLISHPATH)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def write_public(public, date_str):
    """Free tier: 3 picks drawn at random from the highest-edge qualifying
    picks. Random rather than best-first on purpose -- the free list is a
    shop window, and always handing over the single strongest pick leaves
    nothing behind the paywall."""
    txt = f"📊 <b>Free Picks — {date_str}</b>\n\n"
    if public.empty:
        txt += "No picks qualified today.\n"
    else:
        for _, row in public.iterrows():
            # An unpriced market (GG, team-goal) has no odd to show; print
            # the pick without one rather than the literal 'nan'.
            odd = f" @ {row['OddValue']:.2f}" if pd.notna(row['OddValue']) else ""
            txt += f"• <b>{row['Match']}</b> → {row['Prediction']}{odd}\n"
    txt += ("\n📩 <a href='https://rebrand.ly/betprophet-m'>Join BetProphet.AI VIP now</a> "
            "for today's premium picks before kick-off!\n"
            "💎Just €8/month — one winning bet covers your subscription! ✅")
    _write(f"{PUBLISHPATH}/Public_{date_str}.txt", txt)


def write_vip(vip, date_str):
    """VIP tier: the top 10, each with its own reasoning."""
    txt = f"💎 <b>VIP Picks — {date_str}</b>\n\n"
    for _, row in vip.iterrows():
        odd = f" @ {row['OddValue']:.2f}" if pd.notna(row['OddValue']) else ""
        txt += (f"{row['Confidence']} <b>{row['Match']}</b> → {row['Prediction']}"
                f"{odd} ({row['Prediction %']} | {row['History H2H']})\n"
                f"  <i>{row['Reasoning']}</i>\n\n")
    _write(f"{PUBLISHPATH}/VIP_{date_str}.txt", txt)


def write_best_bets(best, date_str):
    """Best Bets, in markdown."""
    if best.empty:
        _write(f"{PUBLISHPATH}/BestBets_{date_str}.txt",
               f"# Best Bets — {date_str}\n\n"
               f"No pick cleared the minimum edge of {MIN_EDGE*100:.0f}pp today. "
               f"Sitting this one out is the right call.\n")
        return

    lines = [f"# Best Bets — {date_str}", "",
             "_Ranked on model edge vs the bookmaker price, adjusted for recent "
             "form, head-to-head record and league standings._", ""]

    for _, row in best.iterrows():
        lines.append(f"## {row['Match']} ({row['Division']})")
        lines.append("")
        lines.append(f"**{row['Prediction']}** @ **{row['OddValue']:.2f}**")
        lines.append("")
        lines.append(f"- Model {row['ModelProb']*100:.0f}% vs market "
                     f"{row['ImpliedValue']*100:.0f}% -> edge **+{row['EdgeValue']*100:.1f}pp**")
        lines.append(f"- Supporting: {row['SupportNotes']}")
        lines.append(f"- {row['Reasoning']}")
        if '+' in str(row['RawPrediction']):
            lines.append("- _Combo: probability assumes independent legs, so the "
                         "edge here is indicative._")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("Edge is how much more likely the model rates an outcome than the "
                 "price implies. Higher edge means better long-run value, not a "
                 "guarantee on any single bet.")
    _write(f"{PUBLISHPATH}/BestBets_{date_str}.txt", "\n".join(lines))


# ================================================================= main

def load_today(day):
    path = os.path.join(DATAPATH, INPUT_TEMPLATE.format(date=day))
    if not os.path.exists(path):
        print(f'No merged prediction file for {day} at {path}.')
        return pd.DataFrame()
    df = pd.read_csv(path)
    print(f'Loaded {len(df)} predictions from {path}.')
    return df


def prepare(df):
    """Shared derived columns. Both tiers and Best Bets used to compute
    these independently, from the same file, in two scripts."""
    df = df.copy()
    df['RawPrediction'] = df['Prediction']
    df['Prediction'] = df['Prediction'].apply(prediction_label)

    code_to_name = {v: k for k, v in LEAGUES.items()}
    df['Division'] = df['Division'].map(code_to_name).fillna(df['Division'])

    df['HomeTeam'] = df['HomeTeam'].apply(team_utils.display_name)
    df['AwayTeam'] = df['AwayTeam'].apply(team_utils.display_name)
    df['Match'] = df['HomeTeam'] + " vs " + df['AwayTeam']

    df['ModelProb'] = pd.to_numeric(df['Prediction %'], errors='coerce')
    df['PredValue'] = df['ModelProb'] * 100
    df['OddValue'] = pd.to_numeric(df['Odd'], errors='coerce')
    df['ImpliedValue'] = pd.to_numeric(df['Implied %'], errors='coerce')
    df['EdgeValue'] = pd.to_numeric(df['Edge %'], errors='coerce')

    df.rename(columns={'History %': 'History H2H'}, inplace=True)
    df['HistValue'] = df['History H2H'].apply(parse_hist)

    # Explicit format first (times are written as HH:MM). The free-parse
    # fallback runs ONLY if something failed that format -- calling it
    # unconditionally makes pandas emit a "could not infer format" warning
    # on every run even when nothing needed it, burying the script's real
    # output in CI logs.
    times = pd.to_datetime(df['Time'], format='%H:%M', errors='coerce')
    if times.isna().any():
        times = times.fillna(pd.to_datetime(df['Time'], errors='coerce'))
    df['Time'] = times.dt.time.fillna(datetime.time(0, 0))
    df['AdjustedDate'] = date_utils.adjusted_date_series(df['Date'], df['Time'])

    w_model, w_hist = model_config.get_blend_weights()
    df['ConfScore'] = (w_model * df['PredValue']) + (w_hist * df['HistValue'].fillna(df['PredValue']))
    df['Confidence'] = df['ConfScore'].apply(
        lambda s: "🟢" if s >= 80 else ("🟠" if s >= 60 else "🔴"))

    df['Prediction %'] = df['ModelProb'].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "-")
    return df


def market_bars(pred):
    """(min_bar, strong_bar, base_rate) for a prediction, as probabilities.

    For a single market these are model_config's values verbatim. For a
    combo each is the PRODUCT across its legs, because a combo is an
    intersection: '1+O2_5' at 36% is not a weak pick judged against Home
    Win's 64% bar, it is roughly what two independent legs at 62% and 58%
    should produce. Judging a combo against one leg's single-market bar
    rejected every combo ever generated, which is how this was found.
    """
    min_bar = strong_bar = base = 1.0
    for c in legs_of(pred):
        min_bar *= model_config.get_market_min_prob(c)
        strong_bar *= model_config.get_strong_prob(c)
        base *= model_config.get_base_rate(c)
    return min_bar, strong_bar, base


def qualifying(df):
    """The tipster-quality gate: a pick clears its own market's bar,
    beating that market's base rate, either outright or with H2H support.
    Per-market rather than one flat number -- Over 1.5 lands ~75% of the
    time and would sail through any threshold set for a 1X2 market."""
    bars = [market_bars(p) for p in df['RawPrediction']]
    prob = df['ModelProb'].fillna(0)

    min_bar = pd.Series([b[0] * 100 for b in bars], index=df.index)
    strong_bar = pd.Series([b[1] * 100 for b in bars], index=df.index)
    lift_ok = pd.Series(
        [(p - b[2]) >= model_config.MIN_LIFT_OVER_BASE for b, p in zip(bars, prob)],
        index=df.index)

    HIST_SUPPORT = 50
    ok = lift_ok & ((df['PredValue'] >= strong_bar) |
                    ((df['PredValue'] >= min_bar) & (df['HistValue'] >= HIST_SUPPORT)))
    picks = df[ok].sort_values(by=['PredValue', 'HistValue'], ascending=False)

    if picks.empty:
        print('WARNING: nothing met the tipster gate; falling back to top 5 by probability.')
        picks = df.sort_values(by='PredValue', ascending=False).head(5)
    return picks


def select_free(vip_all):
    """FREE_PICKS drawn at random from the FREE_POOL highest-edge picks.

    Unpriced markets (GG, the team-goal markets) are excluded from the
    pool because they have no edge to rank on -- but if NOTHING is priced
    today, the pool falls back to the top of the VIP list so the free post
    is never silently empty.
    """
    if vip_all.empty:
        return vip_all

    pool = vip_all[vip_all['EdgeValue'].notna()]
    if pool.empty:
        print('No priced picks today; free list falls back to the top VIP picks.')
        pool = vip_all
    else:
        pool = pool.sort_values('EdgeValue', ascending=False)

    pool = pool.head(FREE_POOL)
    return pool.sample(n=min(FREE_PICKS, len(pool)), random_state=FREE_SEED)


def select_best_bets(df):
    """3-6 picks, one per match, ranked by the edge-dominant composite."""
    pool = df[df['EdgeValue'].notna() & df['OddValue'].notna()].copy()
    pool = pool[(pool['EdgeValue'] >= MIN_EDGE) & (pool['ModelProb'] >= MIN_MODEL_PROB)]
    if pool.empty:
        print(f'WARNING: no pick cleared the {MIN_EDGE:.0%} edge floor.')
        return pool

    pool['BestScore'] = pool.apply(best_bet_score, axis=1)
    pool['SupportNotes'] = pool.apply(support_notes, axis=1)
    pool = pool.sort_values('BestScore', ascending=False).drop_duplicates(subset='Match', keep='first')

    if len(pool) < BEST_MIN_PICKS:
        print(f'Only {len(pool)} pick(s) cleared the floor (target {BEST_MIN_PICKS}-{BEST_MAX_PICKS}).')
    return pool.head(BEST_MAX_PICKS)


def main(reference=None):
    day = (reference or datetime.date.today()).strftime('%Y-%m-%d')
    raw = load_today(day)
    if raw.empty:
        print('Nothing to publish.')
        return

    df_full = prepare(raw)

    for match_date, df_date in df_full.groupby('AdjustedDate'):
        date_str = pd.to_datetime(match_date).strftime('%Y-%m-%d')
        print(f'Processing {date_str}..')

        df_date = df_date.copy()
        df_date['Reasoning'] = df_date.apply(generate_reasoning, axis=1)

        picks = qualifying(df_date)
        vip_all = dedup_one_per_match(picks, limit=None)

        # ---- free tier: 3 at random from the highest-edge picks
        public = select_free(vip_all)
        print(f'Free: {len(public)} pick(s) drawn from the top {FREE_POOL} by edge '
              f'(of {len(vip_all)} qualifying).')
        write_public(public, date_str)

        # ---- VIP: top 10 with reasoning, plus every qualifying prediction
        vip = vip_all.head(VIP_PICKS)
        print(f'VIP: {len(vip)} pick(s) with reasoning.')
        write_vip(vip, date_str)

        excel_picks = resolve_match_conflicts(picks).copy()
        public_pairs = set(zip(public['Match'], public['Prediction'])) if not public.empty else set()
        excel_picks['PickedforFree'] = excel_picks.apply(
            lambda r: (r['Match'], r['Prediction']) in public_pairs, axis=1)
        excel_picks['Odds'] = excel_picks['OddValue'].apply(
            lambda x: f"{x:.2f}" if pd.notna(x) else "-")
        df_save = excel_picks[["Division", "Date", "Time", "HomeTeam", "AwayTeam",
                               "Prediction", "Odds", "Prediction %", "History H2H",
                               "HomeForm", "AwayForm", "Reasoning",
                               "HomeTeam Stats", "AwayTeam Stats", "PickedforFree"]]
        savetoexcel_format(df_save, f"{PUBLISHPATH}/VIP_{date_str}.xlsx")

        # ---- best bets
        best = select_best_bets(df_date)
        print(f'Best Bets: {len(best)} pick(s).')
        write_best_bets(best, date_str)

    print('Publishing completed..')


if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__) or '.')
    main()
