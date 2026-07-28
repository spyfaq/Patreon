#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
best_bets_selector.py

Picks the 4-6 best matches/markets to bet on from the latest merged
prediction file, ranked by EDGE (model probability vs bookmaker's
implied probability) rather than raw model confidence.

Why edge instead of raw confidence:
    A 70%-confidence pick priced at odds of 1.30 (implied ~77%) is a
    bad bet -- the market already rates it higher than your model does.
    A 55%-confidence pick at odds of 2.20 (implied ~45%) is a good bet.
    Betting on raw confidence alone ignores the price you're getting.

Pipeline position: run this AFTER predictions_merger.py has produced
the combined `my_prediction_data_{date1}_{date2}.csv` file.

Usage:
    python best_bets_selector.py
"""

import os
import re
import datetime
import numpy as np
import pandas as pd

DATAPATH = 'predictions_data/'
PUBLISHPATH = 'publish/'

MIN_PICKS = 4
MAX_PICKS = 6

# Minimum edge (model_prob - implied_prob) required to even consider a pick.
# 0.03 = model must think the outcome is at least 3 percentage points more
# likely than the market does. Raise this if you want fewer, higher-conviction picks.
MIN_EDGE = 0.03

# Minimum model probability required regardless of edge, to avoid low-probability
# longshots with technically-large edge but low hit rate.
MIN_MODEL_PROB = 0.45

# Prediction codes we can currently price against bookmaker odds.
# (Extend odd_addition()/fetch_market_odds() below if you want more markets priced.)
PRICED_MARKETS = {'1', 'X', '2', 'O2_5', 'U2_5'}

PREDICTION_LABELS = {
    "O1_5": "Over 1.5 Goals", "O2_5": "Over 2.5 Goals", "O3_5": "Over 3.5 Goals",
    "1": "Home Win", "2": "Away Win", "X": "Draw", "GG": "Both Teams to Score",
    "aO1_5": "Away team Over 1.5 Goals", "aO2_5": "Away team Over 2.5 Goals",
    "hO1_5": "Home team Over 1.5 Goals", "hO2_5": "Home team Over 2.5 Goals",
}


def newest_predictions() -> str:
    print('Searching latest merged prediction file..')
    files = os.listdir(DATAPATH)
    paths = [os.path.join(DATAPATH, f) for f in files if 'my_prediction_data_' in f]
    if not paths:
        print('ERROR: No merged prediction file found..')
        raise FileNotFoundError("No file matching 'my_prediction_data_*' in " + DATAPATH)
    file = max(paths, key=os.path.getctime)
    print('File found..', file)
    return file


def fetch_market_odds() -> pd.DataFrame:
    """Pull fresh fixture odds covering 1X2 and Over/Under 2.5, from both the
    main and 'new league' football-data.co.uk fixture files (mirrors the
    logic already used in predictions_merger.odd_addition, extended to O/U).
    """
    print('Fetching market odds (1X2 + O/U 2.5)..')

    cols_main = ['Date', 'Time', 'Div', 'HomeTeam', 'AwayTeam', 'AvgH', 'AvgD', 'AvgA']
    ou_candidates = ['Avg>2.5', 'Avg<2.5']

    f1 = pd.read_csv('https://www.football-data.co.uk/fixtures.csv', encoding='utf-8-sig')
    have_ou_1 = [c for c in ou_candidates if c in f1.columns]
    f1 = f1[cols_main + have_ou_1]

    f2 = pd.read_csv('https://www.football-data.co.uk/new_league_fixtures.csv', encoding='utf-8-sig')
    f2 = f2.rename(columns={'Country': 'Div', 'Home': 'HomeTeam', 'Away': 'AwayTeam'})
    have_ou_2 = [c for c in ou_candidates if c in f2.columns]
    f2 = f2[[c for c in cols_main if c in f2.columns] + have_ou_2]

    odds = pd.concat([f1, f2], ignore_index=True)
    odds = odds.rename(columns={'Avg>2.5': 'AvgOver25', 'Avg<2.5': 'AvgUnder25'})
    odds['Date'] = pd.to_datetime(odds['Date'], format='%d/%m/%Y', errors='coerce')

    for c in ['AvgH', 'AvgD', 'AvgA', 'AvgOver25', 'AvgUnder25']:
        if c not in odds.columns:
            odds[c] = np.nan

    return odds[['HomeTeam', 'AwayTeam', 'AvgH', 'AvgD', 'AvgA', 'AvgOver25', 'AvgUnder25']]


def implied_prob_1x2(row) -> dict:
    """De-vig the 1X2 market by normalizing 1/odds so the three implied
    probabilities sum to 1 (removes the bookmaker's overround)."""
    raw = {}
    for k, col in [('1', 'AvgH'), ('X', 'AvgD'), ('2', 'AvgA')]:
        odd = row.get(col)
        raw[k] = (1.0 / odd) if pd.notna(odd) and odd > 0 else np.nan
    total = sum(v for v in raw.values() if pd.notna(v))
    if not total or not np.isfinite(total):
        return {'1': np.nan, 'X': np.nan, '2': np.nan}
    return {k: (v / total if pd.notna(v) else np.nan) for k, v in raw.items()}


def implied_prob_ou(row) -> dict:
    """De-vig the Over/Under 2.5 (two-way) market the same way."""
    over = row.get('AvgOver25')
    under = row.get('AvgUnder25')
    raw_o = (1.0 / over) if pd.notna(over) and over > 0 else np.nan
    raw_u = (1.0 / under) if pd.notna(under) and under > 0 else np.nan
    total = np.nansum([raw_o, raw_u])
    if not total or not np.isfinite(total):
        return {'O2_5': np.nan, 'U2_5': np.nan}
    return {
        'O2_5': raw_o / total if pd.notna(raw_o) else np.nan,
        'U2_5': raw_u / total if pd.notna(raw_u) else np.nan,
    }


def market_odd_for_pick(row, pred_code) -> float:
    mapping = {'1': 'AvgH', 'X': 'AvgD', '2': 'AvgA',
               'O2_5': 'AvgOver25', 'U2_5': 'AvgUnder25'}
    col = mapping.get(pred_code)
    return row.get(col) if col else np.nan


def attach_edges(df: pd.DataFrame) -> pd.DataFrame:
    """For each prediction row, compute the model's implied edge over the
    bookmaker's de-vigged probability. Rows for markets we can't price
    (no odds available, e.g. GG, hO1_5, aO2_5) get edge = NaN and are
    excluded from the value-bet ranking, not silently treated as zero-edge.
    """
    df = df.copy()
    df['MarketOdd'] = np.nan
    df['ImpliedProb'] = np.nan
    df['Edge'] = np.nan
    df['ModelProb'] = df['Prediction %']  # already a 0-1 float upstream

    for idx, row in df.iterrows():
        pred = row['Prediction']
        if pred not in PRICED_MARKETS:
            continue

        odds_1x2 = implied_prob_1x2(row)
        odds_ou = implied_prob_ou(row)
        implied = {**odds_1x2, **odds_ou}.get(pred)
        odd_val = market_odd_for_pick(row, pred)

        if pd.notna(implied) and pd.notna(odd_val):
            df.at[idx, 'MarketOdd'] = odd_val
            df.at[idx, 'ImpliedProb'] = implied
            df.at[idx, 'Edge'] = row['ModelProb'] - implied

    return df


def expected_value_per_unit_stake(model_prob, odd) -> float:
    """EV of a 1-unit stake at decimal odds `odd`, given true win prob model_prob."""
    if pd.isna(model_prob) or pd.isna(odd):
        return np.nan
    return model_prob * (odd - 1) - (1 - model_prob)


def select_best_bets(df: pd.DataFrame) -> pd.DataFrame:
    priced = df[df['Edge'].notna()].copy()
    priced = priced[
        (priced['Edge'] >= MIN_EDGE) &
        (priced['ModelProb'] >= MIN_MODEL_PROB)
    ]

    if priced.empty:
        print('WARNING: No matches cleared the edge/probability thresholds today.')
        return priced

    priced['EV'] = priced.apply(
        lambda r: expected_value_per_unit_stake(r['ModelProb'], r['MarketOdd']), axis=1
    )

    # One pick per match: keep the highest-EV market for that fixture
    priced['Match'] = priced['HomeTeam'] + ' vs ' + priced['AwayTeam']
    priced = priced.sort_values('EV', ascending=False).drop_duplicates(subset='Match', keep='first')

    priced = priced.sort_values('EV', ascending=False)
    n_picks = min(MAX_PICKS, max(MIN_PICKS if len(priced) >= MIN_PICKS else len(priced), 0))
    best = priced.head(n_picks).copy()

    best['PredictionLabel'] = best['Prediction'].map(PREDICTION_LABELS).fillna(best['Prediction'])
    return best


def format_telegram(best: pd.DataFrame, date_str: str) -> str:
    if best.empty:
        return f"⚠️ <b>No qualifying value bets found for {date_str}.</b>\nNo picks met the minimum edge threshold today — sitting this one out is the right call."

    msg = f"🎯 <b>Best Bets — {date_str}</b>\n<i>Ranked by model edge vs bookmaker odds, not just confidence</i>\n\n"
    for _, row in best.iterrows():
        msg += (
            f"⚽ <b>{row['Match']}</b> ({row['Division']})\n"
            f"   → {row['PredictionLabel']} @ {row['MarketOdd']:.2f}\n"
            f"   Model: {row['ModelProb']*100:.0f}% | Market implies: {row['ImpliedProb']*100:.0f}% "
            f"| Edge: +{row['Edge']*100:.1f}pp | EV: {row['EV']:+.2f} per unit\n\n"
        )
    msg += "📌 Edge = how much more likely our model thinks this is vs. what the odds imply. Higher edge and EV means better long-run value — not a guarantee on any single bet."
    return msg


def main():
    filename = newest_predictions()
    print('Loading merged predictions..', filename)
    df = pd.read_csv(filename)

    odds = fetch_market_odds()
    df = df.merge(odds, on=['HomeTeam', 'AwayTeam'], how='left')

    df = attach_edges(df)
    best = select_best_bets(df)

    date_str = datetime.date.today().strftime('%Y-%m-%d')
    tg_text = format_telegram(best, date_str)

    if not os.path.exists(PUBLISHPATH):
        os.makedirs(PUBLISHPATH)

    with open(f"{PUBLISHPATH}/BestBets_{date_str}.txt", "w", encoding="utf-8") as f:
        f.write(tg_text)

    if not best.empty:
        out_cols = ['Division', 'Date', 'Time', 'HomeTeam', 'AwayTeam', 'PredictionLabel',
                    'ModelProb', 'MarketOdd', 'ImpliedProb', 'Edge', 'EV']
        best[out_cols].to_csv(f"{PUBLISHPATH}/BestBets_{date_str}.csv", index=False)
        print(f'Selected {len(best)} best bets.', best["Match"].tolist())
    else:
        print('WARNING: No best bets selected today.')

    print(tg_text)


if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))

    try:
        main()
    except Exception as e:
        print("CRITICAL: Exception occurred while running best_bets_selector", e)
        raise
