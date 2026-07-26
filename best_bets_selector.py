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

Also builds a BET BUILDER section: same-match combos (e.g. Home Win +
Over 2.5 Goals) using the EXACT joint probability computed upstream
from the Dixon-Coles score grid (majorleague_predictions.py /
minorleague_predictions.py), not an independence assumption. Combos
built from legs we have real odds for (1X2 + Over/Under 2.5) get
compared against a de-vigged "naive independence" baseline built from
those same singles odds -- there's no bookmaker-quoted price for the
exact combo in our data source, so this baseline is the best available
reference point, not a real market price. Combos involving unpriced
legs (GG, O1.5/O3.5, home/away-specific) are still shown with the
model's probability, just without an edge/EV ranking.

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
from jsonlogger_class import JSONLogger
import date_utils
import team_utils
import odds_client
import odds_utils

LOGPATH = 'logs/bestbets/'
LOGNAME = '{date}_bestbets_logs'
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

# Bet builder (combo) selection is intentionally a smaller, supplementary
# list: combos are intersections of two events, so they're inherently
# lower-probability than either single-market pick alone.
COMBO_MIN_PICKS = 1
COMBO_MAX_PICKS = 3
COMBO_MIN_EDGE = 0.03
COMBO_MIN_MODEL_PROB = 0.25

# Accumulator ("suggested bets") slip: a single combined ticket across
# several DIFFERENT matches (as opposed to best/combo_best above, which
# are independent single-match suggestions). Legs are chosen from the
# same priced singles + combos pools, one leg per match, greedily by
# edge, until the combined (multiplied) odds clears TARGET_AGG_ODD or the
# leg count hits ACC_MAX_LEGS -- whichever comes first.
ACC_MIN_LEGS = 4
ACC_MAX_LEGS = 6
TARGET_AGG_ODD = 4.0

# Prediction codes we can currently price against bookmaker odds. O2_5 has
# a real 2-way market (Over/Under 2.5) to de-vig against; O1_5/O3_5 have no
# real market at all -- their odds are derived from O2_5 (see
# odds_utils.py) and priced as a naive (non-de-vigged) implied probability
# -- see implied_prob_ou(). It never predicts "Under", so there's no
# 'U2_5' row to price.
PRICED_MARKETS = {'1', 'X', '2', 'O1_5', 'O2_5', 'O3_5'}

# Of the goal-market legs a combo can pair with a 1X2 side: O2_5 has a real
# bookmaker odd; O1_5/O3_5 now have a derived one (see odds_utils.py) and
# are priced the same way. Combos using any other goal leg (GG, hO2_5,
# etc.) can still be shown (model probability only) but can't be priced
# against a market.
PRICED_COMBO_GOAL_LEGS = {'O1_5', 'O2_5', 'O3_5'}

PREDICTION_LABELS = {
    "O1_5": "Over 1.5 Goals", "O2_5": "Over 2.5 Goals", "O3_5": "Over 3.5 Goals",
    "1": "Home Win", "2": "Away Win", "X": "Draw", "GG": "Both Teams to Score",
    "aO1_5": "Away team Over 1.5 Goals", "aO2_5": "Away team Over 2.5 Goals",
    "hO1_5": "Home team Over 1.5 Goals", "hO2_5": "Home team Over 2.5 Goals",
}


def prediction_label(pred: str) -> str:
    """Human-readable label for any prediction code, including bet-builder
    combos (e.g. '1+O2_5' -> 'Home Win + Over 2.5 Goals')."""
    if '+' in pred:
        side, goal_leg = pred.split('+', 1)
        return f"{PREDICTION_LABELS.get(side, side)} + {PREDICTION_LABELS.get(goal_leg, goal_leg)}"
    return PREDICTION_LABELS.get(pred, pred)


def newest_predictions() -> str:
    logger.log('info', 'Searching latest merged prediction file..')
    files = os.listdir(DATAPATH)
    paths = [os.path.join(DATAPATH, f) for f in files if 'my_prediction_data_' in f]
    if not paths:
        logger.log('error', 'No merged prediction file found..')
        raise FileNotFoundError("No file matching 'my_prediction_data_*' in " + DATAPATH)
    file = max(paths, key=os.path.getctime)
    logger.log('info', 'File found..', info=file)
    return file


def fetch_market_odds() -> pd.DataFrame:
    """Pull fresh fixture odds covering 1X2 and Over/Under 2.5 (Over 1.5 /
    Over 3.5 are derived from Over 2.5, see odds_utils.py): domestic from
    football-data.co.uk (mirrors predictions_merger.odd_addition, extended
    to O/U), plus international (CL/WC/EC) from The Odds API via
    odds_client.py, since football-data.co.uk has nothing for those 3
    competitions -- this script previously only ever priced domestic
    matches, so Best Bets/Bet Builder/Suggested Bets silently never
    considered an international pick even when predictions_tier.py's
    Excel had odds for one.
    """
    logger.log('info', 'Fetching market odds (1X2 + O/U 2.5)..')

    cols_main = ['Date', 'Time', 'Div', 'HomeTeam', 'AwayTeam', 'AvgH', 'AvgD', 'AvgA']
    ou_candidates = ['Avg>2.5', 'Avg<2.5']

    f1 = pd.read_csv('https://www.football-data.co.uk/fixtures.csv', encoding='utf-8-sig')
    have_ou_1 = team_utils.find_columns(f1.columns, ou_candidates)
    logger.log('info', f'Main fixtures O/U columns found: {have_ou_1 or "NONE"}',
               info=str(list(f1.columns)))
    f1 = f1[cols_main + have_ou_1]

    f2 = pd.read_csv('https://www.football-data.co.uk/new_league_fixtures.csv', encoding='utf-8-sig')
    f2 = f2.rename(columns={'Country': 'Div', 'Home': 'HomeTeam', 'Away': 'AwayTeam'})
    have_ou_2 = team_utils.find_columns(f2.columns, ou_candidates)
    logger.log('info', f'New-league fixtures O/U columns found: {have_ou_2 or "NONE"}',
               info=str(list(f2.columns)))
    f2 = f2[[c for c in cols_main if c in f2.columns] + have_ou_2]

    odds = pd.concat([f1, f2], ignore_index=True)
    odds = odds.rename(columns={c: 'AvgOver25' for c in have_ou_1 + have_ou_2 if c.strip().lower() == 'avg>2.5'})
    odds = odds.rename(columns={c: 'AvgUnder25' for c in have_ou_1 + have_ou_2 if c.strip().lower() == 'avg<2.5'})
    odds['Date'] = pd.to_datetime(odds['Date'], format='%d/%m/%Y', errors='coerce')

    for c in ['AvgH', 'AvgD', 'AvgA', 'AvgOver25', 'AvgUnder25']:
        if c not in odds.columns:
            odds[c] = np.nan

    odds_cols = ['HomeTeam', 'AwayTeam', 'AvgH', 'AvgD', 'AvgA', 'AvgOver25', 'AvgUnder25']
    n_over25 = odds['AvgOver25'].notna().sum()
    logger.log('info', f'Domestic odds: {len(odds)} fixtures, {n_over25} with an Over 2.5 price.')

    try:
        intl_odds = odds_client.fetch_all_international_odds(logger=logger)
    except Exception as e:
        logger.log('warning', 'Could not fetch international odds..', info=str(e))
        intl_odds = pd.DataFrame(columns=odds_cols)

    combined = pd.concat([odds[odds_cols], intl_odds[odds_cols] if not intl_odds.empty else intl_odds],
                          ignore_index=True)

    # football-data.co.uk (and The Odds API's totals market, fetched only
    # at the 2.5 line) never publishes Over 1.5 / Over 3.5 odds at all --
    # approximate them from the real Over 2.5 price (see odds_utils.py).
    combined['AvgOver15'], combined['AvgOver35'] = odds_utils.derive_over_under_odds(combined['AvgOver25'])
    return combined


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
    """De-vig the real Over/Under 2.5 (two-way) market. Over 1.5 and Over
    3.5 have no real market at all -- their odds (AvgOver15/AvgOver35)
    are estimated from Over 2.5 by a fixed offset (odds_utils.py), not a
    genuine priced market with an Under-side to normalize against. Their
    implied probability here is therefore the naive 1/odd (still
    includes the bookmaker's margin) -- a reasonable approximation given
    the data limitation, but not on the same footing as the de-vigged 2.5
    line."""
    over = row.get('AvgOver25')
    under = row.get('AvgUnder25')
    raw_o = (1.0 / over) if pd.notna(over) and over > 0 else np.nan
    raw_u = (1.0 / under) if pd.notna(under) and under > 0 else np.nan
    total = np.nansum([raw_o, raw_u])
    result = {'O2_5': np.nan, 'U2_5': np.nan}
    if total and np.isfinite(total):
        result = {
            'O2_5': raw_o / total if pd.notna(raw_o) else np.nan,
            'U2_5': raw_u / total if pd.notna(raw_u) else np.nan,
        }

    for pred, col in [('O1_5', 'AvgOver15'), ('O3_5', 'AvgOver35')]:
        odd = row.get(col)
        result[pred] = (1.0 / odd) if pd.notna(odd) and odd > 0 else np.nan

    return result


def market_odd_for_pick(row, pred_code) -> float:
    mapping = {'1': 'AvgH', 'X': 'AvgD', '2': 'AvgA',
               'O1_5': 'AvgOver15', 'O2_5': 'AvgOver25', 'O3_5': 'AvgOver35',
               'U2_5': 'AvgUnder25'}
    col = mapping.get(pred_code)
    return row.get(col) if col else np.nan


def attach_edges(df: pd.DataFrame) -> pd.DataFrame:
    """For each prediction row, compute the model's implied edge over the
    bookmaker's de-vigged probability. Rows for markets we can't price
    (no odds available, e.g. GG, hO1_5, aO2_5) get edge = NaN and are
    excluded from the value-bet ranking, not silently treated as zero-edge.
    Bet-builder combo rows (Prediction contains '+') are left alone here --
    see attach_combo_edges below, which prices them against a baseline
    instead of a real market odd (none exists for combos in our data).
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


def attach_combo_edges(df: pd.DataFrame) -> pd.DataFrame:
    """Bet builder pricing: for combo rows (Prediction like '1+O2_5'), build
    a "naive independence" baseline by multiplying the de-vigged single-leg
    probabilities from real odds, then compare the model's EXACT joint
    probability (computed from the score grid upstream) against that
    baseline. The gap between them is informative on its own: since the
    legs are correlated, the model's exact joint probability being higher
    than the naive-independence baseline is expected for a well-correlated
    combo (e.g. home win + over 2.5), and the size of that gap reflects how
    much correlation the model is capturing that a naive multiplication
    would miss.

    This is NOT a real bookmaker-quoted price for the combo -- our data
    source (football-data.co.uk) doesn't publish bet-builder odds, only
    1X2 and O/U 2.5 singles. Treat BaselineOdd/ComboEdge/ComboEV as a
    reference point, not a guaranteed price you'd actually get.
    """
    df = df.copy()
    df['BaselineProb'] = np.nan
    df['BaselineOdd'] = np.nan
    df['ComboEdge'] = np.nan

    is_combo = df['Prediction'].str.contains(r'\+', regex=True, na=False)
    for idx, row in df[is_combo].iterrows():
        side, goal_leg = row['Prediction'].split('+', 1)
        if goal_leg not in PRICED_COMBO_GOAL_LEGS:
            continue  # no odds behind this leg (e.g. GG, hO2_5, aO1_5) -- model prob only

        odds_1x2 = implied_prob_1x2(row)
        odds_ou = implied_prob_ou(row)
        side_prob = odds_1x2.get(side)
        goal_prob = odds_ou.get(goal_leg)

        if pd.notna(side_prob) and pd.notna(goal_prob):
            baseline = side_prob * goal_prob
            df.at[idx, 'BaselineProb'] = baseline
            df.at[idx, 'BaselineOdd'] = (1.0 / baseline) if baseline > 0 else np.nan
            df.at[idx, 'ComboEdge'] = row['ModelProb'] - baseline

    return df


def expected_value_per_unit_stake(model_prob, odd) -> float:
    """EV of a 1-unit stake at decimal odds `odd`, given true win prob model_prob."""
    if pd.isna(model_prob) or pd.isna(odd):
        return np.nan
    return model_prob * (odd - 1) - (1 - model_prob)


def _ranked_singles_pool(df: pd.DataFrame) -> pd.DataFrame:
    """Every single-market pick that clears the edge/probability bar,
    one per match, ranked by EV -- uncapped. select_best_bets() just caps
    and labels this; build_accumulator() draws from the full pool since
    it may need more than MAX_PICKS candidates to find enough legs across
    distinct matches."""
    priced = df[df['Edge'].notna()].copy()
    priced = priced[
        (priced['Edge'] >= MIN_EDGE) &
        (priced['ModelProb'] >= MIN_MODEL_PROB)
    ]
    if priced.empty:
        return priced

    priced['EV'] = priced.apply(
        lambda r: expected_value_per_unit_stake(r['ModelProb'], r['MarketOdd']), axis=1
    )
    priced['Match'] = priced['HomeTeam'] + ' vs ' + priced['AwayTeam']
    priced = priced.sort_values('EV', ascending=False).drop_duplicates(subset='Match', keep='first')
    return priced.sort_values('EV', ascending=False)


def select_best_bets(df: pd.DataFrame) -> pd.DataFrame:
    priced = _ranked_singles_pool(df)

    if priced.empty:
        logger.log('warning', 'No matches cleared the edge/probability thresholds today.')
        return priced

    if len(priced) < MIN_PICKS:
        logger.log('info', f'Only {len(priced)} value bet(s) cleared the threshold today (below the usual {MIN_PICKS}-{MAX_PICKS} target).')
    n_picks = min(MAX_PICKS, len(priced))
    best = priced.head(n_picks).copy()

    best['PredictionLabel'] = best['Prediction'].apply(prediction_label)
    return best


def _ranked_combo_pool(df: pd.DataFrame) -> pd.DataFrame:
    """Every bet-builder combo that clears the combo edge/probability
    bar, one per match, ranked by ComboEV -- uncapped. Mirrors
    _ranked_singles_pool; select_best_combo_bets() caps and labels this,
    build_accumulator() draws from the full pool."""
    is_combo = df['Prediction'].str.contains(r'\+', regex=True, na=False)
    priced = df[is_combo & df['ComboEdge'].notna()].copy()
    priced = priced[
        (priced['ComboEdge'] >= COMBO_MIN_EDGE) &
        (priced['ModelProb'] >= COMBO_MIN_MODEL_PROB)
    ]
    if priced.empty:
        return priced

    priced['ComboEV'] = priced.apply(
        lambda r: expected_value_per_unit_stake(r['ModelProb'], r['BaselineOdd']), axis=1
    )
    priced['Match'] = priced['HomeTeam'] + ' vs ' + priced['AwayTeam']
    priced = priced.sort_values('ComboEV', ascending=False).drop_duplicates(subset='Match', keep='first')
    return priced.sort_values('ComboEV', ascending=False)


def select_best_combo_bets(df: pd.DataFrame) -> pd.DataFrame:
    """Bet builder selection: same idea as select_best_bets, restricted to
    combo rows priced against the independence baseline (see
    attach_combo_edges). Kept as a separate, smaller supplementary list
    rather than mixed into the main single-market picks.
    """
    priced = _ranked_combo_pool(df)

    if priced.empty:
        logger.log('info', 'No bet-builder combos cleared the edge/probability thresholds today.')
        return priced

    if len(priced) < COMBO_MIN_PICKS:
        logger.log('info', f'Only {len(priced)} bet-builder combo(s) cleared the threshold today.')
    n_picks = min(COMBO_MAX_PICKS, len(priced))
    best = priced.head(n_picks).copy()

    best['PredictionLabel'] = best['Prediction'].apply(prediction_label)
    return best


def build_accumulator(df: pd.DataFrame) -> tuple:
    """Single combined ticket ('suggested bets' slip) across 4-6 DIFFERENT
    matches -- distinct from best/combo_best above, which are independent
    single-match suggestions rather than one combined bet.

    Pool: the same priced singles + combo pools used elsewhere, merged and
    reduced to one leg per match (whichever of that match's single or
    combo picks has the higher EV), ranked by EV. Legs are added greedily
    from that ranking: never fewer than ACC_MIN_LEGS, never more than
    ACC_MAX_LEGS, and stop as soon as both the leg-count floor is met AND
    the combined (multiplied) odds clears TARGET_AGG_ODD -- whichever
    happens later. If there aren't enough distinct matches, or the
    combined odds never reaches the target even using all available legs,
    no accumulator is offered for the day rather than forcing a weak one.

    Returns (legs_df, combined_odd) on success, or (empty df, None) if no
    suggestion clears the bar today.
    """
    singles = _ranked_singles_pool(df)
    combos = _ranked_combo_pool(df)

    pool = pd.DataFrame()
    if not singles.empty:
        s = singles.copy()
        s['_odd'] = s['MarketOdd']
        s['_ev'] = s['EV']
        s['_is_combo'] = False
        pool = pd.concat([pool, s])
    if not combos.empty:
        c = combos.copy()
        c['_odd'] = c['BaselineOdd']
        c['_ev'] = c['ComboEV']
        c['_is_combo'] = True
        pool = pd.concat([pool, c])

    if pool.empty:
        logger.log('info', 'No priced picks available to build a suggested-bets accumulator.')
        return pd.DataFrame(), None

    # One leg per match: if both a single and a combo qualified for the
    # same fixture, keep whichever has the higher EV so the accumulator
    # never carries two legs on one match.
    pool = pool[pool['_odd'].notna() & (pool['_odd'] > 1)]
    pool = pool.sort_values('_ev', ascending=False).drop_duplicates(subset='Match', keep='first')
    pool = pool.sort_values('_ev', ascending=False)

    legs = []
    agg_odd = 1.0
    for _, row in pool.iterrows():
        if len(legs) >= ACC_MAX_LEGS:
            break
        legs.append(row)
        agg_odd *= row['_odd']
        if len(legs) >= ACC_MIN_LEGS and agg_odd >= TARGET_AGG_ODD:
            break

    if len(legs) < ACC_MIN_LEGS:
        logger.log('info', f'Only {len(legs)} distinct-match leg(s) available -- below the {ACC_MIN_LEGS}-leg minimum for a suggested-bets slip today.')
        return pd.DataFrame(), None

    if agg_odd < TARGET_AGG_ODD:
        logger.log('info', f'Best available {len(legs)}-leg combination only reaches {agg_odd:.2f}x -- below the {TARGET_AGG_ODD}x target, skipping suggested bets today.')
        return pd.DataFrame(), None

    legs_df = pd.DataFrame(legs)
    legs_df['PredictionLabel'] = legs_df['Prediction'].apply(prediction_label)
    return legs_df, agg_odd


def format_suggested_bets(legs: pd.DataFrame, agg_odd, date_str: str) -> str:
    if legs.empty or agg_odd is None:
        return (
            f"⚠️ <b>No Suggested Bets slip for {date_str}.</b>\n"
            f"Not enough distinct-match legs cleared the value bar to reach a "
            f"{TARGET_AGG_ODD:.1f}x combined odd today -- sitting this one out is the right call."
        )

    msg = (
        f"🎰 <b>Suggested Bets — {date_str}</b>\n"
        f"<i>{len(legs)}-leg accumulator, combined odds ~{agg_odd:.2f}x</i>\n\n"
    )
    for i, (_, row) in enumerate(legs.iterrows(), start=1):
        odd_label = "baseline" if row['_is_combo'] else "odd"
        msg += (
            f"{i}. ⚽ <b>{row['Match']}</b> ({row['Division']})\n"
            f"   → {row['PredictionLabel']} @ {row['_odd']:.2f} ({odd_label})\n"
        )
    msg += (
        f"\n📌 Combined odds multiply individual risk -- ALL legs must win for the "
        f"slip to pay out. Bet-builder legs use an estimated baseline, not a "
        f"bookmaker-quoted price (see Bet Builder Picks for detail). This is one "
        f"illustrative combination, not a guarantee any single leg lands."
    )
    return msg


def format_telegram(best: pd.DataFrame, combo_best: pd.DataFrame, date_str: str) -> str:
    if best.empty and (combo_best is None or combo_best.empty):
        return f"⚠️ <b>No qualifying value bets found for {date_str}.</b>\nNo picks met the minimum edge threshold today — sitting this one out is the right call."

    msg = ""
    if not best.empty:
        msg += f"🎯 <b>Best Bets — {date_str}</b>\n<i>Ranked by model edge vs bookmaker odds, not just confidence</i>\n\n"
        for _, row in best.iterrows():
            msg += (
                f"⚽ <b>{row['Match']}</b> ({row['Division']})\n"
                f"   → {row['PredictionLabel']} @ {row['MarketOdd']:.2f}\n"
                f"   Model: {row['ModelProb']*100:.0f}% | Market implies: {row['ImpliedProb']*100:.0f}% "
                f"| Edge: +{row['Edge']*100:.1f}pp | EV: {row['EV']:+.2f} per unit\n\n"
            )
        msg += "📌 Edge = how much more likely our model thinks this is vs. what the odds imply. Higher edge and EV means better long-run value — not a guarantee on any single bet.\n\n"

    if combo_best is not None and not combo_best.empty:
        msg += f"🧩 <b>Bet Builder Picks — {date_str}</b>\n<i>Same-match combos, priced with the model's exact joint probability (not a naive multiply)</i>\n\n"
        for _, row in combo_best.iterrows():
            msg += (
                f"⚽ <b>{row['Match']}</b> ({row['Division']})\n"
                f"   → {row['PredictionLabel']} (~{row['BaselineOdd']:.2f} baseline)\n"
                f"   Model: {row['ModelProb']*100:.0f}% | Naive-independence baseline: {row['BaselineProb']*100:.0f}% "
                f"| Edge: +{row['ComboEdge']*100:.1f}pp\n\n"
            )
        msg += "📌 Bet builder odds/baselines above are estimated from single-market odds (no bookmaker publishes a price for the exact combo) — treat as a reference point, not a guaranteed price."

    return msg.strip()


def _write_txt(prefix, date_str, content):
    if not os.path.exists(PUBLISHPATH):
        os.makedirs(PUBLISHPATH)
    with open(f"{PUBLISHPATH}/{prefix}_{date_str}.txt", "w", encoding="utf-8") as f:
        f.write(content)


def _failure_message(prefix, date_str, reason):
    label = "Best Bets" if prefix == "BestBets" else "Suggested Bets"
    return (
        f"⚠️ <b>{label} could not be generated for {date_str}.</b>\n"
        f"Reason: {reason}\n"
        f"This means the script hit an error, not that no value was found "
        f"today -- check the run logs for the full traceback."
    )


def _write_failure_files(reason, date_strs=None):
    """Write an explanation file for both BestBets and SuggestedBets when
    something goes wrong badly enough that we never even got to select
    picks -- e.g. the merged predictions file is missing, the odds fetch
    fails outright, or the data has no usable Date/Time to bucket by.
    Without this, a failure here (even with continue-on-error on the
    workflow step) produced NO file at all, silently, which looks
    identical to 'nothing qualified today' from the outside."""
    for date_str in (date_strs or [datetime.date.today().strftime('%Y-%m-%d')]):
        for prefix in ("BestBets", "SuggestedBets"):
            _write_txt(prefix, date_str, _failure_message(prefix, date_str, reason))


def main():
    try:
        _main_impl()
    except Exception as e:
        logger.log('error', 'best_bets_selector.py failed before producing output -- writing failure explanation files.', info=str(e))
        _write_failure_files(str(e), date_strs=date_utils.relevant_date_strs())
        raise  # still surface the failure in the job's exit code/logs


def _main_impl():
    filename = newest_predictions()
    logger.log('info', 'Loading merged predictions..', info=filename)
    df = pd.read_csv(filename)

    odds = fetch_market_odds()
    df = team_utils.fuzzy_merge(df, odds, left_on=('HomeTeam', 'AwayTeam'), right_on=('HomeTeam', 'AwayTeam'))

    df = attach_edges(df)
    df = attach_combo_edges(df)

    # Bucket by the same AdjustedDate rule predictions_tier.py uses (see
    # date_utils.py) instead of stamping every output with date.today() --
    # the pipeline's fetch scripts now pull a window (today from 08:00
    # onward, plus tomorrow up to 08:00) that matches this exact labeling
    # rule, so a run's merged data should only ever resolve to ONE
    # AdjustedDate bucket. Still grouped rather than assumed single as a
    # defensive safety net (e.g. a source that hasn't adopted the fetch
    # window, or a manual/backfilled data file spanning more than one
    # day) -- date.today() would have been one full day behind the actual
    # match date before this fix, since the pipeline predicts fixtures
    # ahead of time.
    df['AdjustedDate'] = date_utils.adjusted_date_series(df['Date'], df['Time'])

    if not os.path.exists(PUBLISHPATH):
        os.makedirs(PUBLISHPATH)

    if df.empty:
        # The merged predictions file itself had no rows -- distinct from
        # "ran fine, nothing cleared the value bar", which format_telegram/
        # format_suggested_bets already handle with their own message.
        logger.log('warning', 'Merged predictions file has no rows -- nothing to select from.')
        _write_failure_files("No predictions available to select from (merged file was empty).",
                              date_strs=date_utils.relevant_date_strs())
        return

    for match_date, df_date in df.groupby('AdjustedDate'):
        date_str = pd.to_datetime(match_date).strftime('%Y-%m-%d')
        df_date = df_date.drop(columns=['AdjustedDate'])
        tg_text = None

        try:
            best = select_best_bets(df_date)
            combo_best = select_best_combo_bets(df_date)
            tg_text = format_telegram(best, combo_best, date_str)
            _write_txt("BestBets", date_str, tg_text)

            if not best.empty:
                logger.log('info', f'Selected {len(best)} best bets for {date_str}.', info=str(best["Match"].tolist()))
            else:
                logger.log('warning', f'No best bets selected for {date_str}.')

            if not combo_best.empty:
                logger.log('info', f'Selected {len(combo_best)} bet-builder combos for {date_str}.', info=str(combo_best["Match"].tolist()))
            else:
                logger.log('info', f'No bet-builder combos selected for {date_str}.')
        except Exception as e:
            logger.log('error', f'Best Bets generation failed for {date_str} -- writing failure explanation.', info=str(e))
            _write_txt("BestBets", date_str, _failure_message("BestBets", date_str, str(e)))

        # Suggested Bets: one combined 4-6 leg accumulator (singles
        # and/or combos, one leg per match) targeting a combined odd of
        # at least TARGET_AGG_ODD -- distinct from the independent
        # per-match suggestions above. Wrapped separately from the Best
        # Bets block so a failure in one doesn't prevent the other's file
        # from being written for this date.
        try:
            legs, agg_odd = build_accumulator(df_date)
            suggested_text = format_suggested_bets(legs, agg_odd, date_str)
            _write_txt("SuggestedBets", date_str, suggested_text)

            if not legs.empty:
                logger.log('info', f'Built a {len(legs)}-leg suggested bets slip at {agg_odd:.2f}x for {date_str}.', info=str(legs["Match"].tolist()))
            else:
                logger.log('info', f'No suggested-bets accumulator built for {date_str}.')
        except Exception as e:
            logger.log('error', f'Suggested Bets generation failed for {date_str} -- writing failure explanation.', info=str(e))
            _write_txt("SuggestedBets", date_str, _failure_message("SuggestedBets", date_str, str(e)))

        print(tg_text if tg_text is not None else f"(Best Bets for {date_str} failed -- see logs)")


if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    datesave = datetime.date.today().strftime('%Y%m%d')
    LOGNAME = LOGNAME.replace('{date}', datesave) + '.json'
    logger = JSONLogger(log_file=LOGNAME, log_dir=LOGPATH)

    try:
        main()
    except Exception as e:
        logger.log('critical', "Exception occurred while running best_bets_selector", info=str(e))
        raise
