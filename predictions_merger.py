#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predictions_merger.py

Combines today's three prediction files (major / minor / international)
into one, resolves contradictory picks, builds priced combo bets, drops
short-priced singles, and writes merged-prediction-<date>.csv.

Reads today's files by name -- majorleague/minorleague/international each
write exactly one file per betting day (see date_utils for what "day"
means: 09:00 through 08:59 the next morning). There is no "find the
newest matching file" search and no date-range reconciliation between
sources any more: all three are named for the same day, so the merger
either finds today's file or that source didn't run.

Odds arrive WITH the predictions. Each prediction row already carries the
bookmaker price for its own market, so this script no longer re-downloads
the football-data.co.uk fixtures feed or calls The Odds API -- that work
now happens once, upstream, in the scripts that already had the fixture
data in hand.
"""

import os
import datetime
import pandas as pd

import team_utils
import market_odds

DATAPATH = 'predictions_data/'

# Exactly one file per source per betting day.
SOURCE_TEMPLATES = {
    'major': 'major_prediction_{date}.csv',
    'minor': 'minor_prediction_{date}.csv',
    'international': 'international_prediction_{date}.csv',
}

OUTPUT_TEMPLATE = 'merged-prediction-{date}.csv'

# --------------------------------------------------------------- rules

# Within one match exactly one of these can happen, so a match must not
# end up with more than one of them in the output.
#
# The goal markets are deliberately NOT here. Over 1.5 / 2.5 / 3.5 are
# nested rather than exclusive (if Over 3.5 lands, so did Over 2.5), as
# are the team-goal markets, and GG is compatible with all of them. Only
# the match-result market is genuinely one-of.
MUTUALLY_EXCLUSIVE = ('1', 'X', '2')

# Combo legs: a match-result side paired with a goal line. Restricted to
# markets that carry a real or derived price, because a combo's whole
# admission test is its combined odd -- an unpriced leg (GG, team-goal
# markets) has no odd to combine.
COMBO_SIDES = ('1', 'X', '2')
COMBO_GOAL_LEGS = ('O1_5', 'O2_5', 'O3_5')

# Combined odd = (leg1 x leg2) less a 5% haircut. Multiplying two prices
# assumes the bookmaker would offer the fair product, which no bookmaker
# does; the haircut stands in for the margin taken on a same-match combo.
COMBO_MARGIN = 0.05

MIN_COMBO_ODD = 1.72   # a combo below this isn't worth the added risk
MIN_SINGLE_ODD = 1.32  # too short to pay for its own variance

OUTPUT_COLUMNS = ["Division", "Date", "Time", "HomeTeam", "AwayTeam",
                  "Prediction", "Prediction %", "Odd", "Implied %", "Edge %",
                  "History %", "HomeTeam Stats", "AwayTeam Stats",
                  "HomeForm", "AwayForm"]

# Identifies one fixture across the three sources.
MATCH_KEY = ['MatchDate', 'HomeTeam', 'AwayTeam']


def today_str(reference=None):
    return (reference or datetime.date.today()).strftime('%Y-%m-%d')


def load_sources(day):
    """Read today's file from each source. A missing file is normal, not an
    error: international only produces one during a tournament window, and
    a domestic script exits early on a day with no fixtures."""
    frames = []
    for name, template in SOURCE_TEMPLATES.items():
        path = os.path.join(DATAPATH, template.format(date=day))
        if not os.path.exists(path):
            print(f'No {name} file for {day} -- skipping.')
            continue
        df = pd.read_csv(path)
        if df.empty:
            print(f'{name} file for {day} is empty -- skipping.')
            continue
        print(f'Loaded {len(df)} {name} predictions.')
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def normalize_dates(df):
    """Add a parsed MatchDate used for grouping and sorting.

    The sources don't agree on Date formatting -- major writes
    '2026-07-28, Tuesday' and minor writes '28-07-2026, Tuesday' -- so the
    weekday suffix is stripped and both layouts are tried. Grouping on the
    raw strings would treat the same fixture from two sources as two
    different matches.
    """
    core = df['Date'].astype(str).str.split(',', n=1).str[0].str.strip()
    parsed = pd.to_datetime(core, format='%Y-%m-%d', errors='coerce')
    parsed = parsed.fillna(pd.to_datetime(core, format='%d-%m-%Y', errors='coerce'))
    parsed = parsed.fillna(pd.to_datetime(core, errors='coerce'))
    df['MatchDate'] = parsed
    return df


def drop_mutually_exclusive(df):
    """Keep at most one of 1/X/2 per match.

    Each market is gated against its own base rate upstream, so a match can
    clear the floor for more than one result -- e.g. a home win at 47% and
    an away win at 31% both qualify, leaving the file recommending both
    sides of the same game.

    The survivor is the highest model probability: that is the pick the
    model actually makes. Edge is deliberately NOT the tie-breaker -- it
    would let a 25%-probability outsider displace a 50% favourite purely
    because it was generously priced, which is a bet-selection judgement
    rather than a contradiction to resolve. Downstream selection can still
    rank on edge; this only removes the self-contradiction.
    """
    excl = df[df['Prediction'].isin(MUTUALLY_EXCLUSIVE)]
    if excl.empty:
        return df

    # Highest probability first, so the first row of each group wins.
    ordered = excl.sort_values('Prediction %', ascending=False, kind='mergesort')
    keep_idx = set(ordered.groupby(MATCH_KEY, dropna=False).head(1).index)

    dropped = len(excl) - len(keep_idx)
    result = df[~df.index.isin(set(excl.index) - keep_idx)]
    if dropped:
        print(f'Removed {dropped} contradictory result picks (kept one of 1/X/2 per match).')
    return result


def build_combos(df):
    """Same-match side + goal-line combos, kept only when the combined odd
    clears MIN_COMBO_ODD.

    Probability caveat: the combined probability here is the PRODUCT of the
    two legs, i.e. it assumes they are independent. They are not -- a home
    win and a high-scoring game are correlated, so the product is biased
    (generally low for 1+Over, high for X+Over). The exact joint
    probability can only be read off the Dixon-Coles score grid, which
    exists in the prediction scripts but is not carried in their output.
    Treat the combo's 'Prediction %' as indicative; its odd is exact.
    """
    priced = df[df['Odd'].notna()]
    if priced.empty:
        print('No priced rows -- no combos built.')
        return pd.DataFrame(columns=df.columns)

    combos = []
    for _, match in priced.groupby(MATCH_KEY, dropna=False):
        sides = match[match['Prediction'].isin(COMBO_SIDES)]
        goals = match[match['Prediction'].isin(COMBO_GOAL_LEGS)]
        if sides.empty or goals.empty:
            continue

        for _, side in sides.iterrows():
            for _, goal in goals.iterrows():
                combined = float(side['Odd']) * float(goal['Odd']) * (1 - COMBO_MARGIN)
                combined = round(combined, 2)
                if combined <= MIN_COMBO_ODD:
                    continue

                prob = float(side['Prediction %']) * float(goal['Prediction %'])
                row = side.copy()
                row['Prediction'] = f"{side['Prediction']}+{goal['Prediction']}"
                row['Prediction %'] = round(prob, 4)
                row['Odd'] = combined
                row['Implied %'] = market_odds.implied_prob(combined)
                row['Edge %'] = market_odds.edge(prob, combined)
                # No head-to-head record exists for a combined outcome.
                row['History %'] = '-'
                combos.append(row)

    if not combos:
        print(f'No combos cleared the {MIN_COMBO_ODD} combined-odd floor.')
        return pd.DataFrame(columns=df.columns)

    print(f'Built {len(combos)} combos above {MIN_COMBO_ODD}.')
    return pd.DataFrame(combos).reset_index(drop=True)


def drop_short_prices(df):
    """Drop rows priced below MIN_SINGLE_ODD.

    Blank odds are KEPT. An empty price means the market has no published
    odd anywhere (GG, the team-goal markets) -- that is missing
    information, not a short price, and dropping those rows would silently
    delete every prediction for markets we simply can't price.
    """
    odd = pd.to_numeric(df['Odd'], errors='coerce')
    too_short = odd.notna() & (odd < MIN_SINGLE_ODD)
    if too_short.any():
        print(f'Removed {int(too_short.sum())} predictions priced under {MIN_SINGLE_ODD}.')
    return df[~too_short]


def save_merged(df, day):
    if not os.path.exists(DATAPATH):
        os.makedirs(DATAPATH)

    df = df.sort_values(['MatchDate', 'Time', 'HomeTeam', 'Prediction'], kind='mergesort')

    # Restate Date in ONE format. The sources disagree (major writes
    # '2026-07-28, Tuesday', minor '28-07-2026, Tuesday'), and a merged
    # file carrying both is ambiguous to read and to re-parse -- '05-06'
    # means two different days depending on which source a row came from.
    # Sorting already used the parsed MatchDate, so this only affects how
    # the column reads.
    formatted = df['MatchDate'].dt.strftime('%Y-%m-%d, %A')
    df['Date'] = formatted.fillna(df['Date'])

    df = df[OUTPUT_COLUMNS]

    filename = os.path.join(DATAPATH, OUTPUT_TEMPLATE.format(date=day))
    df.to_csv(filename, index=False)
    print(f'Saved {len(df)} rows -> {filename}')
    return filename


def merging_func(reference=None):
    day = today_str(reference)
    print(f'Merging predictions for {day}..')

    df = load_sources(day)
    if df.empty:
        print('Nothing to merge -- no source produced predictions today.')
        return

    # Canonical team names, so the same fixture from two sources groups as
    # one match here and settles correctly in update_results.py later.
    df['HomeTeam'] = df['HomeTeam'].apply(team_utils.display_name)
    df['AwayTeam'] = df['AwayTeam'].apply(team_utils.display_name)

    df = normalize_dates(df)
    df = df.reset_index(drop=True)

    df = drop_mutually_exclusive(df)
    combos = build_combos(df)
    if not combos.empty:
        df = pd.concat([df, combos], ignore_index=True)

    df = drop_short_prices(df)
    if df.empty:
        print('Every prediction was filtered out -- nothing to save.')
        return

    save_merged(df, day)
    print('Process completed..')


if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))

    try:
        merging_func()
    except Exception as e:
        print('CRITICAL: Exception occured whie running', e)
        raise
