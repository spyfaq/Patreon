#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
market_odds.py

Maps a prediction's market code (the '1'/'X'/'O2_5'/... codes resultdef()
emits) onto the bookmaker odd for that market, so each exported prediction
row can carry the price it would actually be bet at.

WHAT IS AND ISN'T PRICED
------------------------
The two odds sources between them publish only a narrow set of markets:

  football-data.co.uk fixtures  ->  AvgH / AvgD / AvgA  (1X2)
                                    Avg>2.5 / Avg<2.5   (Over/Under 2.5)
  The Odds API (odds_client)    ->  the same, for CL/WC/EC

From those, Over 1.5 and Over 3.5 are ESTIMATED off the real Over 2.5
price by a fixed offset (odds_utils.derive_over_under_odds) -- they are
not quoted anywhere in the source data.

Everything else the model predicts -- GG (both teams to score) and the
team-specific goal markets (hO1_5, hO2_5, aO1_5, aO2_5) -- has no
published price in either source. Those rows are exported with an EMPTY
odd rather than a guessed one: an invented price would flow straight into
any edge/value calculation downstream and quietly manufacture signal that
isn't there. Blank is the honest answer.

IMPLIED PROBABILITY
-------------------
implied_prob() is the naive 1/odd, NOT de-vigged. The bookmaker's margin
is baked into it, so it systematically overstates the true market
probability (a full 1X2 book sums to ~1.05, not 1.00). It is still the
right comparison point for a like-for-like "is the model above or below
the price" read, and de-vigging needs the complete opposing side of the
market, which we only have for Over/Under 2.5.
"""

import numpy as np
import pandas as pd

import odds_utils

# Market code -> the odds column carrying its price. AvgOver15/AvgOver35
# are derived (see add_derived_columns), the rest are as published.
MARKET_ODD_COLUMN = {
    '1': 'AvgH',
    'X': 'AvgD',
    '2': 'AvgA',
    'O1_5': 'AvgOver15',
    'O2_5': 'AvgOver25',
    'O3_5': 'AvgOver35',
}

# Predicted markets with no published price in either odds source.
UNPRICED_MARKETS = ('GG', 'hO1_5', 'hO2_5', 'aO1_5', 'aO2_5')

# Odds columns a fixture row is expected to be able to supply.
ODDS_COLUMNS = ('AvgH', 'AvgD', 'AvgA', 'AvgOver25', 'AvgUnder25',
                'AvgOver15', 'AvgOver35')


def add_derived_columns(df):
    """Add the derived AvgOver15/AvgOver35 columns to a fixture-odds frame,
    in place-ish (returns the same frame). Safe to call on a frame with no
    AvgOver25 column at all -- the derived columns are then simply NaN,
    which is what an absent Over/Under market should look like."""
    if 'AvgOver25' not in df.columns:
        df['AvgOver25'] = np.nan
    df['AvgOver15'], df['AvgOver35'] = odds_utils.derive_over_under_odds(df['AvgOver25'])
    return df


def odd_for(market, odds_row):
    """The bookmaker odd for `market` given a fixture's odds row (a mapping
    or pandas Series), or None when the market is unpriced, the row is
    missing, or the price is absent/unusable.

    None -- not 0, not a placeholder -- is deliberate: it lands in the CSV
    as an empty cell, so an unpriced market is visibly unpriced instead of
    looking like a real number."""
    if odds_row is None:
        return None
    column = MARKET_ODD_COLUMN.get(market)
    if column is None:
        return None
    try:
        value = odds_row[column]
    except (KeyError, IndexError, TypeError):
        return None
    return _clean_odd(value)


def _clean_odd(value):
    """A usable decimal odd, or None. Rejects NaN, non-numerics, and any
    price at or below 1.0 (an odd of 1.0 pays nothing back above stake, so
    it is a data error rather than a real market)."""
    if value is None:
        return None
    try:
        odd = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(odd) or odd <= 1.0:
        return None
    return round(odd, 2)


def implied_prob(odd):
    """Naive implied probability (1/odd), rounded, or None. Not de-vigged
    -- see the module docstring."""
    odd = _clean_odd(odd)
    if odd is None:
        return None
    return round(1.0 / odd, 4)


def edge(model_prob, odd):
    """model probability - implied probability, or None when unpriced.

    Positive means the model rates the outcome MORE likely than the price
    implies (the direction that makes a bet worth considering); negative
    means the market is already ahead of the model. Because implied_prob()
    keeps the bookmaker's margin in, a genuinely neutral pick scores
    slightly negative here -- treat 0 as "already worse than break-even",
    not as the boundary of value."""
    implied = implied_prob(odd)
    if implied is None or model_prob is None:
        return None
    try:
        return round(float(model_prob) - implied, 4)
    except (TypeError, ValueError):
        return None


def build_lookup(odds_df, normalize):
    """Index a fixture-odds frame by (normalized home, normalized away) for
    O(1) per-fixture lookup, instead of re-scanning or re-merging the frame
    once per match.

    `normalize` is passed in (team_utils.normalize) rather than imported so
    this module stays free of the name-matching logic -- callers already
    depend on team_utils and can decide how strict they want matching to be.

    Later duplicates do not overwrite earlier ones: football-data.co.uk can
    list the same fixture twice across its main and new-league files, and
    the first occurrence is as good as any."""
    if odds_df is None or len(odds_df) == 0:
        return {}

    odds_df = add_derived_columns(odds_df.copy())
    lookup = {}
    for row in odds_df.to_dict('records'):
        key = (normalize(row.get('HomeTeam')), normalize(row.get('AwayTeam')))
        if key[0] and key[1] and key not in lookup:
            lookup[key] = row
    return lookup


def lookup_odds(lookup, home, away, normalize, fuzzy_cutoff=0.82):
    """Odds row for one fixture, exact-matching on normalized names first
    and falling back to a fuzzy match.

    The fallback matters because the odds feed and the fixture feed are
    different sources that disagree on team naming often enough to matter
    ('Nott'm Forest' vs 'Nottingham Forest'). Without it a real price is
    silently dropped and the row exports as unpriced, which is
    indistinguishable from a market that genuinely has no price."""
    if not lookup:
        return None

    key = (normalize(home), normalize(away))
    if key in lookup:
        return lookup[key]

    from difflib import SequenceMatcher

    best_row, best_score = None, fuzzy_cutoff
    for (lh, la), row in lookup.items():
        # Score both sides and take the weaker one, so a fixture only
        # matches when BOTH teams are recognisable -- matching on a strong
        # home-name hit alone would happily pair the wrong away side.
        score = min(SequenceMatcher(None, key[0], lh).ratio(),
                    SequenceMatcher(None, key[1], la).ratio())
        if score > best_score:
            best_row, best_score = row, score

    return best_row


def empty_odds_frame():
    """An empty frame with the full odds column set, for the failure paths
    where a fetch could not produce anything but downstream still expects
    the columns to exist."""
    return pd.DataFrame(columns=['Date', 'Time', 'Div', 'HomeTeam', 'AwayTeam']
                                + list(ODDS_COLUMNS))
