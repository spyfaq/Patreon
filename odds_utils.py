#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
odds_utils.py

Shared odds-math helpers used by predictions_merger.py's odd_addition()
and best_bets_selector.py, so both derive Over 1.5 / Over 3.5 odds the
same way instead of each carrying its own (potentially drifting) copy.

football-data.co.uk only publishes Over/Under 2.5 goals odds -- Over 1.5
and Over 3.5 have no real market price at all in this data. These are
approximated from the real Over 2.5 price with a fixed offset: Over 1.5
is a safer, shorter-priced bet than Over 2.5 (more likely to land), so
its odd should be lower; Over 3.5 is a riskier, longer-priced bet, so its
odd should be higher.
"""

import numpy as np

OVER_1_5_OFFSET = -0.3
OVER_3_5_OFFSET = 0.6

MIN_PAYABLE_ODD = 1.01  # guards against a degenerate/negative derived odd

# Prediction code -> the odds column that prices it. Markets deliberately
# absent from this map (GG, hO1_5, hO2_5, aO1_5, aO2_5) have no published
# price in any source this project uses, so their rows carry no odd at all
# rather than an invented one.
MARKET_ODD_COLUMNS = {
    '1': 'AvgH',
    'X': 'AvgD',
    '2': 'AvgA',
    'O1_5': 'AvgOver15',
    'O2_5': 'AvgOver25',
    'O3_5': 'AvgOver35',
}

# Every odds column a fixture row needs to carry so that any prediction
# made for it can be priced.
ODD_COLUMNS = ['AvgH', 'AvgD', 'AvgA', 'AvgOver25', 'AvgOver15', 'AvgOver35']


def derive_over_under_odds(over25):
    """Given a real Over 2.5 odd (scalar or pandas Series), return
    (over1_5, over3_5) estimated odds: Over2.5 - 0.3 and Over2.5 + 0.6.
    NaN input propagates to NaN output (nothing to derive from). These
    are estimates, not real bookmaker prices -- there's no published
    Under 1.5/Under 3.5 counterpart to de-vig against, unlike the genuine
    Over/Under 2.5 market."""
    over1_5 = over25 + OVER_1_5_OFFSET
    over3_5 = over25 + OVER_3_5_OFFSET

    # Guard against a degenerate/negative odd if Over 2.5 itself is
    # already very short (e.g. a huge favorite's fixture priced near
    # 1.0) -- works for both a scalar and a pandas Series input.
    if hasattr(over1_5, 'where'):
        over1_5 = over1_5.where(over1_5 >= MIN_PAYABLE_ODD, np.nan)
    elif over1_5 is not None and over1_5 < MIN_PAYABLE_ODD:
        over1_5 = np.nan

    return over1_5, over3_5


def odd_for_prediction(pred, odds):
    """Decimal odd for a single-market prediction code, read out of a
    mapping of Avg* odds columns (a dict or a pandas Series row).

    Returns None -- not a placeholder number -- when the market has no
    published price at all (GG and the home/away-specific overs) or when
    the source simply didn't carry one for this fixture. The prediction
    row is still written either way; it just goes out without an odd,
    which is the honest representation.
    """
    col = MARKET_ODD_COLUMNS.get(pred)
    if col is None or odds is None or not hasattr(odds, 'get'):
        return None

    val = odds.get(col)
    try:
        val = float(val)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(val) or val < MIN_PAYABLE_ODD:
        return None
    return round(val, 2)
