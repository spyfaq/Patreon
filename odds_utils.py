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
