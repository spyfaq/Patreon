#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
date_utils.py

Shared "which calendar day's batch does this match belong to" logic.

A match's Date field is its actual UTC calendar date, but very early
kickoffs (e.g. MLS games around 00:00-02:00 UTC, which are evening
kickoffs local US time) are, for scheduling/publishing purposes, really
part of the PREVIOUS day's slate -- originally implemented ad-hoc in
predictions_tier.py only (commit 65130b7, "Shift early-morning matches
(before 06:00) to previous day"). Pulled out here so every script that
needs the same "which day's batch" answer -- predictions_tier.py,
best_bets_selector.py, post_from_dropbox.py -- uses the identical rule
instead of each one potentially drifting out of sync.

Separately: the pipeline runs once daily and predicts TOMORROW's
fixtures. Combined with the shift above, a SINGLE run can produce output
dated either "today" (from early-morning matches that got shifted back)
or "tomorrow" (from everything else), or both at once. Anything that
later searches for "today's" files (e.g. post_from_dropbox.py) needs to
check both dates, or it will silently miss whichever bucket it didn't
check -- see relevant_date_strs().
"""

import datetime
import pandas as pd

# Matches kicking off before this UTC hour are counted as belonging to
# the PREVIOUS calendar day's batch.
EARLY_MORNING_CUTOFF_HOUR = 8


def adjusted_date_series(date_series, time_series, cutoff_hour=EARLY_MORNING_CUTOFF_HOUR):
    """Vectorized: given a Date column (any format pandas/dayfirst can
    parse, e.g. '26-07-2026') and a Time column (string 'HH:MM:SS' or
    already a datetime.time), return a Series of adjusted dates -- one
    per row, shifted back a day where the kickoff hour is before
    cutoff_hour. Missing/unparseable times default to 00:00 (so they DO
    get shifted back), matching the previous inline behavior in
    predictions_tier.py."""
    time_parsed = pd.to_datetime(time_series.astype(str), errors='coerce').dt.time
    time_parsed = time_parsed.fillna(datetime.time(0, 0))

    combined = pd.to_datetime(
        date_series.astype(str) + ' ' + time_parsed.astype(str),
        dayfirst=True, errors='coerce'
    )

    # Fallback for any row where even the date alone won't parse -- keep
    # the run going rather than raising, since a single bad row shouldn't
    # take down the whole batch.
    fallback = pd.to_datetime(date_series, dayfirst=True, errors='coerce').dt.date

    def _shift(dt, fb):
        if pd.isna(dt):
            return fb
        return (dt - datetime.timedelta(days=1)).date() if dt.hour < cutoff_hour else dt.date()

    return pd.Series(
        [_shift(dt, fb) for dt, fb in zip(combined, fallback)],
        index=date_series.index
    )


def relevant_date_strs(reference=None):
    """The set of '%Y-%m-%d' date strings a single pipeline run's output
    can legitimately be dated under, from the perspective of whoever is
    looking for today's files to post. Since the pipeline predicts
    TOMORROW's fixtures but shifts early-morning matches back to what is,
    from that run's perspective, "today", a run can produce files dated
    either today or tomorrow (or both) -- checking only one, as
    post_from_dropbox.py used to, silently misses the other."""
    reference = reference or datetime.date.today()
    return {
        reference.strftime('%Y-%m-%d'),
        (reference + datetime.timedelta(days=1)).strftime('%Y-%m-%d'),
    }
