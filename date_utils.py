#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
date_utils.py

Shared "which day's batch does this match belong to" logic, used both to
decide what to FETCH for today's run and to LABEL matches once fetched.

The betting day runs from 09:00 to 09:00, not midnight to midnight: a
match kicking off at 02:00 (e.g. a late US kickoff, evening local time)
is still part of the previous day's card, not the next one. Originally
implemented ad-hoc in predictions_tier.py only, as a post-fetch label
(commit 65130b7, "Shift early-morning matches ... to previous day", cutoff
initially 06:00 in the comment though the code actually used 08:00).
The cutoff has since moved to 09:00 -- see EARLY_MORNING_CUTOFF_HOUR.

That label-only fix was incomplete: every fetch script (majorleague/
minorleague/international_predictions.py, check_fixtures.py) pulled a
full calendar day ("tomorrow", midnight-to-midnight) regardless of this
cutoff, so a run on day X only ever fetched day X+1's fixtures -- day X's
own daytime/evening matches were never fetched by that run AT ALL, and
day X+1's matches after 08:00 (the bulk of a normal day) stayed labeled
day X+1 instead of day X. The result: a run on day X produced output
mostly dated day X+1, not day X.

Both sides now use the same 09:00 cutoff and live here:
  - fetch_window() / in_fetch_window(): what a script should PULL for
    today's run -- day X from 09:00 onward, plus day X+1 up to 09:00.
  - adjusted_date_series(): which day's batch an ALREADY-FETCHED row
    belongs to -- used for grouping/labeling/filenames.
Every script that needs either answer (predictions_tier.py,
best_bets_selector.py, post_from_dropbox.py, and now the fetch scripts
too) uses these instead of each one potentially drifting out of sync.
"""

import datetime
import pandas as pd

# The betting day starts at this UTC hour. Matches before it belong to
# the PREVIOUS calendar day's batch; matches at or after it belong to
# their own calendar day.
#
# 09:00 is chosen to sit just ahead of the ~11:00 pipeline run: everything
# from 09:00 today through 08:59 tomorrow is "today's card", so a late
# 02:00 kickoff is captured by today's run while last night's 22:00 game
# has already been settled by yesterday's.
EARLY_MORNING_CUTOFF_HOUR = 9


def _combine_date_time(date_series, time_series):
    """Parse a Date column (raw string like '26-07-2026', OR an
    already-parsed Timestamp/datetime column) and a Time column (string
    'HH:MM:SS' or already a datetime.time) into one combined datetime
    Series. Missing/unparseable times default to 00:00. Date is
    normalized to just its date portion first regardless of which form
    it arrives in -- an already-parsed Timestamp carries its own (usually
    00:00:00) time component, which would silently clash with the
    separate Time column if string-concatenated as-is."""
    date_parsed = pd.to_datetime(date_series, dayfirst=True, errors='coerce').dt.strftime('%Y-%m-%d')

    time_parsed = pd.to_datetime(time_series.astype(str), errors='coerce').dt.time
    time_parsed = time_parsed.fillna(datetime.time(0, 0))

    return pd.to_datetime(date_parsed + ' ' + time_parsed.astype(str), errors='coerce')


def fetch_window(reference=None, cutoff_hour=EARLY_MORNING_CUTOFF_HOUR):
    """The [start, end) datetime window whose matches belong to a run on
    `reference`'s day (default: today): from cutoff_hour on that day to
    cutoff_hour the next day. A match at 03:00 the next calendar day
    falls inside this window (early enough that it's still part of
    today's card); a match at 15:00 the next day does not -- that's the
    following day's run's job to fetch."""
    reference = reference or datetime.date.today()
    start = pd.Timestamp(datetime.datetime.combine(reference, datetime.time(cutoff_hour, 0)))
    end = start + pd.Timedelta(days=1)
    return start, end


def in_fetch_window(date_series, time_series, reference=None, cutoff_hour=EARLY_MORNING_CUTOFF_HOUR):
    """Boolean mask over a Date/Time column pair: which rows fall inside
    today's fetch window (see fetch_window()). Use this to FILTER which
    fixtures a script pulls for today's run -- e.g.
    `next_match[date_utils.in_fetch_window(next_match['Date'], next_match['Time'])]`
    in place of the old `Date == tomorrow` exact-day filter."""
    combined = _combine_date_time(date_series, time_series)
    start, end = fetch_window(reference, cutoff_hour)
    return (combined >= start) & (combined < end)


def adjusted_date_series(date_series, time_series, cutoff_hour=EARLY_MORNING_CUTOFF_HOUR):
    """Vectorized: given a Date/Time column pair, return a Series of
    adjusted dates -- one per row, shifted back a day where the kickoff
    hour is before cutoff_hour. Use this to LABEL/GROUP rows that have
    already been fetched (e.g. for filenames, for building
    per-day batches)."""
    combined = _combine_date_time(date_series, time_series)

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
    """The set of '%Y-%m-%d' date strings a run's output can be dated
    under, from the perspective of whoever is looking for today's files
    to post. With fetch_window()/adjusted_date_series() now using the
    same cutoff on both the fetch and label side, a run on day X should
    only ever produce files dated day X. Both today's and tomorrow's date
    are still checked here as a defensive safety net (e.g. leftover files
    from before this fix, or a manual run with mismatched fetch/label
    logic) -- cheap to check, and silently missing a file is worse than
    one harmless extra lookup that finds nothing."""
    reference = reference or datetime.date.today()
    return {
        reference.strftime('%Y-%m-%d'),
        (reference + datetime.timedelta(days=1)).strftime('%Y-%m-%d'),
    }
