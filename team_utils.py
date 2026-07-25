#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
team_utils.py

Team names are spelled differently across every source this project
touches: football-data.co.uk uses short names ("Man United", "Inter"),
football-data.org uses formal registered names ("Manchester United FC",
"FC Internazionale Milano"), and odds providers use yet another
convention ("Manchester United", "Inter Milan"). Matching rows across
these sources (predictions <-> odds, predictions <-> results) on a raw
string-equality merge silently drops anything that isn't spelled
identically, and posting a mix of naming conventions in the same
Telegram/Excel output looks inconsistent to readers.

This module gives every other script a single place to:
  1. normalize(name)   -- canonical lowercase key for matching
  2. display_name(name) -- cleaned-up name for posting to users
  3. best_match(name, candidates) -- fuzzy-match one name into a list
  4. fuzzy_merge(...)  -- merge two DataFrames on (HomeTeam, AwayTeam)
     tolerating spelling/formatting differences between them
"""

import re
import unicodedata
from difflib import get_close_matches

import pandas as pd

# Known cross-source aliases. Exact-match fixes are cheap and more
# reliable than fuzzy matching for cases we already know about --- add to
# this as new mismatches are discovered in production logs (a team that
# silently gets no odds/settlement is the symptom to watch for).
MANUAL_ALIASES = {
    "man united": "manchester united",
    "man utd": "manchester united",
    "man city": "manchester city",
    "spurs": "tottenham",
    "tottenham hotspur": "tottenham",
    "wolves": "wolverhampton wanderers",
    "newcastle": "newcastle united",
    "leeds": "leeds united",
    "nottm forest": "nottingham forest",
    "sheffield utd": "sheffield united",
    "bayern munich": "bayern munchen",
    "bayern münchen": "bayern munchen",
    "fc bayern munchen": "bayern munchen",
    "borussia dortmund": "dortmund",
    "bvb": "dortmund",
    "inter milan": "inter",
    "internazionale": "inter",
    "fc internazionale milano": "inter",
    "ac milan": "milan",
    "real madrid cf": "real madrid",
    "club atletico de madrid": "atletico madrid",
    "atletico de madrid": "atletico madrid",
    "psg": "paris sg",
    "paris saint-germain": "paris sg",
    "paris saint germain": "paris sg",
    "paris saint-germain fc": "paris sg",
}

# Suffixes that are decoration, not identity -- stripped both for the
# matching key and for the cleaned-up display name.
SUFFIXES = (
    " fc", " cf", " afc", " sc", " cd", " ac", " sad", " if",
    " bk", " fk", " sk", " ca", " se",
)


def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )


def normalize(name) -> str:
    """Canonical matching key: lowercase, accent-stripped, punctuation-
    stripped, common-suffix-stripped, with known aliases folded together."""
    if not isinstance(name, str) or not name.strip():
        return ""
    s = _strip_accents(name).lower().strip()
    s = re.sub(r"[.']", "", s)
    s = re.sub(r"[-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    for suf in SUFFIXES:
        if s.endswith(suf):
            s = s[: -len(suf)].strip()
    return MANUAL_ALIASES.get(s, s)


def display_name(name) -> str:
    """Cleaned-up name for posting -- strips the same decorative suffixes
    as normalize() but preserves original casing/spelling otherwise, so
    'Real Madrid CF' -> 'Real Madrid' while 'Manchester United' (already
    short) is untouched. Keeps posted output consistent regardless of
    which source a given match's data came from."""
    if not isinstance(name, str) or not name.strip():
        return name
    s = name.strip()
    for suf in SUFFIXES:
        if s.lower().endswith(suf):
            s = s[: -len(suf)].strip()
            break
    return s


def best_match(name, candidates, cutoff=0.82):
    """Return whichever of `candidates` (original strings) best matches
    `name`, or None if nothing clears `cutoff`. Tries an exact normalized
    match first (cheap, unambiguous); difflib fuzzy matching on the
    normalized forms is the fallback for spelling variants not already
    covered by MANUAL_ALIASES."""
    if not candidates:
        return None
    target = normalize(name)
    if not target:
        return None

    norm_to_orig = {}
    for c in candidates:
        n = normalize(c)
        if n and n not in norm_to_orig:
            norm_to_orig[n] = c

    if target in norm_to_orig:
        return norm_to_orig[target]

    matches = get_close_matches(target, list(norm_to_orig.keys()), n=1, cutoff=cutoff)
    return norm_to_orig[matches[0]] if matches else None


def fuzzy_merge(left, right, left_on=("HomeTeam", "AwayTeam"),
                 right_on=("HomeTeam", "AwayTeam"), cutoff=0.82,
                 how="left"):
    """Merge `left` onto `right` matching on a pair of team-name columns
    (typically HomeTeam+AwayTeam), tolerating naming differences between
    the two sources instead of requiring exact string equality.

    Strategy: build normalized keys on both sides and merge on those
    first (covers the vast majority of rows cheaply and unambiguously);
    for any left rows that still don't match, fall back to fuzzy-matching
    each unmatched team name independently against the right side's team
    list and retry the merge with the resolved names. Original column
    values from `left` are preserved in the output (display names aren't
    silently overwritten by whichever source happened to match).
    """
    l_home, l_away = left_on
    r_home, r_away = right_on

    left = left.copy()
    right = right.copy()
    left["_nh"] = left[l_home].map(normalize)
    left["_na"] = left[l_away].map(normalize)
    right["_nh"] = right[r_home].map(normalize)
    right["_na"] = right[r_away].map(normalize)

    other_cols = [c for c in right.columns if c not in (r_home, r_away, "_nh", "_na")]

    merged = left.merge(right[["_nh", "_na"] + other_cols], on=["_nh", "_na"], how=how)

    unmatched = merged[other_cols].isna().all(axis=1) if other_cols else pd.Series(False, index=merged.index)
    if unmatched.any():
        right_homes = right[r_home].dropna().unique().tolist()
        right_aways = right[r_away].dropna().unique().tolist()

        for idx in merged[unmatched].index:
            home_val = merged.at[idx, l_home]
            away_val = merged.at[idx, l_away]
            mh = best_match(home_val, right_homes, cutoff=cutoff)
            ma = best_match(away_val, right_aways, cutoff=cutoff)
            if mh is None or ma is None:
                continue
            row = right[(right[r_home] == mh) & (right[r_away] == ma)]
            if row.empty:
                continue
            for c in other_cols:
                merged.at[idx, c] = row.iloc[0][c]

    return merged.drop(columns=["_nh", "_na"])
