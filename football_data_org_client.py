#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
football_data_org_client.py

Shared football-data.org v4 API access for the international competitions
(UEFA Champions League, FIFA World Cup, UEFA European Championship) that
football-data.co.uk doesn't cover. Pulled out of international_predictions.py
so check_fixtures.py and update_results.py can hit the exact same
fetch/parsing logic instead of re-implementing it slightly differently in
three places -- three implementations of "extract score from a v4 match
object" is how these subtly drift out of sync with each other.

Requires a free API token (sign up at
https://www.football-data.org/client/register) set as the
FOOTBALL_DATA_ORG_TOKEN environment variable / GitHub secret.
"""

import os
import time
import datetime
import requests
import pandas as pd

API_BASE = "https://api.football-data.org/v4"

# Confirmed-free competitions on football-data.org that fill the
# international/cup gap football-data.co.uk doesn't have.
COMPETITIONS = {
    'UEFA Champions League': 'CL',
    'FIFA World Cup': 'WC',
    'UEFA European Championship': 'EC',
}

REQUEST_DELAY = 6.5  # seconds between calls -- free tier is 10 req/min


def _token():
    token = os.environ.get("FOOTBALL_DATA_ORG_TOKEN")
    if not token:
        raise RuntimeError(
            "FOOTBALL_DATA_ORG_TOKEN is not set. Sign up free at "
            "https://www.football-data.org/client/register and add the "
            "token as a GitHub secret / environment variable."
        )
    return token


def api_get(path, params=None, retries=3):
    headers = {"X-Auth-Token": _token()}
    url = f"{API_BASE}{path}"
    for attempt in range(retries):
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        if resp.status_code == 429:
            print("WARNING: Rate limited, backing off..", url)
            time.sleep(15)
            continue
        resp.raise_for_status()
        time.sleep(REQUEST_DELAY)
        return resp.json()
    raise RuntimeError(f"Failed to fetch {url} after {retries} retries (rate limited).")


def _extract_score(match):
    """v4 uses score.fullTime.{home,away}; handle the older
    {homeTeam,awayTeam} naming defensively too, just in case."""
    ft = (match.get('score') or {}).get('fullTime') or {}
    home = ft.get('home', ft.get('homeTeam'))
    away = ft.get('away', ft.get('awayTeam'))
    return home, away


def fetch_matches(code, date_from=None, date_to=None, status=None, season=None,
                   retries=3):
    """Generic v4 matches fetch for one competition code. Returns the raw
    'matches' list from the API (list of dicts), or [] on failure -- callers
    decide whether a fetch failure for one competition should be fatal."""
    params = {}
    if date_from:
        params["dateFrom"] = date_from
    if date_to:
        params["dateTo"] = date_to
    if status:
        params["status"] = status
    if season:
        params["season"] = season

    try:
        data = api_get(f"/competitions/{code}/matches", params=params,
                        retries=retries)
    except Exception as e:
        print(f"WARNING: Could not fetch {code} matches", e)
        return []
    return data.get('matches', [])


def matches_to_df(matches):
    """Shared match-object -> DataFrame row shape (HomeTeam, AwayTeam,
    HomeGoals, AwayGoals, Date, Time) used across historical fitting,
    fixture-checking, and results settlement. HomeGoals/AwayGoals are
    None for not-yet-played matches."""
    rows = []
    for m in matches:
        home, away = _extract_score(m)
        utc_dt = pd.to_datetime(m['utcDate']).tz_localize(None)
        rows.append({
            'HomeTeam': m['homeTeam']['name'],
            'AwayTeam': m['awayTeam']['name'],
            'HomeGoals': home,
            'AwayGoals': away,
            'Date': utc_dt,
            'Time': utc_dt.strftime('%H:%M'),
        })
    return pd.DataFrame(rows)


def fetch_historical_matches(code, seasons_back=4):
    """Pull recent seasons of FINISHED matches for Dixon-Coles fitting.
    football-data.org's free tier may not expose very old seasons -- fetch
    defensively season by season and use whatever comes back rather than
    failing the whole run over one missing season."""
    rows = []
    current_year = datetime.date.today().year
    for offset in range(seasons_back):
        season = current_year - offset
        matches = fetch_matches(code, status="FINISHED", season=season)
        for m in matches:
            home, away = _extract_score(m)
            if home is None or away is None:
                continue
            rows.append({
                'HomeTeam': m['homeTeam']['name'],
                'AwayTeam': m['awayTeam']['name'],
                'HomeGoals': home,
                'AwayGoals': away,
                'Date': pd.to_datetime(m['utcDate']).tz_localize(None),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values('Date').reset_index(drop=True)
        df['time_diff'] = (df['Date'].max() - df['Date']).dt.days
    return df


def fetch_matches_on_date(code, date_str, status="SCHEDULED"):
    """Matches for a single competition on a single ISO date (used for the
    1-day fixture window the rest of the pipeline uses)."""
    matches = fetch_matches(code, date_from=date_str, date_to=date_str,
                             status=status)
    return matches_to_df(matches)


def fetch_recent_finished(code, days_back=6):
    """FINISHED matches in a trailing window, for results settlement. Wider
    than the 1-day prediction window since settlement can lag a few days
    (weekends, tournament gaps between matchdays)."""
    date_to = datetime.date.today().isoformat()
    date_from = (datetime.date.today() - datetime.timedelta(days=days_back)).isoformat()
    matches = fetch_matches(code, date_from=date_from, date_to=date_to,
                             status="FINISHED")
    return matches_to_df(matches)
