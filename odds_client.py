#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
odds_client.py

football-data.co.uk (the odds source the domestic prediction scripts read
straight off their fixtures feed) only carries domestic-league odds -- it has nothing for the
Champions League, World Cup, or European Championship. This module fills
that specific gap using The Odds API (https://the-odds-api.com), which does
cover those three on its free tier (500 requests/month, no card required).

Requires a free API key (sign up at https://the-odds-api.com) set as the
ODDS_API_KEY environment variable / GitHub secret.

NOTE: this can't be exercised against the live API in this sandbox
(api.the-odds-api.com isn't reachable here) -- verify against the real
API once the key is added as a GitHub secret.
"""

import os
import requests
import pandas as pd

API_BASE = "https://api.the-odds-api.com/v4"

# football-data.org competition code -> The Odds API sport_key. WC/EC only
# resolve to real events during their respective tournament windows -- an
# empty response outside of that is expected, not an error.
SPORT_KEYS = {
    'CL': 'soccer_uefa_champs_league',
    'WC': 'soccer_fifa_world_cup',
    'EC': 'soccer_uefa_european_championship',
}


def _api_key():
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        raise RuntimeError(
            "ODDS_API_KEY is not set. Sign up free at https://the-odds-api.com "
            "and add the key as a GitHub secret / environment variable."
        )
    return key


def _average(values):
    vals = [v for v in values if v]
    return sum(vals) / len(vals) if vals else None


def fetch_odds_for_competition(code):
    """Fetch upcoming h2h + totals odds for one international competition,
    shaped into the same Avg*/Date/Time/Div/HomeTeam/AwayTeam columns the
    domestic fixtures feed provides, so market_odds prices an
    international fixture exactly like a domestic one.
    Returns an empty DataFrame (not an exception) if the competition has
    no events right now or the request fails -- a quiet no-op is correct
    outside of a tournament window."""
    sport_key = SPORT_KEYS.get(code)
    if sport_key is None:
        return pd.DataFrame()

    try:
        resp = requests.get(
            f"{API_BASE}/sports/{sport_key}/odds",
            params={
                "apiKey": _api_key(),
                "regions": "eu,uk",
                "markets": "h2h,totals",
                "oddsFormat": "decimal",
            },
            timeout=30,
        )
        resp.raise_for_status()
        events = resp.json()
    except Exception as e:
        print(f"WARNING: Could not fetch odds for {code} ({sport_key})", e)
        return pd.DataFrame()

    rows = []
    for event in events:
        home_team = event.get('home_team')
        away_team = event.get('away_team')
        if not home_team or not away_team:
            continue

        h_prices, d_prices, a_prices = [], [], []
        over25_prices, under25_prices = [], []

        for bookmaker in event.get('bookmakers', []):
            for market in bookmaker.get('markets', []):
                if market['key'] == 'h2h':
                    for outcome in market.get('outcomes', []):
                        if outcome['name'] == home_team:
                            h_prices.append(outcome.get('price'))
                        elif outcome['name'] == away_team:
                            a_prices.append(outcome.get('price'))
                        elif outcome['name'].lower() == 'draw':
                            d_prices.append(outcome.get('price'))
                elif market['key'] == 'totals':
                    for outcome in market.get('outcomes', []):
                        if outcome.get('point') != 2.5:
                            continue
                        if outcome['name'].lower() == 'over':
                            over25_prices.append(outcome.get('price'))
                        elif outcome['name'].lower() == 'under':
                            under25_prices.append(outcome.get('price'))

        commence = pd.to_datetime(event.get('commence_time')).tz_localize(None)
        rows.append({
            'Date': commence,
            'Time': commence.strftime('%H:%M'),
            'Div': code,
            'HomeTeam': home_team,
            'AwayTeam': away_team,
            'AvgH': _average(h_prices),
            'AvgD': _average(d_prices),
            'AvgA': _average(a_prices),
            'AvgOver25': _average(over25_prices),
            'AvgUnder25': _average(under25_prices),
        })

    return pd.DataFrame(rows)


def fetch_all_international_odds():
    """Odds for all 3 international competitions, concatenated into one
    DataFrame, consumed by international_predictions.py to price its rows."""
    frames = [fetch_odds_for_competition(code) for code in SPORT_KEYS]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=['Date', 'Time', 'Div', 'HomeTeam', 'AwayTeam',
                                      'AvgH', 'AvgD', 'AvgA', 'AvgOver25', 'AvgUnder25'])
    return pd.concat(frames, ignore_index=True)
