#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_fixtures.py

Decides whether tomorrow's prediction run should go ahead for a given
league group (majorleague / minorleague / international).

The problem with the old approach: it polled football-data.co.uk's
fixtures.csv and asked "does this contain rows dated tomorrow?". A "no"
answer is ambiguous -- it means EITHER "there genuinely are no matches
tomorrow" OR "the CSV hasn't been refreshed yet". Those need opposite
responses (give up immediately vs. keep waiting), but the check couldn't
tell them apart, so it blindly burned its whole retry budget either way:
20 minutes of waiting on a day with no football, and a premature give-up
on a day when the file was just running late.

This version separates the two questions:

  1. DO GAMES EXIST TOMORROW?  -> football-data.org's API (independent of
     football-data.co.uk, updated reliably, and already integrated here
     for the international competitions). One request covers every
     competition on the free plan.

  2. IS THE DATA WE NEED READY? -> football-data.co.uk's fixtures.csv,
     plus a cheap Last-Modified HEAD probe to tell a stale file from a
     freshly-published one.

Combining them:
  - CSV already has tomorrow's fixtures  -> proceed immediately, no wait.
  - CSV empty + oracle says no games     -> exit now, don't wait at all.
  - CSV empty + oracle says games exist  -> the file is late, keep retrying.
  - CSV empty + oracle unavailable       -> fall back to the old blind
                                            retry loop (no worse than before).

Coverage caveat: football-data.org's free plan covers most of the major
leagues (see FDO_TO_MAJOR below) but among the minor leagues only Brazil.
So for minorleague the oracle can confirm "games exist" but can never
confirm "no games" -- it returns UNKNOWN in that case and we fall back to
retrying, rather than wrongly skipping a day of picks.
"""

import pandas as pd
import sys, os, argparse, time, email.utils
from datetime import datetime, timezone

import requests

import football_data_org_client as fdo


URLS = {
    "majorleague": "https://www.football-data.co.uk/fixtures.csv",
    "minorleague": "https://www.football-data.co.uk/new_league_fixtures.csv"
}

ALL_LEAGUES = list(URLS.keys()) + ["international"]

# football-data.org competition codes -> the football-data.co.uk division
# codes they correspond to. Used only to decide whether the oracle's
# answer is meaningful for the league group being checked.
FDO_TO_MAJOR = {
    'PL': 'E0',    # Premier League
    'ELC': 'E1',   # Championship
    'BL1': 'D1',   # Bundesliga
    'SA': 'I1',    # Serie A
    'PD': 'SP1',   # La Liga
    'FL1': 'F1',   # Ligue 1
    'DED': 'N1',   # Eredivisie
    'PPL': 'P1',   # Primeira Liga
}

# The only minor-league overlap on football-data.org's free plan.
FDO_TO_MINOR = {
    'BSA': 'Brazil',
}

# Oracle verdicts
GAMES_EXIST = "games_exist"
NO_GAMES = "no_games"
UNKNOWN = "unknown"


def tomorrow_date():
    return datetime.today().date() + pd.Timedelta(days=1)


def get_upcoming_fixtures(url):
    """Tomorrow's fixtures from a football-data.co.uk CSV, or an empty
    frame if the file is unreachable/unparseable/not yet refreshed."""
    try:
        df = pd.read_csv(url)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
        tomorrow = pd.Timestamp(tomorrow_date())
        return df[df["Date"].dt.normalize() == tomorrow]
    except Exception as e:
        print(f"❌ Error fetching {url}: {e}")
        return pd.DataFrame()


def source_last_modified(url):
    """Last-Modified of the remote CSV via a cheap HEAD request, as a
    naive UTC datetime. Used to distinguish "file republished today but
    genuinely lists no matches for tomorrow" from "file is stale". Returns
    None if the server doesn't advertise it."""
    try:
        resp = requests.head(url, timeout=15, allow_redirects=True)
        raw = resp.headers.get("Last-Modified")
        if not raw:
            return None
        return email.utils.parsedate_to_datetime(raw).astimezone(timezone.utc).replace(tzinfo=None)
    except Exception:
        return None


def schedule_oracle(league):
    """Independent answer to "are there matches tomorrow?", from
    football-data.org rather than football-data.co.uk. Returns
    GAMES_EXIST / NO_GAMES / UNKNOWN."""
    if not os.environ.get("FOOTBALL_DATA_ORG_TOKEN"):
        print("ℹ️  No FOOTBALL_DATA_ORG_TOKEN set -- skipping schedule cross-check.")
        return UNKNOWN

    try:
        df = fdo.fetch_all_matches_on_date(tomorrow_date().isoformat(), status="SCHEDULED")
    except Exception as e:
        print(f"⚠️  Schedule cross-check unavailable: {e}")
        return UNKNOWN

    if df.empty or 'Competition' not in df.columns:
        # Distinguishing "the API genuinely has nothing scheduled" from
        # "the request came back malformed" isn't possible here, so treat
        # a blank response as unknown for minor (poor coverage) and as a
        # real "no games" only where coverage is good.
        relevant_codes = set()
    else:
        relevant_codes = set(df['Competition'].dropna())

    if league == "international":
        wanted = set(fdo.COMPETITIONS.values())
    elif league == "majorleague":
        wanted = set(FDO_TO_MAJOR.keys())
    else:
        wanted = set(FDO_TO_MINOR.keys())

    hits = relevant_codes & wanted
    if hits:
        print(f"🔎 Schedule cross-check: matches scheduled tomorrow in {sorted(hits)}")
        return GAMES_EXIST

    if league == "minorleague":
        # Only Brazil is covered, so "no Brazil match" says nothing about
        # Denmark/Norway/Poland/etc. Never conclude NO_GAMES from this.
        print("🔎 Schedule cross-check inconclusive for minor leagues (limited coverage).")
        return UNKNOWN

    print("🔎 Schedule cross-check: no matches scheduled tomorrow.")
    return NO_GAMES


def set_github_output(name, value):
    """Write a step output for the workflow to branch on. Only meaningful
    inside GitHub Actions (GITHUB_OUTPUT is set there); no-op otherwise so
    this still runs fine locally (e.g. via predictions.bat)."""
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a") as f:
        f.write(f"{name}={value}\n")


def found(league, fixtures):
    print(f"✅ Upcoming fixtures found in {league} ({len(fixtures)} matches - {time.ctime()})")
    set_github_output("fixtures_found", "true")
    sys.exit(0)


def not_found(league, reason):
    print(f"❌ No fixtures for {league}: {reason}")
    # Exit non-zero AND record fixtures_found=false, so the calling
    # workflow can skip just this league's remaining steps (via
    # continue-on-error + an `if:` on the output) instead of either
    # silently proceeding or failing the whole pipeline.
    set_github_output("fixtures_found", "false")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("league", choices=ALL_LEAGUES, help="League to check")
    parser.add_argument("--retries", type=int, default=3, help="Number of attempts")
    parser.add_argument("--interval", type=int, default=600, help="Seconds between attempts")
    args = parser.parse_args()

    # International reads its fixtures straight from football-data.org --
    # the same source the oracle would consult -- so there's no
    # stale-third-party-file problem to wait out. One look is definitive.
    if args.league == "international":
        fixtures = pd.DataFrame()
        for name, code in fdo.COMPETITIONS.items():
            try:
                df = fdo.fetch_matches_on_date(code, tomorrow_date().isoformat(), status="SCHEDULED")
            except Exception as e:
                print(f"❌ Error fetching {name} ({code}): {e}")
                continue
            if not df.empty:
                fixtures = pd.concat([fixtures, df], ignore_index=True)
        if not fixtures.empty:
            found(args.league, fixtures)
        not_found(args.league, "no CL/WC/EC matches scheduled tomorrow")

    url = URLS[args.league]

    # First look before consulting anything else -- on most days the file
    # is already current and we can proceed without any waiting at all.
    fixtures = get_upcoming_fixtures(url)
    if not fixtures.empty:
        found(args.league, fixtures)

    verdict = schedule_oracle(args.league)

    if verdict == NO_GAMES:
        # Confident there's simply no football tomorrow -- previously this
        # burned the full retry budget before reaching the same answer.
        not_found(args.league, "schedule cross-check confirms no matches tomorrow")

    last_mod = source_last_modified(url)
    if last_mod is not None:
        age_h = (datetime.now(timezone.utc).replace(tzinfo=None) - last_mod).total_seconds() / 3600
        print(f"🕒 {url} last modified {last_mod} UTC (~{age_h:.1f}h ago)")

    if verdict == GAMES_EXIST:
        print("⏳ Matches are scheduled tomorrow but the fixtures file hasn't caught up yet -- waiting.")
    else:
        print("⏳ Could not confirm the schedule independently -- falling back to plain retries.")

    for attempt in range(2, args.retries + 1):
        time.sleep(args.interval)
        fixtures = get_upcoming_fixtures(url)
        if not fixtures.empty:
            found(args.league, fixtures)
        print(f"⏳ Attempt {attempt}/{args.retries}: still nothing for {args.league} - {time.ctime()}")

    if verdict == GAMES_EXIST:
        not_found(args.league,
                  f"matches ARE scheduled tomorrow but the source file never refreshed "
                  f"within {args.retries} attempts -- likely a football-data.co.uk delay")
    not_found(args.league, f"nothing found after {args.retries} attempts")


if __name__ == "__main__":
    main()
