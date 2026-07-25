#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import sys, os, argparse, time
from datetime import datetime

import football_data_org_client as fdo


URLS = {
    "majorleague": "https://www.football-data.co.uk/fixtures.csv",
    "minorleague": "https://www.football-data.co.uk/new_league_fixtures.csv"
}

def get_upcoming_fixtures(url):
    try:
        df = pd.read_csv(url)

        # Parse date column
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)

        # Only tomorrow's fixtures -- matches the 1-day window the generation
        # scripts now use. Previously unbounded (>= tomorrow, no upper
        # limit), so it reported "fixtures found" even when the only
        # matches were a week away.
        tomorrow = pd.Timestamp(datetime.today().date()) + pd.Timedelta(days=1)
        upcoming = df[df["Date"].dt.normalize() == tomorrow]

        return upcoming

    except Exception as e:
        print(f"❌ Error fetching {url}: {e}")
        return pd.DataFrame()


def get_upcoming_international_fixtures():
    """Tomorrow's scheduled matches across CL/WC/EC on football-data.org.
    Any one of the 3 having a fixture is enough to report found=true --
    international_predictions.py itself loops over all 3 and skips
    whichever have nothing, same as majorleague/minorleague skip whichever
    domestic divisions have no fixtures."""
    tomorrow = (datetime.today().date() + pd.Timedelta(days=1)).isoformat()
    frames = []
    for name, code in fdo.COMPETITIONS.items():
        try:
            df = fdo.fetch_matches_on_date(code, tomorrow, status="SCHEDULED")
        except Exception as e:
            print(f"❌ Error fetching {name} ({code}): {e}")
            continue
        if not df.empty:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def set_github_output(name, value):
    """Write a step output for the workflow to branch on. Only meaningful
    inside GitHub Actions (GITHUB_OUTPUT is set there); no-op otherwise so
    this still runs fine locally (e.g. via predictions.bat).
    """
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a") as f:
        f.write(f"{name}={value}\n")


ALL_LEAGUES = list(URLS.keys()) + ["international"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("league", choices=ALL_LEAGUES, help="League to check")
    parser.add_argument("--retries", type=int, default=12, help="Number of retries")
    parser.add_argument("--interval", type=int, default=1800, help="Seconds between retries")
    args = parser.parse_args()

    def fetch():
        if args.league == "international":
            return get_upcoming_international_fixtures()
        return get_upcoming_fixtures(URLS[args.league])

    for attempt in range(1, args.retries + 1):
        fixtures = fetch()
        if not fixtures.empty:
            print(f"✅ Upcoming fixtures found in {args.league} ({len(fixtures)} matches - {time.ctime()})")
            set_github_output("fixtures_found", "true")
            sys.exit(0)
        else:
            print(f"⏳ Attempt {attempt}/{args.retries}: no fixtures yet in {args.league} - {time.ctime()}")
            if attempt < args.retries:
                time.sleep(args.interval)

    print(f"❌ No fixtures found in {args.league} after {args.retries} attempts")
    # Previously this fell off the end with the default exit code (0), so a
    # "no fixtures" result was indistinguishable from success -- the workflow
    # would run the (pointless) prediction step regardless. Now we exit
    # non-zero AND record fixtures_found=false, so the calling workflow can
    # skip just this league's remaining steps (via continue-on-error + an
    # `if:` on the output) instead of either silently proceeding or failing
    # the whole pipeline.
    set_github_output("fixtures_found", "false")
    sys.exit(1)
