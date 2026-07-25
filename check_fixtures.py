#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import sys, os, argparse, time
from datetime import datetime


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("league", choices=URLS.keys(), help="League to check")
    parser.add_argument("--retries", type=int, default=12, help="Number of retries")
    parser.add_argument("--interval", type=int, default=1800, help="Seconds between retries")
    args = parser.parse_args()

    url = URLS[args.league]

    for attempt in range(1, args.retries + 1):
        fixtures = get_upcoming_fixtures(url)
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
