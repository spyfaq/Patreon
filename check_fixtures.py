#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import io, sys, argparse, time, requests
from datetime import datetime


URLS = {
    "majorleague": "https://www.football-data.co.uk/fixtures.csv",
    "minorleague": "https://www.football-data.co.uk/new_league_fixtures.csv"
}

def get_upcoming_fixtures(url):
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))

        if "Date" not in df.columns:
            print(f"⚠️ No 'Date' column found in {url}")
            return pd.DataFrame()

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
        today = pd.Timestamp(datetime.today().date())
        return df[df["Date"] >= today]

    except Exception as e:
        print(f"❌ Error fetching {url}: {e}")
        return pd.DataFrame()

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
            print(f"✅ Upcoming fixtures found in {args.league} ({len(fixtures)} matches)")
            print(fixtures.head(10))
            sys.exit(0)
        else:
            print(f"⏳ Attempt {attempt}/{args.retries}: no fixtures yet in {args.league}")
            if attempt < args.retries:
                time.sleep(args.interval)

    print(f"❌ No fixtures found in {args.league} after {args.retries} attempts")
    sys.exit(1)
