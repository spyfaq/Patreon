#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import requests
import sys, os, argparse, time
from datetime import datetime

import football_data_org_client as fdo
import date_utils


URLS = {
    "majorleague": "https://www.football-data.co.uk/fixtures.csv",
    "minorleague": "https://www.football-data.co.uk/new_league_fixtures.csv"
}

def get_upcoming_fixtures(url):
    try:
        df = pd.read_csv(url)

        # Today's fetch window (today from the 09:00 cutoff onward, plus
        # tomorrow up to 09:00 -- see date_utils.py). Previously matched
        # tomorrow's calendar date only and ignored the Time column
        # entirely, so a run on day X reported nothing found unless day
        # X+1 already had fixtures listed, even though day X's own
        # matches (which this run should also cover) were sitting right
        # there in the same file.
        upcoming = df[date_utils.in_fetch_window(df["Date"], df["Time"])]

        return upcoming

    except Exception as e:
        print(f"❌ Error fetching {url}: {e}")
        return pd.DataFrame()


def get_upcoming_international_fixtures():
    """Scheduled matches across CL/WC/EC on football-data.org, within
    today's fetch window. Any one of the 3 having a fixture is enough to
    report found=true -- international_predictions.py itself loops over
    all 3 and skips whichever have nothing, same as majorleague/
    minorleague skip whichever domestic divisions have no fixtures."""
    today = datetime.today().date()
    tomorrow = today + pd.Timedelta(days=1)
    frames = []
    for name, code in fdo.COMPETITIONS.items():
        try:
            matches = fdo.fetch_matches(code, date_from=today.isoformat(), date_to=tomorrow.isoformat(),
                                         status="SCHEDULED")
            df = fdo.matches_to_df(matches)
        except Exception as e:
            print(f"❌ Error fetching {name} ({code}): {e}")
            continue
        if not df.empty:
            df = df[date_utils.in_fetch_window(df["Date"], df["Time"])]
        if not df.empty:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# Manual interrupt: while a check is sitting in its retry wait, you can
# short-circuit it from the GitHub UI without cancelling the run --
# Settings -> Secrets and variables -> Actions -> Variables tab -> New
# repository variable, named FORCE_PROCEED_<LEAGUE> (e.g.
# FORCE_PROCEED_MAJORLEAGUE), value "true" (treat as fixtures found,
# proceed now) or "false" (treat as no fixtures, stop waiting now). The
# running check polls for this every POLL_INTERVAL seconds and acts on it
# as soon as it appears, then clears the variable itself so it's a
# one-shot override and doesn't silently affect tomorrow's run too.
POLL_INTERVAL = 15  # seconds between override checks while waiting


def _variable_name(league):
    return f"FORCE_PROCEED_{league.upper()}"


def _variables_url(league):
    repo = os.environ.get("GITHUB_REPOSITORY")
    if not repo:
        return None
    return f"https://api.github.com/repos/{repo}/actions/variables/{_variable_name(league)}"


def get_force_signal(league):
    """Check the per-league override variable. Returns 'true'/'false' if
    a valid override is set, or None if there's nothing to act on (no
    token, no repo context, variable unset/empty/invalid) -- any of which
    just means "keep polling normally"."""
    token = os.environ.get("GITHUB_TOKEN")
    url = _variables_url(league)
    if not token or not url:
        return None
    try:
        resp = requests.get(
            url,
            headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        value = (resp.json().get("value") or "").strip().lower()
        return value if value in ("true", "false") else None
    except Exception:
        return None


def clear_force_signal(league):
    """Blank the variable out after acting on it so it's a one-shot
    trigger, not a standing override future runs would also pick up."""
    token = os.environ.get("GITHUB_TOKEN")
    url = _variables_url(league)
    if not token or not url:
        return
    try:
        requests.patch(
            url,
            headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
            json={"name": _variable_name(league), "value": ""},
            timeout=10,
        )
    except Exception:
        pass


def act_on_signal(league, signal):
    verb = "fixtures found" if signal == "true" else "no fixtures"
    print(f"🧑‍💻 Manual override: {_variable_name(league)}={signal} -- treating as '{verb}', proceeding now.")
    clear_force_signal(league)
    set_github_output("fixtures_found", signal)
    sys.exit(0 if signal == "true" else 1)


def wait_with_override(league, total_seconds):
    """Sleep up to total_seconds in short POLL_INTERVAL chunks, checking
    for a manual override each time so a stalled wait can be cut short
    instead of sitting through the rest of it."""
    elapsed = 0
    while elapsed < total_seconds:
        signal = get_force_signal(league)
        if signal is not None:
            act_on_signal(league, signal)
        chunk = min(POLL_INTERVAL, total_seconds - elapsed)
        time.sleep(chunk)
        elapsed += chunk


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


def main():
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
        signal = get_force_signal(args.league)
        if signal is not None:
            act_on_signal(args.league, signal)

        fixtures = fetch()
        if not fixtures.empty:
            print(f"✅ Upcoming fixtures found in {args.league} ({len(fixtures)} matches - {time.ctime()})")
            set_github_output("fixtures_found", "true")
            sys.exit(0)
        else:
            print(f"⏳ Attempt {attempt}/{args.retries}: no fixtures yet in {args.league} - {time.ctime()}")
            if attempt < args.retries:
                wait_with_override(args.league, args.interval)

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


if __name__ == "__main__":
    main()
