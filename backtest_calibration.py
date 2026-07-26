#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backtest_calibration.py

Walk-forward backtest for the Dixon-Coles prediction pipeline. Answers
two open questions without waiting for any future match to be played:

  #1  Is the model calibrated? i.e. among matches where it says
      "70% confident", do those actually happen ~70% of the time?
  #2  Is the arbitrary 0.7*Prediction + 0.3*History blend in
      predictions_tier.py justified, or should the weights be
      different?

Both are answered using historical, already-settled matches. No need
to wait for outcomes: for each past matchday, we refit the model using
ONLY data available before that date (exactly what production would
have seen), predict, then compare to the real (already known) result.
This is the standard "walk-forward backtest" pattern.

Usage:
    python backtest_calibration.py                # runs default leagues
    python backtest_calibration.py --leagues E0 D1 SP1
    python backtest_calibration.py --season 2425   # e.g. '2425','2324'
"""

import argparse
import json
import os
import model_config
import numpy as np
import pandas as pd
from scipy.optimize import minimize

import majorleague_predictions as mlp

# The imported module only defines `logger` inside its own __main__ guard.
# A couple of its functions log a rare numerical-safety warning; give it a
# harmless no-op logger here so that path can't raise NameError on import.
class _NullLogger:
    def log(self, *args, **kwargs):
        pass

mlp.logger = _NullLogger()

MIN_CALIBRATION_SAMPLES = 200  # below this a fitted shrink is noise, not signal

DEFAULT_LEAGUES = {
    'En PremierLeague': 'E0', 'De Bundesliga': 'D1', 'It Serie A': 'I1',
    'Sp LaLiga': 'SP1', 'Fr Championnat': 'F1',
}

MIN_TRAIN_MATCHES = 60      # need a reasonable sample before trusting a fit
REFIT_EVERY_N_MATCHDAYS = 3  # refitting every matchday is slow; every few is a
                              # fair approximation of the live pipeline (which
                              # itself only refits once per run)


def h2h_home_win_rate(train, ht, at):
    """Proxy for the 'History H2H' feature used in predictions_tier.py:
    fraction of prior head-to-head meetings (either venue) that `ht` won.
    This is a simplified stand-in for calibration purposes -- it doesn't
    need to match historyfunc()'s exact bucketing to tell us whether
    history is predictive at all.
    """
    h2h = train[
        ((train['HomeTeam'] == ht) & (train['AwayTeam'] == at)) |
        ((train['HomeTeam'] == at) & (train['AwayTeam'] == ht))
    ]
    if h2h.empty:
        return np.nan
    ht_wins = (
        ((h2h['HomeTeam'] == ht) & (h2h['HomeGoals'] > h2h['AwayGoals'])).sum() +
        ((h2h['AwayTeam'] == ht) & (h2h['AwayGoals'] > h2h['HomeGoals'])).sum()
    )
    return ht_wins / len(h2h)


def backtest_league(url, league_code):
    raw = pd.read_csv(url, encoding='utf-8-sig')
    raw['Date'] = pd.to_datetime(raw['Date'], format='%d/%m/%Y')
    raw = raw[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']].dropna()
    raw = raw.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals'})
    raw = raw.sort_values('Date').reset_index(drop=True)

    matchdays = sorted(raw['Date'].unique())
    records = []
    params = None

    for md_idx, md in enumerate(matchdays):
        train = raw[raw['Date'] < md].copy()
        if len(train) < MIN_TRAIN_MATCHES:
            continue

        if params is None or md_idx % REFIT_EVERY_N_MATCHDAYS == 0:
            train['time_diff'] = (train['Date'].max() - train['Date']).dt.days
            try:
                params = mlp.solve_parameters_decay(train)
            except Exception:
                continue

        train_teams = set(train['HomeTeam']) | set(train['AwayTeam'])
        todays = raw[raw['Date'] == md]

        for _, m in todays.iterrows():
            ht, at = m['HomeTeam'], m['AwayTeam']
            if ht not in train_teams or at not in train_teams:
                continue  # model has no rating for a team it hasn't seen yet

            try:
                grid = mlp.dixon_coles_simulate_match(params, ht, at)
            except Exception:
                continue

            home_p = float(np.sum(np.tril(grid, -1)))
            draw_p = float(np.sum(np.diag(grid)))
            away_p = float(np.sum(np.triu(grid, 1)))
            max_g, max_ga = grid.shape[0] - 1, grid.shape[1] - 1
            over25_p = float(sum(
                grid[i, j] for i in range(max_g + 1) for j in range(max_ga + 1) if i + j > 2
            ))

            records.append({
                'League': league_code, 'Date': md, 'HomeTeam': ht, 'AwayTeam': at,
                'HomeProb': home_p, 'DrawProb': draw_p, 'AwayProb': away_p,
                'Over25Prob': over25_p,
                'ActualHome': int(m['HomeGoals'] > m['AwayGoals']),
                'ActualDraw': int(m['HomeGoals'] == m['AwayGoals']),
                'ActualAway': int(m['HomeGoals'] < m['AwayGoals']),
                'ActualOver25': int((m['HomeGoals'] + m['AwayGoals']) > 2.5),
                'H2HHomeRate': h2h_home_win_rate(train, ht, at),
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------
# #1: Calibration metrics
# ---------------------------------------------------------------------

def brier_multiclass(df):
    probs = df[['HomeProb', 'DrawProb', 'AwayProb']].values
    actual = df[['ActualHome', 'ActualDraw', 'ActualAway']].values
    return float(np.mean(np.sum((probs - actual) ** 2, axis=1)))


def log_loss_binary(p, y, eps=1e-12):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def reliability_table(prob, actual, n_bins=10):
    """Predicted probability vs realized frequency, bucketed. This is the
    single most useful diagnostic: if the model is well-calibrated,
    predicted_avg ~= actual_rate in every bucket. Consistent gaps in one
    direction mean the model is over- or under-confident.
    """
    prob = pd.Series(prob).reset_index(drop=True)
    actual = pd.Series(actual).reset_index(drop=True)
    bins = pd.cut(prob, np.linspace(0, 1, n_bins + 1), include_lowest=True)
    table = pd.DataFrame({'prob': prob, 'actual': actual, 'bin': bins})
    return table.groupby('bin', observed=True).agg(
        n=('actual', 'size'), predicted_avg=('prob', 'mean'), actual_rate=('actual', 'mean')
    )


# ---------------------------------------------------------------------
# #2: Data-driven confidence blend (replaces the arbitrary 0.7/0.3 split)
# ---------------------------------------------------------------------

def fit_logistic(X, y, l2=1.0):
    """Minimal logistic regression via scipy.optimize (no sklearn dependency,
    matching the rest of this repo's stack). X should NOT include an
    intercept column; one is added automatically.
    """
    X = np.column_stack([np.ones(len(X)), X])
    y = np.asarray(y, dtype=float)

    def neg_log_lik(beta):
        z = X @ beta
        z = np.clip(z, -30, 30)
        p = 1 / (1 + np.exp(-z))
        p = np.clip(p, 1e-12, 1 - 1e-12)
        ll = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
        return -ll + l2 * np.sum(beta[1:] ** 2)  # don't regularize intercept

    beta0 = np.zeros(X.shape[1])
    res = minimize(neg_log_lik, beta0, method='BFGS')
    return res.x  # [intercept, coef_model_prob, coef_h2h_rate]


def fit_confidence_blend(df):
    """Uses the home-win prediction as the running example: for matches
    where the model favoured the home team (HomeProb is the max of the
    three), fit whether that pick was actually correct as a function of
    HomeProb and H2HHomeRate. The resulting coefficients tell you the
    real, data-backed weight to give each signal -- to compare against
    the current hardcoded 0.7 / 0.3 split.
    """
    sub = df[df['H2HHomeRate'].notna()].copy()
    sub = sub[sub['HomeProb'] >= sub[['HomeProb', 'DrawProb', 'AwayProb']].max(axis=1) - 1e-9]
    if len(sub) < 30:
        return None, len(sub)

    X = sub[['HomeProb', 'H2HHomeRate']].values
    y = sub['ActualHome'].values
    beta = fit_logistic(X, y)

    intercept, coef_model, coef_h2h = beta
    total = abs(coef_model) + abs(coef_h2h)
    weight_model = abs(coef_model) / total if total > 0 else 1.0
    weight_h2h = abs(coef_h2h) / total if total > 0 else 0.0

    return {
        'intercept': intercept, 'coef_model_prob': coef_model, 'coef_h2h_rate': coef_h2h,
        'implied_weight_model': weight_model, 'implied_weight_h2h': weight_h2h,
        'n_samples': len(sub),
    }, len(sub)


def main(leagues, season):
    all_results = []
    for name, code in leagues.items():
        url = f"https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
        print(f"Backtesting {name} ({code})...")
        try:
            df = backtest_league(url, code)
        except Exception as e:
            print(f"  Skipped {code}: {e}")
            continue
        if df.empty:
            print(f"  No usable matches for {code} (not enough history yet).")
            continue
        all_results.append(df)
        print(f"  {len(df)} walk-forward predictions collected.")

    if not all_results:
        print("No data collected across any league. Nothing to report.")
        return

    df = pd.concat(all_results, ignore_index=True)

    print("\n" + "=" * 60)
    print("#1 CALIBRATION REPORT")
    print("=" * 60)
    print(f"Total walk-forward predictions: {len(df)}")
    print(f"1X2 multiclass Brier score: {brier_multiclass(df):.4f}  (0 = perfect, 0.667 = uninformative)")
    print(f"Over/Under 2.5 log-loss:    {log_loss_binary(df['Over25Prob'], df['ActualOver25']):.4f}  (lower is better)")

    print("\nReliability -- Home win probability (predicted vs actual):")
    print(reliability_table(df['HomeProb'], df['ActualHome']))

    print("\nReliability -- Over 2.5 goals probability (predicted vs actual):")
    print(reliability_table(df['Over25Prob'], df['ActualOver25']))

    print("\n" + "=" * 60)
    print("#2 DATA-DRIVEN CONFIDENCE BLEND (vs current hardcoded 0.7 / 0.3)")
    print("=" * 60)
    blend, n = fit_confidence_blend(df)
    if blend is None:
        print(f"Not enough matches with head-to-head history to fit a blend (n={n}, need >=30).")
    else:
        print(f"Fitted on {blend['n_samples']} home-favourite predictions with prior H2H data.")
        print(f"  Model-probability coefficient: {blend['coef_model_prob']:.3f}")
        print(f"  H2H-history coefficient:       {blend['coef_h2h_rate']:.3f}")
        print(f"  => Implied weight -- Model: {blend['implied_weight_model']:.2f} | History: {blend['implied_weight_h2h']:.2f}")
        print("  Compare this to the hardcoded 0.7 / 0.3 currently in predictions_tier.py's ConfScore.")

    out_path = "backtest_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nRaw walk-forward predictions saved to {out_path} for further analysis.")

    # ------------------------------------------------------------------
    # Emit machine-readable outputs the live pipeline actually consumes.
    # Previously this script only PRINTED its findings, so the recommended
    # blend weights and any calibration insight had to be hand-copied into
    # predictions_tier.py -- which never happened, leaving the hardcoded
    # 0.7/0.3 in place and the calibration finding unused entirely.
    # ------------------------------------------------------------------
    os.makedirs(model_config.CONFIG_DIR, exist_ok=True)

    tuning = {}
    if blend is not None:
        tuning['blend_model'] = round(float(blend['implied_weight_model']), 4)
        tuning['blend_hist'] = round(float(blend['implied_weight_h2h']), 4)
        tuning['blend_n_samples'] = int(blend['n_samples'])
    if tuning:
        tuning['generated_from'] = f"{len(df)} walk-forward predictions"
        with open(model_config.TUNING_FILE, 'w') as f:
            json.dump(tuning, f, indent=2)
        print(f"Wrote tuning overrides to {model_config.TUNING_FILE}: {tuning}")
    else:
        print("No tuning overrides written (insufficient data to fit a blend).")

    # Per-market calibration: fit shrink toward the base rate, the simple
    # monotone form model_config.calibrate_prob applies. shrink < 1 means
    # the model is overconfident and its probabilities get pulled back
    # toward the observed base rate.
    calibration = {}
    for market, prob_col, actual_col in [
        ('1', 'HomeProb', 'ActualHome'),
        ('O2_5', 'Over25Prob', 'ActualOver25'),
    ]:
        sub = df[[prob_col, actual_col]].dropna()
        if len(sub) < MIN_CALIBRATION_SAMPLES:
            print(f"  {market}: only {len(sub)} samples (need >= {MIN_CALIBRATION_SAMPLES}) -- no calibration fitted.")
            continue
        p, y = sub[prob_col].to_numpy(float), sub[actual_col].to_numpy(float)
        base = float(y.mean())
        var = float(np.sum((p - p.mean()) ** 2))
        if var <= 1e-12:
            continue
        # Least-squares slope of actual-vs-predicted around the base rate.
        shrink = float(np.sum((p - p.mean()) * (y - y.mean())) / var)
        shrink = min(max(shrink, 0.0), 1.5)
        calibration[market] = {'base': round(base, 4), 'shrink': round(shrink, 4),
                                'n_samples': int(len(sub))}
        verdict = "overconfident" if shrink < 0.9 else ("well calibrated" if shrink <= 1.1 else "underconfident")
        print(f"  {market}: base={base:.4f} shrink={shrink:.4f} (n={len(sub)}) -> {verdict}")

    if calibration:
        with open(model_config.CALIBRATION_FILE, 'w') as f:
            json.dump(calibration, f, indent=2)
        print(f"Wrote calibration to {model_config.CALIBRATION_FILE}")
    else:
        print("No calibration written (insufficient data).")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Walk-forward backtest & calibration for the Dixon-Coles pipeline")
    parser.add_argument('--leagues', nargs='+', default=None,
                         help="League codes to test, e.g. E0 D1 SP1 (default: a fixed set of 5 major leagues)")
    parser.add_argument('--season', default='2425',
                         help="football-data.co.uk season code, e.g. 2425 for 2024/25 (default: 2425, a fully completed season)")
    args = parser.parse_args()

    if args.leagues:
        leagues = {code: code for code in args.leagues}
    else:
        leagues = DEFAULT_LEAGUES

    main(leagues, args.season)
