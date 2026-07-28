#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model_config.py

Single home for the model's tunable constants, plus optional
backtest-derived overrides and probability calibration.

Why this exists: values like the time-decay rate, the History/Model
confidence blend, and the VIP selection gates were previously hardcoded
magic numbers scattered across majorleague_predictions.py,
minorleague_predictions.py and predictions_tier.py -- so tuning any of
them meant editing several files consistently, and backtest_calibration.py
(which was written specifically to derive some of them) had nothing to
feed into.

Two files are read if present, both produced by backtest_calibration.py
and both entirely optional -- absent either one, the built-in defaults
below apply and behaviour is unchanged:

  model_params/tuning.json      -- scalar overrides (xi, blend weights)
  model_params/calibration.json -- per-market probability calibration

CALIBRATION, and why it matters more than the rest:
best_bets_selector.py selects picks by Edge = ModelProb - MarketImpliedProb,
taking the LARGEST edges. If the model is overconfident (raw Dixon-Coles
usually is), then ranking by edge preferentially surfaces the model's own
overconfidence rather than genuine market inefficiency -- the selection
rule actively seeks out the errors. Calibration maps raw model
probabilities onto empirically-observed hit rates so the edge is measured
against something real.

The calibration applied here is deliberately the simplest defensible
form -- a per-market shrinkage toward the market's base rate:

    p_cal = base + shrink * (p_raw - base)

with shrink < 1 pulling overconfident probabilities back toward the base
rate. It is monotone (never reorders picks within a market), interpretable
from a reliability table, and needs far less data to fit reliably than
Platt scaling or isotonic regression. Until backtest_calibration.py has
been run over enough settled history to produce calibration.json, this is
a no-op (shrink=1.0) rather than a guess.
"""

import json
import os

CONFIG_DIR = 'model_params'
TUNING_FILE = os.path.join(CONFIG_DIR, 'tuning.json')
CALIBRATION_FILE = os.path.join(CONFIG_DIR, 'calibration.json')


# ---------------------------------------------------------------- defaults

# Dixon-Coles time decay, per day. Half-life = ln(2)/xi.
# 0.0018 -> ~385 days, which is slow: it weights a match from 13 months ago
# at half the weight of today's. Published Dixon-Coles work generally uses
# faster decay (half-lives of a few months). Left at the historical value
# so behaviour doesn't silently change, but this is the single most
# valuable parameter to tune -- backtest_calibration.py can now sweep it
# (see tune_xi there) and write the result to tuning.json.
DEFAULT_XI = 0.0018

# Candidate values a tuning sweep should try. ~385d down to ~58d half-life.
XI_GRID = [0.0018, 0.0025, 0.0035, 0.0050, 0.0075, 0.0120]

# Confidence blend: ConfScore = w_model * Prediction% + w_hist * History%.
# H2H over a handful of meetings is a weak, noisy signal (often different
# squads entirely), so it should inform ranking, not dominate it.
DEFAULT_BLEND_MODEL = 0.7
DEFAULT_BLEND_HIST = 0.3


# Parsed-JSON cache, keyed by path -> (mtime, size, contents).
#
# Without this, every _load() call re-opened and re-parsed the file. That
# is on the hot path: resultdef() calls get_record_floor() once per market
# per fixture, and each of those resolves through get_base_rate() ->
# load_calibration() -> _load(). A single run therefore did roughly
# 11 x (number of fixtures) file reads to answer a question whose answer
# never changes mid-run.
#
# Keyed on (mtime, size) rather than cached outright so a file rewritten
# during the process (backtest_calibration.py writing calibration.json)
# is still picked up rather than served stale.
_CACHE = {}


def _load(path):
    try:
        stat = os.stat(path)
        stamp = (stat.st_mtime, stat.st_size)
    except OSError:
        _CACHE.pop(path, None)
        return {}

    cached = _CACHE.get(path)
    if cached is not None and cached[0] == stamp:
        return cached[1]

    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except Exception:
        data = {}

    _CACHE[path] = (stamp, data)
    return data


def get_xi():
    return float(_load(TUNING_FILE).get('xi', DEFAULT_XI))


def get_blend_weights():
    """(model_weight, history_weight), normalized to sum to 1."""
    t = _load(TUNING_FILE)
    wm = float(t.get('blend_model', DEFAULT_BLEND_MODEL))
    wh = float(t.get('blend_hist', DEFAULT_BLEND_HIST))
    total = wm + wh
    if total <= 0:
        return DEFAULT_BLEND_MODEL, DEFAULT_BLEND_HIST
    return wm / total, wh / total


def half_life_days(xi=None):
    import math
    xi = get_xi() if xi is None else xi
    return math.log(2) / xi if xi > 0 else float('inf')


# ------------------------------------------------------------- calibration

def load_calibration():
    """Per-market calibration params: {market: {'base': b, 'shrink': s}}.
    Empty dict (the default when calibration.json doesn't exist) means no
    calibration is applied anywhere."""
    return _load(CALIBRATION_FILE)


def calibrate_prob(prob, market, calibration=None):
    """Map a raw model probability onto its calibrated value for `market`.
    Unknown market, missing calibration file, or a non-finite input all
    return the input unchanged -- calibration should never be able to
    silently corrupt a probability, only refine one."""
    if prob is None:
        return prob
    try:
        p = float(prob)
    except (TypeError, ValueError):
        return prob
    if not (0.0 <= p <= 1.0):
        return prob

    calibration = load_calibration() if calibration is None else calibration
    entry = calibration.get(market)
    if not entry:
        return prob

    base = float(entry.get('base', p))
    shrink = float(entry.get('shrink', 1.0))
    out = base + shrink * (p - base)
    return min(max(out, 0.0), 1.0)


# ------------------------------------------------- market-relative gating
#
# The problem this solves: a single absolute threshold was applied to every
# market, but different markets have wildly different NATURAL rates. Over
# 1.5 lands ~75% of the time, Over 2.5 ~52%, a draw ~26%. A flat "must be
# >= 64%" gate therefore doesn't mean "confident" -- it mostly just means
# "this market has a high base rate". Over 1.5 sailed through on almost
# every fixture while Over 2.5 could essentially never qualify: it needs
# ~3.3 total expected goals to reach 64%, against a real league average of
# ~2.5-2.8, so it was structurally excluded rather than judged.
#
# Base rates below are the well-established long-run frequencies across
# major European leagues. They are PROVISIONAL defaults -- get_base_rate()
# prefers the empirically measured value from calibration.json as soon as
# backtest_calibration.py has produced one.
MARKET_BASE_RATES = {
    '1': 0.44, 'X': 0.26, '2': 0.30,
    'O1_5': 0.75, 'O2_5': 0.52, 'O3_5': 0.29,
    'GG': 0.51,
    'hO1_5': 0.42, 'hO2_5': 0.19,
    'aO1_5': 0.33, 'aO2_5': 0.13,
}

# Minimum model probability for a pick to reach the VIP list, per market.
#
# 1X2 values are deliberately left at the historical 0.64 so the change is
# neutral for those markets. The goal markets are set relative to their own
# base rates, which is the actual fix:
#   O2_5 0.60 sits ~8pp above its 0.52 base -- a genuinely informative
#        signal, and reachable, where 0.64 was not.
#   O1_5 0.85 replaces a bar its 0.75 base cleared almost automatically,
#        so it now has to earn its place like everything else.
# Net effect: relatively more Over 2.5, fewer near-automatic Over 1.5.
# These are provisional pending calibration -- see MIN_LIFT_OVER_BASE.
MARKET_MIN_PROB = {
    '1': 0.64, 'X': 0.64, '2': 0.64,
    'O1_5': 0.85, 'O2_5': 0.60, 'O3_5': 0.42,
    'GG': 0.62,
    'hO1_5': 0.55, 'hO2_5': 0.32,
    'aO1_5': 0.46, 'aO2_5': 0.24,
}

# Fallback for any market without an explicit entry above.
DEFAULT_MIN_PROB = 0.64

# A pick must also beat its own base rate by at least this margin, so a
# probability that merely matches what the market does anyway never counts
# as a "confident" tip regardless of its absolute value.
MIN_LIFT_OVER_BASE = 0.06

# Floor used when recording a market at all (resultdef). Set per-market to
# its base rate rather than a flat 0.5, so a market is written out only
# when the model rates it at least as likely as typical -- but never below
# this absolute floor, to avoid flooding the intermediate file with noise
# from naturally low-frequency markets.
MIN_RECORD_FLOOR = 0.40


def get_base_rate(market):
    """Empirically measured base rate for `market` if calibration.json has
    one (backtest_calibration.py writes it), else the provisional default."""
    entry = load_calibration().get(market) or {}
    base = entry.get('base')
    if base is not None:
        try:
            return float(base)
        except (TypeError, ValueError):
            pass
    return MARKET_BASE_RATES.get(market, DEFAULT_MIN_PROB)


def get_market_min_prob(market):
    """Absolute probability a pick must reach for this market."""
    t = _load(TUNING_FILE).get('market_min_prob', {})
    if market in t:
        return float(t[market])
    return MARKET_MIN_PROB.get(market, DEFAULT_MIN_PROB)


def get_record_floor(market):
    """Floor for recording a market at all, in resultdef."""
    return max(get_base_rate(market), MIN_RECORD_FLOOR)


# The original gate had TWO bars: a pick qualified at 0.64 WITH H2H
# support, or at 0.80 with none -- a 0.80 ratio between them. MARKET_MIN_PROB
# above is the H2H-supported bar; the unsupported bar is derived from it by
# the same ratio, so 1X2 reproduces the historical 0.64 / 0.80 exactly.
HIST_SUPPORT_RATIO = 0.80
MAX_STRONG_PROB = 0.95  # a derived bar above this is unreachable in practice


def get_strong_prob(market):
    """Probability required to qualify with NO H2H support."""
    return min(get_market_min_prob(market) / HIST_SUPPORT_RATIO, MAX_STRONG_PROB)


def has_min_lift(market, prob):
    """True when `prob` beats this market's own base rate by at least
    MIN_LIFT_OVER_BASE. Applied on top of the bars above so a probability
    that merely matches what the market does anyway never reads as a
    confident tip, whatever its absolute value."""
    if prob is None:
        return False
    try:
        p = float(prob)
    except (TypeError, ValueError):
        return False
    return (p - get_base_rate(market)) >= MIN_LIFT_OVER_BASE
