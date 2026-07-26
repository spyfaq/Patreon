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


def _load(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


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
