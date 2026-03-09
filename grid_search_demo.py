"""
Light-Memory Grid Search — Runnable Demo
=========================================
Strategy: SMA crossover (fast SMA > slow SMA = long, else flat)
Parameters searched: fast_win x slow_win x entry_threshold

Demonstrates the 3-stage light-memory pattern:
  Stage 1 (numba)   : fast sweep of all combos — track cum_ret + trades only
  Stage 2           : cull top 20% by cum_ret
  Stage 3 (parallel): full Sharpe/DD stats on survivors only

Run: python grid_search_demo.py
"""

import numpy as np
import pandas as pd
from numba import jit
import concurrent.futures
import time

# =============================================================================
# PARAMETER GRID
# =============================================================================
FAST_WINS  = np.array([10, 20, 30, 50, 80])
SLOW_WINS  = np.array([50, 100, 150, 200, 300, 500])
THRESHOLDS = np.array([0.0, 0.001, 0.002, 0.005])

TCOST      = 0.001
MIN_TRADES = 10
TOP_PCT    = 0.20
N_WORKERS  = 4


# =============================================================================
# STAGE 1 — NUMBA FAST SWEEP
# For each combo: run a minimal bar-by-bar loop.
# Track cum_ret and trade count only — no Sharpe, no drawdown.
# =============================================================================
@jit(nopython=True)
def stage1_sweep(close, fast_arr, slow_arr, thresholds, min_trades):
    """
    fast_arr[f, t] -- precomputed fast SMA for param index f at bar t
    slow_arr[s, t] -- precomputed slow SMA for param index s at bar t

    Returns list of (f_idx, s_idx, thr_idx, cum_ret, n_trades)
    -- only 5 numbers stored per combo, nothing else.
    """
    n = len(close)
    results = []

    for f in range(fast_arr.shape[0]):
        for s in range(slow_arr.shape[0]):
            for thr_idx in range(len(thresholds)):
                thr = thresholds[thr_idx]

                position = 0.0
                cum_ret  = 0.0
                n_trades = 0.0

                for t in range(1, n):
                    gap = fast_arr[f, t] - slow_arr[s, t]

                    if gap > thr and position == 0.0:
                        position  = 1.0
                        n_trades += 1.0
                    elif gap < -thr and position == 1.0:
                        position = 0.0

                    cum_ret += position * (close[t] / close[t-1] - 1.0)

                if n_trades < min_trades:
                    continue

                results.append((float(f), float(s), float(thr_idx),
                                cum_ret, n_trades))
    return results


# =============================================================================
# STAGE 3 WORKER — full stats for one combo
# =============================================================================
def process_combo(args):
    f_idx, s_idx, thr_idx, close, fast_arr, slow_arr, thresholds, tcost = args

    fast_sma = fast_arr[int(f_idx)]
    slow_sma = slow_arr[int(s_idx)]
    thr      = thresholds[int(thr_idx)]
    n        = len(close)

    # Rebuild full position series
    position = np.zeros(n)
    pos = 0.0
    for t in range(1, n):
        gap = fast_sma[t] - slow_sma[t]
        if gap > thr and pos == 0.0:
            pos = 1.0
        elif gap < -thr and pos == 1.0:
            pos = 0.0
        position[t] = pos

    # NAV with transaction costs
    nav = np.ones(n)
    for t in range(1, n):
        ret    = close[t] / close[t-1] - 1.0
        cost   = tcost * abs(position[t] - position[t-1])
        nav[t] = nav[t-1] * (1.0 + position[t-1] * ret - cost)

    # Sharpe (annualised, 5-min bars)
    rets        = np.diff(nav) / nav[:-1]
    bars_per_yr = 252 * 24 * 12
    sharpe      = (rets.mean() * bars_per_yr /
                   (rets.std() * np.sqrt(bars_per_yr) + 1e-10))

    # Max drawdown
    peak   = np.maximum.accumulate(nav)
    max_dd = float(np.min((nav - peak) / peak))

    n_trades = int(np.abs(np.diff(position)).sum() / 2)

    return {
        'fast_win':  int(FAST_WINS[int(f_idx)]),
        'slow_win':  int(SLOW_WINS[int(s_idx)]),
        'threshold': float(thr),
        'sharpe':    float(sharpe),
        'total_ret': float(nav[-1] - 1.0),
        'max_dd':    max_dd,
        'n_trades':  n_trades,
    }


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':

    # --- Synthetic price data ---
    np.random.seed(42)
    n_bars  = 50_000
    returns = np.random.normal(0.00005, 0.001, n_bars)
    close   = np.cumprod(1 + returns) * 100.0

    total_combos = len(FAST_WINS) * len(SLOW_WINS) * len(THRESHOLDS)
    print(f"Total combos: {total_combos:,}")
    t0 = time.time()

    # ------------------------------------------------------------------
    # STAGE 0: precompute all SMAs once
    # ------------------------------------------------------------------
    print("\nPrecomputing SMAs...")
    all_wins = np.unique(np.concatenate([FAST_WINS, SLOW_WINS]))
    sma_cache = {int(w): pd.Series(close).rolling(int(w), min_periods=1).mean().to_numpy()
                 for w in all_wins}

    fast_arr = np.stack([sma_cache[w] for w in FAST_WINS])   # shape (5, n_bars)
    slow_arr = np.stack([sma_cache[w] for w in SLOW_WINS])   # shape (6, n_bars)
    print(f"  {len(sma_cache)} SMA arrays cached [{time.time()-t0:.2f}s]")

    # ------------------------------------------------------------------
    # STAGE 1: numba fast sweep
    # ------------------------------------------------------------------
    print(f"\nStage 1: numba sweep over {total_combos:,} combos...")
    t1 = time.time()
    stage1_out = stage1_sweep(close, fast_arr, slow_arr, THRESHOLDS, MIN_TRADES)
    print(f"  {len(stage1_out):,} / {total_combos:,} passed  [{time.time()-t1:.2f}s]")

    # ------------------------------------------------------------------
    # STAGE 2: keep top 20% by cum_ret
    # ------------------------------------------------------------------
    stage1_out.sort(key=lambda x: x[3], reverse=True)
    n_keep    = max(10, int(len(stage1_out) * TOP_PCT))
    survivors = stage1_out[:n_keep]
    print(f"\nStage 2: {n_keep} / {len(stage1_out):,} kept by cum_ret")

    # ------------------------------------------------------------------
    # STAGE 3: full calcperf on survivors, parallel
    # ------------------------------------------------------------------
    print(f"\nStage 3: full stats on {n_keep} combos ({N_WORKERS} workers)...")
    t3 = time.time()

    args_list = [(r[0], r[1], r[2], close, fast_arr, slow_arr, THRESHOLDS, TCOST)
                 for r in survivors]

    with concurrent.futures.ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        results = list(ex.map(process_combo, args_list))

    results.sort(key=lambda x: x['sharpe'], reverse=True)
    print(f"  Done [{time.time()-t3:.2f}s]")

    # ------------------------------------------------------------------
    # RESULTS
    # ------------------------------------------------------------------
    print(f"\n{'='*62}")
    print("TOP 10 BY SHARPE")
    print(f"{'='*62}")
    print(f"{'fast':>5} {'slow':>5} {'thr':>6} {'sharpe':>8} {'ret%':>8} {'dd%':>8} {'trades':>7}")
    print('-' * 55)
    for r in results[:10]:
        print(f"  {r['fast_win']:>3}   {r['slow_win']:>4}   {r['threshold']:.3f}"
              f"   {r['sharpe']:>6.3f}   {r['total_ret']*100:>+5.1f}%"
              f"   {r['max_dd']*100:>+5.1f}%   {r['n_trades']:>5}")

    print(f"\nTotal time: {time.time()-t0:.1f}s")
