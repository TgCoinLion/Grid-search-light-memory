# Light-Memory Grid Search — How It Works

## The Problem

A trading strategy grid search tests thousands of parameter combinations
(MACD period, stochastic window, OB thresholds, etc.). For each combo we
need full performance stats: Sharpe ratio, max drawdown, annualised return.

The bottleneck is the performance calculator (`calcperf`). It is accurate
but slow (~0.05s per combo). At 500,000 combos that is **7 hours**.

---

## The Solution: A 3-Stage Funnel

Instead of running `calcperf` on every combo, we use a funnel that
eliminates 95%+ of combos cheaply before the expensive step.

```
500,000 combos
      |
      v  Stage 1: numba fast sweep        (~30 sec)
  ~50,000 survivors
      |
      v  Stage 2: cull by cum_ret         (instant)
  ~10,000 survivors
      |
      v  Stage 3: calcperf, 20 workers    (~5 min)
      |
      v
  Final ranked results
```

See `grid_search_demo.py` for the full simplified implementation.

---

## Stage 0 — Precompute Indicators Once

All rolling indicators (MACD, stochastic, OB ratio, OB trend) are computed
**once** before the sweep and stored as lookup arrays.

```python
macd_pre[n, t]            # MACD histogram for N=macd_N_list[n] at bar t
rt_pre[s, t]              # smoothed OB ratio for sma=sma_list[s] at bar t
cnt_trend_pre[s, l, t]    # OB trend for sma[s], lookback[l] at bar t
```

The sweep loop just indexes into these arrays — no recalculation per combo.
If `ob_sma` has 3 values and `ob_lb` has 3 values, we compute 9 arrays once,
and every combo that shares those parameters reuses the same precomputed result.

---

## Stage 1 — Numba Fast Sweep

The inner loop runs over all 500K combos using **numba** (a Python JIT
compiler that produces native machine code). For each combo it tracks
only two numbers:

- `cum_ret` — cumulative return while in position
- `trades`  — number of round-trip trades

```python
@jit(nopython=True)
def stage1_sweep(close, macd_pre, rt_pre, cnt_trend_pre, ...):

    for each parameter combination:
        position = 0
        cum_ret  = 0
        trades   = 0

        for t in range(n_bars):
            if buy_conditions_met:    position = 1
            elif sell_conditions_met: position = 0

            cum_ret += position * bar_return[t]
            trades  += position_changed

        if trades >= min_trades:
            save(combo_indices, cum_ret, trades)   # <-- only ~10 numbers
```

No Sharpe, no drawdown, no statistics — just a running sum.
500,000 combos complete in **~30 seconds** instead of 7 hours.

---

## Stage 2 — Cull by Cumulative Return

Sort all Stage 1 survivors by `cum_ret`, keep the top 20%.

```python
stage1_results.sort(key=lambda x: x['cum_ret'], reverse=True)
survivors = stage1_results[:int(len(stage1_results) * 0.20)]
```

`cum_ret` ignores volatility and drawdown, but the best Sharpe combos
almost always rank in the top 20% by `cum_ret`. This step is instantaneous
and eliminates 80% of remaining work.

---

## Stage 3 — Full calcperf on Survivors

The remaining ~10,000 combos go through the full performance calculator
using `ProcessPoolExecutor` (parallel workers, one per CPU core).

```python
with ProcessPoolExecutor(max_workers=20) as ex:
    results = ex.map(process_combo, survivors)
```

Each worker reconstructs the full position series for its combo and
computes Sharpe, drawdown, return/tail ratio, etc.

With 20 cores this takes ~5 minutes.

---

## Summary

| Stage | What it does | Combos in | Combos out | Time |
|-------|-------------|-----------|------------|------|
| Precompute | Rolling indicators (MACD, stoch, OB) | — | — | ~10s |
| Stage 1 (numba) | Minimal loop: cum_ret + trades only | 500,000 | ~50,000 | ~30s |
| Stage 2 | Sort by cum_ret, keep top 20% | 50,000 | ~10,000 | instant |
| Stage 3 (calcperf) | Full Sharpe/DD stats, 20 workers | 10,000 | final | ~5 min |

**Total: ~6 minutes vs ~7 hours brute force.**

The approach works because:
1. Most combos are obviously bad (too few trades, negative return) — numba eliminates them cheaply
2. `cum_ret` rank is a reliable pre-filter for Sharpe rank
3. Precomputed arrays eliminate all redundant rolling calculations
4. Parallel workers fully utilise multi-core hardware for the expensive step
