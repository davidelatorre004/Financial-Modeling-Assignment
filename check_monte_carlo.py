"""
check_monte_carlo.py - Monte Carlo with the default (notebook) parameters and the "Full history" pool,
without Streamlit. Prints the key results and checks that the vectorised strategy.monte_carlo()
gives exactly the same numbers as the reference loop with np.random.seed(2270).

Usage:  python check_monte_carlo.py
"""

import numpy as np
import strategy as s

N_SIM, N_MONTHS = 10_000, 120                        # app defaults

data = s.download_data(s.DEFAULTS['start'], s.DEFAULTS['end'])
df = s.run_backtest(data)                            # default parameters = notebook values
ret_monthly = s.monthly_returns(df['r_net'])         # complete months only

# Vectorised version used by the app
paths = s.monte_carlo(ret_monthly, N_SIM, N_MONTHS)
res = s.monte_carlo_summary(paths)

# Reference loop version (global seed, one path at a time)
np.random.seed(2270)
paths_loop = np.ones((N_SIM, N_MONTHS + 1))
for i in range(N_SIM):
    r = np.random.choice(ret_monthly, size=N_MONTHS)
    for t in range(N_MONTHS):                        # compound month by month
        paths_loop[i, t + 1] = paths_loop[i, t] * (1 + r[t])
res_loop = s.monte_carlo_summary(paths_loop)

print(f"Complete months in the pool: {len(ret_monthly)} "
      f"({ret_monthly.index[0]:%b %Y} - {ret_monthly.index[-1]:%b %Y})")
print(f"Horizon {N_MONTHS} months, {N_SIM:,} simulations, seed 2270")
print(f"Median final value:  {res['median']:.4f}")
print(f"5th percentile:      {res['p5']:.4f}")
print(f"95th percentile:     {res['p95']:.4f}")
print(f"Probability of loss: {res['prob_loss']:.2%}")
print()
print('Loop version:', {k: round(v, 6) for k, v in res_loop.items()})
print('Identical draws and paths (exact):', np.array_equal(paths, paths_loop))
print('Max abs difference in paths:', np.max(np.abs(paths - paths_loop)))
