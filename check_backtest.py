"""
check_backtest.py - run the backtest with the default (notebook) parameters, without Streamlit,
and print the headline numbers so they can be compared with the notebook output.

Usage:  python check_backtest.py
"""

import strategy as s

data = s.download_data(s.DEFAULTS['start'], s.DEFAULTS['end'])   # same download as the notebook
df = s.run_backtest(data)                                         # default parameters = notebook values
res = s.summary(df)
reg = s.regime_stats(df)
imp = s.implementation_stats(df)

print(f"Sample: {df.index[0]:%d/%m/%Y} - {df.index[-1]:%d/%m/%Y} ({len(df)} trading days)")
print()
print(f"Final value of $1   net {res['final_net']:.4f} | gross {res['final_gross']:.4f} | 60/40 {res['final_6040']:.4f}")
print(f"Sharpe              net {res['sharpe_net']:.4f} | gross {res['sharpe_gross']:.4f} | 60/40 {res['sharpe_6040']:.4f}")
print(f"Max drawdown        net {res['mdd_net']:.4%} | gross {res['mdd_gross']:.4%} | 60/40 {res['mdd_6040']:.4%}")
print(f"Days in risk parity {reg['days_rp']} of {reg['days_total']} ({res['share_rp']:.2%}), "
      f"{reg['n_episodes']} episodes, average {reg['avg_episode']:.0f} days, longest {reg['longest_episode']} days")
print(f"Implementation      turnover {imp['turnover']:.2f}x | trading costs {imp['tc_cost']:.2%} | "
      f"funding {imp['funding_cost']:.2%} | gross exposure in RP {imp['gross_exposure_rp']:.2f}")
print()
print(s.performance_table(df).round(4).to_string())
