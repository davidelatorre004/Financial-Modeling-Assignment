# Correlation-Switch Risk Parity (SPY & IEI)

An interactive Streamlit app that backtests a two-asset strategy that switches between a static 60/40
portfolio and a levered risk parity portfolio, depending on the stock–bond correlation. You can change
every parameter in the sidebar and see the effect on performance, drawdowns, regime switches,
implementation costs and threshold sensitivity.

**Live app:** https://financial-modeling-assignment-wcbrtezmes4ay4sgygpdew.streamlit.app/  
**Repository:** https://github.com/davidelatorre004/Financial-Modeling-Assignment

## The strategy in 5 lines

1. Two ETFs: SPY (S&P 500) and IEI (US Treasuries 3–7y); benchmark = static 60/40 (60% IEI, 40% SPY), rebalanced daily.
2. Every day we compute the 180-day rolling correlation between the daily returns of SPY and IEI.
3. If the correlation is ≤ −0.25 the portfolio switches to risk parity: each asset is sized to 7% ex-ante volatility (22-day estimate), with gross exposure capped at 2×. Otherwise it holds the 60/40.
4. Leverage is funded at the 1-month T-bill (FRED DGS1MO) + 0.5%; trades cost 5 bp per unit of weight traded.
5. Weights decided at today's close earn tomorrow's return (no look-ahead bias).

## What the app shows

The KPIs stay at the top; the rest is organised in tabs (Overview, Performance, Drawdowns,
Regimes & weights, Implementation & sensitivity).

- KPIs (final value of $1, Sharpe, max drawdown, % of days in risk parity), strategy vs 60/40
- Cumulative return (net, gross of transaction costs, 60/40)
- Performance table: average return, volatility, Sharpe, alpha, beta, information ratio
- Drawdown chart and the 5 worst drawdowns of each portfolio
- Rolling correlation with the threshold and the risk parity periods shaded, plus regime statistics
- Portfolio weights (SPY, IEI) and gross exposure over time, showing when leverage is used
- Implementation: turnover, transaction and funding costs, gross exposure
- Threshold sensitivity (−0.45 … −0.05), with the currently selected threshold highlighted

## Files

| File | Content |
|---|---|
| `strategy.py` | Pure functions: data download, signal, weights, backtest, metrics, drawdowns (no Streamlit code) |
| `app.py` | Streamlit interface; imports from `strategy.py` |
| `check_backtest.py` | Runs the backtest with the default parameters without Streamlit and prints the headline numbers |
| `requirements.txt` | Python dependencies (pinned versions) |
| `.streamlit/config.toml` | App theme (colours matching the charts, light mode) |

The calculations reproduce the course notebook `FM_switch_6040_risk_parity.ipynb` exactly: with the
default parameters, every column of the backtest is identical to the notebook's.

## Run locally

Requires Python 3.12 or newer.

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501. Data is downloaded on the first run (Yahoo Finance via
`yfinance` for prices, FRED for the T-bill rate; no API key needed) and cached for one day.

To check the numbers without the interface:

```bash
python check_backtest.py
```

## Data sources

- Prices: Yahoo Finance via the `yfinance` package (SPY, IEI; adjusted for dividends and splits).
- Risk-free and funding rate: FRED series DGS1MO (1-month Treasury constant maturity), read as CSV.

## Disclaimer

For educational purposes only, not investment advice. Past performance does not guarantee future results.

---

This app is part of the group assignment for the **Nova SBE course 2270 Financial Modeling**.
