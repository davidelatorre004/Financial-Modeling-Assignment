"""
strategy.py - calculations for the "60/40 -> risk parity correlation switch" strategy.

This module contains only pure functions (no Streamlit code), so it can be imported by the
Streamlit app (app.py) and by plain Python scripts. Every step reproduces the course notebook
FM_switch_6040_risk_parity.ipynb operation by operation, so that with the default parameters
the numbers are identical to the notebook's.

Pipeline:
    download_data()  -> prices of SPY and IEI + 1-month T-bill (FRED DGS1MO)
    run_backtest()   -> returns, rolling vol/corr, weights, regime, net/gross returns
    performance_table(), drawdown_series(), drawdown_episodes(), regime_stats(),
    implementation_stats(), threshold_sensitivity() -> results shown in the app
    monthly_returns(), monte_carlo() -> bootstrap Monte Carlo of the net strategy
"""

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------------------------
# Default parameters (the values used in the notebook)
# ---------------------------------------------------------------------------------------------
DEFAULTS = {
    'start': '2007-01-01',       # sample start (IEI trades from January 2007)
    'end': '2026-10-31',         # sample end (yfinance excludes the end date)
    'w_spy_static': 0.4,         # SPY weight in the 60/40 regime
    'w_iei_static': 0.6,         # IEI weight in the 60/40 regime
    'target_vol': 0.07,          # max ex-ante volatility per asset in risk parity (7%)
    'max_lev': 2.0,              # max gross exposure (sum of weights) in risk parity
    'vol_window': 22,            # trading days used to estimate volatility
    'corr_window': 180,          # trading days used to estimate correlation
    'threshold': -0.25,          # correlation at or below which the portfolio switches to risk parity
    'tc': 0.0005,                # transaction cost: 5 bp per unit of weight traded
    'spread': 0.005,             # annual spread paid over the T-bill to fund leverage (0.5%)
}

TRADING_DAYS = 252               # annualisation factor
FRED_URL = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS1MO&cosd={start}&coed={end}'


class DataError(Exception):
    """Raised when market data cannot be downloaded or is empty (shown as a clean message in the app)."""


# ---------------------------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------------------------
def download_prices(start, end):
    """Daily adjusted closes (dividends and splits, i.e. total return) of SPY and IEI from Yahoo Finance."""
    try:
        data = yf.download(['SPY', 'IEI'], start=start, end=end, interval='1d',
                           auto_adjust=True, progress=False)
    except Exception as exc:                                    # network or API failure
        raise DataError(f'Yahoo Finance download failed: {exc}') from exc
    if data is None or data.empty:
        raise DataError('Yahoo Finance returned no data for SPY and IEI (the service may be '
                        'unavailable or rate-limiting requests). Please try again later.')
    prices = data['Close']                                      # keep closing prices only (columns IEI, SPY)
    prices = prices.dropna(how='all')                           # drop rows where both tickers are missing
    if prices.empty or prices.isna().all().any():               # one of the two tickers failed entirely
        raise DataError('Yahoo Finance returned incomplete data (one ticker is missing). '
                        'Please try again later.')
    prices = prices.tz_localize(None)                           # remove timezone so dates align with FRED
    return prices[['IEI', 'SPY']]


def download_rf(start, end):
    """1-month Treasury bill yield (FRED DGS1MO, % per year), read as CSV straight from the FRED URL."""
    url = FRED_URL.format(start=start, end=end)
    try:
        rf_raw = pd.read_csv(url, index_col=0, parse_dates=True)
    except Exception as exc:
        raise DataError(f'FRED download (DGS1MO) failed: {exc}') from exc
    if rf_raw.empty or 'DGS1MO' not in rf_raw.columns:
        raise DataError('FRED returned no data for series DGS1MO. Please try again later.')
    # safety net: older FRED CSVs used '.' for missing values; it becomes NaN (no effect if already numeric)
    rf_raw['DGS1MO'] = pd.to_numeric(rf_raw['DGS1MO'], errors='coerce')
    return rf_raw


def download_data(start=DEFAULTS['start'], end=DEFAULTS['end']):
    """Prices + T-bill in one table, aligned by date; rows with any missing value are dropped.

    Missing T-bill values fall on days when the Treasury market is closed but the stock market
    is open (e.g. Columbus Day): after dropna the next day's return covers two trading days.
    """
    start, end = str(start), str(end)
    prices = download_prices(start, end)
    rf_raw = download_rf(start, end)
    df = prices.copy()
    df['rf_annual'] = rf_raw['DGS1MO']                          # pandas aligns the two series by date
    df = df.dropna()
    if df.empty:
        raise DataError('No dates with both prices and the T-bill rate in the selected sample.')
    return df


# ---------------------------------------------------------------------------------------------
# 2-5. Signal, weights and backtest
# ---------------------------------------------------------------------------------------------
def run_backtest(data, threshold=DEFAULTS['threshold'], corr_window=DEFAULTS['corr_window'],
                 vol_window=DEFAULTS['vol_window'], target_vol=DEFAULTS['target_vol'],
                 max_lev=DEFAULTS['max_lev'], tc=DEFAULTS['tc'], spread=DEFAULTS['spread'],
                 w_spy_static=DEFAULTS['w_spy_static'], w_iei_static=DEFAULTS['w_iei_static']):
    """Run the full backtest on the output of download_data(). Returns a DataFrame with every column.

    Column names follow the notebook: r_spy, r_iei, rf, funding, r_6040, vol_*, corr, w_*_rp,
    scale, rp_on, w_spy, w_iei, lev, r_gross, turnover, costs, r_net, cum_*.
    """
    df = data.copy()

    # 2. Returns: simple daily returns; the annual % rate becomes a daily decimal rate (/100/252)
    df['r_spy'] = df['SPY'].pct_change()
    df['r_iei'] = df['IEI'].pct_change()
    df['rf'] = df['rf_annual'] / 100 / TRADING_DAYS              # daily risk-free rate
    df['funding'] = df['rf'] + spread / TRADING_DAYS             # daily cost of leverage: T-bill + spread
    df['r_6040'] = w_spy_static * df['r_spy'] + w_iei_static * df['r_iei']   # benchmark, rebalanced daily

    # 3. Rolling volatility (annualised) and correlation; each estimate uses data up to that day only
    df['vol_spy'] = np.sqrt(df['r_spy'].rolling(window=vol_window).var() * TRADING_DAYS)
    df['vol_iei'] = np.sqrt(df['r_iei'].rolling(window=vol_window).var() * TRADING_DAYS)
    cov = df['r_spy'].rolling(window=corr_window).cov(df['r_iei'])
    var_spy = df['r_spy'].rolling(window=corr_window).var()
    var_iei = df['r_iei'].rolling(window=corr_window).var()
    df['corr'] = cov / np.sqrt(var_spy * var_iei)               # correlation = cov / (sd * sd)
    df = df.dropna()                                            # drop the warm-up rows
    if len(df) < 3:
        raise DataError('The sample is too short for the chosen rolling windows. '
                        'Choose an earlier start date or shorter windows.')

    # 4a. Risk parity weights: each leg sized to target_vol; scaled down pro rata if gross > max_lev
    df['w_spy_rp'] = target_vol / df['vol_spy']
    df['w_iei_rp'] = target_vol / df['vol_iei']
    df['gross_rp'] = df['w_spy_rp'] + df['w_iei_rp']
    df['cap_binding'] = df['gross_rp'] > max_lev                # True on days when the leverage cap bites
    df['scale'] = max_lev / df['gross_rp']
    df.loc[df['scale'] > 1, 'scale'] = 1                        # no cut when gross exposure is already below the cap
    df['w_spy_rp'] = df['w_spy_rp'] * df['scale']
    df['w_iei_rp'] = df['w_iei_rp'] * df['scale']

    # 4b. Regime switch: risk parity when corr <= threshold, otherwise static 60/40
    df['rp_on'] = 0
    df.loc[df['corr'] <= threshold, 'rp_on'] = 1
    df['w_spy'] = df['rp_on'] * df['w_spy_rp'] + (1 - df['rp_on']) * w_spy_static
    df['w_iei'] = df['rp_on'] * df['w_iei_rp'] + (1 - df['rp_on']) * w_iei_static
    df['lev'] = df['w_spy'] + df['w_iei'] - 1                   # borrowed share of capital
    df.loc[df['lev'] < 0, 'lev'] = 0                            # no debt if weights sum to less than 1

    # 5. Backtest: today's return is earned by yesterday's weights (.shift(1)), so no look-ahead bias
    df['r_gross'] = (df['w_spy'].shift(1) * df['r_spy']
                     + df['w_iei'].shift(1) * df['r_iei']
                     - df['lev'].shift(1) * df['funding'])     # minus the cost of yesterday's debt
    df['turnover'] = df['w_spy'].diff(1).abs() + df['w_iei'].diff(1).abs()
    df['costs'] = df['turnover'] * tc
    df['r_net'] = df['r_gross'] - df['costs']
    df = df.dropna()                                            # drop the first row (no weights for "yesterday")
    first_day = df.index == df.index[0]
    df.loc[first_day, 'r_gross'] = 0                            # day 0: zero return, all series start at 1
    df.loc[first_day, 'r_net'] = 0
    df.loc[first_day, 'r_6040'] = 0

    # Cumulative value of $1 and drawdowns
    df['cum_net'] = (1 + df['r_net']).cumprod()
    df['cum_gross'] = (1 + df['r_gross']).cumprod()
    df['cum_6040'] = (1 + df['r_6040']).cumprod()
    df['drawdown_net'] = drawdown_series(df['cum_net'])
    df['drawdown_gross'] = drawdown_series(df['cum_gross'])
    df['drawdown_6040'] = drawdown_series(df['cum_6040'])
    return df


# ---------------------------------------------------------------------------------------------
# 6. Performance measures
# ---------------------------------------------------------------------------------------------
def sharpe(r, rf):
    """Annualised Sharpe ratio: 252 * mean(r - rf) / (sqrt(252) * std(r - rf))."""
    ex = r - rf
    return TRADING_DAYS * ex.mean() / (np.sqrt(TRADING_DAYS) * ex.std())


def performance(r, r_b, rf):
    """Six measures of r against benchmark r_b: [avg return, vol, Sharpe, alpha, beta, IR]."""
    ex = r - rf                                                 # strategy excess returns
    ex_b = r_b - rf                                             # benchmark excess returns
    avg = TRADING_DAYS * r.mean()                               # annual average return
    vol = np.sqrt(TRADING_DAYS) * r.std()                       # annual volatility
    sr = TRADING_DAYS * ex.mean() / (np.sqrt(TRADING_DAYS) * ex.std())
    cov = ((ex - ex.mean()) * (ex_b - ex_b.mean())).sum() / (len(ex) - 1)   # sample covariance
    beta = cov / ex_b.std() ** 2                                # not annualised
    alpha = TRADING_DAYS * (ex.mean() - beta * ex_b.mean())     # annualised alpha
    te = r - r_b                                                # active return
    ir = TRADING_DAYS * te.mean() / (np.sqrt(TRADING_DAYS) * te.std())
    return [avg, vol, sr, alpha, beta, ir]


def performance_table(df):
    """Table with one row per series (net, gross, 60/40); benchmark = static 60/40.

    The 60/40 row shows alpha 0, beta 1 and IR NaN (not defined against itself), as in the notebook.
    """
    row_net = performance(df['r_net'], df['r_6040'], df['rf'])
    row_gross = performance(df['r_gross'], df['r_6040'], df['rf'])
    row_6040 = performance(df['r_6040'], df['r_net'], df['rf'])[0:3] + [0.0, 1.0, np.nan]
    return pd.DataFrame([row_net, row_gross, row_6040],
                        index=['Strategy (net)', 'Strategy (gross)', 'Static 60/40'],
                        columns=['Avg. annual return', 'Volatility', 'Sharpe',
                                 'Alpha (annual)', 'Beta', 'Information ratio'])


# ---------------------------------------------------------------------------------------------
# 7. Drawdowns
# ---------------------------------------------------------------------------------------------
def drawdown_series(cum):
    """Drawdown from the previous peak (high water mark): cum / running max - 1."""
    return cum / cum.cummax() - 1


def drawdown_episodes(drawdown):
    """All drawdown episodes of a series, from the peak to full recovery (drawdown back to 0)
    or to the end of the sample. Same loop as the notebook.

    Returns a DataFrame with one row per episode: depth, peak/trough/end positions and dates,
    whether it recovered, and its length in trading days (peak to recovery/end).
    """
    dd = drawdown.to_numpy()
    episodes = []
    in_dd = False
    for i in range(len(dd)):
        if dd[i] < 0:                                           # below the previous peak
            if not in_dd:                                       # a new episode starts
                in_dd = True
                start, depth, trough = i - 1, 0.0, i            # the peak was the previous day
            if dd[i] < depth:                                   # deeper than before: new trough
                depth, trough = dd[i], i
        if in_dd and (dd[i] == 0 or i == len(dd) - 1):          # recovery, or end of sample
            episodes.append({'depth': depth, 'start': start, 'trough': trough, 'end': i,
                             'recovered': dd[i] == 0})
            in_dd = False
    ep = pd.DataFrame(episodes, columns=['depth', 'start', 'trough', 'end', 'recovered'])
    idx = drawdown.index
    ep['peak_date'] = [idx[int(s)] for s in ep['start']]
    ep['trough_date'] = [idx[int(t)] for t in ep['trough']]
    ep['end_date'] = [idx[int(e)] for e in ep['end']]
    ep['days'] = ep['end'] - ep['start']                        # trading days from peak to recovery/end
    return ep


def worst_drawdowns(drawdown, n=5):
    """The n deepest drawdown episodes, worst first (stable sort: ties keep chronological order)."""
    ep = drawdown_episodes(drawdown)
    return ep.sort_values('depth', kind='stable').head(n).reset_index(drop=True)


def max_drawdown(drawdown):
    """Worst drawdown of the series (most negative value)."""
    return drawdown.min()


# ---------------------------------------------------------------------------------------------
# 8. Regime statistics
# ---------------------------------------------------------------------------------------------
def regime_stats(df):
    """Days in risk parity, number of episodes, average and longest episode (trading days)."""
    rp = df['rp_on']
    n_entries = int(np.sum((rp == 1) & (rp.shift(1) == 0)))    # switches from 60/40 into risk parity
    n_episodes = n_entries + int(rp.iloc[0])                    # + an episode already active on day 1
    days_rp = int(rp.sum())
    # contiguous runs of rp_on == 1: (start date, length)
    runs = []
    length = 0
    for i in range(len(rp)):
        if rp.iloc[i] == 1:
            if length == 0:
                start_ep = df.index[i]
            length += 1
        if (rp.iloc[i] == 0 or i == len(rp) - 1) and length > 0:
            runs.append((start_ep, length))
            length = 0
    longest = max(runs, key=lambda x: x[1]) if runs else (None, 0)
    return {
        'days_rp': days_rp,
        'days_total': len(df),
        'share_rp': rp.mean(),
        'n_episodes': n_episodes,
        'avg_episode': days_rp / n_episodes if n_episodes > 0 else 0.0,
        'longest_episode': longest[1],
        'longest_start': longest[0],
        'runs': runs,
    }


def rp_periods(df):
    """(first date, last date) of each contiguous risk parity period, for shading charts."""
    periods = []
    for start, length in regime_stats(df)['runs']:
        i0 = df.index.get_loc(start)
        periods.append((df.index[i0], df.index[i0 + length - 1]))
    return periods


# ---------------------------------------------------------------------------------------------
# 9. Implementation
# ---------------------------------------------------------------------------------------------
def implementation_stats(df):
    """Annual turnover, transaction and funding costs, average gross exposure in risk parity."""
    rp_days = df['rp_on'] == 1
    gross_rp = (df.loc[rp_days, 'w_spy'] + df.loc[rp_days, 'w_iei']).mean() if rp_days.any() else np.nan
    return {
        'turnover': TRADING_DAYS * df['turnover'].mean(),                         # x capital per year
        'tc_cost': TRADING_DAYS * df['costs'].mean(),                             # return lost to trading
        'funding_cost': TRADING_DAYS * (df['lev'].shift(1) * df['funding']).mean(),   # cost of leverage
        'gross_exposure_rp': gross_rp,
        'cap_share': df['cap_binding'].mean(),                  # share of days with the leverage cap binding
    }


# ---------------------------------------------------------------------------------------------
# Summary and sensitivity
# ---------------------------------------------------------------------------------------------
def summary(df):
    """Headline numbers: final value of $1, Sharpe and max drawdown of each series, % days in RP."""
    return {
        'final_net': df['cum_net'].iloc[-1],
        'final_gross': df['cum_gross'].iloc[-1],
        'final_6040': df['cum_6040'].iloc[-1],
        'sharpe_net': sharpe(df['r_net'], df['rf']),
        'sharpe_gross': sharpe(df['r_gross'], df['rf']),
        'sharpe_6040': sharpe(df['r_6040'], df['rf']),
        'mdd_net': max_drawdown(df['drawdown_net']),
        'mdd_gross': max_drawdown(df['drawdown_gross']),
        'mdd_6040': max_drawdown(df['drawdown_6040']),
        'share_rp': df['rp_on'].mean(),
    }


def threshold_sensitivity(data, thresholds=(-0.45, -0.35, -0.25, -0.15, -0.05), **params):
    """Re-run the backtest for each threshold (all other parameters unchanged) and collect key results."""
    params = {k: v for k, v in params.items() if k != 'threshold'}
    rows = []
    for th in thresholds:
        res = run_backtest(data, threshold=th, **params)
        rows.append({
            'Threshold': th,
            'Sharpe (net)': sharpe(res['r_net'], res['rf']),
            'Avg. annual return (net)': TRADING_DAYS * res['r_net'].mean(),
            'Max drawdown (net)': max_drawdown(res['drawdown_net']),
            'Days in risk parity': res['rp_on'].mean(),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# 10. Monte Carlo simulation of the strategy (bootstrap of monthly net returns)
# ---------------------------------------------------------------------------------------------
MC_SEED = 2270                   # fixed seed (course number): results are reproducible


def monthly_returns(daily_returns):
    """Monthly returns from daily returns, compounded within each calendar month.

    monthly return = product of (1 + daily return) over the month - 1.
    The first and the last month of the sample are dropped because they are incomplete
    (the backtest starts and ends in the middle of a month).
    Returns a Series indexed by month-end date.
    """
    monthly = (1 + daily_returns).resample('ME').prod() - 1    # compound the days of each month
    return monthly.iloc[1:-1]                                  # drop the first and the last month


def monte_carlo(monthly_returns, n_sim, n_months, seed=MC_SEED):
    """Bootstrap Monte Carlo: n_sim paths of n_months monthly returns drawn WITH replacement
    from the historical monthly returns. Each path starts at 1 (USD):
        value_t = value_{t-1} * (1 + drawn return_t)

    Returns a matrix of shape (n_sim, n_months + 1): column 0 is the starting value 1,
    column t is the value of $1 after t months.

    Vectorised version of this loop (it gives exactly the same numbers):
        np.random.seed(seed)
        for i in range(n_sim):
            r = np.random.choice(ret_monthly, size=n_months)
            paths[i, 1:] = np.cumprod(1 + r)
    A private RandomState(seed) is the same generator as np.random.seed(seed) but does not change
    the global random state, which on a web server is shared by all users.
    """
    pool = np.asarray(monthly_returns, dtype=float)
    rng = np.random.RandomState(seed)
    draws = rng.choice(pool, size=(n_sim, n_months))           # all draws at once, row by row
    paths = np.ones((n_sim, n_months + 1))                     # column 0 = initial capital of 1
    paths[:, 1:] = np.cumprod(1 + draws, axis=1)               # compound month by month
    return paths


def monte_carlo_summary(paths):
    """Statistics of the final value of $1 across the simulated paths."""
    final = paths[:, -1]
    return {
        'median': np.median(final),
        'p5': np.percentile(final, 5),
        'p95': np.percentile(final, 95),
        'prob_loss': np.mean(final < 1),                       # share of paths ending below 1
    }
