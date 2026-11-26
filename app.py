"""
app.py - Streamlit interface for the "60/40 -> risk parity correlation switch" strategy.

All calculations live in strategy.py (pure functions, no Streamlit). This file only:
  1. reads the parameters from the sidebar,
  2. downloads the data (cached for one day) and runs the backtest,
  3. shows KPIs, charts and tables.

Run locally with:  streamlit run app.py
"""

import datetime as dt

import matplotlib
matplotlib.use('Agg')                                   # non-interactive backend (needed on servers)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import streamlit as st

import strategy as s

# ---------------------------------------------------------------------------------------------
# Page setup and colours (net strategy = coral, everything else = shades of blue)
# ---------------------------------------------------------------------------------------------
st.set_page_config(page_title='Correlation-Switch Risk Parity', page_icon='📈', layout='wide')

C_STRAT = 'coral'                 # net strategy (reserved)
C_BLUE_DARK = 'steelblue'         # main lines, gross strategy (dashed)
C_BLUE_MID = 'cornflowerblue'     # static 60/40 benchmark
C_BLUE_LIGHT = 'lightsteelblue'   # secondary lines
C_BAND = 'aliceblue'              # shaded bands

GITHUB_URL = 'https://github.com/davidelatorre004/Financial-Modeling-Assignment'

# ---------------------------------------------------------------------------------------------
# Sidebar: inputs (defaults = notebook values). Each widget has a key so "Reset" can restore it.
# Values shown in % or bp in the sidebar are converted to decimals before calling strategy.py.
# Display defaults are typed literally (7.0, not 0.07 * 100) so that dividing by 100 gives exactly
# the notebook's 0.07: in floating point 0.07 * 100 = 7.000000000000001.
# ---------------------------------------------------------------------------------------------
WIDGET_DEFAULTS = {
    'threshold': s.DEFAULTS['threshold'],                    # correlation threshold
    'corr_window': s.DEFAULTS['corr_window'],                # days
    'vol_window': s.DEFAULTS['vol_window'],                  # days
    'target_vol_pct': 7.0,                                    # % per year (= 0.07 in strategy.py)
    'max_lev': s.DEFAULTS['max_lev'],                        # x capital
    'tc_bp': 5.0,                                             # bp per unit of weight traded (= 0.0005)
    'spread_pct': 0.5,                                        # % per year (= 0.005)
    'equity_pct': 40.0,                                       # % of SPY in the static portfolio (= 0.4)
    'start': dt.date.fromisoformat(s.DEFAULTS['start']),
    'end': dt.date.fromisoformat(s.DEFAULTS['end']),
}


def reset_defaults():
    """Callback of the reset button: put every widget back to its default value."""
    for key, value in WIDGET_DEFAULTS.items():
        st.session_state[key] = value


for key, value in WIDGET_DEFAULTS.items():                 # initialise the widgets on the first run
    st.session_state.setdefault(key, value)

with st.sidebar:
    st.header('Strategy parameters')
    st.slider('Correlation threshold', -0.6, 0.0, step=0.05, key='threshold',
              help='Switch to risk parity when the rolling SPY-IEI correlation is at or below this value.')
    st.slider('Correlation window (days)', 60, 250, step=1, key='corr_window')
    st.slider('Volatility window (days)', 10, 60, step=1, key='vol_window')
    st.slider('Target volatility per asset (%)', 3.0, 12.0, step=0.5, key='target_vol_pct')
    st.slider('Maximum gross leverage (x)', 1.0, 3.0, step=0.1, key='max_lev')
    st.slider('Transaction cost (bp per unit traded)', 0.0, 30.0, step=0.5, key='tc_bp')
    st.slider('Funding spread over T-bill (%)', 0.0, 2.0, step=0.1, key='spread_pct')
    st.slider('Static portfolio: equity share (%)', 20.0, 60.0, step=5.0, key='equity_pct',
              help='SPY weight of the static benchmark; bonds (IEI) get the rest.')
    st.caption(f"Static weights: {st.session_state['equity_pct']:.0f}% SPY / "
               f"{100 - st.session_state['equity_pct']:.0f}% IEI")
    st.date_input('Start date', min_value=dt.date(2007, 1, 1), max_value=dt.date.today(), key='start')
    st.date_input('End date', min_value=dt.date(2007, 1, 1), max_value=dt.date.today(), key='end')
    st.button('Reset to defaults', on_click=reset_defaults, width='stretch')

# Parameters in the units used by strategy.py (decimals)
w_spy_static = st.session_state['equity_pct'] / 100
params = dict(
    threshold=st.session_state['threshold'],
    corr_window=int(st.session_state['corr_window']),
    vol_window=int(st.session_state['vol_window']),
    target_vol=st.session_state['target_vol_pct'] / 100,
    max_lev=st.session_state['max_lev'],
    tc=st.session_state['tc_bp'] / 10000,
    spread=st.session_state['spread_pct'] / 100,
    w_spy_static=w_spy_static,
    w_iei_static=1 - w_spy_static,
)


# ---------------------------------------------------------------------------------------------
# Data (cached for one day) and backtest
# ---------------------------------------------------------------------------------------------
@st.cache_data(ttl=24 * 3600, show_spinner='Downloading prices (Yahoo Finance) and T-bill rate (FRED)...')
def load_data(start, end):
    """Download prices + T-bill once per (start, end) pair and keep them for a day."""
    return s.download_data(start, end)


st.title('Correlation-Switch Risk Parity: SPY & IEI')
st.markdown(
    "By default the portfolio is a **static 60/40** (60% IEI Treasuries 3–7y, 40% SPY S&P 500), rebalanced daily.  \n"
    "When the rolling stock–bond correlation falls to the **threshold or below**, bonds are a reliable hedge and it "
    "switches to **risk parity**: each asset is sized to a target volatility, with leverage up to a cap.  \n"
    "Leverage is funded at the 1-month T-bill plus a spread, and every trade pays a transaction cost."
)

start, end = st.session_state['start'], st.session_state['end']
if start >= end:
    st.error('The start date must be before the end date.')
    st.stop()

try:
    data = load_data(start.isoformat(), end.isoformat())
    df = s.run_backtest(data, **params)
except s.DataError as exc:                                 # known problem: clean message, no traceback
    st.error(f'Could not run the backtest: {exc}')
    st.stop()
except Exception as exc:                                   # anything unexpected (e.g. network down)
    st.error(f'Could not download the data or run the backtest ({type(exc).__name__}). '
             'Please check your connection and try again later.')
    st.stop()

res = s.summary(df)
reg = s.regime_stats(df)
imp = s.implementation_stats(df)

st.caption(f"Backtest sample: **{df.index[0]:%d %b %Y} – {df.index[-1]:%d %b %Y}** "
           f"({len(df):,} trading days; the first {params['corr_window']} days of data are used to "
           f"estimate the first correlation).")


# ---------------------------------------------------------------------------------------------
# Helpers for charts and tables
# ---------------------------------------------------------------------------------------------
def style_axes(ax, ylabel):
    """Quiet axes: light grid, no top/right frame, year ticks."""
    ax.set_ylabel(ylabel)
    ax.grid(True, color='#e6e6e6', linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.margins(x=0.01)


def pct(x, digits=1):
    """Format a decimal as a percentage string."""
    return f'{x:.{digits}%}'


# ---------------------------------------------------------------------------------------------
# 2. KPI row: strategy (net) vs static 60/40
# ---------------------------------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric('Final value of $1 (net)', f"${res['final_net']:.2f}",
          f"60/40: ${res['final_6040']:.2f}", delta_color='off')
k2.metric('Sharpe ratio (net)', f"{res['sharpe_net']:.2f}",
          f"60/40: {res['sharpe_6040']:.2f}", delta_color='off')
k3.metric('Max drawdown (net)', pct(res['mdd_net']),
          f"60/40: {pct(res['mdd_6040'])}", delta_color='off')
k4.metric('Days in risk parity', pct(res['share_rp']),
          '60/40: 0% (never switches)', delta_color='off')

# ---------------------------------------------------------------------------------------------
# 3. Cumulative return
# ---------------------------------------------------------------------------------------------
st.subheader('Cumulative return (base = 1)')
fig, ax = plt.subplots(figsize=(11, 4.8))
ax.plot(df['cum_net'], color=C_STRAT, linewidth=2, label='Strategy (net)')
ax.plot(df['cum_gross'], '--', color=C_BLUE_DARK, linewidth=1.5, label='Strategy (gross of transaction costs)')
ax.plot(df['cum_6040'], color=C_BLUE_MID, linewidth=1.5, label='Static 60/40')
style_axes(ax, 'Value of $1 invested')
ax.legend(frameon=False, loc='upper left')
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ---------------------------------------------------------------------------------------------
# 4. Performance table
# ---------------------------------------------------------------------------------------------
st.subheader('Performance measures (benchmark: static 60/40)')
perf = s.performance_table(df)
perf_display = pd.DataFrame(index=perf.index)               # formatted copy for display (strings)
for col in ['Avg. annual return', 'Volatility', 'Alpha (annual)']:
    perf_display[col] = [pct(x, 2) for x in perf[col]]
for col in ['Sharpe', 'Beta', 'Information ratio']:
    perf_display[col] = ['–' if np.isnan(x) else f'{x:.2f}' for x in perf[col]]   # IR not defined for 60/40
perf_display = perf_display[perf.columns]                   # original column order
st.dataframe(perf_display, width='stretch')
st.caption('Daily simple returns, 252 days a year. Sharpe uses excess returns over the 1-month T-bill. '
           'Beta is the regression slope on the 60/40 excess return (not annualised); alpha is annualised. '
           'Information ratio = annualised mean of (strategy − 60/40) / annualised tracking error.')

# ---------------------------------------------------------------------------------------------
# 5. Drawdowns
# ---------------------------------------------------------------------------------------------
st.subheader('Drawdowns')
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(100 * df['drawdown_6040'], color=C_BLUE_MID, linewidth=1.5, label='Static 60/40')
ax.plot(100 * df['drawdown_net'], color=C_STRAT, linewidth=1.5, label='Strategy (net)')
style_axes(ax, 'Drawdown from previous peak (%)')
ax.legend(frameon=False, loc='lower left')
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)


def drawdown_table(drawdown):
    """The 5 worst episodes, formatted for display, with the worst one highlighted."""
    top = s.worst_drawdowns(drawdown, 5)
    table = pd.DataFrame({
        # plain lists (not Series), so pandas does not try to align them with the new index
        'Depth': [pct(x, 2) for x in top['depth']],
        'Peak': [d.strftime('%d %b %Y') for d in top['peak_date']],
        'Trough': [d.strftime('%d %b %Y') for d in top['trough_date']],
        'Recovery': [d.strftime('%d %b %Y') if rec else 'not recovered'
                     for d, rec in zip(top['end_date'], top['recovered'])],
        'Days': [str(int(x)) for x in top['days']],             # trading days, peak to recovery
    }, index=[f'{i + 1}' for i in range(len(top))])
    table.index.name = 'Rank'
    highlight = lambda row: (['background-color: aliceblue; font-weight: bold'] * len(row)
                             if row.name == '1' else [''] * len(row))
    return table.style.apply(highlight, axis=1)


c1, c2 = st.columns(2)
with c1:
    st.markdown('**Strategy (net): 5 worst drawdowns**')
    st.dataframe(drawdown_table(df['drawdown_net']), width='stretch')
with c2:
    st.markdown('**Static 60/40: 5 worst drawdowns**')
    st.dataframe(drawdown_table(df['drawdown_6040']), width='stretch')
st.caption('An episode runs from the peak to full recovery of that peak (or to the end of the sample). '
           'Days = trading days from the peak to the recovery (or to the end of the sample).')

# ---------------------------------------------------------------------------------------------
# 6. Correlation and regime switches
# ---------------------------------------------------------------------------------------------
st.subheader(f"Rolling {params['corr_window']}-day SPY–IEI correlation and risk parity periods")
fig, ax = plt.subplots(figsize=(11, 4.5))
for first, last in s.rp_periods(df):                       # shade each risk parity period
    i_last = df.index.get_loc(last)
    right = df.index[min(i_last + 1, len(df) - 1)]          # extend to the next day so 1-day periods show
    ax.axvspan(first, right, color=C_BAND, linewidth=0)
ax.plot(df['corr'], color=C_BLUE_DARK, linewidth=1.5)
ax.axhline(params['threshold'], linestyle='--', color=C_BLUE_MID, linewidth=1.5)
ax.axhline(0, color=C_BLUE_LIGHT, linewidth=0.8)
style_axes(ax, 'Correlation')
ax.legend(handles=[plt.Line2D([], [], color=C_BLUE_DARK, linewidth=1.5,
                              label=f"Correlation ({params['corr_window']} days)"),
                   plt.Line2D([], [], color=C_BLUE_MID, linestyle='--', linewidth=1.5,
                              label=f"Switch threshold ({params['threshold']:.2f})"),
                   Patch(facecolor=C_BAND, edgecolor=C_BLUE_LIGHT, label='In risk parity')],
          frameon=False, loc='upper left')
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

r1, r2, r3, r4 = st.columns(4)
r1.metric('Days in risk parity', f"{reg['days_rp']:,} of {reg['days_total']:,}", pct(reg['share_rp']),
          delta_color='off')
r2.metric('Risk parity episodes', f"{reg['n_episodes']}")
r3.metric('Average episode', f"{reg['avg_episode']:.0f} days")
r4.metric('Longest episode', f"{reg['longest_episode']} days",
          f"from {reg['longest_start']:%d %b %Y}" if reg['longest_start'] is not None else None,
          delta_color='off')

# ---------------------------------------------------------------------------------------------
# 7. Implementation
# ---------------------------------------------------------------------------------------------
st.subheader('Implementation: turnover, costs and leverage')
with st.container(border=True):
    i1, i2, i3, i4 = st.columns(4)
    i1.metric('Annual turnover', f"{imp['turnover']:.2f}x capital")
    i2.metric('Annual transaction cost', pct(imp['tc_cost'], 2))
    i3.metric('Annual funding cost', pct(imp['funding_cost'], 2))
    i4.metric('Avg. gross exposure in risk parity',
              f"{imp['gross_exposure_rp']:.2f}x" if not np.isnan(imp['gross_exposure_rp']) else '–')
    st.caption(f"Turnover = |ΔW_SPY| + |ΔW_IEI| per day, times 252. The leverage cap binds on "
               f"{pct(imp['cap_share'])} of all days (risk parity weights scaled down to "
               f"{params['max_lev']:.1f}x).")

# ---------------------------------------------------------------------------------------------
# 8. Threshold sensitivity
# ---------------------------------------------------------------------------------------------
st.subheader('Threshold sensitivity')
sens = s.threshold_sensitivity(data, **params)
st.dataframe(
    sens.set_index('Threshold').style.format({'Sharpe (net)': '{:.2f}', 'Avg. annual return (net)': '{:.2%}',
                                              'Max drawdown (net)': '{:.2%}', 'Days in risk parity': '{:.1%}'})
        .format_index('{:.2f}'),
    width='stretch')
st.caption('Each row re-runs the backtest with a different correlation threshold; all other inputs are '
           'as set in the sidebar. The threshold was chosen on the same data, so treat the default row with '
           'caution (data snooping).')

# ---------------------------------------------------------------------------------------------
# 9. Footer
# ---------------------------------------------------------------------------------------------
st.divider()
st.caption(
    'Data: prices from Yahoo Finance via the `yfinance` package (SPY, IEI, adjusted for dividends and splits); '
    '1-month Treasury bill yield from FRED, series DGS1MO.  \n'
    '**For educational purposes only, not investment advice.** Past performance does not guarantee future results.  \n'
    f'Source code: [GitHub repository]({GITHUB_URL}) · Nova SBE, 2270 Financial Modeling, group assignment.'
)
