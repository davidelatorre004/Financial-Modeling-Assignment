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
    st.slider('Correlation threshold', -0.6, 0.6, step=0.05, key='threshold',
              help='Switch to risk parity when the rolling SPY-IEI correlation is at or below this value.')
    st.slider('Correlation window (days)', 60, 250, step=1, key='corr_window')
    st.slider('Volatility window (days)', 10, 60, step=1, key='vol_window')
    st.slider('Target volatility per asset (%)', 3.0, 100.0, step=0.5, key='target_vol_pct',
              help='Ex-ante volatility cap for each asset in risk parity. High values push the weights '
                   'up until the maximum gross leverage binds.')
    st.slider('Maximum gross leverage (x)', 1.0, 10.0, step=0.1, key='max_lev')
    st.slider('Transaction cost (bp per unit traded)', 0.0, 30.0, step=0.5, key='tc_bp')
    st.slider('Funding spread over T-bill (%)', 0.0, 2.0, step=0.1, key='spread_pct')
    st.slider('Static portfolio: equity share (%)', 0.0, 100.0, step=5.0, key='equity_pct',
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

# Name of the static benchmark, bonds/equity as in the notebook ("60/40" = 60% IEI, 40% SPY).
# It follows the equity share chosen in the sidebar, e.g. 30% equity -> "70/30".
EQ_PCT = st.session_state['equity_pct']
BENCH = f"{100 - EQ_PCT:.0f}/{EQ_PCT:.0f}"


# ---------------------------------------------------------------------------------------------
# Data (cached for one day) and backtest
# ---------------------------------------------------------------------------------------------
@st.cache_data(ttl=24 * 3600, show_spinner='Downloading prices (Yahoo Finance) and T-bill rate (FRED)...')
def load_data(start, end):
    """Download prices + T-bill once per (start, end) pair and keep them for a day."""
    return s.download_data(start, end)


st.title('Correlation-Switch Risk Parity: SPY & IEI')
st.markdown(
    f"By default the portfolio is a **static {BENCH}** ({100 - EQ_PCT:.0f}% IEI Treasuries 3–7y, "
    f"{EQ_PCT:.0f}% SPY S&P 500), rebalanced daily.  \n"
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


def shade_rp_periods(ax):
    """Shade the risk parity periods (light blue bands) on a time-series chart."""
    for first, last in s.rp_periods(df):
        i_last = df.index.get_loc(last)
        right = df.index[min(i_last + 1, len(df) - 1)]      # extend to the next day so 1-day periods show
        ax.axvspan(first, right, color=C_BAND, linewidth=0)


def show(fig):
    """Draw a matplotlib figure in the page and free its memory."""
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# 2. KPI row (always visible, above the tabs): strategy (net) vs the static benchmark.
#    The small coloured number is the difference from the benchmark: green = better, red = worse.
# ---------------------------------------------------------------------------------------------
d_final = res['final_net'] - res['final_6040']
d_sharpe = res['sharpe_net'] - res['sharpe_6040']
d_mdd = 100 * (res['mdd_net'] - res['mdd_6040'])           # in percentage points; > 0 = shallower drawdown

k1, k2, k3, k4 = st.columns(4)
k1.metric('Final value of $1 (net)', f"${res['final_net']:.2f}",
          f"{'+' if d_final >= 0 else '-'}${abs(d_final):.2f}",
          delta_description=f"vs {BENCH} (${res['final_6040']:.2f})", border=True)
k2.metric('Sharpe ratio (net)', f"{res['sharpe_net']:.2f}", f"{d_sharpe:+.2f}",
          delta_description=f"vs {BENCH} ({res['sharpe_6040']:.2f})", border=True)
k3.metric('Max drawdown (net)', pct(res['mdd_net']), f"{d_mdd:+.1f} pp",
          delta_description=f"vs {BENCH} ({pct(res['mdd_6040'])})", border=True,
          help=f'Positive difference = the strategy lost less than the static {BENCH} at its worst point.')
k4.metric('Days in risk parity', pct(res['share_rp']), border=True,
          help=f'Share of trading days in the risk parity regime. The static {BENCH} never switches.')

# ---------------------------------------------------------------------------------------------
# Tabs: one topic per tab instead of one very long page
# ---------------------------------------------------------------------------------------------
tab_overview, tab_perf, tab_dd, tab_regime, tab_impl = st.tabs(
    ['Overview', 'Performance', 'Drawdowns', 'Regimes & weights', 'Implementation & sensitivity'])

# ---- Overview: cumulative return -----------------------------------------------------------
with tab_overview:
    st.subheader('Cumulative return (base = 1)')
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(df['cum_net'], color=C_STRAT, linewidth=2, label='Strategy (net)')
    ax.plot(df['cum_gross'], '--', color=C_BLUE_DARK, linewidth=1.5,
            label='Strategy (gross of transaction costs)')
    ax.plot(df['cum_6040'], color=C_BLUE_MID, linewidth=1.5, label=f'Static {BENCH}')
    style_axes(ax, 'Value of $1 invested')
    ax.legend(frameon=False, loc='upper left')
    show(fig)
    st.caption('All series start at 1 on the first day of the backtest. "Gross" excludes transaction costs '
               'but includes funding costs; "net" includes both.')

    st.subheader('Drawdown from previous peak')
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(100 * df['drawdown_6040'], color=C_BLUE_MID, linewidth=1.5, label=f'Static {BENCH}')
    ax.plot(100 * df['drawdown_net'], color=C_STRAT, linewidth=1.5, label='Strategy (net)')
    style_axes(ax, 'Drawdown (%)')
    ax.legend(frameon=False, loc='lower right')             # bottom right is clear of the 2008-09 trough
    show(fig)

# ---- Performance table -------------------------------------------------------------------
with tab_perf:
    st.subheader(f'Performance measures (benchmark: static {BENCH})')
    perf = s.performance_table(df).rename(index={'Static 60/40': f'Static {BENCH}'})
    perf_display = pd.DataFrame(index=perf.index)           # formatted copy for display (strings)
    for col in ['Avg. annual return', 'Volatility', 'Alpha (annual)']:
        perf_display[col] = [pct(x, 2) for x in perf[col]]
    for col in ['Sharpe', 'Beta', 'Information ratio']:
        perf_display[col] = ['–' if np.isnan(x) else f'{x:.2f}' for x in perf[col]]   # IR not defined for 60/40
    perf_display = perf_display[perf.columns]               # original column order
    st.dataframe(perf_display, width='stretch')
    st.caption('Daily simple returns, 252 days a year. Sharpe uses excess returns over the 1-month T-bill. '
               f'Beta is the regression slope on the {BENCH} excess return (not annualised); alpha is annualised. '
               f'Information ratio = annualised mean of (strategy − {BENCH}) / annualised tracking error. '
               f'Benchmark name = % IEI / % SPY.')

# ---- Drawdowns ---------------------------------------------------------------------------
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
        'Days': [str(int(x)) for x in top['days']],         # trading days, peak to recovery
    }, index=[f'{i + 1}' for i in range(len(top))])
    table.index.name = 'Rank'
    highlight = lambda row: (['background-color: aliceblue; font-weight: bold'] * len(row)
                             if row.name == '1' else [''] * len(row))
    return table.style.apply(highlight, axis=1)


with tab_dd:
    st.subheader('Worst drawdowns')
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('**Strategy (net): 5 worst drawdowns**')
        st.dataframe(drawdown_table(df['drawdown_net']), width='stretch')
    with c2:
        st.markdown(f'**Static {BENCH}: 5 worst drawdowns**')
        st.dataframe(drawdown_table(df['drawdown_6040']), width='stretch')
    st.caption('An episode runs from the peak to full recovery of that peak (or to the end of the sample). '
               'Days = trading days from the peak to the recovery (or to the end of the sample).')

# ---- Regimes: correlation signal and portfolio weights -------------------------------------
with tab_regime:
    st.subheader(f"Rolling {params['corr_window']}-day SPY–IEI correlation and risk parity periods")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    shade_rp_periods(ax)
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
    show(fig)

    # Regime statistics (no deltas: there is nothing to compare them with)
    r1, r2, r3, r4 = st.columns(4)
    r1.metric('Days in risk parity', f"{reg['days_rp']:,} days", border=True,
              help=f"{reg['days_rp']:,} of {reg['days_total']:,} trading days ({pct(reg['share_rp'])}).")
    r2.metric('Risk parity episodes', f"{reg['n_episodes']}", border=True)
    r3.metric('Average episode', f"{reg['avg_episode']:.0f} days", border=True)
    r4.metric('Longest episode', f"{reg['longest_episode']} days", border=True,
              help=(f"Started on {reg['longest_start']:%d %b %Y}." if reg['longest_start'] is not None
                    else 'The strategy never entered risk parity.'))

    # Portfolio weights over time: shows when (and how much) leverage is used
    st.subheader('Portfolio weights and gross exposure')
    gross_exposure = df['w_spy'] + df['w_iei']
    fig, ax = plt.subplots(figsize=(11, 4.5))
    shade_rp_periods(ax)
    ax.axhline(1, color=C_BLUE_LIGHT, linewidth=0.8)        # 100% of capital: above this line = leverage
    ax.plot(gross_exposure, color='midnightblue', linewidth=1.2, label='Gross exposure (SPY + IEI)')
    ax.plot(df['w_iei'], color=C_BLUE_MID, linewidth=1.2, label='IEI weight (bonds)')
    ax.plot(df['w_spy'], color=C_BLUE_DARK, linewidth=1.2, linestyle=':', label='SPY weight (equity)')
    style_axes(ax, 'Weight (× capital)')
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [Patch(facecolor=C_BAND, edgecolor=C_BLUE_LIGHT, label='In risk parity')],
              frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4)   # below the plot
    ax.set_ylim(0, max(params['max_lev'], gross_exposure.max()) * 1.05)
    show(fig)
    st.caption(f"Weights decided at each day's close (they earn the next day's return). In the static regime "
               f"the weights are fixed at {100 * params['w_spy_static']:.0f}% SPY / "
               f"{100 * params['w_iei_static']:.0f}% IEI; in risk parity they follow volatility, and gross "
               f"exposure above 1 (the thin line) is borrowed at the T-bill + spread, up to "
               f"{params['max_lev']:.1f}×.")

# ---- Implementation and threshold sensitivity --------------------------------------------
with tab_impl:
    st.subheader('Implementation: turnover, costs and leverage')
    i1, i2, i3, i4 = st.columns(4)
    i1.metric('Annual turnover', f"{imp['turnover']:.2f}× capital", border=True)
    i2.metric('Annual transaction cost', pct(imp['tc_cost'], 2), border=True)
    i3.metric('Annual funding cost', pct(imp['funding_cost'], 2), border=True)
    i4.metric('Avg. gross exposure in risk parity',
              f"{imp['gross_exposure_rp']:.2f}×" if not np.isnan(imp['gross_exposure_rp']) else '–',
              border=True)
    st.caption(f"Turnover = |ΔW_SPY| + |ΔW_IEI| per day, times 252. The leverage cap binds on "
               f"{pct(imp['cap_share'])} of all days (risk parity weights scaled down to "
               f"{params['max_lev']:.1f}×).")

    st.subheader('Threshold sensitivity')
    # Standard grid, plus the threshold chosen in the sidebar if it is not already in it
    thresholds = [-0.45, -0.35, -0.25, -0.15, -0.05]
    if not any(np.isclose(params['threshold'], t) for t in thresholds):
        thresholds = sorted(thresholds + [round(params['threshold'], 2)])
    sens = s.threshold_sensitivity(data, thresholds=thresholds, **params).set_index('Threshold')

    def highlight_current(row):
        """Highlight the row of the threshold currently selected in the sidebar."""
        current = np.isclose(row.name, params['threshold'])
        return ['background-color: aliceblue; font-weight: bold' if current else ''] * len(row)

    st.dataframe(
        sens.style.apply(highlight_current, axis=1)
            .format({'Sharpe (net)': '{:.2f}', 'Avg. annual return (net)': '{:.2%}',
                     'Max drawdown (net)': '{:.2%}', 'Days in risk parity': '{:.1%}'})
            .format_index('{:.2f}'),
        width='stretch')
    st.caption('Each row re-runs the backtest with a different correlation threshold; all other inputs are '
               'as set in the sidebar. The highlighted row is the threshold currently selected. The default '
               'threshold was chosen on the same data, so treat it with caution (data snooping).')

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
