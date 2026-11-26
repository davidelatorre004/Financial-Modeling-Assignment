"""
app.py - Streamlit interface for the "60/40 -> risk parity correlation switch" strategy.

All calculations live in strategy.py (pure functions, no Streamlit). This file only:
  1. reads the parameters from the sidebar,
  2. downloads the data (cached for one day) and runs the backtest,
  3. shows KPIs, charts and tables.

Run locally with:  streamlit run app.py
"""

import datetime as dt
import importlib
import os

import matplotlib
matplotlib.use('Agg')                                   # non-interactive backend (needed on servers)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import streamlit as st

import strategy as s

# After a redeploy, Streamlit Community Cloud reruns app.py with the new code but can keep the OLD
# strategy.py in memory (modules are imported once per server process). Reload strategy.py whenever
# the file on disk has changed, so app.py and strategy.py are always the same version.
_strategy_mtime = os.path.getmtime(s.__file__)
if getattr(s, '_loaded_mtime', None) != _strategy_mtime:
    s = importlib.reload(s)
    s._loaded_mtime = _strategy_mtime

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
# Small CSS touches (layout only; the matplotlib charts are not affected)
#   - mini charts inside the KPI cards get some space from the card border, like the large charts
#   - the tab bar becomes a row of light-blue "pills"; the selected tab is dark blue with white text
#   - a small pulsing coral dot marks the Monte Carlo tab (6th tab of the main tab bar, key 'main_tabs')
# ---------------------------------------------------------------------------------------------
st.markdown("""
<style>
/* KPI cards: leave a margin around the mini chart */
[data-testid="stMetricChart"] {
    margin: 4px 16px 16px 16px;
    width: calc(100% - 32px) !important;
}
[data-testid="stMetricChart"] svg {     /* scale the chart to the narrower box instead of overflowing */
    width: 100% !important;
    height: auto !important;
}
/* Tab bar: pills instead of an underline */
[data-testid="stTabs"] [role="tablist"] {
    gap: 8px;
}
[data-testid="stTabs"] [role="tablist"]::after,                 /* grey line under the tab bar */
[data-testid="stTabs"] .react-aria-SelectionIndicator {         /* underline of the selected tab */
    display: none;
}
[data-testid="stTab"] {
    background-color: #E3EEF9;          /* light blue */
    border-radius: 8px;
    padding: 6px 16px;
    transition: background-color 0.15s;
}
[data-testid="stTab"]:hover {
    background-color: #CFE2F3;          /* a little darker on hover */
}
[data-testid="stTab"][aria-selected="true"] {
    background-color: #1F4E79;          /* dark blue */
}
[data-testid="stTab"][aria-selected="true"] p {
    color: #FFFFFF;                     /* white text on the selected tab */
}
/* Monte Carlo tab: pulsing coral dot after the label. ".st-key-main_tabs" is the class Streamlit
   gives to st.tabs(..., key='main_tabs'), so no other tab bar is affected. */
.st-key-main_tabs [data-testid="stTab"]:nth-of-type(6)::after {
    content: "";
    display: inline-block;
    width: 8px;
    height: 8px;
    margin-left: 8px;
    border-radius: 50%;
    background-color: coral;
    vertical-align: middle;
    animation: mc-pulse 1.6s ease-out infinite;
}
@keyframes mc-pulse {                   /* a coral ring that grows and fades, then repeats */
    0%   { box-shadow: 0 0 0 0 rgba(255, 127, 80, 0.7); }
    70%  { box-shadow: 0 0 0 7px rgba(255, 127, 80, 0); }
    100% { box-shadow: 0 0 0 0 rgba(255, 127, 80, 0); }
}
</style>
""", unsafe_allow_html=True)

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

# Labels of the parameters, used to list the ones changed from their default value
PARAM_LABELS = {
    'threshold': 'Correlation threshold', 'corr_window': 'Correlation window',
    'vol_window': 'Volatility window', 'target_vol_pct': 'Target volatility',
    'max_lev': 'Maximum leverage', 'tc_bp': 'Transaction cost', 'spread_pct': 'Funding spread',
    'equity_pct': 'Equity share', 'start': 'Start date', 'end': 'End date',
}


def is_changed(key):
    """True if the widget value differs from its default (floats compared with a tolerance)."""
    value, default = st.session_state[key], WIDGET_DEFAULTS[key]
    if isinstance(default, (int, float)):
        return not np.isclose(value, default)
    return value != default


def section(title, icon, keys):
    """Collapsible sidebar section; its title shows how many of its parameters were changed."""
    n = sum(is_changed(k) for k in keys)
    label = f'{title}  ·  {n} changed' if n else title
    return st.expander(label, expanded=True, icon=icon)


with st.sidebar:
    st.header('Strategy parameters')
    changed = [PARAM_LABELS[k] for k in WIDGET_DEFAULTS if is_changed(k)]
    if changed:                                            # summary of what is being tested
        st.markdown(' '.join(f':orange-badge[{name}]' for name in changed))
        st.caption('Changed from the notebook defaults.')
    else:
        st.markdown(':green-badge[:material/check: Notebook defaults]')

    with section('Signal', ':material/sensors:', ['threshold', 'corr_window']):
        st.slider('Correlation threshold', -0.6, 0.6, step=0.05, key='threshold',
                  help='Switch to risk parity when the rolling SPY-IEI correlation is at or below this value.')
        st.slider('Correlation window (days)', 60, 250, step=1, key='corr_window')
    with section('Risk parity', ':material/balance:', ['vol_window', 'target_vol_pct', 'max_lev']):
        st.slider('Volatility window (days)', 10, 60, step=1, key='vol_window')
        st.slider('Target volatility per asset (%)', 3.0, 100.0, step=0.5, key='target_vol_pct',
                  help='Ex-ante volatility cap for each asset in risk parity. High values push the weights '
                       'up until the maximum gross leverage binds.')
        st.slider('Maximum gross leverage (x)', 1.0, 10.0, step=0.1, key='max_lev')
    with section('Costs', ':material/payments:', ['tc_bp', 'spread_pct']):
        st.slider('Transaction cost (bp per unit traded)', 0.0, 30.0, step=0.5, key='tc_bp')
        st.slider('Funding spread over T-bill (%)', 0.0, 2.0, step=0.1, key='spread_pct')
    with section('Benchmark', ':material/pie_chart:', ['equity_pct']):
        st.slider('Static portfolio: equity share (%)', 0.0, 100.0, step=5.0, key='equity_pct',
                  help='SPY weight of the static benchmark; bonds (IEI) get the rest.')
        st.caption(f"Static weights: {st.session_state['equity_pct']:.0f}% SPY / "
                   f"{100 - st.session_state['equity_pct']:.0f}% IEI")
    with section('Sample', ':material/date_range:', ['start', 'end']):
        st.date_input('Start date', min_value=dt.date(2007, 1, 1), max_value=dt.date.today(), key='start')
        st.date_input('End date', min_value=dt.date(2007, 1, 1), max_value=dt.date.today(), key='end')
    st.button('Reset to defaults', on_click=reset_defaults, width='stretch', icon=':material/restart_alt:')

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


st.title('Correlation-Switch Risk Parity')
st.markdown(f'A static **{BENCH}** portfolio of SPY and IEI that switches to levered **risk parity** '
            f'when stocks and bonds hedge each other.')

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

# Badges: assets, sample actually used, course
st.markdown(
    ':blue-badge[:material/show_chart: SPY · IEI] '
    f":blue-badge[:material/calendar_month: {df.index[0]:%d %b %Y} – {df.index[-1]:%d %b %Y}] "
    f':blue-badge[{len(df):,} trading days] '
    ':gray-badge[:material/school: Nova SBE · 2270 Financial Modeling]'
)
with st.expander('How the strategy works', icon=':material/info:'):
    st.markdown(
        f"By default the portfolio is a **static {BENCH}** ({100 - EQ_PCT:.0f}% IEI Treasuries 3–7y, "
        f"{EQ_PCT:.0f}% SPY S&P 500), rebalanced daily.  \n"
        "When the rolling stock–bond correlation falls to the **threshold or below**, bonds are a reliable hedge "
        "and it switches to **risk parity**: each asset is sized to a target volatility, with leverage up to a "
        "cap.  \n"
        "Leverage is funded at the 1-month T-bill plus a spread, and every trade pays a transaction cost."
    )
    st.caption(f"The backtest starts on {df.index[0]:%d %b %Y} because the first {params['corr_window']} "
               f"days of data are used to estimate the first correlation.")


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

# Small series drawn inside the cards (display only, monthly or yearly so they stay light)
spark_value = df['cum_net'].resample('ME').last().round(3).tolist()          # value of $1, month end
ex_net = df['r_net'] - df['rf']
rolling_sharpe = (np.sqrt(252) * ex_net.rolling(252).mean() / ex_net.rolling(252).std()).dropna()
spark_sharpe = rolling_sharpe.resample('ME').last().dropna().round(2).tolist()   # 1-year rolling Sharpe
spark_dd = (100 * df['drawdown_net']).resample('ME').min().round(2).tolist()     # worst drawdown each month
spark_rp = (100 * df['rp_on'].resample('YE').mean()).round(1).tolist()           # % of days in RP, by year

k1, k2, k3, k4 = st.columns(4)
k1.metric('Final value of $1 (net)', f"${res['final_net']:.2f}",
          f"{'+' if d_final >= 0 else '-'}${abs(d_final):.2f}",
          delta_description=f"vs {BENCH} (${res['final_6040']:.2f})", border=True,
          chart_data=spark_value, chart_type='area', height='stretch',
          help='Mini chart: value of $1 in the strategy (net), month by month.')
k2.metric('Sharpe ratio (net)', f"{res['sharpe_net']:.2f}", f"{d_sharpe:+.2f}",
          delta_description=f"vs {BENCH} ({res['sharpe_6040']:.2f})", border=True,
          chart_data=spark_sharpe if spark_sharpe else None, chart_type='line', height='stretch',
          help='Mini chart: 1-year rolling Sharpe ratio of the strategy (net).')
k3.metric('Max drawdown (net)', pct(res['mdd_net']), f"{d_mdd:+.1f} pp",
          delta_description=f"vs {BENCH} ({pct(res['mdd_6040'])})", border=True,
          chart_data=spark_dd, chart_type='area', height='stretch',
          help=f'Positive difference = the strategy lost less than the static {BENCH} at its worst point. '
               'Mini chart: drawdown of the strategy (net), month by month.')
k4.metric('Days in risk parity', pct(res['share_rp']), border=True,
          chart_data=spark_rp, chart_type='bar', height='stretch',
          help=f'Share of trading days in the risk parity regime. The static {BENCH} never switches. '
               'Mini chart: share of days in risk parity, year by year.')

# ---------------------------------------------------------------------------------------------
# Tabs: one topic per tab instead of one very long page
# ---------------------------------------------------------------------------------------------
tab_overview, tab_perf, tab_dd, tab_regime, tab_impl, tab_mc = st.tabs(
    ['Overview', 'Performance', 'Drawdowns', 'Regimes & weights', 'Implementation & sensitivity',
     'Monte Carlo'],
    key='main_tabs')                                       # key -> CSS class used for the pulsing dot

# ---- Overview: cumulative return -----------------------------------------------------------
with tab_overview:
    with st.container(border=True):                     # card around the chart
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

    with st.container(border=True):                     # card around the chart
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
    """The 5 worst episodes, formatted for display, with the worst one highlighted.

    Depth is drawn as a bar (longer bar = deeper loss) by the column configuration in show_drawdowns().
    """
    top = s.worst_drawdowns(drawdown, 5)
    table = pd.DataFrame({
        # plain lists (not Series), so pandas does not try to align them with the new index
        'Depth': [-100 * x for x in top['depth']],          # size of the loss in %, shown as a bar
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


# Same bar scale for both tables, so the two portfolios can be compared at a glance
DD_SCALE = -100 * min(res['mdd_net'], res['mdd_6040'])


def show_drawdowns(drawdown):
    """Draw a drawdown table with the depth as a red bar labelled '-x.xx%'."""
    st.dataframe(drawdown_table(drawdown), width='stretch', column_config={
        'Depth': st.column_config.ProgressColumn('Depth', format='-%.2f%%', min_value=0.0,
                                                 max_value=DD_SCALE, color='red', width='large'),
    })


with tab_dd:
    st.subheader('Worst drawdowns')
    # One table above the other (full width), so the depth bars and all the dates fit
    st.markdown('**Strategy (net): 5 worst drawdowns**')
    show_drawdowns(df['drawdown_net'])
    st.markdown(f'**Static {BENCH}: 5 worst drawdowns**')
    show_drawdowns(df['drawdown_6040'])
    st.caption('An episode runs from the peak to full recovery of that peak (or to the end of the sample). '
               'Days = trading days from the peak to the recovery (or to the end of the sample).')

# ---- Regimes: correlation signal and portfolio weights -------------------------------------
with tab_regime:
    with st.container(border=True):                     # card around the chart
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
    with st.container(border=True):                     # card around the chart
        st.subheader('Portfolio weights and gross exposure')
        # Stacked areas: SPY at the bottom, IEI on top, so the top edge is the gross exposure
        # (SPY + IEI). Everything above the dashed line at 1 is borrowed (leverage).
        gross_exposure = df['w_spy'] + df['w_iei']
        fig, ax = plt.subplots(figsize=(11, 4.5))
        shade_rp_periods(ax)
        ax.stackplot(df.index, df['w_spy'], df['w_iei'], colors=[C_BLUE_DARK, C_BLUE_LIGHT],
                     labels=['SPY weight (equity)', 'IEI weight (bonds)'], linewidth=0)
        ax.axhline(1, color='midnightblue', linestyle='--', linewidth=1.2,
                   label='Capital (1×): above = leverage')
        style_axes(ax, 'Weight (× capital)')
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=handles + [Patch(facecolor=C_BAND, edgecolor=C_BLUE_LIGHT, label='In risk parity')],
                  frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4)   # below the plot
        ax.set_ylim(0, max(params['max_lev'], gross_exposure.max()) * 1.05)
        show(fig)
        st.caption(f"Weights decided at each day's close (they earn the next day's return). The top edge of the "
                   f"two areas is the gross exposure (SPY + IEI). In the static regime the weights are fixed at "
                   f"{100 * params['w_spy_static']:.0f}% SPY / {100 * params['w_iei_static']:.0f}% IEI; in risk "
                   f"parity they follow volatility, and exposure above the dashed line at 1 is borrowed at the "
                   f"T-bill + spread, up to {params['max_lev']:.1f}×.")

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

    sharpe_lo = min(0.0, sens['Sharpe (net)'].min())       # bars start at 0 unless a Sharpe is negative
    st.dataframe(
        sens.style.apply(highlight_current, axis=1)
            .format({'Avg. annual return (net)': '{:.2%}', 'Max drawdown (net)': '{:.2%}'})
            .format_index('{:.2f}'),
        width='stretch',
        column_config={                                    # Sharpe and time in risk parity as bars
            'Sharpe (net)': st.column_config.ProgressColumn(
                'Sharpe (net)', format='%.2f', min_value=sharpe_lo,
                max_value=max(sens['Sharpe (net)'].max(), sharpe_lo + 0.01)),
            'Days in risk parity': st.column_config.ProgressColumn(
                'Days in risk parity', format='percent', min_value=0.0, max_value=1.0),
        })
    st.caption('Each row re-runs the backtest with a different correlation threshold; all other inputs are '
               'as set in the sidebar. The highlighted row is the threshold currently selected. The default '
               'threshold was chosen on the same data, so treat it with caution (data snooping).')


# ---- Monte Carlo simulation of the strategy (net returns) ---------------------------------
# Bootstrap of the strategy's monthly NET returns, from the backtest currently configured in the
# sidebar. The calculation (monthly_returns, monte_carlo) is in strategy.py; here only UI and chart.

# Calibration scenarios: which historical months go into the pool that is resampled
MC_SCENARIOS = {
    'Full history': (None, None),                          # all complete months
    'Prudent (2021 onwards)': ('2021-01-01', None),        # stock-bond correlation turned positive
    'Favourable (up to 2020)': (None, '2020-12-31'),       # mostly in the risk parity regime
}


@st.cache_data(show_spinner='Running the Monte Carlo simulation...')
def run_monte_carlo(pool, n_sim, n_months):
    """Simulate and keep only what the tab shows (cached on all inputs: pool, n_sim, n_months).

    The full (n_sim x n_months+1) matrix is not stored: only the percentile bands per month,
    50 sample paths and the statistics of the final value.
    """
    paths = s.monte_carlo(pool, n_sim, n_months)
    return {
        'summary': s.monte_carlo_summary(paths),
        'bands': np.percentile(paths, [5, 25, 50, 75, 95], axis=0),   # one row per percentile
        'sample_paths': paths[:50],                                    # 50 paths to draw
    }


with tab_mc:
    st.subheader('Monte Carlo simulation of the strategy')
    st.markdown('What could $1 invested in the strategy (net of all costs) become? Each scenario is built '
                'by drawing historical **monthly** returns of the strategy at random, with replacement. '
                'The returns come from the backtest as currently set in the sidebar.')

    # Inputs (inside the tab, above the chart)
    m1, m2 = st.columns(2)
    m1.slider('Horizon (months)', 12, 240, value=120, step=12, key='mc_horizon')
    m2.slider('Number of simulations', 1000, 10000, value=10000, step=1000, key='mc_n_sim')
    st.radio('Calibration scenario', list(MC_SCENARIOS), horizontal=True, key='mc_scenario',
             help='Which historical months are resampled. "Prudent" uses only months from January 2021, '
                  'when the stock–bond correlation turned positive; "Favourable" only months up to '
                  'December 2020.')
    n_months, n_sim = st.session_state['mc_horizon'], st.session_state['mc_n_sim']
    scenario = st.session_state['mc_scenario']

    # Pool of historical monthly net returns for the chosen scenario
    ret_monthly = s.monthly_returns(df['r_net'])            # complete months of the current backtest
    first, last = MC_SCENARIOS[scenario]
    pool = ret_monthly
    if first is not None:
        pool = pool[pool.index >= first]
    if last is not None:
        pool = pool[pool.index <= last]

    if len(pool) > 0:
        st.caption(f'The pool contains **{len(pool)} complete months** of net returns '
                   f'({pool.index[0]:%b %Y} – {pool.index[-1]:%b %Y}).')
    if len(pool) == 0:
        st.error('The selected scenario contains no complete month in the current sample. '
                 'Choose another scenario or change the sample dates in the sidebar.')
    elif len(pool) < 24:
        st.warning(f'Only {len(pool)} historical months in the pool (fewer than 24): the simulation '
                   'resamples very few months, so its results are not reliable.')

    # Everything the results depend on: if it changes after a run, the results are out of date
    mc_inputs = (tuple(sorted(params.items())), str(start), str(end), scenario, n_sim, n_months)

    if st.button('Run simulation', type='primary', icon=':material/casino:', disabled=len(pool) == 0):
        st.session_state['mc_result'] = {                   # kept when switching tabs
            'inputs': mc_inputs, 'scenario': scenario, 'n_sim': n_sim, 'n_months': n_months,
            'pool_size': len(pool), **run_monte_carlo(pool, n_sim, n_months),
        }

    if 'mc_result' not in st.session_state:
        st.info('Choose the inputs and click **Run simulation**.', icon=':material/play_circle:')
    else:
        mc_res = st.session_state['mc_result']
        if mc_res['inputs'] != mc_inputs:
            st.caption(':orange[:material/update: The results below are out of date: the inputs changed '
                       'since the last run. Click **Run simulation** to update them.]')
        summ, H = mc_res['summary'], mc_res['n_months']

        # KPI cards, same style as the cards at the top of the page
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Median final value', f"${summ['median']:.2f}", border=True,
                  help=f'Value of $1 after {H} months: half of the simulated scenarios end above it, '
                       'half below.')
        c2.metric('5th percentile', f"${summ['p5']:.2f}", border=True,
                  help='Bad case: only 5% of the scenarios end below this value.')
        c3.metric('95th percentile', f"${summ['p95']:.2f}", border=True,
                  help='Good case: only 5% of the scenarios end above this value.')
        c4.metric('Probability of loss', pct(summ['prob_loss'], 2), border=True,
                  help='Share of scenarios that end below the initial $1.')

        # Fan chart (matplotlib, like the other charts)
        with st.container(border=True):                     # card around the chart
            months = np.arange(H + 1)
            p5, p25, p50, p75, p95 = mc_res['bands']
            fig, ax = plt.subplots(figsize=(11, 5))
            for path in mc_res['sample_paths']:             # 50 individual scenarios
                ax.plot(months, path, color=C_BLUE_LIGHT, linewidth=0.6)
            ax.fill_between(months, p5, p95, color=C_STRAT, alpha=0.15, linewidth=0)
            ax.fill_between(months, p25, p75, color=C_STRAT, alpha=0.3, linewidth=0)
            ax.plot(months, p50, color=C_STRAT, linewidth=2.5)
            ax.axhline(1, color='black', linestyle=':', linewidth=1.2)
            ax.set_title(f"Monte Carlo simulation of the strategy: {mc_res['n_sim']:,} scenarios "
                         f"over {H} months (50 shown)")
            ax.set_xlabel('Month')
            ax.set_ylabel('Value of $1 invested')
            ax.set_xlim(0, H)
            ax.grid(True, color='#e6e6e6', linewidth=0.8)   # same quiet grid and frame as style_axes()
            ax.set_axisbelow(True)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            ax.legend(handles=[
                plt.Line2D([], [], color=C_BLUE_LIGHT, linewidth=1, label='Simulated paths (50 shown)'),
                Patch(facecolor=C_STRAT, alpha=0.15, label='5th–95th percentile'),
                Patch(facecolor=C_STRAT, alpha=0.3, label='25th–75th percentile'),
                plt.Line2D([], [], color=C_STRAT, linewidth=2.5, label='Median'),
                plt.Line2D([], [], color='black', linestyle=':', linewidth=1.2, label='Initial capital'),
            ], frameon=False, loc='upper left')
            show(fig)
            st.caption(f"**How to read this.** Each thin line is one possible path of $1 over {H} months, "
                       f"built from {mc_res['pool_size']} historical months ({mc_res['scenario']}). "
                       "The thick coral line is the median scenario; the darker band contains the middle "
                       "50% of the scenarios and the lighter band 90%. Scenarios that finish below the "
                       "dotted line lose money.")

    # Limitations of the method
    st.info(
        '**Limitations**\n'
        '- Months are drawn independently, so multi-month crises and correlation regimes are not reproduced.\n'
        '- Outcomes can never be worse than the worst historical month.\n'
        '- The full-history pool is dominated by 2007–2020, when the risk parity regime was active, so '
        'results may be optimistic: compare with the Prudent scenario.\n'
        '- This is a historical simulation, not a forecast, and not investment advice.',
        icon=':material/warning:')

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
