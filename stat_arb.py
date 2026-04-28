"""
BTC/USD – ETH/USD Statistical Arbitrage
========================================
Step 1 : Load & align tick data, run Engle-Granger cointegration test
Step 2 : Estimate hedge ratio via OLS regression
Step 3 : Compute spread, fit Ornstein-Uhlenbeck process
Step 4 : Generate entry (±2σ) and exit (mean reversion) signals
Step 5 : Backtest with 0.1% round-trip costs, report metrics, plot equity curve

Output : stat_arb_equity.png
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

# ── 1. Load & align ───────────────────────────────────────────────────────────

print("=" * 60)
print("  BTC/USD — ETH/USD Statistical Arbitrage")
print("=" * 60)

print("\n[1/5] Loading and aligning data...")

btc = pd.read_csv("btc_data.csv")
eth = pd.read_csv("eth_data.csv")

# Convert ms timestamps to UTC datetime and use as index
btc["dt"] = pd.to_datetime(btc["timestamp"], unit="ms", utc=True)
eth["dt"] = pd.to_datetime(eth["timestamp"], unit="ms", utc=True)

btc = btc.set_index("dt").sort_index()
eth = eth.set_index("dt").sort_index()

# Resample both to 1-minute bars (last mid_price in each minute)
btc_min = btc["mid_price"].resample("1min").last().dropna()
eth_min = eth["mid_price"].resample("1min").last().dropna()

# Inner join on common timestamps
prices = pd.DataFrame({"BTC": btc_min, "ETH": eth_min}).dropna()
n = len(prices)

print(f"  BTC raw rows : {len(btc):,}")
print(f"  ETH raw rows : {len(eth):,}")
print(f"  Aligned bars : {n:,} (1-min)")
print(f"  Date range   : {prices.index[0].strftime('%Y-%m-%d %H:%M')} "
      f"to {prices.index[-1].strftime('%Y-%m-%d %H:%M')} UTC")

# ── 2. Cointegration test ─────────────────────────────────────────────────────

print("\n[2/5] Engle-Granger cointegration test...")

# ADF on each series individually (should be I(1))
adf_btc = adfuller(np.log(prices["BTC"]), autolag="AIC")
adf_eth = adfuller(np.log(prices["ETH"]), autolag="AIC")

print(f"  ADF log(BTC): stat={adf_btc[0]:.4f}  p={adf_btc[1]:.4f}  "
      f"{'non-stationary (I(1)) OK' if adf_btc[1] > 0.05 else 'STATIONARY — check data'}")
print(f"  ADF log(ETH): stat={adf_eth[0]:.4f}  p={adf_eth[1]:.4f}  "
      f"{'non-stationary (I(1)) OK' if adf_eth[1] > 0.05 else 'STATIONARY — check data'}")

# Engle-Granger cointegration test
coint_t, coint_p, crit_vals = coint(np.log(prices["BTC"]), np.log(prices["ETH"]))
print(f"\n  Engle-Granger cointegration:")
print(f"    t-statistic : {coint_t:.4f}")
print(f"    p-value     : {coint_p:.4f}")
print(f"    Critical values: 1%={crit_vals[0]:.4f}  5%={crit_vals[1]:.4f}  10%={crit_vals[2]:.4f}")
cointegrated = coint_p < 0.05
print(f"    Result      : {'COINTEGRATED at 5% (proceed)' if cointegrated else 'NOT cointegrated at 5% — results may be spurious'}")

# ── 3. Hedge ratio via OLS ────────────────────────────────────────────────────

print("\n[3/5] OLS regression for hedge ratio...")

log_btc = np.log(prices["BTC"]).values
log_eth = np.log(prices["ETH"]).values

# Regress log(BTC) on log(ETH): log(BTC) = alpha + beta * log(ETH) + e
X   = add_constant(log_eth)
ols = OLS(log_btc, X).fit()
alpha_ols = ols.params[0]
beta       = ols.params[1]   # hedge ratio

print(f"  log(BTC) = {alpha_ols:.4f} + {beta:.4f} * log(ETH)")
print(f"  R²       = {ols.rsquared:.6f}")
print(f"  Hedge ratio (beta) = {beta:.6f}")

# ── 4. Spread & Ornstein-Uhlenbeck fit ───────────────────────────────────────

print("\n[4/5] Spread construction and OU process fitting...")

# Spread = log(BTC) - beta * log(ETH) - alpha
spread = log_btc - beta * log_eth - alpha_ols
prices["spread"] = spread

# ADF test on spread (should be stationary for cointegration to hold)
adf_spread = adfuller(spread, autolag="AIC")
print(f"  ADF on spread: stat={adf_spread[0]:.4f}  p={adf_spread[1]:.4f}  "
      f"{'stationary (good)' if adf_spread[1] < 0.05 else 'non-stationary (weak cointegration)'}")

# Ornstein-Uhlenbeck via AR(1) regression on spread differences
# dS_t = kappa*(mu - S_t)*dt + sigma*dW
# Discretised: S_t = a + b*S_{t-1} + e  =>  kappa = -ln(b)/dt, mu = a/(1-b)
spread_lag  = spread[:-1]
spread_diff = np.diff(spread)

ar_res   = stats.linregress(spread_lag, spread_diff)
b_ar     = ar_res.slope          # = -(1 - e^{-kappa*dt})
a_ar     = ar_res.intercept      # = mu*(1 - e^{-kappa*dt})
resid_std= ar_res.stderr

dt = 1 / (252 * 390)             # 1 minute as fraction of trading year
kappa    = -math.log(1 + b_ar) / dt   # mean-reversion speed (annualised)
mu_ou    = -a_ar / b_ar               # long-run mean
half_life_min = math.log(2) / (-b_ar) # in bars (minutes)
ou_sigma = np.std(spread_diff - (a_ar + b_ar * spread_lag))

print(f"\n  OU process parameters:")
print(f"    Mean-reversion speed (kappa) : {kappa:.2f}  (annualised)")
print(f"    Long-run mean (mu)           : {mu_ou:.6f}")
print(f"    Half-life                    : {half_life_min:.1f} minutes  "
      f"({half_life_min/60:.1f} hours)")
print(f"    OU sigma                     : {ou_sigma:.6f}")

# ── 5. Signals ────────────────────────────────────────────────────────────────

print("\n[5/5] Generating signals and backtesting...")

ENTRY_Z = 2.0
EXIT_Z  = 0.0
COST    = 0.001    # 0.1% per trade per leg

# Rolling z-score of spread (use expanding window up to 120 bars, then rolling 120)
ZSCORE_WINDOW = 120
spread_mean = pd.Series(spread).rolling(ZSCORE_WINDOW, min_periods=30).mean()
spread_std  = pd.Series(spread).rolling(ZSCORE_WINDOW, min_periods=30).std()
z_score     = (pd.Series(spread) - spread_mean) / spread_std
prices["z_score"] = z_score.values

# ── 6. Backtest ───────────────────────────────────────────────────────────────

# Position convention:
#   +1 = long spread  (long BTC, short ETH*beta)   — spread too low, expect reversion up
#   -1 = short spread (short BTC, long ETH*beta)   — spread too high, expect reversion down
#    0 = flat

btc_price = prices["BTC"].values
eth_price = prices["ETH"].values
z         = z_score.values
N         = len(prices)

position   = 0
equity     = 1.0
equity_curve = np.zeros(N)
n_trades   = 0
trade_log: list[dict] = []

# Track entry prices for cost accounting
entry_btc = 0.0
entry_eth = 0.0

for i in range(ZSCORE_WINDOW, N):
    z_i = z[i]
    if np.isnan(z_i):
        equity_curve[i] = equity
        continue

    prev_position = position

    # Entry signals
    if position == 0:
        if z_i < -ENTRY_Z:          # spread abnormally low → go long spread
            position = 1
        elif z_i > ENTRY_Z:         # spread abnormally high → go short spread
            position = -1

    # Exit signals
    elif position == 1 and z_i >= EXIT_Z:
        position = 0
    elif position == -1 and z_i <= EXIT_Z:
        position = 0

    # P&L: daily return of the spread position
    if i > 0 and position != 0:
        # Return of long-spread position = log(BTC[i]/BTC[i-1]) - beta*log(ETH[i]/ETH[i-1])
        r_btc = math.log(btc_price[i] / btc_price[i - 1])
        r_eth = math.log(eth_price[i] / eth_price[i - 1])
        bar_return = position * (r_btc - beta * r_eth)
        equity *= math.exp(bar_return)

    # Transaction costs on position changes
    if position != prev_position:
        if prev_position == 0:   # new entry: one round of costs
            equity *= (1 - COST)
            n_trades += 1
            entry_btc = btc_price[i]
            entry_eth = eth_price[i]
        elif position == 0:      # exit: one round of costs
            equity *= (1 - COST)
        else:                    # flip: two rounds (exit old + enter new)
            equity *= (1 - 2 * COST)
            n_trades += 2

    equity_curve[i] = equity

# Fill warmup section with 1.0
equity_curve[:ZSCORE_WINDOW] = 1.0

# ── Metrics ───────────────────────────────────────────────────────────────────

eq_series   = pd.Series(equity_curve[ZSCORE_WINDOW:])
eq_returns  = eq_series.pct_change().dropna()

total_return = equity_curve[-1] - 1.0
ann_factor   = 252 * 390          # trading minutes per year
sharpe       = (eq_returns.mean() / eq_returns.std(ddof=1) * math.sqrt(ann_factor)
                if eq_returns.std() > 0 else 0.0)
rolling_max  = eq_series.cummax()
drawdown     = (eq_series - rolling_max) / rolling_max
max_dd       = drawdown.min()

# Signal stats
long_bars  = int((np.array([position]) == 1).sum())   # approximate
z_arr      = z_score.dropna()
n_entries  = n_trades

print(f"\n  {'='*44}")
print(f"  Backtest Results")
print(f"  {'='*44}")
print(f"  Bars traded     : {N - ZSCORE_WINDOW:,} (1-min bars)")
print(f"  Number of trades: {n_trades}")
print(f"  Total return    : {total_return:+.4%}")
print(f"  Ann. Sharpe     : {sharpe:.3f}")
print(f"  Max drawdown    : {max_dd:.2%}")
print(f"  {'='*44}")
print(f"  Entry threshold : ±{ENTRY_Z} z-score")
print(f"  OU half-life    : {half_life_min:.1f} min")
print(f"  Cost per trade  : {COST:.1%} per leg")

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(3, 1, figsize=(13, 10),
                          gridspec_kw={"height_ratios": [3, 2, 2], "hspace": 0.35})
fig.patch.set_facecolor("#080b14")
for ax in axes:
    ax.set_facecolor("#101526")
    ax.tick_params(colors="#3a4060", labelsize=8)
    ax.spines["bottom"].set_color("#1c2238")
    ax.spines["left"].set_color("#1c2238")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=(1, 1, 1, 0.06), linewidth=0.5, linestyle="--")
    ax.yaxis.label.set_color("#6b749e")
    ax.xaxis.label.set_color("#6b749e")

dates = prices.index

# ── Panel 1: Equity curve ──────────────────────────────────────────────────

ax = axes[0]
ax.plot(dates, equity_curve, color="#4f8ef7", linewidth=1.3)
ax.axhline(1.0, color=(1,1,1,0.15), linewidth=0.8, linestyle="--")
ax.fill_between(dates, equity_curve, 1.0,
                where=np.array(equity_curve) >= 1.0,
                alpha=0.12, color="#00d48c")
ax.fill_between(dates, equity_curve, 1.0,
                where=np.array(equity_curve) < 1.0,
                alpha=0.12, color="#f04e6a")
ax.set_ylabel("Equity (normalised)", fontsize=9)
ax.set_title(
    f"BTC/ETH Stat Arb  |  Sharpe {sharpe:.2f}  |  Return {total_return:+.2%}  |  "
    f"MaxDD {max_dd:.2%}  |  {n_trades} trades",
    color="#d8dff5", fontsize=10, pad=8
)

# ── Panel 2: Z-score with entry/exit bands ────────────────────────────────

ax = axes[1]
z_plot = z_score.values.copy()
ax.plot(dates, z_plot, color="#a07ff0", linewidth=0.7, alpha=0.85)
ax.axhline( ENTRY_Z, color="#f04e6a", linewidth=0.9, linestyle="--", alpha=0.7,
            label=f"+{ENTRY_Z}σ entry")
ax.axhline(-ENTRY_Z, color="#00d48c", linewidth=0.9, linestyle="--", alpha=0.7,
            label=f"-{ENTRY_Z}σ entry")
ax.axhline(0, color="#3a4060", linewidth=0.6)
ax.fill_between(dates, z_plot, ENTRY_Z,
                where=np.array(z_plot) > ENTRY_Z, alpha=0.15, color="#f04e6a")
ax.fill_between(dates, z_plot, -ENTRY_Z,
                where=np.array(z_plot) < -ENTRY_Z, alpha=0.15, color="#00d48c")
ax.set_ylabel("Spread Z-score", fontsize=9)
ax.set_ylim(-6, 6)
ax.legend(fontsize=7, loc="upper right",
          facecolor="#101526", edgecolor="#1c2238", labelcolor="#6b749e")

# ── Panel 3: Raw spread ────────────────────────────────────────────────────

ax = axes[2]
ax.plot(dates, spread, color="#f5c518", linewidth=0.7, alpha=0.85)
ax.axhline(mu_ou, color="#4f8ef7", linewidth=0.9, linestyle="--",
           label=f"OU mean ({mu_ou:.4f})")
ax.set_ylabel("Log-price spread", fontsize=9)
ax.legend(fontsize=7, loc="upper right",
          facecolor="#101526", edgecolor="#1c2238", labelcolor="#6b749e")

for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
fig.autofmt_xdate(rotation=30, ha="right")

plt.savefig("stat_arb_equity.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("\nChart saved to stat_arb_equity.png")
