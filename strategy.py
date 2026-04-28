"""Avellaneda–Stoikov market making backtest (tick + OHLCV)."""
from __future__ import annotations
import argparse, math, random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from as_engine import get_quotes
from competitor_floor import compute_competitor_floor
from regime import compute_volatility_regime

# ── Parameters ────────────────────────────────────────────────────────────────
KAPPA            = 150_000.0  # AS market-depth parameter
TIME_HORIZON     = 1.0        # Normalised time horizon τ
INVENTORY_CAP    = 200.0      # Hard inventory limit (units)
EMERGENCY_THRESH = 150.0      # Emergency gamma trigger (abs units)
EMERGENCY_MULT   = 5.0        # Emergency gamma multiplier
FILL_QTY         = 100.0      # Fill size per trade (units)
OHLCV_FILL_PROB  = 0.10       # Candle-touch fill probability
OHLCV_COOLDOWN   = 30         # Min candles between OHLCV fills
FLOOR_WINDOW     = 100        # Competitor floor rolling window
WARMUP_ROWS      = 500        # Rows to skip before trading (Hurst estimator needs 500 obs)
MIN_SPREAD_BPS   = 0.001      # 10 bps total spread floor
MAX_SPREAD_BPS   = 0.005      # 50 bps total spread cap
GAMMA            = {"low": 0.6, "normal": 1.0, "high": 2.5}  # risk aversion per regime


def _load(csv_path: str) -> tuple[pd.DataFrame, str]:
    df   = pd.read_csv(csv_path)
    cols = set(df.columns)
    if {"mid_price", "spread", "last_trade_price"}.issubset(cols):
        return df, "tick"
    if {"open", "high", "low", "close", "volume"}.issubset(cols):
        df = df.copy()
        df["mid_price"]        = pd.to_numeric(df["close"], errors="coerce")
        df["spread"]           = pd.to_numeric(df["high"],  errors="coerce") \
                               - pd.to_numeric(df["low"],   errors="coerce")
        df["last_trade_price"] = pd.to_numeric(df["close"], errors="coerce")
        return df, "ohlcv"
    raise KeyError(
        f"Unsupported CSV schema: {csv_path!r}. "
        "Need tick {mid_price, spread, last_trade_price} or OHLCV {open,high,low,close,volume}."
    )


def run_strategy(csv_path: str, static_gamma: float | None = None) -> dict:
    df, kind = _load(csv_path)
    n = len(df)
    mid        = pd.to_numeric(df["mid_price"],        errors="coerce").astype(float)
    spread_obs = pd.to_numeric(df["spread"],           errors="coerce").astype(float)
    lt         = pd.to_numeric(df["last_trade_price"], errors="coerce").astype(float)
    vol        = mid.pct_change().ewm(span=20, adjust=False).std().fillna(0.0)
    regimes, _, _ = compute_volatility_regime(mid)
    comp_floor    = compute_competitor_floor(spread_obs, window=FLOOR_WINDOW)
    low_s  = pd.to_numeric(df["low"],  errors="coerce").astype(float) if kind == "ohlcv" else None
    high_s = pd.to_numeric(df["high"], errors="coerce").astype(float) if kind == "ohlcv" else None

    inv           = 0.0
    cash          = 500.0
    last_fill_idx = -(10 ** 9)
    n_trades      = 0
    sc_pnl        = 0.0   # spread-capture PnL
    lt_prev       = float("nan")  # previous last_trade_price for crossing detection
    equity: list[float]   = []
    inv_path: list[float] = []

    for i in range(n):
        mid_i = float(mid.iloc[i])
        if not math.isfinite(mid_i):
            equity.append(cash); inv_path.append(inv); continue

        # gamma: regime-adaptive or fixed baseline; emergency widening on top
        gamma = static_gamma if static_gamma is not None else GAMMA.get(str(regimes.iloc[i]), 1.0)
        if abs(inv) > EMERGENCY_THRESH:
            gamma *= EMERGENCY_MULT

        # A–S reservation price + optimal spread → raw bid/ask
        bid, ask = get_quotes(
            mid_price=mid_i, inventory=inv, volatility=float(vol.iloc[i]),
            gamma=gamma, time_horizon=TIME_HORIZON, kappa=KAPPA,
        )

        # Clamp spread to [10 bps, 50 bps]; raise to competitor floor if needed
        r     = 0.5 * (bid + ask)       # AS reservation price
        s     = ask - bid
        min_s = MIN_SPREAD_BPS * mid_i
        max_s = MAX_SPREAD_BPS * mid_i
        s     = max(min_s, min(max_s, s))
        cf    = float(comp_floor.iloc[i])
        if math.isfinite(cf) and cf > min_s:
            s = max(s, min(cf, max_s))
        bid, ask = r - 0.5 * s, r + 0.5 * s

        if i < WARMUP_ROWS:                     # wait for Hurst window to fill
            equity.append(cash + inv * mid_i); inv_path.append(inv); continue

        can_buy  = inv <  INVENTORY_CAP
        can_sell = inv > -INVENTORY_CAP
        filled   = False

        if kind == "ohlcv":
            lo = float(low_s.iloc[i])   # type: ignore[union-attr]
            hi = float(high_s.iloc[i])  # type: ignore[union-attr]
            if (i - last_fill_idx) >= OHLCV_COOLDOWN and random.random() < OHLCV_FILL_PROB:
                buy_t  = can_buy  and math.isfinite(lo) and lo <= bid
                sell_t = can_sell and math.isfinite(hi) and hi >= ask
                if buy_t and sell_t:  # both touched: coin-flip
                    buy_t, sell_t = (True, False) if random.random() < 0.5 else (False, True)
                if buy_t:
                    inv += FILL_QTY;  cash -= bid * FILL_QTY;  filled = True
                elif sell_t:
                    inv -= FILL_QTY;  cash += ask * FILL_QTY;  filled = True
                if filled:
                    last_fill_idx = i
        else:
            lt_i = float(lt.iloc[i])
            if math.isfinite(lt_i) and math.isfinite(lt_prev):
                # Crossing model: fill only when last_trade crosses our quote level.
                # Buy fill: price moves DOWN through our bid (lt_prev > bid, lt_i <= bid).
                # Sell fill: price moves UP through our ask (lt_prev < ask, lt_i >= ask).
                if can_buy  and lt_prev > bid  and lt_i <= bid:
                    inv += FILL_QTY;  cash -= bid * FILL_QTY;  filled = True
                elif can_sell and lt_prev < ask and lt_i >= ask:
                    inv -= FILL_QTY;  cash += ask * FILL_QTY;  filled = True
            lt_prev = lt_i

        if filled:
            n_trades += 1
            sc_pnl   += s * FILL_QTY   # spread captured = full quoted spread × qty

        equity.append(cash + inv * mid_i)
        inv_path.append(inv)

    # ── Metrics ───────────────────────────────────────────────────────────────
    eq        = pd.Series(equity)
    total_pnl = float(eq.iloc[-1]) - 500.0 if n else 0.0
    inv_mtm   = total_pnl - sc_pnl
    max_dd    = float(((eq / eq.cummax()) - 1.0).min() * 100.0) if len(eq) else 0.0
    rets      = eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    std       = float(rets.std(ddof=0))
    sharpe    = float(rets.mean() / std) * math.sqrt(525_600.0) if len(rets) > 1 and std > 0 else 0.0
    fpd       = float(n_trades / max(n / 1440.0, 1.0))
    vc        = regimes.iloc[WARMUP_ROWS:].value_counts(normalize=True)
    regime_pct = {k: float(vc.get(k, 0.0)) * 100 for k in ("high", "normal", "low")}
    return {
        "spread_capture_pnl": sc_pnl,
        "inventory_mtm_pnl":  inv_mtm,
        "total_pnl":          total_pnl,
        "sharpe":             sharpe,
        "fills_per_day":      fpd,
        "max_drawdown_pct":   max_dd,
        "n_trades":           n_trades,
        "equity_series":      eq,
        "inventory_series":   pd.Series(inv_path),
        "regime_pct":         regime_pct,
    }


def _print_regime(regime_pct: dict) -> None:
    p = regime_pct
    print(f"  Regime split (post-warmup):  "
          f"high {p['high']:5.1f}%   normal {p['normal']:5.1f}%   low {p['low']:5.1f}%")


def _side_by_side(dyn: dict, sta: dict) -> None:
    """Print a two-column comparison table: dynamic gamma vs static gamma=0.1."""
    W = 16  # column width
    rows = [
        ("Spread capture PnL", f"{dyn['spread_capture_pnl']:>{W}.4f}", f"{sta['spread_capture_pnl']:>{W}.4f}"),
        ("Inventory MTM PnL",  f"{dyn['inventory_mtm_pnl']:>{W}.4f}",  f"{sta['inventory_mtm_pnl']:>{W}.4f}"),
        ("Total PnL",          f"{dyn['total_pnl']:>{W}.4f}",           f"{sta['total_pnl']:>{W}.4f}"),
        ("Sharpe (annual)",    f"{dyn['sharpe']:>{W}.4f}",              f"{sta['sharpe']:>{W}.4f}"),
        ("Fills per day",      f"{dyn['fills_per_day']:>{W}.2f}",       f"{sta['fills_per_day']:>{W}.2f}"),
        ("Max drawdown",       f"{dyn['max_drawdown_pct']:>{W-1}.3f}%", f"{sta['max_drawdown_pct']:>{W-1}.3f}%"),
        ("Trades",             f"{dyn['n_trades']:>{W}d}",              f"{sta['n_trades']:>{W}d}"),
    ]
    hdr = f"{'':22}{'Dynamic gamma':>{W}}{'Static gamma=0.1':>{W}}"
    sep = "-" * len(hdr)
    print(sep); print(hdr); print(sep)
    for label, d, s in rows:
        print(f"  {label:<20}{d}{s}")
    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(description="A–S market making backtest")
    parser.add_argument("--csv",         default="algo_data.csv")
    parser.add_argument("--monte-carlo", action="store_true")
    args = parser.parse_args()

    if args.monte_carlo:
        runs = 10
        dyn_pnls, dyn_sharpes, sta_pnls, sta_sharpes = [], [], [], []
        regime_pct = None
        for _ in range(runs):
            d = run_strategy(args.csv)
            s = run_strategy(args.csv, static_gamma=0.1)
            dyn_pnls.append(d["total_pnl"]);  dyn_sharpes.append(d["sharpe"])
            sta_pnls.append(s["total_pnl"]);   sta_sharpes.append(s["sharpe"])
            if regime_pct is None:
                regime_pct = d["regime_pct"]
        print(f"\nMonte Carlo ({runs} runs):")
        W = 20
        print(f"{'':22}{'Dynamic gamma':>{W}}{'Static gamma=0.1':>{W}}")
        print(f"  {'Mean PnL':<20}{np.mean(dyn_pnls):>{W}.4f}{np.mean(sta_pnls):>{W}.4f}")
        print(f"  {'Std  PnL':<20}{np.std(dyn_pnls,  ddof=0):>{W}.4f}{np.std(sta_pnls,  ddof=0):>{W}.4f}")
        print(f"  {'Mean Sharpe':<20}{np.mean(dyn_sharpes):>{W}.4f}{np.mean(sta_sharpes):>{W}.4f}")
        print(f"  {'Std  Sharpe':<20}{np.std(dyn_sharpes, ddof=0):>{W}.4f}{np.std(sta_sharpes, ddof=0):>{W}.4f}")
        _print_regime(regime_pct)
        return

    dyn = run_strategy(args.csv)
    sta = run_strategy(args.csv, static_gamma=0.1)
    _side_by_side(dyn, sta)
    _print_regime(dyn["regime_pct"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True)
    for ax, r, title in zip(axes[0], [dyn, sta], ["Equity - Dynamic gamma", "Equity - Static gamma=0.1"]):
        ax.plot(r["equity_series"].values); ax.set_ylabel("Equity ($)"); ax.set_title(title)
    for ax, r, title in zip(axes[1], [dyn, sta], ["Inventory - Dynamic gamma", "Inventory - Static gamma=0.1"]):
        ax.plot(r["inventory_series"].values); ax.set_ylabel("Inventory"); ax.set_xlabel("Step")
        ax.set_title(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
