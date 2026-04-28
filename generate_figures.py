"""
generate_figures.py — QFC vol surface report figures.

Reads vol_surface_data.json from the current directory and produces:

  figure1_dvol_timeline.png    — BTC + ETH DVOL through the FTX collapse
                                  (Oct–Dec 2022) with event annotations
  figure2_smile_comparison.png — IV smile: current SVI baseline vs estimated
                                  FTX peak stress, side by side BTC / ETH
  figure3_term_structure.png   — ATM vol term structure: pre-FTX, FTX peak,
                                  and current, side by side BTC / ETH

Historical scenarios are constructed from documented Deribit DVOL levels and
published IV observations during the Nov 2022 event; they are representative
illustrations, not tick-level replays.

Usage:
    python generate_figures.py
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Design tokens (match QFC website) ────────────────────────────────────────

BG       = "#0a0a0a"
BG_PANEL = "#0e0e0e"
GRID     = "#1a1a1a"
TEXT     = "#e2e2e2"
DIM      = "#555555"
BTC      = "#f7931a"
ETH      = "#8b9ef4"
BLUE     = "#4fa8e8"
GREEN    = "#22c55e"
RED      = "#ef4444"
YELLOW   = "#eab308"

MONO = {"family": "monospace"}

plt.rcParams.update({
    "font.family":        "monospace",
    "text.color":         TEXT,
    "axes.labelcolor":    DIM,
    "xtick.color":        DIM,
    "ytick.color":        DIM,
    "axes.facecolor":     BG_PANEL,
    "axes.edgecolor":     GRID,
    "axes.grid":          True,
    "grid.color":         GRID,
    "grid.linewidth":     0.6,
    "grid.linestyle":     "--",
    "grid.alpha":         0.9,
    "figure.facecolor":   BG,
    "legend.framealpha":  0,
    "legend.fontsize":    8,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
})


def _style_ax(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)


def _fig_note(fig: plt.Figure, note: str) -> None:
    fig.text(0.5, 0.01, note, ha="center", va="bottom",
             fontsize=7, color=DIM, alpha=0.8)


def _watermark(fig: plt.Figure) -> None:
    fig.text(0.995, 0.005, "QFC", ha="right", va="bottom",
             fontsize=7, color=DIM, alpha=0.5)


# ── SVI helpers ───────────────────────────────────────────────────────────────

def _svi_tv(k: np.ndarray, a: float, b: float, rho: float,
            m: float, sigma: float) -> np.ndarray:
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def _svi_iv_pct(k: np.ndarray, params: dict, T: float) -> np.ndarray:
    w = np.maximum(_svi_tv(k, **params), 0.0)
    return np.sqrt(w / T) * 100.0


# ── Synthetic DVOL generator ──────────────────────────────────────────────────

def _make_dates(start: str, end: str) -> list[datetime]:
    d, out = datetime.strptime(start, "%Y-%m-%d"), []
    stop   = datetime.strptime(end,   "%Y-%m-%d")
    while d <= stop:
        out.append(d)
        d += timedelta(days=1)
    return out


def _synthetic_dvol(
    dates: list[datetime],
    anchors: list[tuple[str, float]],
    noise_sd: float = 0.8,
    seed: int = 0,
) -> np.ndarray:
    """
    Piecewise-linear interpolation through (date, vol%) anchors plus a
    seeded smooth random walk for realism.  Random walk is attenuated near
    the peak so the spike shape stays clean.
    """
    rng  = np.random.default_rng(seed)
    t0   = dates[0]
    t_all  = np.array([(d - t0).days for d in dates], dtype=float)
    t_anch = np.array([(datetime.strptime(a[0], "%Y-%m-%d") - t0).days
                       for a in anchors], dtype=float)
    v_anch = np.array([a[1] for a in anchors], dtype=float)

    base   = np.interp(t_all, t_anch, v_anch)
    noise  = np.cumsum(rng.normal(0, noise_sd, len(dates)))
    noise -= noise[0]

    # attenuate noise near peak so the event spike is legible
    peak_t  = t_anch[v_anch.argmax()]
    att     = 1.0 - 0.85 * np.exp(-0.5 * ((t_all - peak_t) / 6) ** 2)
    return np.maximum(base + noise * att, 10.0)


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 1 — DVOL timeline
# ══════════════════════════════════════════════════════════════════════════════

# Key Deribit DVOL anchor points derived from published price/vol data.
# Sources: Deribit blog, The Block research, Kaiko Nov 2022 reports.
_BTC_ANCHORS = [
    ("2022-10-01", 57.5), ("2022-10-08", 55.2), ("2022-10-18", 56.8),
    ("2022-10-28", 58.5), ("2022-11-01", 61.5), ("2022-11-04", 64.0),
    ("2022-11-06", 69.0), ("2022-11-08", 81.0), ("2022-11-09", 97.5),
    ("2022-11-10", 92.0), ("2022-11-12", 86.5), ("2022-11-15", 80.0),
    ("2022-11-20", 74.5), ("2022-11-26", 70.5), ("2022-12-01", 67.0),
    ("2022-12-08", 65.5), ("2022-12-15", 64.0), ("2022-12-22", 63.5),
    ("2022-12-31", 62.5),
]
_ETH_ANCHORS = [
    ("2022-10-01", 68.0), ("2022-10-08", 65.5), ("2022-10-18", 67.5),
    ("2022-10-28", 70.5), ("2022-11-01", 74.0), ("2022-11-04", 78.0),
    ("2022-11-06", 84.0), ("2022-11-08", 99.0), ("2022-11-09", 118.5),
    ("2022-11-10", 111.0), ("2022-11-12", 103.0), ("2022-11-15", 95.0),
    ("2022-11-20", 87.0), ("2022-11-26", 81.5), ("2022-12-01", 77.0),
    ("2022-12-08", 73.5), ("2022-12-15", 71.5), ("2022-12-22", 70.5),
    ("2022-12-31", 69.5),
]

_EVENTS = [
    # (date, label, label-side)  side: 'L'=left of line, 'R'=right of line
    ("2022-11-02", "CoinDesk: Alameda\nbalance sheet leak",  "R"),
    ("2022-11-08", "Binance LOI;\nFTX halts withdrawals",    "R"),
    ("2022-11-09", "Binance withdraws;\nrun accelerates",    "L"),
    ("2022-11-11", "FTX files\nChapter 11",                  "R"),
    ("2022-12-12", "SBF arrested\nin Bahamas",               "R"),
]


def make_figure1() -> None:
    dates    = _make_dates("2022-10-01", "2022-12-31")
    btc_dvol = _synthetic_dvol(dates, _BTC_ANCHORS, noise_sd=0.65, seed=1)
    eth_dvol = _synthetic_dvol(dates, _ETH_ANCHORS, noise_sd=0.85, seed=2)

    fig, axes = plt.subplots(
        2, 1, figsize=(13, 7), sharex=True,
        gridspec_kw={"hspace": 0.06, "top": 0.91, "bottom": 0.10},
    )
    fig.suptitle(
        "DERIBIT DVOL INDEX  —  FTX COLLAPSE  (OCT – DEC 2022)",
        fontsize=11, fontweight="bold", color=TEXT, y=0.97,
    )

    panic_start = datetime(2022, 11, 8)
    panic_end   = datetime(2022, 11, 14)

    for ax, dvol, color, label in [
        (axes[0], btc_dvol, BTC, "BTC DVOL"),
        (axes[1], eth_dvol, ETH, "ETH DVOL"),
    ]:
        _style_ax(ax)
        ax.fill_between(dates, dvol, alpha=0.10, color=color)
        ax.plot(dates, dvol, color=color, linewidth=1.6, label=label, zorder=3)
        ax.axvspan(panic_start, panic_end, alpha=0.07, color=RED, zorder=0,
                   label="Peak panic window")

        for ev_date_str, _, _ in _EVENTS:
            ev_dt = datetime.strptime(ev_date_str, "%Y-%m-%d")
            ax.axvline(ev_dt, color=RED, linewidth=0.7, linestyle="--",
                       alpha=0.45, zorder=2)

        ax.set_ylabel("30D IV  (%)", fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        ax.legend(loc="upper left", fontsize=9, labelcolor=color)
        ax.set_xlim(dates[0], dates[-1])

    # Event labels on the top panel only
    ax0   = axes[0]
    y_top = ax0.get_ylim()[1]
    y_bot = ax0.get_ylim()[0]
    span  = y_top - y_bot

    # Stagger labels vertically so they don't overlap
    v_offsets = [0.95, 0.78, 0.60, 0.95, 0.78]
    for (ev_date_str, ev_label, side), v_frac in zip(_EVENTS, v_offsets):
        ev_dt  = datetime.strptime(ev_date_str, "%Y-%m-%d")
        x_off  = timedelta(days=1 if side == "R" else -1)
        ha     = "left" if side == "R" else "right"
        ax0.text(
            ev_dt + x_off, y_bot + span * v_frac,
            ev_label, color=RED, fontsize=6.5, va="top", ha=ha,
            linespacing=1.35, alpha=0.9, zorder=5,
        )

    # Peak panic shading label (on bottom panel)
    mid_panic = panic_start + (panic_end - panic_start) / 2
    axes[1].text(
        mid_panic, axes[1].get_ylim()[0] + 2,
        "PEAK\nPANIC", color=RED, fontsize=6, ha="center",
        va="bottom", alpha=0.55, linespacing=1.2,
    )

    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    axes[1].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    axes[1].tick_params(axis="x", rotation=30)

    _fig_note(fig,
              "Synthetic illustration constructed from published Deribit DVOL levels "
              "(Deribit Blog · The Block Research · Kaiko Nov 2022).  "
              "Not tick-level replay data.")
    _watermark(fig)

    fig.savefig("figure1_dvol_timeline.png", dpi=150, bbox_inches="tight",
                facecolor=BG)
    plt.close(fig)
    print("Saved figure1_dvol_timeline.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 2 — Smile comparison
# ══════════════════════════════════════════════════════════════════════════════

# Estimated SVI parameters for the FTX peak (Nov 9–11 2022), ~30-day expiry.
# Calibrated to match documented ATM DVOL levels (BTC ~97%, ETH ~118%)
# and the characteristic left-skew steepening observed during liquidation events.
_FTX_T = 30.0 / 365.25   # 30-day expiry (years)

_FTX_SVI = {
    "BTC": {"a": 0.0170, "b": 0.200, "rho": -0.35, "m": 0.00, "sigma": 0.300},
    "ETH": {"a": 0.0268, "b": 0.250, "rho": -0.40, "m": 0.00, "sigma": 0.350},
}
# Sanity-check ATM IVs:  BTC ≈ 97%,  ETH ≈ 118%


def make_figure2(data: dict) -> None:
    k_grid = np.linspace(-0.40, 0.40, 300)

    fig, axes = plt.subplots(
        1, 2, figsize=(13, 5.5),
        gridspec_kw={"top": 0.88, "bottom": 0.14, "left": 0.07,
                     "right": 0.97, "wspace": 0.28},
    )
    fig.suptitle(
        "IMPLIED VOLATILITY SMILE  —  CURRENT  vs  FTX PEAK STRESS",
        fontsize=11, fontweight="bold", color=TEXT, y=0.97,
    )

    for ax, asset, color in [
        (axes[0], "BTC", BTC),
        (axes[1], "ETH", ETH),
    ]:
        _style_ax(ax)

        # ── FTX stress curve ─────────────────────────────────────────────────
        fp     = _FTX_SVI[asset]
        ftx_iv = _svi_iv_pct(k_grid, fp, _FTX_T)
        ax.fill_between(k_grid, ftx_iv, alpha=0.07, color=RED)
        ax.plot(k_grid, ftx_iv, color=RED, linewidth=1.8, linestyle="--",
                label=f"FTX Peak  (Nov 9–11 2022,  ~30d)\nATM ≈ {ftx_iv[len(k_grid)//2]:.0f}%",
                zorder=3)

        # ── Current curve — use longest available fitted expiry ───────────────
        fits = data.get("assets", {}).get(asset, {}).get("svi_fits", {})
        if fits:
            # pick expiry with largest T
            best_key  = max(fits, key=lambda k: fits[k]["T"])
            fit       = fits[best_key]
            cur_T     = fit["T"]
            cur_iv    = _svi_iv_pct(k_grid, fit["params"], cur_T)
            exp_label = f"{best_key[:2]}-{best_key[2:5]}-{best_key[5:]}"
            cur_dte   = cur_T * 365.25
            ax.fill_between(k_grid, cur_iv, alpha=0.13, color=color)
            ax.plot(k_grid, cur_iv, color=color, linewidth=1.8,
                    label=f"Current  ({exp_label},  {cur_dte:.1f}d)\nATM ≈ {cur_iv[len(k_grid)//2]:.0f}%",
                    zorder=4)
        else:
            ax.text(0, 50, "No current data available",
                    ha="center", color=DIM, fontsize=9)

        # ── ATM marker ───────────────────────────────────────────────────────
        ax.axvline(0, color=DIM, linewidth=0.7, linestyle=":", alpha=0.6, zorder=1)
        ax.text(0.005, ax.get_ylim()[0] * 1.01 if ax.get_ylim()[0] > 0 else 2,
                "ATM", color=DIM, fontsize=7, va="bottom", ha="left")

        # ── Skew annotation ──────────────────────────────────────────────────
        ax.annotate("", xy=(-0.30, ftx_iv[int(0.15 * len(k_grid))]),
                    xytext=(-0.05, ftx_iv[int(0.15 * len(k_grid))]),
                    arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))
        ax.text(-0.18, ftx_iv[int(0.15 * len(k_grid))] + 2,
                "put skew steepening", color=RED, fontsize=6.5, ha="center")

        ax.set_xlabel("Log-moneyness   k = log(K / F)")
        ax.set_ylabel("Implied Volatility  (%)")
        ax.set_title(f"{asset} / USD", color=color, fontsize=10,
                     fontweight="bold", pad=8)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        ax.set_xlim(-0.42, 0.42)
        ax.legend(loc="upper right", labelcolor=TEXT, fontsize=7.5,
                  handlelength=1.6, labelspacing=0.6)

    _fig_note(fig,
              "FTX stress smile estimated from Nov 2022 Deribit DVOL levels and published skew observations.  "
              "Current smile: live SVI fit from Deribit public API.")
    _watermark(fig)

    fig.savefig("figure2_smile_comparison.png", dpi=150, bbox_inches="tight",
                facecolor=BG)
    plt.close(fig)
    print("Saved figure2_smile_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 3 — Volatility term structure
# ══════════════════════════════════════════════════════════════════════════════

# ATM (k=0) implied vol by DTE for the two historical scenarios.
# Pre-FTX: mild contango typical of Q4 2022 before the event.
# FTX peak: deeply inverted — front end spiked while long end lagged.
_PRE_FTX = {
    "BTC": [(7, 54), (14, 56), (30, 59), (60, 61.5), (90, 63), (180, 65)],
    "ETH": [(7, 63), (14, 66), (30, 70), (60, 72),   (90, 74), (180, 76)],
}
_FTX_PEAK = {
    "BTC": [(7, 145), (14, 122), (30, 97.5), (60, 85), (90, 79), (180, 73)],
    "ETH": [(7, 175), (14, 148), (30, 118),  (60, 102),(90, 94), (180, 86)],
}


def make_figure3(data: dict) -> None:
    fig, axes = plt.subplots(
        1, 2, figsize=(13, 5.5),
        gridspec_kw={"top": 0.88, "bottom": 0.14, "left": 0.07,
                     "right": 0.97, "wspace": 0.28},
    )
    fig.suptitle(
        "ATM VOLATILITY TERM STRUCTURE  —  PRE-FTX  /  FTX PEAK  /  CURRENT",
        fontsize=11, fontweight="bold", color=TEXT, y=0.97,
    )

    for ax, asset, color in [
        (axes[0], "BTC", BTC),
        (axes[1], "ETH", ETH),
    ]:
        _style_ax(ax)

        # ── Pre-FTX ─────────────────────────────────────────────────────────
        pf    = _PRE_FTX[asset]
        pf_x  = [p[0] for p in pf]
        pf_y  = [p[1] for p in pf]
        ax.fill_between(pf_x, pf_y, alpha=0.07, color=GREEN)
        ax.plot(pf_x, pf_y, color=GREEN, linewidth=1.6, linestyle="--",
                marker="o", markersize=4, label="Pre-FTX  (Oct 2022)", zorder=3)

        # ── FTX peak ─────────────────────────────────────────────────────────
        fp   = _FTX_PEAK[asset]
        fp_x = [p[0] for p in fp]
        fp_y = [p[1] for p in fp]
        ax.fill_between(fp_x, fp_y, alpha=0.07, color=RED)
        ax.plot(fp_x, fp_y, color=RED, linewidth=1.6, linestyle="--",
                marker="o", markersize=4, label="FTX Peak  (Nov 9–11 2022)", zorder=4)

        # Inversion annotation
        ax.annotate(
            "inversion\n(front-end spike)",
            xy=(7, fp_y[0]), xytext=(25, fp_y[0] - 18),
            color=RED, fontsize=6.5, ha="left",
            arrowprops=dict(arrowstyle="->", color=RED, lw=0.7),
        )

        # ── Current (from JSON) ───────────────────────────────────────────────
        fits = data.get("assets", {}).get(asset, {}).get("svi_fits", {})
        if fits:
            cur_dte, cur_atm = [], []
            for fit in fits.values():
                T      = fit["T"]
                p      = fit["params"]
                w_atm  = float(np.maximum(_svi_tv(np.array([0.0]), **p), 0.0)[0])
                iv_atm = math.sqrt(w_atm / T) * 100.0 if T > 0 else 0.0
                cur_dte.append(T * 365.25)
                cur_atm.append(iv_atm)

            order   = sorted(range(len(cur_dte)), key=lambda i: cur_dte[i])
            cur_dte = [cur_dte[i] for i in order]
            cur_atm = [cur_atm[i] for i in order]

            ax.plot(cur_dte, cur_atm, color=color, linewidth=2.0,
                    marker="D", markersize=5,
                    label=f"Current  (live SVI, {len(cur_dte)} expiries)", zorder=5)
            ax.fill_between(cur_dte, cur_atm, alpha=0.14, color=color)

            # Annotate each current point with its DTE
            for dte, atm in zip(cur_dte, cur_atm):
                ax.text(dte + 0.5, atm + 1.5, f"{dte:.0f}d",
                        color=color, fontsize=6.5, va="bottom", ha="left")

        ax.set_xlabel("Days to Expiry  (DTE)")
        ax.set_ylabel("ATM Implied Volatility  (%)")
        ax.set_title(f"{asset} / USD", color=color, fontsize=10,
                     fontweight="bold", pad=8)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        ax.set_xlim(left=-2)
        ax.legend(loc="upper right", labelcolor=TEXT, fontsize=7.5,
                  handlelength=1.6, labelspacing=0.5)

    _fig_note(fig,
              "Pre-FTX and FTX peak ATM levels constructed from Deribit DVOL and published smile data.  "
              "Current: SVI fit at k=0 per expiry, live from Deribit public API.")
    _watermark(fig)

    fig.savefig("figure3_term_structure.png", dpi=150, bbox_inches="tight",
                facecolor=BG)
    plt.close(fig)
    print("Saved figure3_term_structure.png")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading vol_surface_data.json...", flush=True)
    try:
        with open("vol_surface_data.json", "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        print("ERROR: vol_surface_data.json not found in current directory.",
              file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"ERROR: could not parse vol_surface_data.json: {exc}",
              file=sys.stderr)
        sys.exit(1)

    print("Generating figure 1 — DVOL timeline...", flush=True)
    make_figure1()

    print("Generating figure 2 — smile comparison...", flush=True)
    make_figure2(data)

    print("Generating figure 3 — term structure...", flush=True)
    make_figure3(data)

    print("Done — 3 PNG files written.")


if __name__ == "__main__":
    main()
