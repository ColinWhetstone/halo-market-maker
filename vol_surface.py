"""
vol_surface.py — BTC and ETH implied vol surface with SVI fitting.

Pulls live option chains for both assets from Deribit's public API,
fits an SVI (Stochastic Volatility Inspired) smile to each expiry slice,
writes a JSON file with raw observations + fitted parameters + a dense
interpolated grid, and prints a per-expiry RMSE summary.

SVI parametrization (total implied variance):
    w(k) = a + b * (rho*(k-m) + sqrt((k-m)^2 + sigma^2))
    k    = log(K / F)   — log-moneyness
    T    = time to expiry in years

No-arbitrage constraints enforced:
    b >= 0
    |rho| < 1
    a + b*sigma*sqrt(1 - rho^2) >= 0   (ensures w(k) >= 0 for all k)

Usage:
    python vol_surface.py
    python vol_surface.py --expiries 4 --workers 40 --out custom.json
    python vol_surface.py --quiet          # one-line log-friendly output
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import requests
from scipy.optimize import minimize

# ── Configuration ─────────────────────────────────────────────────────────────

BASE        = "https://www.deribit.com/api/v2/public"
ASSETS      = ["BTC", "ETH"]
N_GRID      = 50     # log-moneyness grid points for interpolated surface
K_GRID_MIN  = -0.6   # log(K/F) lower bound for grid
K_GRID_MAX  =  0.6   # log(K/F) upper bound for grid
MIN_POINTS  = 5      # minimum strikes needed to attempt SVI fit

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "vol-surface/1.0"

QUIET = False   # set True by --quiet; suppresses progress and table output


def _vprint(*args, **kwargs) -> None:
    """Print only when not running in quiet mode."""
    if not QUIET:
        print(*args, **kwargs)


# ── API helpers ───────────────────────────────────────────────────────────────

def _get(endpoint: str, params: dict) -> dict:
    url = f"{BASE}/{endpoint}"
    resp = SESSION.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"Deribit error: {data['error']}")
    return data["result"]


def fetch_instruments(currency: str) -> list[dict]:
    return _get("get_instruments", {"currency": currency, "kind": "option"})


def fetch_ticker(instrument_name: str) -> Optional[dict]:
    try:
        return _get("ticker", {"instrument_name": instrument_name})
    except Exception:
        return None


# ── Chain collection ──────────────────────────────────────────────────────────

def collect_chain(currency: str, n_expiries: int, max_workers: int) -> tuple[dict, dict]:
    """
    Fetch the nearest n_expiries option expiry slices for `currency`.

    Returns:
        surface    : { expiry_str: { strike: { "C": ticker, "P": ticker } } }
        expiry_ts  : { expiry_str: expiration_timestamp_ms }
    """
    _vprint(f"\n[{currency}] Fetching instrument list...", flush=True)
    instruments = fetch_instruments(currency)

    now_ms = datetime.now(tz=timezone.utc).timestamp() * 1000
    live   = [i for i in instruments if i["expiration_timestamp"] > now_ms]

    expiry_ts: dict[str, int] = {}
    for inst in live:
        exp = inst["instrument_name"].split("-")[1]   # "28APR26"
        expiry_ts[exp] = inst["expiration_timestamp"]

    chosen     = sorted(expiry_ts, key=lambda e: expiry_ts[e])[:n_expiries]
    chosen_set = set(chosen)
    target     = [i for i in live if i["instrument_name"].split("-")[1] in chosen_set]

    _vprint(f"[{currency}] Fetching {len(target)} tickers across {len(chosen)} expiries...",
            flush=True)

    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(fetch_ticker, inst["instrument_name"]): inst["instrument_name"]
            for inst in target
        }
        done = 0
        for fut in as_completed(futures):
            name   = futures[fut]
            ticker = fut.result()
            if ticker is not None:
                results[name] = ticker
            done += 1
            _vprint(f"\r  {done}/{len(target)} fetched", end="", flush=True)
    _vprint(f"\r[{currency}] Got {len(results)}/{len(target)} tickers.         ", flush=True)

    # Organise into surface dict
    surface: dict[str, dict[float, dict]] = {}
    for name, ticker in results.items():
        parts      = name.split("-")
        expiry_str = parts[1]
        strike     = float(parts[2])
        opt_type   = parts[3]          # "C" or "P"
        surface.setdefault(expiry_str, {})
        surface[expiry_str].setdefault(strike, {})
        surface[expiry_str][strike][opt_type] = ticker

    ordered    = {e: surface[e]   for e in chosen if e in surface}
    chosen_ts  = {e: expiry_ts[e] for e in chosen if e in surface}
    return ordered, chosen_ts


# ── Expiry slice builder ──────────────────────────────────────────────────────

def _forward_from_slice(strikes_dict: dict) -> Optional[float]:
    """
    Extract the expiry-specific forward price from any ticker in the slice.
    Deribit's `underlying_price` is the futures/forward price for that expiry.
    """
    for tickers in strikes_dict.values():
        for ticker in tickers.values():
            f = ticker.get("underlying_price")
            if f and float(f) > 0:
                return float(f)
    return None


def build_expiry_slice(
    expiry_str: str,
    strikes_dict: dict,
    expiry_ts_ms: int,
    now_ms: float,
) -> Optional[dict]:
    """
    Build arrays and raw rows for one expiry.

    Returns dict with keys:
        T, forward, k_arr, w_arr, iv_arr (all floats/np.ndarray), raw_rows (list)
    or None if data is unusable.
    """
    T = (expiry_ts_ms - now_ms) / (1_000 * 365.25 * 24 * 3_600)
    if T <= 1e-4:
        return None

    forward = _forward_from_slice(strikes_dict)
    if forward is None:
        return None

    ks, ws, ivs = [], [], []
    raw_rows: list[dict] = []

    for strike in sorted(strikes_dict):
        call = strikes_dict[strike].get("C")
        put  = strikes_dict[strike].get("P")

        c_iv = (float(call["mark_iv"]) / 100.0
                if call and call.get("mark_iv") is not None else None)
        p_iv = (float(put["mark_iv"]) / 100.0
                if put and put.get("mark_iv") is not None else None)

        # Average call and put IVs — they should be identical under no-arb
        # but averaging guards against tiny model discrepancies.
        if c_iv is not None and p_iv is not None:
            iv = (c_iv + p_iv) / 2.0
        elif c_iv is not None:
            iv = c_iv
        elif p_iv is not None:
            iv = p_iv
        else:
            iv = None

        c_delta = (float(call["greeks"]["delta"])
                   if call and call.get("greeks") else None)
        p_delta = (float(put["greeks"]["delta"])
                   if put and put.get("greeks") else None)

        k = math.log(strike / forward)

        raw_rows.append({
            "strike":     strike,
            "expiry":     expiry_str,
            "k":          round(k, 6),
            "T":          round(T, 6),
            "call_iv":    round(c_iv  * 100, 4) if c_iv  is not None else None,
            "put_iv":     round(p_iv  * 100, 4) if p_iv  is not None else None,
            "call_delta": round(c_delta, 6)     if c_delta is not None else None,
            "put_delta":  round(p_delta, 6)     if p_delta is not None else None,
        })

        if iv is not None and iv > 0:
            ks.append(k)
            ws.append(iv * iv * T)   # total variance
            ivs.append(iv)

    if len(ks) < MIN_POINTS:
        return None

    return {
        "T":        T,
        "forward":  forward,
        "k_arr":    np.array(ks,  dtype=float),
        "w_arr":    np.array(ws,  dtype=float),
        "iv_arr":   np.array(ivs, dtype=float),
        "raw_rows": raw_rows,
    }


# ── SVI model ─────────────────────────────────────────────────────────────────

def svi_tv(k: np.ndarray, a: float, b: float, rho: float,
           m: float, sigma: float) -> np.ndarray:
    """Total implied variance: w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))."""
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def _objective(params: np.ndarray, k_arr: np.ndarray, w_arr: np.ndarray) -> float:
    a, b, rho, m, sigma = params
    w_fit  = svi_tv(k_arr, a, b, rho, m, sigma)
    mse    = float(np.mean((w_fit - w_arr) ** 2))

    # Soft penalty: a + b*sigma*sqrt(1-rho^2) >= 0
    # (guarantees w(k) >= 0 everywhere; the SVI minimum occurs at the vertex)
    min_var = a + b * sigma * math.sqrt(max(1.0 - rho ** 2, 0.0))
    penalty = 0.0
    if min_var < 0:
        penalty += 1e6 * min_var ** 2
    # Also penalise any observed-point negative fitted values (belt-and-suspenders)
    neg = w_fit[w_fit < 0]
    if len(neg):
        penalty += 1e6 * float(np.sum(neg ** 2))

    return mse + penalty


# L-BFGS-B parameter bounds: (a, b, rho, m, sigma)
_SVI_BOUNDS = [
    (None, None),       # a  — unconstrained (penalty handles a+b*sigma*... >= 0)
    (1e-6, None),       # b  >= 0
    (-0.999, 0.999),    # |rho| < 1
    (None, None),       # m  — unconstrained
    (1e-4, None),       # sigma > 0
]


def fit_svi(k_arr: np.ndarray, w_arr: np.ndarray) -> Optional[dict]:
    """
    Fit SVI to (log-moneyness, total-variance) observations.

    Tries five starting points and returns the best result by objective value.
    Returns a dict {a, b, rho, m, sigma} or None on failure.
    """
    # Estimate ATM total variance for sensible initial guesses
    idx_atm = int(np.argmin(np.abs(k_arr)))
    w_atm   = max(float(w_arr[idx_atm]), 1e-6)

    init_guesses = [
        [w_atm * 0.50,  0.10, -0.30,  0.00, 0.30],
        [w_atm * 0.80,  0.20, -0.50,  0.00, 0.20],
        [w_atm * 0.30,  0.30,  0.00,  0.00, 0.40],
        [w_atm * 0.50,  0.08, -0.10, -0.05, 0.35],
        [w_atm * 0.60,  0.15,  0.10,  0.05, 0.50],
    ]

    best_res = None
    best_val = np.inf

    for x0 in init_guesses:
        try:
            res = minimize(
                _objective, x0, args=(k_arr, w_arr),
                method="L-BFGS-B",
                bounds=_SVI_BOUNDS,
                options={"maxiter": 3000, "ftol": 1e-14, "gtol": 1e-9},
            )
            if res.fun < best_val:
                best_val = res.fun
                best_res = res
        except Exception:
            continue

    if best_res is None:
        return None

    a, b, rho, m, sigma = best_res.x
    return {"a": float(a), "b": float(b), "rho": float(rho),
            "m": float(m), "sigma": float(sigma)}


def rmse_iv_pct(params: dict, k_arr: np.ndarray, iv_arr: np.ndarray, T: float) -> float:
    """Root-mean-square error between fitted and observed IV, in percentage points."""
    w_fit  = svi_tv(k_arr, **params)
    iv_fit = np.sqrt(np.maximum(w_fit, 0.0) / T) * 100.0   # in %
    iv_obs = iv_arr * 100.0
    return float(np.sqrt(np.mean((iv_fit - iv_obs) ** 2)))


# ── Interpolated grid ─────────────────────────────────────────────────────────

def build_grid(fitted_slices: dict) -> dict:
    """
    Build a dense 50-strike × N-expiry grid using the fitted SVI curves.

    The log-moneyness axis is shared across expiries (K_GRID_MIN to K_GRID_MAX).
    Strike values differ per expiry because each forward price differs.

    Returns a dict ready for JSON serialisation.
    """
    k_grid = list(np.linspace(K_GRID_MIN, K_GRID_MAX, N_GRID))

    expiry_labels, T_values, strike_grid, iv_grid, tv_grid = [], [], [], [], []

    for expiry_str, sl in fitted_slices.items():
        p       = sl["params"]
        T       = sl["T"]
        forward = sl["forward"]

        k_arr = np.array(k_grid)
        w_arr = np.maximum(svi_tv(k_arr, **p), 0.0)
        iv_pct = np.sqrt(w_arr / T) * 100.0

        expiry_labels.append(expiry_str)
        T_values.append(round(T, 6))
        strike_grid.append([round(float(forward * math.exp(k)), 4) for k in k_arr])
        iv_grid.append([round(float(v), 4) for v in iv_pct])
        tv_grid.append([round(float(v), 8) for v in w_arr])

    return {
        "description": (
            "Dense SVI-interpolated grid. "
            f"k_grid has {N_GRID} points in [{K_GRID_MIN}, {K_GRID_MAX}]. "
            "Dimensions: [expiry_index][strike_index]."
        ),
        "k_grid":      [round(k, 6) for k in k_grid],
        "expiries":    expiry_labels,
        "T_values":    T_values,
        "strike_grid": strike_grid,   # [n_exp × N_GRID]
        "iv_grid":     iv_grid,       # [n_exp × N_GRID], annualised IV in %
        "tv_grid":     tv_grid,       # [n_exp × N_GRID], total variance w(k)
    }


# ── Per-asset pipeline ────────────────────────────────────────────────────────

def process_asset(currency: str, n_expiries: int, max_workers: int) -> dict:
    surface, expiry_ts = collect_chain(currency, n_expiries, max_workers)
    now_ms = datetime.now(tz=timezone.utc).timestamp() * 1000

    _vprint(f"\n[{currency}] Fitting SVI to {len(surface)} expiry slices...", flush=True)

    raw_observations: list[dict] = []
    svi_fits:          dict[str, dict] = {}
    fitted_slices:     dict[str, dict] = {}   # for grid building

    for expiry_str, strikes_dict in surface.items():
        sl = build_expiry_slice(expiry_str, strikes_dict, expiry_ts[expiry_str], now_ms)
        if sl is None:
            print(f"  [{currency}] {expiry_str}: skipped (T<=0 or too few points)",
                  file=sys.stderr)
            continue

        raw_observations.extend(sl["raw_rows"])

        params = fit_svi(sl["k_arr"], sl["w_arr"])
        if params is None:
            print(f"  [{currency}] {expiry_str}: SVI optimizer failed", file=sys.stderr)
            continue

        rmse = rmse_iv_pct(params, sl["k_arr"], sl["iv_arr"], sl["T"])

        svi_fits[expiry_str] = {
            "T":           round(sl["T"], 6),
            "forward":     round(sl["forward"], 4),
            "n_points":    int(len(sl["k_arr"])),
            "params":      {k: round(v, 8) for k, v in params.items()},
            "rmse_iv_pct": round(rmse, 5),
        }
        fitted_slices[expiry_str] = {
            "T":       sl["T"],
            "forward": sl["forward"],
            "params":  params,
        }

        _vprint(f"  [{currency}] {expiry_str}: n={len(sl['k_arr'])}  RMSE={rmse:.3f}%  "
                f"a={params['a']:+.4f}  b={params['b']:.4f}  "
                f"rho={params['rho']:+.4f}  m={params['m']:+.4f}  "
                f"sigma={params['sigma']:.4f}")

    grid = build_grid(fitted_slices) if fitted_slices else {}

    return {
        "raw_observations":  raw_observations,
        "svi_fits":          svi_fits,
        "interpolated_grid": grid,
    }


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(output: dict) -> None:
    W = 72
    HDR = (f"  {'Expiry':<10}  {'T (yr)':<7}  {'Forward':>12}  "
           f"{'N pts':>5}  {'RMSE IV':>8}  {'a':>8}  {'b':>6}  "
           f"{'rho':>7}  {'m':>7}  {'sigma':>6}")
    SEP = "-" * W

    print("\n" + "=" * W)
    print("  SVI Fit Summary")
    print("=" * W)

    for asset in ASSETS:
        adat = output["assets"].get(asset)
        if not adat:
            continue
        fits = adat["svi_fits"]
        if not fits:
            print(f"\n  {asset}: no successful fits.")
            continue

        print(f"\n  {asset}")
        print(SEP)
        print(HDR)
        print(SEP)

        for expiry_str, fit in fits.items():
            label = f"{expiry_str[:2]}-{expiry_str[2:5]}-{expiry_str[5:]}"
            p     = fit["params"]
            print(
                f"  {label:<10}  {fit['T']:<7.4f}  "
                f"{fit['forward']:>12,.2f}  "
                f"{fit['n_points']:>5}  "
                f"{fit['rmse_iv_pct']:>7.3f}%  "
                f"{p['a']:>8.4f}  {p['b']:>6.4f}  "
                f"{p['rho']:>7.4f}  {p['m']:>7.4f}  {p['sigma']:>6.4f}"
            )

        print(SEP)

        # Per-asset aggregate
        rmse_vals = [f["rmse_iv_pct"] for f in fits.values()]
        print(f"  {asset} mean RMSE: {sum(rmse_vals)/len(rmse_vals):.3f}%  "
              f"  max RMSE: {max(rmse_vals):.3f}%")

    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="BTC + ETH implied vol surface with SVI fitting (Deribit public API)"
    )
    parser.add_argument("--expiries", type=int, default=6,
                        help="Nearest expiries per asset (default: 6)")
    parser.add_argument("--workers",  type=int, default=30,
                        help="Parallel ticker fetch concurrency (default: 30)")
    parser.add_argument("--out",      default="vol_surface_data.json",
                        help="Output JSON file path (default: vol_surface_data.json)")
    args = parser.parse_args()

    print("=" * 72)
    print("  Vol Surface Builder — BTC + ETH  |  Deribit public API")
    print(f"  Expiries per asset : {args.expiries}")
    print(f"  Fetch workers      : {args.workers}")
    print(f"  Output file        : {args.out}")
    print("=" * 72)

    generated_at = datetime.now(tz=timezone.utc).isoformat()
    output: dict = {"generated_at": generated_at, "assets": {}}

    for currency in ASSETS:
        try:
            output["assets"][currency] = process_asset(
                currency, args.expiries, args.workers
            )
        except requests.RequestException as exc:
            print(f"[{currency}] Network error: {exc}", file=sys.stderr)
        except Exception as exc:
            print(f"[{currency}] Unexpected error: {exc}", file=sys.stderr)
            raise

    # Write JSON (numpy values serialised to plain Python floats above)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)

    # Count totals for confirmation line
    total_obs = sum(
        len(output["assets"][a]["raw_observations"])
        for a in ASSETS if a in output["assets"]
    )
    total_fits = sum(
        len(output["assets"][a]["svi_fits"])
        for a in ASSETS if a in output["assets"]
    )
    print(f"\nWrote {args.out}  "
          f"({total_obs} raw observations, {total_fits} SVI fits)")

    print_summary(output)


if __name__ == "__main__":
    main()
