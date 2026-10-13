"""
cost_analysis.py  —  Commission & slippage stress-test on backtest trades.

Scenarios
  A  Zero costs (baseline — what the backtest already measures)
  B  $25 round-trip commission + 1-tick slippage each leg
  C  $50 round-trip commission + 2-tick slippage each leg

Tick size: $0.01/share × 100 = $1.00/contract (penny-pilot options — NVDA, AAPL, etc.)
Slippage is applied as: paid more on entry + received less on exit → 2× tick per leg.

Run:  .venv/bin/python3 cost_analysis.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
from strategy_params import PARAMS

_S = PARAMS["simulation"]

N_SIMS       = _S["n_sims"]
STARTING_CAP = _S["starting_capital"]
TICK_SIZE    = _S["tick_size"]

RESULTS_CSV  = Path(__file__).parent / "backtest_results.csv"

SCENARIOS = {
    "A  Zero costs (baseline)":             {"commission": 0.00,  "ticks": 0},
    "B  IBKR realistic (~$3.30)":           {"commission": 1.30,  "ticks": 1},
    "C  $25 commission + 1-tick slippage":  {"commission": 25.00, "ticks": 1},
    "D  $50 commission + 2-tick slippage":  {"commission": 50.00, "ticks": 2},
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _pct(arr: np.ndarray, p: float) -> float:
    return float(np.percentile(arr, p))


def _max_dd_vec(paths: np.ndarray) -> np.ndarray:
    peak = np.maximum.accumulate(paths, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak > 0, (peak - paths) / peak, 0.0)
    return dd.max(axis=1)


def cost_per_trade(commission: float, ticks: int) -> float:
    """Total round-trip cost per contract."""
    return commission + ticks * TICK_SIZE * 2   # ×2: entry leg + exit leg


def run_scenario(pnls_raw: np.ndarray, label: str,
                 commission: float, ticks: int) -> dict:
    cost   = cost_per_trade(commission, ticks)
    pnls   = pnls_raw - cost                    # adjust every trade

    n      = len(pnls)
    rng    = np.random.default_rng(42)
    idx    = rng.integers(0, n, size=(N_SIMS, n))
    samp   = pnls[idx]

    paths       = np.empty((N_SIMS, n + 1))
    paths[:, 0] = STARTING_CAP
    paths[:, 1:] = STARTING_CAP + np.cumsum(samp, axis=1)

    final_eq = paths[:, -1]
    max_dds  = _max_dd_vec(paths) * 100

    actual    = float(STARTING_CAP + pnls.sum())
    actual_mc = float(np.median(final_eq))

    return {
        "label":       label,
        "cost":        cost,
        "n_trades":    n,
        "pnls":        pnls,
        "final_eq":    final_eq,
        "max_dds":     max_dds,
        "prob_profit": float((final_eq > STARTING_CAP).mean() * 100),
        "ev_trade":    float(pnls.mean()),
        "median_pnl":  _pct(final_eq, 50) - STARTING_CAP,
        "p95_pnl":     _pct(final_eq, 95) - STARTING_CAP,
        "p5_pnl":      _pct(final_eq, 5)  - STARTING_CAP,
        "p50_dd":      _pct(max_dds, 50),
        "p95_dd":      _pct(max_dds, 95),
        "actual":      actual - STARTING_CAP,
        "actual_mc":   actual_mc - STARTING_CAP,
        "gap":         actual_mc - actual,   # MC median vs actual (should be ≈0)
    }


# ── Display ───────────────────────────────────────────────────────────────────
def colour(val: float, good_positive: bool = True) -> str:
    GREEN, RED, RESET = "\033[92m", "\033[91m", "\033[0m"
    is_good = val > 0 if good_positive else val < 0
    return f"{GREEN if is_good else RED}{val:>+,.2f}{RESET}"


def print_table(results: list[dict]) -> None:
    BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
    sep = "─" * 76

    print(f"\n{BOLD}{'Commission & Slippage Impact':^76}{RESET}")
    print(f"{DIM}Tick size $0.01/share × 100 = $1.00/contract  ·  "
          f"{N_SIMS:,} bootstrap sims  ·  ${STARTING_CAP:,} start  ·  "
          f"{results[0]['n_trades']} trades{RESET}")
    print(sep)

    # Header
    labels = [r["label"] for r in results]
    col    = 22
    print(f"  {'Metric':<26}" + "".join(f"  {l[:col]:<{col}}" for l in labels))
    print(sep)

    def row(name, getter, fmt=None, good_positive=True):
        GREEN, RED, DIM, RESET = "\033[92m", "\033[91m", "\033[2m", "\033[0m"
        cells = []
        for r in results:
            v = getter(r)
            if fmt == "pct":
                color = GREEN if (v > 50 if good_positive else v < 20) else RED
                cells.append(f"  {color}{v:>+.1f}%{RESET:<{col}}")
            elif fmt == "pprob":
                color = GREEN if v >= 80 else (DIM if v >= 50 else RED)
                cells.append(f"  {color}{v:>5.1f}%{RESET:<{col}}")
            elif fmt == "dd":
                color = GREEN if v <= 20 else (DIM if v <= 40 else RED)
                cells.append(f"  {color}{v:>5.1f}%{RESET:<{col}}")
            else:
                color = GREEN if v >= 0 else RED
                cells.append(f"  {color}${v:>+,.2f}{RESET:<{col}}")
        print(f"  {name:<26}" + "".join(cells))

    row("Round-trip cost/trade",   lambda r: r["cost"],        fmt="$cost")
    print(f"  {DIM}{'─'*72}{RESET}")
    row("Prob of profit",          lambda r: r["prob_profit"],  fmt="pprob")
    row("Median outcome (P50)",    lambda r: r["median_pnl"])
    row("Best case (P95)",         lambda r: r["p95_pnl"])
    row("Worst case (P5)",         lambda r: r["p5_pnl"])
    row("Expected value / trade",  lambda r: r["ev_trade"])
    print(f"  {DIM}{'─'*72}{RESET}")
    row("Actual backtest P&L",     lambda r: r["actual"])
    row("Median MC prediction",    lambda r: r["actual_mc"])
    row("Actual vs MC gap",        lambda r: r["gap"])
    print(f"  {DIM}{'─'*72}{RESET}")
    row("Median max drawdown",     lambda r: r["p50_dd"],       fmt="dd")
    row("Worst-case DD (P95)",     lambda r: r["p95_dd"],       fmt="dd")

    print(sep)

    # Which scenario best explains the actual result
    actual_true = results[0]["actual"]  # Scenario A's actual = raw backtest
    print(f"\n  {BOLD}Scenario closest to actual backtest result  (${actual_true:+,.2f}){RESET}")
    for r in results:
        gap = abs(r["actual_mc"] - actual_true)
        print(f"    {r['label']:<40}  MC median = ${r['actual_mc']:+,.2f}  "
              f"  gap = ${gap:,.2f}")

    print()

    # Break-even cost
    baseline_ev = results[0]["ev_trade"]
    print(f"  {BOLD}Break-even cost analysis{RESET}")
    print(f"  {DIM}Baseline EV/trade = ${baseline_ev:+.2f}  →  strategy breaks even "
          f"at ${baseline_ev:.2f}/trade in total costs{RESET}")
    for r in results[1:]:
        viable = r["ev_trade"] > 0
        status = "\033[92mVIABLE\033[0m" if viable else "\033[91mUNVIABLE\033[0m"
        print(f"    {r['label']:<40}  EV/trade = ${r['ev_trade']:+.2f}  →  {status}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not RESULTS_CSV.exists():
        print("backtest_results.csv not found — run backtest.py first.")
        raise SystemExit(1)

    df = pd.read_csv(RESULTS_CSV)
    df = df[df["score"] == 3].reset_index(drop=True)
    print(f"\nLoaded {len(df)} score-3 trades from backtest_results.csv")
    print(f"Win rate: {(df['pnl_$'] > 0).mean()*100:.1f}%  |  "
          f"Avg premium: ${df['prem_in'].mean():.2f}/share  |  "
          f"Avg P&L: ${df['pnl_$'].mean():+.2f}/trade\n")
    print(f"Running {N_SIMS:,} simulations per scenario...")

    pnls_raw = df["pnl_$"].to_numpy(dtype=float)

    results = []
    for label, params in SCENARIOS.items():
        r = run_scenario(pnls_raw, label, **params)
        results.append(r)
        print(f"  {label}  done")

    print_table(results)


if __name__ == "__main__":
    main()
