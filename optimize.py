"""
optimize.py — walk-forward parameter sweep for the backtest strategy.

Fetches market data once, then re-runs backtest.run_backtest() across a grid
of MIN_SCORE / MAX_STOP / COOLDOWN_BARS combinations against the SAME cached
data (no repeated network calls), ranks them, and compares the best
performers against the current backtest.py defaults.

This is read-only analysis. It never modifies backtest.py, scanner.py, or
auto_trader.py — it only writes optimization_results.json for the dashboard
(or a human) to review. Applying a suggested change is a manual, deliberate
edit to the relevant config constants, not something this script does.

Run: .venv/bin/python3 optimize.py
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import backtest

from strategy_params import PARAMS

_S = PARAMS["simulation"]

MIN_TRADES_FOR_CONFIDENCE = _S["min_trades_for_confidence"]

# One full run_backtest() pass over ~14 tickers / 30 days takes roughly a
# minute (it's the pattern-detection loop, not the thresholds, that's slow).
# Keep the grid small enough that a sweep finishes in a few minutes rather
# than tying up the dashboard for an hour. MIN_SCORE is the highest-leverage,
# most interpretable dial (it mirrors scanner.py's live score gate), so it
# gets the full sweep; MAX_STOP/COOLDOWN_BARS stay fixed at current values
# for now — widen these grids later once MIN_SCORE sweeps prove useful.
MIN_SCORE_GRID     = [1, 2, 3, 4, 5]
MAX_STOP_GRID       = [backtest.MAX_STOP]
COOLDOWN_BARS_GRID  = [backtest.COOLDOWN_BARS]

OUT_PATH = Path(__file__).parent / "optimization_results.json"


def _max_drawdown(df) -> float:
    """Largest peak-to-trough decline in cumulative $ P&L, trades taken in
    chronological order. 0.0 for an empty or never-losing sequence."""
    if df.empty:
        return 0.0
    ordered = df.sort_values(["date", "entry_time"])
    cum = ordered["pnl_$"].cumsum()
    drawdown = cum.cummax() - cum
    return round(float(drawdown.max()), 2)


def _summarize(df, min_score, max_stop, cooldown_bars):
    trades = len(df)
    if trades == 0:
        return {
            "min_score": min_score, "max_stop": max_stop, "cooldown_bars": cooldown_bars,
            "trades": 0, "win_rate": 0.0,
            "expectancy_dollars": 0.0, "expectancy_r": 0.0,
            "total_pnl": 0.0, "max_drawdown": 0.0,
            "sufficient_sample": False,
        }
    wins = int((df["pnl_$"] > 0).sum())
    # R here is the option's own risk basis (premium paid), not the
    # underlying's stop distance — this backtest prices the actual option
    # (Black-Scholes), so "1R" = "1x the premium risked on entry" is the
    # coherent definition; underlying-based R wouldn't track the $ P&L here.
    r_multiples = df["pnl_$"] / df["prem_in"]
    return {
        "min_score": min_score, "max_stop": max_stop, "cooldown_bars": cooldown_bars,
        "trades": trades,
        "win_rate": round(100 * wins / trades, 1),
        "expectancy_dollars": round(df["pnl_$"].mean(), 2),
        "expectancy_r": round(float(r_multiples.mean()), 3),
        "total_pnl": round(df["pnl_$"].sum(), 2),
        "max_drawdown": _max_drawdown(df),
        "sufficient_sample": trades >= MIN_TRADES_FOR_CONFIDENCE,
    }


def run_sweep():
    print("Loading market data once for the whole sweep...\n")
    all_data = backtest.load_all_data()

    current_df = backtest.run_backtest(
        all_data=all_data,
        min_score=backtest.MIN_SCORE,
        max_stop=backtest.MAX_STOP,
        cooldown_bars=backtest.COOLDOWN_BARS,
    )
    current = _summarize(current_df, backtest.MIN_SCORE, backtest.MAX_STOP, backtest.COOLDOWN_BARS)

    results = []
    total_combos = len(MIN_SCORE_GRID) * len(MAX_STOP_GRID) * len(COOLDOWN_BARS_GRID)
    done = 0
    for min_score in MIN_SCORE_GRID:
        for max_stop in MAX_STOP_GRID:
            for cooldown_bars in COOLDOWN_BARS_GRID:
                done += 1
                print(f"  [{done}/{total_combos}] score>={min_score} stop<={max_stop} cooldown={cooldown_bars}...", end="\r")
                if (min_score, max_stop, cooldown_bars) == (backtest.MIN_SCORE, backtest.MAX_STOP, backtest.COOLDOWN_BARS):
                    results.append(current)
                    continue
                df = backtest.run_backtest(
                    all_data=all_data,
                    min_score=min_score,
                    max_stop=max_stop,
                    cooldown_bars=cooldown_bars,
                )
                results.append(_summarize(df, min_score, max_stop, cooldown_bars))
    print(f"  Swept {total_combos} combinations.{' '*20}\n")

    results.sort(key=lambda r: r["total_pnl"], reverse=True)

    confident = [r for r in results if r["sufficient_sample"]]
    top = confident[0] if confident else None

    suggestion = None
    if top and not (top["min_score"] == current["min_score"]
                     and top["max_stop"] == current["max_stop"]
                     and top["cooldown_bars"] == current["cooldown_bars"]):
        improvement = top["total_pnl"] - current["total_pnl"]
        if improvement > max(20.0, 0.1 * abs(current["total_pnl"])):
            suggestion = (
                f"Over the last {backtest.LOOKBACK_DAYS} days, MIN_SCORE={top['min_score']}, "
                f"MAX_STOP=${top['max_stop']:.2f}, COOLDOWN_BARS={top['cooldown_bars']} produced "
                f"${top['total_pnl']:+.2f} ({top['win_rate']}% win rate, {top['trades']} trades) vs. "
                f"the current ${current['total_pnl']:+.2f} ({current['win_rate']}% win rate, "
                f"{current['trades']} trades) — a ${improvement:+.2f} difference. "
                f"Worth a manual look before changing anything live."
            )
    if suggestion is None:
        suggestion = "No alternative combination clearly outperformed the current settings over this window."

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "lookback_days": backtest.LOOKBACK_DAYS,
        "current": current,
        "results": results,
        "suggestion": suggestion,
    }
    OUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"Results → {OUT_PATH.name}")
    print(f"\n{suggestion}")
    return output


if __name__ == "__main__":
    run_sweep()
