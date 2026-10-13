"""signal_logger.py — records every rejected pattern detection, plus a
simulated forward outcome, to rejected_signals.csv.

This module only reads values that scanner.py's detect_patterns()/
score_alert() already compute (or re-derives the same intermediate values
scanner.py itself uses, like VWAP/day-open) — it never changes pattern
detection or scoring, and it never causes anything to trade. It's purely an
observation log, for later analysis of what the strategy is currently
throwing away.

Live signals can't be simulated forward instantly — "the next 12 bars"
haven't happened yet when a signal fires. So rejections are queued in
memory and resolved later, once auto_trader.py's normal re-fetch (it always
pulls the FULL day's bars, not just the latest one) naturally contains the
bars needed. If the process restarts before a queued signal resolves, that
signal is dropped rather than logged — a known tradeoff of the in-memory
queue, not a silent correctness bug.
"""
from datetime import date, time
from pathlib import Path

import pandas as pd

CSV_PATH = Path(__file__).parent / "rejected_signals.csv"
MAX_HOLD_BARS = 12   # bars to simulate forward before giving up (60 min @ 5-min bars)

COLUMNS = [
    "timestamp", "ticker", "setup", "bias",
    "base_points", "mkt_points", "whipsaw_points", "power_points", "total_score",
    "rejection_reason",
    "entry", "stop", "target1", "target2",
    "close", "vwap", "day_open", "below_vwap", "below_open",
    "resolution", "bars_to_resolution", "mfe_r", "mae_r", "pnl_r",
]

_pending: list[dict] = []
_seen: set[tuple] = set()   # (ticker, setup, bar_ts) already queued or written — dedup guard


def _base_points(setup: str) -> int:
    """Mirrors scanner.score_alert()'s own base-point logic exactly — not a
    new rule, just re-deriving the same number for logging."""
    if "VWAP Vol Divergence" in setup:
        return 2
    if "Volume Divergence" in setup:
        return 1
    return 0


def _last_market_bar(df: pd.DataFrame):
    market_df = df[df.index.time >= time(9, 30)]
    return market_df if len(market_df) else df


def record_rejection(alert: dict, df: pd.DataFrame, reason: str) -> None:
    """Queue a rejected/skipped detection for forward simulation.

    Call this after score_alert(alert, df, market_data) has already run, so
    alert["score"]/cond_* are populated. Safe to call for any setup type,
    including the six watching-bucket ones — it never causes a trade.
    """
    market_df = _last_market_bar(df)
    c = market_df.iloc[-1]
    bar_ts = market_df.index[-1]

    key = (alert["ticker"], alert["setup"], bar_ts)
    if key in _seen:
        return   # same signal already queued/logged this bar — don't duplicate
    _seen.add(key)

    vwap = c.get("vwap", float("nan"))
    day_open = market_df.iloc[0]["open"]
    close = c["close"]

    _pending.append({
        "timestamp":  bar_ts,
        "ticker":     alert["ticker"],
        "setup":      alert["setup"],
        "bias":       alert["bias"],
        "base_points":    _base_points(alert["setup"]),
        "mkt_points":     int(alert.get("cond_mkt", False)),
        "whipsaw_points": int(alert.get("cond_clean", False)),
        "power_points":   int(alert.get("cond_power", False)),
        "total_score":    alert.get("score"),
        "rejection_reason": reason,
        "entry": alert["entry"], "stop": alert["stop"],
        "target1": alert["target1"], "target2": alert["target2"],
        "close": round(float(close), 2),
        "vwap": round(float(vwap), 2) if pd.notna(vwap) else None,
        "day_open": round(float(day_open), 2),
        "below_vwap": bool(close < vwap) if pd.notna(vwap) else None,
        "below_open": bool(close < day_open),
        "_bar_ts": bar_ts,   # internal only — stripped before writing
    })


def _outcome(resolution: str, bars, mfe, mae, pnl_r) -> dict:
    return {
        "resolution": resolution,
        "bars_to_resolution": bars,
        "mfe_r": round(mfe, 2) if mfe is not None else None,
        "mae_r": round(mae, 2) if mae is not None else None,
        "pnl_r": round(pnl_r, 2) if pnl_r is not None else None,
    }


def _simulate(row: dict, df: "pd.DataFrame | None") -> "dict | None":
    """Try to resolve a queued signal using bars now available in df.
    Returns an outcome dict once resolved (stop/T1/T2 hit, or
    MAX_HOLD_BARS elapsed with no hit), else None if not enough bars yet.
    """
    if df is None:
        return None

    bar_ts = row["_bar_ts"]

    # Each fetch is scoped to "today" only (durationStr="1 D"). Once the
    # calendar day has turned over, df no longer contains yesterday's bars
    # at all — anything after bar_ts would actually be a brand new session,
    # not a continuation. Finalize honestly rather than let an overnight
    # gap masquerade as normal intraday movement.
    if bar_ts.date() < date.today():
        return _outcome("eod_unresolved", None, None, None, None)

    future = df[df.index > bar_ts]
    if future.empty:
        return None

    entry, stop = row["entry"], row["stop"]
    t1, t2 = row["target1"], row["target2"]
    bias = row["bias"]
    risk = abs(entry - stop)
    if risk == 0:
        return _outcome("no_resolution", 0, 0.0, 0.0, 0.0)

    window = future.iloc[:MAX_HOLD_BARS]
    mfe = mae = 0.0
    for i, (_, bar) in enumerate(window.iterrows(), start=1):
        lo, hi = bar["low"], bar["high"]
        favorable = (hi - entry) if bias == "LONG" else (entry - lo)
        adverse   = (entry - lo) if bias == "LONG" else (hi - entry)
        mfe = max(mfe, favorable / risk)
        mae = max(mae, adverse / risk)

        if bias == "LONG":
            if lo <= stop: return _outcome("stop", i, mfe, mae, -1.0)
            if hi >= t2:   return _outcome("t2",   i, mfe, mae, 2.5)
            if hi >= t1:   return _outcome("t1",   i, mfe, mae, 1.5)
        else:
            if hi >= stop: return _outcome("stop", i, mfe, mae, -1.0)
            if lo <= t2:   return _outcome("t2",   i, mfe, mae, 2.5)
            if lo <= t1:   return _outcome("t1",   i, mfe, mae, 1.5)

    if len(window) < MAX_HOLD_BARS:
        return None   # not enough bars yet — keep waiting

    last_close = window.iloc[-1]["close"]
    pnl_r = ((last_close - entry) / risk) if bias == "LONG" else ((entry - last_close) / risk)
    return _outcome("no_resolution", MAX_HOLD_BARS, mfe, mae, pnl_r)


def _write_row(row: dict) -> None:
    row = {k: v for k, v in row.items() if not k.startswith("_")}
    is_new = not CSV_PATH.exists()
    df = pd.DataFrame([row], columns=COLUMNS)
    df.to_csv(CSV_PATH, mode="a", header=is_new, index=False)


def process_pending(all_data: dict[str, pd.DataFrame]) -> None:
    """Call once per scan cycle. Attempts to resolve queued signals using
    freshly-fetched bars; writes resolved rows to the CSV and drops them
    from the queue. Unresolved ones stay queued for the next cycle."""
    still_pending = []
    for row in _pending:
        outcome = _simulate(row, all_data.get(row["ticker"]))
        if outcome is None:
            still_pending.append(row)
        else:
            _write_row({**row, **outcome})
    _pending[:] = still_pending
