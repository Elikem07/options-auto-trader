"""
backtest.py — 0DTE options backtest
Buys ATM calls/puts at each pattern alert, prices them with Black-Scholes,
and tracks theta decay bar-by-bar until stop/T1/T2 or end of day.

Data source: IBKR, via the local cache in historical_cache/ (see
ibkr_history.py). Previously yfinance, which hard-caps intraday bars
(anything under 1h, including these 5-min bars) at 60 days — no amount of
config could get a multi-year backtest out of it. IBKR supports much
longer history, but only by walking backward in ~1-month chunks (a single
large request just times out), which is slow enough (~30-40s/chunk,
serial to stay polite to IBKR's pacing limits and not disrupt
auto_trader.py sharing the same connection) that it's a separate, cached,
run-once-and-reuse step rather than something backtest.py does live.
Run ibkr_history.py first to populate/update the cache before running this.

Run: .venv/bin/python3 backtest.py
"""
import math
import sys
from datetime import date, time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from scanner import (
    WATCHLIST, _process_df, detect_patterns, whipsaw_detected, ET,
    DAILY_0DTE_TICKERS as DAILY_0DTE, MWF_0DTE_TICKERS as MWF_0DTE,
)
import ibkr_history

# ── Config ────────────────────────────────────────────────────────
from strategy_params import PARAMS

_BT = PARAMS["backtest"]

MIN_SCORE     = _BT["min_score"]
MAX_STOP      = _BT["max_stop"]
LOOKBACK_DAYS = _BT["lookback_days"]   # gracefully
                        # uses whatever's actually cached if less is available
COOLDOWN_BARS = _BT["cooldown_bars"]  # must match execution.cooldown_min
RISK_FREE     = _BT["risk_free"]   # annualized

# Set to None to allow all setups, or a list of exact setup names to filter
SETUPS = ["Volume Divergence Short"]
SKIP_TICKERS = {"SPY", "PLTR", "MSFT", "GOOGL", "TSLA", "META"}

# Only trade tickers that use nearest-Friday expiry (never 0DTE)
# Skips: DAILY_0DTE and MWF_0DTE, both imported from scanner.py (single source
# of truth — no longer duplicated here, since the two copies had drifted out
# of sync with what's actually in scanner.py/auto_trader.py).
WEEKLY_ONLY = False
# Everything else → next Friday (weekly)

# Annualized IV estimates per ticker (0DTE priced higher than 30-day IV)
IV_MAP = {
    "SPY":  0.18, "QQQ":  0.20, "IWM":  0.22, "DIA":  0.18,
    "NVDA": 0.55, "TSLA": 0.60, "AAPL": 0.28, "AMD":  0.50,
    "META": 0.35, "AMZN": 0.30, "MSFT": 0.28, "GOOGL": 0.30,
    "SOXX": 0.30, "XLF":  0.22, "SMH":  0.32, "NFLX": 0.40,
    "BA":   0.35, "COIN": 0.80, "PLTR": 0.70, "MU":   0.45,
}

# ── Black-Scholes ─────────────────────────────────────────────────
def _ncdf(x: float) -> float:
    return (1.0 + math.erf(x / math.sqrt(2))) / 2.0

def _bs_price(S: float, K: float, T: float, r: float, sigma: float, opt: str) -> float:
    if T <= 0:
        return max(S - K, 0) if opt == "call" else max(K - S, 0)
    if S <= 0 or K <= 0 or sigma <= 0:
        return 0.0
    try:
        d1 = (math.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        if opt == "call":
            return S * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)
        else:
            return K * math.exp(-r * T) * _ncdf(-d2) - S * _ncdf(-d1)
    except Exception:
        return 0.0

def _dte(ticker: str, trade_date, bar_time) -> float:
    """
    Fractional trading days to nearest option expiration.
    Daily 0DTE tickers → expires today.
    MWF tickers → 0DTE on Mon/Wed/Fri, else next MWF day.
    Weekly tickers → next Friday.
    """
    weekday  = trade_date.weekday()   # 0=Mon … 4=Fri
    bar_min  = bar_time.hour * 60 + bar_time.minute
    frac_day = max(16 * 60 - bar_min, 1) / 390  # fraction of today remaining

    if ticker in DAILY_0DTE:
        return frac_day

    if ticker in MWF_0DTE:
        if weekday in (0, 2, 4):   # Mon / Wed / Fri
            return frac_day
        for delta in range(1, 5):
            if (weekday + delta) % 7 in (0, 2, 4):
                return delta + frac_day

    # Weekly — next Friday
    if weekday == 4:
        return frac_day
    return (4 - weekday) + frac_day

def _tte(ticker: str, trade_date, bar_time) -> float:
    """Time to expiry in years."""
    return _dte(ticker, trade_date, bar_time) / 252

def _strike(price: float) -> float:
    """Nearest ATM strike."""
    return round(price / 5) * 5 if price > 500 else round(price)

# ── Data ──────────────────────────────────────────────────────────
def fetch_ticker(ticker: str) -> "pd.DataFrame | None":
    """Reads from the local IBKR cache (historical_cache/) — see
    ibkr_history.py to populate/update it. Returns None if nothing's
    cached yet for this ticker (run ibkr_history.py first)."""
    return ibkr_history.fetch_ticker_from_cache(ticker)

def trading_days(df: pd.DataFrame) -> list[date]:
    return sorted(set(df[df.index.time >= time(9, 30)].index.date))

def day_slice(df: pd.DataFrame, day: date) -> pd.DataFrame:
    return df[df.index.date == day]

def stock_is_weak(df: pd.DataFrame) -> bool:
    """
    True when the stock is showing intraday weakness:
      - current close is below VWAP (average price for the day)
      - current close is below the opening bar of the session
    Both must be true to avoid taking shorts on a day that opened weak
    but has already recovered.
    """
    mkt = df[df.index.time >= time(9, 30)]
    if len(mkt) < 2:
        return False
    last = mkt.iloc[-1]
    vwap = last.get("vwap", float("nan"))
    if pd.isna(vwap) or vwap == 0:
        return False
    day_open = mkt.iloc[0]["open"]
    return last["close"] < vwap and last["close"] < day_open

# ── Scoring ───────────────────────────────────────────────────────
def score_backtest(alert: dict, df: pd.DataFrame,
                   spy_vis: "pd.DataFrame | None",
                   qqq_vis: "pd.DataFrame | None",
                   bar_time: time) -> int:
    score  = 0
    ticker = alert["ticker"]
    bias   = alert["bias"]
    setup  = alert["setup"]

    if "VWAP Vol Divergence" in setup:
        score += 2
    elif "Volume Divergence" in setup:
        score += 1

    ref = qqq_vis if ticker in ("QQQ", "TQQQ", "SQQQ", "SOXX", "SMH") else spy_vis
    if ref is not None:
        mkt = ref[ref.index.time >= time(9, 30)]
        if not mkt.empty:
            last = mkt.iloc[-1]
            vwap = last.get("vwap", float("nan"))
            if pd.notna(vwap) and vwap > 0:
                if (bias == "LONG" and last["close"] > vwap) or \
                   (bias == "SHORT" and last["close"] < vwap):
                    score += 1

    if not whipsaw_detected(df):
        score += 1
    if time(9, 30) <= bar_time <= time(10, 30) or time(15, 0) <= bar_time <= time(16, 0):
        score += 1
    return score

# ── Options Outcome ───────────────────────────────────────────────
def check_outcome(alert: dict, future_bars: pd.DataFrame, entry_ts, trade_date) -> dict:
    bias     = alert["bias"]
    entry    = alert["entry"]
    stop     = alert["stop"]
    t1       = alert["target1"]
    t2       = alert["target2"]
    ticker   = alert["ticker"]
    opt      = "call" if bias == "LONG" else "put"
    iv       = IV_MAP.get(ticker, 0.35)
    K        = _strike(entry)
    T_entry  = _tte(ticker, trade_date, entry_ts.time())
    prem_in  = max(_bs_price(entry, K, T_entry, RISK_FREE, iv, opt), 0.01)
    is_0dte  = ticker in DAILY_0DTE or (ticker in MWF_0DTE and trade_date.weekday() in (0, 2, 4))

    entry_time = entry_ts.strftime("%H:%M")

    def make(kind, exit_price, ts):
        T_exit   = _tte(ticker, trade_date, ts.time())
        prem_out = _bs_price(exit_price, K, T_exit, RISK_FREE, iv, opt)
        pnl      = (prem_out - prem_in) * 100
        return {
            "result":      kind,
            "entry_time":  entry_time,
            "exit_time":   ts.strftime("%H:%M"),
            "entry_price": entry,
            "exit_price":  round(exit_price, 2),
            "stop":        alert["stop"],
            "t1":          alert["target1"],
            "t2":          alert["target2"],
            "opt_type":    opt.upper(),
            "strike":      K,
            "prem_in":     round(prem_in, 2),
            "prem_out":    round(prem_out, 2),
            "pnl_$":       round(pnl, 2),
            "pnl_%":       round((prem_out - prem_in) / prem_in * 100, 1),
            "is_0dte":     is_0dte,
        }

    for idx, bar in future_bars.iterrows():
        lo, hi = bar["low"], bar["high"]
        if bias == "LONG":
            if lo <= stop:  return make("stop", stop, idx)
            if hi >= t2:    return make("t2",   t2,   idx)
            if hi >= t1:    return make("t1",   t1,   idx)
        else:
            if hi >= stop:  return make("stop", stop, idx)
            if lo <= t2:    return make("t2",   t2,   idx)
            if lo <= t1:    return make("t1",   t1,   idx)

    if future_bars.empty:
        return make("eod", entry, entry_ts)

    last     = future_bars.iloc[-1]
    last_ts  = future_bars.index[-1]
    T_eod    = 0 if is_0dte else _tte(ticker, trade_date, last_ts.time())
    prem_out = _bs_price(last["close"], K, T_eod, RISK_FREE, iv, opt)
    pnl      = (prem_out - prem_in) * 100
    return {
        "result":      "eod",
        "entry_time":  entry_time,
        "exit_time":   last_ts.strftime("%H:%M"),
        "entry_price": entry,
        "exit_price":  round(last["close"], 2),
        "stop":        alert["stop"],
        "t1":          alert["target1"],
        "t2":          alert["target2"],
        "opt_type":    opt.upper(),
        "strike":      K,
        "prem_in":     round(prem_in, 2),
        "prem_out":    round(prem_out, 2),
        "pnl_$":       round(pnl, 2),
        "pnl_%":       round((prem_out - prem_in) / prem_in * 100, 1),
        "is_0dte":     is_0dte,
    }

# ── Backtest Loop ─────────────────────────────────────────────────
def load_all_data() -> dict[str, "pd.DataFrame"]:
    """Fetch every watchlist ticker's bars once, for reuse across multiple backtest runs."""
    all_data: dict[str, pd.DataFrame] = {}
    for ticker in WATCHLIST:
        print(f"  Loading {ticker}...", end="\r")
        df = fetch_ticker(ticker)
        if df is not None:
            all_data[ticker] = df
    print(f"  Loaded {len(all_data)}/{len(WATCHLIST)} tickers.{' ' * 20}\n")
    return all_data


def run_backtest(
    all_data: "dict[str, pd.DataFrame] | None" = None,
    min_score: int = MIN_SCORE,
    max_stop: float = MAX_STOP,
    cooldown_bars: int = COOLDOWN_BARS,
    lookback_days: int = LOOKBACK_DAYS,
    setups=SETUPS,
    skip_tickers=SKIP_TICKERS,
    weekly_only: bool = WEEKLY_ONLY,
) -> pd.DataFrame:
    """Run the backtest against already-fetched data (or fetch it if not supplied).

    All thresholds default to this module's own config constants, so calling
    run_backtest() with no arguments behaves exactly as before. Passing
    all_data in lets a caller (e.g. a parameter sweep) fetch once and re-run
    the scoring/filtering logic many times without refetching from yfinance.
    """
    if all_data is None:
        print(f"\nBacktest — 0DTE Options — score ≥ {min_score}/5 — last {lookback_days} days")
        print("Fetching data...\n")
        all_data = load_all_data()

    rows = []

    for ticker, df in all_data.items():
        if ticker in skip_tickers:
            continue
        if weekly_only and ticker in (DAILY_0DTE | MWF_0DTE):
            continue
        days   = trading_days(df)[-lookback_days:]
        trades = 0

        for day in days:
            day_df   = day_slice(df, day)
            mkt_bars = day_df[day_df.index.time >= time(9, 30)]
            if len(mkt_bars) < 10:
                continue

            spy_day = day_slice(all_data["SPY"], day) if "SPY" in all_data else None
            qqq_day = day_slice(all_data["QQQ"], day) if "QQQ" in all_data else None
            fired: dict[str, int] = {}

            for i in range(3, len(mkt_bars)):
                bar_ts   = mkt_bars.index[i]
                bar_time = bar_ts.time()
                visible  = day_df[day_df.index <= bar_ts]
                spy_vis  = spy_day[spy_day.index <= bar_ts] if spy_day is not None else None
                qqq_vis  = qqq_day[qqq_day.index <= bar_ts] if qqq_day is not None else None

                active_alerts, watching_alerts = detect_patterns(ticker, visible)
                for alert in active_alerts + watching_alerts:
                    if setups is not None and alert["setup"] not in setups:
                        continue
                    # Only take shorts when the stock is already weak on the day
                    if alert["bias"] == "SHORT" and not stock_is_weak(visible):
                        continue
                    key = f"{alert['setup']}_{alert['bias']}"
                    if fired.get(key, -999) > i - cooldown_bars:
                        continue
                    if abs(alert["entry"] - alert["stop"]) > max_stop:
                        continue

                    score = score_backtest(alert, visible, spy_vis, qqq_vis, bar_time)
                    if score < min_score:
                        continue

                    fired[key] = i
                    outcome = check_outcome(alert, mkt_bars.iloc[i + 1:], bar_ts, day)
                    rows.append({
                        "date":   day,
                        "ticker": ticker,
                        "setup":  alert["setup"],
                        "bias":   alert["bias"],
                        "score":  score,
                        **outcome,
                    })
                    trades += 1

        if trades > 0:
            print(f"  {ticker:<6}  {trades} trades")

    return pd.DataFrame(rows)

# ── Results ───────────────────────────────────────────────────────
def print_results(df: pd.DataFrame) -> None:
    if df.empty:
        print("\nNo trades found.\n")
        return

    wins   = df[df["pnl_$"] > 0]
    losses = df[df["pnl_$"] < 0]
    sep    = "═" * 62

    print(f"\n{sep}")
    print(f"  0DTE OPTIONS BACKTEST  ·  {len(df)} trades  ·  last {LOOKBACK_DAYS} days")
    print(f"{sep}")
    print(f"  Win Rate       {len(wins)/len(df)*100:>6.1f}%   ({len(wins)}W / {len(losses)}L)")
    print(f"  Total P&L      ${df['pnl_$'].sum():>+8.2f}  per contract")
    print(f"  Avg / Trade    ${df['pnl_$'].mean():>+8.2f}")
    print(f"  Avg Win        ${wins['pnl_$'].mean():>+8.2f}" if not wins.empty else "  Avg Win        n/a")
    print(f"  Avg Loss       ${losses['pnl_$'].mean():>+8.2f}" if not losses.empty else "  Avg Loss       n/a")
    print(f"  Avg Premium    ${df['prem_in'].mean():>8.2f}/share  (${df['prem_in'].mean()*100:>6.0f}/contract)")
    print(f"  Best Trade     ${df['pnl_$'].max():>+8.2f}")
    print(f"  Worst Trade    ${df['pnl_$'].min():>+8.2f}")

    print(f"\n  By Setup")
    print(f"  {'─' * 58}")
    print(f"  {'Setup':<30} {'#':>4}  {'Win%':>5}  {'Total$':>8}  {'Avg$':>7}")
    for setup, g in df.groupby("setup"):
        w = len(g[g["pnl_$"] > 0])
        print(f"  {setup:<30} {len(g):>4}  {w/len(g)*100:>4.0f}%  "
              f"${g['pnl_$'].sum():>+7.2f}  ${g['pnl_$'].mean():>+6.2f}")

    print(f"\n  By Score")
    print(f"  {'─' * 58}")
    print(f"  {'Score':<8} {'#':>4}  {'Win%':>5}  {'Total$':>8}  {'Avg$':>7}")
    for score, g in df.groupby("score"):
        w = len(g[g["pnl_$"] > 0])
        print(f"  {score}/5     {len(g):>4}  {w/len(g)*100:>4.0f}%  "
              f"${g['pnl_$'].sum():>+7.2f}  ${g['pnl_$'].mean():>+6.2f}")

    print(f"\n  By Ticker  (ranked by total $)")
    print(f"  {'─' * 58}")
    print(f"  {'Ticker':<8} {'#':>4}  {'Win%':>5}  {'Total$':>8}  {'Avg$':>7}  {'AvgPrem':>8}")
    by_t = (
        df.groupby("ticker")
        .apply(
            lambda g: pd.Series({
                "n":        len(g),
                "win_pct":  len(g[g["pnl_$"] > 0]) / len(g) * 100,
                "total":    g["pnl_$"].sum(),
                "avg":      g["pnl_$"].mean(),
                "avg_prem": g["prem_in"].mean(),
            }),
            include_groups=False,
        )
        .sort_values("total", ascending=False)
    )
    for t, row in by_t.iterrows():
        print(f"  {t:<8} {int(row['n']):>4}  {row['win_pct']:>4.0f}%  "
              f"${row['total']:>+7.2f}  ${row['avg']:>+6.2f}  ${row['avg_prem']:>7.2f}")

    print(f"\n  By Expiry Type")
    print(f"  {'─' * 58}")
    print(f"  {'Type':<16} {'#':>4}  {'Win%':>5}  {'Total$':>8}  {'Avg$':>7}  {'AvgPrem':>8}")
    for label, mask in [("0DTE", df["is_0dte"] == True), ("Nearest Expiry", df["is_0dte"] == False)]:
        g = df[mask]
        if g.empty:
            continue
        w = len(g[g["pnl_$"] > 0])
        print(f"  {label:<16} {len(g):>4}  {w/len(g)*100:>4.0f}%  "
              f"${g['pnl_$'].sum():>+7.2f}  ${g['pnl_$'].mean():>+6.2f}  ${g['prem_in'].mean():>7.2f}")

    print(f"\n{sep}")
    csv_path = Path(__file__).parent / "backtest_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Full results → backtest_results.csv")
    print(f"\n  0DTE tickers: SPY QQQ IWM DIA XLF (daily)")
    print(f"  MWF tickers:  AAPL MSFT NVDA TSLA AMZN GOOGL META AMD")
    print(f"  Weekly:       COIN PLTR BA NFLX SMH SOXX MU (next Friday)")
    print(f"  ATM strike, 1 contract per trade. Stop/targets = underlying price.\n")


if __name__ == "__main__":
    results = run_backtest()
    print_results(results)
