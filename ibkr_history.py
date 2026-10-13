"""ibkr_history.py — fetch and cache multi-year 5-min bar history from IBKR.

yfinance caps intraday bars (anything under 1h, including the 5-min bars
this strategy runs on) at 60 days — a hard API limit, not a setting. IBKR's
reqHistoricalData supports much longer history, but not in one request: a
single request for "6 M" of 5-min bars times out (tested — 0 bars back in
60s). "1 M" works reliably (~4100 bars, ~30-40s). So multi-year history
means walking backward in ~1-month chunks and concatenating.

This is a real cost, not a quick call: 2 years x 14 tickers, 1-month
chunks, is roughly 336 requests at ~30-40s each run serially (~3.5-4h).
Deliberately NOT parallelized — the live auto_trader.py process shares
this same TWS connection/account, and IBKR's historical-data pacing limits
(no more than ~60 requests per rolling 10-minute window, plus per-contract
spacing rules) apply account-wide, not per-client-id. Hammering this
concurrently risks a pacing violation that could disrupt the live bot, not
just this fetch. Keeping it serial with a deliberate pause between
requests is the safe tradeoff for a one-time cost that gets cached to disk
afterward — this should never need to run in full more than once (plus
occasional incremental top-ups for whatever's happened since the last run).

Cache format: one Parquet file per ticker at
historical_cache/{ticker}_5min.parquet, indexed by bar timestamp (ET,
tz-aware), columns open/high/low/close/volume — the same shape
backtest.py's fetch_ticker() already produces via _process_df(), so
swapping the data source doesn't require touching detect_patterns()/
score_backtest()/anything downstream.

Run standalone: .venv/bin/python3 ibkr_history.py [--months N] [--tickers T1,T2,...]
"""
import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())  # must be before ib_insync import on Python 3.12+

import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from ib_insync import IB, Stock

sys.path.insert(0, str(Path(__file__).parent))
from scanner import WATCHLIST, _process_df

CACHE_DIR = Path(__file__).parent / "historical_cache"
CACHE_DIR.mkdir(exist_ok=True)

CHUNK_DURATION = "1 M"          # validated: reliable; "6 M" times out entirely
PACING_DELAY_SEC = 5.0          # deliberate pause between requests, serial only
CHUNK_TIMEOUT_SEC = 60           # single chunks observed taking up to ~35s; leave headroom
MAX_RETRIES = 2                  # transient timeouts happen (observed once in testing) —
                                  # retry before giving up and truncating a ticker's history
CLIENT_ID = 97                  # distinct from auto_trader.py's clientId=10


def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker}_5min.parquet"


async def _connect() -> IB:
    ib = IB()
    await ib.connectAsync("127.0.0.1", 7497, clientId=CLIENT_ID, readonly=True)
    return ib


async def fetch_ticker_history(ib: IB, ticker: str, months_back: int, log=print) -> pd.DataFrame:
    """Walk backward from now in CHUNK_DURATION-sized requests, concatenate,
    dedupe, and return one continuous raw OHLCV dataframe (not yet
    _process_df()'d — caller decides when to compute vwap/delta, since
    those are cheap to recompute but we don't want to pay the fetch cost
    twice)."""
    contract = Stock(ticker, "SMART", "USD")
    chunks = []
    end_dt = ""   # '' = now, for the first request only

    for i in range(months_back):
        bars = None
        for attempt in range(1, MAX_RETRIES + 2):   # first try + MAX_RETRIES retries
            try:
                bars = await asyncio.wait_for(
                    ib.reqHistoricalDataAsync(
                        contract, endDateTime=end_dt, durationStr=CHUNK_DURATION,
                        barSizeSetting="5 mins", whatToShow="TRADES",
                        useRTH=False, formatDate=1,
                    ),
                    timeout=CHUNK_TIMEOUT_SEC,
                )
                break
            except asyncio.TimeoutError:
                if attempt <= MAX_RETRIES:
                    log(f"  {ticker}  chunk {i+1}/{months_back}  timed out (attempt {attempt}), retrying...")
                    await asyncio.sleep(PACING_DELAY_SEC)
                else:
                    log(f"  {ticker}  chunk {i+1}/{months_back}  TIMED OUT after {MAX_RETRIES} retries — stopping here, partial history kept")

        if bars is None:
            break

        if not bars:
            log(f"  {ticker}  chunk {i+1}/{months_back}  empty — reached start of available history")
            break

        chunks.append(bars)
        earliest = bars[0].date
        log(f"  {ticker}  chunk {i+1}/{months_back}  {len(bars)} bars  back to {earliest}")
        end_dt = earliest   # next request ends where this one started

        await asyncio.sleep(PACING_DELAY_SEC)

    if not chunks:
        return pd.DataFrame()

    all_bars = [bar for chunk in chunks for bar in chunk]
    from ib_insync import util as ibu
    df = ibu.df(all_bars).set_index("date")
    df.index = pd.DatetimeIndex(df.index)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


def load_cached(ticker: str) -> "pd.DataFrame | None":
    """Read a ticker's cached raw bars, or None if nothing's cached yet."""
    path = _cache_path(ticker)
    if not path.exists():
        return None
    return pd.read_parquet(path)


def save_cache(ticker: str, df: pd.DataFrame) -> None:
    df.to_parquet(_cache_path(ticker))


async def update_all(tickers: list[str], months_back: int, log=print) -> None:
    ib = await _connect()
    log(f"Connected. Fetching {months_back} months x {len(tickers)} tickers "
        f"({months_back * len(tickers)} requests, ~{months_back * len(tickers) * (35 + PACING_DELAY_SEC) / 60:.0f} min estimated)...")
    try:
        for n, ticker in enumerate(tickers, start=1):
            log(f"[{n}/{len(tickers)}] {ticker}")
            df = await fetch_ticker_history(ib, ticker, months_back, log=log)
            if df.empty:
                log(f"  {ticker}  no data returned at all — skipping cache write")
                continue
            save_cache(ticker, df)
            log(f"  {ticker}  cached {len(df)} total bars "
                f"({df.index[0]} → {df.index[-1]}) → {_cache_path(ticker).name}")
    finally:
        ib.disconnect()


def fetch_ticker_from_cache(ticker: str) -> "pd.DataFrame | None":
    """Drop-in replacement for backtest.py's fetch_ticker() — same output
    shape (_process_df()'d), sourced from the local IBKR cache instead of
    a live yfinance call. Returns None if nothing's cached for this ticker."""
    raw = load_cached(ticker)
    if raw is None or raw.empty:
        return None
    return _process_df(raw.copy())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--months", type=int, default=24)
    parser.add_argument("--tickers", type=str, default=None, help="comma-separated, default: full WATCHLIST")
    args = parser.parse_args()

    tickers = args.tickers.split(",") if args.tickers else WATCHLIST
    t0 = time.time()
    asyncio.run(update_all(tickers, args.months))
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min.")
