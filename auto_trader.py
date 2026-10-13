"""
auto_trader.py — Automated options trader
Strategy : configured via config/strategy.toml ([setup].validated_setup)
           5-min bars, ATM puts
Broker   : Interactive Brokers via ib_insync

Modes (set in config/strategy.toml under [execution])
  dry_run = true   → logs signals only, no orders placed
  dry_run = false  → places real orders (paper or live, depending on tws_port)
  tws_port = 7497  → paper trading  (7496 for live)

Run: .venv/bin/python3 auto_trader.py
"""
import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

import logging
import subprocess
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import pytz
from ib_insync import IB, Option, Stock, LimitOrder, MarketOrder, util

util.logToConsole("ERROR")

from scanner import (
    WATCHLIST, ET, MIN_SCORE, MAX_SCORE,
    DAILY_0DTE_TICKERS, MWF_0DTE_TICKERS,
    detect_patterns, score_alert, _process_df,
    whipsaw_detected,
)
import signal_logger

# ── Config ────────────────────────────────────────────────────────────────────
# Every execution limit lives in config/strategy.toml (gitignored). The
# committed config/strategy.example.toml holds placeholders only, and ships
# with dry_run = true so a fresh clone cannot place an order.
from strategy_params import PARAMS, parse_hhmm

_EXEC = PARAMS["execution"]

DRY_RUN          = _EXEC["dry_run"]
TWS_PORT         = _EXEC["tws_port"]
CONTRACTS        = _EXEC["contracts"]
MAX_TRADES_DAY   = _EXEC["max_trades_day"]
MAX_DAILY_LOSS   = _EXEC["max_daily_loss"]
MAX_PREMIUM_PER_TRADE = _EXEC["max_premium_per_trade"]
MAX_CONCURRENT   = _EXEC["max_concurrent"]
ENTRY_OFFSET     = _EXEC["entry_offset"]
FILL_TIMEOUT     = _EXEC["fill_timeout"]
COOLDOWN_MIN     = _EXEC["cooldown_min"]
CLOSE_BEFORE     = parse_hhmm(_EXEC["close_before"])
NO_NEW_AFTER     = parse_hhmm(_EXEC["no_new_after"])
MAX_HOLD_BARS    = _EXEC["max_hold_bars"]
USE_ALPACA_FALLBACK = _EXEC["use_alpaca_fallback"]

LOG_FILE = Path(__file__).parent / "auto_trader.log"

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    force=True,   # override any handlers ib_insync already registered
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger("auto")


# ── Position state ────────────────────────────────────────────────────────────
@dataclass
class Position:
    ticker:       str
    opt_contract: object        # ib_insync Contract
    und_stop:     float         # underlying stop price (short: above entry)
    und_t1:       float         # underlying T1 price  (short: below entry)
    und_t2:       float         # underlying T2 price  (short: further below)
    entry_opt_px: float         # option premium paid per share
    qty:          int
    entry_time:   datetime
    score:        int
    ib_trade:     object = None # ib_insync Trade (None in DRY_RUN)


# Module-level state (resets at midnight via _reset_day)
_open:        dict[str, Position] = {}   # key = ticker
_cooldowns:   dict[str, datetime] = {}   # ticker → can re-enter after
_daily_trades = 0
_daily_pnl    = 0.0
_day_of:      Optional[date] = None


def _reset_day() -> None:
    global _daily_trades, _daily_pnl, _day_of, _open, _cooldowns
    today = date.today()
    if _day_of != today:
        _daily_trades = 0
        _daily_pnl    = 0.0
        _day_of       = today
        _open.clear()
        _cooldowns.clear()
        log.info(f"── New trading day: {today} ──────────────────────────────")


def _can_trade(ticker: str) -> tuple[bool, str]:
    if _daily_trades >= MAX_TRADES_DAY:
        return False, f"daily cap ({MAX_TRADES_DAY}) reached"
    if _daily_pnl <= MAX_DAILY_LOSS:
        return False, f"daily loss limit (${MAX_DAILY_LOSS}) hit"
    if len(_open) >= MAX_CONCURRENT:
        return False, f"max concurrent positions ({MAX_CONCURRENT}) open"
    if ticker in _open:
        return False, "already in position"
    if ticker in _cooldowns and datetime.now(ET) < _cooldowns[ticker]:
        delta = (_cooldowns[ticker] - datetime.now(ET)).seconds // 60
        return False, f"cooldown ({delta}m remaining)"
    now_time = datetime.now(ET).time()
    if now_time >= NO_NEW_AFTER:
        return False, "no new trades after 3:30 PM"
    return True, ""


# ── IBKR connection ───────────────────────────────────────────────────────────
async def _connect(port: int = TWS_PORT) -> IB:
    ib = IB()
    for p in (port, 7497 if port == 7496 else 7496):
        try:
            await ib.connectAsync("127.0.0.1", p, clientId=10, readonly=False)
            log.info(f"Connected to TWS on port {p} "
                     f"({'PAPER' if p == 7497 else 'LIVE'})")
            ib.reqMarketDataType(3)  # delayed data — no real-time subscription needed
            return ib
        except Exception:
            continue
    raise ConnectionError("Cannot connect to TWS — is it open with API enabled?")


# ── Data fetching ─────────────────────────────────────────────────────────────
def _bars_to_df(bars) -> Optional[pd.DataFrame]:
    if not bars:
        return None
    from ib_insync import util as ibu
    df = ibu.df(bars).set_index("date")
    df.index = pd.DatetimeIndex(df.index)
    return _process_df(df)


async def _fetch_bars(ib: IB, ticker: str) -> Optional[pd.DataFrame]:
    try:
        bars = await asyncio.wait_for(
            ib.reqHistoricalDataAsync(
                Stock(ticker, "SMART", "USD"),
                endDateTime="", durationStr="1 D",
                barSizeSetting="5 mins", whatToShow="TRADES",
                useRTH=False, formatDate=1,
            ), timeout=8
        )
        return _bars_to_df(bars)
    except Exception:
        return None


async def _fetch_all(ib: IB) -> dict[str, pd.DataFrame]:
    result = {}
    for ticker in WATCHLIST + ["SPY"]:
        df = await _fetch_bars(ib, ticker)
        if df is not None:
            result[ticker] = df
    return result


# ── Options contract lookup ───────────────────────────────────────────────────
def _target_expiry(ticker: str) -> str:
    """Return the target expiry as YYYYMMDD (IBKR format).

    Mirrors backtest.py's _dte() day-selection so live and backtest resolve
    the same calendar day: same-day for true daily-0DTE tickers, the next
    Mon/Wed/Fri for MWF single names on an off day, else nearest Friday.
    """
    today   = date.today()
    weekday = today.weekday()   # 0=Mon .. 4=Fri

    if ticker in DAILY_0DTE_TICKERS:
        return today.strftime("%Y%m%d")

    if ticker in MWF_0DTE_TICKERS:
        if weekday in (0, 2, 4):
            return today.strftime("%Y%m%d")
        for delta in range(1, 5):
            if (weekday + delta) % 7 in (0, 2, 4):
                return (today + timedelta(days=delta)).strftime("%Y%m%d")

    days = (4 - weekday) % 7   # nearest Friday (0 if today is Friday)
    return (today + timedelta(days=days)).strftime("%Y%m%d")


async def _find_atm_put(ib: IB, ticker: str, price: float) -> Optional[object]:
    expiry = _target_expiry(ticker)

    # Build candidate strikes around ATM
    # COIN, NFLX, BA trade in $5 increments; most ETFs/stocks in $1 or $2.50
    WIDE_STRIKE_TICKERS = {"COIN", "NFLX", "BA"}
    if price > 500:
        base   = round(price / 5) * 5
        steps  = [0, -5, 5, -10, 10, -15, 15, -20, 20]
    elif ticker in WIDE_STRIKE_TICKERS:
        base   = round(price / 5) * 5
        steps  = [0, -5, 5, -10, 10, -15, 15, -20, 20]
    elif price > 50:
        base   = round(price)
        steps  = [0, -1, 1, -2, 2, -3, 3, -5, 5, -7, 7, -10, 10]
    else:
        base   = round(price * 2) / 2
        steps  = [0, -0.5, 0.5, -1.0, 1.0, -1.5, 1.5, -2.0, 2.0]

    for d in steps:
        s = base + d
        if s <= 0:
            continue
        contract = Option(ticker, expiry, s, "P", "SMART", currency="USD")
        try:
            qualified = await asyncio.wait_for(
                ib.qualifyContractsAsync(contract), timeout=3
            )
            if qualified:
                return qualified[0]
        except Exception:
            continue
    return None


async def _get_mid(ib: IB, contract) -> Optional[float]:
    ticker = ib.reqMktData(contract, "", False, False)
    try:
        for _ in range(20):   # up to 10 s
            await asyncio.sleep(0.5)
            b, a = ticker.bid, ticker.ask
            if b and a and b > 0 and a > 0 and b < a:
                return (b + a) / 2
        return float(ticker.last) if ticker.last and ticker.last > 0 else None
    finally:
        ib.cancelMktData(contract)


async def _get_bid_ask(ib: IB, contract) -> tuple[Optional[float], Optional[float]]:
    ticker = ib.reqMktData(contract, "", False, False)
    try:
        for _ in range(20):   # up to 10 s
            await asyncio.sleep(0.5)
            b, a = ticker.bid, ticker.ask
            if b and a and b > 0 and a > 0 and b < a:
                return b, a
        return None, None
    finally:
        ib.cancelMktData(contract)


async def _validate_contracts(ib: IB, all_data: dict[str, pd.DataFrame]) -> None:
    """Resolve and log the actual option contract for every watchlist ticker
    at startup. Raises RuntimeError if any ticker can't resolve a contract —
    that's a structural problem (bad expiry/strike/ticker), not a data one,
    so it always halts startup. A missing bid/ask is logged as a warning,
    never a hard failure — this account doesn't carry live options market
    data entitlements, so a missing quote here reflects that account-level
    gap rather than a broken contract, and would otherwise fail every
    single ticker during market hours. Revisit this once/if a market data
    subscription is added.
    """
    log.info("Startup validation — resolving option contracts for all tickers...")
    failures = []

    for i, ticker in enumerate(WATCHLIST):
        if i > 0:
            await asyncio.sleep(1.0)   # avoid an IBKR API pacing violation across the whole watchlist

        df = all_data.get(ticker)
        if df is None or df.empty:
            log.error(f"  VALIDATE  {ticker:<6}  no price data — cannot resolve contract")
            failures.append(ticker)
            continue

        price = float(df.iloc[-1]["close"])
        contract = await _find_atm_put(ib, ticker, price)
        if contract is None:
            log.error(
                f"  VALIDATE  {ticker:<6}  could not resolve ATM PUT "
                f"(expiry={_target_expiry(ticker)}, ref price=${price:.2f})"
            )
            failures.append(ticker)
            continue

        bid, ask = await _get_bid_ask(ib, contract)
        if bid is None or ask is None:
            log.warning(
                f"  VALIDATE  {ticker:<6}  resolved {contract.localSymbol} "
                f"but no bid/ask quote available (not treated as a failure)"
            )
            continue

        log.info(
            f"  VALIDATE  {ticker:<6}  {contract.localSymbol:<18}  "
            f"expiry={contract.lastTradeDateOrContractMonth}  "
            f"bid=${bid:.2f}  ask=${ask:.2f}  spread=${ask - bid:.2f}"
        )

    if failures:
        raise RuntimeError(
            f"Startup validation failed for {len(failures)} ticker(s): "
            f"{', '.join(failures)} — see ERROR lines above. Fix before trading."
        )
    log.info(f"Startup validation passed — {len(WATCHLIST)}/{len(WATCHLIST)} tickers resolved.")


# ── Order placement ───────────────────────────────────────────────────────────
async def _buy_limit(ib: IB, contract, limit_px: float, qty: int
                     ) -> tuple[Optional[object], Optional[float]]:
    """Place a limit buy. Returns (trade, fill_price) or (None, None)."""
    order = LimitOrder("BUY", qty, round(limit_px, 2))
    trade = ib.placeOrder(contract, order)
    for _ in range(FILL_TIMEOUT):
        await asyncio.sleep(1)
        if trade.orderStatus.status == "Filled":
            return trade, trade.orderStatus.avgFillPrice
        if trade.orderStatus.status in ("Cancelled", "Inactive"):
            return None, None
    ib.cancelOrder(order)
    await asyncio.sleep(2)
    return None, None


async def _sell_market(ib: IB, contract, qty: int) -> Optional[float]:
    """Market sell. Returns fill price."""
    order = MarketOrder("SELL", qty)
    trade = ib.placeOrder(contract, order)
    for _ in range(30):
        await asyncio.sleep(1)
        if trade.orderStatus.status == "Filled":
            return trade.orderStatus.avgFillPrice
    return None


# ── Trade entry ───────────────────────────────────────────────────────────────
async def _enter_trade(ib: IB, alert: dict, df: pd.DataFrame) -> None:
    global _daily_trades

    ticker = alert["ticker"]
    ok, reason = _can_trade(ticker)
    if not ok:
        log.info(f"  SKIP  {ticker}  {reason}")
        signal_logger.record_rejection(alert, df, reason)
        return

    log.info(
        f"SIGNAL  {ticker}  score {alert['score']}/3  "
        f"entry ${alert['entry']}  stop ${alert['stop']}  "
        f"T1 ${alert['target1']}  T2 ${alert['target2']}"
    )

    if DRY_RUN:
        log.info(f"  [DRY RUN] would buy {CONTRACTS} ATM PUT — signal logged only")
        _notify(ticker, alert)
        _daily_trades += 1
        _cooldowns[ticker] = datetime.now(ET) + timedelta(minutes=COOLDOWN_MIN)
        return

    # Find option contract
    contract = await _find_atm_put(ib, ticker, alert["entry"])
    if contract is None:
        log.warning(f"  Could not find ATM PUT for {ticker} — skipping")
        return

    # Get mid price
    mid = await _get_mid(ib, contract)
    if (mid is None or mid <= 0) and USE_ALPACA_FALLBACK:
        try:
            import alpaca_options
            exp = contract.lastTradeDateOrContractMonth   # IBKR format YYYYMMDD
            exp_dashed = f"{exp[:4]}-{exp[4:6]}-{exp[6:]}"
            alp = alpaca_options.get_quote_for_contract(ticker, exp_dashed, contract.strike, "P")
        except Exception as e:
            alp = None
            log.warning(f"  Alpaca fallback failed for {ticker}: {e}")
        if alp:
            mid = alp["mid"]
            log.info(f"  Using Alpaca fallback quote for {ticker}: mid=${mid:.2f} (bid=${alp['bid']:.2f} ask=${alp['ask']:.2f})")

    if mid is None or mid <= 0:
        log.warning(f"  No market data for {contract.localSymbol} — skipping")
        return

    projected_premium = mid * 100 * CONTRACTS

    if projected_premium > MAX_PREMIUM_PER_TRADE:
        reason = f"premium cap (${projected_premium:.0f} > ${MAX_PREMIUM_PER_TRADE:.0f} max/trade)"
        log.warning(f"  SKIP  {ticker}  {reason}")
        signal_logger.record_rejection(alert, df, reason)
        return

    remaining_budget = _daily_pnl - MAX_DAILY_LOSS   # $ still left before the daily halt
    if projected_premium > remaining_budget:
        reason = (
            f"exceeds remaining daily loss budget "
            f"(${projected_premium:.0f} premium > ${remaining_budget:.0f} remaining)"
        )
        log.warning(f"  SKIP  {ticker}  {reason}")
        signal_logger.record_rejection(alert, df, reason)
        return

    limit_px = round(mid + ENTRY_OFFSET, 2)
    log.info(
        f"  ORDER  BUY {CONTRACTS} {contract.localSymbol} "
        f"@ ${limit_px:.2f} LMT  (mid ${mid:.2f})"
    )

    trade, fill_px = await _buy_limit(ib, contract, limit_px, CONTRACTS)
    if fill_px is None:
        log.warning(f"  Order not filled — skipping {ticker}")
        return

    log.info(f"  FILLED @ ${fill_px:.2f}  (${fill_px * 100:.0f}/contract)")

    pos = Position(
        ticker       = ticker,
        opt_contract = contract,
        und_stop     = alert["stop"],
        und_t1       = alert["target1"],
        und_t2       = alert["target2"],
        entry_opt_px = fill_px,
        qty          = CONTRACTS,
        entry_time   = datetime.now(ET),
        score        = alert["score"],
        ib_trade     = trade,
    )
    _open[ticker] = pos
    _daily_trades += 1
    _cooldowns[ticker] = datetime.now(ET) + timedelta(minutes=COOLDOWN_MIN)
    _notify(ticker, alert, fill_px)


# ── Position exit ─────────────────────────────────────────────────────────────
async def _exit_position(ib: IB, ticker: str, reason: str,
                         latest_bars: dict) -> None:
    global _daily_pnl
    pos = _open.get(ticker)
    if pos is None:
        return

    log.info(f"EXIT  {ticker}  reason={reason}")

    if DRY_RUN:
        del _open[ticker]
        return

    fill_px = await _sell_market(ib, pos.opt_contract, pos.qty)
    if fill_px is None:
        fill_px = pos.entry_opt_px   # fallback estimate
        log.warning(f"  No fill confirmation — estimating exit at ${fill_px:.2f}")

    pnl = (fill_px - pos.entry_opt_px) * 100 * pos.qty
    _daily_pnl += pnl
    del _open[ticker]

    sign = "+" if pnl >= 0 else ""
    log.info(
        f"  P&L  {sign}${pnl:.2f}  "
        f"(entry ${pos.entry_opt_px:.2f} → exit ${fill_px:.2f})  "
        f"daily total: {'+' if _daily_pnl >= 0 else ''}${_daily_pnl:.2f}"
    )


async def _check_exits(ib: IB, all_data: dict) -> None:
    now_time = datetime.now(ET).time()
    for ticker in list(_open.keys()):
        df = all_data.get(ticker)
        if df is None:
            continue
        mkt = df[df.index.time >= time(9, 30)]
        if mkt.empty:
            continue

        last  = mkt.iloc[-1]
        hi    = last["high"]
        lo    = last["low"]
        close = last["close"]
        pos   = _open[ticker]

        # Force-close before market close
        if now_time >= CLOSE_BEFORE:
            await _exit_position(ib, ticker, "end-of-day", all_data)
            continue

        # Short position exits: stop is above, targets are below
        if hi >= pos.und_stop:
            await _exit_position(ib, ticker, f"stop (${pos.und_stop})", all_data)
        elif lo <= pos.und_t2:
            await _exit_position(ib, ticker, f"T2 (${pos.und_t2})", all_data)
        elif lo <= pos.und_t1:
            await _exit_position(ib, ticker, f"T1 (${pos.und_t1})", all_data)
        elif len(mkt[mkt.index > pos.entry_time]) >= MAX_HOLD_BARS:
            await _exit_position(ib, ticker, f"max hold ({MAX_HOLD_BARS} bars)", all_data)


# ── Notification ──────────────────────────────────────────────────────────────
def _notify(ticker: str, alert: dict, fill_px: Optional[float] = None) -> None:
    title = f"{ticker} — Vol Div Short  [AUTO{'•DRY' if DRY_RUN else ''}]"
    body  = (f"Score {alert['score']}/3  Entry ${alert['entry']}  "
             f"Stop ${alert['stop']}")
    if fill_px:
        body += f"  Filled PUT @ ${fill_px:.2f}"
    subprocess.run(
        ["osascript", "-e",
         f'display notification "{body}" with title "{title}" sound name "Funk"'],
        capture_output=True,
    )


# ── Market hours ──────────────────────────────────────────────────────────────
def _market_open() -> bool:
    t = datetime.now(ET).time()
    return time(9, 25) <= t <= time(16, 5)


# ── Main loop ─────────────────────────────────────────────────────────────────
async def _main() -> None:
    ib = await _connect()

    startup_data = await _fetch_all(ib)
    await _validate_contracts(ib, startup_data)

    while True:
        _reset_day()

        if not _market_open():
            now_str = datetime.now(ET).strftime("%H:%M:%S")
            print(f"\r[{now_str} ET] Market closed — waiting...", end="", flush=True)
            await asyncio.sleep(30)
            continue

        now_str = datetime.now(ET).strftime("%H:%M:%S")
        log.info(
            f"[{now_str}] scan  trades={_daily_trades}/{MAX_TRADES_DAY}  "
            f"open={len(_open)}  pnl={'+' if _daily_pnl>=0 else ''}${_daily_pnl:.2f}"
        )

        # Fetch all bars (includes SPY for market_aligned check)
        all_data = await _fetch_all(ib)

        # Check exits on open positions first
        if _open:
            await _check_exits(ib, all_data)

        # Daily loss limit check
        if _daily_pnl <= MAX_DAILY_LOSS:
            log.warning(
                f"Daily loss limit reached (${_daily_pnl:.2f}) — "
                f"no new trades for the rest of the day"
            )
            await asyncio.sleep(60)
            continue

        # Scan for new signals
        for ticker, df in all_data.items():
            if ticker == "SPY":
                continue
            active, watching = detect_patterns(ticker, df)

            for alert in active:
                sc = score_alert(alert, df, all_data)
                alert["score"] = sc
                if sc < MIN_SCORE or sc > MAX_SCORE:
                    signal_logger.record_rejection(
                        alert, df, f"score_gate (score={sc}, need exactly {MIN_SCORE})"
                    )
                    continue
                await _enter_trade(ib, alert, df)

            # Watching-bucket setups (including Vol Div Short on a non-weak
            # day) never trade — score them anyway, log-only, for analysis.
            for alert in watching:
                alert["score"] = score_alert(alert, df, all_data)
                reason = alert.get("watch_reason", "setup_not_live (not Volume Divergence Short)")
                signal_logger.record_rejection(alert, df, reason)

        signal_logger.process_pending(all_data)

        await asyncio.sleep(60)   # 5-min bars update ~every 60s


def main() -> None:
    mode = "DRY RUN — signals logged, no orders placed" if DRY_RUN else (
        "PAPER TRADING" if TWS_PORT == 7497 else "⚠  LIVE TRADING"
    )
    print(f"""
┌─────────────────────────────────────────────────────────┐
│  AUTO TRADER  —  Vol Divergence Short · Score 3          │
│  Mode     : {mode:<44} │
│  Max/day  : {MAX_TRADES_DAY} trades   Max loss: ${abs(MAX_DAILY_LOSS):.0f}/day            │
│  Contracts: {CONTRACTS}           Max concurrent: {MAX_CONCURRENT}                │
│  Log file : auto_trader.log                             │
│  Stop     : Ctrl+C                                      │
└─────────────────────────────────────────────────────────┘
""")
    asyncio.run(_main())


if __name__ == "__main__":
    main()
