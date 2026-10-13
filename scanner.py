import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())  # must be before ib_insync import on Python 3.12+

import math
import subprocess
import time as time_module
from datetime import date, datetime, time

import pandas as pd
import pytz
import yfinance as yf  # used only for options chain data
from ib_insync import IB, Stock, util

util.logToConsole("ERROR")

ET = pytz.timezone("America/New_York")

# Validated watchlist — MSFT/GOOGL/TSLA/META/SPY/PLTR removed (negative on 5-min Vol Div Short)
from strategy_params import PARAMS, parse_hhmm

# Every tunable lives in config/strategy.toml -- see strategy_params.py
_SESSION  = PARAMS["session"]
_VOL      = PARAMS["volume"]
_BREAKOUT = PARAMS["breakout"]
_VOLDIV   = PARAMS["volume_divergence"]
_ORB      = PARAMS["opening_range"]
_VWAPDIV  = PARAMS["vwap_divergence"]
_SCORING  = PARAMS["scoring"]
_REF      = PARAMS["scoring"]["reference"]
_WHIPSAW  = PARAMS["whipsaw"]

MARKET_OPEN      = parse_hhmm(_SESSION["market_open"])
MARKET_CLOSE     = parse_hhmm(_SESSION["market_close"])
SCAN_START       = parse_hhmm(_SESSION["scan_start"])
PREMARKET_CUTOFF = parse_hhmm(_SESSION["premarket_cutoff"])
POWER_WINDOWS    = [
    (parse_hhmm(w["start"]), parse_hhmm(w["end"])) for w in _SESSION["power_windows"]
]

WATCHLIST = list(PARAMS["universe"]["watchlist"])

# Only this setup is backtested and validated
VALIDATED_SETUP = PARAMS["setup"]["validated_setup"]
MIN_SCORE = PARAMS["setup"]["min_score"]
MAX_SCORE = PARAMS["setup"]["max_score"]

DAILY_0DTE_TICKERS = set(PARAMS["universe"]["daily_0dte"])
MWF_0DTE_TICKERS   = set(PARAMS["universe"]["mwf_0dte"])

# ANSI colors
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"

# Daily alert log — reset at midnight
_alert_log: list[dict] = []
_log_date: "date | None" = None


def _log_alert(alert: dict) -> None:
    global _alert_log, _log_date
    today = date.today()
    if _log_date != today:
        _alert_log.clear()
        _log_date = today
    _alert_log.append({
        "time":   datetime.now(ET).strftime("%H:%M"),
        "ticker": alert["ticker"],
        "setup":  alert["setup"],
        "bias":   alert["bias"],
        "entry":  alert["entry"],
        "score":  alert.get("score", 0),
    })


def print_alert_history() -> None:
    if not _alert_log:
        return
    print(f"\n{DIM}{'─' * 56}")
    print(f"  Today's alerts ({len(_alert_log)} total)")
    print(f"{'─' * 56}{RESET}")
    for a in _alert_log:
        print(f"  {DIM}{a['time']}{RESET}  "
              f"{BOLD}{a['ticker']:<6}{RESET}  "
              f"{RED}{a['setup']:<28}{RESET}  "
              f"{RED}[SHORT]{RESET}  "
              f"{YELLOW}${a['entry']}{RESET}  "
              f"{DIM}score {a['score']}{RESET}")
    print(f"{DIM}{'─' * 56}{RESET}")


def _process_df(df: pd.DataFrame) -> "pd.DataFrame | None":
    df.columns = [col.lower() for col in df.columns]
    df = df.drop(columns=[c for c in df.columns if c not in ("open","high","low","close","volume")], errors="ignore")
    df = df.dropna(how="all")
    if df.empty:
        return None
    if df.index.tzinfo is None:
        df.index = pd.DatetimeIndex(df.index).tz_localize("UTC").tz_convert(ET)
    else:
        df.index = pd.DatetimeIndex(df.index).tz_convert(ET)
    mkt_mask = df.index.time >= MARKET_OPEN
    if mkt_mask.any():
        mkt = df.loc[mkt_mask]
        df["vwap"] = float("nan")
        df.loc[mkt_mask, "vwap"] = (
            (mkt["close"] * mkt["volume"]).cumsum() / mkt["volume"].cumsum()
        )
        hl = (mkt["high"] - mkt["low"]).replace(0, float("nan"))
        bar_delta = ((mkt["close"] - mkt["open"]) / hl) * mkt["volume"]
        df["delta"] = float("nan")
        df["cum_delta"] = float("nan")
        df.loc[mkt_mask, "delta"] = bar_delta
        df.loc[mkt_mask, "cum_delta"] = bar_delta.cumsum()
    return df


def _ibkr_bars_to_df(bars) -> "pd.DataFrame | None":
    if not bars:
        return None
    df = util.df(bars).set_index("date")
    df.index = pd.DatetimeIndex(df.index)
    return _process_df(df)


async def _ibkr_connect() -> IB:
    ib = IB()
    for port in (7497, 7496):
        try:
            await ib.connectAsync("127.0.0.1", port, clientId=1, readonly=True)
            return ib
        except Exception:
            continue
    raise ConnectionError("Could not connect to TWS. Make sure it is open and API is enabled.")


async def _fetch_data_async(ticker: str) -> "pd.DataFrame | None":
    ib = await _ibkr_connect()
    try:
        bars = await asyncio.wait_for(
            ib.reqHistoricalDataAsync(
                Stock(ticker, "SMART", "USD"),
                endDateTime="", durationStr="1 D",
                barSizeSetting="5 mins", whatToShow="TRADES",
                useRTH=False, formatDate=1,
            ), timeout=8
        )
        return _ibkr_bars_to_df(bars)
    except (asyncio.TimeoutError, Exception):
        return None
    finally:
        ib.disconnect()


async def _fetch_all_async() -> dict[str, pd.DataFrame]:
    result = {}
    ib = await _ibkr_connect()
    try:
        for ticker in WATCHLIST:
            try:
                bars = await asyncio.wait_for(
                    ib.reqHistoricalDataAsync(
                        Stock(ticker, "SMART", "USD"),
                        endDateTime="", durationStr="1 D",
                        barSizeSetting="5 mins", whatToShow="TRADES",
                        useRTH=False, formatDate=1,
                    ), timeout=8
                )
                processed = _ibkr_bars_to_df(bars)
                if processed is not None:
                    result[ticker] = processed
            except (asyncio.TimeoutError, Exception):
                continue
    finally:
        ib.disconnect()
    return result


def fetch_data(ticker: str) -> "pd.DataFrame | None":
    try:
        return asyncio.run(_fetch_data_async(ticker))
    except Exception:
        return None


def fetch_all() -> dict[str, pd.DataFrame]:
    try:
        return asyncio.run(_fetch_all_async())
    except ConnectionError as e:
        print(f"\n{RED}{e}{RESET}")
        print(f"{DIM}Make sure TWS is open: Edit → Global Configuration → API → Settings → Enable Socket Clients{RESET}")
        return {}
    except Exception:
        return {}


def get_pm_levels(df):
    pm = df[df.index.time < PREMARKET_CUTOFF]
    if pm.empty:
        return None, None
    return round(pm["high"].max(), 2), round(pm["low"].min(), 2)


def _stock_is_weak(market_df: pd.DataFrame) -> bool:
    """True when stock is below VWAP AND below the day's opening price."""
    if len(market_df) < 2:
        return False
    last = market_df.iloc[-1]
    vwap = last.get("vwap", float("nan"))
    if pd.isna(vwap) or vwap == 0:
        return False
    day_open = market_df.iloc[0]["open"]
    return last["close"] < vwap and last["close"] < day_open


def detect_patterns(ticker, df):
    """
    Returns (active_alerts, watching_alerts).
    active_alerts  — Vol Div Short on a weak day. Backtested and validated.
    watching_alerts — all other patterns + Vol Div Short on a non-weak day.
                      Not yet validated — observe only.
    """
    active   = []
    watching = []

    pm_high, pm_low = get_pm_levels(df)
    if pm_high is None:
        return active, watching

    market_df = df[df.index.time >= MARKET_OPEN].copy()
    if len(market_df) < 3:
        return active, watching

    avg_vol_series = market_df["volume"].rolling(
        _VOL["avg_lookback"], min_periods=_VOL["avg_min_periods"]
    ).mean()
    avg_vol = avg_vol_series.iloc[-1]
    if pd.isna(avg_vol) or avg_vol == 0:
        return active, watching

    c    = market_df.iloc[-1]
    prev = market_df.iloc[-2]
    vol_ratio = c["volume"] / avg_vol

    vwap = c.get("vwap", float("nan"))
    vwap_valid = not pd.isna(vwap) and vwap > 0

    # ── PM HIGH BREAKOUT (watching) ──────────────────────────────
    if prev["close"] <= pm_high and c["close"] > pm_high and vol_ratio >= _BREAKOUT["min_vol_ratio"]:
        stop = c["low"]
        risk = c["close"] - stop
        watching.append({
            "setup":     "PM High Breakout",
            "bias":      "LONG",
            "ticker":    ticker,
            "entry":     round(c["close"], 2),
            "stop":      round(stop, 2),
            "target1":   round(c["close"] + risk * _BREAKOUT["target1_r"], 2),
            "target2":   round(c["close"] + risk * _BREAKOUT["target2_r"], 2),
            "vol_ratio": round(vol_ratio, 1),
            "rr":        "1:2 — 1:3",
            "note":      f"Broke above PM High (${pm_high}) with {vol_ratio:.1f}x vol",
        })

    # ── PM LOW BREAKDOWN (watching) ──────────────────────────────
    if prev["close"] >= pm_low and c["close"] < pm_low and vol_ratio >= _BREAKOUT["min_vol_ratio"]:
        stop = c["high"]
        risk = stop - c["close"]
        watching.append({
            "setup":     "PM Low Breakdown",
            "bias":      "SHORT",
            "ticker":    ticker,
            "entry":     round(c["close"], 2),
            "stop":      round(stop, 2),
            "target1":   round(c["close"] - risk * _BREAKOUT["target1_r"], 2),
            "target2":   round(c["close"] - risk * _BREAKOUT["target2_r"], 2),
            "vol_ratio": round(vol_ratio, 1),
            "rr":        "1:2 — 1:3",
            "note":      f"Broke below PM Low (${pm_low}) with {vol_ratio:.1f}x vol",
        })

    # ── VOLUME DIVERGENCE SHORT ───────────────────────────────────
    _vd_n = _VOLDIV["lookback_bars"]
    if len(market_df) >= _vd_n:
        last5 = market_df.iloc[-_vd_n:]
        vols  = last5["volume"].values
        vol_declining = sum(
            vols[i] < vols[i - 1] for i in range(1, _vd_n)
        ) >= _VOLDIV["min_declining_bars"]

        if last5["high"].iloc[-1] > last5["high"].iloc[0] and vol_declining and c["close"] < c["open"]:
            stop     = last5["high"].max()
            risk     = stop - c["close"]
            near_res = abs(c["high"] - pm_high) / pm_high < _VOLDIV["level_proximity"]
            is_weak  = _stock_is_weak(market_df)
            alert = {
                "setup":     VALIDATED_SETUP,
                "bias":      "SHORT",
                "color":     RED,
                "ticker":    ticker,
                "entry":     round(c["close"], 2),
                "stop":      round(stop, 2),
                "target1":   round(c["close"] - risk * _VOLDIV["target1_r"], 2),
                "target2":   round(c["close"] - risk * _VOLDIV["target2_r"], 2),
                "vol_ratio": round(vol_ratio, 1),
                "rr":        "1:1.5 — 1:2.5",
                "note":      f"Higher highs, declining volume{' — near PM High resistance' if near_res else ''}",
                "weak_day":  is_weak,
                "vwap":      round(vwap, 2) if vwap_valid else None,
            }
            if is_weak:
                active.append(alert)
            else:
                alert["watch_reason"] = "stock not weak (above VWAP or above open)"
                watching.append(alert)

        # ── VOLUME DIVERGENCE LONG (watching) ────────────────────
        if last5["low"].iloc[-1] < last5["low"].iloc[0] and vol_declining and c["close"] > c["open"]:
            stop = last5["low"].min()
            risk = c["close"] - stop
            near_sup = abs(c["low"] - pm_low) / pm_low < _VOLDIV["level_proximity"]
            watching.append({
                "setup":     "Volume Divergence Long",
                "bias":      "LONG",
                "ticker":    ticker,
                "entry":     round(c["close"], 2),
                "stop":      round(stop, 2),
                "target1":   round(c["close"] + risk * _VOLDIV["target1_r"], 2),
                "target2":   round(c["close"] + risk * _VOLDIV["target2_r"], 2),
                "vol_ratio": round(vol_ratio, 1),
                "rr":        "1:1.5 — 1:2.5",
                "note":      f"Lower lows, volume drying up{' — at PM Low support' if near_sup else ''}",
            })

    # ── OPENING RANGE BREAKOUT (watching) ────────────────────────
    n = len(market_df)
    _or_n = _ORB["window_bars"]
    if _or_n < n < _ORB["max_bars_elapsed"]:
        or_candles = market_df.iloc[:_or_n]
        or_high    = or_candles["high"].max()
        or_low     = or_candles["low"].min()
        or_range   = or_high - or_low

        if prev["close"] <= or_high and c["close"] > or_high and vol_ratio >= _ORB["min_vol_ratio"]:
            watching.append({
                "setup":     "ORB Long",
                "bias":      "LONG",
                "ticker":    ticker,
                "entry":     round(c["close"], 2),
                "stop":      round(or_low, 2),
                "target1":   round(or_high + or_range * _ORB["target1_range_mult"], 2),
                "target2":   round(or_high + or_range * _ORB["target2_range_mult"], 2),
                "vol_ratio": round(vol_ratio, 1),
                "rr":        "1:1 — 1:2",
                "note":      f"OR (${or_low:.2f}–${or_high:.2f}) broken up with {vol_ratio:.1f}x vol",
            })

        if prev["close"] >= or_low and c["close"] < or_low and vol_ratio >= _ORB["min_vol_ratio"]:
            watching.append({
                "setup":     "ORB Short",
                "bias":      "SHORT",
                "ticker":    ticker,
                "entry":     round(c["close"], 2),
                "stop":      round(or_high, 2),
                "target1":   round(or_low - or_range * _ORB["target1_range_mult"], 2),
                "target2":   round(or_low - or_range * _ORB["target2_range_mult"], 2),
                "vol_ratio": round(vol_ratio, 1),
                "rr":        "1:1 — 1:2",
                "note":      f"OR broken down with {vol_ratio:.1f}x vol",
            })

    # ── VWAP VOLUME DIVERGENCE (watching) ────────────────────────
    _vw_n = _VWAPDIV["lookback_bars"]
    if vwap_valid and len(market_df) >= _vw_n:
        last4 = market_df.iloc[-_vw_n:]
        vols4 = last4["volume"].values
        vol_declining4 = sum(
            vols4[i] < vols4[i - 1] for i in range(1, _vw_n)
        ) >= _VWAPDIV["min_declining_bars"]
        near_vwap = abs(c["close"] - vwap) / vwap < _VWAPDIV["vwap_proximity"]

        if near_vwap and vol_declining4:
            if c["close"] <= vwap and c["close"] > c["open"]:
                stop = last4["low"].min()
                risk = c["close"] - stop
                if risk > 0:
                    watching.append({
                        "setup":     "VWAP Vol Divergence Long",
                        "bias":      "LONG",
                        "ticker":    ticker,
                        "entry":     round(c["close"], 2),
                        "stop":      round(stop, 2),
                        "target1":   round(c["close"] + risk * _VWAPDIV["target1_r"], 2),
                        "target2":   round(c["close"] + risk * _VWAPDIV["target2_r"], 2),
                        "vol_ratio": round(vol_ratio, 1),
                        "rr":        "1:1.5 — 1:2.5",
                        "note":      f"Vol drying up at VWAP (${vwap:.2f}) — reclaim",
                    })

            if c["close"] >= vwap and c["close"] < c["open"]:
                stop = last4["high"].max()
                risk = stop - c["close"]
                if risk > 0:
                    watching.append({
                        "setup":     "VWAP Vol Divergence Short",
                        "bias":      "SHORT",
                        "ticker":    ticker,
                        "entry":     round(c["close"], 2),
                        "stop":      round(stop, 2),
                        "target1":   round(c["close"] - risk * _VWAPDIV["target1_r"], 2),
                        "target2":   round(c["close"] - risk * _VWAPDIV["target2_r"], 2),
                        "vol_ratio": round(vol_ratio, 1),
                        "rr":        "1:1.5 — 1:2.5",
                        "note":      f"Vol drying up at VWAP (${vwap:.2f}) — rejection",
                    })

    return active, watching


def _norm_cdf(x):
    return (1.0 + math.erf(x / math.sqrt(2))) / 2.0


def _bs_delta(S, K, T, r, sigma, option_type):
    if T <= 0 or sigma <= 0:
        return 0.5
    d1 = (math.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * math.sqrt(T))
    return _norm_cdf(d1) if option_type == "call" else _norm_cdf(d1) - 1.0


def _preferred_expiry(ticker: str, exps: list[str]) -> "tuple[str, str]":
    """
    Returns (expiry_date_str, label).
    Skips today's 0DTE for daily/MWF tickers and uses the next weekly Friday.
    """
    today = date.today()
    weekday = today.weekday()

    is_0dte_day = (
        ticker in DAILY_0DTE_TICKERS or
        (ticker in MWF_0DTE_TICKERS and weekday in (0, 2, 4))
    )

    if is_0dte_day:
        valid = [e for e in exps if datetime.strptime(e, "%Y-%m-%d").date() > today]
        label = "next weekly (skipping 0DTE)"
    else:
        valid = [e for e in exps if datetime.strptime(e, "%Y-%m-%d").date() >= today]
        label = "nearest expiry"

    if not valid:
        valid = [e for e in exps if datetime.strptime(e, "%Y-%m-%d").date() >= today]
        label = "nearest expiry (fallback)"

    return (valid[0], label) if valid else ("", label)


def get_options_context(ticker, price, bias, target1, target2, stop):
    try:
        t    = yf.Ticker(ticker)
        exps = t.options
        if not exps:
            return None

        exp, exp_label = _preferred_expiry(ticker, exps)
        if not exp:
            return None

        chain     = t.option_chain(exp)
        opt_type  = "call" if bias == "LONG" else "put"
        contracts = chain.calls if bias == "LONG" else chain.puts
        if contracts.empty:
            return None

        contracts         = contracts.copy()
        contracts["diff"] = abs(contracts["strike"] - price)
        atm    = contracts.sort_values("diff").iloc[0]
        bid    = float(atm["bid"])
        ask    = float(atm["ask"])
        mid    = (bid + ask) / 2 if bid > 0 and ask > 0 else float(atm["lastPrice"])
        if mid <= 0:
            return None

        iv     = float(atm.get("impliedVolatility", 0.30))
        strike = float(atm["strike"])
        today  = date.today()
        days   = max((datetime.strptime(exp, "%Y-%m-%d").date() - today).days, 0)
        T      = max(days, 0.5) / 365.0
        delta  = abs(_bs_delta(price, strike, T, 0.05, iv, opt_type))

        def est(target_price):
            move = (target_price - price) if bias == "LONG" else (price - target_price)
            return round(max(0.01, mid + delta * move), 2)

        val_t1, val_t2, val_stop = est(target1), est(target2), est(stop)
        return {
            "type":      "CALL" if bias == "LONG" else "PUT",
            "strike":    strike,
            "expiry":    exp,
            "exp_label": exp_label,
            "days":      days,
            "mid":       round(mid, 2),
            "premium":   round(mid * 100, 2),
            "delta":     round(delta, 2),
            "iv":        round(iv * 100, 1),
            "val_t1":    val_t1,  "pnl_t1":   round((val_t1  - mid) * 100, 2),
            "val_t2":    val_t2,  "pnl_t2":   round((val_t2  - mid) * 100, 2),
            "val_stop":  val_stop,"pnl_stop":  round((val_stop - mid) * 100, 2),
        }
    except Exception:
        return None


def print_options(opt):
    if opt is None:
        print(f"  {DIM}Options: unavailable{RESET}")
        return
    pnl_color = lambda v: GREEN if v >= 0 else RED
    fmt_pnl   = lambda v: f"+${v:.2f}" if v >= 0 else f"-${abs(v):.2f}"
    print(f"  {DIM}{'─' * 54}{RESET}")
    print(f"  {BOLD}{RED}{opt['type']}{RESET}  "
          f"${opt['strike']} strike  "
          f"Exp {opt['expiry']} ({opt['days']}d)  "
          f"{DIM}{opt['exp_label']}{RESET}  "
          f"IV {opt['iv']}%  Δ {opt['delta']}")
    print(f"  Premium: {YELLOW}${opt['mid']}/share{RESET}  "
          f"({YELLOW}${opt['premium']}/contract{RESET})")
    print(f"  Stop: ≈ {RED}${opt['val_stop']}{RESET} "
          f"({pnl_color(opt['pnl_stop'])}{fmt_pnl(opt['pnl_stop'])}/contract{RESET})  "
          f"T1: ≈ {GREEN}${opt['val_t1']}{RESET} "
          f"({pnl_color(opt['pnl_t1'])}{fmt_pnl(opt['pnl_t1'])}/contract{RESET})  "
          f"T2: ≈ {GREEN}${opt['val_t2']}{RESET} "
          f"({pnl_color(opt['pnl_t2'])}{fmt_pnl(opt['pnl_t2'])}/contract{RESET})")


def notify(alert):
    title   = f"{alert['ticker']} — {alert['setup']} [SHORT]"
    score   = alert.get("score", "?")
    message = f"Entry ${alert['entry']}  Stop ${alert['stop']}  Score {score}/3  Vol {alert['vol_ratio']}x"
    subprocess.run([
        "osascript", "-e",
        f'display notification "{message}" with title "{title}" sound name "Funk"'
    ], capture_output=True)


def print_alert(alert):
    score = alert.get("score", 0)

    # Build condition tags for score breakdown
    tags = []
    if alert.get("cond_vol_div"):   tags.append(f"{GREEN}VolDiv✓{RESET}")
    if alert.get("cond_mkt"):       tags.append(f"{GREEN}MktWeak✓{RESET}")
    if alert.get("cond_clean"):     tags.append(f"{GREEN}Clean✓{RESET}")
    if alert.get("cond_power"):     tags.append(f"{GREEN}PowerHour✓{RESET}")
    tag_str = "  ".join(tags) if tags else ""

    vwap_str = f"  VWAP ${alert['vwap']}" if alert.get("vwap") else ""

    print(f"\n{BOLD}{RED}{'━' * 54}{RESET}")
    print(f"  {BOLD}{alert['ticker']}{RESET}  {RED}{alert['setup']}{RESET}  {RED}[SHORT]{RESET}  "
          f"{DIM}score {score}/3{RESET}")
    print(f"  Entry: {YELLOW}${alert['entry']}{RESET}  "
          f"Stop: {RED}${alert['stop']}{RESET}  "
          f"R:R: {alert['rr']}{vwap_str}")
    print(f"  T1: {GREEN}${alert['target1']}{RESET}  "
          f"T2: {GREEN}${alert['target2']}{RESET}  "
          f"Vol: {YELLOW}{alert['vol_ratio']}x{RESET}")
    if tag_str:
        print(f"  {tag_str}")
    print(f"  {DIM}{alert['note']}{RESET}")
    opt = get_options_context(
        alert["ticker"], alert["entry"], alert["bias"],
        alert["target1"], alert["target2"], alert["stop"],
    )
    print_options(opt)
    print(f"{BOLD}{RED}{'━' * 54}{RESET}")
    _log_alert(alert)
    notify(alert)


def print_watching(watching: list[dict]) -> None:
    if not watching:
        return
    print(f"\n{DIM}{'─' * 54}")
    print(f"  Watching ({len(watching)})  —  not yet validated, observe only")
    print(f"{'─' * 54}{RESET}")
    for a in watching:
        bias_sym   = "▲" if a["bias"] == "LONG" else "▼"
        reason     = a.get("watch_reason", "")
        reason_str = f"  {DIM}({reason}){RESET}" if reason else ""
        print(f"  {DIM}{bias_sym} {a['ticker']:<6}  {a['setup']:<28}  "
              f"${a['entry']}  Vol {a['vol_ratio']}x{RESET}{reason_str}")
    print(f"{DIM}{'─' * 54}{RESET}")


def print_market_context(all_data: dict) -> None:
    """Print SPY and QQQ bias at the top of each scan."""
    refs = [("SPY", "SPY"), ("QQQ", "QQQ")]
    parts = []
    for label, ticker in refs:
        df = all_data.get(ticker)
        if df is None:
            continue
        mkt = df[df.index.time >= MARKET_OPEN]
        if mkt.empty:
            continue
        last     = mkt.iloc[-1]
        vwap     = last.get("vwap", float("nan"))
        day_open = mkt.iloc[0]["open"]
        close    = last["close"]

        if pd.isna(vwap) or vwap == 0:
            parts.append(f"{label} ${close:.2f}")
            continue

        vs_vwap    = close - vwap
        vs_open    = close - day_open
        is_weak    = close < vwap and close < day_open
        bias_label = f"{RED}WEAK{RESET}" if is_weak else f"{GREEN}STRONG{RESET}"
        arrow      = "▼" if vs_vwap < 0 else "▲"

        parts.append(
            f"{BOLD}{label}{RESET} ${close:.2f}  "
            f"{arrow} {abs(vs_vwap):.2f} vs VWAP  "
            f"{'▼' if vs_open < 0 else '▲'} {abs(vs_open):.2f} vs open  "
            f"[{bias_label}]"
        )

    if parts:
        print(f"\n{DIM}Market context:{RESET}  " + "    ".join(parts))


def in_power_hour() -> bool:
    t = datetime.now(ET).time()
    return any(start <= t <= end for start, end in POWER_WINDOWS)


def market_aligned(ticker: str, bias: str, market_data: dict) -> bool:
    ref = (_REF["nasdaq_proxy"] if ticker in set(_REF["nasdaq_symbols"])
           else _REF["default_proxy"])
    df  = market_data.get(ref)
    if df is None or df.empty:
        return False
    mkt  = df[df.index.time >= MARKET_OPEN]
    if mkt.empty:
        return False
    c    = mkt.iloc[-1]
    vwap = c.get("vwap", float("nan"))
    if pd.isna(vwap) or vwap == 0:
        return False
    return c["close"] > vwap if bias == "LONG" else c["close"] < vwap


def whipsaw_detected(df: pd.DataFrame) -> bool:
    market_df = df[df.index.time >= MARKET_OPEN]
    _n = _WHIPSAW["lookback_bars"]
    if len(market_df) < _n:
        return False
    closes     = market_df["close"].iloc[-_n:].values
    directions = [1 if closes[i] > closes[i - 1] else -1 for i in range(1, len(closes))]
    return all(directions[i] != directions[i - 1] for i in range(1, len(directions)))


def score_alert(alert: dict, df: pd.DataFrame, market_data: dict) -> int:
    """
    Point weights come from [scoring] in config/strategy.toml; the tradeable
    band is [setup].min_score .. [setup].max_score.
    Conditions stored on the alert dict for display.
    """
    score = 0

    cond_vol_div = "Volume Divergence" in alert["setup"] or "VWAP Vol Divergence" in alert["setup"]
    if "VWAP Vol Divergence" in alert["setup"]:
        score += _SCORING["vwap_divergence_setup"]
    elif cond_vol_div:
        score += _SCORING["volume_divergence_setup"]

    cond_mkt = market_aligned(alert["ticker"], alert["bias"], market_data)
    if cond_mkt:
        score += _SCORING["market_aligned"]

    cond_clean = not whipsaw_detected(df)
    if cond_clean:
        score += _SCORING["no_whipsaw"]

    cond_power = in_power_hour()
    if cond_power:
        score += _SCORING["in_power_hour"]

    alert["cond_vol_div"] = cond_vol_div
    alert["cond_mkt"]     = cond_mkt
    alert["cond_clean"]   = cond_clean
    alert["cond_power"]   = cond_power

    return score


def is_market_hours():
    t = datetime.now(ET).time()
    return SCAN_START <= t <= MARKET_CLOSE


def run_scan():
    now = datetime.now(ET)
    print_alert_history()
    print(f"\n{DIM}[{now.strftime('%H:%M:%S')} ET] Scanning {len(WATCHLIST)} tickers...{RESET}",
          end="", flush=True)

    all_data = fetch_all()

    # Market context header (uses SPY even though it's not in watchlist)
    spy_df = fetch_data("SPY") if "SPY" not in all_data else all_data["SPY"]
    context_data = dict(all_data)
    if spy_df is not None:
        context_data["SPY"] = spy_df
    print_market_context(context_data)

    fired        = 0
    all_watching = []

    for ticker, df in all_data.items():
        active, watching = detect_patterns(ticker, df)
        all_watching.extend(watching)

        for alert in active:
            score = score_alert(alert, df, context_data)
            if score < MIN_SCORE or score > MAX_SCORE:
                # Score 1 underperforms on 5-min; score 4 fires late — skip both
                continue
            alert["score"] = score
            print_alert(alert)
            fired += 1

    if fired == 0:
        print(f" {DIM}no active signals{RESET}")
    else:
        print(f" {GREEN}{fired} signal(s){RESET}")

    print_watching(all_watching)


def main():
    print(f"\n{BOLD}Vol Divergence Short Scanner  —  5-min bars{RESET}")
    print(f"Watchlist ({len(WATCHLIST)}): {', '.join(WATCHLIST)}")
    print(f"Active setup : Volume Divergence Short  (weak-day filter · score 3 only)")
    print(f"Watching     : PM High/Low · ORB · VWAP Vol Div · Vol Div Long  (unvalidated)")
    print(f"Runs every 60 sec  ·  9:25 AM – 4:00 PM ET  ·  Ctrl+C to stop\n")

    while True:
        if is_market_hours():
            run_scan()
        else:
            now = datetime.now(ET)
            print(f"\r{DIM}[{now.strftime('%H:%M:%S')} ET] Market closed — waiting...{RESET}",
                  end="", flush=True)
        time_module.sleep(60)


if __name__ == "__main__":
    main()
