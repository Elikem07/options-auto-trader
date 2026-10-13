import math
import os
import subprocess
import time as time_module
from datetime import date, datetime, time

import pandas as pd
import pytz
import yfinance as yf

yf.set_tz_cache_location("/tmp/yfinance_cache")
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

# ── Alpaca paper trading config ───────────────────────────────
# Set these in your shell before running:
#   export ALPACA_API_KEY="your_key"
#   export ALPACA_SECRET_KEY="your_secret"
ALPACA_API_KEY    = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
from strategy_params import PARAMS

_AS = PARAMS["alpaca_scanner"]
TRADE_SIZE_USD = _AS["trade_size_usd"]   # dollar amount per trade
WATCHLIST = list(_AS["watchlist"] or PARAMS["universe"]["watchlist"])

# ANSI colors
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
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
        "color":  alert["color"],
    })


def print_alert_history() -> None:
    if not _alert_log:
        return
    print(f"\n{DIM}{'─' * 52}")
    print(f"  Today's alerts ({len(_alert_log)} total)")
    print(f"{'─' * 52}{RESET}")
    for a in _alert_log:
        bias_color = GREEN if a["bias"] == "LONG" else RED
        print(f"  {DIM}{a['time']}{RESET}  "
              f"{BOLD}{a['ticker']:<6}{RESET}  "
              f"{a['color']}{a['setup']:<28}{RESET}  "
              f"{bias_color}[{a['bias']}]{RESET}  "
              f"{YELLOW}${a['entry']}{RESET}")
    print(f"{DIM}{'─' * 52}{RESET}")


def _occ_symbol(ticker: str, expiry: str, opt_type: str, strike: float) -> str:
    """Build the OCC option symbol, e.g. AAPL240119C00185000."""
    exp = datetime.strptime(expiry, "%Y-%m-%d")
    cp  = "C" if opt_type == "CALL" else "P"
    return f"{ticker}{exp.strftime('%y%m%d')}{cp}{int(round(strike * 1000)):08d}"


def execute_trade(alert: dict, opt) -> None:
    """Buy a call or put on Alpaca paper account for the given alert."""
    ticker = alert["ticker"]

    if opt is None:
        print(f"  {DIM}Options trade skipped — no contract data for {ticker}{RESET}")
        return

    occ           = _occ_symbol(ticker, opt["expiry"], opt["type"], opt["strike"])
    num_contracts = max(1, int(TRADE_SIZE_USD / opt["premium"]))

    # Skip if already holding this contract
    try:
        trading_client.get_open_position(occ)
        print(f"  {DIM}Trade skipped — already holding {occ}{RESET}")
        return
    except Exception:
        pass

    try:
        order = trading_client.submit_order(
            MarketOrderRequest(
                symbol=occ,
                qty=num_contracts,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
        )
        opt_color = GREEN if opt["type"] == "CALL" else RED
        print(f"  {BOLD}PAPER TRADE{RESET}  "
              f"BUY {num_contracts}x {opt_color}{occ}{RESET}  "
              f"~${opt['mid']}/share  (${opt['premium']}/contract)  "
              f"Stop ≈${opt['val_stop']}  T1 ≈${opt['val_t1']}  "
              f"{DIM}order {str(order.id)[:8]}{RESET}")
    except Exception as e:
        print(f"  {RED}Options trade failed — {e}{RESET}")


def _build_df(bars_df: pd.DataFrame) -> "pd.DataFrame | None":
    """Normalize Alpaca bars df, compute VWAP and delta."""
    df = bars_df.copy()
    df.columns = [col.lower() for col in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].dropna(how="all")
    if df.empty:
        return None
    if df.index.tzinfo is None:
        df.index = pd.DatetimeIndex(df.index).tz_localize("UTC").tz_convert(ET)
    else:
        df.index = pd.DatetimeIndex(df.index).tz_convert(ET)
    mkt_mask = df.index.time >= time(9, 30)
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


def fetch_data(ticker: str) -> "pd.DataFrame | None":
    """Single-ticker fetch via Alpaca data API."""
    try:
        now   = datetime.now(ET)
        start = now.replace(hour=4, minute=0, second=0, microsecond=0)
        req   = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame(2, TimeFrameUnit.Minute),
            start=start, feed="iex",
        )
        bars = data_client.get_stock_bars(req)
        df   = bars[ticker].df
        return _build_df(df) if not df.empty else None
    except Exception:
        return None


def fetch_all() -> dict[str, pd.DataFrame]:
    """Batch-fetch all watchlist tickers via Alpaca data API."""
    result = {}
    try:
        now   = datetime.now(ET)
        start = now.replace(hour=4, minute=0, second=0, microsecond=0)
        req   = StockBarsRequest(
            symbol_or_symbols=WATCHLIST,
            timeframe=TimeFrame(2, TimeFrameUnit.Minute),
            start=start, feed="iex",
        )
        bars = data_client.get_stock_bars(req)
        for ticker in WATCHLIST:
            try:
                df = bars[ticker].df
                if df.empty:
                    continue
                processed = _build_df(df)
                if processed is not None:
                    result[ticker] = processed
            except Exception:
                continue
    except Exception:
        pass
    return result


def get_pm_levels(df):
    pm = df[df.index.time < time(9, 30)]
    if pm.empty:
        return None, None
    return round(pm["high"].max(), 2), round(pm["low"].min(), 2)


def detect_patterns(ticker, df):
    alerts = []

    pm_high, pm_low = get_pm_levels(df)
    if pm_high is None:
        return alerts

    market_df = df[df.index.time >= time(9, 30)].copy()
    if len(market_df) < 3:
        return alerts

    avg_vol_series = market_df["volume"].rolling(20).mean()
    avg_vol = avg_vol_series.iloc[-1]
    if pd.isna(avg_vol) or avg_vol == 0:
        return alerts

    c    = market_df.iloc[-1]
    prev = market_df.iloc[-2]
    vol_ratio = c["volume"] / avg_vol

    vwap = c.get("vwap", float("nan"))
    vwap_valid = not pd.isna(vwap) and vwap > 0

    # ── 1) PM HIGH BREAKOUT ──────────────────────────────────────
    if prev["close"] <= pm_high and c["close"] > pm_high and vol_ratio >= 1.5:
        stop = c["low"]
        risk = c["close"] - stop
        alerts.append({
            "setup":     "PM High Breakout",
            "bias":      "LONG",
            "color":     GREEN,
            "ticker":    ticker,
            "entry":     round(c["close"], 2),
            "stop":      round(stop, 2),
            "target1":   round(c["close"] + risk * 2, 2),
            "target2":   round(c["close"] + risk * 3, 2),
            "vol_ratio": round(vol_ratio, 1),
            "rr":        "1:2 — 1:3",
            "note":      f"Broke above PM High (${pm_high}) with {vol_ratio:.1f}x avg volume",
        })

    # ── 2) PM LOW BREAKDOWN ──────────────────────────────────────
    if prev["close"] >= pm_low and c["close"] < pm_low and vol_ratio >= 1.5:
        stop = c["high"]
        risk = stop - c["close"]
        alerts.append({
            "setup":     "PM Low Breakdown",
            "bias":      "SHORT",
            "color":     RED,
            "ticker":    ticker,
            "entry":     round(c["close"], 2),
            "stop":      round(stop, 2),
            "target1":   round(c["close"] - risk * 2, 2),
            "target2":   round(c["close"] - risk * 3, 2),
            "vol_ratio": round(vol_ratio, 1),
            "rr":        "1:2 — 1:3",
            "note":      f"Broke below PM Low (${pm_low}) with {vol_ratio:.1f}x avg volume",
        })

    # ── 3 & 4) VOLUME DIVERGENCE ─────────────────────────────────
    if len(market_df) >= 5:
        last5 = market_df.iloc[-5:]
        vols  = last5["volume"].values
        vol_declining = vols[0] > vols[1] > vols[2] > vols[3] > vols[4]

        if last5["high"].iloc[-1] > last5["high"].iloc[0] and vol_declining and c["close"] < c["open"]:
            stop = last5["high"].max()
            risk = stop - c["close"]
            near_res = abs(c["high"] - pm_high) / pm_high < 0.003
            alerts.append({
                "setup":     "Volume Divergence Short",
                "bias":      "SHORT",
                "color":     RED,
                "ticker":    ticker,
                "entry":     round(c["close"], 2),
                "stop":      round(stop, 2),
                "target1":   round(c["close"] - risk * 1.5, 2),
                "target2":   round(c["close"] - risk * 2.5, 2),
                "vol_ratio": round(vol_ratio, 1),
                "rr":        "1:1.5 — 1:2.5",
                "note":      f"Higher highs, declining volume{' — near PM High resistance' if near_res else ''}",
            })

        if last5["low"].iloc[-1] < last5["low"].iloc[0] and vol_declining and c["close"] > c["open"]:
            stop = last5["low"].min()
            risk = c["close"] - stop
            near_sup = abs(c["low"] - pm_low) / pm_low < 0.003
            alerts.append({
                "setup":     "Volume Divergence Long",
                "bias":      "LONG",
                "color":     GREEN,
                "ticker":    ticker,
                "entry":     round(c["close"], 2),
                "stop":      round(stop, 2),
                "target1":   round(c["close"] + risk * 1.5, 2),
                "target2":   round(c["close"] + risk * 2.5, 2),
                "vol_ratio": round(vol_ratio, 1),
                "rr":        "1:1.5 — 1:2.5",
                "note":      f"Lower lows, selling volume drying up{' — at PM Low support' if near_sup else ''}",
            })

    # ── 5) OPENING RANGE BREAKOUT (first ~16 min = 8 candles) ───
    n = len(market_df)
    if 8 < n < 30:
        or_candles = market_df.iloc[:8]
        or_high    = or_candles["high"].max()
        or_low     = or_candles["low"].min()
        or_range   = or_high - or_low

        if prev["close"] <= or_high and c["close"] > or_high and vol_ratio >= 1.2:
            alerts.append({
                "setup":     "ORB Long",
                "bias":      "LONG",
                "color":     YELLOW,
                "ticker":    ticker,
                "entry":     round(c["close"], 2),
                "stop":      round(or_low, 2),
                "target1":   round(or_high + or_range, 2),
                "target2":   round(or_high + or_range * 2, 2),
                "vol_ratio": round(vol_ratio, 1),
                "rr":        "1:1 — 1:2",
                "note":      f"Opening range (${or_low:.2f}–${or_high:.2f}) broken to upside with {vol_ratio:.1f}x volume",
            })

        if prev["close"] >= or_low and c["close"] < or_low and vol_ratio >= 1.2:
            alerts.append({
                "setup":     "ORB Short",
                "bias":      "SHORT",
                "color":     YELLOW,
                "ticker":    ticker,
                "entry":     round(c["close"], 2),
                "stop":      round(or_high, 2),
                "target1":   round(or_low - or_range, 2),
                "target2":   round(or_low - or_range * 2, 2),
                "vol_ratio": round(vol_ratio, 1),
                "rr":        "1:1 — 1:2",
                "note":      f"Opening range broken to downside with {vol_ratio:.1f}x volume",
            })

    # ── 6) VWAP VOLUME DIVERGENCE ────────────────────────────────
    if vwap_valid and len(market_df) >= 4:
        last4 = market_df.iloc[-4:]
        vols4 = last4["volume"].values
        vol_declining4 = vols4[0] > vols4[1] > vols4[2] > vols4[3]
        near_vwap = abs(c["close"] - vwap) / vwap < 0.002

        if near_vwap and vol_declining4:
            if c["close"] <= vwap and c["close"] > c["open"]:
                stop = last4["low"].min()
                risk = c["close"] - stop
                if risk > 0:
                    alerts.append({
                        "setup":     "VWAP Vol Divergence Long",
                        "bias":      "LONG",
                        "color":     GREEN,
                        "ticker":    ticker,
                        "entry":     round(c["close"], 2),
                        "stop":      round(stop, 2),
                        "target1":   round(c["close"] + risk * 1.5, 2),
                        "target2":   round(c["close"] + risk * 2.5, 2),
                        "vol_ratio": round(vol_ratio, 1),
                        "rr":        "1:1.5 — 1:2.5",
                        "note":      f"Selling volume drying up at VWAP (${vwap:.2f}) — reclaim attempt",
                    })

            if c["close"] >= vwap and c["close"] < c["open"]:
                stop = last4["high"].max()
                risk = stop - c["close"]
                if risk > 0:
                    alerts.append({
                        "setup":     "VWAP Vol Divergence Short",
                        "bias":      "SHORT",
                        "color":     RED,
                        "ticker":    ticker,
                        "entry":     round(c["close"], 2),
                        "stop":      round(stop, 2),
                        "target1":   round(c["close"] - risk * 1.5, 2),
                        "target2":   round(c["close"] - risk * 2.5, 2),
                        "vol_ratio": round(vol_ratio, 1),
                        "rr":        "1:1.5 — 1:2.5",
                        "note":      f"Buying volume drying up at VWAP (${vwap:.2f}) — rejection",
                    })

    return alerts


def _norm_cdf(x):
    return (1.0 + math.erf(x / math.sqrt(2))) / 2.0


def _bs_delta(S, K, T, r, sigma, option_type):
    if T <= 0 or sigma <= 0:
        return 0.5
    d1 = (math.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * math.sqrt(T))
    return _norm_cdf(d1) if option_type == "call" else _norm_cdf(d1) - 1.0


def get_options_context(ticker, price, bias, target1, target2, stop):
    try:
        t    = yf.Ticker(ticker)
        exps = t.options
        if not exps:
            return None

        today      = date.today()
        valid_exps = [e for e in exps if datetime.strptime(e, "%Y-%m-%d").date() >= today]
        if not valid_exps:
            return None

        exp       = valid_exps[0]
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
        days   = max((datetime.strptime(exp, "%Y-%m-%d").date() - today).days, 0)
        T      = max(days, 0.5) / 365.0
        delta  = abs(_bs_delta(price, strike, T, 0.05, iv, opt_type))

        def est(target_price):
            move = (target_price - price) if bias == "LONG" else (price - target_price)
            return round(max(0.01, mid + delta * move), 2)

        val_t1, val_t2, val_stop = est(target1), est(target2), est(stop)
        return {
            "type":    "CALL" if bias == "LONG" else "PUT",
            "strike":  strike,
            "expiry":  exp,
            "days":    days,
            "mid":     round(mid, 2),
            "premium": round(mid * 100, 2),
            "delta":   round(delta, 2),
            "iv":      round(iv * 100, 1),
            "val_t1":  val_t1,  "pnl_t1":  round((val_t1  - mid) * 100, 2),
            "val_t2":  val_t2,  "pnl_t2":  round((val_t2  - mid) * 100, 2),
            "val_stop":val_stop,"pnl_stop": round((val_stop - mid) * 100, 2),
        }
    except Exception:
        return None


def print_options(opt, color):
    if opt is None:
        print(f"  {DIM}Options: unavailable{RESET}")
        return
    def pnl_color(v): return GREEN if v >= 0 else RED
    def fmt_pnl(v): return f"+${v:.2f}" if v >= 0 else f"-${abs(v):.2f}"
    print(f"  {DIM}{'─' * 50}{RESET}")
    print(f"  {BOLD}{color}{opt['type']}{RESET}  "
          f"${opt['strike']} strike  "
          f"Exp {opt['expiry']} ({opt['days']}d)  "
          f"IV {opt['iv']}%  Δ {opt['delta']}")
    print(f"  Premium: {YELLOW}${opt['mid']}/share{RESET}  "
          f"({YELLOW}${opt['premium']}/contract{RESET})")
    print(f"  Stop:  ≈ {RED}${opt['val_stop']}{RESET}  "
          f"({pnl_color(opt['pnl_stop'])}{fmt_pnl(opt['pnl_stop'])}/contract{RESET})  "
          f"T1: ≈ {GREEN}${opt['val_t1']}{RESET}  "
          f"({pnl_color(opt['pnl_t1'])}{fmt_pnl(opt['pnl_t1'])}/contract{RESET})  "
          f"T2: ≈ {GREEN}${opt['val_t2']}{RESET}  "
          f"({pnl_color(opt['pnl_t2'])}{fmt_pnl(opt['pnl_t2'])}/contract{RESET})")


def notify(alert):
    title   = f"{alert['ticker']} — {alert['setup']} [{alert['bias']}]"
    message = f"Entry ${alert['entry']}  Stop ${alert['stop']}  Vol {alert['vol_ratio']}x  {alert['rr']}"
    subprocess.run([
        "osascript", "-e",
        f'display notification "{message}" with title "{title}" sound name "Funk"'
    ], capture_output=True)


def print_alert(alert):
    color      = alert["color"]
    bias_color = GREEN if alert["bias"] == "LONG" else RED
    print(f"\n{BOLD}{color}{'━' * 52}{RESET}")
    print(f"  {BOLD}{alert['ticker']}{RESET}  {color}{alert['setup']}{RESET}  {bias_color}[{alert['bias']}]{RESET}")
    print(f"  Entry: {YELLOW}${alert['entry']}{RESET}  "
          f"Stop: {RED}${alert['stop']}{RESET}  "
          f"R:R: {alert['rr']}")
    print(f"  T1: {GREEN}${alert['target1']}{RESET}  "
          f"T2: {GREEN}${alert['target2']}{RESET}  "
          f"Vol: {YELLOW}{alert['vol_ratio']}x{RESET}  "
          f"Score: {BOLD}{alert.get('score', '?')}/7{RESET}")
    of_tags = []
    if alert.get("of_confirms"):   of_tags.append(f"{GREEN}Δ Flow ✓{RESET}")
    if alert.get("of_absorption"): of_tags.append(f"{YELLOW}Absorption ✓{RESET}")
    if of_tags:
        print(f"  OrderFlow: {'  '.join(of_tags)}")
    print(f"  {DIM}{alert['note']}{RESET}")
    opt = get_options_context(
        alert["ticker"], alert["entry"], alert["bias"],
        alert["target1"], alert["target2"], alert["stop"],
    )
    print_options(opt, color)
    print(f"{BOLD}{color}{'━' * 52}{RESET}")
    _log_alert(alert)
    execute_trade(alert, opt)
    notify(alert)


def run_scan():
    now = datetime.now(ET)
    print_alert_history()
    print(f"\n{DIM}[{now.strftime('%H:%M:%S')} ET] Scanning {len(WATCHLIST)} tickers...{RESET}",
          end="", flush=True)

    found    = 0
    all_data = fetch_all()
    for ticker, df in all_data.items():
        for alert in detect_patterns(ticker, df):
            alert["of_confirms"]   = orderflow_confirms(alert, df)
            alert["of_absorption"] = absorption_detected(df)
            alert["score"]         = score_alert(alert, df, all_data)
            print_alert(alert)
            found += 1

    if found == 0:
        print(f" {DIM}no patterns{RESET}")
    else:
        print(f" {GREEN}{found} alert(s) found{RESET}")


def in_power_hour() -> bool:
    t = datetime.now(ET).time()
    return time(9, 30) <= t <= time(10, 30) or time(15, 0) <= t <= time(16, 0)


def market_aligned(ticker: str, bias: str, market_data: dict) -> bool:
    ref = "QQQ" if ticker in ("QQQ", "TQQQ", "SQQQ", "SOXX", "SMH") else "SPY"
    df  = market_data.get(ref)
    if df is None or df.empty:
        return False
    c    = df.iloc[-1]
    vwap = c.get("vwap", float("nan"))
    if pd.isna(vwap) or vwap == 0:
        return False
    return c["close"] > vwap if bias == "LONG" else c["close"] < vwap


def whipsaw_detected(df: pd.DataFrame) -> bool:
    market_df = df[df.index.time >= time(9, 30)]
    if len(market_df) < 4:
        return False
    closes     = market_df["close"].iloc[-4:].values
    directions = [1 if closes[i] > closes[i - 1] else -1 for i in range(1, len(closes))]
    return all(directions[i] != directions[i - 1] for i in range(1, len(directions)))


def orderflow_confirms(alert: dict, df: pd.DataFrame) -> bool:
    market_df = df[df.index.time >= time(9, 30)]
    if len(market_df) < 5 or "cum_delta" not in market_df.columns:
        return False
    cd = market_df["cum_delta"].iloc[-5:].values
    if alert["bias"] == "LONG":
        return float(cd[-1]) > float(cd[0])
    return float(cd[-1]) < float(cd[0])


def absorption_detected(df: pd.DataFrame) -> bool:
    market_df = df[df.index.time >= time(9, 30)]
    if len(market_df) < 20:
        return False
    avg_vol   = market_df["volume"].rolling(20).mean().iloc[-1]
    avg_range = (market_df["high"] - market_df["low"]).rolling(20).mean().iloc[-1]
    if pd.isna(avg_vol) or pd.isna(avg_range) or avg_range == 0:
        return False
    for i in range(-3, 0):
        bar = market_df.iloc[i]
        if bar["volume"] > avg_vol * 1.5 and (bar["high"] - bar["low"]) < avg_range * 0.6:
            return True
    return False


def score_alert(alert: dict, df: pd.DataFrame, market_data: dict) -> int:
    score = 0
    if "VWAP Vol Divergence" in alert["setup"]:
        score += 2
    elif "Volume Divergence" in alert["setup"]:
        score += 1
    if market_aligned(alert["ticker"], alert["bias"], market_data):
        score += 1
    if not whipsaw_detected(df):
        score += 1
    if in_power_hour():
        score += 1
    if orderflow_confirms(alert, df):
        score += 1
    if absorption_detected(df):
        score += 1
    return score


def is_market_hours():
    t = datetime.now(ET).time()
    return time(9, 25) <= t <= time(16, 0)


def main():
    print(f"\n{BOLD}Pattern Scanner + Alpaca Paper Trading — yfinance{RESET}")
    print(f"Watchlist ({len(WATCHLIST)}): {', '.join(WATCHLIST)}")
    print(f"Patterns: PM High/Low Break · Volume Divergence · ORB · VWAP Vol Divergence")
    print(f"Trade size: ${TRADE_SIZE_USD} per alert  ·  Paper trading ON")
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