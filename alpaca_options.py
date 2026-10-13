"""alpaca_options.py — optional Alpaca options data source.

IBKR's live options market data is currently gated behind a paid
subscription (see auto_trader.py's startup validation — every ticker
resolves a contract fine, but bid/ask comes back empty). Alpaca added
options trading/data alongside its existing stock API, and alpaca-py
(already a dependency here, used by scanner_alpaca.py) exposes a full
OptionHistoricalDataClient. This module wraps that, to test whether it
can fill the gap.

UNVERIFIED: built entirely from the alpaca-py SDK's method signatures,
docstrings, and Pydantic model fields (confirmed via introspection) — no
working ALPACA_API_KEY/ALPACA_SECRET_KEY with options entitlements was
available to test against a live account. Options trading is a separate
approval tier on Alpaca, independent of a funded/paper account existing.
Run this file directly once you have real keys to find out whether it
actually returns data.

Credentials are read lazily (only when a function is called), not at
import time, so importing this module is always safe even without keys
set — unlike scanner_alpaca.py, which raises SystemExit at import if
credentials are missing.
"""
import os
import re
from datetime import date, timedelta

from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionChainRequest

# OCC symbol suffix is always [C|P] + 8-digit strike (strike*1000, zero
# padded), regardless of how the root symbol itself is padded/formatted —
# parsing from the right avoids guessing at root-symbol width.
_OCC_SUFFIX_RE = re.compile(r"([CP])(\d{8})$")


def _parse_occ_suffix(symbol: str) -> "tuple[str, float] | None":
    m = _OCC_SUFFIX_RE.search(symbol)
    if not m:
        return None
    right, strike_digits = m.group(1), m.group(2)
    return right, int(strike_digits) / 1000.0


def _client() -> OptionHistoricalDataClient:
    key = os.environ.get("ALPACA_API_KEY", "")
    secret = os.environ.get("ALPACA_SECRET_KEY", "")
    if not key or not secret:
        raise RuntimeError(
            "ALPACA_API_KEY / ALPACA_SECRET_KEY not set — export both before "
            "calling into alpaca_options.py"
        )
    return OptionHistoricalDataClient(key, secret)


def get_chain_near(ticker: str, price: float, expiry: str, strike_window: float = 10.0) -> dict:
    """Chain snapshot for `ticker`/`expiry`, restricted to strikes within
    `strike_window` of `price` (keeps the request/response small).
    `expiry` must be 'YYYY-MM-DD'. Returns {occ_symbol: OptionsSnapshot}.
    Raises whatever the SDK raises on a bad request/auth failure — not
    wrapped, since this is meant to be exercised directly while validating
    account access, and a swallowed exception would hide exactly the
    failure mode you're trying to diagnose.
    """
    client = _client()
    req = OptionChainRequest(
        underlying_symbol=ticker,
        expiration_date=expiry,
        strike_price_gte=price - strike_window,
        strike_price_lte=price + strike_window,
    )
    return client.get_option_chain(req)


def get_quote_for_contract(ticker: str, expiry: str, strike: float, right: str) -> "dict | None":
    """Bid/ask/mid for one EXACT contract (strike/expiry/right must match
    exactly — this is for pricing a contract another broker already
    resolved, e.g. IBKR's ATM put, not for picking a strike yourself).
    `expiry` must be 'YYYY-MM-DD'. `right` is 'C' or 'P'.
    Returns None if the chain is empty, the exact contract isn't in it, or
    it has no live quote.
    """
    chain = get_chain_near(ticker, strike, expiry, strike_window=0.5)
    for symbol, snap in chain.items():
        parsed = _parse_occ_suffix(symbol)
        if parsed is None:
            continue
        sym_right, sym_strike = parsed
        if sym_right != right or abs(sym_strike - strike) > 0.01:
            continue
        quote = snap.latest_quote
        if quote is None or not quote.bid_price or not quote.ask_price:
            return None
        bid, ask = float(quote.bid_price), float(quote.ask_price)
        return {"occ_symbol": symbol, "bid": bid, "ask": ask, "mid": round((bid + ask) / 2, 2)}
    return None


def find_atm_quote(ticker: str, price: float, expiry: str, bias: str) -> "dict | None":
    """Convenience lookup for manual testing: nearest-strike contract to
    `price` (LONG->call, SHORT->put), rather than an exact strike match.
    `expiry` must be 'YYYY-MM-DD'.
    """
    right = "C" if bias == "LONG" else "P"
    chain = get_chain_near(ticker, price, expiry)
    candidates = []
    for symbol, snap in chain.items():
        parsed = _parse_occ_suffix(symbol)
        if parsed is None or parsed[0] != right:
            continue
        candidates.append((symbol, parsed[1], snap))
    if not candidates:
        return None

    symbol, strike, snap = min(candidates, key=lambda c: abs(c[1] - price))
    quote = snap.latest_quote
    if quote is None or not quote.bid_price or not quote.ask_price:
        return None
    bid, ask = float(quote.bid_price), float(quote.ask_price)
    return {"occ_symbol": symbol, "strike": strike, "bid": bid, "ask": ask, "mid": round((bid + ask) / 2, 2)}


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: .venv/bin/python3 alpaca_options.py TICKER PRICE [EXPIRY YYYY-MM-DD] [LONG|SHORT]")
        print("Example: .venv/bin/python3 alpaca_options.py AAPL 230.50")
        raise SystemExit(1)

    ticker = sys.argv[1].upper()
    price = float(sys.argv[2])
    today = date.today()
    default_expiry = (today + timedelta(days=(4 - today.weekday()) % 7)).isoformat()
    expiry = sys.argv[3] if len(sys.argv) > 3 else default_expiry
    bias = sys.argv[4].upper() if len(sys.argv) > 4 else "SHORT"

    print(f"Fetching {ticker} chain near ${price} for expiry {expiry} ({bias})...")
    try:
        result = find_atm_quote(ticker, price, expiry, bias)
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        print("Check: are ALPACA_API_KEY/ALPACA_SECRET_KEY set? Is options trading enabled on the account?")
        raise SystemExit(1)

    if result is None:
        print("No quote found — chain came back empty or nothing has a live quote yet (market open?).")
    else:
        print(f"SUCCESS: {result}")
