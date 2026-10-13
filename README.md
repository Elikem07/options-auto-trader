# Options Auto-Trader

An end-to-end intraday options system: a pattern scanner, a confluence scorer,
an automated IBKR execution loop, a backtester, a parameter sweep, a Monte Carlo
risk model and a live dashboard.

> **This software can place real orders.** It ships with `dry_run = true` and
> placeholder parameters that block all trading. Read
> [Safety](#safety) before changing either. Nothing here is investment advice.

## What it does

A scanner polls a watchlist on 5-minute bars and detects four setup families —
pre-market high/low breaks, opening-range breakouts, volume divergence and VWAP
volume divergence. Each candidate is scored 0–5 on confluence (setup quality,
index alignment, absence of chop, time-of-day). Scores inside the configured
band are forwarded to the trader, which selects an ATM option, places a limit
order at mid plus an offset, and manages the position to a target, a stop, a
time stop or the end-of-day flat.

## Components

| File | Role |
|---|---|
| `scanner.py` | Pattern detection, confluence scoring, IBKR bar handling |
| `scanner_alpaca.py` | Standalone scanner against Alpaca market data |
| `auto_trader.py` | Live execution loop: contract selection, limit orders, exits |
| `alpaca_options.py` | Optional Alpaca options-quote fallback |
| `ibkr_history.py` | Historical bar fetch and caching |
| `backtest.py` | Replays scored signals against cached bars |
| `optimize.py` | Parameter sweep over the backtest |
| `monte_carlo.py` | Bootstrap resampling of trade outcomes into equity curves |
| `cost_analysis.py` | Commission and slippage sensitivity |
| `dashboard.py` | Live monitoring UI |
| `signal_logger.py` | Append-only signal/trade journal |
| `process_manager.py` | Start/stop supervision for the trading loop |

## Install

```bash
pip install -r requirements.txt   # or: ib_insync pandas numpy pytz
```

Python 3.11+ (`tomllib`). Live trading requires IBKR TWS or IB Gateway with API
access enabled and market-data entitlements for the instruments you trade.

## Configuration

**All parameters are gitignored. The committed example is inert by design.**

```bash
cp config/strategy.example.toml config/strategy.toml
# then edit config/strategy.toml
```

| File | Committed? | Contents |
|---|---|---|
| `config/strategy.example.toml` | yes | Full schema, placeholders, `dry_run = true` |
| `config/strategy.toml` | **no** | Real thresholds, limits, watchlist |

Override with `AUTOTRADER_STRATEGY_CONFIG`. Sections:

- `[universe]` — watchlist, and which tickers genuinely list 0DTE/MWF expirations
- `[setup]` — which setup is cleared for live trading, and the tradeable score band
- `[session]` — market open/close, scan start, power-hour windows
- `[breakout]`, `[volume_divergence]`, `[opening_range]`, `[vwap_divergence]` — per-setup entry thresholds and target multiples
- `[scoring]` — confluence point weights
- `[execution]` — dry-run flag, TWS port, position/premium/loss limits, order offsets, timeouts, cutoffs
- `[backtest]`, `[simulation]` — replay filters and Monte Carlo settings

## Safety

Two independent switches must be changed before any order can reach a broker:

1. `[execution].dry_run` — `true` logs signals and places nothing.
2. `[execution].tws_port` — `7497` is the TWS **paper** port; `7496` is live.

The example config also sets `max_trades_day = 0`, `max_concurrent = 0` and
`min_score = 99`, so even with `dry_run = false` an unconfigured clone cannot
trade. Risk controls in `[execution]`:

- `max_daily_loss` halts trading for the day once breached
- `max_premium_per_trade` rejects expensive contracts so one position cannot
  consume the whole daily loss budget
- `max_concurrent`, `max_trades_day`, `cooldown_min` bound exposure and churn
- `no_new_after` / `close_before` stop new entries and force a flat close
- `max_hold_bars` time-stops a position that neither target nor stop resolved

Verify `dry_run` behavior and paper-port fills before considering live use.

## Credentials

Keys are read from the environment only — never from a committed file. The
original launcher script is gitignored because it contained a local absolute
path; export the variables yourself instead:

```bash
export ALPACA_API_KEY=...
export ALPACA_SECRET_KEY=...   # only needed for scanner_alpaca.py
```

`scanner_alpaca.py` exits immediately if either is unset. IBKR needs no key —
it authenticates through the running TWS/Gateway session.

## Usage

```bash
python scanner.py                 # scan only, no orders
python auto_trader.py             # execution loop (honors dry_run)
python backtest.py                # replay cached bars
python optimize.py                # parameter sweep
python monte_carlo.py             # equity-curve distribution
python dashboard.py               # live dashboard
```

## Limitations

- Backtests replay **scored signals against cached bars**; they do not model the
  options chain tick-by-tick. Fills are assumed at mid plus the configured
  offset, which is optimistic for wide spreads.
- 0DTE expiration availability is hardcoded per ticker in config and must be
  verified against the broker's contract data — it changes over time.
- Monte Carlo resamples historical trade outcomes and therefore inherits every
  bias in the sample, including its length.
- No modeling of assignment, pin risk, halts, or data outages.
- Backtested results are not evidence of live performance.

## License

No license granted. All rights reserved.
