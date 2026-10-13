import json
import math
import os
import re
import threading
import time as time_module
import uuid
from datetime import datetime, date, time

import pandas as pd
from flask import Flask, jsonify, render_template_string, request, send_file

import process_manager
from scanner import ET, WATCHLIST, detect_patterns, fetch_data, get_options_context, is_market_hours

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRADES_FILE = os.path.join(BASE_DIR, "trades.json")
AUTO_TRADER_SCRIPT = "auto_trader.py"
AUTO_TRADER_LOG = os.path.join(BASE_DIR, "auto_trader.log")
BACKTEST_SCRIPT = "backtest.py"
BACKTEST_RESULTS_FILE = os.path.join(BASE_DIR, "backtest_results.csv")
MONTECARLO_SCRIPT = "monte_carlo.py"
MONTECARLO_HTML_FILE = os.path.join(BASE_DIR, "monte_carlo.html")
OPTIMIZE_SCRIPT = "optimize.py"
OPTIMIZE_RESULTS_FILE = os.path.join(BASE_DIR, "optimization_results.json")

# ── Trades persistence ─────────────────────────────────────────────────────────

_trades_lock = threading.Lock()


def load_trades():
    if os.path.exists(TRADES_FILE):
        with open(TRADES_FILE) as f:
            return json.load(f)
    return []


def save_trades(trades):
    with open(TRADES_FILE, "w") as f:
        json.dump(trades, f, indent=2)


# ── Outcome detection ──────────────────────────────────────────────────────────

def detect_outcome(trade):
    df = fetch_data(trade["ticker"])
    if df is None:
        return {"outcome": "open"}

    market_df = df[df.index.time >= time(9, 30)]
    if market_df.empty:
        return {"outcome": "open"}

    h, m = map(int, trade["entry_time"].split(":"))
    entry_t = time(h, m)
    after = market_df[market_df.index.time > entry_t]
    if after.empty:
        return {"outcome": "open"}

    entry = float(trade["entry"])
    stop  = float(trade["stop"])
    t1    = float(trade["target1"])
    t2    = float(trade["target2"])
    risk  = abs(entry - stop)
    if risk == 0:
        return {"outcome": "open"}

    for ts, bar in after.iterrows():
        bar_time = ts.strftime("%H:%M")
        if trade["bias"] == "LONG":
            if bar["low"] <= stop:
                return {"outcome": "loss",   "exit_price": stop, "exit_time": bar_time, "r_multiple": -1.0}
            if bar["high"] >= t2:
                return {"outcome": "win_t2", "exit_price": t2,   "exit_time": bar_time, "r_multiple": round((t2 - entry) / risk, 2)}
            if bar["high"] >= t1:
                return {"outcome": "win_t1", "exit_price": t1,   "exit_time": bar_time, "r_multiple": round((t1 - entry) / risk, 2)}
        else:
            if bar["high"] >= stop:
                return {"outcome": "loss",   "exit_price": stop, "exit_time": bar_time, "r_multiple": -1.0}
            if bar["low"] <= t2:
                return {"outcome": "win_t2", "exit_price": t2,   "exit_time": bar_time, "r_multiple": round((entry - t2) / risk, 2)}
            if bar["low"] <= t1:
                return {"outcome": "win_t1", "exit_price": t1,   "exit_time": bar_time, "r_multiple": round((entry - t1) / risk, 2)}

    return {"outcome": "open"}


def calc_stats(trades):
    closed = [t for t in trades if t.get("outcome") not in ("open", None)]
    wins   = [t for t in closed if t.get("outcome", "").startswith("win")]
    losses = [t for t in closed if t.get("outcome") == "loss"]

    win_rate   = round(len(wins) / len(closed) * 100, 1) if closed else 0
    avg_r_win  = round(sum(t.get("r_multiple", 0) for t in wins) / len(wins), 2) if wins else 0
    avg_r_loss = round(sum(t.get("r_multiple", 0) for t in losses) / len(losses), 2) if losses else 0
    expectancy = round((win_rate / 100 * avg_r_win) + ((1 - win_rate / 100) * avg_r_loss), 2) if closed else 0

    by_setup = {}
    for t in closed:
        s = t["setup"]
        if s not in by_setup:
            by_setup[s] = {"trades": 0, "wins": 0, "r_total": 0.0}
        by_setup[s]["trades"] += 1
        if t.get("outcome", "").startswith("win"):
            by_setup[s]["wins"] += 1
        by_setup[s]["r_total"] += t.get("r_multiple", 0)

    breakdown = []
    for s, v in by_setup.items():
        breakdown.append({
            "setup":    s,
            "trades":   v["trades"],
            "wins":     v["wins"],
            "losses":   v["trades"] - v["wins"],
            "win_rate": round(v["wins"] / v["trades"] * 100, 1),
            "avg_r":    round(v["r_total"] / v["trades"], 2),
        })

    return {
        "total":      len(trades),
        "closed":     len(closed),
        "open":       len(trades) - len(closed),
        "wins":       len(wins),
        "losses":     len(losses),
        "win_rate":   win_rate,
        "avg_r_win":  avg_r_win,
        "expectancy": expectancy,
        "breakdown":  sorted(breakdown, key=lambda x: x["trades"], reverse=True),
    }


# ── Live scanner ───────────────────────────────────────────────────────────────

_alerts = []
_alert_lock = threading.Lock()
_status = {"last_scan": "Never", "scanning": False}


def background_scan():
    while True:
        if is_market_hours():
            _status["scanning"] = True
            found = []
            for ticker in WATCHLIST:
                df = fetch_data(ticker)
                if df is None:
                    continue
                for alert in detect_patterns(ticker, df):
                    alert["time"] = datetime.now(ET).strftime("%H:%M:%S")
                    alert.pop("color", None)
                    found.append(alert)
            with _alert_lock:
                for a in found:
                    _alerts.insert(0, a)
                del _alerts[200:]
            _status["last_scan"] = datetime.now(ET).strftime("%H:%M:%S ET")
            _status["scanning"] = False
        time_module.sleep(120)


# ── Replay ─────────────────────────────────────────────────────────────────────

def replay_ticker(ticker):
    df = fetch_data(ticker)
    if df is None:
        return []

    pm_df     = df[df.index.time < time(9, 30)]
    market_df = df[df.index.time >= time(9, 30)].copy()
    if len(market_df) < 3:
        return []

    results, seen = [], set()
    for i in range(2, len(market_df)):
        full_slice = pd.concat([pm_df, market_df.iloc[: i + 1]])
        bar_time   = market_df.index[i].strftime("%H:%M")
        for alert in detect_patterns(ticker, full_slice):
            key = (alert["setup"], alert["entry"])
            if key not in seen:
                seen.add(key)
                alert["time"] = bar_time
                alert.pop("color", None)
                results.append(alert)
    return results


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route("/api/state")
def state():
    with _alert_lock:
        data = list(_alerts)
    return jsonify({
        "alerts":      data,
        "last_scan":   _status["last_scan"],
        "scanning":    _status["scanning"],
        "market_open": is_market_hours(),
        "time":        datetime.now(ET).strftime("%H:%M:%S ET"),
        "watchlist":   WATCHLIST,
    })


@app.route("/api/replay/<ticker>")
def replay(ticker):
    ticker = ticker.upper()
    if ticker not in WATCHLIST:
        return jsonify({"error": f"{ticker} not in watchlist"}), 400
    return jsonify({"ticker": ticker, "setups": replay_ticker(ticker)})


@app.route("/api/log_trade", methods=["POST"])
def log_trade():
    data = request.get_json()
    required = ["ticker", "setup", "bias", "entry", "stop", "target1", "target2", "entry_time"]
    if not all(k in data for k in required):
        return jsonify({"error": "Missing fields"}), 400

    trade = {
        "id":         str(uuid.uuid4())[:8],
        "date":       datetime.now(ET).strftime("%Y-%m-%d"),
        "entry_time": data["entry_time"],
        "ticker":     data["ticker"],
        "setup":      data["setup"],
        "bias":       data["bias"],
        "entry":      float(data["entry"]),
        "stop":       float(data["stop"]),
        "target1":    float(data["target1"]),
        "target2":    float(data["target2"]),
        "note":       data.get("note", ""),
        "outcome":    "open",
        "exit_price": None,
        "exit_time":  None,
        "r_multiple": None,
    }

    trade.update(detect_outcome(trade))

    with _trades_lock:
        trades = load_trades()
        trades.append(trade)
        save_trades(trades)

    return jsonify({"trade": trade})


@app.route("/api/trades")
def get_trades():
    with _trades_lock:
        trades = load_trades()
    changed = False
    for t in trades:
        if t.get("outcome") == "open":
            result = detect_outcome(t)
            if result.get("outcome") != "open":
                t.update(result)
                changed = True
    if changed:
        with _trades_lock:
            save_trades(trades)
    return jsonify({"trades": list(reversed(trades)), "stats": calc_stats(trades)})


@app.route("/api/delete_trade/<trade_id>", methods=["DELETE"])
def delete_trade(trade_id):
    with _trades_lock:
        trades = load_trades()
        trades = [t for t in trades if t.get("id") != trade_id]
        save_trades(trades)
    return jsonify({"ok": True})


@app.route("/api/options")
def options_route():
    ticker  = request.args.get("ticker", "").upper()
    price   = float(request.args.get("price",   0))
    bias    = request.args.get("bias", "LONG")
    target1 = float(request.args.get("target1", 0))
    target2 = float(request.args.get("target2", 0))
    stop    = float(request.args.get("stop",    0))
    if ticker not in WATCHLIST:
        return jsonify({"error": "Ticker not in watchlist"}), 400
    result = get_options_context(ticker, price, bias, target1, target2, stop)
    if result is None:
        return jsonify({"error": "Options data unavailable"})
    return jsonify(result)


# ── Auto trader control ─────────────────────────────────────────────────────────

def _read_auto_trader_mode():
    with open(os.path.join(BASE_DIR, AUTO_TRADER_SCRIPT)) as f:
        src = f.read()
    dry_run  = re.search(r"^DRY_RUN\s*=\s*(\w+)", src, re.MULTILINE)
    tws_port = re.search(r"^TWS_PORT\s*=\s*(\d+)", src, re.MULTILINE)
    if dry_run and dry_run.group(1) == "True":
        return "DRY_RUN"
    if tws_port and tws_port.group(1) == "7497":
        return "PAPER"
    if tws_port and tws_port.group(1) == "7496":
        return "LIVE"
    return "UNKNOWN"


@app.route("/api/autotrader/status")
def autotrader_status():
    st = process_manager.status("auto_trader", AUTO_TRADER_SCRIPT)
    st["mode"] = _read_auto_trader_mode()
    st["log_tail"] = process_manager.tail_log(AUTO_TRADER_LOG, 50)
    return jsonify(st)


@app.route("/api/autotrader/start", methods=["POST"])
def autotrader_start():
    return jsonify(process_manager.start("auto_trader", AUTO_TRADER_SCRIPT))


@app.route("/api/autotrader/stop", methods=["POST"])
def autotrader_stop():
    return jsonify(process_manager.stop("auto_trader"))


# ── Reports (backtest / monte carlo) ────────────────────────────────────────────

@app.route("/api/tools/backtest/run", methods=["POST"])
def backtest_run():
    return jsonify(process_manager.start("backtest", BACKTEST_SCRIPT))


@app.route("/api/tools/backtest/status")
def backtest_status():
    return jsonify(process_manager.status("backtest", BACKTEST_SCRIPT))


@app.route("/api/tools/backtest/results")
def backtest_results():
    if not os.path.exists(BACKTEST_RESULTS_FILE):
        return jsonify({"error": "No results yet"}), 404
    df = pd.read_csv(BACKTEST_RESULTS_FILE)
    df = df.astype(object).where(pd.notna(df), None)
    return jsonify({
        "updated": datetime.fromtimestamp(os.path.getmtime(BACKTEST_RESULTS_FILE)).strftime("%Y-%m-%d %H:%M:%S"),
        "columns": list(df.columns),
        "rows": df.to_dict(orient="records"),
    })


@app.route("/api/tools/montecarlo/run", methods=["POST"])
def montecarlo_run():
    return jsonify(process_manager.start("montecarlo", MONTECARLO_SCRIPT))


@app.route("/api/tools/montecarlo/status")
def montecarlo_status():
    return jsonify(process_manager.status("montecarlo", MONTECARLO_SCRIPT))


@app.route("/monte_carlo.html")
def montecarlo_html():
    if not os.path.exists(MONTECARLO_HTML_FILE):
        return "Monte Carlo report not generated yet.", 404
    return send_file(MONTECARLO_HTML_FILE)


@app.route("/api/tools/optimize/run", methods=["POST"])
def optimize_run():
    return jsonify(process_manager.start("optimize", OPTIMIZE_SCRIPT))


@app.route("/api/tools/optimize/status")
def optimize_status():
    return jsonify(process_manager.status("optimize", OPTIMIZE_SCRIPT))


@app.route("/api/tools/optimize/results")
def optimize_results():
    if not os.path.exists(OPTIMIZE_RESULTS_FILE):
        return jsonify({"error": "No optimization results yet"}), 404
    with open(OPTIMIZE_RESULTS_FILE) as f:
        return jsonify(json.load(f))


# ── HTML ───────────────────────────────────────────────────────────────────────

DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>Pattern Scanner</title>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <style>
    *{box-sizing:border-box;margin:0;padding:0}
    :root{--cyan:#00e5ff;--green:#00e676;--red:#ff1744;--yellow:#ffd600;--bg:#070a0f;--panel:#0e131a;--border:#1a2530}
    body{
      background:
        repeating-linear-gradient(0deg, rgba(0,229,255,.03) 0px, rgba(0,229,255,.03) 1px, transparent 1px, transparent 3px),
        radial-gradient(circle at 12% -10%, rgba(0,229,255,.07), transparent 40%),
        radial-gradient(circle at 90% 110%, rgba(0,230,118,.05), transparent 40%),
        var(--bg);
      color:#c3d1da;font-family:ui-monospace,'SF Mono','JetBrains Mono','Courier New',monospace;min-height:100vh;position:relative
    }
    body::before{
      content:'';position:fixed;left:0;right:0;height:140px;top:-140px;
      background:linear-gradient(180deg, transparent, rgba(0,229,255,.05), transparent);
      animation:scan 7s linear infinite;pointer-events:none;z-index:1
    }
    @keyframes scan{0%{top:-140px}100%{top:100vh}}

    header{background:linear-gradient(180deg,#0b0f15,#090c11);border-bottom:1px solid var(--border);box-shadow:0 1px 0 rgba(0,229,255,.18);padding:14px 24px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:10}
    .logo{font-size:1rem;font-weight:700;color:#eaf6fa;letter-spacing:3px;text-transform:uppercase}
    .logo span{color:var(--cyan);text-shadow:0 0 12px rgba(0,229,255,.55)}
    .status-bar{display:flex;gap:24px;align-items:center;font-size:.78rem;color:#5a6b78;letter-spacing:.3px}
    .dot{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:5px}
    .dot.green{background:#00e676;box-shadow:0 0 6px #00e676}
    .dot.red{background:#ff1744;box-shadow:0 0 6px #ff1744}
    .dot.yellow{background:#ffd600;box-shadow:0 0 6px #ffd600;animation:pulse 1s infinite}
    @keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}

    .tabs{display:flex;border-bottom:1px solid var(--border);background:#090c11;padding:0 24px;position:relative;z-index:2}
    .tab{padding:12px 20px;cursor:pointer;font-size:.74rem;color:#4a5a66;border-bottom:2px solid transparent;transition:color .15s,border-color .15s,text-shadow .15s;text-transform:uppercase;letter-spacing:1.5px;font-weight:600}
    .tab.active{color:var(--cyan);border-bottom-color:var(--cyan);text-shadow:0 0 8px rgba(0,229,255,.5)}
    .tab:hover:not(.active){color:#93a6b2}
    .tab-panel{display:none}
    .tab-panel.active{display:block}

    .main{padding:24px;max-width:1200px;margin:0 auto;position:relative;z-index:2}
    #alert-count{font-size:.8rem;color:#5a6b78;margin-bottom:18px}

    .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:16px}
    .card{background:var(--panel);border:1px solid var(--border);border-radius:5px;padding:16px;position:relative;transition:transform .15s,border-color .15s,box-shadow .15s}
    .card:hover{transform:translateY(-2px);border-color:rgba(0,229,255,.35);box-shadow:0 10px 26px rgba(0,229,255,.06)}
    .card::before,.card::after{content:'';position:absolute;top:-1px;width:9px;height:9px;border-top:1px solid rgba(0,229,255,.45);pointer-events:none}
    .card::before{left:-1px;border-left:1px solid rgba(0,229,255,.45)}
    .card::after{right:-1px;border-right:1px solid rgba(0,229,255,.45)}
    .card.long{border-left:3px solid #00e676}
    .card.short{border-left:3px solid #ff1744}
    .card.orb{border-left:3px solid #ffd600}
    .card-time{position:absolute;top:12px;right:12px;font-size:.68rem;color:#3a4854}
    .card-header{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px}
    .ticker{font-size:1.3rem;font-weight:bold;color:#eaf6fa}
    .setup-name{font-size:.75rem;color:#5a6b78;margin-top:2px;text-transform:uppercase;letter-spacing:.5px}
    .bias-badge{padding:3px 10px;border-radius:3px;font-size:.72rem;font-weight:bold;text-transform:uppercase;letter-spacing:.5px}
    .bias-badge.long{background:rgba(0,230,118,.12);color:#00e676;border:1px solid #00e676}
    .bias-badge.short{background:rgba(255,23,68,.12);color:#ff1744;border:1px solid #ff1744}
    .levels{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px}
    .targets{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:12px}
    .level-item label{font-size:.63rem;color:#4a5a66;text-transform:uppercase;letter-spacing:1px;display:block;margin-bottom:2px}
    .val{font-size:1rem;font-weight:bold}
    .val.entry{color:#eaf6fa}
    .val.stop{color:#ff1744}
    .val.target{color:#00e676}
    .card-footer{display:flex;justify-content:space-between;align-items:flex-start;border-top:1px solid var(--border);padding-top:10px}
    .note{font-size:.72rem;color:#5a6b78;flex:1;margin-right:14px;line-height:1.5}
    .meta-right{text-align:right;font-size:.72rem;white-space:nowrap}
    .vol{color:#ffd600}
    .rr{color:#4a5a66;margin-top:2px}
    .new-badge{position:absolute;top:-7px;right:10px;background:#ffd600;color:#000;font-size:.6rem;font-weight:bold;padding:2px 7px;border-radius:10px;box-shadow:0 0 10px rgba(255,214,0,.5)}
    .empty{grid-column:1/-1;text-align:center;padding:80px 20px;color:#263340}
    .empty .icon{font-size:2.5rem;margin-bottom:14px}
    .empty p{font-size:.85rem;line-height:1.8}

    .replay-controls{display:flex;gap:12px;align-items:center;margin-bottom:24px;flex-wrap:wrap}
    select{background:var(--panel);border:1px solid var(--border);color:#c3d1da;padding:8px 14px;border-radius:4px;font-family:inherit;font-size:.85rem;cursor:pointer}
    select:focus{outline:none;border-color:var(--cyan);box-shadow:0 0 0 2px rgba(0,229,255,.15)}
    .btn{background:var(--cyan);color:#00161a;border:none;padding:9px 22px;border-radius:4px;font-family:inherit;font-size:.76rem;font-weight:700;letter-spacing:1px;text-transform:uppercase;cursor:pointer;transition:box-shadow .15s,transform .15s}
    .btn:hover{box-shadow:0 0 16px rgba(0,229,255,.45);transform:translateY(-1px)}
    .btn:disabled{opacity:.35;cursor:not-allowed;box-shadow:none;transform:none}
    .btn.small{padding:5px 14px;font-size:.68rem}
    .btn.logged{background:var(--panel);color:#00e676;border:1px solid rgba(0,230,118,.4);cursor:default;box-shadow:none}
    #replay-status{font-size:.8rem;color:#5a6b78}

    .timeline{display:flex;flex-direction:column}
    .tl-item{display:flex;position:relative}
    .tl-left{width:70px;flex-shrink:0;text-align:right;padding-right:16px;padding-top:14px;font-size:.75rem;color:#4a5a66;font-weight:bold}
    .tl-line{display:flex;flex-direction:column;align-items:center;flex-shrink:0;width:20px}
    .tl-dot{width:10px;height:10px;border-radius:50%;margin-top:16px;flex-shrink:0}
    .tl-dot.long{background:#00e676;box-shadow:0 0 6px #00e676}
    .tl-dot.short{background:#ff1744;box-shadow:0 0 6px #ff1744}
    .tl-dot.orb{background:#ffd600;box-shadow:0 0 6px #ffd600}
    .tl-connector{width:1px;background:var(--border);flex:1}
    .tl-card{flex:1;background:var(--panel);border:1px solid var(--border);border-radius:5px;padding:12px 16px;margin:8px 0 8px 12px}
    .tl-card.long{border-left:3px solid #00e676}
    .tl-card.short{border-left:3px solid #ff1744}
    .tl-card.orb{border-left:3px solid #ffd600}
    .tl-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}
    .tl-ticker{font-size:1.1rem;font-weight:bold;color:#eaf6fa}
    .tl-setup{font-size:.75rem;color:#5a6b78;text-transform:uppercase;letter-spacing:.5px}
    .tl-levels{display:flex;gap:16px;font-size:.78rem;margin-bottom:6px;flex-wrap:wrap}
    .tl-levels span{color:#4a5a66}
    .tl-levels span b{color:#eaf6fa}
    .tl-levels span b.stop{color:#ff1744}
    .tl-levels span b.target{color:#00e676}
    .tl-levels span b.vol{color:#ffd600}
    .tl-note{font-size:.7rem;color:#4a5a66;margin-top:4px;margin-bottom:8px}
    .tl-actions{display:flex;gap:8px;align-items:center;margin-top:6px}
    .tl-empty{text-align:center;padding:60px 20px;color:#263340;font-size:.85rem}

    .opts-panel{margin-top:10px;background:var(--bg);border:1px solid var(--border);border-radius:5px;padding:12px 14px;display:none}
    .opts-panel.open{display:block}
    .opts-header{display:flex;gap:16px;align-items:center;margin-bottom:10px;flex-wrap:wrap}
    .opts-type{font-size:.85rem;font-weight:bold;color:#eaf6fa}
    .opts-type.call{color:#00e676}
    .opts-type.put{color:#ff1744}
    .opts-meta{font-size:.72rem;color:#5a6b78}
    .opts-meta span{margin-right:12px}
    .opts-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
    .opts-cell{background:var(--panel);border-radius:4px;padding:8px 10px;border:1px solid var(--border)}
    .opts-cell label{font-size:.6rem;color:#4a5a66;text-transform:uppercase;letter-spacing:1px;display:block;margin-bottom:3px}
    .opts-cell .ov{font-size:.9rem;font-weight:bold;color:#eaf6fa}
    .opts-cell .ov.pos{color:#00e676}
    .opts-cell .ov.neg{color:#ff1744}
    .opts-cell .sub{font-size:.68rem;color:#5a6b78;margin-top:2px}
    .opts-disclaimer{font-size:.65rem;color:#3a4854;margin-top:8px}

    .stats-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:16px;margin-bottom:28px}
    .stat-card{background:var(--panel);border:1px solid var(--border);border-radius:5px;padding:16px 20px;position:relative}
    .stat-card::before{content:'';position:absolute;top:0;left:0;width:100%;height:2px;background:linear-gradient(90deg,rgba(0,229,255,.5),transparent)}
    .stat-label{font-size:.63rem;color:#4a5a66;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px}
    .stat-value{font-size:1.6rem;font-weight:bold;color:#eaf6fa}
    .stat-value.green{color:#00e676}
    .stat-value.red{color:#ff1744}
    .stat-value.yellow{color:#ffd600}
    .stat-sub{font-size:.7rem;color:#4a5a66;margin-top:4px}
    .section-title{font-size:.7rem;color:var(--cyan);text-transform:uppercase;letter-spacing:2px;margin-bottom:14px;margin-top:28px;opacity:.75}

    table{width:100%;border-collapse:collapse;font-size:.78rem}
    th{text-align:left;color:#4a5a66;font-size:.62rem;text-transform:uppercase;letter-spacing:1px;padding:8px 12px;border-bottom:1px solid var(--border)}
    td{padding:9px 12px;border-bottom:1px solid #10161d;color:#a8b8c2;vertical-align:middle}
    tr:hover td{background:rgba(0,229,255,.03)}

    .outcome-badge{padding:2px 8px;border-radius:3px;font-size:.7rem;font-weight:bold;display:inline-block;text-transform:uppercase;letter-spacing:.5px}
    .outcome-badge.win_t1{background:rgba(0,230,118,.12);color:#00e676;border:1px solid #00e67644}
    .outcome-badge.win_t2{background:rgba(0,230,118,.2);color:#00e676;border:1px solid #00e676}
    .outcome-badge.loss{background:rgba(255,23,68,.12);color:#ff1744;border:1px solid #ff174444}
    .outcome-badge.open{background:rgba(255,214,0,.08);color:#ffd600;border:1px solid #ffd60044}
    .r-pos{color:#00e676;font-weight:bold}
    .r-neg{color:#ff1744;font-weight:bold}
    .del-btn{background:none;border:none;color:#263340;cursor:pointer;font-size:.9rem;padding:2px 6px}
    .del-btn:hover{color:#ff1744}
    .perf-empty{text-align:center;padding:80px 20px;color:#263340}
    .perf-empty .icon{font-size:2.5rem;margin-bottom:14px}
    .perf-empty p{font-size:.85rem;line-height:1.8}

    .mode-badge{padding:3px 12px;border-radius:3px;font-size:.72rem;font-weight:bold;letter-spacing:1px;text-transform:uppercase}
    .mode-badge.paper{background:rgba(0,230,118,.12);color:#00e676;border:1px solid #00e676}
    .mode-badge.dry_run{background:rgba(150,150,150,.12);color:#999;border:1px solid #666}
    .mode-badge.live{background:rgba(255,23,68,.15);color:#ff1744;border:1px solid #ff1744;animation:pulse 1.2s infinite}
    .mode-badge.unknown{background:rgba(255,214,0,.1);color:#ffd600;border:1px solid #ffd60044}
    .control-row{display:flex;gap:14px;align-items:center;margin-bottom:20px;flex-wrap:wrap}
    .btn.stop-btn{background:var(--panel);color:#ff1744;border:1px solid rgba(255,23,68,.4)}
    .btn.stop-btn:hover{box-shadow:0 0 16px rgba(255,23,68,.3)}
    .warn-note{font-size:.75rem;color:#ffd600;background:rgba(255,214,0,.05);border:1px solid rgba(255,214,0,.15);border-radius:4px;padding:10px 14px;margin-bottom:20px;line-height:1.6}
    .log-tail{background:#04060a;border:1px solid var(--border);border-radius:4px;padding:14px;font-size:.72rem;line-height:1.7;color:#5a9aa8;max-height:360px;overflow-y:auto;white-space:pre-wrap;box-shadow:inset 0 0 24px rgba(0,229,255,.03)}
    .tool-card{background:var(--panel);border:1px solid var(--border);border-top:2px solid rgba(0,229,255,.35);border-radius:5px;padding:18px 20px;margin-bottom:16px}
    .tool-card h3{font-size:.8rem;color:#eaf6fa;margin-bottom:10px;text-transform:uppercase;letter-spacing:1px}
    .tool-card .sub{font-size:.72rem;color:#5a6b78;margin-bottom:14px}
    .spinner{display:inline-block;width:11px;height:11px;border:2px solid #1a2530;border-top-color:var(--cyan);border-radius:50%;animation:spin .7s linear infinite;margin-right:6px;vertical-align:middle}
    @keyframes spin{to{transform:rotate(360deg)}}
  </style>
</head>
<body>

<header>
  <div class="logo">Pattern <span>Scanner</span></div>
  <div class="status-bar">
    <span id="market-status"><span class="dot red"></span>Market Closed</span>
    <span id="scan-status">Last scan: —</span>
    <span id="clock">—</span>
  </div>
</header>

<div class="tabs">
  <div class="tab active" onclick="switchTab('live')">Live Alerts</div>
  <div class="tab"        onclick="switchTab('replay')">Replay Today</div>
  <div class="tab"        onclick="switchTab('perf')">Performance</div>
  <div class="tab"        onclick="switchTab('auto')">Auto Trader</div>
  <div class="tab"        onclick="switchTab('reports')">Reports</div>
  <div class="tab"        onclick="switchTab('strategy')">Strategy Ideas</div>
</div>

<!-- LIVE -->
<div id="tab-live" class="tab-panel active">
  <div class="main">
    <div id="alert-count">Connecting...</div>
    <div class="grid" id="alert-grid"></div>
  </div>
</div>

<!-- REPLAY -->
<div id="tab-replay" class="tab-panel">
  <div class="main">
    <div class="replay-controls">
      <select id="replay-ticker"><option value="">Select ticker...</option></select>
      <button class="btn" id="replay-btn" onclick="runReplay()" disabled>Replay Day</button>
      <span id="replay-status"></span>
    </div>
    <div id="replay-output"></div>
  </div>
</div>

<!-- PERFORMANCE -->
<div id="tab-perf" class="tab-panel">
  <div class="main">
    <div id="perf-content"><div class="perf-empty"><div class="icon">◎</div><p>Loading...</p></div></div>
  </div>
</div>

<!-- AUTO TRADER -->
<div id="tab-auto" class="tab-panel">
  <div class="main">
    <div class="warn-note">Stopping the auto trader does <b>not</b> close any open option positions — check TWS directly. Starting it places real orders through Interactive Brokers (paper or live, per the mode shown below).</div>
    <div class="control-row">
      <span id="auto-status-dot"><span class="dot red"></span>Stopped</span>
      <span id="auto-mode-badge"></span>
      <button class="btn" id="auto-start-btn" onclick="autoTraderStart()">Start</button>
      <button class="btn stop-btn" id="auto-stop-btn" onclick="autoTraderStop()" disabled>Stop</button>
    </div>
    <div class="section-title">Recent Activity</div>
    <div class="log-tail" id="auto-log-tail">No activity yet.</div>
  </div>
</div>

<!-- REPORTS -->
<div id="tab-reports" class="tab-panel">
  <div class="main">
    <div class="tool-card">
      <h3>Backtest</h3>
      <div class="sub">Runs backtest.py against the current watchlist/strategy and writes backtest_results.csv.</div>
      <button class="btn" id="backtest-run-btn" onclick="runTool('backtest')">Run Backtest</button>
      <button class="btn" style="background:#1a1a1a;color:#e0e0e0;border:1px solid #2a2a2a" onclick="loadBacktestResults()">View Results</button>
      <span id="backtest-status" style="font-size:.78rem;margin-left:10px;color:#555"></span>
      <div id="backtest-results" style="margin-top:16px;overflow-x:auto"></div>
    </div>
    <div class="tool-card">
      <h3>Monte Carlo</h3>
      <div class="sub">Runs monte_carlo.py and regenerates monte_carlo.html.</div>
      <button class="btn" id="montecarlo-run-btn" onclick="runTool('montecarlo')">Run Monte Carlo</button>
      <span id="montecarlo-status" style="font-size:.78rem;margin-left:10px;color:#555"></span>
    </div>
  </div>
</div>

<!-- STRATEGY IDEAS -->
<div id="tab-strategy" class="tab-panel">
  <div class="main">
    <div class="warn-note" style="color:var(--cyan);background:rgba(0,229,255,.05);border-color:rgba(0,229,255,.15)">
      Advisory only — nothing here changes live behavior automatically. Sweeps MIN_SCORE against backtest.py's cached
      data and compares it to the current setting. Applying a suggestion means manually editing the constant yourself.
    </div>
    <div class="tool-card">
      <h3>Parameter Sweep</h3>
      <div class="sub">Re-runs the backtest across MIN_SCORE = 1–5 (same data, no re-fetch) and ranks the results. Takes a few minutes.</div>
      <button class="btn" id="optimize-run-btn" onclick="runTool('optimize')">Run Sweep</button>
      <button class="btn" style="background:#1a1a1a;color:#e0e0e0;border:1px solid #2a2a2a" onclick="loadOptimizeResults()">View Results</button>
      <span id="optimize-status" style="font-size:.78rem;margin-left:10px;color:#555"></span>
    </div>
    <div id="optimize-results"></div>
  </div>
</div>

<script>
const TAB_NAMES = ['live','replay','perf','auto','reports','strategy'];
function switchTab(name) {
  document.querySelectorAll('.tab').forEach((t,i) => t.classList.toggle('active', TAB_NAMES[i] === name));
  TAB_NAMES.forEach(n => document.getElementById('tab-'+n).classList.toggle('active', n === name));
  if (name === 'perf') loadPerf();
  if (name === 'auto') pollAutoTrader();
  if (name === 'reports') loadBacktestResults();
  if (name === 'strategy') loadOptimizeResults();
}

function biasClass(a) { return a.bias === 'LONG' ? 'long' : 'short'; }
function borderClass(a) { return a.setup.startsWith('ORB') ? 'orb' : biasClass(a); }

function card(a, isNew) {
  return `<div class="card ${borderClass(a)}">
    ${isNew ? '<span class="new-badge">NEW</span>' : ''}
    <div class="card-time">${a.time||''}</div>
    <div class="card-header">
      <div><div class="ticker">${a.ticker}</div><div class="setup-name">${a.setup}</div></div>
      <div class="bias-badge ${biasClass(a)}">${a.bias}</div>
    </div>
    <div class="levels">
      <div class="level-item"><label>Entry</label><div class="val entry">$${a.entry}</div></div>
      <div class="level-item"><label>Stop</label><div class="val stop">$${a.stop}</div></div>
    </div>
    <div class="targets">
      <div class="level-item"><label>Target 1</label><div class="val target">$${a.target1}</div></div>
      <div class="level-item"><label>Target 2</label><div class="val target">$${a.target2}</div></div>
    </div>
    <div class="card-footer">
      <div class="note">${a.note}</div>
      <div class="meta-right"><div class="vol">${a.vol_ratio}x vol</div><div class="rr">${a.rr}</div></div>
    </div>
  </div>`;
}

function tlItem(a, isLast) {
  const bc   = borderClass(a);
  const key  = `${a.ticker}-${a.setup}-${a.time}`.replace(/[^a-zA-Z0-9]/g,'_');
  const aJson = JSON.stringify(a).replace(/"/g, '&quot;');
  const params = `ticker=${a.ticker}&price=${a.entry}&bias=${a.bias}&target1=${a.target1}&target2=${a.target2}&stop=${a.stop}`;
  return `<div class="tl-item">
    <div class="tl-left">${a.time}</div>
    <div class="tl-line">
      <div class="tl-dot ${bc}"></div>
      ${isLast ? '' : '<div class="tl-connector"></div>'}
    </div>
    <div class="tl-card ${bc}">
      <div class="tl-header">
        <div><span class="tl-ticker">${a.ticker}</span>&nbsp;<span class="tl-setup">${a.setup}</span></div>
        <div class="bias-badge ${biasClass(a)}">${a.bias}</div>
      </div>
      <div class="tl-levels">
        <span>Entry <b>$${a.entry}</b></span>
        <span>Stop <b class="stop">$${a.stop}</b></span>
        <span>T1 <b class="target">$${a.target1}</b></span>
        <span>T2 <b class="target">$${a.target2}</b></span>
        <span>Vol <b class="vol">${a.vol_ratio}x</b></span>
        <span>${a.rr}</span>
      </div>
      <div class="tl-note">${a.note}</div>
      <div class="tl-actions">
        <button class="btn small" id="log-${key}" onclick='logTrade("${key}", ${aJson})'>Log Trade</button>
        <button class="btn small" style="background:#1a1a1a;color:#ffd600;border:1px solid #ffd60044"
          id="opts-btn-${key}" onclick='loadOptions("${key}","${params}")'>Options</button>
        <span id="log-result-${key}" style="font-size:.72rem"></span>
      </div>
      <div class="opts-panel" id="opts-${key}">
        <div id="opts-content-${key}" style="color:#555;font-size:.78rem">Loading...</div>
      </div>
    </div>
  </div>`;
}

// Live polling
let knownKeys = new Set();
function alertKey(a) { return `${a.ticker}-${a.setup}-${a.time}`; }

async function poll() {
  try {
    const res = await fetch('/api/state');
    const data = await res.json();
    document.getElementById('clock').textContent = data.time;
    document.getElementById('market-status').innerHTML = data.market_open
      ? '<span class="dot green"></span>Market Open'
      : '<span class="dot red"></span>Market Closed';
    document.getElementById('scan-status').innerHTML = data.scanning
      ? '<span class="dot yellow"></span>Scanning...'
      : `Last scan: ${data.last_scan}`;

    const newKeys   = new Set(data.alerts.map(alertKey));
    const freshKeys = new Set([...newKeys].filter(k => !knownKeys.has(k)));
    knownKeys = newKeys;

    const grid = document.getElementById('alert-grid');
    const countEl = document.getElementById('alert-count');
    if (!data.alerts.length) {
      grid.innerHTML = `<div class="empty"><div class="icon">◎</div><p>No patterns detected yet.<br>Scans run every 2 minutes during market hours.</p></div>`;
      countEl.textContent = 'No alerts this session';
    } else {
      grid.innerHTML = data.alerts.map(a => card(a, freshKeys.has(alertKey(a)))).join('');
      countEl.textContent = `${data.alerts.length} alert${data.alerts.length!==1?'s':''} this session`;
    }

    const sel = document.getElementById('replay-ticker');
    if (data.watchlist && sel.options.length === 1) {
      data.watchlist.forEach(t => { const o = document.createElement('option'); o.value = o.textContent = t; sel.appendChild(o); });
    }
  } catch(e) {
    document.getElementById('scan-status').textContent = 'Connection error — retrying...';
  }
}

document.getElementById('replay-ticker').addEventListener('change', function() {
  document.getElementById('replay-btn').disabled = !this.value;
});

// Replay
async function runReplay() {
  const ticker = document.getElementById('replay-ticker').value;
  if (!ticker) return;
  const btn = document.getElementById('replay-btn');
  const statusEl = document.getElementById('replay-status');
  btn.disabled = true;
  statusEl.innerHTML = '<span class="dot yellow"></span>Fetching data...';
  document.getElementById('replay-output').innerHTML = '';
  try {
    const res  = await fetch(`/api/replay/${ticker}`);
    const data = await res.json();
    if (data.error) { statusEl.textContent = data.error; btn.disabled = false; return; }
    const setups = data.setups;
    statusEl.textContent = setups.length
      ? `${setups.length} setup${setups.length!==1?'s':''} found today for ${ticker}`
      : `No setups triggered today for ${ticker}`;
    document.getElementById('replay-output').innerHTML = setups.length
      ? `<div class="timeline">${setups.map((a,i) => tlItem(a, i===setups.length-1)).join('')}</div>`
      : `<div class="tl-empty">No pattern setups detected for ${ticker} today.</div>`;
  } catch(e) { statusEl.textContent = 'Error fetching replay data.'; }
  btn.disabled = false;
}

// Log trade
async function logTrade(key, alert) {
  const btn   = document.getElementById(`log-${key}`);
  const resEl = document.getElementById(`log-result-${key}`);
  btn.disabled = true;
  btn.textContent = 'Checking...';
  try {
    const res  = await fetch('/api/log_trade', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        ticker: alert.ticker, setup: alert.setup, bias: alert.bias,
        entry: alert.entry, stop: alert.stop, target1: alert.target1, target2: alert.target2,
        entry_time: alert.time, note: alert.note,
      })
    });
    const data = await res.json();
    const t = data.trade;
    btn.className = 'btn small logged';
    btn.textContent = 'Logged';
    const oc = t.outcome;
    if (oc === 'win_t1') resEl.innerHTML = `<span style="color:#00e676">Win T1 &nbsp;+${t.r_multiple}R @ $${t.exit_price} (${t.exit_time})</span>`;
    else if (oc === 'win_t2') resEl.innerHTML = `<span style="color:#00e676">Win T2 &nbsp;+${t.r_multiple}R @ $${t.exit_price} (${t.exit_time})</span>`;
    else if (oc === 'loss')   resEl.innerHTML = `<span style="color:#ff1744">Loss &nbsp;-1R @ $${t.exit_price} (${t.exit_time})</span>`;
    else resEl.innerHTML = `<span style="color:#ffd600">Open — outcome will update in Performance tab</span>`;
  } catch(e) {
    btn.disabled = false; btn.textContent = 'Log Trade';
    resEl.textContent = 'Error';
  }
}

// Performance
function outcomeBadge(oc) {
  const labels = {win_t1:'Win T1', win_t2:'Win T2', loss:'Loss', open:'Open'};
  return `<span class="outcome-badge ${oc}">${labels[oc]||oc}</span>`;
}
function rBadge(r) {
  if (r === null || r === undefined) return '—';
  return r >= 0 ? `<span class="r-pos">+${r}R</span>` : `<span class="r-neg">${r}R</span>`;
}

async function loadPerf() {
  const el = document.getElementById('perf-content');
  try {
    const res  = await fetch('/api/trades');
    const data = await res.json();
    const s = data.stats, trades = data.trades;

    if (!trades.length) {
      el.innerHTML = `<div class="perf-empty"><div class="icon">◎</div>
        <p>No trades logged yet.<br>Go to <b>Replay Today</b>, pick a ticker, and click <b>Log Trade</b> on any setup.</p></div>`;
      return;
    }

    const wrColor = s.win_rate >= 55 ? 'green' : s.win_rate >= 45 ? 'yellow' : 'red';
    const exColor = s.expectancy >= 0 ? 'green' : 'red';

    let html = `<div class="stats-grid">
      <div class="stat-card">
        <div class="stat-label">Win Rate</div>
        <div class="stat-value ${wrColor}">${s.win_rate}%</div>
        <div class="stat-sub">${s.wins}W / ${s.losses}L &nbsp;(${s.open} open)</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Total Trades</div>
        <div class="stat-value">${s.total}</div>
        <div class="stat-sub">${s.closed} closed</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Expectancy</div>
        <div class="stat-value ${exColor}">${s.expectancy>=0?'+':''}${s.expectancy}R</div>
        <div class="stat-sub">per trade avg</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Avg Win</div>
        <div class="stat-value green">${s.avg_r_win>=0?'+':''}${s.avg_r_win}R</div>
        <div class="stat-sub">on winners</div>
      </div>
    </div>`;

    if (s.breakdown.length) {
      html += `<div class="section-title">By Setup</div>
        <table>
          <tr><th>Setup</th><th>Trades</th><th>Wins</th><th>Losses</th><th>Win %</th><th>Avg R</th></tr>
          ${s.breakdown.map(b => {
            const wc = b.win_rate >= 55 ? '#00e676' : b.win_rate < 45 ? '#ff1744' : '#ffd600';
            return `<tr>
              <td>${b.setup}</td><td>${b.trades}</td>
              <td style="color:#00e676">${b.wins}</td>
              <td style="color:#ff1744">${b.losses}</td>
              <td style="color:${wc};font-weight:bold">${b.win_rate}%</td>
              <td>${rBadge(b.avg_r)}</td>
            </tr>`;
          }).join('')}
        </table>`;
    }

    html += `<div class="section-title">Trade History</div>
      <table>
        <tr><th>Date</th><th>Time</th><th>Ticker</th><th>Setup</th><th>Bias</th><th>Entry</th><th>Exit</th><th>Outcome</th><th>R</th><th></th></tr>
        ${trades.map(t => `<tr>
          <td>${t.date}</td><td>${t.entry_time}</td>
          <td style="font-weight:bold;color:#fff">${t.ticker}</td>
          <td>${t.setup}</td>
          <td><span class="bias-badge ${t.bias.toLowerCase()}">${t.bias}</span></td>
          <td>$${t.entry}</td>
          <td>${t.exit_price ? '$'+t.exit_price+' ('+t.exit_time+')' : '—'}</td>
          <td>${outcomeBadge(t.outcome)}</td>
          <td>${rBadge(t.r_multiple)}</td>
          <td><button class="del-btn" onclick="deleteTrade('${t.id}')" title="Remove">✕</button></td>
        </tr>`).join('')}
      </table>`;

    el.innerHTML = html;
  } catch(e) {
    el.innerHTML = '<div class="perf-empty"><p>Error loading performance data.</p></div>';
  }
}

async function deleteTrade(id) {
  await fetch(`/api/delete_trade/${id}`, {method:'DELETE'});
  loadPerf();
}

// Options
async function loadOptions(key, params) {
  const panel   = document.getElementById(`opts-${key}`);
  const content = document.getElementById(`opts-content-${key}`);
  const btn     = document.getElementById(`opts-btn-${key}`);

  if (panel.classList.contains('open')) {
    panel.classList.remove('open');
    btn.style.color = '#ffd600';
    return;
  }

  panel.classList.add('open');
  btn.style.color = '#fff';
  content.textContent = 'Fetching options data...';

  try {
    const res  = await fetch(`/api/options?${params}`);
    const data = await res.json();

    if (data.error) {
      content.textContent = `Options unavailable: ${data.error}`;
      return;
    }

    const pnlColor = v => v >= 0 ? 'pos' : 'neg';
    const pnlFmt   = v => (v >= 0 ? '+' : '') + '$' + Math.abs(v).toFixed(2);

    content.innerHTML = `
      <div class="opts-header">
        <span class="opts-type ${data.type.toLowerCase()}">${data.type}</span>
        <div class="opts-meta">
          <span>Strike $${data.strike}</span>
          <span>Exp ${data.expiry} (${data.days_to_exp}d)</span>
          <span>IV ${data.iv}%</span>
          <span>Delta ${data.delta}</span>
        </div>
      </div>
      <div class="opts-grid">
        <div class="opts-cell">
          <label>Premium (mid)</label>
          <div class="ov">$${data.mid}</div>
          <div class="sub">$${data.premium} / contract</div>
        </div>
        <div class="opts-cell">
          <label>Bid / Ask</label>
          <div class="ov">$${data.bid} / $${data.ask}</div>
          <div class="sub">spread $${(data.ask - data.bid).toFixed(2)}</div>
        </div>
        <div class="opts-cell">
          <label>Max Risk</label>
          <div class="ov neg">-$${data.premium}</div>
          <div class="sub">if expires worthless</div>
        </div>
        <div class="opts-cell">
          <label>If Stop Hit</label>
          <div class="ov ${pnlColor(data.pnl_stop)}">${pnlFmt(data.pnl_stop)}</div>
          <div class="sub">val ≈ $${data.val_stop} ($${(data.val_stop*100).toFixed(0)}/contract)</div>
        </div>
        <div class="opts-cell">
          <label>If T1 Hit</label>
          <div class="ov ${pnlColor(data.pnl_t1)}">${pnlFmt(data.pnl_t1)}</div>
          <div class="sub">val ≈ $${data.val_t1} ($${(data.val_t1*100).toFixed(0)}/contract)</div>
        </div>
        <div class="opts-cell">
          <label>If T2 Hit</label>
          <div class="ov ${pnlColor(data.pnl_t2)}">${pnlFmt(data.pnl_t2)}</div>
          <div class="sub">val ≈ $${data.val_t2} ($${(data.val_t2*100).toFixed(0)}/contract)</div>
        </div>
      </div>
      <div class="opts-disclaimer">* Delta-based estimates only. Actual value depends on theta, IV change, and time remaining.</div>`;
  } catch(e) {
    content.textContent = 'Error loading options data.';
  }
}

// Auto trader
const MODE_LABELS = {PAPER:'PAPER', DRY_RUN:'DRY RUN', LIVE:'⚠ LIVE', UNKNOWN:'UNKNOWN'};
async function pollAutoTrader() {
  const tab = document.getElementById('tab-auto');
  if (!tab.classList.contains('active')) return;
  try {
    const res  = await fetch('/api/autotrader/status');
    const data = await res.json();
    document.getElementById('auto-status-dot').innerHTML = data.running
      ? '<span class="dot green"></span>Running (pid ' + data.pid + (data.external ? ', started outside this app' : '') + ')'
      : '<span class="dot red"></span>Stopped';
    document.getElementById('auto-mode-badge').innerHTML =
      `<span class="mode-badge ${data.mode.toLowerCase()}">${MODE_LABELS[data.mode] || data.mode}</span>`;
    document.getElementById('auto-start-btn').disabled = data.running;
    document.getElementById('auto-stop-btn').disabled = !data.running || data.external;
    document.getElementById('auto-stop-btn').title = data.external
      ? 'Started outside this app — stop it from wherever it was launched' : '';
    document.getElementById('auto-log-tail').textContent = data.log_tail.length
      ? data.log_tail.join('\n') : 'No activity yet.';
  } catch(e) {}
}

async function autoTraderStart() {
  document.getElementById('auto-start-btn').disabled = true;
  try { await fetch('/api/autotrader/start', {method:'POST'}); } catch(e) {}
  pollAutoTrader();
}

async function autoTraderStop() {
  document.getElementById('auto-stop-btn').disabled = true;
  try { await fetch('/api/autotrader/stop', {method:'POST'}); } catch(e) {}
  pollAutoTrader();
}

setInterval(pollAutoTrader, 5000);

// Reports
async function runTool(tool) {
  const btn = document.getElementById(`${tool}-run-btn`);
  const statusEl = document.getElementById(`${tool}-status`);
  btn.disabled = true;
  statusEl.innerHTML = '<span class="spinner"></span>Running...';
  try {
    await fetch(`/api/tools/${tool}/run`, {method:'POST'});
    const poll = setInterval(async () => {
      const res  = await fetch(`/api/tools/${tool}/status`);
      const data = await res.json();
      if (!data.running) {
        clearInterval(poll);
        btn.disabled = false;
        statusEl.textContent = 'Done.';
        if (tool === 'montecarlo') window.open('/monte_carlo.html', '_blank');
        if (tool === 'backtest') loadBacktestResults();
        if (tool === 'optimize') loadOptimizeResults();
      }
    }, 2000);
  } catch(e) {
    btn.disabled = false;
    statusEl.textContent = 'Error starting run.';
  }
}

async function loadBacktestResults() {
  const el = document.getElementById('backtest-results');
  el.innerHTML = '<span style="color:#555;font-size:.78rem">Loading...</span>';
  try {
    const res = await fetch('/api/tools/backtest/results');
    if (res.status === 404) {
      el.innerHTML = '<span style="color:#555;font-size:.78rem">No backtest results yet — click Run Backtest.</span>';
      return;
    }
    const data = await res.json();
    const rows = [...data.rows].reverse();
    const wins = rows.filter(r => String(r.result).startsWith('t')).length;
    const totalPnl = rows.reduce((s, r) => s + (r['pnl_$'] || 0), 0);
    el.innerHTML = `
      <div style="font-size:.78rem;color:#555;margin-bottom:10px">
        Updated ${data.updated} &nbsp;·&nbsp; ${rows.length} trades &nbsp;·&nbsp;
        ${rows.length ? (100*wins/rows.length).toFixed(1) : 0}% win rate &nbsp;·&nbsp;
        <span class="${totalPnl>=0?'r-pos':'r-neg'}">${totalPnl>=0?'+':''}$${totalPnl.toFixed(2)} total</span>
      </div>
      <table>
        <thead><tr>${data.columns.map(c => `<th>${c}</th>`).join('')}</tr></thead>
        <tbody>${rows.map(r => `<tr>${data.columns.map(c => `<td>${r[c] === null ? '—' : r[c]}</td>`).join('')}</tr>`).join('')}</tbody>
      </table>`;
  } catch(e) {
    el.innerHTML = '<span style="color:#ff1744;font-size:.78rem">Error loading results.</span>';
  }
}

function isCurrentConfig(r, current) {
  return r.min_score === current.min_score && r.max_stop === current.max_stop && r.cooldown_bars === current.cooldown_bars;
}

async function loadOptimizeResults() {
  const el = document.getElementById('optimize-results');
  el.innerHTML = '<span style="color:#555;font-size:.78rem">Loading...</span>';
  try {
    const res = await fetch('/api/tools/optimize/results');
    if (res.status === 404) {
      el.innerHTML = '<span style="color:#555;font-size:.78rem">No sweep results yet — click Run Sweep.</span>';
      return;
    }
    const data = await res.json();
    const rows = data.results;
    el.innerHTML = `
      <div class="tool-card" style="border-top-color:rgba(0,229,255,.5)">
        <div style="font-size:.85rem;color:#eaf6fa;line-height:1.6">${data.suggestion}</div>
        <div style="font-size:.72rem;color:#5a6b78;margin-top:8px">Generated ${new Date(data.generated_at).toLocaleString()} · ${data.lookback_days}-day window</div>
      </div>
      <div style="overflow-x:auto">
        <table>
          <thead><tr><th>Min Score</th><th>Max Stop</th><th>Cooldown</th><th># Trades</th><th>Win Rate</th><th>Expectancy $</th><th>Expectancy R</th><th>Total P&L</th><th>Max Drawdown</th><th></th></tr></thead>
          <tbody>${rows.map(r => `
            <tr style="${isCurrentConfig(r, data.current) ? 'background:rgba(0,229,255,.06)' : ''}">
              <td>${r.min_score}</td>
              <td>$${r.max_stop.toFixed(2)}</td>
              <td>${r.cooldown_bars}</td>
              <td>${r.trades}</td>
              <td>${r.win_rate}%</td>
              <td class="${r.expectancy_dollars>=0?'r-pos':'r-neg'}">${r.expectancy_dollars>=0?'+':''}$${r.expectancy_dollars.toFixed(2)}</td>
              <td class="${r.expectancy_r>=0?'r-pos':'r-neg'}">${r.expectancy_r>=0?'+':''}${r.expectancy_r.toFixed(2)}R</td>
              <td class="${r.total_pnl>=0?'r-pos':'r-neg'}">${r.total_pnl>=0?'+':''}$${r.total_pnl.toFixed(2)}</td>
              <td>$${r.max_drawdown.toFixed(2)}</td>
              <td>${isCurrentConfig(r, data.current) ? '<span class="mode-badge paper">current</span>' : (!r.sufficient_sample ? '<span style="color:#4a5a66;font-size:.68rem">low sample</span>' : '')}</td>
            </tr>`).join('')}</tbody>
        </table>
      </div>`;
  } catch(e) {
    el.innerHTML = '<span style="color:#ff1744;font-size:.78rem">Error loading results.</span>';
  }
}

poll();
setInterval(poll, 15000);
</script>
</body>
</html>
"""

if __name__ == "__main__":
    t = threading.Thread(target=background_scan, daemon=True)
    t.start()
    print("Dashboard → http://localhost:8080")
    app.run(host="127.0.0.1", port=8080, debug=False)
