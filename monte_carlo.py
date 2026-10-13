"""
monte_carlo.py  –  Bootstrap Monte Carlo on backtest trade outcomes.
Loads backtest_results.csv (or runs the backtest fresh) and resamples the
trade P&Ls N_SIMS times to show the distribution of possible equity curves.

Run:  .venv/bin/python3 monte_carlo.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from strategy_params import PARAMS

_S = PARAMS["simulation"]

# ── Config ───────────────────────────���──────────────────────────────────────
N_SIMS           = _S["n_sims"]    # bootstrap iterations
STARTING_CAPITAL = _S["starting_capital"]
CONTRACTS        = _S["contracts"]
MIN_SCORE        = _S["min_score"]
RESULTS_CSV      = Path(__file__).parent / "backtest_results.csv"
HTML_OUT         = Path(__file__).parent / "monte_carlo.html"


# ── Helpers ──────��─────────────────────────────────────────────���─────────────
def _pct(arr: np.ndarray, p: float) -> float:
    return float(np.percentile(arr, p))


def _max_dd_vec(paths: np.ndarray) -> np.ndarray:
    """Vectorised max drawdown for every row in a (N, T) paths array."""
    peak = np.maximum.accumulate(paths, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak > 0, (peak - paths) / peak, 0.0)
    return dd.max(axis=1)


def _max_dd(equity: np.ndarray) -> float:
    return float(_max_dd_vec(equity[np.newaxis, :])[0])


# ── Data ──��───────────────────────────���──────────────────────────────────────
def load_trades() -> pd.DataFrame:
    if RESULTS_CSV.exists():
        df = pd.read_csv(RESULTS_CSV)
        print(f"  Loaded {len(df)} trades from backtest_results.csv")
    else:
        print("  No CSV found – running backtest first …")
        from backtest import run_backtest
        df = run_backtest()
    if MIN_SCORE > 0:
        df = df[df["score"] >= MIN_SCORE]
        print(f"  Filtered to score ≥ {MIN_SCORE}: {len(df)} trades remain")
    return df


# ── Full bootstrap MC ──────────────��─────────────────────────��────────────────
def run_mc(pnls: np.ndarray) -> dict:
    n = len(pnls)
    print(f"  Running {N_SIMS:,} simulations ({n} trades each) …\n")

    rng     = np.random.default_rng(42)
    samples = pnls[rng.integers(0, n, size=(N_SIMS, n))] * CONTRACTS

    paths        = np.empty((N_SIMS, n + 1))
    paths[:, 0]  = STARTING_CAPITAL
    paths[:, 1:] = STARTING_CAPITAL + np.cumsum(samples, axis=1)

    final_eq  = paths[:, -1]
    max_dds   = _max_dd_vec(paths)
    win_rates = (samples > 0).mean(axis=1) * 100

    step  = max(1, n // 150)
    x_idx = sorted(set(range(0, n + 1, step)) | {0, n})

    pct_curves = {str(p): np.percentile(paths[:, x_idx], p, axis=0).tolist()
                  for p in (5, 25, 50, 75, 95)}

    vi  = rng.choice(N_SIMS, size=min(200, N_SIMS), replace=False)
    viz = paths[vi][:, x_idx].tolist()

    actual_full = np.concatenate([[STARTING_CAPITAL],
                                   STARTING_CAPITAL + np.cumsum(pnls * CONTRACTS)])
    actual_path = actual_full[x_idx].tolist()

    return {
        "n": n, "n_sims": N_SIMS, "starting": STARTING_CAPITAL,
        "x_idx":       x_idx,
        "pct_curves":  pct_curves,
        "viz":         viz,
        "actual_path": actual_path,
        "final_eq":    final_eq,
        "max_dds":     max_dds,
        "win_rates":   win_rates,
        "actual":      float(actual_full[-1]),
        "pnls":        pnls,
        "label":       "All signals",
        "x_label":     "Trade #",
    }


# ── Capital sweep ───────��───────────────────────────────���─────────────────────
def capital_sweep(pnls: np.ndarray) -> int:
    """
    Sweep starting capital from $1k to $20k in $500 steps.
    Find the minimum capital where P50 max drawdown ≤ 20%.
    Returns that capital level.
    """
    n   = len(pnls)
    rng = np.random.default_rng(42)
    # Run once, reuse paths for all capital levels
    samples  = pnls[rng.integers(0, n, size=(2_000, n))] * CONTRACTS
    cum_pnl  = np.cumsum(samples, axis=1)           # (2000, n) — dollar gains/losses

    sep = "─" * 62
    print(f"\n  Capital Sweep  —  target: P50 max drawdown ≤ 20%")
    print(f"  {sep}")
    print(f"  {'Capital':>10}  {'P50 MaxDD':>10}  {'P5 Final':>10}  "
          f"{'P50 Final':>10}  {'PProfit':>8}")
    print(f"  {sep}")

    floor = None
    for capital in range(1_000, 20_001, 500):
        paths     = np.empty((2_000, n + 1))
        paths[:, 0]  = capital
        paths[:, 1:] = capital + cum_pnl

        max_dds   = _max_dd_vec(paths)
        final_eq  = paths[:, -1]
        p50_dd    = _pct(max_dds, 50) * 100
        p5_final  = _pct(final_eq, 5)
        p50_final = _pct(final_eq, 50)
        prob      = (final_eq > capital).mean() * 100

        marker = "  ◄ FLOOR" if (p50_dd <= 20 and floor is None) else ""
        print(f"  ${capital:>9,}  {p50_dd:>9.1f}%  ${p5_final:>9,.0f}  "
              f"${p50_final:>9,.0f}  {prob:>7.1f}%{marker}")

        if p50_dd <= 20 and floor is None:
            floor = capital

    print(f"  {sep}")
    if floor:
        print(f"\n  Minimum capital for P50 max DD ≤ 20%: ${floor:,}\n")
    return floor or 20_000


# ── Daily-cap MC ──────────────���───────────────────────────────────────────────
def run_mc_daily_cap(df: pd.DataFrame, cap: int, starting: float) -> dict:
    """
    Monte Carlo where each simulation randomly selects at most `cap` trades
    per trading day (without replacement within each day).
    Equity steps are per day, not per trade.
    """
    # Sort by date then entry_time so "first N of the day" is chronological
    df2 = df.copy()
    df2["date"] = df2["date"].astype(str)
    if "entry_time" in df2.columns:
        df2 = df2.sort_values(["date", "entry_time"]).reset_index(drop=True)
    else:
        df2 = df2.sort_values("date").reset_index(drop=True)

    daily = df2.groupby("date")["pnl_$"].apply(np.array).to_dict()
    days  = sorted(daily.keys())
    n_days = len(days)

    print(f"  Running {N_SIMS:,} daily-cap sims "
          f"(max {cap} trades/day × {n_days} days) …\n")

    rng = np.random.default_rng(42)

    # Build (N_SIMS, n_days+1) equity paths
    paths = np.empty((N_SIMS, n_days + 1))
    paths[:, 0] = starting

    for d, day in enumerate(days):
        day_pnls = daily[day] * CONTRACTS
        k        = min(cap, len(day_pnls))
        # For each sim: pick k random indices from this day's trades
        if k == len(day_pnls):
            daily_gain = np.full(N_SIMS, day_pnls.sum())
        else:
            # Sample k trades per sim — vectorised via sorting random scores
            scores   = rng.random((N_SIMS, len(day_pnls)))
            top_k    = np.argsort(scores, axis=1)[:, :k]
            daily_gain = day_pnls[top_k].sum(axis=1)

        paths[:, d + 1] = paths[:, d] + daily_gain

    final_eq  = paths[:, -1]
    max_dds   = _max_dd_vec(paths)
    win_rates = np.array([])   # not meaningful per-day

    # Actual path: take first `cap` trades each day chronologically
    actual_eq = starting
    actual_path = [actual_eq]
    for day in days:
        day_pnls = daily[day] * CONTRACTS
        actual_eq += day_pnls[:cap].sum()
        actual_path.append(actual_eq)

    x_idx = list(range(n_days + 1))
    pct_curves = {str(p): np.percentile(paths, p, axis=0).tolist()
                  for p in (5, 25, 50, 75, 95)}

    vi  = rng.choice(N_SIMS, size=min(200, N_SIMS), replace=False)
    viz = paths[vi].tolist()

    total_trades = sum(min(cap, len(daily[d])) for d in days)

    return {
        "n":           n_days,
        "n_sims":      N_SIMS,
        "starting":    starting,
        "x_idx":       x_idx,
        "pct_curves":  pct_curves,
        "viz":         viz,
        "actual_path": actual_path,
        "final_eq":    final_eq,
        "max_dds":     max_dds,
        "win_rates":   win_rates,
        "actual":      float(actual_path[-1]),
        "pnls":        np.array([]),
        "label":       f"Max {cap} trades/day  ({total_trades} total)",
        "x_label":     "Trading Day",
    }


# ── Terminal output ─────────────────��─────────────────────────────────────────
def print_results(st: dict, starting: float = None) -> None:
    final   = st["final_eq"]
    dds     = st["max_dds"] * 100
    capital = starting or st["starting"]
    sep     = "═" * 62

    prob_profit = (final > capital).mean() * 100
    prob_double = (final >= capital * 2).mean() * 100
    prob_50loss = (final <= capital * 0.5).mean() * 100
    prob_ruin   = (final <= capital * 0.1).mean() * 100

    label = st.get("label", "")
    print(f"\n{sep}")
    print(f"  {label}  ·  {st['n_sims']:,} sims  ·  ${capital:,.0f} start")
    print(sep)
    print(f"  Actual result             ${st['actual']:>+10,.2f}")
    print(f"  Median outcome   (P50)    ${_pct(final, 50):>+10,.2f}")
    print(f"  Mean expected value       ${float(final.mean()):>+10,.2f}")
    print(f"  Best case        (P95)    ${_pct(final, 95):>+10,.2f}")
    print(f"  Worst case       (P5)     ${_pct(final, 5):>+10,.2f}")
    print()
    print(f"  Probability of profit     {prob_profit:>6.1f}%")
    print(f"  Probability of 2× gain    {prob_double:>6.1f}%")
    print(f"  Probability of 50% loss   {prob_50loss:>6.1f}%")
    print(f"  Probability of ruin       {prob_ruin:>6.1f}%")
    print()
    print(f"  Max Drawdown              P5      P50     P95")
    print(f"  {'─' * 44}")
    print(f"    Drawdown (%)       {_pct(dds,5):>6.1f}%  {_pct(dds,50):>6.1f}%  {_pct(dds,95):>6.1f}%")
    print()
    print(f"  Final Equity Percentiles")
    print(f"  {'─' * 44}")
    for p in (5, 10, 25, 50, 75, 90, 95):
        v = _pct(final, p)
        print(f"    P{p:<5}  ${v:>10,.2f}   ({(v/capital-1)*100:>+.1f}%)")
    print(f"\n{sep}\n")


# ── HTML generation ─────────────────────────────────────────���─────────────────
_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Monte Carlo Backtest</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#f1f5f9;--card:#ffffff;--border:rgba(0,0,0,.08);
  --tx:#1e293b;--tx2:#64748b;--accent:#4f46e5;--green:#16a34a;--red:#dc2626;
}
@media(prefers-color-scheme:dark){:root{
  --bg:#0d1117;--card:#161b22;--border:rgba(255,255,255,.07);
  --tx:#e6edf3;--tx2:#8b949e;--accent:#818cf8;--green:#4ade80;--red:#f87171;
}}
:root[data-theme=light]{
  --bg:#f1f5f9;--card:#ffffff;--border:rgba(0,0,0,.08);
  --tx:#1e293b;--tx2:#64748b;--accent:#4f46e5;--green:#16a34a;--red:#dc2626;
}
:root[data-theme=dark]{
  --bg:#0d1117;--card:#161b22;--border:rgba(255,255,255,.07);
  --tx:#e6edf3;--tx2:#8b949e;--accent:#818cf8;--green:#4ade80;--red:#f87171;
}
body{background:var(--bg);color:var(--tx);font-family:system-ui,-apple-system,sans-serif;
     padding:24px;max-width:1200px;margin:0 auto}
h1{font-size:1.35rem;font-weight:700;margin-bottom:4px}
.sub{color:var(--tx2);font-size:.82rem;margin-bottom:22px}
.tabs{display:flex;gap:8px;margin-bottom:16px;flex-wrap:wrap}
.tab{padding:6px 16px;border-radius:6px;border:1px solid var(--border);background:var(--card);
     cursor:pointer;font-size:.82rem;color:var(--tx2);transition:all .15s}
.tab.active{background:var(--accent);color:#fff;border-color:var(--accent)}
.panel{display:none}.panel.active{display:block}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:16px}
.tile{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:14px 16px}
.tl{font-size:.68rem;color:var(--tx2);text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px}
.tv{font-size:1.4rem;font-weight:700;line-height:1.1}
.ts{font-size:.72rem;color:var(--tx2);margin-top:4px}
.g{color:var(--green)}.r{color:var(--red)}.a{color:var(--accent)}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:20px;margin-bottom:14px}
.ct{font-size:.72rem;font-weight:600;color:var(--tx2);text-transform:uppercase;
    letter-spacing:.07em;margin-bottom:14px}
.row{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:600px){.row{grid-template-columns:1fr}}
canvas{display:block;width:100%}
.leg{display:flex;gap:14px;flex-wrap:wrap;margin-top:10px}
.li{display:flex;align-items:center;gap:5px;font-size:.72rem;color:var(--tx2)}
.ld{width:18px;height:3px;border-radius:2px}
.lb{width:10px;height:10px;border-radius:2px}
</style>
</head>
<body>
<h1>Monte Carlo Backtest</h1>
<div class="sub" id="sub"></div>
<div class="tabs" id="tabs"></div>
<div id="panels"></div>

<script>
const DATASETS=%%DATA%%;
let active=0;

function isDark(){
  const t=document.documentElement.getAttribute('data-theme');
  if(t==='dark')return true;if(t==='light')return false;
  return window.matchMedia('(prefers-color-scheme:dark)').matches;
}
function clr(){
  const d=isDark();
  return d?{
    muted:'#8b949e',grid:'rgba(255,255,255,.045)',axis:'rgba(255,255,255,.12)',
    b95:'rgba(99,102,241,.11)',b75:'rgba(99,102,241,.24)',
    p50:'#a5b4fc',spaghetti:'rgba(99,102,241,.045)',
    actual:'#4ade80',startLine:'rgba(255,255,255,.15)',
    winBar:'#22c55e',lossBar:'#ef4444'
  }:{
    muted:'#64748b',grid:'rgba(0,0,0,.045)',axis:'rgba(0,0,0,.15)',
    b95:'rgba(99,102,241,.07)',b75:'rgba(99,102,241,.16)',
    p50:'#4f46e5',spaghetti:'rgba(99,102,241,.035)',
    actual:'#16a34a',startLine:'rgba(0,0,0,.18)',
    winBar:'#16a34a',lossBar:'#dc2626'
  };
}

function prep(id,H){
  const el=document.getElementById(id);if(!el)return null;
  const r=window.devicePixelRatio||1;
  const w=el.parentElement.clientWidth-40;
  el.width=w*r;el.height=H*r;el.style.width=w+'px';el.style.height=H+'px';
  const ctx=el.getContext('2d');ctx.scale(r,r);
  return{ctx,W:w,H};
}

function fmtK(v){
  const a=Math.abs(v),s=v<0?'-':'';
  if(a>=1e6)return s+'$'+(a/1e6).toFixed(1)+'M';
  if(a>=1e3)return s+'$'+(a/1e3).toFixed(0)+'k';
  return s+'$'+a.toFixed(0);
}
function niceTicks(lo,hi){
  const rng=hi-lo;const mag=Math.pow(10,Math.floor(Math.log10(rng)));
  const norm=rng/mag;const step=norm<2?mag/5:norm<5?mag/2:mag;
  const first=Math.floor(lo/step)*step;const tks=[];
  for(let v=first;v<=hi+step*.01;v+=step)tks.push(+v.toFixed(10));
  return tks;
}

function drawEq(D, pfx){
  const r=prep(pfx+'-eq',320);if(!r)return;
  const{ctx,W,H}=r;const c=clr();
  const pad={t:16,r:20,b:38,l:68};
  const cW=W-pad.l-pad.r,cH=H-pad.t-pad.b;

  const p5=D.pct_curves["5"],p95=D.pct_curves["95"];
  const allY=[...p5,...p95,...D.actual_path];
  const rawMin=allY.reduce((m,v)=>Math.min(m,v),Infinity);
  const rawMax=allY.reduce((m,v)=>Math.max(m,v),-Infinity);
  const py=(rawMax-rawMin)*.06;
  const yLo=rawMin-py,yHi=rawMax+py;
  const nx=D.x_idx.length;
  const xs=i=>pad.l+(i/(nx-1))*cW;
  const ys=v=>pad.t+cH*(1-(v-yLo)/(yHi-yLo));

  ctx.font='11px system-ui';ctx.textAlign='right';
  niceTicks(yLo,yHi).forEach(t=>{
    const y=ys(t);if(y<pad.t-4||y>pad.t+cH+4)return;
    ctx.strokeStyle=c.grid;ctx.lineWidth=1;
    ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(pad.l+cW,y);ctx.stroke();
    ctx.fillStyle=c.muted;ctx.fillText(fmtK(t),pad.l-5,y+4);
  });

  ctx.textAlign='center';ctx.fillStyle=c.muted;
  [0,.25,.5,.75,1].forEach(f=>{
    const trN=Math.round(f*D.n);
    const idx=D.x_idx.reduce((b,v,i)=>Math.abs(v-trN)<Math.abs(D.x_idx[b]-trN)?i:b,0);
    const x=xs(idx);
    ctx.strokeStyle=c.grid;ctx.lineWidth=1;
    ctx.beginPath();ctx.moveTo(x,pad.t);ctx.lineTo(x,pad.t+cH);ctx.stroke();
    ctx.fillText((D.x_label||'T')+trN,x,pad.t+cH+16);
  });

  ctx.strokeStyle=c.axis;ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(pad.l,pad.t);ctx.lineTo(pad.l,pad.t+cH);
  ctx.lineTo(pad.l+cW,pad.t+cH);ctx.stroke();

  const yS=ys(D.starting);
  ctx.strokeStyle=c.startLine;ctx.lineWidth=1;ctx.setLineDash([4,5]);
  ctx.beginPath();ctx.moveTo(pad.l,yS);ctx.lineTo(pad.l+cW,yS);ctx.stroke();
  ctx.setLineDash([]);

  D.viz.forEach(path=>{
    ctx.beginPath();ctx.strokeStyle=c.spaghetti;ctx.lineWidth=0.7;
    path.forEach((v,i)=>{i===0?ctx.moveTo(xs(i),ys(v)):ctx.lineTo(xs(i),ys(v));});
    ctx.stroke();
  });

  const drawFill=(a,b,col)=>{
    ctx.beginPath();
    a.forEach((v,i)=>{i===0?ctx.moveTo(xs(i),ys(v)):ctx.lineTo(xs(i),ys(v));});
    for(let i=b.length-1;i>=0;i--)ctx.lineTo(xs(i),ys(b[i]));
    ctx.closePath();ctx.fillStyle=col;ctx.fill();
  };
  drawFill(p5,p95,c.b95);
  drawFill(D.pct_curves["25"],D.pct_curves["75"],c.b75);

  const p50=D.pct_curves["50"];
  ctx.beginPath();ctx.strokeStyle=c.p50;ctx.lineWidth=2;
  p50.forEach((v,i)=>{i===0?ctx.moveTo(xs(i),ys(v)):ctx.lineTo(xs(i),ys(v));});
  ctx.stroke();

  ctx.beginPath();ctx.strokeStyle=c.actual;ctx.lineWidth=2.5;
  D.actual_path.forEach((v,i)=>{i===0?ctx.moveTo(xs(i),ys(v)):ctx.lineTo(xs(i),ys(v));});
  ctx.stroke();

  const dl=isDark();
  const leg=document.getElementById(pfx+'-leg');if(leg){leg.innerHTML='';
  [['200 paths',dl?'rgba(99,102,241,.45)':'rgba(99,102,241,.35)','ld'],
   ['P5–P95',   dl?'rgba(99,102,241,.45)':'rgba(99,102,241,.3)','lb'],
   ['P25–P75',  dl?'rgba(99,102,241,.75)':'rgba(99,102,241,.55)','lb'],
   ['Median',   c.p50,'ld'],['Actual',c.actual,'ld']
  ].forEach(([t,col,cls])=>{
    leg.insertAdjacentHTML('beforeend',
      `<div class="li"><div class="${cls}" style="background:${col}"></div>${t}</div>`);
  });}
}

function drawHist(id,values,nBins,fmtX,threshold,higherBetter){
  const r=prep(id,190);if(!r)return;
  const{ctx,W,H}=r;const c=clr();
  const pad={t:10,r:16,b:36,l:52};
  const cW=W-pad.l-pad.r,cH=H-pad.t-pad.b;
  const vMin=values.reduce((m,v)=>Math.min(m,v),Infinity);
  const vMax=values.reduce((m,v)=>Math.max(m,v),-Infinity);
  const bw=(vMax-vMin)/nBins;
  const bins=Array(nBins).fill(0);
  values.forEach(v=>{const b=Math.min(Math.floor((v-vMin)/bw),nBins-1);bins[b]++;});
  const mx=bins.reduce((m,v)=>Math.max(m,v),0);
  const xs=(b,f=0)=>pad.l+((b+f)/nBins)*cW;
  const ys=cnt=>pad.t+cH*(1-cnt/mx);
  [.25,.5,.75,1].forEach(f=>{
    const y=pad.t+cH*(1-f);ctx.strokeStyle=c.grid;ctx.lineWidth=1;
    ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(pad.l+cW,y);ctx.stroke();
  });
  bins.forEach((cnt,b)=>{
    if(cnt===0)return;
    const mid=vMin+(b+.5)*bw;
    const good=higherBetter?mid>=threshold:mid<=threshold;
    ctx.fillStyle=(good?c.winBar:c.lossBar)+'cc';
    ctx.fillRect(xs(b),ys(cnt),cW/nBins-1,pad.t+cH-ys(cnt));
  });
  if(threshold!==null){
    const tx=pad.l+(threshold-vMin)/(vMax-vMin)*cW;
    if(tx>pad.l&&tx<pad.l+cW){
      ctx.strokeStyle=c.startLine;ctx.lineWidth=1.5;ctx.setLineDash([3,3]);
      ctx.beginPath();ctx.moveTo(tx,pad.t);ctx.lineTo(tx,pad.t+cH);ctx.stroke();
      ctx.setLineDash([]);
    }
  }
  ctx.strokeStyle=c.axis;ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(pad.l,pad.t);ctx.lineTo(pad.l,pad.t+cH);
  ctx.lineTo(pad.l+cW,pad.t+cH);ctx.stroke();
  ctx.fillStyle=c.muted;ctx.font='10px system-ui';ctx.textAlign='center';
  for(let i=0;i<=4;i++){
    const v=vMin+(vMax-vMin)*i/4;
    ctx.fillText(fmtX(v),pad.l+(i/4)*cW,pad.t+cH+16);
  }
}

function buildTiles(D, pfx){
  const feq=D.final_eq,dds=D.max_dds;
  const pct=(arr,p)=>{const s=[...arr].sort((a,b)=>a-b);return s[Math.floor(s.length*p/100)];};
  const probP=feq.filter(v=>v>D.starting).length/feq.length*100;
  const probD=feq.filter(v=>v>=D.starting*2).length/feq.length*100;
  const medEq=pct(feq,50);const medDD=pct(dds,50)*100;const p95DD=pct(dds,95)*100;
  const tiles=[
    {l:'Actual Result',  v:fmtK(D.actual),      s:((D.actual/D.starting-1)*100).toFixed(1)+'%',c:D.actual>=D.starting?'g':'r'},
    {l:'Prob of Profit', v:probP.toFixed(1)+'%', s:D.n_sims.toLocaleString()+' sims',c:probP>=50?'g':'r'},
    {l:'Median Outcome', v:fmtK(medEq),          s:((medEq/D.starting-1)*100).toFixed(1)+'%',c:medEq>=D.starting?'g':'r'},
    {l:'Median Max DD',  v:medDD.toFixed(1)+'%', s:'typical drawdown',c:medDD<=20?'g':medDD<=40?'a':'r'},
    {l:'Best Case P95',  v:fmtK(pct(feq,95)),    s:'top 5%',c:'a'},
    {l:'Worst Case P5',  v:fmtK(pct(feq,5)),     s:'bottom 5%',c:pct(feq,5)>=D.starting?'g':'r'},
    {l:'Prob 2× Gain',   v:probD.toFixed(1)+'%', s:'doubling account',c:probD>=20?'g':probD>=5?'a':'r'},
    {l:'P95 Max DD',     v:p95DD.toFixed(1)+'%', s:'worst-case DD',c:p95DD<=30?'g':p95DD<=60?'a':'r'},
  ];
  const grid=document.getElementById(pfx+'-tiles');if(!grid)return;
  grid.innerHTML='';
  tiles.forEach(t=>{
    const d=document.createElement('div');d.className='tile';
    d.innerHTML=`<div class="tl">${t.l}</div><div class="tv ${t.c}">${t.v}</div><div class="ts">${t.s}</div>`;
    grid.appendChild(d);
  });
}

function renderPanel(idx){
  const D=DATASETS[idx];const pfx='p'+idx;
  buildTiles(D,pfx);drawEq(D,pfx);
  drawHist(pfx+'-eh',D.final_eq,40,fmtK,D.starting,true);
  drawHist(pfx+'-dh',D.max_dds.map(v=>v*100),40,v=>v.toFixed(0)+'%',20,false);
}

function buildUI(){
  document.getElementById('sub').textContent=
    `${DATASETS[0].n_sims.toLocaleString()} bootstrap simulations · $${DATASETS[0].starting.toLocaleString()} starting capital`;
  const tabsEl=document.getElementById('tabs');
  const panelsEl=document.getElementById('panels');
  DATASETS.forEach((D,i)=>{
    const tab=document.createElement('button');
    tab.className='tab'+(i===0?' active':'');
    tab.textContent=D.label;
    tab.onclick=()=>{
      document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
      document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
      tab.classList.add('active');
      document.getElementById('panel-'+i).classList.add('active');
      active=i;renderPanel(i);
    };
    tabsEl.appendChild(tab);

    const panel=document.createElement('div');
    panel.className='panel'+(i===0?' active':'');
    panel.id='panel-'+i;
    panel.innerHTML=`
      <div class="tiles" id="p${i}-tiles"></div>
      <div class="card">
        <div class="ct">Equity Curves — ${D.label}</div>
        <canvas id="p${i}-eq" height="320"></canvas>
        <div class="leg" id="p${i}-leg"></div>
      </div>
      <div class="row">
        <div class="card"><div class="ct">Final Equity Distribution</div><canvas id="p${i}-eh" height="190"></canvas></div>
        <div class="card"><div class="ct">Max Drawdown Distribution</div><canvas id="p${i}-dh" height="190"></canvas></div>
      </div>`;
    panelsEl.appendChild(panel);
  });
  renderPanel(0);
}

new MutationObserver(()=>renderPanel(active))
  .observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
window.matchMedia('(prefers-color-scheme:dark)').addEventListener('change',()=>renderPanel(active));
window.addEventListener('resize',()=>renderPanel(active));
window.addEventListener('load',buildUI);
</script>
</body>
</html>"""


def generate_html(datasets: list[dict]) -> None:
    serialisable = []
    for st in datasets:
        serialisable.append({
            "n":           int(st["n"]),
            "n_sims":      st["n_sims"],
            "starting":    st["starting"],
            "x_idx":       st["x_idx"],
            "pct_curves":  st["pct_curves"],
            "viz":         st["viz"],
            "actual_path": st["actual_path"],
            "final_eq":    st["final_eq"].tolist(),
            "max_dds":     st["max_dds"].tolist(),
            "actual":      st["actual"],
            "label":       st.get("label", ""),
            "x_label":     st.get("x_label", "T"),
        })
    html = _HTML.replace("%%DATA%%", json.dumps(serialisable))
    HTML_OUT.write_text(html, encoding="utf-8")
    print(f"  Chart saved → {HTML_OUT.name}")


# ── Main ───────��─────────────────────────────────���───────────────────────────
if __name__ == "__main__":
    print(f"\nMonte Carlo Backtest")
    print("─" * 40)

    df = load_trades()
    if df.empty:
        print("No trades to simulate.")
        raise SystemExit(1)

    df = df.sort_values("date").reset_index(drop=True)
    pnls = df["pnl_$"].to_numpy(dtype=float)

    # 1 — Capital sweep
    floor = capital_sweep(pnls)

    # 2 — Full MC at $1k (baseline)
    print("── Full signal MC  ($1,000 start) ──────────────────────")
    st_full = run_mc(pnls)
    print_results(st_full)

    # 3 — Daily-cap MC: 3 and 4 trades/day
    results = [st_full]
    for cap in (3, 4):
        print(f"── Daily cap: {cap} trades/day ─────────────────────────────")
        st = run_mc_daily_cap(df, cap=cap, starting=STARTING_CAPITAL)
        print_results(st)
        results.append(st)

    # 4 — Re-run full MC at the capital floor too
    if floor != STARTING_CAPITAL:
        print(f"── Full signal MC  (${floor:,} floor capital) ───────────────")
        st_floor = run_mc(pnls)
        st_floor["starting"] = floor
        # rebuild paths at floor capital
        rng     = np.random.default_rng(42)
        n       = len(pnls)
        samples = pnls[rng.integers(0, n, size=(N_SIMS, n))] * CONTRACTS
        paths   = np.empty((N_SIMS, n + 1))
        paths[:, 0]  = floor
        paths[:, 1:] = floor + np.cumsum(samples, axis=1)
        st_floor["final_eq"]    = paths[:, -1]
        st_floor["max_dds"]     = _max_dd_vec(paths)
        st_floor["win_rates"]   = (samples > 0).mean(axis=1) * 100
        st_floor["actual"]      = float(floor + pnls.sum() * CONTRACTS)
        step  = max(1, n // 150)
        x_idx = sorted(set(range(0, n + 1, step)) | {0, n})
        st_floor["pct_curves"]  = {str(p): np.percentile(paths[:, x_idx], p, axis=0).tolist()
                                   for p in (5, 25, 50, 75, 95)}
        vi = rng.choice(N_SIMS, size=min(200, N_SIMS), replace=False)
        st_floor["viz"]         = paths[vi][:, x_idx].tolist()
        actual_full = np.concatenate([[floor], floor + np.cumsum(pnls * CONTRACTS)])
        st_floor["actual_path"] = actual_full[x_idx].tolist()
        st_floor["label"]       = f"All signals  (${floor:,} start)"
        st_floor["x_label"]     = "Trade #"
        print_results(st_floor, starting=floor)
        results.append(st_floor)

    generate_html(results)
