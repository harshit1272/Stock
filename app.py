from __future__ import annotations

from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
import os
import json
from math import exp, sqrt
import random
from statistics import mean, pstdev
from typing import Any

SEED = 42
RNG = random.Random(SEED)


@dataclass
class StockInput:
    symbol: str
    sector: str
    revenue_growth: float
    eps_growth: float
    roe: float
    debt_to_equity: float
    pe: float
    growth_rate: float
    relative_strength_percentile: float
    roic: float
    wacc: float
    gross_margin_5y: list[float]
    cfo: float
    net_income: float
    sector_cagr_3y: float
    close: float
    dma50: float
    dma150: float
    dma200: float
    up_volume: float
    down_volume: float
    bb_width: float
    sentiment: float
    severity_rank: int
    win_rate: float
    win_loss_ratio: float
    atr: float
    stop_pct: float = 0.08
    target_pct: float = 0.16


@dataclass
class PipelineRequest:
    macro: dict[str, float]
    nse_universe: list[StockInput]
    capital: float = 1_000_000


@dataclass
class RegimeResult:
    regime: str
    tilt: str
    risk_off: bool


@dataclass
class Decision:
    symbol: str
    regime: str
    conviction: float
    strategy: str
    position_size: float
    entry: float
    stop: float
    target: float
    win_probability: float
    scores: dict[str, float]


def clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def phase1_market_regime(macro: dict[str, float]) -> RegimeResult:
    pmi = macro.get("pmi", 50)
    cpi = macro.get("cpi", 5)
    gdp = macro.get("gdp_growth", 6)
    spread = macro.get("yield_spread", 1)
    vix = macro.get("vix", 18)

    if gdp > 5 and pmi > 50 and cpi < 6 and spread > 0:
        regime = "Reflationary Expansion"
    elif gdp > 2 and cpi >= 6:
        regime = "Stagflation"
    elif gdp <= 2 and spread < 0:
        regime = "Recession"
    else:
        regime = "Crisis"

    risk_off = vix > 30
    return RegimeResult(regime=regime, tilt="Defensive" if risk_off else "Pro-Risk", risk_off=risk_off)


def phase2_sieve(stock: StockInput, settings: dict[str, float] | None = None) -> tuple[bool, dict[str, bool]]:
    settings = settings or {}
    rev_growth_min = settings.get("revenue_growth_min", 15)
    eps_growth_min = settings.get("eps_growth_min", 15)
    roe_min = settings.get("roe_min", 15)
    de_max = settings.get("de_max", 0.5)
    peg_max = settings.get("peg_max", 1.0)
    rs_min = settings.get("rs_min", 70)

    stage1 = (
        stock.revenue_growth >= rev_growth_min
        and stock.eps_growth >= eps_growth_min
        and stock.roe >= roe_min
        and stock.debt_to_equity < de_max
    )
    stage2 = (stock.pe / max(stock.growth_rate, 1e-6)) <= peg_max
    stage3 = stock.relative_strength_percentile >= rs_min
    return stage1 and stage2 and stage3, {"stage1": stage1, "stage2": stage2, "stage3": stage3}


def phase3_fundamental_score(stock: StockInput, nifty_cagr: float = 12.0) -> float:
    mgmt = clamp((stock.roic - stock.wacc) * 5)
    gm_mean = mean(stock.gross_margin_5y)
    gm_std = pstdev(stock.gross_margin_5y) if len(stock.gross_margin_5y) > 1 else 0
    moat = clamp((gm_mean * 2) - (gm_std * 10))
    earnings_quality = clamp((stock.cfo / max(stock.net_income, 1e-6)) * 60)
    sector_tailwind = clamp(50 + (stock.sector_cagr_3y - nifty_cagr) * 4)
    return round(0.3 * mgmt + 0.25 * moat + 0.25 * earnings_quality + 0.2 * sector_tailwind, 2)


def phase4_technical_score(stock: StockInput) -> float:
    trend = 100.0 if stock.close > stock.dma50 > stock.dma150 > stock.dma200 else 25.0
    ad_score = clamp((stock.up_volume / max(stock.down_volume, 1e-6)) * 40)
    compression = clamp(100 - stock.bb_width * 100)
    return round(0.5 * trend + 0.3 * ad_score + 0.2 * compression, 2)


def phase5_sentiment_adjustment(stock: StockInput) -> tuple[float, str | None]:
    if stock.severity_rank >= 5 and stock.sentiment < -30:
        return -40.0, "Sell/Hedge Override"
    return stock.sentiment * 0.2, None


def pick_strategy(regime: RegimeResult, conviction: float, stock: StockInput, override: str | None) -> str:
    if override:
        return override
    vol_ratio = stock.atr / max(stock.close, 1e-6)
    if conviction >= 80 and vol_ratio > 0.03:
        return "Bull Call Spread"
    if conviction >= 75 and regime.risk_off:
        return "Protective Put / Collar"
    if conviction >= 75:
        return "Directional Equity"
    return "Watchlist"


def phase6_conviction(regime: RegimeResult, fund: float, tech: float, stock: StockInput) -> tuple[float, str]:
    valuation = clamp(120 - (stock.pe / max(stock.growth_rate, 1e-6)) * 80)
    sentiment_adj, override = phase5_sentiment_adjustment(stock)
    conviction = clamp(0.4 * fund + 0.3 * valuation + 0.25 * tech + 0.05 * (50 + sentiment_adj))
    return conviction, pick_strategy(regime, conviction, stock, override)


def phase7_position_size(stock: StockInput, capital: float, risk_budget: float = 0.02, kelly_fraction: float = 0.5) -> float:
    p = stock.win_rate
    b = stock.win_loss_ratio
    kelly = max(0.0, (b * p - (1 - p)) / b)
    vol_ratio = stock.atr / max(stock.close, 1e-6)
    volatility_scale = max(0.25, min(1.0, 1 / max(vol_ratio, 1e-6)))
    return round(capital * risk_budget * (kelly * kelly_fraction) * volatility_scale, 2)


def phase8_monte_carlo(stock: StockInput, n_paths: int = 2000, horizon_days: int = 40) -> float:
    sigma = max(stock.atr / max(stock.close, 1e-6), 0.01)
    entry = stock.close
    target = entry * (1 + stock.target_pct)
    stop = entry * (1 - stock.stop_pct)
    wins = 0

    for _ in range(n_paths):
        price = entry
        for _ in range(horizon_days):
            price *= exp(RNG.gauss(0.0008, sigma))
            if price >= target:
                wins += 1
                break
            if price <= stop:
                break
    return round(wins / n_paths * 100, 2)


def run_pipeline(
    req: PipelineRequest,
    settings: dict[str, float] | None = None,
    selected_symbols: set[str] | None = None,
) -> list[Decision]:
    regime = phase1_market_regime(req.macro)
    output: list[Decision] = []

    for stock in req.nse_universe:
        if selected_symbols and stock.symbol not in selected_symbols:
            continue

        passed, _ = phase2_sieve(stock, settings=settings)
        if not passed:
            continue

        fund = phase3_fundamental_score(stock)
        tech = phase4_technical_score(stock)
        conviction, strategy = phase6_conviction(regime, fund, tech, stock)
        entry = stock.close

        output.append(
            Decision(
                symbol=stock.symbol,
                regime=regime.regime,
                conviction=round(conviction, 2),
                strategy=strategy,
                position_size=phase7_position_size(stock, req.capital),
                entry=round(entry, 2),
                stop=round(entry * (1 - stock.stop_pct), 2),
                target=round(entry * (1 + stock.target_pct), 2),
                win_probability=phase8_monte_carlo(stock),
                scores={
                    "fundamental": fund,
                    "technical": tech,
                    "sentiment": stock.sentiment,
                },
            )
        )

    return sorted(output, key=lambda item: item.conviction, reverse=True)


def parse_request(payload: dict[str, Any]) -> PipelineRequest:
    stocks = [StockInput(**item) for item in payload.get("nse_universe", [])]
    return PipelineRequest(
        macro=payload.get("macro", {}),
        nse_universe=stocks,
        capital=payload.get("capital", 1_000_000),
    )


def demo_payload() -> dict[str, Any]:
    return {
        "macro": {"pmi": 54, "cpi": 4.9, "gdp_growth": 6.8, "yield_spread": 0.8, "vix": 18},
        "capital": 1_000_000,
        "nse_universe": [
            {
                "symbol": "RELIANCE",
                "sector": "Energy",
                "revenue_growth": 18,
                "eps_growth": 20,
                "roe": 17,
                "debt_to_equity": 0.35,
                "pe": 22,
                "growth_rate": 24,
                "relative_strength_percentile": 86,
                "roic": 19,
                "wacc": 10,
                "gross_margin_5y": [42, 43, 44, 44, 45],
                "cfo": 95000,
                "net_income": 82000,
                "sector_cagr_3y": 15,
                "close": 3000,
                "dma50": 2920,
                "dma150": 2790,
                "dma200": 2700,
                "up_volume": 2400000,
                "down_volume": 1300000,
                "bb_width": 0.12,
                "sentiment": 28,
                "severity_rank": 2,
                "win_rate": 0.58,
                "win_loss_ratio": 1.7,
                "atr": 65,
            },
            {
                "symbol": "INFY",
                "sector": "IT",
                "revenue_growth": 17,
                "eps_growth": 18,
                "roe": 28,
                "debt_to_equity": 0.08,
                "pe": 27,
                "growth_rate": 30,
                "relative_strength_percentile": 79,
                "roic": 26,
                "wacc": 11,
                "gross_margin_5y": [35, 35.4, 36, 35.8, 36.3],
                "cfo": 39800,
                "net_income": 33400,
                "sector_cagr_3y": 17,
                "close": 1650,
                "dma50": 1610,
                "dma150": 1540,
                "dma200": 1510,
                "up_volume": 1500000,
                "down_volume": 780000,
                "bb_width": 0.09,
                "sentiment": 18,
                "severity_rank": 1,
                "win_rate": 0.55,
                "win_loss_ratio": 1.6,
                "atr": 32,
            },
            {
                "symbol": "SBIN",
                "sector": "Financials",
                "revenue_growth": 16,
                "eps_growth": 21,
                "roe": 16,
                "debt_to_equity": 0.42,
                "pe": 12,
                "growth_rate": 18,
                "relative_strength_percentile": 73,
                "roic": 15,
                "wacc": 9,
                "gross_margin_5y": [48, 49, 49, 50, 51],
                "cfo": 59000,
                "net_income": 52000,
                "sector_cagr_3y": 14,
                "close": 820,
                "dma50": 790,
                "dma150": 745,
                "dma200": 730,
                "up_volume": 4800000,
                "down_volume": 2900000,
                "bb_width": 0.1,
                "sentiment": 12,
                "severity_rank": 2,
                "win_rate": 0.54,
                "win_loss_ratio": 1.5,
                "atr": 18,
            },
        ],
    }


def demo_request() -> PipelineRequest:
    return parse_request(demo_payload())


def build_response(req: PipelineRequest, decisions: list[Decision]) -> dict[str, Any]:
    regime = phase1_market_regime(req.macro)
    vol = mean([stock.atr / max(stock.close, 1e-6) for stock in req.nse_universe]) if req.nse_universe else 0.02
    var_95 = round(req.capital * 1.65 * vol / sqrt(252), 2)
    drawdown = round(min(15.0, vol * 180), 2)
    return {
        "count": len(decisions),
        "macro_regime": regime.regime,
        "tilt": regime.tilt,
        "portfolio_var_95": var_95,
        "drawdown_pct": drawdown,
        "decisions": [asdict(d) for d in decisions],
    }


INDEX_HTML = """<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>NSE Holistic Portfolio OS</title>
<style>
:root{--bg:#050816;--panel:#0f172ab8;--line:#243247;--text:#e5edf9;--muted:#8ea0bb;--accent:#4f8cff;--ok:#22c55e;--warn:#f59e0b}
*{box-sizing:border-box} body{margin:0;font-family:Inter,system-ui;background:radial-gradient(circle at 10% 0,#1d4ed8 0,#050816 46%);color:var(--text)}
.app{max-width:1280px;margin:0 auto;padding:20px}
.topbar,.panel{background:var(--panel);border:1px solid var(--line);backdrop-filter:blur(10px);border-radius:16px}
.topbar{padding:16px;display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}
.brand h1{margin:0;font-size:1.35rem}.brand p{margin:4px 0 0;color:var(--muted)}
.actions{display:flex;gap:8px;flex-wrap:wrap}
button{border:0;border-radius:10px;padding:10px 12px;font-weight:700;color:#fff;cursor:pointer}
.primary{background:linear-gradient(90deg,#22d3ee,#2563eb)} .secondary{background:#1e293b;border:1px solid #32445d}
.layout{display:grid;grid-template-columns:290px 1fr;gap:14px;margin-top:14px}
.sidebar{padding:12px}.nav-btn{width:100%;text-align:left;margin-bottom:8px;background:#0b1226;border:1px solid var(--line);color:#c7d6ee}.nav-btn.active{background:#13264d;border-color:#3b82f6}
.main{display:grid;gap:12px}
.screen{display:none}.screen.active{display:block}
.grid{display:grid;gap:10px}.kpi-grid{grid-template-columns:repeat(auto-fit,minmax(180px,1fr))}
.kpi{padding:12px;border-radius:12px;border:1px solid var(--line);background:#0a1226}.kpi small{color:var(--muted);display:block}.kpi b{font-size:1.15rem}
.pill{display:inline-block;padding:4px 10px;border-radius:999px;background:#1d3152;color:#bfdbfe;font-size:.8rem}
.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:12px}
table{width:100%;border-collapse:collapse}th,td{padding:9px;border-bottom:1px solid #213148;text-align:left}th{font-size:.78rem;color:#93c5fd;text-transform:uppercase;letter-spacing:.05em}
.muted{color:var(--muted)}
.controls{display:grid;grid-template-columns:2fr 1fr;gap:10px}
.checklist{max-height:220px;overflow:auto;border:1px solid var(--line);padding:10px;border-radius:10px;background:#091024}
input,textarea{width:100%;padding:8px;border-radius:8px;border:1px solid var(--line);background:#0a1226;color:#dbeafe}
.timeline{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}.step{border:1px solid var(--line);padding:10px;border-radius:10px;background:#091024}
.badge-ok{color:#86efac}.badge-warn{color:#fcd34d}
@media (max-width:1000px){.layout{grid-template-columns:1fr}.controls{grid-template-columns:1fr}.timeline{grid-template-columns:1fr 1fr}}
</style>
</head>
<body>
<div class='app'>
  <header class='topbar'>
    <div class='brand'>
      <h1>NSE Holistic Portfolio OS</h1>
      <p>Macro-aware screening, conviction scoring, strategy selection, and execution dashboard in one interface.</p>
    </div>
    <div class='actions'>
      <button id='oneClick' class='primary'>⚡ One-Click Analyze</button>
      <button id='privacy' class='secondary'>🙈 Privacy ON</button>
    </div>
  </header>

  <div class='layout'>
    <aside class='panel sidebar'>
      <button class='nav-btn active' data-screen='overview'>Overview</button>
      <button class='nav-btn' data-screen='screener'>Screener</button>
      <button class='nav-btn' data-screen='strategy'>Strategy</button>
      <button class='nav-btn' data-screen='education'>Education</button>
      <hr style='border-color:#1f2d42'>
      <div class='muted'>Pipeline status</div>
      <div class='timeline' style='margin-top:8px'>
        <div class='step'>Phase 1<br><small class='muted'>Regime</small></div>
        <div class='step'>Phase 2-3<br><small class='muted'>Sieve+Fund</small></div>
        <div class='step'>Phase 4-6<br><small class='muted'>Tech+Conviction</small></div>
        <div class='step'>Phase 7-8<br><small class='muted'>Risk+MC</small></div>
      </div>
    </aside>

    <main class='main'>
      <section id='overview' class='panel screen active' style='padding:14px'>
        <div style='display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap'>
          <h2 style='margin:0'>Portfolio Summary & Risk</h2>
          <span class='pill' id='regimePill'>Regime: --</span>
        </div>
        <div class='grid kpi-grid' id='summaryKpis' style='margin-top:10px'></div>
        <p class='muted'>Privacy mode masks INR figures while preserving relative performance context.</p>
      </section>

      <section id='screener' class='panel screen' style='padding:14px'>
        <h2 style='margin-top:0'>Dynamic Screener & Settings</h2>
        <div class='controls'>
          <div>
            <label class='muted'>NSE Universe</label>
            <div id='universeList' class='checklist'></div>
          </div>
          <div class='grid'>
            <label>PEG Max <input id='pegMax' type='number' step='0.1' value='1.0'></label>
            <label>ROE Min <input id='roeMin' type='number' value='15'></label>
            <label>RS Min <input id='rsMin' type='number' value='70'></label>
            <label>Auto Analyze (minutes) <input id='intervalMin' type='number' min='1' value='3'></label>
            <button id='runCustom' class='primary'>Analyze Selected</button>
          </div>
        </div>
      </section>

      <section id='strategy' class='panel screen' style='padding:14px'>
        <h2 style='margin-top:0'>Strategy & Execution</h2>
        <div class='table-wrap'>
          <table>
            <thead><tr><th>Symbol</th><th>Conviction</th><th>Strategy</th><th>Win %</th><th>Position</th><th>Entry / Stop / Target</th><th>Action</th></tr></thead>
            <tbody id='strategyRows'><tr><td colspan='7' class='muted'>Run analysis to populate signals.</td></tr></tbody>
          </table>
        </div>
      </section>

      <section id='education' class='panel screen' style='padding:14px'>
        <h2 style='margin-top:0'>Educational Hub</h2>
        <div class='grid kpi-grid'>
          <div class='kpi'><small>Greeks</small><b>Delta • Theta • Vega</b><p class='muted'>Used to choose between directional and hedged option structures.</p></div>
          <div class='kpi'><small>Factor Logic</small><b>Quality + Value + Momentum</b><p class='muted'>Blends fundamentals, valuation, technicals, and sentiment overlays.</p></div>
          <div class='kpi'><small>Risk Discipline</small><b>Kelly + VaR + Drawdown</b><p class='muted'>Position sizing and exposure controls prevent capital concentration.</p></div>
          <div class='kpi'><small>Event Handling</small><b>Severity Override Matrix</b><p class='muted'>High-severity news can force hedge/sell regardless of technical trend.</p></div>
        </div>
      </section>
    </main>
  </div>
</div>

<script>
const nav=[...document.querySelectorAll('.nav-btn')];
const screens=[...document.querySelectorAll('.screen')];
const universeList=document.getElementById('universeList');
const demoPayload=__DEMO_PAYLOAD__;
let privacy=true, latest=null, timer=null;

nav.forEach(btn=>btn.addEventListener('click',()=>{
  nav.forEach(b=>b.classList.remove('active')); btn.classList.add('active');
  screens.forEach(s=>s.classList.remove('active'));
  document.getElementById(btn.dataset.screen).classList.add('active');
}));

function loadUniverse(){
  universeList.innerHTML=demoPayload.nse_universe.map(s=>`<label style='display:block;margin:4px 0'><input type='checkbox' class='stock' value='${s.symbol}' checked> ${s.symbol} <span class='muted'>${s.sector}</span></label>`).join('');
}
function selectedSymbols(){return [...document.querySelectorAll('.stock:checked')].map(n=>n.value);}
function money(v){return privacy ? '•••••' : `₹${Number(v).toLocaleString()}`;}

function render(result){
  latest=result;
  document.getElementById('regimePill').textContent=`Regime: ${result.macro_regime} • ${result.tilt}`;
  const top=result.decisions && result.decisions[0];
  const avg=(result.decisions||[]).length?((result.decisions.reduce((a,b)=>a+b.win_probability,0))/result.decisions.length).toFixed(2)+'%':'--';
  document.getElementById('summaryKpis').innerHTML=[
    ['Portfolio VaR (95%)', money(result.portfolio_var_95)],
    ['Current Drawdown', result.drawdown_pct + '%'],
    ['Candidates', result.count],
    ['Top Symbol', top?top.symbol:'--'],
    ['Top Conviction', top?top.conviction:'--'],
    ['Average Win Probability', avg]
  ].map(([k,v])=>`<div class='kpi'><small>${k}</small><b>${v}</b></div>`).join('');

  const rows=(result.decisions||[]).map(d=>`<tr>
    <td>${d.symbol}</td>
    <td>${d.conviction}</td>
    <td>${d.strategy}</td>
    <td>${d.win_probability}%</td>
    <td>${money(d.position_size)}</td>
    <td>${d.entry} / ${d.stop} / ${d.target}</td>
    <td><button class='secondary' onclick='tradeNow(${JSON.stringify(JSON.stringify(d))})'>Trade Now</button></td>
  </tr>`).join('');
  document.getElementById('strategyRows').innerHTML=rows || `<tr><td colspan='7' class='muted'>No shortlisted stocks for current settings.</td></tr>`;
}

function tradeNow(raw){
  const d=JSON.parse(raw);
  alert(`Dhan Order Draft
Symbol: ${d.symbol}
Strategy: ${d.strategy}
Position: ₹${d.position_size}
Entry: ${d.entry}
Stop: ${d.stop}
Target: ${d.target}`);
}

async function runAnalyze(payload){
  const res=await fetch('/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  render(await res.json());
}

document.getElementById('oneClick').addEventListener('click', async()=>{
  const res=await fetch('/demo');
  render(await res.json());
});

document.getElementById('privacy').addEventListener('click',()=>{
  privacy=!privacy;
  document.getElementById('privacy').textContent=privacy?'🙈 Privacy ON':'👁 Privacy OFF';
  if(latest) render(latest);
});

document.getElementById('runCustom').addEventListener('click',()=>{
  const payload=JSON.parse(JSON.stringify(demoPayload));
  payload.settings={
    peg_max:Number(document.getElementById('pegMax').value),
    roe_min:Number(document.getElementById('roeMin').value),
    rs_min:Number(document.getElementById('rsMin').value)
  };
  payload.selected_symbols=selectedSymbols();
  runAnalyze(payload);
  if(timer) clearInterval(timer);
  timer=setInterval(()=>runAnalyze(payload), Math.max(1,Number(document.getElementById('intervalMin').value))*60000);
});

loadUniverse();
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, data: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_HEAD(self) -> None:
        if self.path in {"/", "/demo", "/health"}:
            self.send_response(200)
            self.end_headers()
            return
        self.send_error(404)

    def do_GET(self) -> None:
        if self.path == "/":
            html = INDEX_HTML.replace("__DEMO_PAYLOAD__", json.dumps(demo_payload()))
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/demo":
            req = demo_request()
            decisions = run_pipeline(req)
            self._send_json(build_response(req, decisions))
            return
        if self.path == "/health":
            self._send_json({"status": "ok"})
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if self.path != "/analyze":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode() or "{}")
        req = parse_request(payload)
        settings = payload.get("settings") if isinstance(payload.get("settings"), dict) else None
        selected = set(payload.get("selected_symbols", [])) if isinstance(payload.get("selected_symbols"), list) else None
        decisions = run_pipeline(req, settings=settings, selected_symbols=selected)
        self._send_json(build_response(req, decisions))


def serve(host: str = "0.0.0.0", port: int = 8000) -> None:
    try:
        server = HTTPServer((host, port), Handler)
    except OSError as exc:
        raise SystemExit(f"Failed to start server on {host}:{port} ({exc}). Try a different PORT.") from exc
    print(f"Serving on http://{host}:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    serve(port=int(os.getenv("PORT", "8000")))
