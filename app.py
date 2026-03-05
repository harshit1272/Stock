from __future__ import annotations

from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from math import exp
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
        regime = "Expansion"
    elif gdp > 2 and cpi >= 6:
        regime = "Stagflation"
    elif gdp <= 2 and spread < 0:
        regime = "Recession"
    else:
        regime = "Crisis"

    risk_off = vix > 30
    return RegimeResult(regime=regime, tilt="Defensive" if risk_off else "Pro-Risk", risk_off=risk_off)


def phase2_sieve(stock: StockInput) -> tuple[bool, dict[str, bool]]:
    stage1 = (
        stock.revenue_growth >= 15
        and stock.eps_growth >= 15
        and stock.roe >= 15
        and stock.debt_to_equity < 0.5
    )
    stage2 = (stock.pe / max(stock.growth_rate, 1e-6)) <= 1
    stage3 = stock.relative_strength_percentile >= 70
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


def phase6_conviction(regime: RegimeResult, fund: float, tech: float, stock: StockInput) -> tuple[float, str]:
    valuation = clamp(120 - (stock.pe / max(stock.growth_rate, 1e-6)) * 80)
    sentiment_adj, override = phase5_sentiment_adjustment(stock)
    conviction = clamp(0.4 * fund + 0.3 * valuation + 0.25 * tech + 0.05 * (50 + sentiment_adj))
    if override:
        return conviction, override
    if conviction >= 75:
        return conviction, "Collar" if regime.risk_off else "Directional Equity"
    return conviction, "Watchlist"


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


def run_pipeline(req: PipelineRequest) -> list[Decision]:
    regime = phase1_market_regime(req.macro)
    output: list[Decision] = []

    for stock in req.nse_universe:
        passed, _ = phase2_sieve(stock)
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
                scores={"fundamental": fund, "technical": tech},
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


def demo_request() -> PipelineRequest:
    return parse_request(
        {
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
            ],
        }
    )


INDEX_HTML = """<!doctype html>
<html lang='en'>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1'>
  <title>NSE Conviction Engine</title>
  <style>
    :root { color-scheme: dark; }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      background: radial-gradient(circle at 0% 0%, #1e3a8a 0%, #020617 45%, #020617 100%);
      color: #e2e8f0;
      min-height: 100vh;
    }
    .shell { max-width: 1100px; margin: 0 auto; padding: 2rem 1.2rem 3rem; }
    .hero {
      border: 1px solid rgba(148, 163, 184, .24);
      border-radius: 18px;
      padding: 1.5rem;
      background: rgba(15, 23, 42, .65);
      backdrop-filter: blur(8px);
      box-shadow: 0 16px 45px rgba(2, 6, 23, .45);
    }
    h1 { margin: 0 0 .4rem 0; font-size: 1.75rem; }
    .sub { color: #cbd5e1; margin: 0 0 1.2rem; }
    .actions { display: flex; gap: .7rem; flex-wrap: wrap; }
    button {
      border: 0;
      border-radius: 12px;
      padding: .75rem 1rem;
      font-weight: 700;
      cursor: pointer;
      transition: transform .15s ease, opacity .15s ease;
    }
    button:hover { transform: translateY(-1px); }
    #runDemo { background: linear-gradient(90deg, #22d3ee, #2563eb); color: white; }
    #runCustom { background: #1e293b; color: #e2e8f0; border: 1px solid rgba(148,163,184,.22); }
    textarea {
      margin-top: 1rem;
      width: 100%;
      min-height: 180px;
      border-radius: 12px;
      border: 1px solid rgba(148,163,184,.22);
      background: #0f172a;
      color: #dbeafe;
      padding: .9rem;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      display: none;
    }
    .grid { margin-top: 1rem; display: grid; grid-template-columns: repeat(auto-fit, minmax(175px, 1fr)); gap: .7rem; }
    .metric {
      background: rgba(15, 23, 42, .72);
      border: 1px solid rgba(100, 116, 139, .36);
      border-radius: 12px;
      padding: .85rem;
    }
    .metric small { color: #94a3b8; display: block; }
    .metric b { font-size: 1.1rem; }
    .panel {
      margin-top: 1rem;
      background: rgba(15,23,42,.7);
      border: 1px solid rgba(100,116,139,.3);
      border-radius: 14px;
      overflow: hidden;
    }
    table { width: 100%; border-collapse: collapse; }
    th, td { text-align: left; padding: .75rem .9rem; border-bottom: 1px solid rgba(100,116,139,.2); }
    th { color: #93c5fd; font-size: .85rem; text-transform: uppercase; letter-spacing: .04em; }
    .empty { margin-top: 1rem; color: #94a3b8; }
    .pill { display: inline-block; padding: .22rem .5rem; border-radius: 999px; font-size: .76rem; font-weight: 700; }
    .ok { background: rgba(16,185,129,.2); color: #86efac; }
    .watch { background: rgba(245,158,11,.2); color: #fcd34d; }
  </style>
</head>
<body>
  <main class='shell'>
    <section class='hero'>
      <h1>NSE 8-Phase Conviction Engine</h1>
      <p class='sub'>Modern one-click workflow: run the full pipeline instantly with a curated demo portfolio, or switch to custom JSON mode.</p>
      <div class='actions'>
        <button id='runDemo'>⚡ One-Click Run</button>
        <button id='runCustom'>Use Custom JSON</button>
      </div>
      <textarea id='payload' placeholder='Paste payload JSON here'></textarea>
      <div id='metrics' class='grid'></div>
      <div id='tableWrap' class='panel' style='display:none'>
        <table>
          <thead>
            <tr>
              <th>Symbol</th><th>Conviction</th><th>Strategy</th><th>Position Size</th><th>Entry</th><th>Stop</th><th>Target</th><th>Win Prob.</th>
            </tr>
          </thead>
          <tbody id='rows'></tbody>
        </table>
      </div>
      <p id='empty' class='empty'>No run yet. Click <b>One-Click Run</b> to generate a portfolio instantly.</p>
    </section>
  </main>
  <script>
    const payloadArea = document.getElementById('payload');
    const runDemoButton = document.getElementById('runDemo');
    const runCustomButton = document.getElementById('runCustom');

    runCustomButton.addEventListener('click', () => {
      payloadArea.style.display = payloadArea.style.display === 'none' ? 'block' : 'none';
      if (!payloadArea.value) {
        payloadArea.value = JSON.stringify(window.demoPayload, null, 2);
      }
      runCustomButton.textContent = payloadArea.style.display === 'none' ? 'Use Custom JSON' : 'Run Custom JSON';
      if (payloadArea.style.display === 'block' && runCustomButton.textContent === 'Run Custom JSON') {
        runCustomButton.onclick = runCustom;
      }
    });

    async function runCustom() {
      const payload = JSON.parse(payloadArea.value);
      const res = await fetch('/analyze', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload),
      });
      render(await res.json());
    }

    runDemoButton.addEventListener('click', async () => {
      const res = await fetch('/demo');
      render(await res.json());
    });

    function metric(label, value) {
      return `<article class='metric'><small>${label}</small><b>${value}</b></article>`;
    }

    function render(result) {
      const decisions = result.decisions || [];
      const top = decisions[0];
      const avgWin = decisions.length ? (decisions.reduce((a, d) => a + d.win_probability, 0) / decisions.length).toFixed(2) + '%' : '--';
      document.getElementById('metrics').innerHTML = [
        metric('Shortlisted', decisions.length),
        metric('Top Symbol', top ? top.symbol : '--'),
        metric('Top Conviction', top ? top.conviction : '--'),
        metric('Average Win Probability', avgWin)
      ].join('');

      const rows = decisions.map((d) => {
        const klass = d.strategy === 'Watchlist' ? 'watch' : 'ok';
        return `<tr>
          <td>${d.symbol}</td>
          <td>${d.conviction}</td>
          <td><span class='pill ${klass}'>${d.strategy}</span></td>
          <td>₹${Number(d.position_size).toLocaleString()}</td>
          <td>${d.entry}</td>
          <td>${d.stop}</td>
          <td>${d.target}</td>
          <td>${d.win_probability}%</td>
        </tr>`;
      }).join('');

      document.getElementById('rows').innerHTML = rows;
      document.getElementById('tableWrap').style.display = decisions.length ? 'block' : 'none';
      document.getElementById('empty').textContent = decisions.length ? '' : 'No stocks passed filters in this run.';
    }

    window.demoPayload = {"macro":{"pmi":54,"cpi":4.9,"gdp_growth":6.8,"yield_spread":0.8,"vix":18},"capital":1000000,"nse_universe":[{"symbol":"RELIANCE","sector":"Energy","revenue_growth":18,"eps_growth":20,"roe":17,"debt_to_equity":0.35,"pe":22,"growth_rate":24,"relative_strength_percentile":86,"roic":19,"wacc":10,"gross_margin_5y":[42,43,44,44,45],"cfo":95000,"net_income":82000,"sector_cagr_3y":15,"close":3000,"dma50":2920,"dma150":2790,"dma200":2700,"up_volume":2400000,"down_volume":1300000,"bb_width":0.12,"sentiment":28,"severity_rank":2,"win_rate":0.58,"win_loss_ratio":1.7,"atr":65},{"symbol":"INFY","sector":"IT","revenue_growth":17,"eps_growth":18,"roe":28,"debt_to_equity":0.08,"pe":27,"growth_rate":30,"relative_strength_percentile":79,"roic":26,"wacc":11,"gross_margin_5y":[35,35.4,36,35.8,36.3],"cfo":39800,"net_income":33400,"sector_cagr_3y":17,"close":1650,"dma50":1610,"dma150":1540,"dma200":1510,"up_volume":1500000,"down_volume":780000,"bb_width":0.09,"sentiment":18,"severity_rank":1,"win_rate":0.55,"win_loss_ratio":1.6,"atr":32}]};
  </script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, data: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/":
            body = INDEX_HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/demo":
            decisions = run_pipeline(demo_request())
            self._send_json({"count": len(decisions), "decisions": [asdict(d) for d in decisions]})
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if self.path != "/analyze":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode() or "{}")
        req = parse_request(payload)
        decisions = run_pipeline(req)
        self._send_json({"count": len(decisions), "decisions": [asdict(d) for d in decisions]})


def serve(host: str = "0.0.0.0", port: int = 8000) -> None:
    server = HTTPServer((host, port), Handler)
    print(f"Serving on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    serve()
