# NSE 8-Phase Portfolio Engine

A Python web app that converts a raw NSE stock list into a structured high-conviction trading decision template using an 8-phase pipeline.

## What’s new
- Modern dashboard-style UI (glassmorphism cards, summary metrics, and result table)
- **One-click run** via `/demo` endpoint from the homepage (`⚡ One-Click Run`)
- Optional custom JSON mode for advanced users

## Features
- Market regime classification (Expansion/Stagflation/Recession/Crisis)
- 3-stage stock sieve: fundamentals, PEG valuation, and momentum
- Fundamental scoring with proxies for management quality, moat, earnings quality, and sector tailwind
- Technical scoring for trend stack, volume footprint, and volatility compression
- News sentiment override with severity matrix
- Final conviction scoring and strategy selection (Directional / Collar / Watchlist)
- Position sizing via fractional Kelly + ATR volatility scaling
- Monte Carlo win probability estimation and final entry/stop/target output

## Run
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:8000 and click **One-Click Run**.

## API
- `GET /demo`: one-click analyzed sample portfolio output
- `POST /analyze` accepts:
  - `macro`: `{ pmi, cpi, gdp_growth, yield_spread, vix }`
  - `nse_universe`: list of stock objects with the required fields in the UI sample
  - `capital`: optional total capital
