# NSE 8-Phase Portfolio Engine

A self-contained Python web app that transforms an NSE stock universe into high-conviction, risk-managed trade ideas.

## Holistic interface (rebuilt)
- Sidebar navigation with dedicated workspaces: Overview, Screener, Strategy, Education.
- Unified top action bar with one-click analysis and privacy mode toggle.
- KPI-first portfolio overview with regime badge, VaR, drawdown, and conviction analytics.
- Integrated screener controls + auto-refresh analysis loop for iterative threshold tuning.
- Strategy table with execution-ready trade draft popups for operational handoff.

## Run (one click local)

### macOS / Linux (bash/zsh)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

### Windows PowerShell
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:8000` and click **⚡ One-Click Analyze**.

## API
- `GET /demo` → quick analyzed demo portfolio
- `POST /analyze` → custom payload with optional controls:
  - `settings`: `{ peg_max, roe_min, rs_min, revenue_growth_min, eps_growth_min, de_max }`
  - `selected_symbols`: `[...]`

## Troubleshooting
- **PowerShell `source` error**: `source` is a Unix shell command. In PowerShell use:
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
- If PowerShell blocks script activation, run this once in the same terminal session:
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  .\.venv\Scripts\Activate.ps1
  ```
- If nothing appears in terminal, check server health in another terminal:
  ```bash
  curl http://127.0.0.1:8000/health
  ```
  Expected: `{"status":"ok"}`
- If port 8000 is busy:
  - macOS/Linux:
    ```bash
    PORT=8001 python app.py
    ```
  - Windows PowerShell:
    ```powershell
    $env:PORT=8001
    python app.py
    ```
