# Backtesting MCP server

Python MCP server built with the official `mcp.server.fastmcp.FastMCP` helper, modeled after the Apps SDK example servers in `openai-apps-sdk-examples/`.

## Tools

- `backtesting-widget` – simple demo tool that returns `{"note": ...}` for widget hydration.
- `get-equity-prices` – fetch OHLCV bars from Alpaca and renders `ui://widget/equity-chart.html`.
- `backtest-strategy` – moving-average crossover backtest rendered with `ui://widget/equity-chart.html`.
- `dynamic-backtest` – AI-generated strategy backtest (requires `OPENAI_API_KEY`) rendered with `ui://widget/equity-chart.html`.

## Setup

```bash
cd mcp_server
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Create an environment file:

```bash
cp .env.example .env
```

Then edit `.env` and set:

- `ALPACA_KEY_ID` / `ALPACA_SECRET_KEY` (required for Alpaca tools)
- `OPENAI_API_KEY` (required for `dynamic-backtest`)
- `HOST` / `PORT` (optional; defaults to `0.0.0.0:8090`)

## Run

```bash
python main.py
```

This runs a FastAPI app with the streaming HTTP transport enabled. The endpoints match the Apps SDK examples:

- `GET /mcp` – SSE stream
- `POST /mcp/messages?sessionId=...` – follow-up messages for an active session
