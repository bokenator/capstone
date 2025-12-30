"""Backtesting-themed MCP server built with the FastMCP helper.

This mirrors the Pizzaz demo from the OpenAI apps SDK examples while keeping the
footprint small. The server exposes a HTML widget as both a resource and a tool.
The tool echoes a user-provided note back as structured content so MCP clients
can hydrate the widget.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, List, cast

from dotenv import load_dotenv
import mcp.types as types
from mcp.server.fastmcp import FastMCP
from pydantic import ValidationError
from pydantic.networks import AnyUrl

from tools import (
    BACKTEST_TOOL_NAME,
    BACKTEST_TOOL_SCHEMA,
    EQUITY_TOOL_NAME,
    EQUITY_TOOL_SCHEMA,
    MIME_TYPE,
    WIDGET_TOOL_SCHEMA,
    BacktestInput,
    EquityPricesInput,
    Widget,
    WidgetInput,
    fetch_equity_prices,
    get_widgets,
    ma_crossover_backtest,
    resource_description,
    tool_invocation_meta,
    tool_meta,
)


# Load environment variables from a local .env when present.
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)


# Initialize widgets
widgets: List[Widget] = get_widgets()
WIDGETS_BY_ID: Dict[str, Widget] = {widget.identifier: widget for widget in widgets}
WIDGETS_BY_URI: Dict[str, Widget] = {widget.template_uri: widget for widget in widgets}


mcp = FastMCP(name="backtesting-mcp", stateless_http=True)


@mcp._mcp_server.list_tools()
async def _list_tools() -> List[types.Tool]:
    tools = [
        types.Tool(
            name=widget.identifier,
            title=widget.title,
            description=widget.title,
            inputSchema=deepcopy(WIDGET_TOOL_SCHEMA),
            _meta=tool_meta(widget),
            annotations=types.ToolAnnotations(
                destructiveHint=False,
                openWorldHint=False,
                readOnlyHint=True,
            ),
        )
        for widget in widgets
    ]
    tools.append(
        types.Tool(
            name=EQUITY_TOOL_NAME,
            title="Get Equity Prices (Alpaca)",
            description="Fetch historical equity bars from Alpaca",
            inputSchema=deepcopy(EQUITY_TOOL_SCHEMA),
            _meta={
                "openai/outputTemplate": "ui://widget/equity-chart.html",
                "openai/toolInvocation/invoking": "Fetching equity prices",
                "openai/toolInvocation/invoked": "Fetched equity prices",
                "openai/resultCanProduceWidget": True,
                "openai/widgetAccessible": True,
            },
            annotations=types.ToolAnnotations(
                destructiveHint=False,
                openWorldHint=False,
                readOnlyHint=True,
            ),
        )
    )
    tools.append(
        types.Tool(
            name=BACKTEST_TOOL_NAME,
            title="Backtest Strategy (MA Crossover)",
            description="Run a moving-average crossover backtest over one or more symbols.",
            inputSchema=deepcopy(BACKTEST_TOOL_SCHEMA),
            _meta={
                "openai/outputTemplate": "ui://widget/equity-chart.html",
                "openai/toolInvocation/invoking": "Running backtest",
                "openai/toolInvocation/invoked": "Backtest complete",
                "openai/resultCanProduceWidget": True,
                "openai/widgetAccessible": True,
            },
            annotations=types.ToolAnnotations(
                destructiveHint=False,
                openWorldHint=False,
                readOnlyHint=True,
            ),
        )
    )
    return tools


@mcp._mcp_server.list_resources()
async def _list_resources() -> List[types.Resource]:
    return [
        types.Resource(
            name=widget.title,
            title=widget.title,
            uri=cast(AnyUrl, widget.template_uri),
            description=resource_description(widget),
            mimeType=MIME_TYPE,
            _meta=tool_meta(widget),
        )
        for widget in widgets
    ]


@mcp._mcp_server.list_resource_templates()
async def _list_resource_templates() -> List[types.ResourceTemplate]:
    return [
        types.ResourceTemplate(
            name=widget.title,
            title=widget.title,
            uriTemplate=widget.template_uri,
            description=resource_description(widget),
            mimeType=MIME_TYPE,
            _meta=tool_meta(widget),
        )
        for widget in widgets
    ]


async def _handle_read_resource(req: types.ReadResourceRequest) -> types.ServerResult:
    widget = WIDGETS_BY_URI.get(str(req.params.uri))
    if widget is None:
        return types.ServerResult(
            types.ReadResourceResult(
                contents=[],
                _meta={"error": f"Unknown resource: {req.params.uri}"},
            )
        )

    contents: List[types.TextResourceContents | types.BlobResourceContents] = [
        types.TextResourceContents(
            uri=cast(AnyUrl, widget.template_uri),
            mimeType=MIME_TYPE,
            text=widget.html,
            _meta=tool_meta(widget),
        )
    ]
    return types.ServerResult(types.ReadResourceResult(contents=contents))


async def _call_tool_request(req: types.CallToolRequest) -> types.ServerResult:
    if req.params.name == BACKTEST_TOOL_NAME:
        arguments = req.params.arguments or {}
        try:
            payload = BacktestInput.model_validate(arguments)
        except ValidationError as exc:
            return types.ServerResult(
                types.CallToolResult(
                    content=[
                        types.TextContent(
                            type="text",
                            text=f"Input validation error: {exc.errors()}",
                        )
                    ],
                    isError=True,
                )
            )
        try:
            result = ma_crossover_backtest(payload)
        except Exception as exc:
            return types.ServerResult(
                types.CallToolResult(
                    content=[types.TextContent(type="text", text=f"Failed to run backtest: {exc}")],
                    isError=True,
                )
            )
        text = f"Backtest complete for {len(payload.symbols)} symbols; bars: {len(result['data'])}"
        return types.ServerResult(
            types.CallToolResult(
                content=[types.TextContent(type="text", text=text)],
                structuredContent=result,
                _meta={
                    "openai/outputTemplate": "ui://widget/equity-chart.html",
                    "openai/toolInvocation/invoking": "Running backtest",
                    "openai/toolInvocation/invoked": "Backtest complete",
                },
            )
        )

    if req.params.name == EQUITY_TOOL_NAME:
        arguments = req.params.arguments or {}
        try:
            payload = EquityPricesInput.model_validate(arguments)
        except ValidationError as exc:
            return types.ServerResult(
                types.CallToolResult(
                    content=[
                        types.TextContent(
                            type="text",
                            text=f"Input validation error: {exc.errors()}",
                        )
                    ],
                    isError=True,
                )
            )

        try:
            result = fetch_equity_prices(payload)
        except Exception as exc:
            return types.ServerResult(
                types.CallToolResult(
                    content=[
                        types.TextContent(
                            type="text",
                            text=f"Failed to fetch prices: {exc}",
                        )
                    ],
                    isError=True,
                )
            )

        text = f"Fetched {len(result['data'])} bars for {result['symbol']} ({result['timeframe']})."
        return types.ServerResult(
            types.CallToolResult(
                content=[types.TextContent(type="text", text=text)],
                structuredContent=result,
                _meta={
                    "openai/outputTemplate": "ui://widget/equity-chart.html",
                    "openai/toolInvocation/invoking": "Fetching equity prices",
                    "openai/toolInvocation/invoked": "Fetched equity prices",
                },
            )
        )

    widget = WIDGETS_BY_ID.get(req.params.name)
    if widget is None:
        return types.ServerResult(
            types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=f"Unknown tool: {req.params.name}",
                    )
                ],
                isError=True,
            )
        )

    arguments = req.params.arguments or {}
    try:
        payload = WidgetInput.model_validate(arguments)
    except ValidationError as exc:
        return types.ServerResult(
            types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=f"Input validation error: {exc.errors()}",
                    )
                ],
                isError=True,
            )
        )

    note = payload.note
    meta = tool_invocation_meta(widget)

    return types.ServerResult(
        types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=widget.response_text,
                )
            ],
            structuredContent={"note": note},
            _meta=meta,
        )
    )


mcp._mcp_server.request_handlers[types.CallToolRequest] = _call_tool_request
mcp._mcp_server.request_handlers[types.ReadResourceRequest] = _handle_read_resource


app = mcp.streamable_http_app()

try:
    from starlette.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=False,
    )
except Exception:
    # CORS middleware is optional; skip if starlette is unavailable.
    pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8090)
