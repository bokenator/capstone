"""Widget tools - HTML widget rendering for MCP."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field


@dataclass(frozen=True)
class Widget:
    """Definition of an HTML widget."""
    identifier: str
    title: str
    template_uri: str
    invoking: str
    invoked: str
    html: str
    response_text: str


MIME_TYPE = "text/html+skybridge"

_ASSET_CANDIDATES = [
    Path(__file__).resolve().parent.parent / "assets",
    Path.cwd() / "assets",
]


@lru_cache(maxsize=None)
def load_widget_html(filename: str) -> str:
    """Load widget HTML from disk, falling back to a tiny placeholder."""
    for base in _ASSET_CANDIDATES:
        html_path = base / filename
        if html_path.exists():
            return html_path.read_text(encoding="utf8")
    return "<html><body><p>Widget asset missing.</p></body></html>"


class WidgetInput(BaseModel):
    """Schema for widget tools."""

    note: str = Field(..., description="Text to inject into the widget.")

    model_config = ConfigDict(extra="forbid")


WIDGET_TOOL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "note": {
            "type": "string",
            "description": "Text to inject into the widget.",
        }
    },
    "required": ["note"],
    "additionalProperties": False,
}


def get_widgets() -> List[Widget]:
    """Get all available widgets (for resources)."""
    return [
        Widget(
            identifier="backtesting-widget",
            title="Backtesting Widget",
            template_uri="ui://widget/backtesting-widget.html",
            invoking="Preparing backtest widget",
            invoked="Rendered backtest widget",
            html=load_widget_html("backtesting-widget.html"),
            response_text="Hydrated the backtesting widget with your note.",
        ),
        Widget(
            identifier="equity-chart",
            title="Equity Chart",
            template_uri="ui://widget/equity-chart.html",
            invoking="Rendering equity chart",
            invoked="Rendered equity chart",
            html=load_widget_html("equity-chart.html"),
            response_text="Rendered the equity price view.",
        ),
    ]


# Widget identifiers that should NOT be exposed as standalone tools.
# These widgets are rendered via outputTemplate from other tools (e.g., backtest).
# Exposing them as tools causes ChatGPT to call them redundantly.
WIDGETS_NOT_AS_TOOLS = {"backtesting-widget"}


def resource_description(widget: Widget) -> str:
    """Generate a description for a widget resource."""
    return f"{widget.title} markup"


def tool_meta(widget: Widget) -> Dict[str, Any]:
    """Generate metadata for a widget tool."""
    return {
        "openai/outputTemplate": widget.template_uri,
        "openai/toolInvocation/invoking": widget.invoking,
        "openai/toolInvocation/invoked": widget.invoked,
        "openai/widgetAccessible": True,
    }


def tool_invocation_meta(widget: Widget) -> Dict[str, Any]:
    """Generate invocation metadata for a widget tool."""
    return {
        "openai/toolInvocation/invoking": widget.invoking,
        "openai/toolInvocation/invoked": widget.invoked,
    }
