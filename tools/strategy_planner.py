"""Strategy Planner - Interprets user intent and produces strategy specifications.

This module uses an LLM to analyze user requests and determine what strategies
need to be backtested. It follows the "everything is a strategy" philosophy
where buy-and-hold is just a trivial strategy and all strategies are equal peers.

Example:
    User: "RSI long-only and long-short on SPY with buy-and-hold"
    Output: [
        StrategySpec(name="RSI", symbols=["SPY"], directions=["longonly", "both"]),
        StrategySpec(name="Buy & Hold SPY", symbols=["SPY"], directions=["longonly"], slippage=0)
    ]
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .models import PlannerOutput, StrategySpec

logger = logging.getLogger(__name__)


PLANNER_SYSTEM_PROMPT = """You are a trading strategy planner. Your job is to analyze user requests and determine what strategies need to be backtested.

OUTPUT FORMAT - You MUST output valid JSON:
{
    "strategies": [
        {
            "name": "Display name for this strategy",
            "description": "Natural language description for the strategy generator",
            "symbols": ["AAPL", "MSFT"],
            "directions": ["longonly"],
            "execution_price": "open",
            "slippage": 0.0005,
            "stop_loss": null,
            "take_profit": null,
            "init_cash": 100.0
        }
    ]
}

IMPORTANT: `directions` is a LIST. One strategy can be run multiple times with different directions.
- The same strategy CODE is generated once
- But executed N times, once per direction in the list
- Each execution produces a separate equity curve in the output

RULES:

1. EVERY STRATEGY IS EQUAL:
   - There is no special "benchmark" or "primary" - all strategies are treated the same
   - Buy-and-hold is just a strategy where position = 1 always
   - The widget displays all strategies as equal peers

2. BUY-AND-HOLD DETECTION:
   - If user mentions "vs SPY", "compare to SPY", "SPY benchmark" -> add buy-and-hold SPY strategy
   - If user mentions "vs buy-and-hold", "compare to holding" -> add buy-and-hold on same symbol
   - Buy-and-hold strategies typically have slippage=0 (theoretical, no trading)
   - Buy-and-hold description should be: "Buy and hold {SYMBOL} - always maintain long position"

3. SLIPPAGE:
   - Active strategies: use user-specified or default 0.0005 (5 bps)
   - Buy-and-hold strategies: always 0 (theoretical, single entry)

4. DIRECTIONS (plural):
   - Default to ["longonly"] unless user specifies otherwise
   - If user says "long/short" or "both directions" -> ["both"]
   - If user wants to COMPARE long-only vs long-short -> ["longonly", "both"]
     This generates ONE strategy but runs it TWICE with different direction params

5. MULTI-SYMBOL STRATEGIES:
   - If user lists multiple symbols for ONE strategy (e.g., "momentum on AAPL, MSFT, GOOGL")
     -> single strategy with symbols=["AAPL", "MSFT", "GOOGL"]
   - If user wants SEPARATE strategies per symbol (e.g., "compare RSI on AAPL vs MSFT")
     -> multiple strategies, each with one symbol

6. NAMING CONVENTIONS:
   - Use concise, descriptive names
   - For comparison strategies, include direction in name if multiple directions
   - Examples: "RSI Mean Reversion", "MA Crossover", "Buy & Hold SPY"

EXAMPLES:

User: "Backtest RSI mean reversion on AAPL"
-> strategies: [{name: "RSI Mean Reversion", description: "RSI mean reversion: buy when RSI < 30, sell when RSI > 70", symbols: ["AAPL"], directions: ["longonly"]}]
-> Total runs: 1

User: "Backtest MA crossover on AAPL vs SPY"
-> strategies: [
    {name: "MA Crossover", description: "Moving average crossover strategy", symbols: ["AAPL"], directions: ["longonly"], slippage: 0.0005},
    {name: "Buy & Hold SPY", description: "Buy and hold SPY - always maintain long position", symbols: ["SPY"], directions: ["longonly"], slippage: 0}
  ]
-> Total runs: 2

User: "Test RSI on SPY, show long-only vs long-short"
-> strategies: [
    {name: "RSI", description: "RSI mean reversion strategy", symbols: ["SPY"], directions: ["longonly", "both"]}
  ]
-> Total runs: 2 (same code, different direction params)

User: "RSI long-only and long-short on SPY with buy-and-hold"
-> strategies: [
    {name: "RSI", description: "RSI mean reversion strategy", symbols: ["SPY"], directions: ["longonly", "both"], slippage: 0.0005},
    {name: "Buy & Hold SPY", description: "Buy and hold SPY - always maintain long position", symbols: ["SPY"], directions: ["longonly"], slippage: 0}
  ]
-> Total runs: 3 (RSI×2 + B&H×1)

User: "Compare MACD vs RSI vs buy-and-hold on AAPL"
-> strategies: [
    {name: "MACD Strategy", description: "MACD crossover strategy", symbols: ["AAPL"], directions: ["longonly"]},
    {name: "RSI Strategy", description: "RSI mean reversion strategy", symbols: ["AAPL"], directions: ["longonly"]},
    {name: "Buy & Hold AAPL", description: "Buy and hold AAPL - always maintain long position", symbols: ["AAPL"], directions: ["longonly"], slippage: 0}
  ]
-> Total runs: 3

User: "Pairs trading on GLD and SLV"
-> strategies: [
    {name: "Pairs Trading", description: "Pairs trading strategy between GLD and SLV using z-score of spread", symbols: ["GLD", "SLV"], directions: ["both"]}
  ]
-> Total runs: 1 (multi-asset strategy)

User: "Momentum strategy on tech stocks AAPL, MSFT, GOOGL vs SPY"
-> strategies: [
    {name: "Momentum", description: "Momentum strategy", symbols: ["AAPL", "MSFT", "GOOGL"], directions: ["longonly"]},
    {name: "Buy & Hold SPY", description: "Buy and hold SPY - always maintain long position", symbols: ["SPY"], directions: ["longonly"], slippage: 0}
  ]
-> Total runs: 2 (momentum produces 3 equity curves, B&H produces 1)
"""


class StrategyPlanner:
    """Plans what strategies to run based on user prompt."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
    ):
        """Initialize the planner.

        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
            model: Model to use for planning. gpt-4o-mini is fast and cheap.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("Missing OPENAI_API_KEY in environment")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model

    def plan(
        self,
        prompt: str,
        default_symbols: Optional[List[str]] = None,
        default_direction: str = "longonly",
        default_slippage: float = 0.0005,
    ) -> PlannerOutput:
        """Plan what strategies to run based on user prompt.

        Args:
            prompt: User's natural language request.
            default_symbols: Default symbols if not specified in prompt.
            default_direction: Default direction if not specified.
            default_slippage: Default slippage for active strategies.

        Returns:
            PlannerOutput with list of StrategySpecs.
        """
        logger.info("=" * 60)
        logger.info("STRATEGY PLANNER")
        logger.info("=" * 60)
        logger.info(f"User prompt: {prompt}")
        logger.info(f"Default symbols: {default_symbols}")

        # Build context message
        context_parts = []
        if default_symbols:
            context_parts.append(f"Default symbols (if not specified): {default_symbols}")
        if default_direction != "longonly":
            context_parts.append(f"Default direction: {default_direction}")
        if default_slippage != 0.0005:
            context_parts.append(f"Default slippage: {default_slippage}")

        user_message = f"User request: {prompt}"
        if context_parts:
            user_message += f"\n\nContext: {', '.join(context_parts)}"

        try:
            logger.info("Calling OpenAI API for planning...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,  # Low temperature for consistent output
            )
            logger.info("OpenAI API call successful")
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

        raw_response = response.choices[0].message.content or "{}"
        logger.info(f"Raw response: {raw_response}")

        # Parse response
        try:
            parsed = json.loads(raw_response)
            strategies_data = parsed.get("strategies", [])
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            raise ValueError(f"Invalid JSON from planner: {e}")

        # Convert to StrategySpec objects
        strategies = []
        for s in strategies_data:
            # Apply defaults
            if "symbols" not in s and default_symbols:
                s["symbols"] = default_symbols
            if "slippage" not in s:
                # Check if this is a buy-and-hold strategy
                name = s.get("name", "").lower()
                desc = s.get("description", "").lower()
                if "buy" in name and "hold" in name or "buy" in desc and "hold" in desc:
                    s["slippage"] = 0.0
                else:
                    s["slippage"] = default_slippage

            try:
                spec = StrategySpec(**s)
                strategies.append(spec)
                logger.info(f"  Strategy: {spec.name}")
                logger.info(f"    Symbols: {spec.symbols}")
                logger.info(f"    Directions: {spec.directions}")
                logger.info(f"    Slippage: {spec.slippage}")
            except Exception as e:
                logger.warning(f"Failed to parse strategy spec: {e}")
                continue

        if not strategies:
            # Fallback: create a simple strategy from the prompt
            logger.warning("No strategies parsed, creating fallback")
            strategies = [
                StrategySpec(
                    name="Strategy",
                    description=prompt,
                    symbols=default_symbols or ["SPY"],
                    directions=[default_direction],
                    slippage=default_slippage,
                )
            ]

        # Calculate total runs
        total_runs = sum(
            len(s.directions) * len(s.symbols)
            for s in strategies
        )
        logger.info(f"Total strategies: {len(strategies)}")
        logger.info(f"Total backtest runs: {total_runs}")

        return PlannerOutput(strategies=strategies)


def get_strategy_planner(
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
) -> StrategyPlanner:
    """Factory function to create a StrategyPlanner instance."""
    return StrategyPlanner(api_key=api_key, model=model)
