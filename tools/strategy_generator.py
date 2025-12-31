"""Codex-powered strategy agent for iterative code generation and fixing.

This module implements an agentic approach where Codex generates strategy code,
sees execution errors, and iteratively fixes them until the code works.

Phase 2 Update:
- LLM outputs structured JSON with param_schema and code
- Uses OpenAI JSON mode for reliable parsing
- Validates param_schema structure
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import OpenAI

# Configure logging
logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an expert quantitative trading strategy developer using vectorbt. Your task is to generate Python code that implements trading strategies using POSITION TARGETS.

CRITICAL: Your strategy must generate position changes throughout the data period. A strategy that stays flat or never changes position is useless.

OUTPUT FORMAT - You MUST output valid JSON with exactly these fields:
{
    "data_schema": {
        "slot_name": {
            "frequency": "1Day|1Hour|1Week|quarterly",
            "columns": ["open", "high", "low", "close", "volume"],
            "data_type": "ohlcv|fundamental",
            "description": "What this data slot contains"
        }
    },
    "param_schema": {
        "param_name": {
            "type": "int|float|bool|enum",
            "default": <default_value>,
            "min": <optional_min>,
            "max": <optional_max>,
            "values": [<optional_enum_values>],
            "description": "What this parameter does"
        }
    },
    "code": "def generate_signals(data: dict, params: dict) -> dict[str, pd.Series]:\\n    ..."
}

DATA_SCHEMA REQUIREMENTS:
- Define ALL data slots your strategy needs (e.g., "prices" for OHLCV data)
- For single-asset strategies, use "prices" as the slot name
- For multi-asset strategies, use descriptive names like "asset_a", "asset_b"
- Symbols are injected at runtime - NEVER hardcode symbols in your code
- frequency: "1Day" for daily, "1Hour" for hourly, etc.
- data_type: "ohlcv" for price data, "fundamental" for earnings/balance sheet

PARAM_SCHEMA REQUIREMENTS:
- Define ONLY STRATEGY-SPECIFIC parameters (e.g., fast_window, rsi_threshold)
- DO NOT include built-in parameters - these are injected by the orchestrator:
  * direction (longonly/shortonly/both) - handled by orchestrator
  * execution_price (open/close) - handled by orchestrator
  * stop_loss, take_profit, trailing_stop, slippage - handled by orchestrator
- Each parameter needs: type, default, description
- For int/float: include min and max bounds
- For enum: include values array
- Types: "int", "float", "bool", "enum"

CODE REQUIREMENTS - Your code MUST follow this exact pattern:
```python
def generate_signals(data: dict, params: dict) -> dict[str, pd.Series]:
    prices = data['prices']  # OHLCV DataFrame
    close = prices['close']

    # Get parameters (use params.get with defaults matching param_schema)
    fast_window = params.get('fast_window', 10)

    # Calculate indicators using vectorbt
    fast_ma = vbt.MA.run(close, window=fast_window).ma

    # Use explicit state machine: nan = no change, then ffill
    position = pd.Series(np.nan, index=prices.index, dtype=float)
    position[condition] = 1   # Go long
    position[other_condition] = 0  # Go flat
    position = position.ffill().fillna(0)

    return {"prices": position}
```

EXAMPLE JSON OUTPUT - RSI Strategy:
{
    "data_schema": {
        "prices": {
            "frequency": "1Day",
            "columns": ["open", "high", "low", "close", "volume"],
            "data_type": "ohlcv",
            "description": "Daily OHLCV price data"
        }
    },
    "param_schema": {
        "rsi_window": {
            "type": "int",
            "default": 14,
            "min": 2,
            "max": 100,
            "description": "RSI calculation window"
        },
        "oversold": {
            "type": "float",
            "default": 30,
            "min": 0,
            "max": 50,
            "description": "RSI level considered oversold (go long)"
        },
        "overbought": {
            "type": "float",
            "default": 70,
            "min": 50,
            "max": 100,
            "description": "RSI level considered overbought (go flat)"
        }
    },
    "code": "def generate_signals(data: dict, params: dict) -> dict[str, pd.Series]:\\n    prices = data['prices']\\n    close = prices['close']\\n\\n    rsi_window = params.get('rsi_window', 14)\\n    oversold = params.get('oversold', 30)\\n    overbought = params.get('overbought', 70)\\n\\n    rsi = vbt.RSI.run(close, window=rsi_window).rsi\\n\\n    position = pd.Series(np.nan, index=prices.index, dtype=float)\\n    position[rsi < oversold] = 1\\n    position[rsi > overbought] = 0\\n    position = position.ffill().fillna(0)\\n\\n    return {\\"prices\\": position}"
}

EXAMPLE JSON OUTPUT - MA Crossover:
{
    "data_schema": {
        "prices": {
            "frequency": "1Day",
            "columns": ["open", "high", "low", "close", "volume"],
            "data_type": "ohlcv",
            "description": "Daily OHLCV price data"
        }
    },
    "param_schema": {
        "fast_window": {
            "type": "int",
            "default": 10,
            "min": 2,
            "max": 100,
            "description": "Fast moving average window"
        },
        "slow_window": {
            "type": "int",
            "default": 30,
            "min": 5,
            "max": 500,
            "description": "Slow moving average window"
        }
    },
    "code": "def generate_signals(data: dict, params: dict) -> dict[str, pd.Series]:\\n    prices = data['prices']\\n    close = prices['close']\\n\\n    fast_window = params.get('fast_window', 10)\\n    slow_window = params.get('slow_window', 30)\\n\\n    fast_ma = vbt.MA.run(close, window=fast_window).ma\\n    slow_ma = vbt.MA.run(close, window=slow_window).ma\\n\\n    position = pd.Series(np.nan, index=prices.index, dtype=float)\\n    position[fast_ma > slow_ma] = 1\\n    position[fast_ma < slow_ma] = 0\\n    position = position.ffill().fillna(0)\\n\\n    return {\\"prices\\": position}"
}

EXAMPLE JSON OUTPUT - Pairs Trading (Multi-Asset):
{
    "data_schema": {
        "asset_a": {
            "frequency": "1Day",
            "columns": ["open", "high", "low", "close", "volume"],
            "data_type": "ohlcv",
            "description": "First asset in the pair"
        },
        "asset_b": {
            "frequency": "1Day",
            "columns": ["open", "high", "low", "close", "volume"],
            "data_type": "ohlcv",
            "description": "Second asset in the pair"
        }
    },
    "param_schema": {
        "zscore_entry": {
            "type": "float",
            "default": 2.0,
            "min": 0.5,
            "max": 5.0,
            "description": "Z-score threshold to enter trade"
        },
        "zscore_exit": {
            "type": "float",
            "default": 0.5,
            "min": 0,
            "max": 2.0,
            "description": "Z-score threshold to exit trade"
        },
        "lookback": {
            "type": "int",
            "default": 20,
            "min": 5,
            "max": 100,
            "description": "Lookback period for mean/std calculation"
        }
    },
    "code": "def generate_signals(data: dict, params: dict) -> dict[str, pd.Series]:\\n    a = data['asset_a']['close']\\n    b = data['asset_b']['close']\\n\\n    zscore_entry = params.get('zscore_entry', 2.0)\\n    zscore_exit = params.get('zscore_exit', 0.5)\\n    lookback = params.get('lookback', 20)\\n\\n    spread = a / b\\n    spread_mean = spread.rolling(lookback).mean()\\n    spread_std = spread.rolling(lookback).std()\\n    zscore = (spread - spread_mean) / spread_std\\n\\n    pos_a = pd.Series(np.nan, index=a.index)\\n    pos_b = pd.Series(np.nan, index=b.index)\\n\\n    low_spread = zscore < -zscore_entry\\n    pos_a[low_spread] = 1\\n    pos_b[low_spread] = -1\\n\\n    high_spread = zscore > zscore_entry\\n    pos_a[high_spread] = -1\\n    pos_b[high_spread] = 1\\n\\n    normalized = abs(zscore) < zscore_exit\\n    pos_a[normalized] = 0\\n    pos_b[normalized] = 0\\n\\n    pos_a = pos_a.ffill().fillna(0)\\n    pos_b = pos_b.ffill().fillna(0)\\n\\n    return {\\"asset_a\\": pos_a, \\"asset_b\\": pos_b}"
}

POSITION TARGET FORMAT:
- +1 = long position (buy and hold)
- 0 = flat (no position, in cash)
- -1 = short position (sell short and hold)

The orchestrator automatically:
- Converts position changes to entry/exit signals
- Applies direction filtering (longonly clamps to [0,1], shortonly to [-1,0])
- Shifts signals by 1 bar to prevent lookahead bias

CODE RULES:
1. Function MUST be named `generate_signals`
2. Function MUST accept `data: dict` and `params: dict` as arguments
3. Function MUST return `dict[str, pd.Series]` mapping slot names to position Series
4. For single-asset strategies, return {"prices": position}
5. Position values MUST be +1, 0, or -1 (use np.nan for "no change" before ffill)
6. ALWAYS use the state machine pattern: pd.Series(np.nan, index=..., dtype=float), assign positions, then ffill().fillna(0)
7. Access parameters via params.get('name', default_value)
8. Available pre-imported modules: pandas (pd), numpy (np), vectorbt (vbt)
9. DO NOT import any modules - they are pre-imported
10. DO NOT use file I/O, network calls, or system commands

AVAILABLE VECTORBT INDICATORS:
- vbt.MA.run(close, window) → .ma for values
- vbt.RSI.run(close, window) → .rsi for values
- vbt.BBANDS.run(close, window, alpha) → .upper, .middle, .lower
- vbt.MACD.run(close, fast_window, slow_window, signal_window) → .macd, .signal
- vbt.ATR.run(high, low, close, window) → .atr
- vbt.STOCH.run(high, low, close, k_window, d_window) → .percent_k, .percent_d

MODIFYING EXISTING STRATEGIES:
When asked to modify an existing strategy, preserve the overall structure and only change what's requested.
- Keep working parts intact
- Only modify the specific parameters, indicators, or logic mentioned
- Output the COMPLETE JSON with updated param_schema and code"""


DANGEROUS_PATTERNS = [
    r'\bimport\s+',
    r'\bfrom\s+\w+\s+import\b',
    r'\bopen\s*\(',
    r'\bexec\s*\(',
    r'\beval\s*\(',
    r'\bcompile\s*\(',
    r'\b__import__\s*\(',
    r'\bos\.',
    r'\bsys\.',
    r'\bsubprocess\.',
    r'\bshutil\.',
    r'\brequests\.',
    r'\burllib\.',
    r'\bsocket\.',
]


@dataclass
class ExecutionResult:
    """Result from executing generated code."""
    success: bool
    error: Optional[str] = None
    error_type: Optional[str] = None
    traceback: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


@dataclass
class CodexSession:
    """Maintains conversation context for iterative code generation."""
    messages: List[Dict[str, str]] = field(default_factory=list)
    generated_code: str = ""
    data_schema: Dict[str, Any] = field(default_factory=dict)
    param_schema: Dict[str, Any] = field(default_factory=dict)
    attempts: int = 0
    max_attempts: int = 5


class CodexAgent:
    """Agentic code generator using Codex that can iteratively fix errors."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-5.1",
        max_attempts: int = 5,
    ):
        """Initialize the Codex agent.

        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
            model: Model to use. Default is gpt-5.1 for code generation tasks.
            max_attempts: Maximum number of generate-execute-fix cycles.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("Missing OPENAI_API_KEY in environment")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.max_attempts = max_attempts

    def create_session(
        self,
        strategy_prompt: str,
        base_code: Optional[str] = None,
    ) -> CodexSession:
        """Create a new Codex session for a strategy request.

        Args:
            strategy_prompt: Natural language description of the trading strategy,
                or modifications to make if base_code is provided.
            base_code: Optional existing strategy code to modify.

        Returns:
            CodexSession with initialized conversation.
        """
        logger.info("=" * 60)
        logger.info("CREATING NEW CODEX SESSION")
        logger.info("=" * 60)
        logger.info(f"Strategy prompt: {strategy_prompt}")
        logger.info(f"Base code provided: {bool(base_code)}")

        session = CodexSession(max_attempts=self.max_attempts)

        if base_code:
            # Modification mode: refine existing strategy
            logger.info("Mode: MODIFICATION (modifying existing strategy)")
            logger.info(f"Base code:\n{base_code}")
            user_message = f"""Modify this existing trading strategy:

```python
{base_code}
```

Requested changes: {strategy_prompt}

Output ONLY the modified Python function code."""
        else:
            # Creation mode: generate new strategy
            logger.info("Mode: CREATION (generating new strategy)")
            user_message = f"Create a trading strategy: {strategy_prompt}"

        session.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        logger.info("Session created successfully")
        return session

    def generate(self, session: CodexSession) -> str:
        """Generate or regenerate code based on current session state.

        Args:
            session: The current Codex session with conversation history.

        Returns:
            Generated Python code.
        """
        session.attempts += 1

        logger.info("-" * 60)
        logger.info(f"GENERATING CODE (Attempt {session.attempts}/{session.max_attempts})")
        logger.info("-" * 60)
        logger.info(f"Model: {self.model}")
        logger.info(f"Messages in conversation: {len(session.messages)}")

        try:
            logger.info("Calling OpenAI API...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=session.messages,
                temperature=0.2,
                max_completion_tokens=4000,
                response_format={"type": "json_object"},
            )
            logger.info("OpenAI API call successful")
        except Exception as e:
            logger.error(f"OpenAI API error: {type(e).__name__}: {e}")
            raise

        raw_response = response.choices[0].message.content or ""
        logger.info(f"Raw response length: {len(raw_response)} chars")

        # Parse JSON response
        try:
            parsed = json.loads(raw_response)
            code = parsed.get("code", "")
            data_schema = parsed.get("data_schema", {"prices": {"frequency": "1Day", "columns": ["open", "high", "low", "close", "volume"], "data_type": "ohlcv"}})
            param_schema = parsed.get("param_schema", {})
            logger.info("JSON parsed successfully")
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}, falling back to code extraction")
            # Fallback: try to extract code from non-JSON response
            code = self._extract_code(raw_response)
            data_schema = {"prices": {"frequency": "1Day", "columns": ["open", "high", "low", "close", "volume"], "data_type": "ohlcv"}}
            param_schema = {}

        session.generated_code = code
        session.data_schema = data_schema
        session.param_schema = param_schema

        # Log the generated code
        logger.info("=" * 60)
        logger.info("GENERATED DATA_SCHEMA:")
        logger.info("=" * 60)
        logger.info(json.dumps(data_schema, indent=2))

        logger.info("=" * 60)
        logger.info("GENERATED PARAM_SCHEMA:")
        logger.info("=" * 60)
        logger.info(json.dumps(param_schema, indent=2))

        logger.info("=" * 60)
        logger.info("GENERATED CODE:")
        logger.info("=" * 60)
        logger.info(code)
        logger.info("=" * 60)

        # Add assistant response to conversation history
        session.messages.append({
            "role": "assistant",
            "content": raw_response,
        })

        return code

    def feed_error(self, session: CodexSession, execution_result: ExecutionResult) -> None:
        """Feed an execution error back to the agent for fixing.

        Args:
            session: The current Codex session.
            execution_result: The result from executing the generated code.
        """
        logger.warning("-" * 60)
        logger.warning("FEEDING EXECUTION ERROR TO CODEX")
        logger.warning("-" * 60)
        logger.warning(f"Error Type: {execution_result.error_type}")
        logger.warning(f"Error Message: {execution_result.error}")
        if execution_result.traceback:
            logger.warning(f"Traceback:\n{execution_result.traceback}")

        error_message = f"""The code you generated failed during execution.

Error Type: {execution_result.error_type}
Error Message: {execution_result.error}

{f"Traceback:{chr(10)}{execution_result.traceback}" if execution_result.traceback else ""}

Please fix the code. Remember:
1. Function must be named `generate_signals`
2. Must accept `data: dict` and `params: dict` as arguments
3. Must return `dict[str, pd.Series]` mapping slot names to position Series
4. Position values: +1 (long), 0 (flat), -1 (short)
5. Use the state machine pattern: pd.Series(np.nan, dtype=float), assign positions, then ffill().fillna(0)
6. Only use pd, np, vbt - no imports

Output valid JSON with "data_schema", "param_schema", and "code" fields."""

        session.messages.append({
            "role": "user",
            "content": error_message,
        })

    def feed_validation_error(self, session: CodexSession, error: str) -> None:
        """Feed a validation error back to the agent.

        Args:
            session: The current Codex session.
            error: The validation error message.
        """
        logger.warning("-" * 60)
        logger.warning("FEEDING VALIDATION ERROR TO CODEX")
        logger.warning("-" * 60)
        logger.warning(f"Validation Error: {error}")

        error_message = f"""The code you generated failed validation:

{error}

Please fix the code following the exact requirements:
1. Function must be named `generate_signals`
2. Must accept `data: dict` and `params: dict` as arguments
3. Must return `dict[str, pd.Series]` mapping slot names to position Series
4. Position values: +1 (long), 0 (flat), -1 (short)
5. No imports - pd, np, vbt are pre-imported

Output valid JSON with "data_schema", "param_schema", and "code" fields."""

        session.messages.append({
            "role": "user",
            "content": error_message,
        })

    def feed_signal_error(self, session: CodexSession, position_changes: int, total_bars: int) -> None:
        """Feed a signal count error back to the agent.

        Args:
            session: The current Codex session.
            position_changes: Number of position changes detected.
            total_bars: Total number of bars in the data.
        """
        logger.warning("-" * 60)
        logger.warning("FEEDING SIGNAL ERROR TO CODEX")
        logger.warning("-" * 60)
        logger.warning(f"Position changes: {position_changes} (need at least 5)")
        logger.warning(f"Total bars: {total_bars}")

        error_message = f"""The strategy code executed but generated too few position changes:
- Position changes: {position_changes}
- Total bars: {total_bars}

A useful trading strategy should generate multiple position changes throughout the data period (at least 5-10 position changes).

The problem is likely that your conditions are too strict or you're not properly using the state machine pattern.

CORRECT approach - generates multiple position changes:
```python
def generate_signals(data: dict, params: dict) -> dict[str, pd.Series]:
    prices = data['prices']
    close = prices['close']
    rsi = vbt.RSI.run(close, window=14).rsi

    # State machine: nan for no change, then ffill
    position = pd.Series(np.nan, index=prices.index, dtype=float)
    position[rsi < 30] = 1   # Go long when oversold
    position[rsi > 70] = 0   # Go flat when overbought
    position = position.ffill().fillna(0)

    return {{"prices": position}}
```

WRONG approach - conditions that rarely trigger:
```python
position[rsi < 5] = 1   # Too strict, rarely triggers
position[rsi > 95] = 0  # Too strict, rarely triggers
```

Please adjust your strategy to use conditions that will generate more position changes.

Output valid JSON with "data_schema", "param_schema", and "code" fields."""

        session.messages.append({
            "role": "user",
            "content": error_message,
        })

    def validate_code(self, code: str) -> tuple[bool, Optional[str]]:
        """Validate generated code for safety and correctness.

        Returns:
            Tuple of (is_valid, error_message).
        """
        logger.info("-" * 60)
        logger.info("VALIDATING GENERATED CODE")
        logger.info("-" * 60)

        if not code or not code.strip():
            logger.error("Validation failed: Empty code generated")
            return False, "Empty code generated"

        # Check for dangerous patterns
        for pattern in DANGEROUS_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                logger.error(f"Validation failed: Dangerous pattern detected: {pattern}")
                return False, f"Dangerous pattern detected: {pattern}"
        logger.info("Security check passed: No dangerous patterns")

        # Check syntax
        try:
            ast.parse(code)
            logger.info("Syntax check passed: Code is valid Python")
        except SyntaxError as e:
            logger.error(f"Validation failed: Syntax error on line {e.lineno}: {e.msg}")
            return False, f"Syntax error on line {e.lineno}: {e.msg}"

        # Check for required function
        if 'def generate_signals' not in code:
            logger.error("Validation failed: Missing required function 'generate_signals'")
            return False, "Missing required function 'generate_signals'"
        logger.info("Function check passed: 'generate_signals' found")

        # Check function signature - must accept data and params
        if 'data' not in code:
            logger.error("Validation failed: Function must accept 'data' parameter")
            return False, "Function must accept 'data' parameter"
        if 'params' not in code:
            logger.error("Validation failed: Function must accept 'params' parameter")
            return False, "Function must accept 'params' parameter"
        logger.info("Signature check passed: 'data' and 'params' parameters found")

        # Check for return statement
        if 'return' not in code:
            logger.error("Validation failed: Function must have a return statement")
            return False, "Function must have a return statement"

        logger.info("Validation PASSED: Code is valid")
        return True, None

    def validate_param_schema(self, param_schema: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate param_schema structure.

        Args:
            param_schema: The param_schema dict from LLM output.

        Returns:
            Tuple of (is_valid, error_message).
        """
        if not isinstance(param_schema, dict):
            return False, "param_schema must be a dictionary"

        valid_types = {"int", "float", "bool", "enum"}

        for name, definition in param_schema.items():
            if not isinstance(definition, dict):
                return False, f"Parameter '{name}' definition must be a dictionary"

            # Check required fields
            if "type" not in definition:
                return False, f"Parameter '{name}' missing 'type' field"
            if "default" not in definition:
                return False, f"Parameter '{name}' missing 'default' field"

            # Validate type
            param_type = definition.get("type")
            if param_type not in valid_types:
                return False, f"Parameter '{name}' has invalid type '{param_type}', must be one of {valid_types}"

            # Validate enum has values
            if param_type == "enum" and "values" not in definition:
                return False, f"Enum parameter '{name}' missing 'values' field"

        return True, None

    def validate_data_schema(self, data_schema: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate data_schema structure.

        Args:
            data_schema: The data_schema dict from LLM output.

        Returns:
            Tuple of (is_valid, error_message).
        """
        if not isinstance(data_schema, dict):
            return False, "data_schema must be a dictionary"

        if not data_schema:
            return False, "data_schema cannot be empty"

        valid_data_types = {"ohlcv", "fundamental"}
        valid_frequencies = {"1Min", "5Min", "15Min", "1Hour", "1Day", "1Week", "1Month", "quarterly"}

        for slot_name, definition in data_schema.items():
            if not isinstance(definition, dict):
                return False, f"Data slot '{slot_name}' definition must be a dictionary"

            # Check required fields
            if "frequency" not in definition:
                return False, f"Data slot '{slot_name}' missing 'frequency' field"
            if "columns" not in definition:
                return False, f"Data slot '{slot_name}' missing 'columns' field"

            # Validate frequency
            freq = definition.get("frequency")
            if freq not in valid_frequencies:
                return False, f"Data slot '{slot_name}' has invalid frequency '{freq}'"

            # Validate columns is a list
            columns = definition.get("columns")
            if not isinstance(columns, list):
                return False, f"Data slot '{slot_name}' columns must be a list"

            # Validate data_type if present
            data_type = definition.get("data_type", "ohlcv")
            if data_type not in valid_data_types:
                return False, f"Data slot '{slot_name}' has invalid data_type '{data_type}'"

        return True, None

    def _extract_code(self, raw_response: str) -> str:
        """Extract Python code from the model response."""
        # Try to extract from markdown code block
        code_block_pattern = r'```(?:python)?\s*\n(.*?)```'
        matches = re.findall(code_block_pattern, raw_response, re.DOTALL)
        if matches:
            return matches[0].strip()

        # If no code block, look for function definition
        lines = raw_response.strip().split('\n')
        code_lines = []
        in_function = False
        indent_level = 0

        for line in lines:
            if line.strip().startswith('def generate_signals'):
                in_function = True
                indent_level = len(line) - len(line.lstrip())

            if in_function:
                # Check if we've exited the function (non-empty line with less indentation)
                if line.strip() and not line.startswith(' ' * (indent_level + 1)) and not line.strip().startswith('def'):
                    if code_lines:  # We have some code already
                        current_indent = len(line) - len(line.lstrip())
                        if current_indent <= indent_level and not line.strip().startswith('#'):
                            break
                code_lines.append(line)

        if code_lines:
            return '\n'.join(code_lines)

        return raw_response.strip()


def get_codex_agent(
    api_key: Optional[str] = None,
    model: str = "gpt-5.1",
    max_attempts: int = 5,
) -> CodexAgent:
    """Factory function to create a CodexAgent instance."""
    return CodexAgent(api_key=api_key, model=model, max_attempts=max_attempts)
