"""
Experiment Runner
=================

Runs a single experiment: generates code via OpenAI agent, verifies it, saves results.

Install dependencies:
    pip install openai-agents
"""

import asyncio
import importlib
import importlib.util
import json
import os
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

# Load environment variables from .env file
EXPERIMENT_DIR = Path(__file__).parent
load_dotenv(EXPERIMENT_DIR / ".env")

# Add current directory to path for imports (local modules)
sys.path.insert(0, str(EXPERIMENT_DIR))

# Import agents (may not be installed)
try:
    from agents import Agent, Runner, RunConfig, function_tool, RunHooks, ModelSettings
    from agents.models.openai_provider import OpenAIProvider
    from openai.types.shared import Reasoning
    HAS_AGENTS = True
except ImportError:
    HAS_AGENTS = False
    Agent = None  # type: ignore
    Runner = None  # type: ignore
    RunConfig = None  # type: ignore
    OpenAIProvider = None  # type: ignore
    function_tool = None  # type: ignore
    RunHooks = None  # type: ignore
    ModelSettings = None  # type: ignore
    Reasoning = None  # type: ignore

# Configuration from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

from common import VerificationResult
from verify import CONDITION_MODULES
from backtests import BACKTEST_FUNCTIONS, BacktestResult


# =============================================================================
# PATHS
# =============================================================================

RESULTS_DIR = EXPERIMENT_DIR / "results"

# Ensure base results directory exists
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_experiment_dir(condition: str, complexity: str, run_number: int) -> Path:
    """
    Get the directory for an experiment run.

    Args:
        condition: Condition ID (c0-c7)
        complexity: Complexity level (simple, medium, complex)
        run_number: Run number

    Returns:
        Path to experiment directory (e.g., results/c0_simple_1/)
    """
    dir_name = f"{condition.lower()}_{complexity}_{run_number}"
    return RESULTS_DIR / dir_name


# =============================================================================
# PROMPT LOADING
# =============================================================================

def load_prompt(condition: str, complexity: str) -> dict[str, Any]:
    """
    Load a prompt from the prompts module.

    Args:
        condition: Condition ID (e.g., "c0", "c1", ..., "c7")
        complexity: Complexity level ("simple", "medium", "complex")

    Returns:
        Dict with 'name', 'interface', 'prompt' keys
    """
    condition_lower = condition.lower()
    module_name = f"prompts.{condition_lower}_control" if condition_lower == "c0" else f"prompts.{condition_lower}_{'schema' if condition_lower == 'c1' else 'docs' if condition_lower == 'c2' else 'tdd' if condition_lower == 'c3' else 'schema_docs' if condition_lower == 'c4' else 'schema_tdd' if condition_lower == 'c5' else 'docs_tdd' if condition_lower == 'c6' else 'all'}"

    # Map condition to module name
    CONDITION_TO_MODULE = {
        "c0": "c0_control",
        "c1": "c1_schema",
        "c2": "c2_docs",
        "c3": "c3_tdd",
        "c4": "c4_schema_docs",
        "c5": "c5_schema_tdd",
        "c6": "c6_docs_tdd",
        "c7": "c7_all",
    }

    module_suffix = CONDITION_TO_MODULE.get(condition_lower)
    if not module_suffix:
        raise ValueError(f"Unknown condition: {condition}. Expected c0-c7.")

    module_name = f"prompts.{module_suffix}"
    module = importlib.import_module(module_name)

    if complexity not in module.PROMPTS:
        raise ValueError(f"Unknown complexity: {complexity}. Expected simple, medium, or complex.")

    return module.PROMPTS[complexity]


# =============================================================================
# AGENT SETUP
# =============================================================================

def get_openai_provider() -> "OpenAIProvider":
    """Create an OpenAI provider with configured endpoint and API key."""
    if not HAS_AGENTS:
        raise ImportError(
            "openai-agents package not installed. "
            "Install with: pip install openai-agents"
        )

    return OpenAIProvider(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_ENDPOINT,
    )


# =============================================================================
# EXPERIMENT CONTEXT (for tool access)
# =============================================================================

class ExperimentContext:
    """Holds state for a single experiment run, accessible by tools."""

    def __init__(
        self,
        condition: str,
        complexity: str,
        run_number: int,
        skip_backtest: bool = False,
    ):
        self.condition = condition
        self.complexity = complexity
        self.run_number = run_number
        self.skip_backtest = skip_backtest

        # Results tracking
        self.code_path: Optional[Path] = None
        self.verify_results: dict[str, VerificationResult] = {}
        self.backtest_result: Optional[BacktestResult] = None
        self.submission_count = 0
        self.all_passed = False

        # Token usage tracking (accumulated via hooks)
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0


# Global context for current experiment (set before running agent)
_current_context: Optional[ExperimentContext] = None


class TokenTrackingHooks(RunHooks):
    """Hooks to track token usage during agent runs, even when max_turns is exceeded."""

    async def on_llm_end(self, context, agent, response) -> None:
        """Called after each LLM response. Accumulate token usage."""
        global _current_context
        if _current_context is None:
            return

        # Extract usage from response
        if hasattr(response, 'usage') and response.usage:
            usage = response.usage
            _current_context.input_tokens += getattr(usage, 'input_tokens', 0) or 0
            _current_context.output_tokens += getattr(usage, 'output_tokens', 0) or 0
            _current_context.total_tokens += getattr(usage, 'total_tokens', 0) or 0


def _run_verification_internal(code_path: Path, condition: str, complexity: str) -> dict[str, VerificationResult]:
    """Internal verification runner."""
    condition_upper = condition.upper()
    verify_module = CONDITION_MODULES.get(condition_upper)

    if not verify_module:
        raise ValueError(f"No verify module for condition: {condition}")

    with open(code_path, "r") as f:
        source_code = f.read()

    try:
        spec = importlib.util.spec_from_file_location("generated_module", code_path)
        generated_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generated_module)
    except Exception as e:
        result = VerificationResult(passed=False)
        result.add_error(f"Failed to import generated code: {e}\n{traceback.format_exc()}")
        return {"schema": result, "documentation": result, "tests": result}

    # Load test data for TDD verification
    from backtests.shared import load_sample_data
    sample_data = load_sample_data(complexity)

    # Format test data based on complexity
    if complexity == "simple":
        test_data = sample_data["ohlcv"]
    elif complexity == "medium":
        test_data = sample_data["ohlcv"]
    elif complexity == "complex":
        test_data = (sample_data["asset_a"], sample_data["asset_b"])
    else:
        test_data = None

    return verify_module.run_verification(
        strategy=complexity,
        generated_module=generated_module,
        source_code=source_code,
        test_data=test_data,
    )


def _load_strategy_module(code_path: Path) -> Any:
    """Load a strategy module from a code file."""
    spec = importlib.util.spec_from_file_location("strategy_module", code_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {code_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Required functions for each complexity level
REQUIRED_FUNCTIONS = {
    "simple": ["generate_signals"],
    "medium": ["compute_indicators", "order_func"],
    "complex": ["compute_spread_indicators", "order_func"],
}


def _run_backtest_internal(code_path: Path, complexity: str) -> BacktestResult:
    """Internal backtest runner."""
    result = BacktestResult()

    # Check complexity is valid
    backtest_func = BACKTEST_FUNCTIONS.get(complexity)
    if not backtest_func:
        result.error = f"No backtest function for complexity: {complexity}"
        return result

    # Load strategy module
    try:
        module = _load_strategy_module(code_path)
    except Exception as e:
        result.error = f"Failed to import strategy module: {e}\n{traceback.format_exc()}"
        return result

    # Check for required functions
    required_funcs = REQUIRED_FUNCTIONS.get(complexity, [])
    for func_name in required_funcs:
        if not hasattr(module, func_name):
            result.error = f"Module missing '{func_name}' function"
            return result

    # Call the appropriate backtest function with extracted functions
    if complexity == "simple":
        return backtest_func(module.generate_signals)
    elif complexity == "medium":
        return backtest_func(module.compute_indicators, module.order_func)
    elif complexity == "complex":
        return backtest_func(module.compute_spread_indicators, module.order_func)
    else:
        result.error = f"Unknown complexity: {complexity}"
        return result


def _log_status(msg: str) -> None:
    """Print a timestamped status message."""
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"    [{timestamp}] {msg}")


def _simplify_error(error: str) -> str:
    """Extract first line of error, removing traceback."""
    if not error:
        return error
    # Get just the first line or up to the traceback
    lines = error.split('\n')
    first_line = lines[0].strip()
    # If it's a traceback header, try to get the actual error
    if first_line.startswith('Traceback'):
        # Find the last line which is usually the actual error
        for line in reversed(lines):
            line = line.strip()
            if line and not line.startswith('File ') and not line.startswith('Traceback'):
                return line
    return first_line


@function_tool
def submit_code(code: str) -> str:
    """
    Submit generated code for verification and backtesting.

    Call this function with your complete Python code to test it.
    The code will be saved, verified against schema/documentation requirements,
    and backtested. You will receive feedback about any errors.

    Args:
        code: The complete Python code to submit

    Returns:
        Feedback string describing verification and backtest results
    """
    global _current_context
    if _current_context is None:
        return "Error: No experiment context set"

    ctx = _current_context
    ctx.submission_count += 1

    _log_status(f"Submission #{ctx.submission_count}")

    # Save code
    exp_dir = get_experiment_dir(ctx.condition, ctx.complexity, ctx.run_number)
    exp_dir.mkdir(parents=True, exist_ok=True)
    code_path = exp_dir / "code.py"

    with open(code_path, "w") as f:
        f.write(code)

    ctx.code_path = code_path

    # Run verification
    try:
        ctx.verify_results = _run_verification_internal(code_path, ctx.condition, ctx.complexity)
        verification_success = True
        verification_error = None
    except Exception as e:
        verification_success = False
        verification_error = str(e).split('\n')[0]  # First line only
        ctx.verify_results = {}

    # Log verification results
    if not verification_success:
        _log_status(f"  Verification error: {verification_error}")
    else:
        for name, result in ctx.verify_results.items():
            status = "PASS" if result.passed else "FAIL"
            if result.passed:
                _log_status(f"  {name}: {status}")
            else:
                err_summary = result.errors[0] if result.errors else "unknown error"
                err_summary = _simplify_error(err_summary)
                num_errors = len(result.errors)
                if num_errors > 1:
                    _log_status(f"  {name}: {status} ({num_errors} errors, first: {err_summary})")
                else:
                    _log_status(f"  {name}: {status} ({err_summary})")

    # Run backtest only if ALL verifications passed (including tests)
    ctx.backtest_result = None
    all_verifications_passed = verification_success and all(
        r.passed for r in ctx.verify_results.values()
    ) if ctx.verify_results else False

    if not ctx.skip_backtest and all_verifications_passed:
        try:
            ctx.backtest_result = _run_backtest_internal(code_path, ctx.complexity)
        except Exception as e:
            ctx.backtest_result = BacktestResult()
            ctx.backtest_result.error = f"Backtest failed: {str(e).split(chr(10))[0]}"

    # Log backtest result
    if ctx.backtest_result:
        if ctx.backtest_result.success:
            metrics = []
            if ctx.backtest_result.total_return is not None:
                metrics.append(f"return={ctx.backtest_result.total_return:.2%}")
            if ctx.backtest_result.sharpe_ratio is not None:
                metrics.append(f"sharpe={ctx.backtest_result.sharpe_ratio:.2f}")
            if ctx.backtest_result.total_trades is not None:
                metrics.append(f"trades={ctx.backtest_result.total_trades}")
            _log_status(f"  backtest: PASS ({', '.join(metrics)})")
        else:
            err_summary = _simplify_error(ctx.backtest_result.error)
            _log_status(f"  backtest: FAIL ({err_summary})")

    # Check if all passed
    verification_passed = all(
        r.passed for name, r in ctx.verify_results.items() if name != "tests"
    ) if ctx.verify_results else False

    backtest_passed = (
        ctx.backtest_result.success if ctx.backtest_result else ctx.skip_backtest
    )

    ctx.all_passed = verification_passed and backtest_passed

    # Format feedback
    lines = [f"## Submission #{ctx.submission_count} Results\n"]

    if not verification_success:
        lines.append(f"### Verification Error: {verification_error}")
        return "\n".join(lines)

    for name, result in ctx.verify_results.items():
        status = "PASS" if result.passed else "FAIL"
        lines.append(f"### {name.title()}: {status}")
        if result.errors:
            lines.append("Errors:")
            for err in result.errors:
                lines.append(f"  - {err}")
        if result.warnings:
            lines.append("Warnings:")
            for warn in result.warnings:
                lines.append(f"  - {warn}")
        lines.append("")

    if ctx.backtest_result:
        if ctx.backtest_result.success:
            lines.append("### Backtest: PASS")
            if ctx.backtest_result.total_return is not None:
                lines.append(f"  - Total Return: {ctx.backtest_result.total_return:.2%}")
            if ctx.backtest_result.sharpe_ratio is not None:
                lines.append(f"  - Sharpe Ratio: {ctx.backtest_result.sharpe_ratio:.2f}")
            # Trade statistics
            if ctx.backtest_result.total_trades is not None:
                lines.append(f"  - Total Trades: {ctx.backtest_result.total_trades}")
            if ctx.backtest_result.win_rate is not None:
                lines.append(f"  - Win Rate: {ctx.backtest_result.win_rate:.2%}")
            if ctx.backtest_result.profit_factor is not None:
                lines.append(f"  - Profit Factor: {ctx.backtest_result.profit_factor:.2f}")
            if ctx.backtest_result.avg_trade_return is not None:
                lines.append(f"  - Avg Trade Return: {ctx.backtest_result.avg_trade_return:.2%}")
        else:
            lines.append("### Backtest: FAIL")
            lines.append(f"  - Error: {ctx.backtest_result.error}")

    if ctx.all_passed:
        lines.append("\n**All checks passed! Your code is complete.**")
    else:
        lines.append("\n**Please fix the errors above and call submit_code again with corrected code.**")

    return "\n".join(lines)


def create_code_generation_agent(condition: str, complexity: str) -> "Agent":
    """
    Create an OpenAI agent for code generation with tools.

    Args:
        condition: Condition ID (c0-c7)
        complexity: Complexity level (simple, medium, complex)

    Returns:
        Configured Agent instance with submit_code tool

    Raises:
        ImportError: If openai-agents package is not installed
    """
    if not HAS_AGENTS:
        raise ImportError(
            "openai-agents package not installed. "
            "Install with: pip install openai-agents"
        )

    return Agent(
        name=f"CodeGenerator_{condition}_{complexity}",
        instructions="""You are an expert Python developer specializing in quantitative finance and algorithmic trading.

Your task is to generate clean, well-documented Python code for trading strategies using vectorbt.

WORKFLOW:
1. Read the strategy requirements carefully
2. Generate complete Python code
3. Call the submit_code tool with your code to test it
4. If there are errors, fix them and submit again
5. Repeat until all checks pass

IMPORTANT RULES:
1. Include all necessary imports at the top
2. Follow the exact function signatures provided in the prompt
3. Use type hints for all function parameters and return values
4. Handle edge cases (NaN values, warmup periods, etc.)
5. Do NOT use any APIs or functions not specified in the prompt (if VAS is provided)
6. Always call submit_code to validate your code - do not just output code without testing it""",
        model=OPENAI_MODEL,
        model_settings=ModelSettings(
            reasoning=Reasoning(effort="high")
        ),
        tools=[submit_code],
    )


# =============================================================================
# RUN NUMBER TRACKING
# =============================================================================

def get_next_run_number(condition: str, complexity: str) -> int:
    """
    Get the next available run number for a condition/complexity.

    Args:
        condition: Condition ID (c0-c7)
        complexity: Complexity level (simple, medium, complex)

    Returns:
        Next available run number (starts at 1)
    """
    # Look for existing experiment folders like c0_simple_1, c0_simple_2, etc.
    pattern = f"{condition.lower()}_{complexity}_*"
    existing_dirs = [d for d in RESULTS_DIR.glob(pattern) if d.is_dir()]

    if not existing_dirs:
        return 1

    # Extract numbers from existing folders
    numbers = []
    for d in existing_dirs:
        match = re.search(rf'{condition.lower()}_{complexity}_(\d+)$', d.name)
        if match:
            numbers.append(int(match.group(1)))

    if not numbers:
        return 1

    return max(numbers) + 1


def extract_run_number_from_path(path: Path) -> Optional[int]:
    """
    Extract run number from an experiment path or folder name.

    Args:
        path: Path to experiment folder (e.g., results/c1_simple_1/) or folder name

    Returns:
        Run number or None if not found
    """
    # Pattern: {condition}_{complexity}_{number}
    name = path.name if isinstance(path, Path) else str(path)
    match = re.search(r'_(\d+)$', name)
    if match:
        return int(match.group(1))
    return None


# =============================================================================
# VERIFICATION (standalone mode)
# =============================================================================

def run_verification(
    code_path: Path,
    condition: str,
    complexity: str,
) -> dict[str, VerificationResult]:
    """
    Run verification on generated code.

    Args:
        code_path: Path to the generated code file
        condition: Condition ID (c0-c7)
        complexity: Complexity level (simple, medium, complex)

    Returns:
        Dict mapping verification type to VerificationResult
    """
    return _run_verification_internal(code_path, condition, complexity)


# =============================================================================
# BACKTEST
# =============================================================================

def run_backtest(
    code_path: Path,
    complexity: str,
) -> BacktestResult:
    """
    Run backtest on generated code.

    Args:
        code_path: Path to the generated code file
        complexity: Complexity level (simple, medium, complex)

    Returns:
        BacktestResult with metrics and raw data
    """
    return _run_backtest_internal(code_path, complexity)


def save_experiment_results(
    condition: str,
    complexity: str,
    run_number: int,
    results_data: dict[str, Any],
) -> Path:
    """
    Save all experiment results to a single JSON file.

    Args:
        condition: Condition ID
        complexity: Complexity level
        run_number: Run number
        results_data: Complete results dictionary

    Returns:
        Path to the saved results file (e.g., results/c0_simple_1/results.json)
    """
    exp_dir = get_experiment_dir(condition, complexity, run_number)
    exp_dir.mkdir(parents=True, exist_ok=True)

    filepath = exp_dir / "results.json"

    with open(filepath, "w") as f:
        json.dump(results_data, f, indent=2)

    return filepath


# =============================================================================
# MAIN EXPERIMENT RUNNER
# =============================================================================

async def run_experiment_async(
    condition: str,
    complexity: str,
    verbose: bool = True,
    skip_backtest: bool = False,
    max_turns: int = 10,
    run_number: Optional[int] = None,
) -> dict[str, Any]:
    """
    Run a single experiment asynchronously with agentic refinement.

    The agent uses the submit_code tool to test code, and the SDK's max_turns
    parameter controls how many iterations the agent can make.

    Args:
        condition: Condition ID (c0-c7)
        complexity: Complexity level (simple, medium, complex)
        verbose: Whether to print progress
        skip_backtest: Whether to skip running the backtest
        max_turns: Maximum number of agent turns (default: 10)
        run_number: Specific run number to use (auto-increments if None)

    Returns:
        Dict with experiment results
    """
    global _current_context

    # Get the next run number for this condition/complexity (or use provided)
    if run_number is None:
        run_number = get_next_run_number(condition, complexity)

    if verbose:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting experiment: {condition} / {complexity} (run #{run_number})")
        print(f"  Max turns: {max_turns}")

    # Load prompt
    if verbose:
        print(f"  Loading prompt...")
    prompt_data = load_prompt(condition, complexity)

    # Create agent with tools
    if verbose:
        print(f"  Creating agent...")
    agent = create_code_generation_agent(condition, complexity)

    # Set up experiment context for the submit_code tool
    _current_context = ExperimentContext(
        condition=condition,
        complexity=complexity,
        run_number=run_number,
        skip_backtest=skip_backtest,
    )

    # Run agent with SDK's max_turns
    if verbose:
        print(f"  Running agent...")

    provider = get_openai_provider()
    run_config = RunConfig(model_provider=provider)

    generation_error = None
    try:
        run_result = await Runner.run(
            agent,
            input=prompt_data["prompt"],
            run_config=run_config,
            max_turns=max_turns,
            hooks=TokenTrackingHooks(),
        )
    except Exception as e:
        generation_error = str(e)
        run_result = None

    # Get token usage from context (tracked via hooks, works even when max_turns exceeded)
    total_tokens = {
        "input_tokens": _current_context.input_tokens,
        "output_tokens": _current_context.output_tokens,
        "total_tokens": _current_context.total_tokens,
    }

    # Get results from context
    ctx = _current_context
    code_path = ctx.code_path
    verify_results = ctx.verify_results
    backtest_result = ctx.backtest_result
    submission_count = ctx.submission_count
    all_passed = ctx.all_passed

    # Handle case where no code was ever submitted
    if code_path is None:
        exp_dir = get_experiment_dir(condition, complexity, run_number)
        exp_dir.mkdir(parents=True, exist_ok=True)
        code_path = exp_dir / "code.py"
        with open(code_path, "w") as f:
            f.write(f"# Generation failed: {generation_error or 'No code submitted'}\n")

    # Determine success
    generation_success = submission_count > 0
    verification_success = all(
        r.passed for name, r in verify_results.items() if name != "tests"
    ) if verify_results else False

    # Compile results
    result = {
        "condition": condition,
        "complexity": complexity,
        "run_number": run_number,
        "prompt_name": prompt_data["name"],
        "prompt_interface": prompt_data["interface"],
        "max_turns": max_turns,
        "submissions": submission_count,
        "total_tokens": total_tokens,
        "generation": {
            "success": generation_success,
            "error": generation_error,
        },
        "verification": {
            "success": verification_success,
            "error": None,
            "results": {
                k: {"passed": v.passed, "errors": v.errors}
                for k, v in verify_results.items()
            } if verify_results else {},
        },
        "backtest": {
            "success": backtest_result.success if backtest_result else False,
            "error": backtest_result.error if backtest_result else ("Skipped" if skip_backtest else "No successful verification"),
            "metrics": {
                "total_return": backtest_result.total_return if backtest_result else None,
                "sharpe_ratio": backtest_result.sharpe_ratio if backtest_result else None,
                "max_drawdown": backtest_result.max_drawdown if backtest_result else None,
                "total_trades": backtest_result.total_trades if backtest_result else None,
            } if backtest_result else {},
        },
    }

    # Save combined results to single JSON file
    save_experiment_results(condition, complexity, run_number, result)

    # Print final summary
    if verbose:
        status = "PASSED" if all_passed else "FAILED"
        print(f"  Done: {status} ({submission_count} submissions, {total_tokens['total_tokens']} tokens)")
        print()

    # Clear context
    _current_context = None

    return result


def run_experiment(
    condition: str,
    complexity: str,
    verbose: bool = True,
    skip_backtest: bool = False,
    max_turns: int = 10,
    run_number: Optional[int] = None,
) -> dict[str, Any]:
    """
    Run a single experiment synchronously.

    Args:
        condition: Condition ID (c0-c7)
        complexity: Complexity level (simple, medium, complex)
        verbose: Whether to print progress
        skip_backtest: Whether to skip running the backtest
        max_turns: Maximum number of agent turns (default: 10)
        run_number: Specific run number to use (auto-increments if None)

    Returns:
        Dict with experiment results
    """
    return asyncio.run(run_experiment_async(
        condition, complexity, verbose, skip_backtest, max_turns, run_number
    ))


# =============================================================================
# VERIFY-ONLY MODE
# =============================================================================

def run_verify_only(
    condition: str,
    complexity: str,
    run_number: int,
    verbose: bool = True,
    skip_backtest: bool = False,
) -> dict[str, Any]:
    """
    Run verification on an existing code file without regenerating.

    Args:
        condition: Condition ID (c0-c7)
        complexity: Complexity level (simple, medium, complex)
        run_number: Run number to verify
        verbose: Whether to print progress
        skip_backtest: Whether to skip running the backtest

    Returns:
        Dict with verification and backtest results
    """
    # Get the experiment directory and code file
    exp_dir = get_experiment_dir(condition, complexity, run_number)
    code_file = exp_dir / "code.py"

    if verbose:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Verifying existing code: {code_file}")
        print(f"  Condition: {condition}, Complexity: {complexity}, Run: #{run_number}")

    if not code_file.exists():
        raise FileNotFoundError(f"Code file not found: {code_file}")

    # Run verification
    if verbose:
        print(f"  Running verification...")
    try:
        verify_results = run_verification(code_file, condition, complexity)
        verification_success = True
        verification_error = None
    except Exception as e:
        verification_success = False
        verification_error = str(e)
        verify_results = {}

    # Run backtest
    backtest_result = None
    if not skip_backtest:
        if verbose:
            print(f"  Running backtest...")
        try:
            backtest_result = run_backtest(code_file, complexity)
        except Exception as e:
            backtest_result = BacktestResult()
            backtest_result.error = f"Backtest failed: {e}"
    else:
        if verbose:
            print(f"  Skipping backtest...")

    # Compile results
    result = {
        "condition": condition,
        "complexity": complexity,
        "run_number": run_number,
        "mode": "verify-only",
        "generation": {
            "success": True,  # Code already exists
            "error": None,
        },
        "verification": {
            "success": verification_success,
            "error": verification_error,
            "results": {
                k: {"passed": v.passed, "errors": v.errors}
                for k, v in verify_results.items()
            } if verify_results else {},
        },
        "backtest": {
            "success": backtest_result.success if backtest_result else False,
            "error": backtest_result.error if backtest_result else "Skipped",
            "metrics": {
                "total_return": backtest_result.total_return if backtest_result else None,
                "sharpe_ratio": backtest_result.sharpe_ratio if backtest_result else None,
                "max_drawdown": backtest_result.max_drawdown if backtest_result else None,
                "total_trades": backtest_result.total_trades if backtest_result else None,
            } if backtest_result else {},
        },
    }

    # Save combined results
    if verbose:
        print(f"  Saving results...")
    save_experiment_results(condition, complexity, run_number, result)

    # Print summary
    if verbose:
        status = "PASSED" if (verification_success and
                             all(r.passed for r in verify_results.values())) else "FAILED"
        print(f"  Result: {status}")
        if verify_results:
            for name, res in verify_results.items():
                print(f"    {name}: {'PASS' if res.passed else 'FAIL'}")
                if res.errors:
                    for err in res.errors:
                        print(f"      - {err}")
        if backtest_result and backtest_result.success:
            print(f"    backtest: PASS (return={backtest_result.total_return:.2%}, sharpe={backtest_result.sharpe_ratio:.2f})" if backtest_result.total_return else "    backtest: PASS")
        elif backtest_result:
            print(f"    backtest: FAIL ({backtest_result.error})")
        print()

    return result


def find_latest_code_file(condition: str, complexity: str) -> Optional[Path]:
    """Find the highest numbered code file for a condition/complexity."""
    # Look for experiment folders like c0_simple_1, c0_simple_2, etc.
    pattern = f"{condition.lower()}_{complexity}_*"
    exp_dirs = [d for d in RESULTS_DIR.glob(pattern) if d.is_dir()]

    if not exp_dirs:
        return None

    # Find the folder with the highest run number
    best_dir = None
    best_number = -1
    for d in exp_dirs:
        num = extract_run_number_from_path(d)
        if num is not None and num > best_number:
            best_number = num
            best_dir = d

    if best_dir is None:
        return None

    # Return the code.py file from that folder
    code_file = best_dir / "code.py"
    return code_file if code_file.exists() else None


# =============================================================================
# BACKTEST-ONLY MODE
# =============================================================================

def run_backtest_only(
    condition: str,
    complexity: str,
    run_number: int,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run backtest on an existing code file without verification.

    Args:
        condition: Condition ID (c0-c7)
        complexity: Complexity level (simple, medium, complex)
        run_number: Run number to backtest
        verbose: Whether to print progress

    Returns:
        Dict with backtest results
    """
    # Get the experiment directory and code file
    exp_dir = get_experiment_dir(condition, complexity, run_number)
    code_file = exp_dir / "code.py"

    if verbose:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Running backtest on: {code_file}")
        print(f"  Condition: {condition}, Complexity: {complexity}, Run: #{run_number}")

    if not code_file.exists():
        raise FileNotFoundError(f"Code file not found: {code_file}")

    # Run backtest
    backtest_result = None
    if verbose:
        print(f"  Running backtest...")
    try:
        backtest_result = run_backtest(code_file, complexity)
    except Exception as e:
        backtest_result = BacktestResult()
        backtest_result.error = f"Backtest failed: {e}"

    # Compile results
    result = {
        "condition": condition,
        "complexity": complexity,
        "run_number": run_number,
        "mode": "backtest-only",
        "generation": {
            "success": True,  # Code already exists
            "error": None,
        },
        "backtest": {
            "success": backtest_result.success if backtest_result else False,
            "error": backtest_result.error if backtest_result else "Unknown error",
            "metrics": {
                "total_return": backtest_result.total_return if backtest_result else None,
                "sharpe_ratio": backtest_result.sharpe_ratio if backtest_result else None,
                "max_drawdown": backtest_result.max_drawdown if backtest_result else None,
                "total_trades": backtest_result.total_trades if backtest_result else None,
            } if backtest_result else {},
        },
    }

    # Save combined results
    if verbose:
        print(f"  Saving results...")
    save_experiment_results(condition, complexity, run_number, result)

    # Print summary
    if verbose:
        if backtest_result and backtest_result.success:
            print(f"  Result: PASSED")
            if backtest_result.total_return is not None:
                print(f"    return: {backtest_result.total_return:.2%}")
            if backtest_result.sharpe_ratio is not None:
                print(f"    sharpe: {backtest_result.sharpe_ratio:.2f}")
            if backtest_result.max_drawdown is not None:
                print(f"    max_drawdown: {backtest_result.max_drawdown:.2%}")
            if backtest_result.total_trades is not None:
                print(f"    trades: {backtest_result.total_trades}")
        else:
            print(f"  Result: FAILED")
            if backtest_result and backtest_result.error:
                print(f"    error: {backtest_result.error}")
        print()

    return result


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run a code generation experiment")
    parser.add_argument(
        "condition",
        type=str,
        help="Condition ID (c0-c7)",
    )
    parser.add_argument(
        "complexity",
        type=str,
        choices=["simple", "medium", "complex"],
        help="Complexity level",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only run verification on existing code (skip generation)",
    )
    parser.add_argument(
        "--backtest-only",
        action="store_true",
        help="Only run backtest on existing code (skip generation and verification)",
    )
    parser.add_argument(
        "--run-number",
        type=int,
        help="Run number to use. For normal runs, overrides auto-increment. "
             "For --verify-only or --backtest-only, specifies which run to use "
             "(defaults to most recent if not specified).",
    )
    parser.add_argument(
        "--skip-backtest",
        action="store_true",
        help="Skip running the backtest after verification",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Maximum number of generation turns for agentic refinement (default: 10)",
    )

    args = parser.parse_args()

    # Validate mutually exclusive options
    if args.verify_only and args.backtest_only:
        print("Error: --verify-only and --backtest-only are mutually exclusive")
        sys.exit(1)

    if args.backtest_only:
        # Backtest-only mode
        if args.run_number is not None:
            run_number = args.run_number
            # Verify the run exists
            exp_dir = get_experiment_dir(args.condition, args.complexity, run_number)
            if not (exp_dir / "code.py").exists():
                print(f"Error: Run #{run_number} does not exist for {args.condition}/{args.complexity}")
                print(f"  Expected: {exp_dir / 'code.py'}")
                sys.exit(1)
        else:
            # Find the latest run
            code_file = find_latest_code_file(args.condition, args.complexity)
            if not code_file:
                print(f"Error: No existing runs found for {args.condition}/{args.complexity}")
                print(f"  Looked in: {RESULTS_DIR}")
                sys.exit(1)
            run_number = extract_run_number_from_path(code_file.parent)
            if run_number is None:
                print(f"Error: Could not determine run number from {code_file.parent}")
                sys.exit(1)
            assert run_number is not None  # for type checker

        result = run_backtest_only(
            condition=args.condition,
            complexity=args.complexity,
            run_number=run_number,
            verbose=not args.quiet,
        )
    elif args.verify_only:
        # Verify-only mode
        if args.run_number is not None:
            run_number = args.run_number
            # Verify the run exists
            exp_dir = get_experiment_dir(args.condition, args.complexity, run_number)
            if not (exp_dir / "code.py").exists():
                print(f"Error: Run #{run_number} does not exist for {args.condition}/{args.complexity}")
                print(f"  Expected: {exp_dir / 'code.py'}")
                sys.exit(1)
        else:
            # Find the latest run
            code_file = find_latest_code_file(args.condition, args.complexity)
            if not code_file:
                print(f"Error: No existing runs found for {args.condition}/{args.complexity}")
                print(f"  Looked in: {RESULTS_DIR}")
                sys.exit(1)
            run_number = extract_run_number_from_path(code_file.parent)
            if run_number is None:
                print(f"Error: Could not determine run number from {code_file.parent}")
                sys.exit(1)
            assert run_number is not None  # for type checker

        result = run_verify_only(
            condition=args.condition,
            complexity=args.complexity,
            run_number=run_number,
            verbose=not args.quiet,
            skip_backtest=args.skip_backtest,
        )
    else:
        # Normal mode: generate + verify + backtest
        result = run_experiment(
            condition=args.condition,
            complexity=args.complexity,
            verbose=not args.quiet,
            skip_backtest=args.skip_backtest,
            max_turns=args.max_turns,
            run_number=args.run_number,
        )

    # Print final result as JSON
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
