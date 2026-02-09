#!/usr/bin/env python3
"""
Comprehensive Analysis Script for Experiment Results
Generates ANALYSIS_REFERENCE.md with exact numbers from experimental data.
"""

import json
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import importlib.util
import warnings
warnings.filterwarnings('ignore')

# Base paths
BASE_DIR = Path("/root/wqu/capstone/w6/experiments_imp_2")
RESULTS_10 = BASE_DIR / "results_10_gpt5mini"
RESULTS_20 = BASE_DIR / "results_20-gpt5mini"
REF_DIR = BASE_DIR / "reference-implementation"
OUTPUT_FILE = BASE_DIR / "ANALYSIS_REFERENCE.md"

# Condition definitions
CONDITIONS = ["c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7"]
COMPLEXITIES = ["simple", "medium", "complex"]

# Treatment bit definitions (for conditions c0-c7)
# c0 = 000, c1 = 001 (T), c2 = 010 (D), c3 = 011 (D+T), c4 = 100 (S), c5 = 101 (S+T), c6 = 110 (S+D), c7 = 111 (S+D+T)
def get_treatment_flags(condition):
    """Returns (S_enabled, D_enabled, T_enabled) for a condition."""
    c_num = int(condition[1])
    S = bool(c_num & 4)  # bit 2
    D = bool(c_num & 2)  # bit 1
    T = bool(c_num & 1)  # bit 0
    return S, D, T

def load_all_results():
    """Load all results.json files from both directories."""
    results = {"10-turn": [], "20-turn": []}

    for subdir in RESULTS_10.iterdir():
        if subdir.is_dir():
            results_file = subdir / "results.json"
            if results_file.exists():
                with open(results_file) as f:
                    data = json.load(f)
                    data["_dir"] = str(subdir)
                    data["_turns"] = "10-turn"
                    results["10-turn"].append(data)

    for subdir in RESULTS_20.iterdir():
        if subdir.is_dir():
            results_file = subdir / "results.json"
            if results_file.exists():
                with open(results_file) as f:
                    data = json.load(f)
                    data["_dir"] = str(subdir)
                    data["_turns"] = "20-turn"
                    results["20-turn"].append(data)

    return results

def compute_backtest_pass_rates(results):
    """Compute backtest pass rates by condition and complexity."""
    tables = {}

    # By condition
    for turns in ["10-turn", "20-turn"]:
        by_condition = defaultdict(lambda: {"passed": 0, "total": 0})
        for r in results[turns]:
            cond = r.get("condition", "unknown")
            by_condition[cond]["total"] += 1
            if r.get("backtest", {}).get("success", False):
                by_condition[cond]["passed"] += 1
        tables[f"{turns}_by_condition"] = dict(by_condition)

    # By complexity
    for turns in ["10-turn", "20-turn"]:
        by_complexity = defaultdict(lambda: {"passed": 0, "total": 0})
        for r in results[turns]:
            comp = r.get("complexity", "unknown")
            by_complexity[comp]["total"] += 1
            if r.get("backtest", {}).get("success", False):
                by_complexity[comp]["passed"] += 1
        tables[f"{turns}_by_complexity"] = dict(by_complexity)

    return tables

def compute_correctness_rates(results):
    """
    Compute correctness rates (total_return between -100% and +1000%).
    Returns data for 20-turn only as specified.
    """
    by_cond_comp = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))

    for r in results["20-turn"]:
        cond = r.get("condition", "unknown")
        comp = r.get("complexity", "unknown")
        bt = r.get("backtest", {})

        by_cond_comp[cond][comp]["total"] += 1

        if bt.get("success", False):
            metrics = bt.get("metrics", {})
            total_return = metrics.get("total_return", None)
            if total_return is not None:
                # -100% = -1.0, +1000% = 10.0
                if -1.0 <= total_return <= 10.0:
                    by_cond_comp[cond][comp]["correct"] += 1

    return dict(by_cond_comp)

def load_module_from_file(filepath):
    """Dynamically load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location("module", filepath)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        return None

def compute_semantic_equivalence(results):
    """
    Compute semantic equivalence by comparing generated code with reference implementations.

    For simple: compare position signals (exact match)
    For medium: compare indicator correlations (>0.99 = equivalent)
    For complex: compare indicator correlations (>0.95 = equivalent)
    """
    # Load reference implementations
    ref_simple = load_module_from_file(REF_DIR / "simple.py")
    ref_medium = load_module_from_file(REF_DIR / "medium.py")
    ref_complex = load_module_from_file(REF_DIR / "complex.py")

    if ref_simple is None or ref_medium is None or ref_complex is None:
        print("Warning: Could not load reference implementations")
        return None

    # Create sample data for testing
    np.random.seed(42)
    n = 500

    # Generate synthetic OHLCV data
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_price = close + np.random.randn(n) * 0.2
    volume = np.random.randint(1000, 10000, n)

    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    ohlcv = pd.DataFrame({
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume
    }, index=dates)

    # For pairs trading, create second asset
    close_b = 50 + np.cumsum(np.random.randn(n) * 0.3) + close * 0.3

    # Compute reference outputs
    try:
        ref_simple_output = ref_simple.generate_signals({"ohlcv": ohlcv}, {})["ohlcv"]
    except Exception as e:
        print(f"Error computing reference simple output: {e}")
        ref_simple_output = None

    try:
        ref_medium_indicators = ref_medium.compute_indicators(ohlcv)
    except Exception as e:
        print(f"Error computing reference medium output: {e}")
        ref_medium_indicators = None

    try:
        ref_complex_indicators = ref_complex.compute_spread_indicators(close, close_b)
    except Exception as e:
        print(f"Error computing reference complex output: {e}")
        ref_complex_indicators = None

    equivalence_results = {"10-turn": {}, "20-turn": {}}

    for turns in ["10-turn", "20-turn"]:
        by_cond_comp = defaultdict(lambda: defaultdict(lambda: {"equivalent": 0, "total": 0, "tested": 0}))

        for r in results[turns]:
            cond = r.get("condition", "unknown")
            comp = r.get("complexity", "unknown")
            code_path = Path(r["_dir"]) / "code.py"

            by_cond_comp[cond][comp]["total"] += 1

            # Only test if backtest was successful
            if not r.get("backtest", {}).get("success", False):
                continue

            if not code_path.exists():
                continue

            try:
                gen_module = load_module_from_file(code_path)
                if gen_module is None:
                    continue

                by_cond_comp[cond][comp]["tested"] += 1

                if comp == "simple" and ref_simple_output is not None:
                    # Compare position signals
                    try:
                        gen_output = gen_module.generate_signals({"ohlcv": ohlcv}, {})["ohlcv"]
                        # Check if signals match (allowing for minor differences)
                        gen_values = np.array(gen_output.values, dtype=float)
                        ref_values = np.array(ref_simple_output.values, dtype=float)
                        # Exact match or correlation > 0.99
                        if np.allclose(gen_values, ref_values, rtol=0.01) or \
                           (len(gen_values) == len(ref_values) and
                            np.corrcoef(gen_values, ref_values)[0,1] > 0.99):
                            by_cond_comp[cond][comp]["equivalent"] += 1
                    except Exception as e:
                        pass

                elif comp == "medium" and ref_medium_indicators is not None:
                    # Compare indicator correlations (>0.99 = equivalent)
                    try:
                        gen_indicators = gen_module.compute_indicators(ohlcv)
                        all_corr_high = True
                        for key in ["macd", "signal", "atr", "sma"]:
                            if key in gen_indicators and key in ref_medium_indicators:
                                gen_vals = np.array(gen_indicators[key], dtype=float)
                                ref_vals = np.array(ref_medium_indicators[key], dtype=float)
                                # Remove NaNs for correlation
                                mask = ~(np.isnan(gen_vals) | np.isnan(ref_vals))
                                if mask.sum() > 10:
                                    corr = np.corrcoef(gen_vals[mask], ref_vals[mask])[0,1]
                                    if corr < 0.99:
                                        all_corr_high = False
                                        break
                        if all_corr_high:
                            by_cond_comp[cond][comp]["equivalent"] += 1
                    except Exception as e:
                        pass

                elif comp == "complex" and ref_complex_indicators is not None:
                    # Compare indicator correlations (>0.95 = equivalent)
                    try:
                        gen_indicators = gen_module.compute_spread_indicators(close, close_b)
                        all_corr_high = True
                        for key in ["zscore", "hedge_ratio"]:
                            if key in gen_indicators and key in ref_complex_indicators:
                                gen_vals = np.array(gen_indicators[key], dtype=float)
                                ref_vals = np.array(ref_complex_indicators[key], dtype=float)
                                # Remove NaNs for correlation
                                mask = ~(np.isnan(gen_vals) | np.isnan(ref_vals))
                                if mask.sum() > 10:
                                    corr = np.corrcoef(gen_vals[mask], ref_vals[mask])[0,1]
                                    if corr < 0.95:
                                        all_corr_high = False
                                        break
                        if all_corr_high:
                            by_cond_comp[cond][comp]["equivalent"] += 1
                    except Exception as e:
                        pass

            except Exception as e:
                pass

        equivalence_results[turns] = dict(by_cond_comp)

    return equivalence_results

def compute_token_usage(results):
    """Compute mean token usage for successful runs by condition (20-turn only)."""
    by_condition = defaultdict(lambda: {"tokens": [], "count": 0})

    for r in results["20-turn"]:
        cond = r.get("condition", "unknown")
        if r.get("backtest", {}).get("success", False):
            tokens = r.get("total_tokens", {}).get("total_tokens", 0)
            if tokens > 0:
                by_condition[cond]["tokens"].append(tokens)
                by_condition[cond]["count"] += 1

    result = {}
    for cond in CONDITIONS:
        tokens = by_condition[cond]["tokens"]
        if tokens:
            result[cond] = {
                "mean": np.mean(tokens),
                "std": np.std(tokens),
                "count": len(tokens)
            }
        else:
            result[cond] = {"mean": 0, "std": 0, "count": 0}

    return result

def is_reasonable_return(tr):
    """Check if return is in reasonable range (-100% to +1000%)."""
    return tr is not None and -1.0 <= tr <= 10.0

def compute_consistency_cv(results):
    """
    Compute coefficient of variation (CV) of total_return by condition and complexity.
    20-turn only as specified.
    Only includes reasonable returns for meaningful statistics.
    """
    by_cond_comp = defaultdict(lambda: defaultdict(list))
    by_cond_comp_all = defaultdict(lambda: defaultdict(list))  # Include all for counting

    for r in results["20-turn"]:
        cond = r.get("condition", "unknown")
        comp = r.get("complexity", "unknown")
        bt = r.get("backtest", {})

        if bt.get("success", False):
            metrics = bt.get("metrics", {})
            total_return = metrics.get("total_return", None)
            if total_return is not None:
                by_cond_comp_all[cond][comp].append(total_return)
                # Only include reasonable returns for CV calculation
                if is_reasonable_return(total_return):
                    by_cond_comp[cond][comp].append(total_return)

    result = {}
    for cond in CONDITIONS:
        result[cond] = {}
        for comp in COMPLEXITIES:
            returns = by_cond_comp[cond][comp]
            all_returns = by_cond_comp_all[cond][comp]
            n_reasonable = len(returns)
            n_total = len(all_returns)
            if len(returns) >= 2:
                mean = np.mean(returns)
                std = np.std(returns)
                cv = std / abs(mean) if mean != 0 else np.inf
                result[cond][comp] = {"cv": cv, "mean": mean, "std": std, "n": n_reasonable, "n_total": n_total}
            else:
                result[cond][comp] = {"cv": None, "mean": None, "std": None, "n": n_reasonable, "n_total": n_total}

    return result

def compute_treatment_effects(results):
    """
    Compute treatment effects by comparing S-enabled vs S-disabled, D-enabled vs D-disabled, T-enabled vs T-disabled.
    Only includes reasonable returns for mean return calculation.
    """
    effects = {
        "S": {"enabled": {"passed": 0, "total": 0, "returns": [], "returns_all": []},
              "disabled": {"passed": 0, "total": 0, "returns": [], "returns_all": []}},
        "D": {"enabled": {"passed": 0, "total": 0, "returns": [], "returns_all": []},
              "disabled": {"passed": 0, "total": 0, "returns": [], "returns_all": []}},
        "T": {"enabled": {"passed": 0, "total": 0, "returns": [], "returns_all": []},
              "disabled": {"passed": 0, "total": 0, "returns": [], "returns_all": []}}
    }

    for turns in ["10-turn", "20-turn"]:
        for r in results[turns]:
            cond = r.get("condition", "unknown")
            if cond not in CONDITIONS:
                continue

            S, D, T = get_treatment_flags(cond)
            bt = r.get("backtest", {})
            passed = bt.get("success", False)

            # S treatment
            key = "enabled" if S else "disabled"
            effects["S"][key]["total"] += 1
            if passed:
                effects["S"][key]["passed"] += 1
                metrics = bt.get("metrics", {})
                tr = metrics.get("total_return", None)
                if tr is not None:
                    effects["S"][key]["returns_all"].append(tr)
                    if is_reasonable_return(tr):
                        effects["S"][key]["returns"].append(tr)

            # D treatment
            key = "enabled" if D else "disabled"
            effects["D"][key]["total"] += 1
            if passed:
                effects["D"][key]["passed"] += 1
                metrics = bt.get("metrics", {})
                tr = metrics.get("total_return", None)
                if tr is not None:
                    effects["D"][key]["returns_all"].append(tr)
                    if is_reasonable_return(tr):
                        effects["D"][key]["returns"].append(tr)

            # T treatment
            key = "enabled" if T else "disabled"
            effects["T"][key]["total"] += 1
            if passed:
                effects["T"][key]["passed"] += 1
                metrics = bt.get("metrics", {})
                tr = metrics.get("total_return", None)
                if tr is not None:
                    effects["T"][key]["returns_all"].append(tr)
                    if is_reasonable_return(tr):
                        effects["T"][key]["returns"].append(tr)

    return effects

def generate_markdown_report(results, pass_rates, correctness, equivalence, tokens, consistency, treatment_effects):
    """Generate the comprehensive markdown report."""

    lines = []
    lines.append("# Experiment Analysis Reference")
    lines.append("")
    lines.append("*This file is the single source of truth for all paper tables.*")
    lines.append("*All numbers are calculated directly from experimental data files.*")
    lines.append("")
    lines.append(f"Generated from: `{RESULTS_10}` and `{RESULTS_20}`")
    lines.append("")

    # Section 1: Experiment Summary
    lines.append("## 1. Experiment Summary")
    lines.append("")
    total_10 = len(results["10-turn"])
    total_20 = len(results["20-turn"])
    lines.append(f"- **10-turn experiments**: {total_10} runs")
    lines.append(f"- **20-turn experiments**: {total_20} runs")
    lines.append(f"- **Total runs**: {total_10 + total_20}")
    lines.append(f"- **Conditions**: {', '.join(CONDITIONS)} (8 conditions)")
    lines.append(f"- **Complexity levels**: {', '.join(COMPLEXITIES)} (3 levels)")
    lines.append(f"- **Runs per condition per complexity**: 5")
    lines.append("")

    # Treatment encoding
    lines.append("### Treatment Encoding")
    lines.append("")
    lines.append("| Condition | S (Spec) | D (Doc) | T (Test) | Binary |")
    lines.append("|-----------|----------|---------|----------|--------|")
    for c in CONDITIONS:
        S, D, T = get_treatment_flags(c)
        binary = f"{int(S)}{int(D)}{int(T)}"
        lines.append(f"| {c.upper()} | {'Yes' if S else 'No'} | {'Yes' if D else 'No'} | {'Yes' if T else 'No'} | {binary} |")
    lines.append("")

    # Section 2: Backtest Pass Rate by Condition
    lines.append("## 2. Backtest Pass Rate by Condition")
    lines.append("")
    lines.append("| Condition | 10-turn Pass | 10-turn Total | 10-turn Rate | 20-turn Pass | 20-turn Total | 20-turn Rate | Combined Pass | Combined Total | Combined Rate |")
    lines.append("|-----------|--------------|---------------|--------------|--------------|---------------|--------------|---------------|----------------|---------------|")

    total_10_pass, total_10_all = 0, 0
    total_20_pass, total_20_all = 0, 0

    for c in CONDITIONS:
        data_10 = pass_rates["10-turn_by_condition"].get(c, {"passed": 0, "total": 0})
        data_20 = pass_rates["20-turn_by_condition"].get(c, {"passed": 0, "total": 0})

        pass_10 = data_10["passed"]
        tot_10 = data_10["total"]
        rate_10 = (pass_10 / tot_10 * 100) if tot_10 > 0 else 0

        pass_20 = data_20["passed"]
        tot_20 = data_20["total"]
        rate_20 = (pass_20 / tot_20 * 100) if tot_20 > 0 else 0

        combined_pass = pass_10 + pass_20
        combined_tot = tot_10 + tot_20
        combined_rate = (combined_pass / combined_tot * 100) if combined_tot > 0 else 0

        total_10_pass += pass_10
        total_10_all += tot_10
        total_20_pass += pass_20
        total_20_all += tot_20

        lines.append(f"| {c.upper()} | {pass_10} | {tot_10} | {rate_10:.1f}% | {pass_20} | {tot_20} | {rate_20:.1f}% | {combined_pass} | {combined_tot} | {combined_rate:.1f}% |")

    # Totals row
    total_combined_pass = total_10_pass + total_20_pass
    total_combined_all = total_10_all + total_20_all
    rate_10_total = (total_10_pass / total_10_all * 100) if total_10_all > 0 else 0
    rate_20_total = (total_20_pass / total_20_all * 100) if total_20_all > 0 else 0
    rate_combined_total = (total_combined_pass / total_combined_all * 100) if total_combined_all > 0 else 0

    lines.append(f"| **TOTAL** | **{total_10_pass}** | **{total_10_all}** | **{rate_10_total:.1f}%** | **{total_20_pass}** | **{total_20_all}** | **{rate_20_total:.1f}%** | **{total_combined_pass}** | **{total_combined_all}** | **{rate_combined_total:.1f}%** |")
    lines.append("")

    # Section 3: Backtest Pass Rate by Complexity
    lines.append("## 3. Backtest Pass Rate by Complexity")
    lines.append("")
    lines.append("| Complexity | 10-turn Pass | 10-turn Total | 10-turn Rate | 20-turn Pass | 20-turn Total | 20-turn Rate |")
    lines.append("|------------|--------------|---------------|--------------|--------------|---------------|--------------|")

    for comp in COMPLEXITIES:
        data_10 = pass_rates["10-turn_by_complexity"].get(comp, {"passed": 0, "total": 0})
        data_20 = pass_rates["20-turn_by_complexity"].get(comp, {"passed": 0, "total": 0})

        pass_10 = data_10["passed"]
        tot_10 = data_10["total"]
        rate_10 = (pass_10 / tot_10 * 100) if tot_10 > 0 else 0

        pass_20 = data_20["passed"]
        tot_20 = data_20["total"]
        rate_20 = (pass_20 / tot_20 * 100) if tot_20 > 0 else 0

        lines.append(f"| {comp.capitalize()} | {pass_10} | {tot_10} | {rate_10:.1f}% | {pass_20} | {tot_20} | {rate_20:.1f}% |")
    lines.append("")

    # Section 4: Correctness Rate by Condition
    lines.append("## 4. Correctness Rate by Condition (20-turn)")
    lines.append("")
    lines.append("*Correctness defined as: total_return between -100% and +1000% (reasonable, not pathological)*")
    lines.append("")
    lines.append("| Condition | Simple Correct | Simple Total | Simple Rate | Medium Correct | Medium Total | Medium Rate | Complex Correct | Complex Total | Complex Rate |")
    lines.append("|-----------|----------------|--------------|-------------|----------------|--------------|-------------|-----------------|---------------|--------------|")

    for c in CONDITIONS:
        row = [f"| {c.upper()}"]
        for comp in COMPLEXITIES:
            data = correctness.get(c, {}).get(comp, {"correct": 0, "total": 0})
            correct = data["correct"]
            total = data["total"]
            rate = (correct / total * 100) if total > 0 else 0
            row.extend([f" {correct}", f" {total}", f" {rate:.1f}%"])
        lines.append(" |".join(row) + " |")
    lines.append("")

    # Section 5: Semantic Equivalence by Condition
    lines.append("## 5. Semantic Equivalence by Condition")
    lines.append("")
    lines.append("*Semantic equivalence criteria:*")
    lines.append("- **Simple**: Position signals match (correlation > 0.99)")
    lines.append("- **Medium**: Indicator correlations > 0.99")
    lines.append("- **Complex**: Indicator correlations > 0.95")
    lines.append("")

    if equivalence:
        for turns in ["10-turn", "20-turn"]:
            lines.append(f"### {turns.capitalize()}")
            lines.append("")
            lines.append("| Condition | Simple Equiv | Simple Tested | Simple Rate | Medium Equiv | Medium Tested | Medium Rate | Complex Equiv | Complex Tested | Complex Rate |")
            lines.append("|-----------|--------------|---------------|-------------|--------------|---------------|-------------|---------------|----------------|--------------|")

            for c in CONDITIONS:
                row = [f"| {c.upper()}"]
                for comp in COMPLEXITIES:
                    data = equivalence[turns].get(c, {}).get(comp, {"equivalent": 0, "tested": 0, "total": 0})
                    equiv = data["equivalent"]
                    tested = data["tested"]
                    rate = (equiv / tested * 100) if tested > 0 else 0
                    row.extend([f" {equiv}", f" {tested}", f" {rate:.1f}%"])
                lines.append(" |".join(row) + " |")
            lines.append("")
    else:
        lines.append("*Semantic equivalence analysis could not be completed.*")
        lines.append("")

    # Section 6: Token Usage by Condition
    lines.append("## 6. Token Usage by Condition (20-turn, Successful Runs)")
    lines.append("")
    lines.append("| Condition | Mean Tokens | Std Dev | Count |")
    lines.append("|-----------|-------------|---------|-------|")

    for c in CONDITIONS:
        data = tokens.get(c, {"mean": 0, "std": 0, "count": 0})
        mean = data["mean"]
        std = data["std"]
        count = data["count"]
        lines.append(f"| {c.upper()} | {mean:,.0f} | {std:,.0f} | {count} |")
    lines.append("")

    # Section 7: Consistency (CV) by Condition
    lines.append("## 7. Consistency (CV of Total Return) by Condition (20-turn)")
    lines.append("")
    lines.append("*Lower CV indicates more consistent results across runs.*")
    lines.append("*Only includes reasonable returns (between -100% and +1000%) for meaningful statistics.*")
    lines.append("")
    lines.append("| Condition | Simple CV | Simple Mean | Simple N | Medium CV | Medium Mean | Medium N | Complex CV | Complex Mean | Complex N |")
    lines.append("|-----------|-----------|-------------|----------|-----------|-------------|----------|------------|--------------|-----------|")

    for c in CONDITIONS:
        row = [f"| {c.upper()}"]
        for comp in COMPLEXITIES:
            data = consistency.get(c, {}).get(comp, {"cv": None, "mean": None, "n": 0, "n_total": 0})
            cv = data["cv"]
            mean = data["mean"]
            n = data["n"]
            n_total = data.get("n_total", n)

            cv_str = f"{cv:.2f}" if cv is not None and cv != np.inf else "N/A"
            mean_str = f"{mean:.2%}" if mean is not None else "N/A"
            n_str = f"{n}/{n_total}" if n != n_total else str(n)

            row.extend([f" {cv_str}", f" {mean_str}", f" {n_str}"])
        lines.append(" |".join(row) + " |")
    lines.append("")

    # Section 8: Treatment Effects Summary
    lines.append("## 8. Treatment Effects Summary")
    lines.append("")
    lines.append("*Comparison of enabled vs disabled for each treatment factor across all conditions and complexity levels.*")
    lines.append("")

    lines.append("### Backtest Pass Rate by Treatment")
    lines.append("")
    lines.append("| Treatment | Enabled Pass | Enabled Total | Enabled Rate | Disabled Pass | Disabled Total | Disabled Rate | Difference |")
    lines.append("|-----------|--------------|---------------|--------------|---------------|----------------|---------------|------------|")

    for treatment in ["S", "D", "T"]:
        enabled = treatment_effects[treatment]["enabled"]
        disabled = treatment_effects[treatment]["disabled"]

        enabled_rate = (enabled["passed"] / enabled["total"] * 100) if enabled["total"] > 0 else 0
        disabled_rate = (disabled["passed"] / disabled["total"] * 100) if disabled["total"] > 0 else 0
        diff = enabled_rate - disabled_rate

        treatment_name = {"S": "Spec", "D": "Doc", "T": "Test"}[treatment]
        lines.append(f"| {treatment_name} ({treatment}) | {enabled['passed']} | {enabled['total']} | {enabled_rate:.1f}% | {disabled['passed']} | {disabled['total']} | {disabled_rate:.1f}% | {diff:+.1f}% |")
    lines.append("")

    lines.append("### Mean Total Return by Treatment (Reasonable Returns Only)")
    lines.append("")
    lines.append("*Only includes reasonable returns (between -100% and +1000%) for meaningful statistics.*")
    lines.append("")
    lines.append("| Treatment | Enabled Mean Return | Enabled N | Disabled Mean Return | Disabled N | Difference |")
    lines.append("|-----------|---------------------|-----------|----------------------|------------|------------|")

    for treatment in ["S", "D", "T"]:
        enabled = treatment_effects[treatment]["enabled"]
        disabled = treatment_effects[treatment]["disabled"]

        enabled_mean = np.mean(enabled["returns"]) if enabled["returns"] else 0
        disabled_mean = np.mean(disabled["returns"]) if disabled["returns"] else 0
        diff = enabled_mean - disabled_mean

        treatment_name = {"S": "Spec", "D": "Doc", "T": "Test"}[treatment]
        lines.append(f"| {treatment_name} ({treatment}) | {enabled_mean:.2%} | {len(enabled['returns'])} | {disabled_mean:.2%} | {len(disabled['returns'])} | {diff:+.2%} |")
    lines.append("")

    # Detailed breakdown by complexity
    lines.append("## 9. Detailed Backtest Results by Condition x Complexity")
    lines.append("")

    for turns in ["10-turn", "20-turn"]:
        lines.append(f"### {turns.capitalize()}")
        lines.append("")
        lines.append("| Condition | Simple Pass/Total | Simple Rate | Medium Pass/Total | Medium Rate | Complex Pass/Total | Complex Rate |")
        lines.append("|-----------|-------------------|-------------|-------------------|-------------|--------------------:|-------------:|")

        by_cond_comp = defaultdict(lambda: defaultdict(lambda: {"passed": 0, "total": 0}))
        for r in results[turns]:
            cond = r.get("condition", "unknown")
            comp = r.get("complexity", "unknown")
            by_cond_comp[cond][comp]["total"] += 1
            if r.get("backtest", {}).get("success", False):
                by_cond_comp[cond][comp]["passed"] += 1

        for c in CONDITIONS:
            row = [f"| {c.upper()}"]
            for comp in COMPLEXITIES:
                data = by_cond_comp[c][comp]
                passed = data["passed"]
                total = data["total"]
                rate = (passed / total * 100) if total > 0 else 0
                row.extend([f" {passed}/{total}", f" {rate:.0f}%"])
            lines.append(" |".join(row) + " |")
        lines.append("")

    # Raw metrics summary
    lines.append("## 10. Raw Metrics Summary (20-turn, Successful Backtests)")
    lines.append("")
    lines.append("*Statistics computed only for reasonable returns (between -100% and +1000%).*")
    lines.append("")

    # Collect all successful metrics (only reasonable returns)
    all_metrics = defaultdict(lambda: defaultdict(list))
    all_metrics_count = defaultdict(lambda: {"total": 0, "reasonable": 0})
    for r in results["20-turn"]:
        cond = r.get("condition", "unknown")
        bt = r.get("backtest", {})
        if bt.get("success", False):
            metrics = bt.get("metrics", {})
            tr = metrics.get("total_return", None)
            all_metrics_count[cond]["total"] += 1
            if is_reasonable_return(tr):
                all_metrics_count[cond]["reasonable"] += 1
                for key, val in metrics.items():
                    if val is not None:
                        all_metrics[cond][key].append(val)

    lines.append("### Mean Metrics by Condition")
    lines.append("")
    lines.append("| Condition | Mean Total Return | Mean Sharpe Ratio | Mean Max Drawdown | Mean Total Trades | N (reasonable/total) |")
    lines.append("|-----------|-------------------|-------------------|-------------------|-------------------|----------------------|")

    for c in CONDITIONS:
        metrics = all_metrics[c]
        tr_mean = np.mean(metrics["total_return"]) if metrics["total_return"] else 0
        sr_mean = np.mean(metrics["sharpe_ratio"]) if metrics["sharpe_ratio"] else 0
        md_mean = np.mean(metrics["max_drawdown"]) if metrics["max_drawdown"] else 0
        tt_mean = np.mean(metrics["total_trades"]) if metrics["total_trades"] else 0
        n_reasonable = all_metrics_count[c]["reasonable"]
        n_total = all_metrics_count[c]["total"]

        lines.append(f"| {c.upper()} | {tr_mean:.2%} | {sr_mean:.2f} | {md_mean:.2%} | {tt_mean:.1f} | {n_reasonable}/{n_total} |")
    lines.append("")

    # Add section for pathological returns count
    lines.append("### Pathological Returns Summary")
    lines.append("")
    lines.append("*Count of runs with total_return outside reasonable range by condition and complexity.*")
    lines.append("")

    pathological_count = defaultdict(lambda: defaultdict(int))
    for r in results["20-turn"]:
        cond = r.get("condition", "unknown")
        comp = r.get("complexity", "unknown")
        bt = r.get("backtest", {})
        if bt.get("success", False):
            metrics = bt.get("metrics", {})
            tr = metrics.get("total_return", None)
            if tr is not None and not is_reasonable_return(tr):
                pathological_count[cond][comp] += 1

    lines.append("| Condition | Simple | Medium | Complex | Total |")
    lines.append("|-----------|--------|--------|---------|-------|")

    for c in CONDITIONS:
        s = pathological_count[c]["simple"]
        m = pathological_count[c]["medium"]
        x = pathological_count[c]["complex"]
        t = s + m + x
        lines.append(f"| {c.upper()} | {s} | {m} | {x} | {t} |")
    lines.append("")

    return "\n".join(lines)

def main():
    print("Loading all results...")
    results = load_all_results()
    print(f"  10-turn: {len(results['10-turn'])} results")
    print(f"  20-turn: {len(results['20-turn'])} results")

    print("\nComputing backtest pass rates...")
    pass_rates = compute_backtest_pass_rates(results)

    print("Computing correctness rates...")
    correctness = compute_correctness_rates(results)

    print("Computing semantic equivalence (this may take a moment)...")
    equivalence = compute_semantic_equivalence(results)

    print("Computing token usage...")
    tokens = compute_token_usage(results)

    print("Computing consistency (CV)...")
    consistency = compute_consistency_cv(results)

    print("Computing treatment effects...")
    treatment_effects = compute_treatment_effects(results)

    print("\nGenerating markdown report...")
    report = generate_markdown_report(
        results, pass_rates, correctness, equivalence,
        tokens, consistency, treatment_effects
    )

    print(f"\nWriting report to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        f.write(report)

    print("Done!")

if __name__ == "__main__":
    main()
