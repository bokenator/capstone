#!/usr/bin/env python3
"""
Compute statistical tests for the experiment results.
- Fisher's exact test for treatment effects
- Wilson score confidence intervals
- Logistic regression with interactions
- Welch's t-test for token usage
"""

import json
import os
import numpy as np
from scipy import stats
from scipy.stats import fisher_exact
import warnings
warnings.filterwarnings('ignore')

# Treatment encoding
# C0: ---, C1: S--, C2: -D-, C3: --T, C4: SD-, C5: S-T, C6: -DT, C7: SDT
TREATMENTS = {
    'c0': {'S': 0, 'D': 0, 'T': 0},
    'c1': {'S': 1, 'D': 0, 'T': 0},
    'c2': {'S': 0, 'D': 1, 'T': 0},
    'c3': {'S': 0, 'D': 0, 'T': 1},
    'c4': {'S': 1, 'D': 1, 'T': 0},
    'c5': {'S': 1, 'D': 0, 'T': 1},
    'c6': {'S': 0, 'D': 1, 'T': 1},
    'c7': {'S': 1, 'D': 1, 'T': 1},
}

def wilson_score_interval(successes, total, confidence=0.95):
    """Calculate Wilson score confidence interval."""
    if total == 0:
        return (0, 0)
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = successes / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
    return (max(0, center - margin), min(1, center + margin))

def load_results(base_dir):
    """Load all results from JSON files."""
    results = []
    for condition in ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7']:
        for complexity in ['simple', 'medium', 'complex']:
            for run in range(1, 6):
                run_dir = os.path.join(base_dir, f"{condition}_{complexity}_{run}")
                result_file = os.path.join(run_dir, "results.json")
                if os.path.exists(result_file):
                    with open(result_file) as f:
                        data = json.load(f)
                    data['condition'] = condition
                    data['complexity'] = complexity
                    data['run'] = run
                    data['S'] = TREATMENTS[condition]['S']
                    data['D'] = TREATMENTS[condition]['D']
                    data['T'] = TREATMENTS[condition]['T']
                    results.append(data)
    return results

def get_backtest_passed(r):
    """Check if backtest passed."""
    return r.get('backtest', {}).get('success', False)

def get_total_tokens(r):
    """Get total tokens from result."""
    tokens = r.get('total_tokens', {})
    if isinstance(tokens, dict):
        return tokens.get('total_tokens', 0)
    return tokens if isinstance(tokens, (int, float)) else 0

def compute_backtest_stats(results):
    """Compute backtest pass rate statistics."""
    # S-enabled vs S-disabled
    s_enabled = [r for r in results if r['S'] == 1]
    s_disabled = [r for r in results if r['S'] == 0]

    s_enabled_pass = sum(1 for r in s_enabled if get_backtest_passed(r))
    s_disabled_pass = sum(1 for r in s_disabled if get_backtest_passed(r))

    # Fisher's exact test
    # [[s_enabled_pass, s_enabled_fail], [s_disabled_pass, s_disabled_fail]]
    table_s = [[s_enabled_pass, len(s_enabled) - s_enabled_pass],
               [s_disabled_pass, len(s_disabled) - s_disabled_pass]]
    _, p_s = fisher_exact(table_s)

    # D-enabled vs D-disabled
    d_enabled = [r for r in results if r['D'] == 1]
    d_disabled = [r for r in results if r['D'] == 0]

    d_enabled_pass = sum(1 for r in d_enabled if get_backtest_passed(r))
    d_disabled_pass = sum(1 for r in d_disabled if get_backtest_passed(r))

    table_d = [[d_enabled_pass, len(d_enabled) - d_enabled_pass],
               [d_disabled_pass, len(d_disabled) - d_disabled_pass]]
    _, p_d = fisher_exact(table_d)

    # T-enabled vs T-disabled
    t_enabled = [r for r in results if r['T'] == 1]
    t_disabled = [r for r in results if r['T'] == 0]

    t_enabled_pass = sum(1 for r in t_enabled if get_backtest_passed(r))
    t_disabled_pass = sum(1 for r in t_disabled if get_backtest_passed(r))

    table_t = [[t_enabled_pass, len(t_enabled) - t_enabled_pass],
               [t_disabled_pass, len(t_disabled) - t_disabled_pass]]
    _, p_t = fisher_exact(table_t)

    return {
        'S': {
            'enabled_pass': s_enabled_pass, 'enabled_total': len(s_enabled),
            'disabled_pass': s_disabled_pass, 'disabled_total': len(s_disabled),
            'p_value': p_s,
            'enabled_ci': wilson_score_interval(s_enabled_pass, len(s_enabled)),
            'disabled_ci': wilson_score_interval(s_disabled_pass, len(s_disabled)),
        },
        'D': {
            'enabled_pass': d_enabled_pass, 'enabled_total': len(d_enabled),
            'disabled_pass': d_disabled_pass, 'disabled_total': len(d_disabled),
            'p_value': p_d,
            'enabled_ci': wilson_score_interval(d_enabled_pass, len(d_enabled)),
            'disabled_ci': wilson_score_interval(d_disabled_pass, len(d_disabled)),
        },
        'T': {
            'enabled_pass': t_enabled_pass, 'enabled_total': len(t_enabled),
            'disabled_pass': t_disabled_pass, 'disabled_total': len(t_disabled),
            'p_value': p_t,
            'enabled_ci': wilson_score_interval(t_enabled_pass, len(t_enabled)),
            'disabled_ci': wilson_score_interval(t_disabled_pass, len(t_disabled)),
        },
    }

def compute_semantic_stats(results, semantic_data):
    """Compute semantic equivalence statistics using provided semantic data."""
    # semantic_data is a dict: {(condition, complexity): equiv_count}

    # Build success/fail counts by treatment
    s_enabled_equiv = 0
    s_enabled_total = 0
    s_disabled_equiv = 0
    s_disabled_total = 0

    d_enabled_equiv = 0
    d_enabled_total = 0
    d_disabled_equiv = 0
    d_disabled_total = 0

    t_enabled_equiv = 0
    t_enabled_total = 0
    t_disabled_equiv = 0
    t_disabled_total = 0

    for (cond, complexity), (equiv, total) in semantic_data.items():
        s = TREATMENTS[cond]['S']
        d = TREATMENTS[cond]['D']
        t = TREATMENTS[cond]['T']

        if s == 1:
            s_enabled_equiv += equiv
            s_enabled_total += total
        else:
            s_disabled_equiv += equiv
            s_disabled_total += total

        if d == 1:
            d_enabled_equiv += equiv
            d_enabled_total += total
        else:
            d_disabled_equiv += equiv
            d_disabled_total += total

        if t == 1:
            t_enabled_equiv += equiv
            t_enabled_total += total
        else:
            t_disabled_equiv += equiv
            t_disabled_total += total

    # Fisher's exact tests
    table_s = [[s_enabled_equiv, s_enabled_total - s_enabled_equiv],
               [s_disabled_equiv, s_disabled_total - s_disabled_equiv]]
    _, p_s = fisher_exact(table_s)

    table_d = [[d_enabled_equiv, d_enabled_total - d_enabled_equiv],
               [d_disabled_equiv, d_disabled_total - d_disabled_equiv]]
    _, p_d = fisher_exact(table_d)

    table_t = [[t_enabled_equiv, t_enabled_total - t_enabled_equiv],
               [t_disabled_equiv, t_disabled_total - t_disabled_equiv]]
    _, p_t = fisher_exact(table_t)

    return {
        'S': {
            'enabled_equiv': s_enabled_equiv, 'enabled_total': s_enabled_total,
            'disabled_equiv': s_disabled_equiv, 'disabled_total': s_disabled_total,
            'p_value': p_s,
            'enabled_ci': wilson_score_interval(s_enabled_equiv, s_enabled_total),
            'disabled_ci': wilson_score_interval(s_disabled_equiv, s_disabled_total),
        },
        'D': {
            'enabled_equiv': d_enabled_equiv, 'enabled_total': d_enabled_total,
            'disabled_equiv': d_disabled_equiv, 'disabled_total': d_disabled_total,
            'p_value': p_d,
            'enabled_ci': wilson_score_interval(d_enabled_equiv, d_enabled_total),
            'disabled_ci': wilson_score_interval(d_disabled_equiv, d_disabled_total),
        },
        'T': {
            'enabled_equiv': t_enabled_equiv, 'enabled_total': t_enabled_total,
            'disabled_equiv': t_disabled_equiv, 'disabled_total': t_disabled_total,
            'p_value': p_t,
            'enabled_ci': wilson_score_interval(t_enabled_equiv, t_enabled_total),
            'disabled_ci': wilson_score_interval(t_disabled_equiv, t_disabled_total),
        },
    }

def compute_token_stats(results):
    """Compute token usage statistics with Welch's t-test."""
    # Only include successful runs
    successful = [r for r in results if get_backtest_passed(r)]

    s_enabled_tokens = [get_total_tokens(r) for r in successful if r['S'] == 1]
    s_disabled_tokens = [get_total_tokens(r) for r in successful if r['S'] == 0]

    d_enabled_tokens = [get_total_tokens(r) for r in successful if r['D'] == 1]
    d_disabled_tokens = [get_total_tokens(r) for r in successful if r['D'] == 0]

    t_enabled_tokens = [get_total_tokens(r) for r in successful if r['T'] == 1]
    t_disabled_tokens = [get_total_tokens(r) for r in successful if r['T'] == 0]

    # Welch's t-test
    t_stat_s, p_s = stats.ttest_ind(s_enabled_tokens, s_disabled_tokens, equal_var=False)
    t_stat_d, p_d = stats.ttest_ind(d_enabled_tokens, d_disabled_tokens, equal_var=False)
    t_stat_t, p_t = stats.ttest_ind(t_enabled_tokens, t_disabled_tokens, equal_var=False)

    return {
        'S': {
            'enabled_mean': np.mean(s_enabled_tokens) if s_enabled_tokens else 0,
            'enabled_std': np.std(s_enabled_tokens) if s_enabled_tokens else 0,
            'disabled_mean': np.mean(s_disabled_tokens) if s_disabled_tokens else 0,
            'disabled_std': np.std(s_disabled_tokens) if s_disabled_tokens else 0,
            't_stat': t_stat_s,
            'p_value': p_s,
        },
        'D': {
            'enabled_mean': np.mean(d_enabled_tokens) if d_enabled_tokens else 0,
            'enabled_std': np.std(d_enabled_tokens) if d_enabled_tokens else 0,
            'disabled_mean': np.mean(d_disabled_tokens) if d_disabled_tokens else 0,
            'disabled_std': np.std(d_disabled_tokens) if d_disabled_tokens else 0,
            't_stat': t_stat_d,
            'p_value': p_d,
        },
        'T': {
            'enabled_mean': np.mean(t_enabled_tokens) if t_enabled_tokens else 0,
            'enabled_std': np.std(t_enabled_tokens) if t_enabled_tokens else 0,
            'disabled_mean': np.mean(t_disabled_tokens) if t_disabled_tokens else 0,
            'disabled_std': np.std(t_disabled_tokens) if t_disabled_tokens else 0,
            't_stat': t_stat_t,
            'p_value': p_t,
        },
    }

def fit_logistic_regression(results, outcome_func):
    """Fit logistic regression with interaction terms."""
    try:
        import statsmodels.api as sm
    except ImportError:
        return None

    # Build design matrix
    X = []
    y = []
    for r in results:
        s, d, t = r['S'], r['D'], r['T']
        # Include main effects and all interactions
        features = [
            1,  # intercept
            s, d, t,  # main effects
            s*d, s*t, d*t,  # two-way interactions
            s*d*t  # three-way interaction
        ]
        X.append(features)
        y.append(1 if outcome_func(r) else 0)

    X = np.array(X)
    y = np.array(y)

    # Check for perfect separation
    if y.sum() == 0 or y.sum() == len(y):
        return None

    try:
        model = sm.Logit(y, X)
        result = model.fit(disp=0, maxiter=100)

        coef_names = ['intercept', 'S', 'D', 'T', 'SD', 'ST', 'DT', 'SDT']
        return {
            'coefficients': dict(zip(coef_names, result.params)),
            'pvalues': dict(zip(coef_names, result.pvalues)),
            'converged': result.mle_retvals['converged'],
        }
    except Exception as e:
        return {'error': str(e)}

def main():
    # Load 20-turn results
    base_dir_20 = "/root/wqu/capstone/w6/experiments_imp_2/results_20-gpt5mini"
    results_20 = load_results(base_dir_20)

    # Load 10-turn results
    base_dir_10 = "/root/wqu/capstone/w6/experiments_imp_2/results_10_gpt5mini"
    results_10 = load_results(base_dir_10)

    # Combined results
    results_combined = results_10 + results_20

    print("=" * 60)
    print("STATISTICAL ANALYSIS RESULTS")
    print("=" * 60)

    # Backtest pass rate (combined)
    print("\n" + "=" * 60)
    print("1. BACKTEST PASS RATE (Combined 10+20 turn)")
    print("=" * 60)

    backtest_stats = compute_backtest_stats(results_combined)
    for treatment in ['S', 'D', 'T']:
        s = backtest_stats[treatment]
        enabled_rate = s['enabled_pass'] / s['enabled_total'] * 100
        disabled_rate = s['disabled_pass'] / s['disabled_total'] * 100
        diff = enabled_rate - disabled_rate
        print(f"\n{treatment} treatment:")
        print(f"  Enabled:  {s['enabled_pass']}/{s['enabled_total']} ({enabled_rate:.1f}%) "
              f"95% CI: [{s['enabled_ci'][0]*100:.1f}%, {s['enabled_ci'][1]*100:.1f}%]")
        print(f"  Disabled: {s['disabled_pass']}/{s['disabled_total']} ({disabled_rate:.1f}%) "
              f"95% CI: [{s['disabled_ci'][0]*100:.1f}%, {s['disabled_ci'][1]*100:.1f}%]")
        print(f"  Difference: {diff:+.1f}pp")
        print(f"  Fisher's exact p-value: {s['p_value']:.4f}" +
              (" *" if s['p_value'] < 0.05 else ""))

    # Backtest pass rate (20-turn only)
    print("\n" + "=" * 60)
    print("2. BACKTEST PASS RATE (20-turn only)")
    print("=" * 60)

    backtest_stats_20 = compute_backtest_stats(results_20)
    for treatment in ['S', 'D', 'T']:
        s = backtest_stats_20[treatment]
        enabled_rate = s['enabled_pass'] / s['enabled_total'] * 100
        disabled_rate = s['disabled_pass'] / s['disabled_total'] * 100
        diff = enabled_rate - disabled_rate
        print(f"\n{treatment} treatment:")
        print(f"  Enabled:  {s['enabled_pass']}/{s['enabled_total']} ({enabled_rate:.1f}%) "
              f"95% CI: [{s['enabled_ci'][0]*100:.1f}%, {s['enabled_ci'][1]*100:.1f}%]")
        print(f"  Disabled: {s['disabled_pass']}/{s['disabled_total']} ({disabled_rate:.1f}%) "
              f"95% CI: [{s['disabled_ci'][0]*100:.1f}%, {s['disabled_ci'][1]*100:.1f}%]")
        print(f"  Difference: {diff:+.1f}pp")
        print(f"  Fisher's exact p-value: {s['p_value']:.4f}" +
              (" *" if s['p_value'] < 0.05 else ""))

    # Semantic equivalence (from ANALYSIS_REFERENCE.md data)
    # Data from 20-turn semantic analysis - CORRECTED 2025-02-08
    # Both EWM and SMA-based RSI are valid, only fundamentally wrong implementations are "Different"
    semantic_data = {
        # (condition, complexity): (equiv_count, total)
        ('c0', 'simple'): (5, 5),   # All Partial (custom EWM RSI, valid)
        ('c0', 'medium'): (5, 5),   # All Partial (custom pandas, valid)
        ('c0', 'complex'): (0, 5),  # All Different (pathological returns)
        ('c1', 'simple'): (5, 5),   # All Partial
        ('c1', 'medium'): (5, 5),   # All Identical (vbt APIs)
        ('c1', 'complex'): (5, 5),  # All Partial
        ('c2', 'simple'): (5, 5),   # All Identical (vbt.RSI.run)
        ('c2', 'medium'): (4, 4),   # All Identical, 1 failed
        ('c2', 'complex'): (5, 5),  # All Partial
        ('c3', 'simple'): (5, 5),   # All Partial
        ('c3', 'medium'): (5, 5),   # All Partial
        ('c3', 'complex'): (0, 5),  # All Different (pathological returns)
        ('c4', 'simple'): (5, 5),   # All Partial
        ('c4', 'medium'): (5, 5),   # All Identical
        ('c4', 'complex'): (5, 5),  # All Partial
        ('c5', 'simple'): (5, 5),   # All Partial
        ('c5', 'medium'): (5, 5),   # All Identical
        ('c5', 'complex'): (1, 5),  # 1 Partial, 4 Different (expanding window)
        ('c6', 'simple'): (4, 5),   # 4 Partial, 1 Different (wrong crossing logic)
        ('c6', 'medium'): (4, 4),   # 3 Identical + 1 Partial, 1 failed
        ('c6', 'complex'): (3, 5),  # 3 Partial, 2 Different
        ('c7', 'simple'): (5, 5),   # All Partial
        ('c7', 'medium'): (5, 5),   # All Identical
        ('c7', 'complex'): (3, 5),  # 3 Partial, 2 Different
    }

    print("\n" + "=" * 60)
    print("3. SEMANTIC EQUIVALENCE (20-turn)")
    print("=" * 60)

    semantic_stats = compute_semantic_stats(results_20, semantic_data)
    for treatment in ['S', 'D', 'T']:
        s = semantic_stats[treatment]
        enabled_rate = s['enabled_equiv'] / s['enabled_total'] * 100
        disabled_rate = s['disabled_equiv'] / s['disabled_total'] * 100
        diff = enabled_rate - disabled_rate
        print(f"\n{treatment} treatment:")
        print(f"  Enabled:  {s['enabled_equiv']}/{s['enabled_total']} ({enabled_rate:.1f}%) "
              f"95% CI: [{s['enabled_ci'][0]*100:.1f}%, {s['enabled_ci'][1]*100:.1f}%]")
        print(f"  Disabled: {s['disabled_equiv']}/{s['disabled_total']} ({disabled_rate:.1f}%) "
              f"95% CI: [{s['disabled_ci'][0]*100:.1f}%, {s['disabled_ci'][1]*100:.1f}%]")
        print(f"  Difference: {diff:+.1f}pp")
        print(f"  Fisher's exact p-value: {s['p_value']:.4f}" +
              (" *" if s['p_value'] < 0.05 else ""))

    # Token usage
    print("\n" + "=" * 60)
    print("4. TOKEN USAGE (20-turn, successful runs)")
    print("=" * 60)

    token_stats = compute_token_stats(results_20)
    for treatment in ['S', 'D', 'T']:
        s = token_stats[treatment]
        print(f"\n{treatment} treatment:")
        print(f"  Enabled mean:  {s['enabled_mean']:,.0f} (SD: {s['enabled_std']:,.0f})")
        print(f"  Disabled mean: {s['disabled_mean']:,.0f} (SD: {s['disabled_std']:,.0f})")
        print(f"  Difference: {s['enabled_mean'] - s['disabled_mean']:+,.0f}")
        print(f"  Welch's t-test: t = {s['t_stat']:.2f}, p = {s['p_value']:.4f}" +
              (" *" if s['p_value'] < 0.05 else ""))

    # Logistic regression for backtest pass
    print("\n" + "=" * 60)
    print("5. LOGISTIC REGRESSION (Backtest Pass, Combined)")
    print("=" * 60)

    lr_backtest = fit_logistic_regression(results_combined,
                                           lambda r: get_backtest_passed(r))
    if lr_backtest and 'coefficients' in lr_backtest:
        print("\nCoefficients (log-odds):")
        for name in ['S', 'D', 'T', 'SD', 'ST', 'DT', 'SDT']:
            coef = lr_backtest['coefficients'].get(name, 0)
            pval = lr_backtest['pvalues'].get(name, 1)
            sig = " *" if pval < 0.05 else ""
            print(f"  {name:6s}: {coef:+.3f} (p = {pval:.4f}){sig}")
    else:
        print("  Logistic regression could not be fitted (likely perfect separation)")

    # Logistic regression for semantic equivalence
    print("\n" + "=" * 60)
    print("6. LOGISTIC REGRESSION (Semantic Equivalence, 20-turn)")
    print("=" * 60)

    # Create pseudo-results for semantic equivalence
    semantic_results = []
    for (cond, complexity), (equiv, total) in semantic_data.items():
        for i in range(total):
            r = {
                'condition': cond,
                'complexity': complexity,
                'S': TREATMENTS[cond]['S'],
                'D': TREATMENTS[cond]['D'],
                'T': TREATMENTS[cond]['T'],
                'semantic_equiv': i < equiv,
            }
            semantic_results.append(r)

    lr_semantic = fit_logistic_regression(semantic_results,
                                           lambda r: r.get('semantic_equiv', False))
    if lr_semantic and 'coefficients' in lr_semantic:
        print("\nCoefficients (log-odds):")
        for name in ['S', 'D', 'T', 'SD', 'ST', 'DT', 'SDT']:
            coef = lr_semantic['coefficients'].get(name, 0)
            pval = lr_semantic['pvalues'].get(name, 1)
            sig = " *" if pval < 0.05 else ""
            print(f"  {name:6s}: {coef:+.3f} (p = {pval:.4f}){sig}")
    else:
        print("  Logistic regression could not be fitted")
        if lr_semantic and 'error' in lr_semantic:
            print(f"  Error: {lr_semantic['error']}")

    # Summary for LaTeX
    print("\n" + "=" * 60)
    print("LATEX SUMMARY TABLE DATA")
    print("=" * 60)

    print("\n% Treatment effects on Backtest Pass Rate (Combined)")
    for treatment in ['S', 'D', 'T']:
        s = backtest_stats[treatment]
        enabled_rate = s['enabled_pass'] / s['enabled_total'] * 100
        disabled_rate = s['disabled_pass'] / s['disabled_total'] * 100
        diff = enabled_rate - disabled_rate
        print(f"% {treatment}: {enabled_rate:.1f}% vs {disabled_rate:.1f}% ({diff:+.1f}pp), "
              f"p={s['p_value']:.4f}, "
              f"CI_en=[{s['enabled_ci'][0]*100:.1f},{s['enabled_ci'][1]*100:.1f}], "
              f"CI_dis=[{s['disabled_ci'][0]*100:.1f},{s['disabled_ci'][1]*100:.1f}]")

    print("\n% Treatment effects on Semantic Equivalence (20-turn)")
    for treatment in ['S', 'D', 'T']:
        s = semantic_stats[treatment]
        enabled_rate = s['enabled_equiv'] / s['enabled_total'] * 100
        disabled_rate = s['disabled_equiv'] / s['disabled_total'] * 100
        diff = enabled_rate - disabled_rate
        print(f"% {treatment}: {enabled_rate:.1f}% vs {disabled_rate:.1f}% ({diff:+.1f}pp), "
              f"p={s['p_value']:.4f}, "
              f"CI_en=[{s['enabled_ci'][0]*100:.1f},{s['enabled_ci'][1]*100:.1f}], "
              f"CI_dis=[{s['disabled_ci'][0]*100:.1f},{s['disabled_ci'][1]*100:.1f}]")

    print("\n% Token usage (20-turn)")
    for treatment in ['S', 'D', 'T']:
        s = token_stats[treatment]
        print(f"% {treatment}: {s['enabled_mean']:,.0f} vs {s['disabled_mean']:,.0f}, "
              f"p={s['p_value']:.4f}")

if __name__ == "__main__":
    main()
