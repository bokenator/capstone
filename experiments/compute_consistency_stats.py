#!/usr/bin/env python3
"""
Compute statistical significance for treatment effects on consistency (CV).
Uses Welch's t-test for comparing mean CVs between treatment groups.
"""

import numpy as np
from scipy import stats

# CV data by condition and complexity (from ANALYSIS_REFERENCE.md)
cv_data = {
    'C0': {'S': 0, 'D': 0, 'T': 0, 'simple': 0.59, 'medium': 1.22, 'complex': None},  # N/A
    'C1': {'S': 1, 'D': 0, 'T': 0, 'simple': 0.70, 'medium': 0.32, 'complex': 0.47},
    'C2': {'S': 0, 'D': 1, 'T': 0, 'simple': 0.07, 'medium': 0.64, 'complex': 1.26},
    'C3': {'S': 0, 'D': 0, 'T': 1, 'simple': 0.63, 'medium': 2.68, 'complex': None},  # N/A
    'C4': {'S': 1, 'D': 1, 'T': 0, 'simple': 0.79, 'medium': 0.29, 'complex': 0.11},
    'C5': {'S': 1, 'D': 0, 'T': 1, 'simple': 0.56, 'medium': 0.33, 'complex': 0.20},
    'C6': {'S': 0, 'D': 1, 'T': 1, 'simple': 0.61, 'medium': 0.44, 'complex': 0.82},
    'C7': {'S': 1, 'D': 1, 'T': 1, 'simple': 0.76, 'medium': 0.33, 'complex': 0.19},
}

def get_cv_values(treatment, enabled, complexity):
    """Get CV values for conditions where treatment is enabled/disabled."""
    values = []
    for cond, data in cv_data.items():
        if data[treatment] == enabled:
            cv = data[complexity]
            if cv is not None:
                values.append(cv)
    return values

def compute_treatment_effect(treatment, complexity):
    """Compute treatment effect with Welch's t-test."""
    enabled = get_cv_values(treatment, 1, complexity)
    disabled = get_cv_values(treatment, 0, complexity)

    if len(enabled) < 2 or len(disabled) < 2:
        return None, None, None, None, len(enabled), len(disabled)

    mean_enabled = np.mean(enabled)
    mean_disabled = np.mean(disabled)
    effect = mean_enabled - mean_disabled  # Negative = improvement (lower CV)

    t_stat, p_value = stats.ttest_ind(enabled, disabled, equal_var=False)

    return mean_enabled, mean_disabled, effect, p_value, len(enabled), len(disabled)

print("=" * 80)
print("Treatment Effects on Consistency (CV) - Welch's t-test")
print("=" * 80)
print()

for treatment in ['S', 'D', 'T']:
    treatment_name = {'S': 'Schema', 'D': 'Docs', 'T': 'Tests'}[treatment]
    print(f"\n{treatment_name} ({treatment}):")
    print("-" * 70)
    print(f"{'Complexity':<12} {'Enabled':<12} {'Disabled':<12} {'Effect':<12} {'t-stat':<10} {'p-value':<10} {'n'}")
    print("-" * 70)

    all_enabled = []
    all_disabled = []

    for complexity in ['simple', 'medium', 'complex']:
        enabled_vals = get_cv_values(treatment, 1, complexity)
        disabled_vals = get_cv_values(treatment, 0, complexity)

        all_enabled.extend(enabled_vals)
        all_disabled.extend(disabled_vals)

        if len(enabled_vals) >= 2 and len(disabled_vals) >= 2:
            mean_e = np.mean(enabled_vals)
            mean_d = np.mean(disabled_vals)
            effect = mean_e - mean_d
            t_stat, p_val = stats.ttest_ind(enabled_vals, disabled_vals, equal_var=False)

            sig = ""
            if p_val < 0.01:
                sig = "**"
            elif p_val < 0.05:
                sig = "*"
            elif p_val < 0.1:
                sig = "."

            print(f"{complexity:<12} {mean_e:<12.2f} {mean_d:<12.2f} {effect:+.2f}{'':8} {t_stat:<10.2f} {p_val:.4f}{sig:<3} n={len(enabled_vals)},{len(disabled_vals)}")
        else:
            print(f"{complexity:<12} n={len(enabled_vals)},{len(disabled_vals)} (insufficient data)")

    # Overall (pooled across complexities)
    if len(all_enabled) >= 2 and len(all_disabled) >= 2:
        mean_e = np.mean(all_enabled)
        mean_d = np.mean(all_disabled)
        effect = mean_e - mean_d
        t_stat, p_val = stats.ttest_ind(all_enabled, all_disabled, equal_var=False)

        sig = ""
        if p_val < 0.01:
            sig = "**"
        elif p_val < 0.05:
            sig = "*"
        elif p_val < 0.1:
            sig = "."

        print("-" * 70)
        print(f"{'OVERALL':<12} {mean_e:<12.2f} {mean_d:<12.2f} {effect:+.2f}{'':8} {t_stat:<10.2f} {p_val:.4f}{sig:<3} n={len(all_enabled)},{len(all_disabled)}")

print("\n")
print("=" * 80)
print("LaTeX Table Row Updates")
print("=" * 80)
print()

# Compute overall stats for each treatment
for treatment in ['S', 'D', 'T']:
    treatment_name = {'S': 'Schema (S)', 'D': 'Docs (D)', 'T': 'Tests (T)'}[treatment]

    all_enabled = []
    all_disabled = []

    for complexity in ['simple', 'medium', 'complex']:
        all_enabled.extend(get_cv_values(treatment, 1, complexity))
        all_disabled.extend(get_cv_values(treatment, 0, complexity))

    mean_e = np.mean(all_enabled)
    mean_d = np.mean(all_disabled)
    effect = mean_e - mean_d
    t_stat, p_val = stats.ttest_ind(all_enabled, all_disabled, equal_var=False)

    sig = ""
    if p_val < 0.01:
        sig = "**"
    elif p_val < 0.05:
        sig = "*"

    effect_str = f"{effect:+.2f}" if effect >= 0 else f"$-${abs(effect):.2f}"
    if abs(effect) > 0.3:
        effect_str = f"\\textbf{{{effect_str}}}"

    print(f"{treatment_name} & {mean_e:.2f} & {mean_d:.2f} & {effect_str} & {p_val:.3f} & {sig} \\\\")
