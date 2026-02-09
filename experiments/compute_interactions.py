#!/usr/bin/env python3
"""
Analyze treatment combinations for semantic equivalence.
Uses corrected semantic equivalence data.
"""

import numpy as np
from scipy import stats

# Corrected semantic equivalence data (20-turn)
conditions = {
    'C0': {'S': 0, 'D': 0, 'T': 0, 'equiv': 10, 'total': 15, 'label': '---'},
    'C1': {'S': 1, 'D': 0, 'T': 0, 'equiv': 15, 'total': 15, 'label': 'S--'},
    'C2': {'S': 0, 'D': 1, 'T': 0, 'equiv': 14, 'total': 14, 'label': '-D-'},
    'C3': {'S': 0, 'D': 0, 'T': 1, 'equiv': 10, 'total': 15, 'label': '--T'},
    'C4': {'S': 1, 'D': 1, 'T': 0, 'equiv': 15, 'total': 15, 'label': 'SD-'},
    'C5': {'S': 1, 'D': 0, 'T': 1, 'equiv': 11, 'total': 15, 'label': 'S-T'},
    'C6': {'S': 0, 'D': 1, 'T': 1, 'equiv': 11, 'total': 14, 'label': '-DT'},
    'C7': {'S': 1, 'D': 1, 'T': 1, 'equiv': 13, 'total': 15, 'label': 'SDT'},
}

print("=" * 70)
print("Treatment Combination Analysis")
print("=" * 70)
print()

# Summary table
print("Semantic Equivalence by Condition:")
print("-" * 50)
print(f"{'Condition':<12} {'Treatments':<10} {'Equiv':<10} {'Rate':<10}")
print("-" * 50)

for cond, data in conditions.items():
    rate = 100 * data['equiv'] / data['total']
    print(f"{cond:<12} {data['label']:<10} {data['equiv']}/{data['total']:<7} {rate:.0f}%")

print()
print("=" * 70)
print("Interaction Analysis")
print("=" * 70)
print()

# Compare T-absent vs T-present for each S,D combination
print("Effect of Tests (T) on each S,D combination:")
print("-" * 60)

# No S, No D: C0 (---) vs C3 (--T)
c0_rate = conditions['C0']['equiv'] / conditions['C0']['total']
c3_rate = conditions['C3']['equiv'] / conditions['C3']['total']
diff_00 = (c3_rate - c0_rate) * 100
print(f"  Base (---):     C0={c0_rate*100:.0f}% → C3={c3_rate*100:.0f}% (T effect: {diff_00:+.0f}pp)")

# S only: C1 (S--) vs C5 (S-T)
c1_rate = conditions['C1']['equiv'] / conditions['C1']['total']
c5_rate = conditions['C5']['equiv'] / conditions['C5']['total']
diff_S0 = (c5_rate - c1_rate) * 100
print(f"  With S (S--):   C1={c1_rate*100:.0f}% → C5={c5_rate*100:.0f}% (T effect: {diff_S0:+.0f}pp)")

# D only: C2 (-D-) vs C6 (-DT)
c2_rate = conditions['C2']['equiv'] / conditions['C2']['total']
c6_rate = conditions['C6']['equiv'] / conditions['C6']['total']
diff_0D = (c6_rate - c2_rate) * 100
print(f"  With D (-D-):   C2={c2_rate*100:.0f}% → C6={c6_rate*100:.0f}% (T effect: {diff_0D:+.0f}pp)")

# S and D: C4 (SD-) vs C7 (SDT)
c4_rate = conditions['C4']['equiv'] / conditions['C4']['total']
c7_rate = conditions['C7']['equiv'] / conditions['C7']['total']
diff_SD = (c7_rate - c4_rate) * 100
print(f"  With SD (SD-):  C4={c4_rate*100:.0f}% → C7={c7_rate*100:.0f}% (T effect: {diff_SD:+.0f}pp)")

print()
print("Summary: Tests (T) effect varies by context:")
print(f"  - Without S or D: {diff_00:+.0f}pp (no effect)")
print(f"  - With S only:    {diff_S0:+.0f}pp (NEGATIVE)")
print(f"  - With D only:    {diff_0D:+.0f}pp (NEGATIVE)")
print(f"  - With S and D:   {diff_SD:+.0f}pp (NEGATIVE)")

print()
print("=" * 70)
print("T-absent vs T-present (Aggregate)")
print("=" * 70)

# T-absent: C0, C1, C2, C4
t_absent = ['C0', 'C1', 'C2', 'C4']
t_present = ['C3', 'C5', 'C6', 'C7']

t_absent_equiv = sum(conditions[c]['equiv'] for c in t_absent)
t_absent_total = sum(conditions[c]['total'] for c in t_absent)
t_present_equiv = sum(conditions[c]['equiv'] for c in t_present)
t_present_total = sum(conditions[c]['total'] for c in t_present)

print(f"T-absent (C0,C1,C2,C4):  {t_absent_equiv}/{t_absent_total} = {100*t_absent_equiv/t_absent_total:.1f}%")
print(f"T-present (C3,C5,C6,C7): {t_present_equiv}/{t_present_total} = {100*t_present_equiv/t_present_total:.1f}%")

# Fisher's exact test
table = [[t_absent_equiv, t_absent_total - t_absent_equiv],
         [t_present_equiv, t_present_total - t_present_equiv]]
odds_ratio, p_value = stats.fisher_exact(table)
print(f"Fisher's exact test: p = {p_value:.4f}")

print()
print("=" * 70)
print("Perfect Separation Analysis")
print("=" * 70)
print()
print("Conditions with 100% equivalence:")
for cond in ['C1', 'C2', 'C4']:
    data = conditions[cond]
    print(f"  {cond} ({data['label']}): {data['equiv']}/{data['total']}")

print()
print("Common factor: T is ABSENT in all 100% conditions")
print()
print("Conditions with <100% equivalence:")
for cond in ['C0', 'C3', 'C5', 'C6', 'C7']:
    data = conditions[cond]
    rate = 100 * data['equiv'] / data['total']
    t_status = "T present" if data['T'] else "T absent (Control)"
    print(f"  {cond} ({data['label']}): {data['equiv']}/{data['total']} = {rate:.0f}% [{t_status}]")

print()
print("=" * 70)
print("LaTeX Table for Results Section")
print("=" * 70)
print()
print(r"""\begin{table}[h]
\centering
\caption{Effect of adding Tests (T) to each treatment combination.}
\label{tab:test-interactions}
\vspace{0.5em}
\begin{tabular}{lcccl}
\toprule
Base & Without T & With T & T Effect & Interpretation \\
\midrule""")
print(f"None (Control) & C0: 67\\% & C3: 67\\% & {diff_00:+.0f}pp & No change \\\\")
print(f"Schema (S) & C1: 100\\% & C5: 73\\% & {diff_S0:+.0f}pp & T hurts S \\\\")
print(f"Docs (D) & C2: 100\\% & C6: 78\\% & {diff_0D:+.0f}pp & T hurts D \\\\")
print(f"Schema+Docs (SD) & C4: 100\\% & C7: 87\\% & {diff_SD:+.0f}pp & T hurts SD \\\\")
print(r"""\bottomrule
\end{tabular}
\end{table}""")

print()
print("=" * 70)
print("LaTeX Table for Discussion Section")
print("=" * 70)
print()
print(r"""\begin{table}[h]
\centering
\caption{Semantic equivalence by treatment combination (20-turn).}
\label{tab:combination-summary}
\vspace{0.5em}
\begin{tabular}{lcccc}
\toprule
Condition & S & D & T & Equiv Rate \\
\midrule
C0 (Control) & --- & --- & --- & 67\% \\
C1 & $\checkmark$ & --- & --- & \textbf{100\%} \\
C2 & --- & $\checkmark$ & --- & \textbf{100\%} \\
C3 & --- & --- & $\checkmark$ & 67\% \\
C4 & $\checkmark$ & $\checkmark$ & --- & \textbf{100\%} \\
C5 & $\checkmark$ & --- & $\checkmark$ & 73\% \\
C6 & --- & $\checkmark$ & $\checkmark$ & 78\% \\
C7 & $\checkmark$ & $\checkmark$ & $\checkmark$ & 87\% \\
\bottomrule
\end{tabular}

\vspace{0.3em}
\par\small\textit{Bold indicates 100\% equivalence. Note: all 100\% conditions have T absent.}
\end{table}""")
