"""
RAG Documentation Grounding
============================

Instructs the agent to use the OpenSearch documentation search tool
to verify APIs before using them.

Used by: C2 (Docs), C4 (Schema+Docs), C6 (Docs+TDD), C7 (All)

Replaces the former static VAS (Verified API Surface) allowlist.
"""

RAG_DESCRIPTION = """
## Documentation Search

You have access to a `search_docs` tool that searches official library documentation
for numpy, pandas, scipy, and vectorbt.

Use it to look up APIs you are **unsure about** — especially vectorbt-specific APIs
(e.g., `vbt.RSI.run`, `vbt.Portfolio.from_order_func`, `order_nb`) which have
non-obvious signatures.

**Do NOT search for every API call.** Standard numpy/pandas operations you already
know well (e.g., `np.where`, `pd.Series.rolling`) do not need verification.
Focus searches on the 2-3 APIs that are most critical or unfamiliar.

**Prioritize writing and submitting code.** Search only what you need, then call
`submit_code` as early as possible. You can always fix issues after getting feedback.
"""
