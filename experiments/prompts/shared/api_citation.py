"""
API Citation Requirements
=========================

Instructions for documentation-grounded conditions requiring RAG-based verification.
Used by: C2 (Docs), C4 (Schema+Docs), C6 (Docs+TDD), C7 (All)
"""

API_CITATION_SIMPLE = """
## API Usage

Use `search_docs` to look up unfamiliar vectorbt APIs (e.g., `vbt.RSI.run` signature).
Use fully-qualified module paths (e.g., `pd.Series.rolling()`, not `.rolling()`).
"""

API_CITATION_MEDIUM = """
## API Usage

Use `search_docs` to look up unfamiliar vectorbt APIs (e.g., `vbt.MACD.run`,
`vbt.ATR.run`, `order_nb` signatures). Use fully-qualified module paths.
"""

API_CITATION_COMPLEX = """
## API Usage

Use `search_docs` to look up unfamiliar vectorbt APIs (e.g., `vbt.Portfolio.from_order_func`,
`order_nb` signatures) and scipy APIs if needed. Use fully-qualified module paths.
"""
