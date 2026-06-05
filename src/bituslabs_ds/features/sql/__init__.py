"""SQL CTE builders for the feature engineering pipeline.

Each module emits one CTE (or a small related group) as a plain SQL string.
``pipeline.py`` orchestrates them into the two final queries:
``compose_enriched_query`` and ``compose_grouped_query``.
"""
