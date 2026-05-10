import re
from typing import Dict, Optional

import pandas as pd


COMPARATORS = {
    "greater than": ">",
    "more than": ">",
    "above": ">",
    "over": ">",
    ">": ">",
    "less than": "<",
    "below": "<",
    "under": "<",
    "<": "<",
    "equal to": "==",
    "equals": "==",
    "is": "==",
    "=": "==",
}

ROW_QUERY_WORDS = (
    "who",
    "which",
    "list",
    "show rows",
    "show data",
    "get me data",
    "records",
    "students",
    "people",
    "entries",
)

AGGREGATE_QUERY_WORDS = (
    "percentage",
    "percent",
    "proportion",
    "ratio",
    "rate",
    "count",
    "how many",
    "distribution",
    "breakdown",
    "compare",
    "vs",
    "versus",
)

TRUTHY_VALUES = {"1", "yes", "y", "true", "placed", "selected", "pass", "passed"}


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def _find_column(query: str, columns) -> Optional[str]:
    normalized_query = _normalize(query)
    matches = []
    for column in columns:
        normalized_column = _normalize(column)
        if normalized_column and normalized_column in normalized_query:
            matches.append((len(normalized_column), column))
    if not matches:
        return None
    return sorted(matches, reverse=True)[0][1]


def _apply_numeric_filters(query: str, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    filtered = df
    applied = []
    query_text = query

    for column in df.select_dtypes(include=["number"]).columns:
        column_pattern = re.escape(str(column))
        for phrase, operator in COMPARATORS.items():
            pattern = rf"\b{column_pattern}\b\s*(?:is\s+)?{re.escape(phrase)}\s*(-?\d+(?:\.\d+)?)"
            match = re.search(pattern, query_text, re.IGNORECASE)
            if not match:
                continue

            value = float(match.group(1))
            if operator == ">":
                filtered = filtered[filtered[column] > value]
            elif operator == "<":
                filtered = filtered[filtered[column] < value]
            else:
                filtered = filtered[filtered[column] == value]
            applied.append(f"{column} {operator} {value:g}")
            break

    return filtered, applied


def _apply_boolean_intent_filters(query: str, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    filtered = df
    applied = []
    query_lower = query.lower()

    if "placed" in query_lower:
        column = _find_column("placed", df.columns)
        if column:
            values = df[column].astype(str).str.strip().str.lower()
            filtered = filtered[values.isin(TRUTHY_VALUES)]
            applied.append(f"{column} is truthy/placed")

    return filtered, applied


def answer_table_query(query: str, df: pd.DataFrame, max_rows: int = 100) -> Optional[Dict]:
    query_lower = query.lower()
    if any(word in query_lower for word in AGGREGATE_QUERY_WORDS):
        return None

    if not any(word in query_lower for word in ROW_QUERY_WORDS):
        return None

    filtered, applied_numeric = _apply_numeric_filters(query, df)
    filtered, applied_boolean = _apply_boolean_intent_filters(query, filtered)
    applied_filters = applied_numeric + applied_boolean

    if not applied_filters:
        return None

    result = filtered.head(max_rows).copy()
    result = result.where(pd.notnull(result), None)

    return {
        "status": "success",
        "mode": "table",
        "columns": [str(column) for column in result.columns],
        "rows": result.to_dict(orient="records"),
        "row_count": int(len(filtered)),
        "returned_count": int(len(result)),
        "filters": applied_filters,
        "interpretation": (
            f"Found {len(filtered)} matching rows using filters: "
            + ", ".join(applied_filters)
        ),
    }
