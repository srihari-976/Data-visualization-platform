import pandas as pd


def clean_for_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean user-uploaded data while preserving original columns for querying."""
    cleaned = df.copy()
    missing_before = cleaned.isnull().sum().to_dict()
    fill_values = {}

    for column in cleaned.columns:
        series = cleaned[column]

        if pd.api.types.is_numeric_dtype(series):
            value = series.mean()
            if pd.isna(value):
                value = 0
            cleaned[column] = series.fillna(value)
            fill_values[column] = {
                "strategy": "mean",
                "value": float(value),
            }
        else:
            mode = series.dropna().mode()
            value = mode.iloc[0] if not mode.empty else "Unknown"
            cleaned[column] = series.fillna(value)
            fill_values[column] = {
                "strategy": "mode",
                "value": str(value),
            }

    missing_after = cleaned.isnull().sum().to_dict()

    return cleaned, {
        "missing_before": {str(k): int(v) for k, v in missing_before.items()},
        "missing_after": {str(k): int(v) for k, v in missing_after.items()},
        "fill_values": fill_values,
    }
