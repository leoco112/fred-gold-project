import pandas as pd
from typing import List
from src.fetch_fred_data import fetch_fred_series


def load_and_clean_fred_series(
    series_ids: List[str],
    weekly: bool = False,
) -> pd.DataFrame:
    """
    Download up to 3 FRED series, clean them, optionally resample weekly,
    and merge them on common dates.

    Parameters
    ----------
    series_ids : list of str
        FRED series IDs (1 to 3).
    weekly : bool
        If True, aggregate data to weekly frequency (mean).

    Returns
    -------
    pd.DataFrame
        date | series_1 | series_2 | series_3
    """

    if len(series_ids) == 0:
        raise ValueError("At least one FRED series is required.")
    if len(series_ids) > 3:
        raise ValueError("Maximum 3 FRED series allowed.")

    dfs = []

    for i, sid in enumerate(series_ids, start=1):
        df = fetch_fred_series(sid)
        df = df.rename(columns={"value": f"series_{i}"})
        df = df.set_index("date")

        # Optional weekly aggregation
        if weekly:
            df = df.resample("W").mean()

        dfs.append(df)

    # Merge on common dates
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.join(df, how="inner")

    merged = merged.dropna().reset_index()

    if len(merged) < 30:
        raise ValueError("Too few observations after merging.")

    return merged
