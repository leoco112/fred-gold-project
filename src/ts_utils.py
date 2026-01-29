from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR


@dataclass(frozen=True)
class FrequencyInfo:
    """
    Summary of inferred frequency and recommended seasonal period.
    """
    freq_label: str
    median_delta_days: float
    seasonal_period: Optional[int]  # e.g. 7, 12, 52, 365 or None


def infer_frequency(dates: pd.Series) -> FrequencyInfo:
    """
    Infer frequency from median date deltas (heuristic, robust for typical FRED series).
    """
    dates = pd.to_datetime(dates, errors="coerce").dropna().sort_values()
    deltas = dates.diff().dropna().dt.total_seconds() / (24 * 3600)

    if deltas.empty:
        return FrequencyInfo("irregular", np.nan, None)

    median_days = float(np.median(deltas))

    if 0.5 <= median_days <= 2.0:
        return FrequencyInfo("daily", median_days, 365)
    if 5.0 <= median_days <= 9.0:
        return FrequencyInfo("weekly", median_days, 52)
    if 25.0 <= median_days <= 35.0:
        return FrequencyInfo("monthly", median_days, 12)
    if 80.0 <= median_days <= 100.0:
        return FrequencyInfo("quarterly", median_days, 4)
    if 350.0 <= median_days <= 380.0:
        return FrequencyInfo("yearly", median_days, None)

    return FrequencyInfo("irregular", median_days, None)


def adf_stationarity_test(x: pd.Series, alpha: float = 0.05) -> Tuple[float, float, bool]:
    """
    ADF test: returns (statistic, p-value, is_stationary).
    """
    x_clean = pd.to_numeric(x, errors="coerce").dropna()
    if len(x_clean) < 20:
        raise ValueError("Not enough data for ADF test (< 20 non-missing values).")
    stat, pval, *_ = adfuller(x_clean, autolag="AIC")
    return float(stat), float(pval), bool(pval < alpha)


def try_stl_decomposition(
    y: pd.Series,
    period: Optional[int],
    robust: bool = True,
) -> Optional[Dict[str, pd.Series]]:
    """
    STL decomposition if period exists and length >= 2*period.
    """
    if period is None:
        return None
    y_clean = pd.to_numeric(y, errors="coerce").dropna()
    if len(y_clean) < 2 * period:
        return None

    stl = STL(y_clean, period=period, robust=robust)
    res = stl.fit()
    return {
        "observed": res.observed,
        "trend": res.trend,
        "seasonal": res.seasonal,
        "resid": res.resid,
    }


def seasonal_naive_forecast(train: pd.Series, test_len: int, season: int) -> np.ndarray:
    """
    Seasonal naive: repeats last observed seasonal cycle.
    """
    train_vals = np.asarray(train, dtype=float)
    if len(train_vals) < season:
        raise ValueError("Train sample shorter than season length for seasonal naive.")
    preds = np.empty(test_len, dtype=float)
    for i in range(test_len):
        idx = len(train_vals) - season + (i % season)
        preds[i] = train_vals[idx]
    return preds


def fit_arima_grid_aic(
    y: pd.Series,
    d: int,
    p_max: int = 3,
    q_max: int = 3,
) -> Tuple[sm.tsa.arima.model.ARIMAResults, Tuple[int, int, int], float]:
    """
    ARIMA(p,d,q) selection on a small grid using AIC.
    """
    y_clean = pd.to_numeric(y, errors="coerce").dropna()
    if len(y_clean) < 30:
        raise ValueError("Not enough data (< 30) for ARIMA grid search.")

    best_aic = np.inf
    best_order = (0, d, 0)
    best_model = None

    for p in range(p_max + 1):
        for q in range(q_max + 1):
            try:
                res = sm.tsa.ARIMA(y_clean, order=(p, d, q)).fit()
                aic = float(res.aic)
                if np.isfinite(aic) and aic < best_aic:
                    best_aic = aic
                    best_order = (p, d, q)
                    best_model = res
            except Exception:
                continue

    if best_model is None:
        raise RuntimeError("ARIMA grid search failed to fit any model.")

    return best_model, best_order, best_aic


def align_two_series(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    name1: str = "x",
    name2: str = "y",
) -> pd.DataFrame:
    """
    Inner join on date; returns columns: ['date', name1, name2].
    """
    a = df1[["date", "value"]].rename(columns={"value": name1})
    b = df2[["date", "value"]].rename(columns={"value": name2})
    merged = (
        a.merge(b, on="date", how="inner")
        .dropna()
        .sort_values("date")
        .reset_index(drop=True)
    )
    if len(merged) < 30:
        raise ValueError("Too few overlapping observations between the two series (< 30).")
    return merged


def rolling_correlation(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    """
    Rolling Pearson correlation.
    """
    if len(x) != len(y):
        raise ValueError("Series must have same length for rolling correlation.")
    if window < 5:
        raise ValueError("Rolling window too small.")
    return x.rolling(window=window).corr(y)


def fit_var_model(
    df: pd.DataFrame,
    cols: List[str],
    maxlags: int = 10,
    ic: str = "aic",
) -> Tuple[VAR, int]:
    """
    Fit a VAR model with lag selection by information criterion.

    Returns
    -------
    (fitted_results, selected_lag)
    """
    data = df[cols].dropna()
    if len(data) < 50:
        raise ValueError("Not enough observations (< 50) for VAR estimation.")

    model = VAR(data)
    sel = model.select_order(maxlags=maxlags)

    if ic.lower() == "aic":
        p = int(sel.aic)
    elif ic.lower() == "bic":
        p = int(sel.bic)
    elif ic.lower() == "hqic":
        p = int(sel.hqic)
    else:
        raise ValueError("ic must be one of: 'aic', 'bic', 'hqic'.")

    if p < 1:
        p = 1

    res = model.fit(p)
    return res, p
