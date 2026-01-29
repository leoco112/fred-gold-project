import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from src.fetch_fred_data import fetch_fred_series

# ======================================================
# Streamlit configuration
# ======================================================
st.set_page_config(page_title="FRED Time Series Analyzer", layout="centered")

st.title("FRED Time Series Analyzer")
st.write(
    "End-to-end pipeline for FRED time series analysis: "
    "temporal disaggregation, preprocessing, STL decomposition, "
    "ARMA/ARIMA modeling, rolling cross-validation, and benchmark comparison."
)

# ======================================================
# User input
# ======================================================
fred_url = st.text_input(
    "FRED series URL",
    placeholder="https://fred.stlouisfed.org/series/CPIAUCSL"
)

convert_weekly = st.checkbox(
    "Convert monthly data to weekly frequency (linear interpolation)",
    value=True
)

run = st.button("Run analysis")

# ======================================================
# Helper: monthly → weekly interpolation
# ======================================================
def monthly_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.set_index("date").asfreq("MS")
    df = df.resample("W").interpolate(method="linear")
    return df.reset_index()

# ======================================================
# Helper: rolling cross-validation
# ======================================================
def rolling_cv(series, order, initial_window, horizon=1):
    """
    Expanding-window rolling cross-validation for ARMA / ARIMA models.
    """
    errors = []

    for t in range(initial_window, len(series) - horizon):
        train = series.iloc[:t]

        try:
            model = sm.tsa.ARIMA(train, order=order).fit()
            forecast = model.get_forecast(steps=horizon)
            pred = forecast.predicted_mean.iloc[-1]
            true = series.iloc[t]
            errors.append(true - pred)
        except Exception:
            continue

    return np.array(errors)

# ======================================================
# Main logic
# ======================================================
if fred_url and run:
    try:
        # ==================================================
        # 1. Data retrieval
        # ==================================================
        series_id = fred_url.rstrip("/").split("/")[-1]
        df = fetch_fred_series(series_id)
        df = df.sort_values("date").reset_index(drop=True)

        if convert_weekly:
            df = monthly_to_weekly(df)
            st.info("Monthly data converted to weekly frequency via linear interpolation.")

        # ==================================================
        # 2. Preprocessing
        # ==================================================
        scaler = StandardScaler()
        df["value_std"] = scaler.fit_transform(df[["value"]])

        # ==================================================
        # 3. STL decomposition (DESCRIPTIVE)
        # ==================================================
        st.subheader("STL decomposition (descriptive analysis)")

        if len(df) >= 2 * 52:
            stl = STL(
                df.set_index("date")["value"],
                period=52,
                robust=True
            ).fit()

            fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
            axes[0].plot(stl.observed); axes[0].set_title("Observed")
            axes[1].plot(stl.trend); axes[1].set_title("Trend")
            axes[2].plot(stl.seasonal); axes[2].set_title("Seasonal")
            axes[3].plot(stl.resid); axes[3].set_title("Residual")
            plt.tight_layout()
            st.pyplot(fig)

        # ==================================================
        # 4. Stationarity test (ADF)
        # ==================================================
        adf_stat, adf_pvalue, *_ = adfuller(df["value_std"].dropna(), autolag="AIC")

        st.subheader("Stationarity test (ADF)")
        st.write(f"ADF p-value: **{adf_pvalue:.4f}**")

        stationary = adf_pvalue < 0.05

        if stationary:
            order = (1, 0, 1)
            model_name = "ARMA(1,1)"
        else:
            order = (1, 1, 1)
            model_name = "ARIMA(1,1,1)"

        st.write(f"Selected model: **{model_name}**")

        # ==================================================
        # 5. Rolling cross-validation
        # ==================================================
        st.subheader("Rolling cross-validation")

        initial_window = int(0.5 * len(df))
        errors = rolling_cv(
            df["value_std"],
            order=order,
            initial_window=initial_window
        )

        rmse_cv = np.sqrt(np.mean(errors**2))
        mae_cv = np.mean(np.abs(errors))

        st.write(f"CV RMSE: **{rmse_cv:.4f}**")
        st.write(f"CV MAE: **{mae_cv:.4f}**")

        # ==================================================
        # 6. Naive benchmark (random walk)
        # ==================================================
        naive_errors = df["value_std"].diff().dropna()

        rmse_naive = np.sqrt(np.mean(naive_errors**2))
        mae_naive = np.mean(np.abs(naive_errors))

        st.write("Naive random walk benchmark:")
        st.write(f"RMSE: **{rmse_naive:.4f}**")
        st.write(f"MAE: **{mae_naive:.4f}**")

        # ==================================================
        # 7. Comparison table
        # ==================================================
        results = pd.DataFrame(
            {
                "RMSE": [rmse_cv, rmse_naive],
                "MAE": [mae_cv, mae_naive]
            },
            index=[model_name, "Naive random walk"]
        )

        st.dataframe(results)

    except Exception as e:
        st.error("An error occurred during analysis.")
        st.exception(e)
