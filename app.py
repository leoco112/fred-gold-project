import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL
import statsmodels.api as sm

from src.data_cleaning import load_and_clean_fred_series

# ======================================================
# App configuration
# ======================================================
st.set_page_config(page_title="FRED Multi-Series Analyzer", layout="centered")

st.title("FRED Multi-Series Analyzer")
st.write(
    "Simple data-driven application for the analysis of up to three FRED time series."
)

# ======================================================
# User inputs
# ======================================================
st.subheader("FRED links (max 3)")

url1 = st.text_input("FRED URL 1 (required)")
url2 = st.text_input("FRED URL 2 (optional)")
url3 = st.text_input("FRED URL 3 (optional)")

st.subheader("Options")
resample_weekly = st.checkbox(
    "Aggregate data to weekly frequency",
    value=False,
    help="Recommended for STL decomposition (seasonal period = 52)."
)

run = st.button("Run analysis")

# ======================================================
# Main logic
# ======================================================
if run and url1:

    try:
        # ----------------------------------------------
        # 1) Extract series IDs
        # ----------------------------------------------
        urls = [url1, url2, url3]
        series_ids = [
            u.rstrip("/").split("/")[-1]
            for u in urls if u is not None and u.strip() != ""
        ]

        # ----------------------------------------------
        # 2) Load, clean, merge
        # ----------------------------------------------
        df = load_and_clean_fred_series(series_ids, weekly=resample_weekly)

        st.subheader("Cleaned & merged data")
        st.write(df.head())

        # ----------------------------------------------
        # 3) Description
        # ----------------------------------------------
        st.subheader("Series description")

        for col in df.columns:
            if col != "date":
                st.markdown(f"**{col}**")
                st.write(f"- Observations: {df[col].notna().sum()}")
                st.write(
                    f"- Date range: {df['date'].min().date()} → {df['date'].max().date()}"
                )

        # ----------------------------------------------
        # 4) Train / Test split (80 / 20)
        # ----------------------------------------------
        split_idx = int(0.8 * len(df))
        train = df.iloc[:split_idx].copy()
        test = df.iloc[split_idx:].copy()

        # ----------------------------------------------
        # 5) Standardisation
        # ----------------------------------------------
        scalers = {}
        for col in df.columns:
            if col != "date":
                scaler = StandardScaler()
                train[col + "_std"] = scaler.fit_transform(train[[col]])
                test[col + "_std"] = scaler.transform(test[[col]])
                scalers[col] = scaler

        # ----------------------------------------------
        # 6) Per-series analysis
        # ----------------------------------------------
        for col in df.columns:
            if col == "date":
                continue

            st.divider()
            st.header(f"Analysis of {col}")

            y_train = train[col + "_std"]

            # ----- ADF test
            adf_stat, pval, *_ = adfuller(y_train)
            st.write(f"ADF statistic: {adf_stat:.3f}")
            st.write(f"p-value: {pval:.4f}")

            d = 0 if pval < 0.05 else 1
            st.write(f"Selected differencing order: d = {d}")

            # ----- STL (only if weekly)
            if resample_weekly:
                try:
                    stl = STL(
                        train.set_index("date")[col],
                        period=52,
                        robust=True
                    )
                    res = stl.fit()

                    fig, ax = plt.subplots(4, 1, figsize=(8, 6), sharex=True)
                    ax[0].plot(res.observed); ax[0].set_title("Observed")
                    ax[1].plot(res.trend); ax[1].set_title("Trend")
                    ax[2].plot(res.seasonal); ax[2].set_title("Seasonal")
                    ax[3].plot(res.resid); ax[3].set_title("Residual")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception:
                    st.info("STL decomposition failed.")
            else:
                st.info("STL available only with weekly aggregation.")

            # ----- Distribution
            fig, ax = plt.subplots()
            sns.histplot(df[col], bins=40, kde=True, ax=ax)
            ax.set_title("Distribution")
            st.pyplot(fig)
            plt.close(fig)

            # ----- Mean per year
            mean_year = (
                df.assign(year=df["date"].dt.year)
                  .groupby("year")[col]
                  .mean()
                  .to_frame("mean")
            )
            st.write("Average per year")
            st.dataframe(mean_year)

            # ----- ARMA / ARIMA
            model = sm.tsa.ARIMA(y_train, order=(1, d, 1)).fit()
            st.write("ARMA / ARIMA summary")
            st.text(model.summary())

        # ----------------------------------------------
        # 7) Joint plot (standardized)
        # ----------------------------------------------
        st.divider()
        st.header("Joint visualization (standardized series)")

        fig, ax = plt.subplots(figsize=(9, 4))
        for col in df.columns:
            if col != "date":
                ax.plot(
                    df["date"],
                    scalers[col].transform(df[[col]]),
                    label=col
                )

        ax.legend()
        ax.set_title("Standardized series")
        st.pyplot(fig)
        plt.close(fig)

        # ----------------------------------------------
        # 8) Correlation
        # ----------------------------------------------
        st.divider()
        st.header("Correlation between series")

        std_cols = [c + "_std" for c in df.columns if c != "date"]
        corr = pd.concat(
            [train[std_cols], test[std_cols]]
        ).corr()

        st.dataframe(corr)

        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
        plt.close(fig)

    except Exception as e:
        st.error(str(e))
