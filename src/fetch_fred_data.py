import os
import requests
import pandas as pd
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ======================================================
# Load environment variables
# ======================================================
load_dotenv()

FRED_API_KEY = os.getenv("FRED_API_KEY")
if FRED_API_KEY is None:
    raise ValueError("FRED_API_KEY not found. Please check your .env file.")

BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# ======================================================
# HTTP session with retries (rate limits, network errors)
# ======================================================
def create_session() -> requests.Session:
    session = requests.Session()

    retry_strategy = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session


# ======================================================
# Fetch raw FRED data + basic data cleaning
# ======================================================
def fetch_fred_series(series_id: str) -> pd.DataFrame:
    """
    Fetch a FRED time series and perform basic data cleaning.

    Data cleaning steps:
    - Parsing dates into datetime format
    - Numeric coercion of values
    - Handling missing values
    """

    session = create_session()

    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json"
    }

    response = session.get(BASE_URL, params=params, timeout=10)
    response.raise_for_status()

    data = response.json()

    if "observations" not in data:
        raise ValueError("No observations found for this FRED series.")

    # -------------------------------
    # Data cleaning
    # -------------------------------
    df = pd.DataFrame(data["observations"])

    # Parse dates
    df["date"] = pd.to_datetime(df["date"])

    # Numeric coercion (FRED missing values such as ".")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Remove missing values
    df = df.dropna(subset=["value"])

    return df[["date", "value"]]
