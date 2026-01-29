import os
import requests
import pandas as pd
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Final

from src.logger import setup_logger

logger = setup_logger(__name__)

load_dotenv()

FRED_API_KEY = os.getenv("FRED_API_KEY")
if FRED_API_KEY is None:
    raise ValueError("FRED_API_KEY not found. Please check your .env file.")

BASE_URL: Final[str] = "https://api.stlouisfed.org/fred/series/observations"


def create_session() -> requests.Session:
    """
    HTTP session with retry strategy (rate limits, transient errors).
    """
    session = requests.Session()
    retry_strategy = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def fetch_fred_series(series_id: str, timeout: int = 10) -> pd.DataFrame:
    """
    Fetch a FRED series and apply basic cleaning.

    Returns
    -------
    DataFrame with columns: ['date', 'value'] sorted by date.
    """
    if not series_id or not isinstance(series_id, str):
        raise ValueError("Invalid series_id. Please provide a non-empty string.")

    session = create_session()
    params = {"series_id": series_id, "api_key": FRED_API_KEY, "file_type": "json"}

    logger.info("Fetching FRED series: %s", series_id)
    response = session.get(BASE_URL, params=params, timeout=timeout)
    response.raise_for_status()

    data = response.json()
    if "observations" not in data:
        raise ValueError("No observations found for this FRED series.")

    df = pd.DataFrame(data["observations"])
    if df.empty:
        raise ValueError("Empty observations returned by the API.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)

    if len(df) < 20:
        raise ValueError("Series too short (< 20 observations) for meaningful analysis.")

    return df[["date", "value"]]
