import os
import requests
import pandas as pd
from dotenv import load_dotenv

# ======================================================
# Chargement des variables d'environnement (.env)
# ======================================================
load_dotenv()

FRED_API_KEY = os.getenv("FRED_API_KEY")

if FRED_API_KEY is None:
    raise ValueError("FRED_API_KEY non trouvée. Vérifie le fichier .env.")

# ======================================================
# Paramètres FRED
# ======================================================
SERIES_ID = "GVZCLS"  # CBOE Gold ETF Volatility Index
BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# ======================================================
# Fonction de récupération des données
# ======================================================
def fetch_fred_series(start_date="2006-01-01"):
    print("=== Début de la récupération des données FRED ===")
    print(f"Série demandée : {SERIES_ID}")

    params = {
        "series_id": SERIES_ID,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": start_date
    }

    response = requests.get(BASE_URL, params=params, timeout=10)
    response.raise_for_status()

    json_data = response.json()

    if "observations" not in json_data:
        raise ValueError("Aucune observation trouvée dans la réponse FRED.")

    df = pd.DataFrame(json_data["observations"])

    # Nettoyage
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])

    print("=== Données récupérées avec succès ===")
    print(f"Nombre d'observations : {len(df)}")

    return df[["date", "value"]]

# ======================================================
# Exécution directe du script
# ======================================================
if __name__ == "__main__":
    print("SCRIPT LANCÉ")
    df = fetch_fred_series()
    print("\nAperçu des dernières observations :")
    print(df.tail())
