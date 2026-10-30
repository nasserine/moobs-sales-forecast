
"""Prédictions avec les modèles de ventes sauvegardés."""

from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"

SUPPORTED_HORIZONS = (1, 2, 4, 12)


def predict_from_dataframe(
    input_df: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """Prédit les ventes à partir d'un tableau contenant les variables requises."""

    if horizon not in SUPPORTED_HORIZONS:
        raise ValueError(
            f"Horizon invalide : {horizon}. "
            f"Valeurs possibles : {SUPPORTED_HORIZONS}"
        )

    model_path = MODELS_DIR / f"sales_forecast_h{horizon}.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {model_path}"
        )

    saved_model = joblib.load(model_path)
    model = saved_model["model"]
    features = saved_model["features"]

    missing_features = [
        feature for feature in features
        if feature not in input_df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Variables manquantes pour l'horizon {horizon} : "
            f"{missing_features}"
        )

    X = input_df[features].copy()

    if X.isna().any().any():
        raise ValueError(
            "Le tableau contient des valeurs manquantes dans les variables requises."
        )

    predictions = model.predict(X)

    results = input_df.copy()
    results["predicted_sales"] = predictions

    return results
