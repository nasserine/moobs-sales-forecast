"""Entraînement final des modèles de forecasting retenus."""

from pathlib import Path

import joblib

from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from moobs.data import load_data, prepare_data
from moobs.features import create_features
from moobs.dataset import build_dataset
from moobs.train import train_model
from moobs.validation import SHORT_LAGS, ROLLING, DEMOGRAPHIC


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"

# Configurations sélectionnées lors du backtesting.
FINAL_CONFIGS = {
    1: {
        "features": SHORT_LAGS + ROLLING,
        "algorithm": "XGBoost",
    },
    2: {
        "features": SHORT_LAGS + ROLLING,
        "algorithm": "XGBoost",
    },
    4: {
        "features": SHORT_LAGS + ROLLING,
        "algorithm": "XGBoost",
    },
    12: {
        "features": DEMOGRAPHIC,
        "algorithm": "LinearRegression",
    },
}


def make_final_model(algorithm: str):
    """Crée le modèle avec les paramètres du backtesting."""
    if algorithm == "XGBoost":
        return XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )

    if algorithm == "LinearRegression":
        return LinearRegression()

    raise ValueError(f"Algorithme inconnu : {algorithm}")


def train_final_models():
    """Entraîne et sauvegarde les quatre modèles finaux."""
    raw_df = prepare_data(load_data())
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for horizon, config in FINAL_CONFIGS.items():
        print(f"\nEntraînement : horizon {horizon} semaine(s)")

        feature_df = create_features(raw_df, horizon=horizon)

        # Toutes les observations historiques dont la cible est connue.
        # Aucun jeu de test n'est créé pour cet entraînement final.
        split = build_dataset(
            feature_df,
            features=config["features"],
            cutoff_date=feature_df["target_date"].max()
            + __import__("pandas").Timedelta(days=1),
            target="target_sales",
        )

        model = make_final_model(config["algorithm"])
        model = train_model(model, split.X_train, split.y_train)

        model_path = MODELS_DIR / f"sales_forecast_h{horizon}.joblib"
        joblib.dump(
            {
                "model": model,
                "features": config["features"],
                "horizon": horizon,
                "algorithm": config["algorithm"],
            },
            model_path,
        )

        print(f"Observations d'entraînement : {len(split.X_train)}")
        print(f"Modèle sauvegardé : {model_path}")

    print("\nEntraînement final terminé.")


if __name__ == "__main__":
    train_final_models()