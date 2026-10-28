
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Évalue les prédictions avec la MAE et la RMSE.

    y_true : ventes réellement observées.
    y_pred : ventes prédites par le modèle.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
    }
