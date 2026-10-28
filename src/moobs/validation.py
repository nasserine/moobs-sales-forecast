
"""Backtesting temporel pour comparer les stratégies de forecasting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from moobs.data import load_data, prepare_data
from moobs.dataset import build_dataset, common_test_index
from moobs.evaluate import evaluate_model
from moobs.features import create_features
from moobs.train import predict, train_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"

# Chaque période teste les prévisions sur trois mois.
BACKTEST_PERIODS = [
    ("2024-04-01", "2024-07-01"),
    ("2024-07-01", "2024-10-01"),
    ("2024-10-01", "2025-01-01"),
]

HORIZONS = [1, 2, 4, 12]

# Variables de base.
SHORT_LAGS = ["sales_t", "lag_1", "lag_2", "lag_4"]
LONG_LAGS = ["lag_26", "lag_52"]
ROLLING = [
    "rolling_mean_4",
    "rolling_mean_8",
    "rolling_std_4",
]
DEMOGRAPHIC = [
    "Population",
    "Pregnancy_T1",
    "Pregnancy_T2",
    "Pregnancy_T3",
    "Birth",
    "First_baby_birth",
    "Infants",
    "sellers",
]
CALENDAR = ["target_year", "target_week"]

# Stratégies comparées.
FEATURE_SETS = {
    "A_short_lags": SHORT_LAGS,
    "B_lag52": SHORT_LAGS + ["lag_52"],
    "C_long_lags": SHORT_LAGS + LONG_LAGS,
    "D_rolling_only": ROLLING,
    "E_lags_rolling": SHORT_LAGS + ROLLING,
    "F_demographic": DEMOGRAPHIC,
    "G_all": (
        SHORT_LAGS
        + LONG_LAGS
        + ROLLING
        + DEMOGRAPHIC
        + CALENDAR
    ),
}


def make_models() -> dict:
    """Crée les algorithmes à comparer."""
    return {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=42,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        ),
    }


def run_backtesting(
    horizons: list[int] | None = None,
    periods: list[tuple[str, str]] | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Compare le modèle naïf et plusieurs algorithmes.

    Pour chaque horizon et période :
    - l'entraînement utilise les cibles antérieures au test ;
    - tous les modèles sont évalués sur les mêmes lignes de test ;
    - les résultats MAE et RMSE sont enregistrés.

    Le modèle naïf prédit target_sales avec sales_t.
    """
    horizons = HORIZONS if horizons is None else horizons
    periods = BACKTEST_PERIODS if periods is None else periods

    if not horizons or any(h < 1 for h in horizons):
        raise ValueError("Les horizons doivent être des entiers positifs.")

    if not periods:
        raise ValueError("Au moins une période de backtesting est nécessaire.")

    # Chargement et préparation une seule fois.
    raw_df = prepare_data(load_data())
    all_results = []

    for horizon in horizons:
        print(f"\n{'=' * 65}")
        print(f"HORIZON : {horizon} semaine(s)")
        print(f"{'=' * 65}")

        # La cible dépend de l'horizon.
        feature_df = create_features(raw_df, horizon=horizon)

        for start, end in periods:
            print(f"\nBacktest : {start} -> {end}")

            start_date = pd.Timestamp(start)
            end_date = pd.Timestamp(end)

            if start_date >= end_date:
                raise ValueError(
                    f"Période invalide : {start} -> {end}"
                )

            # On retire les cibles postérieures à la fin du test.
            period_df = feature_df.loc[
                feature_df["target_date"].lt(end_date)
            ].copy()

            # Les mêmes lignes de test sont imposées à toutes les
            # stratégies de variables. Cela inclut les lags longs.
            test_index = common_test_index(
                period_df,
                feature_sets=list(FEATURE_SETS.values()),
                cutoff_date=start_date,
                target="target_sales",
            )

            if len(test_index) == 0:
                print("Aucune ligne de test commune : période ignorée.")
                continue

            # ---------- Modèle naïf ----------
            naive_split = build_dataset(
                period_df,
                features=["sales_t"],
                cutoff_date=start_date,
                target="target_sales",
                test_index=test_index,
            )

            naive_predictions = naive_split.X_test["sales_t"]
            naive_metrics = evaluate_model(
                naive_split.y_test,
                naive_predictions,
            )

            all_results.append({
                "horizon": horizon,
                "period_start": start,
                "period_end": end,
                "feature_set": "Naive",
                "algorithm": "NaiveLastValue",
                "n_train": len(naive_split.X_train),
                "n_test": len(naive_split.X_test),
                "MAE": naive_metrics["MAE"],
                "RMSE": naive_metrics["RMSE"],
            })

            print(
                f"Naïf | MAE={naive_metrics['MAE']:.2f} "
                f"| RMSE={naive_metrics['RMSE']:.2f}"
            )

            # ---------- Stratégies et algorithmes ----------
            for feature_name, features in FEATURE_SETS.items():
                split = build_dataset(
                    period_df,
                    features=features,
                    cutoff_date=start_date,
                    target="target_sales",
                    test_index=test_index,
                )

                for algorithm_name, model in make_models().items():
                    # Chaque algorithme reçoit un modèle neuf.
                    trained_model = train_model(
                        model,
                        split.X_train,
                        split.y_train,
                    )

                    predictions = predict(
                        trained_model,
                        split.X_test,
                    )

                    metrics = evaluate_model(
                        split.y_test,
                        predictions,
                    )

                    all_results.append({
                        "horizon": horizon,
                        "period_start": start,
                        "period_end": end,
                        "feature_set": feature_name,
                        "algorithm": algorithm_name,
                        "n_train": len(split.X_train),
                        "n_test": len(split.X_test),
                        "MAE": metrics["MAE"],
                        "RMSE": metrics["RMSE"],
                    })

                    print(
                        f"{feature_name:18s} | "
                        f"{algorithm_name:20s} | "
                        f"MAE={metrics['MAE']:.2f} | "
                        f"RMSE={metrics['RMSE']:.2f}"
                    )

    results_df = pd.DataFrame(all_results)

    if results_df.empty:
        raise ValueError(
            "Aucun résultat obtenu. Vérifie les dates et les données."
        )

    # Export des résultats détaillés.
    destination = (
        Path(output_path)
        if output_path is not None
        else REPORTS_DIR / "backtesting_results.csv"
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(destination, index=False)

    print(f"\nRésultats enregistrés dans : {destination}")
    print(f"Nombre d'évaluations : {len(results_df)}")

    return results_df


if __name__ == "__main__":
    results = run_backtesting()

    # Classement global par MAE moyenne, à titre exploratoire.
    ranking = (
        results.groupby(
            ["horizon", "feature_set", "algorithm"],
            as_index=False,
        )
        .agg(
            MAE_moyenne=("MAE", "mean"),
            RMSE_moyenne=("RMSE", "mean"),
            Nombre_backtests=("MAE", "count"),
        )
        .sort_values(["horizon", "MAE_moyenne"])
    )

    print("\n=== Classement par horizon (MAE moyenne) ===")

    for horizon, group in ranking.groupby("horizon"):
        print(f"\nHorizon : {horizon} semaine(s)")
        print(group.head(10).to_string(index=False))
