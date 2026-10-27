
import pandas as pd


def create_features(
    df: pd.DataFrame,
    horizon: int = 2,
) -> pd.DataFrame:
    """
    Construit la cible et les variables explicatives temporelles.

    Convention :
    - À la date t, les ventes connues vont jusqu'à t.
    - La cible est la vente à t + horizon.
    - Les lags et rolling features utilisent uniquement
      les observations antérieures à t pour les rolling,
      et les valeurs historiques définies par leur lag.

    Parameters
    ----------
    df : pd.DataFrame
        Données préparées avec store_id, date et sales.
    horizon : int, default=2
        Nombre de semaines entre la date d'origine et la cible.

    Returns
    -------
    pd.DataFrame
        DataFrame enrichi avec la cible et les variables temporelles.
    """
    required_columns = {"store_id", "date", "sales"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Colonnes obligatoires manquantes : {sorted(missing_columns)}"
        )

    if horizon < 1:
        raise ValueError("horizon doit être supérieur ou égal à 1.")

    features_df = df.copy()

    features_df["date"] = pd.to_datetime(features_df["date"])
    features_df = features_df.sort_values(
        ["store_id", "date"]
    ).reset_index(drop=True)

    grouped_sales = features_df.groupby("store_id")["sales"]

    # Cible : ventes à t + horizon
    features_df["target_date"] = (
        features_df.groupby("store_id")["date"].shift(-horizon)
    )
    features_df["target_sales"] = grouped_sales.shift(-horizon)

    # Variables calendaires à la date cible
    features_df["target_year"] = features_df["target_date"].dt.isocalendar().year
    features_df["target_week"] = features_df["target_date"].dt.isocalendar().week

    # Lags : ventes connues à t-1, t-2, etc.
    for lag in [1, 2, 4, 26, 52]:
        features_df[f"lag_{lag}"] = grouped_sales.shift(lag)

    # Ventes à la date d'origine t, connues au moment de la prévision
    features_df["sales_t"] = features_df["sales"]

    # Rolling features : uniquement les semaines t-4 à t-1
    features_df["rolling_mean_4"] = (
        grouped_sales.transform(
            lambda s: s.shift(1).rolling(window=4, min_periods=4).mean()
        )
    )

    features_df["rolling_mean_8"] = (
        grouped_sales.transform(
            lambda s: s.shift(1).rolling(window=8, min_periods=8).mean()
        )
    )

    features_df["rolling_std_4"] = (
        grouped_sales.transform(
            lambda s: s.shift(1).rolling(window=4, min_periods=4).std()
        )
    )

    return features_df
