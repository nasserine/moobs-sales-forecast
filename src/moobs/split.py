
import pandas as pd


def temporal_split(
    df: pd.DataFrame,
    cutoff_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sépare les données en entraînement et test selon target_date.

    Train : target_date < cutoff_date
    Test  : target_date >= cutoff_date
    """
    if "target_date" not in df.columns:
        raise ValueError("La colonne target_date est absente.")

    df = df.copy()
    df["target_date"] = pd.to_datetime(df["target_date"])
    cutoff_date = pd.Timestamp(cutoff_date)

    train_df = df[df["target_date"] < cutoff_date].copy()
    test_df = df[df["target_date"] >= cutoff_date].copy()

    return train_df, test_df
