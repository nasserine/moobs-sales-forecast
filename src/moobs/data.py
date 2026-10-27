from pathlib import Path

import pandas as pd


# Racine du projet : .../moobs-sales-forecast
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "sapin_dataset_stores.csv"
REQUIRED_COLUMNS = {"store_id", "date", "sales"}


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    """Charge le CSV des ventes depuis le chemin fourni ou le chemin par défaut."""
    csv_path = Path(path) if path is not None else DATA_PATH

    if not csv_path.exists():
        raise FileNotFoundError(f"Fichier de données introuvable : {csv_path}")

    return pd.read_csv(csv_path)


def validate_data(df: pd.DataFrame) -> None:
    """Valide les colonnes, les valeurs et la fréquence hebdomadaire."""
    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Colonnes obligatoires manquantes : {sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError("Le jeu de données est vide.")

    if df["store_id"].isna().any():
        raise ValueError("La colonne 'store_id' contient des valeurs manquantes.")

    if df["date"].isna().any():
        raise ValueError("La colonne 'date' contient des valeurs manquantes.")

    if df["sales"].isna().any():
        raise ValueError("La colonne 'sales' contient des valeurs manquantes.")

    try:
        parsed_dates = pd.to_datetime(df["date"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "La colonne 'date' contient des valeurs qui ne sont pas des dates valides."
        ) from exc

    try:
        pd.to_numeric(df["sales"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "La colonne 'sales' doit contenir uniquement des valeurs numériques."
        ) from exc

    duplicate_mask = pd.DataFrame(
        {"store_id": df["store_id"], "date": parsed_dates},
        index=df.index,
    ).duplicated(subset=["store_id", "date"], keep=False)

    if duplicate_mask.any():
        examples = df.loc[duplicate_mask, ["store_id", "date"]].head(10)
        raise ValueError(
            "Plusieurs observations existent pour un même magasin et une même date. "
            f"Exemples : {examples.to_dict(orient='records')}"
        )

    # Vérifie la fréquence hebdomadaire séparément pour chaque magasin.
    check_df = pd.DataFrame(
        {"store_id": df["store_id"].to_numpy(), "date": parsed_dates.to_numpy()}
    ).sort_values(["store_id", "date"])

    gaps_days = check_df.groupby("store_id", sort=False)["date"].diff().dt.days
    irregular_mask = gaps_days.notna() & gaps_days.ne(7)

    if irregular_mask.any():
        examples = check_df.loc[irregular_mask, ["store_id", "date"]].copy()
        examples["gap_days"] = gaps_days.loc[irregular_mask]
        raise ValueError(
            "La série temporelle n'est pas strictement hebdomadaire : certains "
            "écarts entre observations successives ne font pas 7 jours. "
            "Aucune semaine manquante n'a été créée automatiquement. "
            f"Exemples : {examples.head(10).to_dict(orient='records')}"
        )


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Valide les données, convertit les types et trie par magasin/date."""
    prepared_df = df.copy()

    # Centralise la vérification des colonnes avant d'accéder à leurs valeurs.
    missing_columns = REQUIRED_COLUMNS - set(prepared_df.columns)
    if missing_columns:
        raise ValueError(
            f"Colonnes obligatoires manquantes : {sorted(missing_columns)}"
        )

    try:
        prepared_df["date"] = pd.to_datetime(
            prepared_df["date"], errors="raise"
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "La colonne 'date' contient des valeurs qui ne sont pas des dates valides."
        ) from exc

    try:
        prepared_df["sales"] = pd.to_numeric(
            prepared_df["sales"], errors="raise"
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "La colonne 'sales' doit contenir uniquement des valeurs numériques."
        ) from exc

    validate_data(prepared_df)

    return (
        prepared_df.sort_values(["store_id", "date"])
        .reset_index(drop=True)
    )
