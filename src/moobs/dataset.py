"""Construction des jeux d'entraînement et de test pour le forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd


@dataclass(frozen=True)
class DatasetSplit:
    """Jeux X/y et lignes source correspondant au découpage temporel."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    train_df: pd.DataFrame
    test_df: pd.DataFrame


def _validate_inputs(
    df: pd.DataFrame,
    features: Sequence[str],
    target: str,
) -> list[str]:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df doit être un pandas.DataFrame.")
    if df.empty:
        raise ValueError("Le DataFrame fourni est vide.")
    if not df.index.is_unique:
        raise ValueError(
            "L'index de df doit être unique pour pouvoir réutiliser des indices de lignes."
        )

    feature_list = list(features)
    if not feature_list:
        raise ValueError("La liste des variables explicatives ne peut pas être vide.")
    if len(feature_list) != len(set(feature_list)):
        raise ValueError("La liste des variables explicatives contient des doublons.")
    if target in feature_list:
        raise ValueError(
            f"La cible '{target}' ne doit pas figurer dans les variables explicatives."
        )

    required = set(feature_list) | {target, "target_date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes requises absentes : {sorted(missing)}")
    return feature_list


def usable_rows(
    df: pd.DataFrame,
    features: Sequence[str],
    target: str = "target_sales",
) -> pd.Series:
    """Renvoie un masque booléen des lignes complètes pour les features et la cible."""
    feature_list = _validate_inputs(df, features, target)
    required = feature_list + [target, "target_date"]
    return df[required].notna().all(axis=1).rename("usable")


def common_test_index(
    df: pd.DataFrame,
    feature_sets: Sequence[Sequence[str]],
    cutoff_date: str | pd.Timestamp,
    target: str = "target_sales",
) -> pd.Index:
    """Renvoie les lignes de test utilisables par tous les jeux de variables."""
    if not feature_sets:
        raise ValueError("Fournir au moins un jeu de variables.")
    cutoff = pd.Timestamp(cutoff_date)
    if pd.isna(cutoff):
        raise ValueError("cutoff_date doit être une date valide.")
    if "target_date" not in df.columns:
        raise ValueError("La colonne 'target_date' est absente.")

    target_dates = pd.to_datetime(df["target_date"], errors="coerce")
    common_mask = target_dates.ge(cutoff)
    for features in feature_sets:
        common_mask &= usable_rows(df, features, target=target)
    return df.index[common_mask]


def build_dataset(
    df: pd.DataFrame,
    features: Sequence[str],
    cutoff_date: str | pd.Timestamp,
    target: str = "target_sales",
    test_index: pd.Index | Sequence | None = None,
) -> DatasetSplit:
    """Prépare X/y après filtrage et séparation temporelle sur target_date.

    Si test_index est fourni, toutes les lignes demandées doivent exister, être
    utilisables pour ces variables et appartenir à la période de test (>= cutoff).
    """
    feature_list = _validate_inputs(df, features, target)
    cutoff = pd.Timestamp(cutoff_date)
    if pd.isna(cutoff):
        raise ValueError("cutoff_date doit être une date valide.")

    work = df.copy()
    work["target_date"] = pd.to_datetime(work["target_date"], errors="coerce")
    usable = usable_rows(work, feature_list, target=target)
    train_mask = usable & work["target_date"].lt(cutoff)
    test_period_mask = work["target_date"].ge(cutoff)

    train_df = work.loc[train_mask].copy()

    if test_index is None:
        test_mask = usable & test_period_mask
        test_df = work.loc[test_mask].copy()
    else:
        requested_index = pd.Index(test_index)
        if not requested_index.is_unique:
            raise ValueError("test_index contient des indices en double.")
        missing_index = requested_index.difference(work.index)
        if len(missing_index):
            raise ValueError(
                f"test_index contient des indices absents de df : {list(missing_index[:10])}"
            )
        requested_df = work.loc[requested_index]
        requested_usable = usable.loc[requested_index]
        if not requested_usable.all():
            bad = requested_index[~requested_usable.to_numpy()]
            raise ValueError(
                "Certaines lignes imposées dans test_index ne sont pas utilisables "
                f"pour les variables {feature_list} : {list(bad[:10])}"
            )
        requested_dates = requested_df["target_date"]
        if requested_dates.isna().any() or requested_dates.lt(cutoff).any():
            bad = requested_index[
                requested_dates.isna().to_numpy() | requested_dates.lt(cutoff).to_numpy()
            ]
            raise ValueError(
                "Toutes les lignes de test_index doivent avoir target_date >= "
                f"{cutoff.date()} : indices concernés {list(bad[:10])}"
            )
        # Conserver l'ordre fourni par test_index pour garantir l'alignement entre modèles.
        test_df = requested_df.copy()

    X_train = train_df.loc[:, feature_list].copy()
    y_train = train_df.loc[:, target].copy()
    X_test = test_df.loc[:, feature_list].copy()
    y_test = test_df.loc[:, target].copy()

    # Défenses explicites contre les fuites et les valeurs manquantes.
    if target in X_train.columns or target in X_test.columns:
        raise AssertionError("La cible ne doit jamais être présente dans X.")
    if X_train.isna().any().any() or y_train.isna().any():
        raise AssertionError("Des valeurs manquantes subsistent dans l'entraînement.")
    if X_test.isna().any().any() or y_test.isna().any():
        raise AssertionError("Des valeurs manquantes subsistent dans le test.")
    if not train_df.empty and train_df["target_date"].max() >= cutoff:
        raise AssertionError("Une ligne d'entraînement atteint ou dépasse la date de coupure.")
    if not test_df.empty and test_df["target_date"].min() < cutoff:
        raise AssertionError("Une ligne de test précède la date de coupure.")

    return DatasetSplit(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        train_df=train_df,
        test_df=test_df,
    )
