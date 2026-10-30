"""Analyse exploratoire des ventes hebdomadaires par magasin.

Chaque fonction :
- affiche un résumé dans la console ;
- enregistre ses graphiques en PNG ;
- renvoie ses résultats chiffrés sous forme de DataFrame ou dict.

Utilisation depuis la racine du projet :
    PYTHONPATH=src python -m moobs.analyse
    PYTHONPATH=src python -m moobs.analyse --csv data/raw/sapin_dataset_stores.csv
    PYTHONPATH=src python -m moobs.analyse --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf

from moobs.data import load_data, prepare_data


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"

MONTH_LABELS = [
    "Jan", "Fév", "Mar", "Avr", "Mai", "Juin",
    "Juil", "Août", "Sep", "Oct", "Nov", "Déc"
]


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def _check_columns(df: pd.DataFrame, columns: set[str]) -> None:
    """Vérifie la présence des colonnes obligatoires."""

    missing = columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Colonnes obligatoires manquantes : {sorted(missing)}"
        )


def _prepare_frame(
    df: pd.DataFrame,
    required_columns: set[str],
) -> pd.DataFrame:
    """Vérifie et prépare les colonnes nécessaires à une analyse."""

    _check_columns(df, required_columns)

    if df.empty:
        raise ValueError("Le DataFrame est vide.")

    work = df.copy()

    if "date" in required_columns:
        work["date"] = pd.to_datetime(
            work["date"],
            errors="coerce"
        )

        if work["date"].isna().any():
            raise ValueError(
                "La colonne date contient des dates invalides ou manquantes."
            )

    if "sales" in required_columns:
        work["sales"] = pd.to_numeric(
            work["sales"],
            errors="coerce"
        )

        if work["sales"].isna().any():
            raise ValueError(
                "La colonne sales contient des valeurs manquantes "
                "ou non numériques."
            )

        if not np.isfinite(work["sales"]).all():
            raise ValueError(
                "La colonne sales contient des valeurs infinies."
            )

    return work


def _save_figure(
    fig: plt.Figure,
    name: str,
    output_dir: Path | str | None,
    show: bool,
) -> Path:
    """Enregistre un graphique et le ferme après affichage éventuel."""

    folder = (
        Path(output_dir)
        if output_dir is not None
        else DEFAULT_OUTPUT_DIR
    )

    folder.mkdir(parents=True, exist_ok=True)

    path = folder / f"{name}.png"

    fig.savefig(
        path,
        dpi=150,
        bbox_inches="tight"
    )

    if show:
        plt.show()

    plt.close(fig)

    print(f"  -> Graphique enregistré : {path}")

    return path


def _short_list(
    values: list[int],
    limit: int = 12,
) -> str:
    """Limite la longueur d'une liste affichée dans la console."""

    if len(values) <= limit:
        return str(values)

    return f"{values[:limit]} ... ({len(values)} retards au total)"


def _title(text: str) -> None:
    """Affiche un titre dans la console."""

    print(
        f"\n{'=' * 70}\n"
        f"{text}\n"
        f"{'=' * 70}"
    )


# ---------------------------------------------------------------------------
# 1. Structure du dataset
# ---------------------------------------------------------------------------

def analyze_dataset(df: pd.DataFrame) -> dict:
    """Analyse les dimensions, types, valeurs manquantes et dates."""

    df = _prepare_frame(
        df,
        {"store_id", "date", "sales"}
    )

    _title("1. STRUCTURE DU DATASET")

    dates = df["date"]

    unique_dates = (
        dates
        .drop_duplicates()
        .sort_values()
    )

    gaps = (
        unique_dates
        .diff()
        .dt.days
        .value_counts(dropna=True)
        .sort_index()
    )

    duplicate_rows = int(df.duplicated().sum())

    duplicate_store_dates = int(
        df.duplicated(subset=["store_id", "date"]).sum()
    )

    result = {
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "n_stores": df["store_id"].nunique(),
        "n_dates": len(unique_dates),
        "date_min": unique_dates.min(),
        "date_max": unique_dates.max(),
        "gap_days_counts": gaps,
        "dtypes": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "describe": df.describe(),
        "duplicate_rows": duplicate_rows,
        "duplicate_store_dates": duplicate_store_dates,
    }

    print(
        f"Dimensions : "
        f"{result['n_rows']} lignes x {result['n_columns']} colonnes"
    )

    print(f"Magasins : {result['n_stores']}")

    print(
        f"Période : {result['date_min'].date()} "
        f"-> {result['date_max'].date()} "
        f"({result['n_dates']} dates distinctes)"
    )

    print(f"\nÉcarts entre dates (jours) :\n{gaps.to_string()}")

    print(f"\nTypes des colonnes :\n{result['dtypes'].to_string()}")

    print(f"\nValeurs manquantes :\n{result['missing'].to_string()}")

    print(
        f"\nStatistiques descriptives :\n"
        f"{result['describe'].round(2).to_string()}"
    )

    print(f"\nLignes dupliquées : {duplicate_rows}")

    print(
        f"Lignes avec doublon magasin/date : "
        f"{duplicate_store_dates}"
    )

    return result


# ---------------------------------------------------------------------------
# 2. Évolution des ventes
# ---------------------------------------------------------------------------

def analyze_sales_trend(
    df: pd.DataFrame,
    output_dir: Path | str | None = None,
    show: bool = False,
    rolling_window: int = 4,
) -> pd.DataFrame:
    """Analyse l'évolution des ventes hebdomadaires."""

    if rolling_window < 1:
        raise ValueError("rolling_window doit être supérieur ou égal à 1.")

    df = _prepare_frame(
        df,
        {"date", "sales"}
    )

    _title("2. ÉVOLUTION DES VENTES")

    weekly = (
        df.groupby("date")["sales"]
        .agg(
            total_sales="sum",
            mean_sales="mean",
            median_sales="median",
        )
        .sort_index()
    )

    weekly[f"mean_sales_roll{rolling_window}"] = (
        weekly["mean_sales"]
        .rolling(rolling_window)
        .mean()
    )

    first = weekly["mean_sales"].iloc[0]
    last = weekly["mean_sales"].iloc[-1]

    print(
        f"Ventes totales par semaine : "
        f"min {weekly['total_sales'].min():.0f} | "
        f"moyenne {weekly['total_sales'].mean():.0f} | "
        f"max {weekly['total_sales'].max():.0f}"
    )

    print(
        f"Ventes moyennes par magasin : "
        f"min {weekly['mean_sales'].min():.1f} | "
        f"moyenne {weekly['mean_sales'].mean():.1f} | "
        f"max {weekly['mean_sales'].max():.1f}"
    )

    print(
        f"Semaine du pic : {weekly['mean_sales'].idxmax().date()} | "
        f"Semaine du creux : {weekly['mean_sales'].idxmin().date()}"
    )

    print(
        f"Première -> dernière semaine : "
        f"{first:.1f} -> {last:.1f} ventes moyennes"
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        sharex=True
    )

    axes[0].plot(
        weekly.index,
        weekly["total_sales"],
        color="#7a2e3a"
    )

    axes[0].set_title(
        "Ventes totales hebdomadaires (tous magasins)"
    )

    axes[0].set_ylabel("Ventes totales")

    axes[1].plot(
        weekly.index,
        weekly["mean_sales"],
        color="#b5485a",
        alpha=0.5,
        label="Moyenne par magasin"
    )

    axes[1].plot(
        weekly.index,
        weekly[f"mean_sales_roll{rolling_window}"],
        color="#2b2b2b",
        label=f"Moyenne mobile {rolling_window} semaines"
    )

    axes[1].set_title(
        "Ventes moyennes hebdomadaires par magasin"
    )

    axes[1].set_ylabel("Ventes moyennes")
    axes[1].set_xlabel("Date")
    axes[1].legend()

    for ax in axes:
        ax.grid(alpha=0.3)

    fig.tight_layout()

    _save_figure(
        fig,
        "02_evolution_ventes",
        output_dir,
        show
    )

    return weekly


# ---------------------------------------------------------------------------
# 3. Distribution des ventes
# ---------------------------------------------------------------------------

def analyze_sales_distribution(
    df: pd.DataFrame,
    output_dir: Path | str | None = None,
    show: bool = False,
) -> dict:
    """Analyse la distribution globale et par magasin des ventes."""

    df = _prepare_frame(
        df,
        {"store_id", "date", "sales"}
    )

    _title("3. DISTRIBUTION DES VENTES")

    sales = df["sales"]

    q1, q3 = sales.quantile([0.25, 0.75])

    iqr = q3 - q1

    upper_fence = q3 + 1.5 * iqr
    lower_fence = q1 - 1.5 * iqr

    n_outliers = int(
        ((sales > upper_fence) | (sales < lower_fence)).sum()
    )

    stats = {
        "describe": sales.describe(),
        "skewness": float(sales.skew()),
        "kurtosis": float(sales.kurt()),
        "iqr_upper_fence": float(upper_fence),
        "iqr_lower_fence": float(lower_fence),
        "n_iqr_outliers": n_outliers,
        "share_iqr_outliers": n_outliers / len(sales),
    }

    print(sales.describe().round(2).to_string())

    print(
        f"\nAsymétrie (skew) : {stats['skewness']:.2f} | "
        f"Aplatissement (kurtosis) : {stats['kurtosis']:.2f}"
    )

    print(
        f"Valeurs hors [{lower_fence:.0f} ; {upper_fence:.0f}] "
        f"(règle 1,5 x IQR) : "
        f"{n_outliers} ({stats['share_iqr_outliers']:.1%})"
    )

    years = df["date"].dt.year

    store_means = (
        df.groupby("store_id")["sales"]
        .mean()
    )

    unique_years = sorted(years.unique())

    by_year = [
        sales[years == year].to_numpy()
        for year in unique_years
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8)
    )

    # Histogramme global

    axes[0, 0].hist(
        sales,
        bins=40,
        color="#b5485a",
        edgecolor="white"
    )

    axes[0, 0].axvline(
        sales.mean(),
        color="black",
        linestyle="--",
        label="Moyenne"
    )

    axes[0, 0].axvline(
        sales.median(),
        color="orange",
        linestyle=":",
        label="Médiane"
    )

    axes[0, 0].set_title(
        "Histogramme des ventes hebdomadaires"
    )

    axes[0, 0].set_xlabel("Ventes")
    axes[0, 0].legend()

    # Boxplot global

    axes[0, 1].boxplot(
        sales,
        flierprops={
            "markersize": 3,
            "alpha": 0.4
        }
    )

    axes[0, 1].set_xticks([])

    axes[0, 1].set_title(
        "Boxplot des ventes (toutes observations)"
    )

    axes[0, 1].set_ylabel("Ventes")

    # Histogramme des moyennes par magasin

    axes[1, 0].hist(
        store_means,
        bins=30,
        color="#7a2e3a",
        edgecolor="white"
    )

    axes[1, 0].set_title(
        "Histogramme des ventes moyennes par magasin"
    )

    axes[1, 0].set_xlabel(
        "Ventes moyennes du magasin"
    )

    # Boxplot par année

    axes[1, 1].boxplot(
        by_year,
        flierprops={
            "markersize": 3,
            "alpha": 0.4
        }
    )

    axes[1, 1].set_xticks(
        range(1, len(unique_years) + 1)
    )

    axes[1, 1].set_xticklabels(
        [str(year) for year in unique_years]
    )

    axes[1, 1].set_title(
        "Boxplot des ventes par année"
    )

    axes[1, 1].set_ylabel("Ventes")

    for ax in axes.ravel():
        ax.grid(alpha=0.3)

    fig.tight_layout()

    _save_figure(
        fig,
        "03_distribution_ventes",
        output_dir,
        show
    )

    return stats


# ---------------------------------------------------------------------------
# 4. Comparaison des magasins
# ---------------------------------------------------------------------------

def analyze_stores(
    df: pd.DataFrame,
    output_dir: Path | str | None = None,
    show: bool = False,
    top_n: int = 10,
    sigma: float = 2.0,
) -> pd.DataFrame:
    """Compare les magasins selon leur niveau et leur dispersion de ventes."""

    if top_n < 1:
        raise ValueError("top_n doit être supérieur ou égal à 1.")

    if sigma <= 0:
        raise ValueError("sigma doit être strictement positif.")

    df = _prepare_frame(
        df,
        {"store_id", "sales"}
    )

    _title("4. COMPARAISON DES MAGASINS")

    stats = (
        df.groupby("store_id")["sales"]
        .agg(
            mean_sales="mean",
            std_sales="std",
            min_sales="min",
            max_sales="max",
            n_obs="count",
        )
    )

    # Coefficient de variation sécurisé
    stats["cv"] = (
        stats["std_sales"]
        / stats["mean_sales"].replace(0, np.nan)
    )

    if "Province" in df.columns:
        stats["Province"] = (
            df.groupby("store_id")["Province"]
            .first()
        )

    mean_of_means = stats["mean_sales"].mean()
    std_of_means = stats["mean_sales"].std()

    high = mean_of_means + sigma * std_of_means
    low = mean_of_means - sigma * std_of_means

    stats["outlier"] = (
        (stats["mean_sales"] > high)
        | (stats["mean_sales"] < low)
    )

    stats = stats.sort_values(
        "mean_sales",
        ascending=False
    )

    stats["rank"] = np.arange(
        1,
        len(stats) + 1
    )

    show_cols = [
        column
        for column in [
            "rank",
            "Province",
            "mean_sales",
            "std_sales",
            "cv",
            "outlier",
        ]
        if column in stats.columns
    ]

    print(
        f"Moyenne des magasins : {mean_of_means:.1f} | "
        f"Écart-type : {std_of_means:.1f} | "
        f"Seuils {sigma:.0f} sigma : [{low:.1f} ; {high:.1f}]"
    )

    print(
        f"\nTop {top_n} :\n"
        f"{stats[show_cols].head(top_n).round(2).to_string()}"
    )

    print(
        f"\nFlop {top_n} :\n"
        f"{stats[show_cols].tail(top_n).round(2).to_string()}"
    )

    outliers = stats[stats["outlier"]]

    print(
        f"\nMagasins atypiques ({len(outliers)}) : "
        f"{list(outliers.index)}"
    )

    print(
        f"Dispersion (coefficient de variation) : "
        f"médiane {stats['cv'].median():.3f} | "
        f"max {stats['cv'].max():.3f} "
        f"({stats['cv'].idxmax()})"
    )

    # Couleur rouge pour les magasins atypiques
    colors = np.where(
        stats["outlier"],
        "#d32f2f",
        "#4682b4"
    )

    # Graphique 1 : classement des magasins

    fig, ax = plt.subplots(
        figsize=(14, 5)
    )

    ax.bar(
        stats.index.astype(str),
        stats["mean_sales"],
        color=colors
    )

    ax.errorbar(
        stats.index.astype(str),
        stats["mean_sales"],
        yerr=stats["std_sales"].fillna(0),
        fmt="none",
        ecolor="black",
        alpha=0.35,
        linewidth=0.7
    )

    ax.axhline(
        mean_of_means,
        color="black",
        linestyle="--",
        label="Moyenne des magasins"
    )

    ax.axhline(
        high,
        color="orange",
        linestyle=":",
        label=f"+{sigma:.0f} écarts-types"
    )

    ax.axhline(
        low,
        color="orange",
        linestyle="-.",
        label=f"-{sigma:.0f} écarts-types"
    )

    ax.set_title(
        "Classement des magasins par ventes moyennes "
        "(barres d'erreur = écart-type)"
    )

    ax.set_ylabel("Ventes moyennes")

    ax.tick_params(
        axis="x",
        rotation=90,
        labelsize=6
    )

    ax.legend()

    fig.tight_layout()

    _save_figure(
        fig,
        "04a_classement_magasins",
        output_dir,
        show
    )

    # Graphique 2 : niveau et dispersion

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5)
    )

    axes[0].scatter(
        stats["mean_sales"],
        stats["std_sales"],
        c=colors,
        alpha=0.8
    )

    for store, row in outliers.iterrows():
        axes[0].annotate(
            str(store),
            (row["mean_sales"], row["std_sales"]),
            fontsize=7
        )

    axes[0].set_xlabel("Ventes moyennes")
    axes[0].set_ylabel("Écart-type des ventes")

    axes[0].set_title(
        "Niveau vs dispersion des ventes par magasin"
    )

    axes[0].grid(alpha=0.3)

    # Boxplots des meilleurs et derniers magasins

    selected_stores = list(
        dict.fromkeys(
            list(stats.head(top_n).index)
            + list(stats.tail(top_n).index)
        )
    )

    data = [
        df.loc[
            df["store_id"] == store,
            "sales"
        ].to_numpy()
        for store in selected_stores
    ]

    axes[1].boxplot(
        data,
        flierprops={
            "markersize": 3,
            "alpha": 0.4
        }
    )

    axes[1].set_xticks(
        range(1, len(selected_stores) + 1)
    )

    axes[1].set_xticklabels(
        [str(store) for store in selected_stores]
    )

    axes[1].set_title(
        f"Distribution des ventes : "
        f"{top_n} meilleurs et {top_n} derniers magasins"
    )

    axes[1].tick_params(
        axis="x",
        rotation=90,
        labelsize=7
    )

    axes[1].grid(alpha=0.3)

    fig.tight_layout()

    _save_figure(
        fig,
        "04b_dispersion_magasins",
        output_dir,
        show
    )

    return stats


# ---------------------------------------------------------------------------
# 5. Saisonnalité
# ---------------------------------------------------------------------------

def analyze_seasonality(
    df: pd.DataFrame,
    output_dir: Path | str | None = None,
    show: bool = False,
) -> dict[str, pd.DataFrame]:
    """Analyse les ventes moyennes par semaine ISO et par mois."""

    df = _prepare_frame(
        df,
        {"date", "sales"}
    )

    _title("5. SAISONNALITÉ")

    dates = df["date"]

    iso = dates.dt.isocalendar()

    work = pd.DataFrame({
        "sales": df["sales"].to_numpy(),
        "iso_year": iso["year"].to_numpy(),
        "iso_week": iso["week"].to_numpy(),
        "year": dates.dt.year.to_numpy(),
        "month": dates.dt.month.to_numpy(),
    })

    # Profil des ventes par semaine ISO et par année
    by_week = (
        work.groupby(["iso_year", "iso_week"])["sales"]
        .mean()
        .unstack(0)
    )

    # Profil des ventes par mois et par année
    by_month = (
        work.groupby(["year", "month"])["sales"]
        .mean()
        .unstack(0)
    )

    # Moyenne des profils entre les années disponibles
    week_profile = by_week.mean(axis=1)
    month_profile = by_month.mean(axis=1)

    print(
        f"Semaine ISO la plus forte : "
        f"{int(week_profile.idxmax())} "
        f"({week_profile.max():.1f}) | "
        f"la plus faible : "
        f"{int(week_profile.idxmin())} "
        f"({week_profile.min():.1f})"
    )

    print(
        f"Mois le plus fort : "
        f"{MONTH_LABELS[int(month_profile.idxmax()) - 1]} "
        f"({month_profile.max():.1f}) | "
        f"le plus faible : "
        f"{MONTH_LABELS[int(month_profile.idxmin()) - 1]} "
        f"({month_profile.min():.1f})"
    )

    amplitude = (
        (month_profile.max() - month_profile.min())
        / month_profile.mean()
    )

    print(
        f"Amplitude saisonnière mensuelle : {amplitude:.1%} "
        f"de la moyenne"
    )

    if by_week.shape[1] > 1:
        corr = by_week.corr().iloc[0, 1]

        print(
            f"Corrélation du profil hebdomadaire entre "
            f"{by_week.columns[0]} et {by_week.columns[1]} : "
            f"{corr:.2f}"
        )

    print(
        f"\nVentes moyennes par mois :\n"
        f"{by_month.round(1).to_string()}"
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15, 5)
    )

    # Profil hebdomadaire

    for year in by_week.columns:
        axes[0].plot(
            by_week.index,
            by_week[year],
            marker="o",
            markersize=3,
            label=str(year)
        )

    axes[0].set_title(
        "Ventes moyennes par semaine ISO"
    )

    axes[0].set_xlabel("Semaine de l'année")
    axes[0].set_ylabel("Ventes moyennes par magasin")
    axes[0].set_xticks(range(1, 54, 4))

    # Profil mensuel

    for year in by_month.columns:
        axes[1].plot(
            by_month.index,
            by_month[year],
            marker="o",
            label=str(year)
        )

    axes[1].set_title(
        "Ventes moyennes par mois"
    )

    axes[1].set_xlabel("Mois")

    axes[1].set_xticks(range(1, 13))
    axes[1].set_xticklabels(MONTH_LABELS)

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(title="Année")

    fig.tight_layout()

    _save_figure(
        fig,
        "05_saisonnalite",
        output_dir,
        show
    )

    return {
        "by_week": by_week,
        "by_month": by_month,
    }


# ---------------------------------------------------------------------------
# 6. Autocorrélation
# ---------------------------------------------------------------------------

def analyze_autocorrelation(
    df: pd.DataFrame,
    output_dir: Path | str | None = None,
    show: bool = False,
    store_id: str | int | None = None,
    nlags: int = 52,
    diff: int = 0,
) -> pd.DataFrame:
    """Analyse les dépendances temporelles avec ACF et PACF.

    Par défaut, la série analysée est la moyenne des ventes de tous
    les magasins par semaine.

    store_id permet d'analyser un magasin précis.

    diff applique une différenciation :
        diff=0 : série originale ;
        diff=1 : première différence ;
        diff=2 : deuxième différence.
    """

    if nlags < 1:
        raise ValueError("nlags doit être supérieur ou égal à 1.")

    if diff < 0:
        raise ValueError("diff doit être positif ou nul.")

    df = _prepare_frame(
        df,
        {"date", "sales"}
    )

    _title("6. DÉPENDANCES TEMPORELLES (ACF / PACF)")

    if store_id is None:

        # Moyenne des ventes de tous les magasins par semaine
        series = (
            df.groupby("date")["sales"]
            .mean()
            .sort_index()
        )

        label = "moyenne de tous les magasins"

    else:

        _check_columns(
            df,
            {"store_id"}
        )

        subset = df[
            df["store_id"] == store_id
        ]

        if subset.empty:
            raise ValueError(
                f"Magasin introuvable : {store_id}"
            )

        if subset["date"].duplicated().any():
            raise ValueError(
                f"Plusieurs observations existent pour le magasin "
                f"{store_id} à une même date."
            )

        series = (
            subset
            .set_index("date")["sales"]
            .sort_index()
        )

        label = f"magasin {store_id}"

    # Vérification de la régularité hebdomadaire
    series.index = pd.to_datetime(series.index)

    series = series.sort_index()

    expected_dates = pd.date_range(
        start=series.index.min(),
        end=series.index.max(),
        freq="7D"
    )

    missing_dates = expected_dates.difference(
        series.index
    )

    if len(missing_dates) > 0:
        raise ValueError(
            f"La série contient {len(missing_dates)} semaines manquantes. "
            f"Il faut traiter ces semaines avant de calculer l'ACF/PACF."
        )

    # Différenciation éventuelle
    if diff > 0:
        series = series.diff(diff).dropna()

        label += f", différenciée {diff} fois"

    # Vérification de la longueur
    max_lags = len(series) // 2 - 1

    if max_lags < 2:
        raise ValueError(
            "Série trop courte pour calculer ACF/PACF."
        )

    nlags = min(nlags, max_lags)

    # Calcul ACF et PACF
    acf_values = acf(
        series,
        nlags=nlags,
        fft=False
    )

    pacf_values = pacf(
        series,
        nlags=nlags
    )

    result = pd.DataFrame(
        {
            "acf": acf_values,
            "pacf": pacf_values,
        },
        index=pd.RangeIndex(
            0,
            nlags + 1,
            name="lag"
        )
    )

    # Seuil de significativité approximatif à 95 %
    threshold = 1.96 / np.sqrt(len(series))

    significant_acf = [
        int(lag)
        for lag in result.index[1:]
        if abs(result.loc[lag, "acf"]) > threshold
    ]

    significant_pacf = [
        int(lag)
        for lag in result.index[1:]
        if abs(result.loc[lag, "pacf"]) > threshold
    ]

    print(
        f"Série : {label} "
        f"({len(series)} points, {nlags} retards)"
    )

    print(
        f"Seuil de significativité approximatif (95 %) : "
        f"+/- {threshold:.3f}"
    )

    print(
        f"Retards ACF significatifs : "
        f"{_short_list(significant_acf)}"
    )

    print(
        f"Retards PACF significatifs : "
        f"{_short_list(significant_pacf)}"
    )

    print(
        "ACF aux retards clés : "
        + ", ".join(
            f"lag {lag} = {result.loc[lag, 'acf']:.2f}"
            for lag in (1, 2, 4, 26, 52)
            if lag <= nlags
        )
    )

    # Graphiques
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 7)
    )

    plot_acf(
        series,
        lags=nlags,
        ax=axes[0],
        fft=False
    )

    axes[0].set_title(
        f"ACF — {label}"
    )

    plot_pacf(
        series,
        lags=nlags,
        ax=axes[1]
    )

    axes[1].set_title(
        f"PACF — {label}"
    )

    for ax in axes:
        ax.set_xlabel("Retard (semaines)")
        ax.grid(alpha=0.3)

    fig.tight_layout()

    _save_figure(
        fig,
        "06_acf_pacf",
        output_dir,
        show
    )

    return result


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_all(
    df: pd.DataFrame | None = None,
    output_dir: Path | str | None = None,
    show: bool = False,
) -> dict:
    """Exécute les six analyses exploratoires."""

    if df is None:
        df = prepare_data(
            load_data()
        )

    return {
        "dataset": analyze_dataset(df),

        "trend": analyze_sales_trend(
            df,
            output_dir,
            show
        ),

        "distribution": analyze_sales_distribution(
            df,
            output_dir,
            show
        ),

        "stores": analyze_stores(
            df,
            output_dir,
            show
        ),

        "seasonality": analyze_seasonality(
            df,
            output_dir,
            show
        ),

        "autocorrelation": analyze_autocorrelation(
            df,
            output_dir,
            show
        ),
    }


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description="Analyse exploratoire des ventes Moobs."
    )

    parser.add_argument(
        "--csv",
        default=None,
        help="Chemin du CSV."
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Dossier de sauvegarde des graphiques."
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Afficher les graphiques à l'écran."
    )

    args = parser.parse_args()

    # Chargement et préparation des données
    df = prepare_data(
        load_data(args.csv)
    )

    # Exécution des analyses
    run_all(
        df=df,
        output_dir=args.output_dir,
        show=args.show
    )

    print("\nAnalyse exploratoire terminée.")
    print(
        f"Graphiques disponibles dans : "
        f"{args.output_dir or DEFAULT_OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()