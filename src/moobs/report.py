
"""Génère un rapport PDF complet sur l'EDA et le backtesting des ventes."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "sapin_dataset_stores.csv"
RESULTS_PATH = REPORTS_DIR / "backtesting_results.csv"
PDF_PATH = REPORTS_DIR / "backtesting_report.pdf"
CHARTS_DIR = REPORTS_DIR / "charts"

# Répertoire contenant les graphiques produits par analysis.py.
EDA_FIGURES_DIR = REPORTS_DIR / "figures"


def make_chart_comparison(best, naive):
    """Graphique : MAE du meilleur modèle contre le modèle naïf."""
    merged = best.merge(naive, on="horizon", how="left")
    if merged.empty:
        return None

    x = range(len(merged))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(
        [i - width / 2 for i in x],
        merged["MAE_naif"],
        width,
        label="Modèle naïf",
    )
    ax.bar(
        [i + width / 2 for i in x],
        merged["MAE_moyenne"],
        width,
        label="Meilleur modèle",
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(
        [f"{int(h)} semaine(s)" for h in merged["horizon"]]
    )
    ax.set_ylabel("MAE moyenne (plus bas = meilleur)")
    ax.set_title("Comparaison avec le modèle naïf")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    path = CHARTS_DIR / "mae_naif_vs_meilleur.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def make_chart_algorithms(ranking):
    """Graphique : MAE moyenne par algorithme et horizon."""
    if ranking.empty:
        return None

    fig, ax = plt.subplots(figsize=(9, 4.5))

    for horizon in sorted(ranking["horizon"].unique()):
        subset = ranking[ranking["horizon"] == horizon]
        algorithm_scores = (
            subset.groupby("algorithm")["MAE_moyenne"]
            .mean()
            .sort_index()
        )
        ax.plot(
            algorithm_scores.index,
            algorithm_scores.values,
            marker="o",
            label=f"Horizon {int(horizon)}",
        )

    ax.set_ylabel("MAE moyenne")
    ax.set_title("Comparaison des algorithmes")
    ax.tick_params(axis="x", rotation=20)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    path = CHARTS_DIR / "comparaison_algorithmes.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def make_chart_periods(results, best):
    """Graphique : MAE du modèle retenu pour chaque période."""
    selected = best[
        ["horizon", "feature_set", "algorithm"]
    ].drop_duplicates()

    chosen = results.merge(
        selected,
        on=["horizon", "feature_set", "algorithm"],
        how="inner",
    )

    if chosen.empty:
        return None

    fig, ax = plt.subplots(figsize=(9, 4.5))

    for horizon in sorted(chosen["horizon"].unique()):
        subset = chosen[chosen["horizon"] == horizon].sort_values(
            "period_start"
        )
        labels = subset["period_start"].astype(str)
        ax.plot(
            labels,
            subset["MAE"],
            marker="o",
            label=f"Horizon {int(horizon)}",
        )

    ax.set_ylabel("MAE")
    ax.set_title("Stabilité des meilleurs modèles selon la période")
    ax.tick_params(axis="x", rotation=20)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    path = CHARTS_DIR / "mae_par_periode.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def add_chart(story, chart_path, width=22 * cm):
    """Ajoute un graphique de backtesting au rapport."""
    if chart_path and chart_path.exists():
        story.append(Image(str(chart_path), width=width, height=10 * cm))
        story.append(Spacer(1, 0.3 * cm))


def add_eda_chart(story, filename, title, description):
    """
    Ajoute un graphique EDA avec son titre et son explication.

    Si le fichier n'existe pas, un message est ajouté au rapport.
    """
    chart_path = EDA_FIGURES_DIR / filename

    story.append(Paragraph(title, styles_for_report["Heading2"]))
    story.append(Paragraph(description, styles_for_report["BodyText"]))
    story.append(Spacer(1, 0.3 * cm))

    if chart_path.exists():
        story.append(
            Image(str(chart_path), width=22 * cm, height=10 * cm)
        )
    else:
        story.append(
            Paragraph(
                f"Graphique non disponible : {filename}",
                styles_for_report["Italic"],
            )
        )

    story.append(Spacer(1, 0.4 * cm))


def add_eda_section(story):
    """Ajoute les six graphiques d'analyse exploratoire au rapport."""

    story.append(
        Paragraph(
            "2. Analyse exploratoire des données (EDA)",
            styles_for_report["Heading1"],
        )
    )

    story.append(
        Paragraph(
            "Cette section présente une analyse descriptive des données "
            "brutes avant la phase de modélisation. Elle permet de mieux "
            "comprendre l'évolution des ventes, leur distribution, les "
            "différences entre magasins, les effets saisonniers et les "
            "dépendances temporelles présentes dans les séries.",
            styles_for_report["BodyText"],
        )
    )
    story.append(Spacer(1, 0.4 * cm))

    # 1. Évolution des ventes.
    add_eda_chart(
        story,
        "02_evolution_ventes.png",
        "2.1 Évolution des ventes",
        "Ce graphique présente l'évolution des ventes au cours du temps. "
        "Il permet d'observer la tendance générale, les variations "
        "temporelles et les éventuelles périodes de hausse ou de baisse "
        "de l'activité.",
    )
    story.append(PageBreak())

    # 2. Distribution des ventes.
    add_eda_chart(
        story,
        "03_distribution_ventes.png",
        "2.2 Distribution des ventes",
        "Cette analyse décrit la répartition des valeurs de ventes. "
        "Elle permet d'observer leur concentration, leur dispersion "
        "et la présence éventuelle de valeurs particulièrement élevées "
        "ou faibles.",
    )
    story.append(PageBreak())

    # 3. Classement des magasins.
    add_eda_chart(
        story,
        "04a_classement_magasins.png",
        "2.3 Comparaison des magasins : classement",
        "Ce graphique compare les magasins selon leur niveau de ventes. "
        "Il permet d'identifier les magasins ayant les volumes de ventes "
        "les plus élevés et ceux ayant les volumes les plus faibles.",
    )
    story.append(PageBreak())

    # 4. Dispersion des magasins.
    add_eda_chart(
        story,
        "04b_dispersion_magasins.png",
        "2.4 Comparaison des magasins : dispersion",
        "Ce graphique complète le classement en montrant la dispersion "
        "des ventes entre les magasins. Il permet d'évaluer les écarts "
        "de niveau d'activité et l'hétérogénéité du réseau.",
    )
    story.append(PageBreak())

    # 5. Saisonnalité.
    add_eda_chart(
        story,
        "05_saisonnalite.png",
        "2.5 Analyse de la saisonnalité",
        "Cette analyse recherche des variations récurrentes des ventes "
        "au fil des périodes. Elle permet d'examiner l'existence "
        "d'éventuels effets saisonniers dans les données hebdomadaires.",
    )
    story.append(PageBreak())

    # 6. Autocorrélation.
    add_eda_chart(
        story,
        "06_acf_pacf.png",
        "2.6 Autocorrélation : ACF et PACF",
        "Les fonctions ACF et PACF permettent d'étudier les relations "
        "entre les ventes observées et leurs valeurs passées. Elles "
        "aident à repérer les dépendances temporelles susceptibles "
        "d'être exploitées dans la construction des variables retardées.",
    )

    story.append(PageBreak())


# Styles du rapport.
styles_for_report = getSampleStyleSheet()

styles_for_report.add(
    ParagraphStyle(
        name="ReportTitle",
        parent=styles_for_report["Title"],
        alignment=TA_CENTER,
        fontSize=21,
        leading=26,
        spaceAfter=18,
    )
)


def generate_report():
    """Construit le rapport PDF depuis les résultats existants."""

    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Résultats introuvables : {RESULTS_PATH}\n"
            "Lance d'abord le backtesting avec "
            "'PYTHONPATH=src python -m moobs.validation'."
        )

    results = pd.read_csv(RESULTS_PATH)

    required = {
        "horizon", "period_start", "period_end",
        "feature_set", "algorithm", "n_train",
        "n_test", "MAE", "RMSE",
    }
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes : {sorted(missing)}")

    if results.empty:
        raise ValueError("Le fichier de résultats est vide.")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    # Informations sur les données brutes.
    if DATA_PATH.exists():
        raw = pd.read_csv(DATA_PATH)
        n_rows = len(raw)
        n_stores = raw["store_id"].nunique() if "store_id" in raw else None
        date_col = "date" if "date" in raw else None

        if date_col:
            dates = pd.to_datetime(raw[date_col], errors="coerce")
            date_min = dates.min()
            date_max = dates.max()
        else:
            date_min = date_max = None

        n_columns = len(raw.columns)
        missing_cells = int(raw.isna().sum().sum())
    else:
        raw = None
        n_rows = n_stores = n_columns = missing_cells = None
        date_min = date_max = None

    model_results = results[
        results["algorithm"] != "NaiveLastValue"
    ]

    ranking = (
        model_results.groupby(
            ["horizon", "feature_set", "algorithm"],
            as_index=False,
        )
        .agg(
            MAE_moyenne=("MAE", "mean"),
            RMSE_moyenne=("RMSE", "mean"),
            Nombre_backtests=("MAE", "count"),
        )
    )

    best = (
        ranking.sort_values(["horizon", "MAE_moyenne"])
        .groupby("horizon", as_index=False)
        .first()
    )

    naive = (
        results[results["algorithm"] == "NaiveLastValue"]
        .groupby("horizon", as_index=False)
        .agg(
            MAE_naif=("MAE", "mean"),
            RMSE_naif=("RMSE", "mean"),
        )
    )

    story = []

    # 1. Couverture et données.
    story.append(
        Paragraph(
            "Rapport de performance du forecasting des ventes",
            styles_for_report["ReportTitle"],
        )
    )
    story.append(
        Paragraph("1. Données utilisées", styles_for_report["Heading1"])
    )

    data_rows = [["Information", "Valeur"]]
    data_rows.append(["Fichier source", str(DATA_PATH.name)])
    data_rows.append([
        "Nombre de lignes",
        str(n_rows) if n_rows is not None else "Indisponible",
    ])
    data_rows.append([
        "Nombre de magasins",
        str(n_stores) if n_stores is not None else "Indisponible",
    ])
    data_rows.append([
        "Nombre de colonnes",
        str(n_columns) if n_columns is not None else "Indisponible",
    ])
    data_rows.append([
        "Période des données",
        (
            f"{date_min.date()} au {date_max.date()}"
            if pd.notna(date_min) and pd.notna(date_max)
            else "Indisponible"
        ),
    ])
    data_rows.append([
        "Cellules manquantes dans le fichier brut",
        str(missing_cells) if missing_cells is not None else "Indisponible",
    ])
    data_rows.append(["Fréquence visée", "Hebdomadaire"])

    table = Table(data_rows, colWidths=[8 * cm, 15 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#243B53")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#EDF2F7")]),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.4 * cm))

    story.append(
        Paragraph(
            "La cible du projet est le chiffre de ventes futur, "
            "noté target_sales dans le dataset de modélisation. "
            "Les variables explicatives varient selon la stratégie : "
            "historique des ventes, statistiques glissantes, données "
            "démographiques et variables calendaires.",
            styles_for_report["BodyText"],
        )
    )

    # 2. Analyse exploratoire des données.
    add_eda_section(story)

    # 3. Approche de backtesting.
    story.append(
        Paragraph("3. Approche d'évaluation", styles_for_report["Heading1"])
    )
    story.append(
        Paragraph(
            "Le backtesting temporel teste les prévisions sur trois "
            "fenêtres : du 1er avril au 1er juillet 2024, du 1er juillet "
            "au 1er octobre 2024 et du 1er octobre 2024 au 1er janvier "
            "2025. Pour chaque fenêtre, l'entraînement utilise les cibles "
            "antérieures au début du test. Les stratégies sont comparées "
            "sur les mêmes lignes de test communes.",
            styles_for_report["BodyText"],
        )
    )
    story.append(Spacer(1, 0.2 * cm))

    story.append(
        Paragraph(
            "Horizons étudiés : 1, 2, 4 et 12 semaines. "
            "Le protocole compare sept ensembles de variables et quatre "
            "algorithmes (régression linéaire, Random Forest, "
            "HistGradientBoosting et XGBoost), ainsi qu'un modèle naïf "
            "qui prédit la dernière vente observée.",
            styles_for_report["BodyText"],
        )
    )
    story.append(Spacer(1, 0.2 * cm))

    story.append(
        Paragraph(
            "MAE : erreur absolue moyenne, dans l'unité de sales. "
            "RMSE : racine de l'erreur quadratique moyenne, qui pénalise "
            "davantage les erreurs importantes. Pour les deux métriques, "
            "une valeur plus faible est préférable.",
            styles_for_report["BodyText"],
        )
    )

    story.append(PageBreak())

    # 4. Résumé des meilleurs résultats.
    story.append(
        Paragraph(
            "4. Meilleures configurations",
            styles_for_report["Heading1"],
        )
    )

    summary = [[
        "Horizon", "Variables", "Algorithme",
        "MAE moyenne", "RMSE moyenne", "Backtests",
    ]]

    for _, row in best.iterrows():
        summary.append([
            f"{int(row['horizon'])} sem.",
            str(row["feature_set"]),
            str(row["algorithm"]),
            f"{row['MAE_moyenne']:.2f}",
            f"{row['RMSE_moyenne']:.2f}",
            str(int(row["Nombre_backtests"])),
        ])

    table = Table(summary, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#243B53")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#EDF2F7")]),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.3 * cm))

    chart = make_chart_comparison(best, naive)
    add_chart(story, chart)

    story.append(PageBreak())

    # 5. Graphiques comparatifs.
    story.append(
        Paragraph(
            "5. Comparaison des modèles",
            styles_for_report["Heading1"],
        )
    )
    add_chart(story, make_chart_algorithms(ranking))
    add_chart(story, make_chart_periods(results, best))

    # 6. Classement complet.
    story.append(PageBreak())
    story.append(
        Paragraph(
            "6. Classement détaillé",
            styles_for_report["Heading1"],
        )
    )

    for horizon in sorted(ranking["horizon"].unique()):
        story.append(
            Paragraph(
                f"Horizon : {int(horizon)} semaine(s)",
                styles_for_report["Heading2"],
            )
        )

        subset = ranking[ranking["horizon"] == horizon].sort_values(
            "MAE_moyenne"
        )

        rows = [[
            "Rang", "Variables", "Algorithme",
            "MAE", "RMSE", "Tests",
        ]]

        for rank, (_, row) in enumerate(subset.iterrows(), start=1):
            rows.append([
                str(rank),
                str(row["feature_set"]),
                str(row["algorithm"]),
                f"{row['MAE_moyenne']:.2f}",
                f"{row['RMSE_moyenne']:.2f}",
                str(int(row["Nombre_backtests"])),
            ])

        table = Table(rows, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#243B53")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#EDF2F7")]),
        ]))
        story.append(table)
        story.append(Spacer(1, 0.3 * cm))

    # 7. Limites.
    story.append(
        Paragraph(
            "7. Interprétation et limites",
            styles_for_report["Heading1"],
        )
    )
    story.append(
        Paragraph(
            "Les scores présentés sont les résultats historiques du "
            "backtesting et servent à comparer les configurations. "
            "Les meilleures configurations sont sélectionnées sur la "
            "MAE moyenne des périodes disponibles ; ce classement ne "
            "constitue pas une nouvelle évaluation indépendante des "
            "modèles finaux sauvegardés. Les résultats peuvent varier "
            "selon les périodes futures. Le nombre de cellules manquantes "
            "affiché concerne le fichier brut et ne représente pas "
            "nécessairement les valeurs manquantes après la création "
            "des variables.",
            styles_for_report["BodyText"],
        )
    )

    document = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=landscape(A4),
        rightMargin=1.3 * cm,
        leftMargin=1.3 * cm,
        topMargin=1.3 * cm,
        bottomMargin=1.3 * cm,
        title="Rapport de performance du forecasting des ventes",
        author="SalesForecasting",
    )

    document.build(story)
    print(f"Rapport PDF créé : {PDF_PATH}")


if __name__ == "__main__":
    generate_report()
