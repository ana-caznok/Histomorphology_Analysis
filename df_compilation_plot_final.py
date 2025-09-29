#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
df_compilation_plot_SIMPLE_ARGPARSE.py
======================================

Goal (in plain words)
---------------------
- Read previously computed tables (morphometry, immuno, clinical).
- Merge them into a single tidy DataFrame.
- Generate the same plots the notebook produced (bar, grouped regression plots,
  violin/box plots, and PCA visuals).
- **Save all plots to disk** in one place.

Main inputs (expected under <base_path>)
----------------------------------------
- morpho_results/area_morpho.csv
- imuno_results/area_imunos_marcadas.csv          (optional; skip if missing)
- final_results/Planilha_final_04-02.csv          (optional; skip if missing)

Main outputs (created under <base_path>)
----------------------------------------
- final_results/final_measurements_table.csv
- final_results/final_measurements_table_pop.csv   (subset: CASE/P0P only, if applicable)
- final_results/stats_table.csv
- final_results/plots/*.png                        (ALL figures saved here)

This script is intentionally **simple and well-commented** so it reads like a tutorial.

"""

# ============================
# 1) IMPORTS (kept compact)
# ============================
from pathlib import Path
import argparse

import numpy as np
import pandas as pd

# Use a non-interactive backend so we can save figures without a GUI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats


# =====================
# 2) SMALL UTILITIES
# =====================
def ensure_dir(p: Path) -> None:
    """Create parent dir if needed."""
    p.parent.mkdir(parents=True, exist_ok=True)


def find_cols(symbol: str, df: pd.DataFrame) -> list[str]:
    """Return all column names that contain a given substring."""
    return [c for c in df.columns if symbol in str(c)]


# ==========================
# 3) PLOTTING HELPER FUNCS
# ==========================
def plot_morpho_proportions(df: pd.DataFrame, out_png: Path, perc_cols: list[str]) -> None:
    """
    Bar chart of the *average* tissue composition by 'type' (e.g., CONTROL/CASE).
    """
    ensure_dir(out_png)
    gmean = df.groupby("type")[perc_cols].mean(numeric_only=True)
    ax = gmean.plot.bar(rot=0, figsize=(12, 6), ylim=(0, 100), title="Average Tissue Composition per Patient")
    ax.set_ylabel("% of tissue")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def grouped_plot(df: pd.DataFrame, x_col: str, y_col: str, out_png: Path) -> None:
    """
    Scatter + line for each 'type' group with a simple linear regression fit.
    Saves '<x>_x_<y>.png'.
    """
    ensure_dir(out_png)
    plt.figure(figsize=(8, 6))
    for group in df["type"].dropna().unique():
        mini = df[df["type"] == group][[x_col, y_col]].dropna()
        if len(mini) < 2:
            continue
        x = mini[x_col].to_numpy(dtype=float)
        y = mini[y_col].to_numpy(dtype=float)
        # Regression (slope/intercept)
        slope, intercept, r, p, _ = stats.linregress(x, y)
        y_pred = intercept + slope * x
        # Scatter
        plt.scatter(x, y, label=f"{group} (r={r:.2f}, p={p:.3g})", alpha=0.8)
        # Fit line (sorted by x for a neat line)
        order = np.argsort(x)
        plt.plot(x[order], y_pred[order], linestyle="--")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"{y_col} vs {x_col}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def col_violinplot(df: pd.DataFrame, col: str, out_png: Path) -> None:
    """
    Violin plot of a single column grouped by 'type'.
    """
    ensure_dir(out_png)
    plt.figure(figsize=(8, 5))
    sns.violinplot(data=df, x="type", y=col, cut=0, inner="box")
    plt.title(f"Violin Plot of {col} by Type")
    plt.xlabel("Type")
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def col_boxplot(df: pd.DataFrame, col: str, out_png: Path) -> None:
    """
    Box plot of a single column grouped by 'type'.
    """
    ensure_dir(out_png)
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="type", y=col)
    plt.title(f"Boxplot of {col} by Type")
    plt.xlabel("Type")
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def perform_pca(df: pd.DataFrame, features: list[str], n_components: int = 5):
    """
    Standardize features and run PCA. Returns (loadings_df, scores_df).
    loadings_df.index are features; columns are PC1..PCk.
    scores_df has columns PC1..PCk plus 'type' for plotting if available.
    """
    # Drop rows with any missing values in selected features
    df_clean = df.dropna(subset=features).copy()
    if "type" in df_clean.columns:
        y = df_clean["type"]
    else:
        y = None

    X = df_clean[features].astype(float).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = PCA(n_components=min(n_components, Xs.shape[1]))
    scores = pca.fit_transform(Xs)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=features,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)],
    )
    scores_df = pd.DataFrame(
        scores,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)],
        index=df_clean.index,
    )
    if y is not None:
        scores_df["type"] = y.values

    return loadings, scores_df


def plot_pca_loadings_heatmap(loadings_df: pd.DataFrame, out_png: Path) -> None:
    """Heatmap of PCA loadings."""
    ensure_dir(out_png)
    plt.figure(figsize=(14, 6))
    sns.heatmap(loadings_df, annot=True, center=0, fmt=".2f", linewidths=0.5, cmap="vlag")
    plt.title("PCA Loadings Heatmap")
    plt.xlabel("Principal Components")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_pca_2d(scores_df: pd.DataFrame, out_png: Path) -> None:
    """2D PCA scatter (PC1 vs PC2) colored by 'type' if present."""
    ensure_dir(out_png)
    if not {"PC1", "PC2"}.issubset(scores_df.columns):
        return
    plt.figure(figsize=(7, 6))
    if "type" in scores_df.columns:
        for t in scores_df["type"].dropna().unique():
            sub = scores_df[scores_df["type"] == t]
            plt.scatter(sub["PC1"], sub["PC2"], label=str(t))
        plt.legend()
    else:
        plt.scatter(scores_df["PC1"], scores_df["PC2"])
    plt.title("2D PCA Projection (PC1 vs. PC2)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


# =====================
# 4) MAIN PROCEDURE
# =====================
def main():
    parser = argparse.ArgumentParser(
        description="Compile tables and save ALL plots from the analysis."
    )
    parser.add_argument("base_path", type=str, help="Project base path.")
    parser.add_argument("--plots-subdir", type=str, default="final_results/plots",
                        help="Relative folder where PNGs will be saved.")
    parser.add_argument("--n-pca", type=int, default=5, help="Number of PCA components.")
    args = parser.parse_args()

    base = Path(args.base_path)
    plots_dir = base / args.plots_subdir
    ensure_dir(plots_dir / "dummy.png")  # just to create the folder

    # ---------- Load inputs (be forgiving if some are missing) ----------
    morpho_csv = base / "morpho_results/area_morpho.csv"
    imuno_csv  = base / "imuno_results/area_imunos_marcadas.csv"
    clin_csv   = base / "final_results/Planilha_final_04-02.csv"

    if not morpho_csv.exists():
        raise FileNotFoundError(f"Required file missing: {morpho_csv}")

    df_morpho = pd.read_csv(morpho_csv)

    df_imuno = pd.read_csv(imuno_csv) if imuno_csv.exists() else None
    df_clin  = pd.read_csv(clin_csv) if clin_csv.exists() else None

    # ---------- Merge tables (simple, transparent logic) ----------
    # Start from morpho (per-patient). Ensure 'patient' and 'type' are strings.
    df_morpho["patient"] = df_morpho["patient"].astype(str)
    if "type" in df_morpho.columns:
        df_morpho["type"] = df_morpho["type"].astype(str)

    df_merge = df_morpho.copy()

    # If immuno exists, join on patient
    if df_imuno is not None and "patient" in df_imuno.columns:
        df_imuno["patient"] = df_imuno["patient"].astype(str)
        # common immuno columns (example): area_px2, area_perc, area_um2
        # Also allow pre-aggregated columns like 'area_perc_mean' if present.
        df_merge = df_merge.merge(df_imuno, on="patient", how="left", suffixes=("", "_imuno"))

    # If clinical exists, join on patient (or 'HC' mapped to patient)
    if df_clin is not None:
        if "HC" in df_clin.columns and "patient" not in df_clin.columns:
            df_clin = df_clin.rename(columns={"HC": "patient"})
        if "patient" in df_clin.columns:
            df_clin["patient"] = df_clin["patient"].astype(str)
            df_merge = df_merge.merge(df_clin, on="patient", how="left", suffixes=("", "_clin"))

    # ---------- Clean/rename a few common columns (friendly names) ----------
    rename_map = {
        "area_perc_mean": "% Hoxa marked area",
        "idade": "Age",
        "area_um2_mean": "Avg Hoxa marked area (um2)",
    }
    for old, new in rename_map.items():
        if old in df_merge.columns and new not in df_merge.columns:
            df_merge = df_merge.rename(columns={old: new})

    # ---------- Save a tidy table for downstream stats ----------
    final_table_csv = base / "final_results/final_measurements_table.csv"
    ensure_dir(final_table_csv)
    df_merge.to_csv(final_table_csv, index=False)

    # Also save a CASE-only table if types exist
    if "type" in df_merge.columns and (df_merge["type"] == "CASE").any():
        df_pop = df_merge[df_merge["type"] == "CASE"].copy()
        pop_csv = base / "final_results/final_measurements_table_pop.csv"
        ensure_dir(pop_csv)
        df_pop.to_csv(pop_csv, index=False)

    # ---------- Determine percentage columns for morpho plot ----------
    perc_cols = [c for c in df_morpho.columns if c.startswith("%")]
    if perc_cols:
        plot_morpho_proportions(df_morpho, plots_dir / "avg_tissue.png", perc_cols)

    # ---------- Grouped regression plots (pick some sensible pairs) ----------
    # If available, try a few common relationships
    candidates_x = [c for c in ["Age", "IMC", "BMI", "PV", "gestacoes", "partos", "cesarias", "abortos"] if c in df_merge.columns]
    candidates_y = perc_cols + [c for c in ["% Hoxa marked area"] if c in df_merge.columns]
    for x in candidates_x:
        for y in candidates_y:
            out = plots_dir / f"grouped_plot_{x}_x_{y}.png"
            try:
                grouped_plot(df_merge, x, y, out)
            except Exception as e:
                # Skip pair if it fails (e.g., non-numeric)
                print(f"[WARN] Could not plot {y} vs {x}: {e}")

    # ---------- Violin and box plots for % columns ----------
    for col in candidates_y:
        try:
            col_violinplot(df_merge, col, plots_dir / f"violin_{col.replace(' ', '_')}.png")
            col_boxplot(df_merge, col, plots_dir / f"boxplot_{col.replace(' ', '_')}.png")
        except Exception as e:
            print(f"[WARN] Could not plot distributions for {col}: {e}")

    # ---------- PCA: use numeric feature columns only ----------
    numeric_cols = df_merge.select_dtypes(include=[np.number]).columns.tolist()
    # Keep a modest set (avoid id-like or count-only columns)
    feature_candidates = [c for c in numeric_cols if c not in {"patient"}]
    if len(feature_candidates) >= 3:
        loadings_df, scores_df = perform_pca(df_merge, feature_candidates, n_components=args.n_pca)
        plot_pca_loadings_heatmap(loadings_df, plots_dir / "pca_loadings_heatmap.png")
        plot_pca_2d(scores_df, plots_dir / "pca_2d_projection.png")

    # ---------- Simple stats table by type (means of a few columns) ----------
    small_cols = [c for c in ["HG", "Age", "PV", "gestacoes", "partos", "cesarias", "abortos", "IMC", "BMI"] if c in df_merge.columns]
    if "type" in df_merge.columns and small_cols:
        grouped_df = df_merge.groupby("type")[small_cols].mean(numeric_only=True)
        stats_csv = base / "final_results/stats_table.csv"
        ensure_dir(stats_csv)
        grouped_df.to_csv(stats_csv, index=True)

    # Console summary
    print("\nSaved outputs:")
    print(f"  - {final_table_csv}")
    if (base / 'final_results/final_measurements_table_pop.csv').exists():
        print(f"  - {base / 'final_results/final_measurements_table_pop.csv'}")
    if (base / 'final_results/stats_table.csv').exists():
        print(f"  - {base / 'final_results/stats_table.csv'}")
    print(f"  - plots (*.png) in: {plots_dir}")


if __name__ == "__main__":
    main()
