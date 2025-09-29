#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
results_to_dataframe_SIMPLE_ARGPARSE.py
======================================

Goal (in plain words)
---------------------
1) Read ALL CSV result files produced by your image analysis pipeline from:
      <base_path>/database_csvs/*.csv
2) Turn their semi-structured "long" rows into a clean table (one row per image).
3) Compute **per-image** and **per-patient** tissue areas and percentages.
4) Save tidy outputs to CSV/Excel for stats/plots.

Main outputs
------------
- <base_path>/morpho_results/area_morpho.csv
    * One row per **patient**, with:
      patient id, type (CT/POP), number of images, total area (per tissue) and % per tissue.

- <base_path>/df_hoxa_final.csv           (optional; only if you provide --hoxa-csv)
    * One row per **patient** with the same columns as above **plus** a 'hoxa' column
      merged from your provided mapping file (patient,hoxa).

- <base_path>/df_hoxa_extra.xlsx          (optional; only if you provide --extra-csv)
    * Same as df_hoxa_final.csv but merged with extra clinical columns
      (e.g., IDADE, IMC, etc.) from your extra CSV. Useful for correlation plots.

Keep it simple
--------------
This script is intentionally **simple and heavily commented** so it reads like a tutorial.
If you don't have HOXA or extra clinical data, just omit those flags—the main morphometry
table `area_morpho.csv` will still be produced.

"""

# ==============
# 1) IMPORTS
# ==============
from pathlib import Path
import argparse
import glob
import csv

import numpy as np
import pandas as pd


# =====================
# 2) HELPER FUNCTIONS
# =====================
def read_csv2df(path: str) -> pd.DataFrame:
    """
    Convert a "long" CSV exported by the upstream tool into a single-row record.

    The CSV typically has rows like: [<idx>, <key>, <value>], for example:
        0, 'Image', 'CT123_..._P01.tif'
        1, '(Live) adiposo area px^2', '12345'
        2, '(Live) musculo area px^2', '6789'
        ...

    We scan the file and build a dict for each 'Image' block, then make a DataFrame.
    """
    raw = pd.read_csv(path, header=None).dropna(how="all")
    records = []
    current = {}

    for _, row in raw.iterrows():
        key = row[1]
        val = row[2]
        if key == "Image":
            if current:
                records.append(current)
            current = {"Image": val}
        else:
            # try to parse numeric values; if fail, keep as text
            try:
                current[key] = pd.to_numeric(val)
            except Exception:
                current[key] = val

    if current:
        records.append(current)

    df = pd.DataFrame(records).fillna(0)
    return df


def find_cols(symbol: str, df: pd.DataFrame) -> list[str]:
    """Return all column names that contain a given substring."""
    return [c for c in df.columns if symbol in str(c)]


def add_patient_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    From the 'Image' filename, extract:
      - type: 'CT' (control) or 'POP' (case). If neither is found, mark 'ERROR'.
      - patient: numeric id found before the first underscore.
      - photo: image index or label at the end of the name.
    Adjust this if your naming scheme is different.
    """
    type_list, patient_list, photo_list = [], [], []
    for name in df["Image"].astype(str).values:
        photo = name.split(".")[0].split("_")[-1]
        photo_list.append(photo)

        first = name.split("_")[0]
        if "CT" in name:
            type_list.append("CT")
            pid = first.split("T")[1]
        elif "POP" in name:
            type_list.append("POP")
            pid = first.split("OP")[1]
        else:
            type_list.append("ERROR")
            pid = ""

        patient_list.append(pid)

    df = df.copy()
    df["type"] = type_list
    df["patient"] = patient_list
    df["photo"] = photo_list
    return df


def clean_and_compute_per_image(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Keep only relevant columns (drop generic non-feature columns commonly present).
    - Convert area columns to float.
    - Compute total non-background area per image.
    - Compute % per tissue for each image.
    """
    # Drop obvious metadata columns if they exist
    drop_cols = [c for c in ["Name", "Object type"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Identify tissue area columns (contain 'area' but not 'background')
    area_cols = []
    for c in df.columns:
        if "area" in str(c) and "background" not in str(c).lower():
            area_cols.append(c)

    # Coerce strings like "12.345" to numeric (remove dots used as thousands separators if present)
    for col in area_cols:
        if df[col].map(type).eq(str).any():
            df.loc[df[col].map(type).eq(str), col] = df.loc[df[col].map(type).eq(str), col].str.replace(".", "")
    df[area_cols] = df[area_cols].astype(float)

    # Total area (excluding background)
    df["tot-area_no-bkg"] = df[area_cols].sum(axis=1)

    # Friendly names for percentages
    perc_map = {
        "(Live) conj_frouxo area px^2": "% Loose Connective Tissue",
        "(Live) adiposo area px^2": "% Adipose Tissue",
        "(Live) musculo area px^2": "% Smooth Muscle",
        "(Live) conj_denso area px^2": "% Dense Connective Tissue",
    }
    # Fill missing keys automatically with a generic "% <name>"
    for col in area_cols:
        pct_col = perc_map.get(col, f"% {col}")
        df[pct_col] = np.where(df["tot-area_no-bkg"] > 0, df[col] / df["tot-area_no-bkg"] * 100.0, 0.0)

    return df, area_cols, perc_map


def aggregate_per_patient(df: pd.DataFrame, area_cols: list[str], perc_map: dict) -> pd.DataFrame:
    """
    Sum areas by patient, then recompute percentages at the patient level.
    Also counts how many images belong to each patient.
    """
    # Sum all numeric columns per patient
    df_sum = df.groupby("patient").sum(numeric_only=True)

    # Remove per-image percentage columns before recomputing
    perc_cols = [c for c in df.columns if str(c).startswith("% ") or str(c).startswith("%")]
    df_sum = df_sum.drop(columns=[c for c in perc_cols if c in df_sum.columns], errors="ignore")

    # Recompute patient-level percentages + copy total area columns with clearer names
    total_map = {
        "(Live) conj_frouxo area px^2": "tot_area conj_frouxo px^2",
        "(Live) adiposo area px^2": "tot_area adiposo px^2",
        "(Live) musculo area px^2": "tot_area musculo px^2",
        "(Live) conj_denso area px^2": "tot_area conj_denso px^2",
    }

    for col in area_cols:
        if col in df_sum.columns:
            df_sum[total_map.get(col, f"tot_area {col}")] = df_sum[col]
            pct_col = perc_map.get(col, f"% {col}")
            df_sum[pct_col] = np.where(df_sum["tot-area_no-bkg"] > 0, df_sum[col] / df_sum["tot-area_no-bkg"] * 100.0, 0.0)

    # Derive patient "type" from the images belonging to each patient:
    # we take the most frequent (mode) of the per-image types for that patient.
    type_mode = df.groupby("patient")["type"].agg(lambda s: s.mode().iat[0] if not s.mode().empty else "ERROR")
    # Build a count of images per patient
    img_counts = df.groupby("patient")["Image"].count().rename("num_imgs")

    df_sum = df_sum.join(type_mode).join(img_counts)

    # Optional: normalize type labels if needed (e.g., CT/POP only)
    df_sum["type"] = df_sum["type"].replace({"CONTROL": "CT", "CASE": "POP"})

    # Sort for neatness
    df_sum = df_sum.sort_values(by=["type", "patient"])

    return df_sum.reset_index()


def attach_hoxa(df_pat: pd.DataFrame, hoxa_csv: Path | None) -> pd.DataFrame | None:
    """
    Optionally merge a (patient,hoxa) CSV onto the patient summary.
    The CSV must have at least two columns named 'patient' and 'hoxa'.
    Returns the merged DataFrame, or None if no CSV provided.
    """
    if hoxa_csv is None:
        return None
    if not hoxa_csv.exists():
        raise FileNotFoundError(f"HOXA CSV not found: {hoxa_csv}")

    hoxa = pd.read_csv(hoxa_csv)
    # Be resilient about column names
    cols = {c.lower(): c for c in hoxa.columns}
    pcol = cols.get("patient")
    hcol = cols.get("hoxa") or cols.get("% hoxa") or cols.get("hoxa_percent")
    if not pcol or not hcol:
        raise ValueError("The HOXA CSV must contain columns named 'patient' and 'hoxa'.")

    # Make patient ids comparable types
    hoxa = hoxa.rename(columns={pcol: "patient", hcol: "hoxa"})
    hoxa["patient"] = hoxa["patient"].astype(str)
    df = df_pat.copy()
    df["patient"] = df["patient"].astype(str)

    merged = df.merge(hoxa[["patient", "hoxa"]], on="patient", how="inner")
    return merged


def attach_extra(df: pd.DataFrame, extra_csv: Path | None) -> pd.DataFrame | None:
    """
    Optionally merge any extra clinical CSV that contains a column called 'patient' or 'HC'.
    Keeps all columns from the extra file and the patient summary.
    Returns the merged DataFrame, or None if no CSV provided.
    """
    if extra_csv is None:
        return None
    if not extra_csv.exists():
        raise FileNotFoundError(f"Extra CSV not found: {extra_csv}")

    extra = pd.read_csv(extra_csv)
    # Support both 'patient' and 'HC' as identifiers
    id_col = "patient" if "patient" in extra.columns else ("HC" if "HC" in extra.columns else None)
    if id_col is None:
        raise ValueError("Extra CSV must have a 'patient' or 'HC' column to merge on.")

    extra = extra.rename(columns={id_col: "patient"})
    extra["patient"] = extra["patient"].astype(str)

    base = df.copy()
    base["patient"] = base["patient"].astype(str)

    merged = base.merge(extra, on="patient", how="left")
    return merged


# =============
# 3) MAIN FLOW
# =============
def main():
    parser = argparse.ArgumentParser(
        description="Aggregate morphometry CSVs into tidy per-patient tables and (optionally) merge HOXA/extra data."
    )
    parser.add_argument("base_path", type=str, help="Base folder that contains the 'database_csvs' subfolder.")
    parser.add_argument("--patterns", nargs="+", default=["*.csv"],
                        help="Glob patterns (relative to <base_path>/database_csvs) to read. Default: *.csv")
    parser.add_argument("--hoxa-csv", type=str, default=None,
                        help="Optional CSV file with columns 'patient','hoxa' to merge (absolute or relative path).")
    parser.add_argument("--extra-csv", type=str, default=None,
                        help="Optional CSV file with extra clinical columns to merge (must include 'patient' or 'HC').")
    parser.add_argument("--out-morpho", type=str, default="morpho_results/area_morpho.csv",
                        help="Output CSV (relative to base_path). Default: morpho_results/area_morpho.csv")
    parser.add_argument("--out-hoxa", type=str, default="df_hoxa_final.csv",
                        help="Output CSV for HOXA merge (relative to base_path).")
    parser.add_argument("--out-extra", type=str, default="df_hoxa_extra.xlsx",
                        help="Output Excel for HOXA+extra merge (relative to base_path).")

    args = parser.parse_args()

    base = Path(args.base_path)
    db_folder = base / "database_csvs"
    if not db_folder.exists():
        raise FileNotFoundError(f"'database_csvs' folder was not found under: {base}")

    # 1) Read & stack all CSVs
    paths = []
    for pat in args.patterns:
        paths.extend(glob.glob(str(db_folder / pat)))
    if not paths:
        raise FileNotFoundError(f"No CSV files found under: {db_folder} with patterns {args.patterns}")

    df_list = [read_csv2df(p) for p in paths]
    df = pd.concat(df_list, axis=0, ignore_index=True)

    # 2) Add patient/type/photo columns
    df = add_patient_cols(df)

    # 3) Per-image cleaning and percentages
    df, area_cols, perc_map = clean_and_compute_per_image(df)

    # 4) Aggregate per patient
    df_pat = aggregate_per_patient(df, area_cols, perc_map)

    # 5) Save morphometry CSV
    out_morpho = base / args.out_morpho
    out_morpho.parent.mkdir(parents=True, exist_ok=True)
    df_pat.to_csv(out_morpho, index=False)
    print(f"[OK] Wrote per-patient morphometry to: {out_morpho}")

    # 6) Optional: attach HOXA
    df_hoxa = attach_hoxa(df_pat, Path(args.hoxa_csv)) if args.hoxa_csv else None
    if df_hoxa is not None:
        out_hoxa = base / args.out_hoxa
        df_hoxa.to_csv(out_hoxa, index=False)
        print(f"[OK] Wrote HOXA-merged table to: {out_hoxa}")

    # 7) Optional: attach extra clinical data
    if df_hoxa is not None and args.extra_csv:
        df_hoxa_extra = attach_extra(df_hoxa, Path(args.extra_csv))
        if df_hoxa_extra is not None:
            out_extra = base / args.out_extra
            df_hoxa_extra.to_excel(out_extra, index=False)
            print(f"[OK] Wrote HOXA+extra table to: {out_extra}")

    # Summary for the console
    print("\nSummary:")
    print(f"  Images read        : {len(df)}")
    print(f"  Patients summarized: {len(df_pat)}")
    if df_hoxa is not None:
        print(f"  HOXA patients      : {len(df_hoxa)}")

if __name__ == "__main__":
    main()
