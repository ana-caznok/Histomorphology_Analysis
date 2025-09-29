#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_area_proportion_SIMPLE_ARGPARSE.py
==========================================

Goal (in plain words)
---------------------
Walk through a base folder, open each image mask, count how many pixels are
"positive" (non‑zero or above a small threshold), and save a small summary
table to a CSV file called `area_imunos_marcadas.csv` (or a name you choose).

Who is this for?
----------------
This version is intentionally **simple and highly commented** so that a
biologist or a beginner programmer can read and modify it without stress.

How to run it (without editing the file)
----------------------------------------
Use the command line and pass your folder path (and optional settings):

Windows example:
    python extract_area_proportion_SIMPLE_ARGPARSE.py "D:\analise de imagem\luana\imunos_c_morpho\imunos_c_morpho" --threshold 0 --pixel-size-um 0.325

macOS / Linux example:
    python3 extract_area_proportion_SIMPLE_ARGPARSE.py "/path/to/imunos_c_morpho" --threshold 0 --pixel-size-um 0.325

What you need to run it
-----------------------
- Python 3.x
- Libraries: Pillow (PIL) and NumPy
  Install with:  pip install pillow numpy

Important definitions
---------------------
- "Positive pixel": any pixel whose intensity is **greater than THRESHOLD**.
  If your masks are binary (0 = background, 255 = foreground), leave THRESHOLD = 0.
- "Area in pixels²": the number of positive pixels (same as counting foreground).
- "Area %": (positive pixels / total pixels) * 100.
- "Area in µm²": only computed if you provide PIXEL_SIZE_UM (micrometers per pixel).

File types
----------
We look for images with these extensions by default: .tif, .tiff, .png, .jpg, .jpeg
(You can change or add more via the --patterns argument.)
"""

# ============================
# 1) IMPORTS (kept very small)
# ============================
from pathlib import Path
import csv
import argparse
from typing import Optional, List

import numpy as np
from PIL import Image


# ============================================
# 2) DEFAULTS (simple knobs you might tweak)
# ============================================
# These are the *defaults* used by argparse. You can override them from the command line.
DEFAULT_THRESHOLD: float = 0.0                 # pixels > THRESHOLD are considered "positive"
DEFAULT_PIXEL_SIZE_UM: Optional[float] = None  # set e.g. 0.325 to also compute area in µm²
DEFAULT_OUTPUT_CSV_NAME: str = "area_imunos_marcadas.csv"
DEFAULT_PATTERNS: List[str] = ["*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"]


# ====================================================
# 3) HELPER FUNCTIONS (like in a notebook: define first)
# ====================================================
def load_image_as_grayscale(image_path: Path) -> np.ndarray:
    """
    Open an image and convert it to a simple 2D grayscale NumPy array.

    Why grayscale?
    --------------
    For area counting, we only need intensity values. Grayscale makes thresholding
    straightforward and avoids color complications.
    """
    # Image.open handles many formats (including TIFF). Convert("L") -> grayscale [0..255].
    img = Image.open(image_path).convert("L")
    arr = np.array(img)
    return arr


def measure_area_from_mask(
    image_array: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    pixel_size_um: Optional[float] = DEFAULT_PIXEL_SIZE_UM
) -> dict:
    """
    Count how many pixels are "positive" (value > threshold) and compute simple metrics.

    Returns a dictionary ready to be written as a CSV row with these keys:
    - width_px, height_px, total_px
    - positive_px, area_px2, area_perc, area_um2 (area_um2 may be None if pixel_size_um is None)
    """
    # Ensure the array is 2D. If a stack or multi-channel sneaks in, take a simple
    # max projection across extra axes (this treats any signal in a stack as "positive").
    if image_array.ndim > 2:
        image_array = image_array.max(axis=0)

    height_px, width_px = image_array.shape[:2]
    total_px = int(width_px * height_px)

    # Positive mask: pixels strictly greater than the chosen threshold
    positive_mask = image_array.astype(np.float32) > float(threshold)
    positive_px = int(positive_mask.sum())

    # Area in pixel units (just the count of positives)
    area_px2 = positive_px

    # Percentage of the image that is positive
    area_perc = (positive_px / total_px * 100.0) if total_px > 0 else 0.0

    # Optional physical area in µm², if calibration is known
    area_um2: Optional[float] = None
    if pixel_size_um is not None:
        area_um2 = float(positive_px) * (float(pixel_size_um) ** 2)

    return dict(
        width_px=width_px,
        height_px=height_px,
        total_px=total_px,
        positive_px=positive_px,
        area_px2=area_px2,
        area_perc=area_perc,
        area_um2=area_um2,
    )


def find_images_recursively(base_path: Path, patterns: List[str]) -> List[Path]:
    """
    Search the base folder and all subfolders for files that match the given patterns.
    Example patterns: ["*.tif", "*.png"]
    """
    all_files: List[Path] = []
    for pat in patterns:
        all_files.extend(base_path.rglob(pat))
    # Remove duplicates and sort for a stable, tidy order
    all_files = sorted(set(all_files))
    return all_files


def save_rows_to_csv(rows: List[dict], output_csv_path: Path) -> None:
    """
    Save a list of dictionaries as a CSV file. Each dictionary is one row.
    The CSV column order is defined here for clarity and consistency.
    """
    # Define the CSV column order clearly (beginner-friendly)
    fieldnames = [
        "file_path", "file_name", "parent_dir",
        "width_px", "height_px", "total_px",
        "positive_px", "area_px2", "area_perc", "area_um2",
    ]

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with output_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ===========================================
# 4) MAIN LOOP (do the work now, step by step)
# ===========================================
def main() -> None:
    # --- argparse setup ---
    parser = argparse.ArgumentParser(
        description=(
            "Compute area proportions from mask images (positive pixels > threshold) "
            "and export results to a CSV file."
        )
    )
    parser.add_argument(
        "base_path",
        type=str,
        help="Base folder containing your images (searches recursively).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Pixel intensity threshold (default: {DEFAULT_THRESHOLD}). Pixels > threshold are positive.",
    )
    parser.add_argument(
        "--pixel-size-um",
        type=float,
        default=DEFAULT_PIXEL_SIZE_UM,
        help=(
            "Pixel size in micrometers per pixel (optional). "
            "If provided, area_um2 is computed as positive_px * (pixel_size_um ** 2)."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_CSV_NAME,
        help=f"Name of the output CSV file (default: {DEFAULT_OUTPUT_CSV_NAME}).",
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=DEFAULT_PATTERNS,
        help=(
            "File patterns to search for (space-separated). "
            f'Default: {" ".join(DEFAULT_PATTERNS)}'
        ),
    )
    args = parser.parse_args()

    # --- convert to Path and check existence ---
    base = Path(args.base_path)
    if not base.exists():
        raise FileNotFoundError(
            f"Base path does not exist:\n{base}\n"
            "Please check the path you passed as the first argument."
        )

    print(f"Looking for images under:\n  {base}\n")

    # 1) Find images using the patterns provided (or defaults)
    image_paths = find_images_recursively(base, args.patterns)
    if not image_paths:
        print("No images found. Adjust --patterns or check your folder.")
        return

    print(f"Found {len(image_paths)} image(s). Measuring areas...\n")

    # 2) Measure each image and collect rows
    rows: List[dict] = []
    for p in image_paths:
        try:
            arr = load_image_as_grayscale(p)
            metrics = measure_area_from_mask(
                arr,
                threshold=args.threshold,
                pixel_size_um=args.pixel_size_um
            )

            # Add file identity info alongside the numeric metrics
            row = dict(
                file_path=str(p),
                file_name=p.name,
                parent_dir=p.parent.name,
                **metrics,
            )
            rows.append(row)
        except Exception as e:
            # If a file fails, print a warning and move on to the next one
            print(f"[WARN] Could not process {p}: {e}")

    # 3) Save results as CSV inside the base folder
    output_csv = base / args.output
    save_rows_to_csv(rows, output_csv)

    # 4) Print a tiny summary to the screen
    avg_area = float(np.mean([r["area_perc"] for r in rows])) if rows else float("nan")
    print(f"\nDone! Wrote {len(rows)} rows to:\n  {output_csv}")
    print(f"Average area% across files: {avg_area:.3f}")


if __name__ == "__main__":
    main()
