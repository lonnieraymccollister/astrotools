#!/usr/bin/env python3

import csv
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table
from photutils.segmentation import detect_threshold, detect_sources, SourceCatalog
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line
import matplotlib.pyplot as plt

# Configuration
ELLIPTICITY_THRESHOLD = 1.00
NSIGMA                = 3.0
MIN_AREA              = 5

# Trail‐detection tuning
HOUGH_THRESHOLD       = 1313
HOUGH_MIN_LENGTH      = 30
HOUGH_LINE_GAP        = 10

def contrast_stretch(image, p_low=1, p_high=99):
    """
    Linearly stretch image contrast between the p_low and p_high percentiles.
    Returns a float image clipped to [0,1].
    """
    lo, hi = np.percentile(image[np.isfinite(image)], (p_low, p_high))
    stretched = (image - lo) / (hi - lo)
    return np.clip(stretched, 0.0, 1.0)

def display_trails(norm, lines, title="Detected Trails"):
    """
    Show the normalized image with red lines overlaid.
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(norm, cmap='gray', origin='lower')
    for p0, p1 in lines:
        plt.plot([p0[0], p1[0]],
                 [p0[1], p1[1]],
                 'r-', linewidth=1.5)
    plt.title(title)
    plt.axis('off')
    plt.show()

def analyze_fits_photutils(fits_path, show_plot=True):
    # Load raw data
    with fits.open(fits_path) as hdul:
        data = hdul[0].data.astype('float64')

    # Contrast stretch & normalize
    norm = contrast_stretch(data)

    # Star detection via segmentation + moments
    thresh = detect_threshold(data, nsigma=NSIGMA)
    segm   = detect_sources(data, thresh, npixels=MIN_AREA)
    catalog = SourceCatalog(data, segm) if segm is not None else None

    stars_flags = []
    if catalog is not None and len(catalog) > 0:
        tbl = catalog.to_table(columns=[
            'xcentroid', 'ycentroid',
            'semimajor_sigma', 'semiminor_sigma'
        ])
        tbl['ellipticity'] = 1.0 - (tbl['semiminor_sigma'] / tbl['semimajor_sigma'])
        for row in tbl:
            if row['ellipticity'] > ELLIPTICITY_THRESHOLD:
                stars_flags.append({
                    'x': float(row['xcentroid']),
                    'y': float(row['ycentroid']),
                    'ellipticity': float(row['ellipticity'])
                })

    # Trail detection via edges + Hough on stretched data
    edges = canny(norm, sigma=1.0)
    lines = probabilistic_hough_line(
        edges,
        threshold=HOUGH_THRESHOLD,
        line_length=HOUGH_MIN_LENGTH,
        line_gap=HOUGH_LINE_GAP
    )
    trails = [{'x0': p0[0], 'y0': p0[1], 'x1': p1[0], 'y1': p1[1]}
              for p0, p1 in lines]

    # Optionally show each detection
    if show_plot:
        display_trails(norm, lines, title=fits_path.name)

    return stars_flags, trails

def main():
    input_dir = Path(input("Enter the directory containing FITS files --> ").strip())
    if not input_dir.is_dir():
        print(f"Directory not found: {input_dir}")
        return

    fits_files = sorted(
        [f for f in input_dir.iterdir()
         if f.is_file() and f.suffix.lower() in ('.fit', '.fits')]
    )
    if not fits_files:
        print("No FIT or FITS files found in directory.")
        return

    star_csv  = input_dir / "flagged_stars_report.csv"
    trail_csv = input_dir / "detected_trails_report.csv"

    with star_csv.open("w", newline="") as sf, trail_csv.open("w", newline="") as tf:
        star_writer  = csv.DictWriter(sf, fieldnames=["filename","x","y","ellipticity"])
        trail_writer = csv.DictWriter(tf, fieldnames=["filename","x0","y0","x1","y1"])
        star_writer.writeheader()
        trail_writer.writeheader()

        for fits_file in fits_files:
            stars, trails = analyze_fits_photutils(fits_file, show_plot=True)

            for s in stars:
                star_writer.writerow({"filename": fits_file.name, **s})
            for t in trails:
                trail_writer.writerow({"filename": fits_file.name, **t})

            print(f"{fits_file.name}: {len(stars)} flagged stars, {len(trails)} trails")

    print("\nReports generated:")
    print(" -", star_csv.name)
    print(" -", trail_csv.name)

if __name__ == "__main__":
    main()