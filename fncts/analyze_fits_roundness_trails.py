#!/usr/bin/env python3

import csv
from pathlib import Path

import numpy as np
from astropy.io import fits
from photutils.segmentation import detect_threshold, detect_sources, SourceCatalog
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line
import matplotlib.pyplot as plt

# Configuration
ROUNDNESS_THRESHOLD   = 0.5
NSIGMA                = 3.0
MIN_AREA              = 5

def contrast_stretch(image, p_low=1, p_high=99):
    lo, hi = np.percentile(image[np.isfinite(image)], (p_low, p_high))
    stretched = (image - lo) / (hi - lo)
    return np.clip(stretched, 0.0, 1.0)

def display_trails(norm, lines, title="Detected Trails"):
    plt.figure(figsize=(8, 8))
    plt.imshow(norm, cmap='gray', origin='lower')
    for p0, p1 in lines:
        plt.plot([p0[0], p1[0]], [p0[1], p1[1]], 'r-', linewidth=1.5)
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
    tbl = None
    if catalog is not None and len(catalog) > 0:
        tbl = catalog.to_table(columns=[
            'xcentroid', 'ycentroid',
            'semimajor_sigma', 'semiminor_sigma'
        ])
        # Roundness: b/a
        tbl['roundness'] = tbl['semiminor_sigma'] / tbl['semimajor_sigma']

        for row in tbl:
            # Current convention: flagged = fail (roundness <= ROUNDNESS_THRESHOLD)
            if row['roundness'] <= ROUNDNESS_THRESHOLD:
                stars_flags.append({
                    'x': float(row['xcentroid']),
                    'y': float(row['ycentroid']),
                    'roundness': float(row['roundness'])
                })

    # --- Adaptive Hough tuning for trail detection ---
    edges = canny(norm, sigma=1.0)
    edge_count = int(np.sum(edges))
    total_pixels = edges.size
    edge_ratio = edge_count / total_pixels if total_pixels > 0 else 0.0

    # Adaptive threshold: base + scaled by edge density
    hough_threshold = int(max(5, 10 + edge_ratio * 200))

    # Adaptive minimum line length based on image diagonal
    height, width = data.shape
    diag = np.sqrt(height**2 + width**2)
    hough_min_length = int(max(20, diag * 0.02))   # 2% of diagonal

    # Adaptive line gap
    hough_line_gap = int(max(5, diag * 0.005))     # 0.5% of diagonal

    lines = probabilistic_hough_line(
        edges,
        threshold=hough_threshold,
        line_length=hough_min_length,
        line_gap=hough_line_gap
    )

    trails = [{'x0': p0[0], 'y0': p0[1], 'x1': p1[0], 'y1': p1[1]}
              for p0, p1 in lines]

    if show_plot:
        display_trails(norm, lines, title=fits_path.name)

    # Return flagged stars, trails, the full table (or None), and Hough diagnostics
    return stars_flags, trails, tbl, {
        'hough_threshold': hough_threshold,
        'hough_min_length': hough_min_length,
        'hough_line_gap': hough_line_gap,
        'edge_count': edge_count,
        'edge_ratio': edge_ratio
    }

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
        star_writer  = csv.DictWriter(sf, fieldnames=["filename","x","y","roundness"])
        trail_writer = csv.DictWriter(tf, fieldnames=["filename","x0","y0","x1","y1"])
        star_writer.writeheader()
        trail_writer.writeheader()

        for fits_file in fits_files:
            # unpack the returned values
            stars, trails, tbl, hough_info = analyze_fits_photutils(fits_file, show_plot=True)

            # write flagged stars (no per-star console print)
            for s in stars:
                star_writer.writerow({"filename": fits_file.name, **s})

            # write trails
            for t in trails:
                trail_writer.writerow({"filename": fits_file.name, **t})

            # Compute totals and ratio
            total_detected = len(tbl) if tbl is not None else 0
            flagged_count = len(stars)   # flagged = roundness <= ROUNDNESS_THRESHOLD (fail)
            passed_count = total_detected - flagged_count

            if total_detected > 0:
                if flagged_count > 0:
                    ratio_passed_to_flagged = passed_count / flagged_count
                    ratio_str = f"{ratio_passed_to_flagged:.3f}"
                else:
                    ratio_str = "inf"  # all passed

                print(
                    f"{fits_file.name}: total_detected={total_detected}, "
                    f"passed={passed_count}, flagged(fail)={flagged_count}, trails={len(trails)}, "
                    f"passed:flagged={ratio_str}"
                )
            else:
                print(
                    f"{fits_file.name}: total_detected=0, passed=0, flagged=0, trails={len(trails)}, "
                    f"no roundness measured"
                )

            # Print adaptive Hough parameters for diagnostics
            print(
                f"  Hough params: threshold={hough_info['hough_threshold']}, "
                f"min_length={hough_info['hough_min_length']}, gap={hough_info['hough_line_gap']}, "
                f"edge_count={hough_info['edge_count']}, edge_ratio={hough_info['edge_ratio']:.4f}"
            )

    print("\nReports generated:")
    print(" -", star_csv.name)
    print(" -", trail_csv.name)

if __name__ == "__main__":
    main()
