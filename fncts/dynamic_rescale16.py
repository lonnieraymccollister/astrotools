#!/usr/bin/env python3
"""
dynamic_rescale16_histmatch.py
Modified from dynamic_rescale16.py to perform histogram matching of processed tiles
(using the brightest processed tile as reference) before reassembly.

Usage: python dynamic_rescale16_histmatch.py
Follow interactive prompts to choose mode 01 (split), 02 (process tiles), or 03 (reassemble with histogram matching).
"""
import os
import sys
import re
import math
from pathlib import Path

import numpy as np
import cv2
from astropy.io import fits

# -------------------------
# Helper I/O functions
# -------------------------
def load_fits(file_path):
    with fits.open(file_path) as hdul:
        return hdul[0].data, hdul[0].header

def write_fits(data, header, outpath):
    if header is None:
        fits.writeto(outpath, data, overwrite=True)
    else:
        fits.writeto(outpath, data, header=header, overwrite=True)

# -------------------------
# Tile split / reassemble
# -------------------------
def split_image(image, tile_size=(600, 600), output_dir="tiles"):
    os.makedirs(output_dir, exist_ok=True)
    h, w = image.shape
    tile_h, tile_w = tile_size
    tiles = []

    for i in range(0, h, tile_h):
        for j in range(0, w, tile_w):
            sub_image = image[i:i+tile_h, j:j+tile_w]
            sub_h, sub_w = sub_image.shape
            padded_tile = np.zeros(tile_size, dtype=image.dtype)
            padded_tile[:sub_h, :sub_w] = sub_image
            tile_file = os.path.join(output_dir, f"tile_{i}_{j}_{sub_h}_{sub_w}.fits")
            fits.writeto(tile_file, padded_tile, overwrite=True)
            tiles.append(tile_file)

    return tiles

def reassemble_image(tiles, original_shape):
    final_image = np.zeros(original_shape, dtype=np.float64)
    pattern = r"tile_(\d+)_(\d+)_(\d+)_(\d+)"
    for tile_file in tiles:
        base = os.path.basename(tile_file)
        match = re.search(pattern, base)
        if not match:
            print(f"Filename {base} does not match expected pattern, skipping.")
            continue
        i_str, j_str, sub_h_str, sub_w_str = match.groups()
        i, j, sub_h, sub_w = map(int, [i_str, j_str, sub_h_str, sub_w_str])
        with fits.open(tile_file) as hdul:
            data = hdul[0].data
            final_image[i:i+sub_h, j:j+sub_w] = data[:sub_h, :sub_w]
    return final_image

# -------------------------
# Block rescale fallback
# -------------------------
def block_rescale_float64(img64, out64, block_size):
    """
    Pure-Python fallback for warp_affine_mask_rescale.
    Applies per-block linear normalization (min..max -> 0..65535) over non-overlapping square blocks.
    img64 and out64 are 2D float64 arrays of the same shape.
    block_size should be int > 0.
    """
    H, W = img64.shape
    bs = max(1, int(block_size))
    for y0 in range(0, H, bs):
        for x0 in range(0, W, bs):
            y1 = min(y0+bs, H)
            x1 = min(x0+bs, W)
            block = img64[y0:y1, x0:x1]
            bmin = np.nanmin(block)
            bmax = np.nanmax(block)
            if bmax > bmin:
                norm = (block - bmin) / (bmax - bmin)
            else:
                norm = np.zeros_like(block)
            out64[y0:y1, x0:x1] = norm * 65535.0

# -------------------------
# Tile processing
# -------------------------
def process_tile(tile_file, width_of_square, bin_value, gamma_value, resize_factor, resize_div):
    print(f"\nProcessing tile: {tile_file}")

    with fits.open(tile_file) as hdul:
        header = hdul[0].header
        image_data = hdul[0].data

    if image_data is None:
        print(f"Tile {tile_file} contains no data, skipping.")
        return

    # Ensure 2D array
    if image_data.ndim != 2:
        raise ValueError("Tile data must be 2D")

    # Normalize to 0..65535 as float
    minv = np.nanmin(image_data)
    maxv = np.nanmax(image_data)
    if maxv == minv:
        norm_image = np.zeros_like(image_data, dtype=np.uint16)
    else:
        norm_image = ((image_data - minv) / (maxv - minv) * 65535.0).astype(np.uint16)

    # Resize using OpenCV (keep as uint16 for resize then cast)
    fx = (resize_factor / resize_div)
    fy = fx
    if fx <= 0:
        fx = fy = 1.0
    resized = cv2.resize(norm_image, None, fx=fx, fy=fy, interpolation=cv2.INTER_LANCZOS4)

    # Convert to float64 for block processing
    img64 = resized.astype(np.float64)
    out64 = np.empty_like(img64, dtype=np.float64)

    # Use fallback block rescale in Python
    block_rescale_float64(img64, out64, block_size=int(width_of_square))

    # Gamma correction (apply to scaled 0..65535 values)
    gamma_corrected = np.round(65535.0 * (out64 / 65535.0) ** float(gamma_value)).astype(np.float64)

    # Normalize before binning (using original division factor from your code)
    img_array = (gamma_corrected / 6553500.0).astype(np.float64)

    # Binning
    bin_factor = max(1, int(bin_value))
    h_img, w_img = img_array.shape
    new_height = h_img // bin_factor
    new_width = w_img // bin_factor
    if new_height == 0 or new_width == 0:
        raise ValueError("Bin factor too large for the resized image size.")

    # reshape trick for fast block sum
    trimmed = img_array[:new_height*bin_factor, :new_width*bin_factor]
    binned = trimmed.reshape(new_height, bin_factor, new_width, bin_factor).sum(axis=(1,3))

    out_filename = tile_file + '_binned_gamma_corrected_drs.fits'
    fits.writeto(out_filename, binned.astype(np.float32), header, overwrite=True)
    print(f"Tile processed and saved to {out_filename}")

# -------------------------
# Histogram matching utilities
# -------------------------
def robust_brightness_metric(data, method="p99"):
    """
    Compute a robust brightness metric for a tile.
    Masks zeros and NaNs (padding) and returns:
      - 'p99' : 99th percentile
      - 'median' : median
      - 'top_mean' : mean of top 1%
    """
    if data is None:
        return -np.inf
    flat = data.ravel()
    mask = np.isfinite(flat) & (flat != 0)
    if not np.any(mask):
        return -np.inf
    vals = flat[mask]
    if method == "p99":
        return float(np.percentile(vals, 99.0))
    if method == "median":
        return float(np.median(vals))
    if method == "top_mean":
        k = max(1, int(len(vals) * 0.01))
        top_vals = np.partition(vals, -k)[-k:]
        return float(np.mean(top_vals))
    return float(np.mean(vals))

def pick_brightest_tile(tiles, method="p99"):
    """
    Given a list of tile file paths, return the path and metric of the brightest tile.
    """
    best_tile = None
    best_val = -np.inf
    for t in tiles:
        try:
            with fits.open(t, memmap=False) as hdul:
                data = hdul[0].data
            val = robust_brightness_metric(data, method=method)
            if val > best_val:
                best_val = val
                best_tile = t
        except Exception as e:
            print(f"Warning: skipping {t} due to error: {e}")
    return best_tile, best_val

def histogram_match(source, reference, n_bins=65536):
    """
    Histogram match source to reference using CDF mapping.
    Works on 2D arrays. Ignores zero-padding (treats zeros as mask).
    Returns matched array with same dtype as source (float).
    """
    if source is None or reference is None:
        return source

    src = np.asarray(source, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)

    # Mask zeros and NaNs (assume padding zeros)
    src_mask = np.isfinite(src) & (src != 0)
    ref_mask = np.isfinite(ref) & (ref != 0)

    if not np.any(src_mask) or not np.any(ref_mask):
        return source

    src_vals = src[src_mask].ravel()
    ref_vals = ref[ref_mask].ravel()

    # If values are already small range, use unique mapping
    # Compute histograms over the value range present in reference and source
    src_min, src_max = src_vals.min(), src_vals.max()
    ref_min, ref_max = ref_vals.min(), ref_vals.max()

    if src_min == src_max or ref_min == ref_max:
        return source

    # Use a fixed number of bins but adapt range
    bins = np.linspace(min(src_min, ref_min), max(src_max, ref_max), n_bins, endpoint=True)

    src_hist, bin_edges = np.histogram(src_vals, bins=bins, density=False)
    ref_hist, _ = np.histogram(ref_vals, bins=bin_edges, density=False)

    # CDFs
    src_cdf = np.cumsum(src_hist).astype(np.float64)
    src_cdf /= src_cdf[-1]
    ref_cdf = np.cumsum(ref_hist).astype(np.float64)
    ref_cdf /= ref_cdf[-1]

    # Bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Create mapping from source bin center to reference value via interpolation of CDFs
    # For each source cdf value, find corresponding reference bin center
    interp_ref_values = np.interp(src_cdf, ref_cdf, bin_centers)

    # Map source pixels: find bin index for each source pixel and replace with interp_ref_values[idx]
    src_bin_idx = np.searchsorted(bin_edges, src_vals, side='right') - 1
    src_bin_idx = np.clip(src_bin_idx, 0, len(interp_ref_values)-1)
    mapped_vals = interp_ref_values[src_bin_idx]

    # Create output array and fill masked positions with mapped values
    out = src.copy()
    out_flat = out.ravel()
    mask_flat = src_mask.ravel()
    out_flat[mask_flat] = mapped_vals
    out = out_flat.reshape(src.shape)

    # Keep zeros (padding) as zeros
    out[~src_mask] = 0.0

    return out

# -------------------------
# Main interactive flow
# -------------------------
def main():
    print("Enter 01 to split tiles, 02 to process tiles, or 03 to combine tiles (with histogram matching)")
    choice = input("--> ").strip()
    if choice == '01':
        input_file_name = input("Enter the input FITS file name --> ").strip()
        if not input_file_name:
            print("No input provided.")
            return
        image, header = load_fits(input_file_name)
        if image is None:
            print("Input has no data.")
            return
        if image.ndim != 2:
            print("This split routine expects a 2D image in the primary HDU.")
            return
        tile_h = int(input("Tile height (default 600) --> ").strip() or 600)
        tile_w = int(input("Tile width (default 600)  --> ").strip() or 600)
        output_dir = input("Enter the output_dir name (default 'tiles') --> ").strip() or "tiles"
        tiles = split_image(image, tile_size=(tile_h, tile_w), output_dir=output_dir)
        print("Image split into tiles:")
        for t in tiles:
            print(" ", t)

    elif choice == '02':
        output_dir = input("Enter the tiles directory name (default 'tiles') --> ").strip() or "tiles"
        if not os.path.isdir(output_dir):
            print("Directory not found:", output_dir)
            return
        width_of_square = input("Enter the width of square (e.g., 5) --> ").strip() or "5"
        bin_value = input("Enter the bin value (e.g., 25) --> ").strip() or "25"
        gamma_value = float(input("Enter gamma (e.g., 0.3981) for 1 magnitude --> ").strip() or "0.3981")
        resize_factor = int(input("Resize factor (default 25) --> ").strip() or "25")
        resize_div = int(input("Resize div (default 1) --> ").strip() or "1")

        tiles = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".fits")])
        if not tiles:
            print("No .fits tiles found in", output_dir)
            return
        for tile_file in tiles:
            try:
                process_tile(tile_file, width_of_square, bin_value, gamma_value, resize_factor, resize_div)
            except Exception as e:
                print(f"Failed processing {tile_file}: {e}")

    elif choice == '03':
        output_dir = input("Enter the tiles directory name (default 'tiles') --> ").strip() or "tiles"
        input_file_name = input("Enter the original input FITS file name (for shape/header) --> ").strip()
        if not input_file_name:
            print("Original input FITS required to know final shape/header.")
            return
        image, header = load_fits(input_file_name)

        # Find processed tiles (the ones produced by process_tile)
        processed_suffix = "_binned_gamma_corrected_drs.fits"
        processed_tiles = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(processed_suffix)])
        if not processed_tiles:
            print("No processed tiles found with suffix", processed_suffix, "in", output_dir)
            return

        # Pick brightest processed tile as reference
        ref_tile, ref_val = pick_brightest_tile(processed_tiles, method="p99")
        if ref_tile is None:
            print("Could not determine a reference tile for histogram matching.")
            return
        print("Reference (brightest) tile for histogram matching:", ref_tile, "metric:", ref_val)

        # Load reference data
        with fits.open(ref_tile) as hdul:
            ref_data = hdul[0].data.astype(np.float64)

        # Create matched tile files (suffix _matched.fits)
        matched_tiles = []
        for t in processed_tiles:
            try:
                with fits.open(t) as hdul:
                    data = hdul[0].data.astype(np.float64)
                    hdr = hdul[0].header
                matched = histogram_match(data, ref_data, n_bins=4096)  # 4096 bins is usually enough and faster
                outname = t.replace(processed_suffix, processed_suffix.replace(".fits", "_matched.fits"))
                fits.writeto(outname, matched.astype(np.float32), hdr, overwrite=True)
                matched_tiles.append(outname)
                print("Wrote matched tile:", outname)
            except Exception as e:
                print(f"Failed histogram matching for {t}: {e}")

        if not matched_tiles:
            print("No matched tiles produced, aborting reassembly.")
            return

        # Reassemble using matched tiles
        final_image = reassemble_image(matched_tiles, image.shape)
        filename = "output_" + os.path.basename(output_dir) + ".fits"
        fits.writeto(filename, final_image.astype(np.float32), header, overwrite=True)

        # Optional cleanup
        cleanup = input("Remove matched tiles? y/N --> ").strip().lower()
        if cleanup == 'y':
            for tile_file in matched_tiles:
                try:
                    os.remove(tile_file)
                except Exception:
                    pass
        print("Processing complete. Final image saved as", filename)

    else:
        print("Invalid option entered.")

if __name__ == "__main__":
    main()