#!/usr/bin/env python3
"""
dynamic_rescale16.py
Standalone utility: split a large 2D FITS into tiles, process tiles (dynamic block rescale + gamma + bin),
and reassemble processed tiles back to the full image.

Usage: python dynamic_rescale16.py
Follow interactive prompts to choose mode 01 (split), 02 (process tiles), or 03 (reassemble).
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
# Main interactive flow
# -------------------------
def main():
    print("Enter 01 to split tiles, 02 to process tiles, or 03 to combine tiles")
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
        tiles = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith("_binned_gamma_corrected_drs.fits")])
        if not tiles:
            print("No processed tiles found with suffix '_binned_gamma_corrected_drs.fits' in", output_dir)
            return
        final_image = reassemble_image(tiles, image.shape)
        filename = "output_" + os.path.basename(output_dir) + ".fits"
        fits.writeto(filename, final_image.astype(np.float32), header, overwrite=True)
        # Optional cleanup
        cleanup = input("Remove processed tiles? y/N --> ").strip().lower()
        if cleanup == 'y':
            for tile_file in tiles:
                os.remove(tile_file)
        print("Processing complete. Final image saved as", filename)
    else:
        print("Invalid option entered.")

if __name__ == "__main__":
    main()