#!/usr/bin/env python3
"""
stacker_gui.py

Dust-map–oriented FITS stacker with:
 - Optional reprojection to a mosaic WCS using find_optimal_celestial_wcs (oldest API)
 - Fallback to first-frame WCS if mosaicking is unavailable
 - Per-frame background normalization (median subtraction, S/N-safe)
 - Optionally add synthetic background noise back into the stacked image
 - Optionally match synthetic noise to the weight map (physically correct)
 - Flux-linear weighted combine (no sqrt scaling)
 - Primary HDU = stacked image, WEIGHTS extension = per-pixel counts
 - Optional masking of low-coverage pixels (weight < 2)
 - Optional writing of extended plate-solve metadata into output header
 - Header normalization for Siril/MaxIm compatibility
 - Chunked, RAM-safe processing (tile-based reprojection + stacking)
"""

import sys
import os
import glob
import traceback
import math
import numpy as np

from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clip
from astropy.wcs.utils import proj_plane_pixel_scales

# --- Reproject imports with safe fallback ---
from reproject import reproject_interp
try:
    from reproject.mosaicking import find_optimal_celestial_wcs
except Exception:
    find_optimal_celestial_wcs = None

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QFileDialog, QGridLayout, QComboBox, QCheckBox, QTextEdit, QMessageBox,
    QDoubleSpinBox, QProgressBar, QSpinBox
)
from PyQt6.QtCore import QThread, pyqtSignal

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# -------------------------
# FITS helpers
# -------------------------
def load_fits(path):
    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            if hdu.data is not None:
                return np.array(hdu.data), hdu.header
    raise ValueError(f"No image data in {path}")

def sanitize_header_for_float32(header):
    hdr = header.copy()
    hdr["BITPIX"] = -32
    for k in ("BZERO", "BSCALE"):
        if k in hdr:
            hdr.pop(k)
    return hdr

def normalize_fits_shape(arr):
    a = np.asarray(arr)
    if a.ndim == 2:
        return a
    if a.ndim == 3:
        if a.shape[2] == 1:
            return a[:, :, 0]
        if a.shape[2] == 3:
            return np.transpose(a, (2, 0, 1))
        if a.shape[0] == 1:
            return a[0]
        if a.shape[0] == 3:
            return a
    s = np.squeeze(a)
    if s.ndim == 2:
        return s
    if s.ndim == 3 and s.shape[0] in (1, 3):
        return s[0] if s.shape[0] == 1 else s
    raise ValueError(f"Unsupported FITS shape: {a.shape}")

def to_luma(arr):
    a = np.asarray(arr)
    if a.ndim == 2:
        return a.astype(np.float64)
    if a.ndim == 3 and a.shape[0] == 3:
        return np.mean(a.astype(np.float64), axis=0)
    raise ValueError(f"Unsupported shape for luma: {a.shape}")

# -------------------------
# Stacking helpers
# -------------------------
def compute_stack_mean(cube):
    return np.nanmean(cube, axis=0)

def compute_stack_median(cube):
    return np.nanmedian(cube, axis=0)

def compute_stack_sigma_clip(cube, sigma=3.0, maxiters=5):
    clipped = sigma_clip(cube, sigma=sigma, maxiters=maxiters, axis=0, masked=True)
    arr = clipped.filled(np.nan)
    return np.nanmean(arr, axis=0)

def weighted_combine_from_cube(cube, mask_threshold=0.0):
    if cube.ndim != 3:
        raise ValueError("weighted_combine_from_cube expects (N, Y, X)")
    finite_mask = np.isfinite(cube)
    contrib_mask = finite_mask & (cube > mask_threshold)
    weights = contrib_mask.sum(axis=0).astype(np.float64)
    sum_vals = np.where(contrib_mask, cube, 0.0).sum(axis=0)
    valid = weights > 0
    avg = np.full_like(sum_vals, np.nan, dtype=np.float64)
    avg[valid] = sum_vals[valid] / weights[valid]
    return avg, weights

# -------------------------
# Header normalization helpers (Siril-friendly)
# -------------------------
def ensure_cd_or_pc(hdr):
    if 'CD1_1' in hdr or 'PC1_1' in hdr:
        return hdr
    if 'CDELT1' in hdr and 'CDELT2' in hdr:
        try:
            cd1 = float(hdr['CDELT1'])
            cd2 = float(hdr['CDELT2'])
            hdr['CD1_1'] = cd1
            hdr['CD1_2'] = 0.0
            hdr['CD2_1'] = 0.0
            hdr['CD2_2'] = cd2
        except Exception:
            pass
    return hdr

def clean_radesys(hdr):
    if 'RADESYS' in hdr:
        try:
            hdr['RADESYS'] = str(hdr['RADESYS']).strip()
        except Exception:
            hdr['RADESYS'] = 'ICRS'
    else:
        hdr['RADESYS'] = 'ICRS'
    if 'RADECSYS' not in hdr:
        hdr['RADECSYS'] = hdr['RADESYS']
    else:
        try:
            hdr['RADECSYS'] = str(hdr['RADECSYS']).strip()
        except Exception:
            hdr['RADECSYS'] = hdr['RADESYS']
    return hdr

def ensure_ctypes_upper(hdr):
    for k in ('CTYPE1', 'CTYPE2'):
        if k in hdr:
            try:
                hdr[k] = str(hdr[k]).upper()
            except Exception:
                pass
    return hdr

def add_plate_solve_metadata(header_out, mosaic_wcs, pixel_size_um=None):
    try:
        header_out['WCSAXES'] = 2
        header_out['RADESYS'] = 'ICRS'
        header_out['EQUINOX'] = 2000.0
        try:
            scales = proj_plane_pixel_scales(mosaic_wcs) * 3600.0
            pixscale = float(np.mean(scales))
            header_out['PIXSCALE'] = (pixscale, 'arcsec/pixel')
        except Exception:
            pass
        cd11 = header_out.get('CD1_1')
        cd12 = header_out.get('CD1_2')
        cd21 = header_out.get('CD2_1')
        cd22 = header_out.get('CD2_2')
        if cd11 is not None and cd12 is not None and cd21 is not None and cd22 is not None:
            try:
                orient = np.degrees(np.arctan2(-float(cd12), float(cd22)))
                header_out['ORIENTAT'] = (orient, 'Orientation angle (deg)')
            except Exception:
                pass
        if 'CRVAL1' in header_out and 'CRVAL2' in header_out:
            header_out['OBJCTRA'] = (header_out['CRVAL1'], 'Field center RA')
            header_out['OBJCTDEC'] = (header_out['CRVAL2'], 'Field center Dec')
        if pixel_size_um is not None:
            try:
                pixscale = float(header_out.get('PIXSCALE', np.nan))
                if np.isfinite(pixscale) and pixscale != 0.0:
                    focal_mm = 206.265 * float(pixel_size_um) / pixcale
                    header_out['FOCALLEN'] = (float(focal_mm), 'Derived focal length (mm)')
                    header_out['PIXSIZE'] = (float(pixel_size_um), 'Pixel size (um)')
            except Exception:
                pass
    except Exception:
        pass

def normalize_header_for_siril(header_out, mosaic_wcs=None, pixel_size_um=None):
    hdr = header_out.copy()
    hdr = ensure_ctypes_upper(hdr)
    hdr = clean_radesys(hdr)
    hdr = ensure_cd_or_pc(hdr)
    if mosaic_wcs is not None:
        try:
            wcs_hdr = mosaic_wcs.to_header()
            for k, v in wcs_hdr.items():
                hdr[k] = v
        except Exception:
            pass
    try:
        add_plate_solve_metadata(hdr, mosaic_wcs if mosaic_wcs is not None else WCS(hdr), pixel_size_um=pixel_size_um)
    except Exception:
        pass
    for k in list(hdr.keys()):
        try:
            if isinstance(hdr[k], str):
                hdr[k] = hdr[k].strip()
        except Exception:
            pass
    return hdr

# -------------------------
# Preview canvas
# -------------------------
class PreviewCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure(figsize=(6, 4))
        super().__init__(fig)
        self.ax_img = fig.add_subplot(1, 2, 1)
        self.ax_hist = fig.add_subplot(1, 2, 2)
        fig.tight_layout()
        self._cbar = None

    def plot(self, image, header=None, hist_bins=256):
        self.ax_img.clear()
        self.ax_hist.clear()

        if image is None:
            self.ax_img.set_title("No image")
        else:
            try:
                if header is not None:
                    wcs = WCS(header)
                    self.figure.delaxes(self.ax_img)
                    self.ax_img = self.figure.add_subplot(1, 2, 1, projection=wcs)
                else:
                    self.figure.delaxes(self.ax_img)
                    self.ax_img = self.figure.add_subplot(1, 2, 1)
            except Exception:
                try:
                    self.figure.delaxes(self.ax_img)
                except Exception:
                    pass
                self.ax_img = self.figure.add_subplot(1, 2, 1)

            im = self.ax_img.imshow(image, origin="lower", cmap="gray")
            if self._cbar:
                try:
                    self._cbar.remove()
                except Exception:
                    pass
            self._cbar = self.figure.colorbar(im, ax=self.ax_img, orientation="vertical")
            self.ax_img.set_title("Stacked image")

        if image is not None:
            clean = image[np.isfinite(image)]
            if clean.size:
                self.ax_hist.hist(clean.ravel(), bins=hist_bins, color="red", histtype="step")
                self.ax_hist.set_title("Histogram (stacked)")
        self.draw()

# -------------------------
# Utility: create tile WCS from mosaic WCS
# -------------------------
def tile_wcs_from_mosaic(mosaic_wcs, tile_x0, tile_y0):
    """
    Create a WCS for a tile whose origin is (tile_x0, tile_y0) in mosaic pixel coordinates.
    This adjusts CRPIX so that reproject_interp will produce an array aligned to the tile.
    """
    hdr = mosaic_wcs.to_header()
    # CRPIX in header is 1-based; adjust by tile origin
    try:
        crpix1 = float(hdr.get('CRPIX1', 0.0))
        crpix2 = float(hdr.get('CRPIX2', 0.0))
        hdr['CRPIX1'] = crpix1 - float(tile_x0)
        hdr['CRPIX2'] = crpix2 - float(tile_y0)
    except Exception:
        pass
    return WCS(hdr)

# -------------------------
# Worker thread (chunked)
# -------------------------
class StackWorker(QThread):
    log = pyqtSignal(str)
    progress = pyqtSignal(int)
    finished = pyqtSignal(bool, str, object, object)

    def __init__(self, files, reproject_flag, method, sigma,
                 weighted_combine_flag, mask_threshold,
                 mask_lowcov_flag, lowcov_threshold,
                 outpath, overwrite,
                 normalize_background, write_full_plate=False, pixel_size_um=None,
                 add_synthetic_bg=False, match_noise_to_weight=False,
                 chunked=False, tile_size=512):
        super().__init__()
        self.files = files
        self.reproject_flag = reproject_flag
        self.method = method
        self.sigma = sigma
        self.weighted_combine_flag = weighted_combine_flag
        self.mask_threshold = float(mask_threshold)
        self.mask_lowcov_flag = mask_lowcov_flag
        self.lowcov_threshold = int(lowcov_threshold)
        self.outpath = outpath
        self.overwrite = overwrite
        self.normalize_background = normalize_background
        self.write_full_plate = write_full_plate
        self.pixel_size_um = pixel_size_um
        self.add_synthetic_bg = add_synthetic_bg
        self.match_noise_to_weight = match_noise_to_weight
        self.chunked = chunked
        self.tile_size = int(tile_size)
        self._abort = False

        # For estimating background noise to re-inject
        self._per_frame_bg_stds = []

    def abort(self):
        self._abort = True

    def _emit(self, *args):
        self.log.emit(" ".join(str(a) for a in args))

    def _create_empty_output(self, header_out, shape_out):
        """
        Create an output FITS file with PrimaryHDU zeros and WEIGHTS extension zeros.
        Overwrite behavior controlled by self.overwrite.
        """
        ny, nx = shape_out
        hdr_out = sanitize_header_for_float32(header_out) if header_out is not None else fits.Header()
        hdr_out = normalize_header_for_siril(hdr_out, mosaic_wcs=(WCS(header_out) if header_out is not None else None), pixel_size_um=self.pixel_size_um)
        primary = fits.PrimaryHDU(np.zeros((ny, nx), dtype=np.float32), header=hdr_out)
        hdu_weights = fits.ImageHDU(np.zeros((ny, nx), dtype=np.float32), name="WEIGHTS")
        hdul = fits.HDUList([primary, hdu_weights])
        hdul.writeto(self.outpath, overwrite=self.overwrite)
        hdul.close()

    def _update_tile_in_output(self, tile_y0, tile_x0, tile_arr, tile_weights):
        """
        Write tile_arr and tile_weights into the output FITS at the given origin.
        This function opens the output in update mode and writes the slice.
        """
        with fits.open(self.outpath, mode='update', memmap=True) as hdul:
            data = hdul[0].data
            weights = hdul['WEIGHTS'].data
            y0, y1 = tile_y0, tile_y0 + tile_arr.shape[0]
            x0, x1 = tile_x0, tile_x0 + tile_arr.shape[1]
            # accumulate: existing sum stored in data, existing counts in weights
            # We store sums in primary and counts in WEIGHTS (consistent with weighted combine)
            # tile_arr contains sum contributions for this tile; tile_weights contains counts
            # Use float32 storage but compute in float64
            data_slice = data[y0:y1, x0:x1].astype(np.float64)
            weights_slice = weights[y0:y1, x0:x1].astype(np.float64)
            # accumulate
            data_slice += tile_arr.astype(np.float64)
            weights_slice += tile_weights.astype(np.float64)
            # write back
            data[y0:y1, x0:x1] = data_slice.astype(np.float32)
            weights[y0:y1, x0:x1] = weights_slice.astype(np.float32)
            hdul.flush()

    def run(self):
        try:
            N = len(self.files)
            self._emit(f"Processing {N} files...")

            frames_meta = []
            self._per_frame_bg_stds = []

            # First pass: read headers, estimate per-frame bg std, and prepare frames metadata
            for i, fn in enumerate(self.files, start=1):
                if self._abort:
                    self._emit("Aborted by user.")
                    self.finished.emit(False, "Aborted", None, None)
                    return
                self._emit(f"[meta {i}/{N}] Reading header: {os.path.basename(fn)}")
                try:
                    data, hdr = load_fits(fn)
                    data = normalize_fits_shape(data)
                    if data.ndim == 3 and data.shape[0] == 3:
                        data_for_wcs = np.mean(data.astype(np.float64), axis=0)
                    else:
                        data_for_wcs = data.astype(np.float64)
                    # Estimate per-frame background std BEFORE subtraction
                    try:
                        clipped = sigma_clip(data_for_wcs, sigma=3.0, maxiters=3)
                        bg_std = float(np.nanstd(clipped.filled(np.nan)))
                        if not np.isfinite(bg_std) or bg_std <= 0:
                            bg_std = float(np.nanstd(data_for_wcs[np.isfinite(data_for_wcs)]))
                    except Exception:
                        bg_std = float(np.nanstd(data_for_wcs[np.isfinite(data_for_wcs)]))
                    if np.isfinite(bg_std) and bg_std > 0:
                        self._per_frame_bg_stds.append(bg_std)
                    frames_meta.append((fn, hdr))
                except Exception as e:
                    self._emit(f"Failed to read header {fn}: {e}; skipping")
                self.progress.emit(int(100.0 * i / max(1, N)))

            if not frames_meta:
                raise RuntimeError("No valid input files loaded.")

            # Determine mosaic WCS and output shape
            mosaic_wcs = None
            header_out = None
            shape_out = None

            if self.reproject_flag:
                if find_optimal_celestial_wcs is None:
                    self._emit("find_optimal_celestial_wcs not available; using first-frame WCS.")
                    fn0, hdr0 = frames_meta[0]
                    data0, _ = load_fits(fn0)
                    data0 = normalize_fits_shape(data0)
                    if data0.ndim == 3 and data0.shape[0] == 3:
                        data0 = np.mean(data0.astype(np.float64), axis=0)
                    else:
                        data0 = data0.astype(np.float64)
                    mosaic_wcs = WCS(hdr0)
                    ny, nx = data0.shape
                    shape_out = (ny, nx)
                    header_out = hdr0
                else:
                    self._emit("Computing optimal mosaic WCS for all inputs...")
                    pairs = []
                    for fn, hdr in frames_meta:
                        data, _ = load_fits(fn)
                        data = normalize_fits_shape(data)
                        if data.ndim == 3 and data.shape[0] == 3:
                            data_for_wcs = np.mean(data.astype(np.float64), axis=0)
                        else:
                            data_for_wcs = data.astype(np.float64)
                        pairs.append((data_for_wcs, WCS(hdr)))
                    mosaic_wcs, shape_out = find_optimal_celestial_wcs(pairs)
                    header_out = mosaic_wcs.to_header()
                ny, nx = shape_out
            else:
                # No reprojection: ensure all shapes equal and use first header
                shapes = set()
                for fn, hdr in frames_meta:
                    data, _ = load_fits(fn)
                    data = normalize_fits_shape(data)
                    if data.ndim == 3 and data.shape[0] == 3:
                        data = np.mean(data.astype(np.float64), axis=0)
                    else:
                        data = data.astype(np.float64)
                    shapes.add(data.shape)
                if len(shapes) != 1:
                    raise RuntimeError("Input FITS files have differing shapes. Enable reprojection or make shapes equal.")
                ny, nx = next(iter(shapes))
                shape_out = (ny, nx)
                header_out = frames_meta[0][1]
                mosaic_wcs = WCS(header_out)

            self._emit(f"Output mosaic shape: {shape_out}")

            # Create empty output FITS (Primary = sum, WEIGHTS = counts)
            if os.path.exists(self.outpath) and not self.overwrite:
                raise FileExistsError(f"Output exists and overwrite disabled: {self.outpath}")
            self._create_empty_output(header_out, shape_out)
            self._emit(f"Created output file: {self.outpath}")

            # Tile loop
            tile_h = min(self.tile_size, shape_out[0])
            tile_w = min(self.tile_size, shape_out[1])
            tiles = []
            for y0 in range(0, shape_out[0], tile_h):
                for x0 in range(0, shape_out[1], tile_w):
                    y1 = min(shape_out[0], y0 + tile_h)
                    x1 = min(shape_out[1], x0 + tile_w)
                    tiles.append((y0, x0, y1 - y0, x1 - x0))
            total_tiles = len(tiles)
            self._emit(f"Processing in {total_tiles} tiles of up to {tile_h}x{tile_w} pixels")

            # Precompute stacked_noise_std global estimate (used when adding synthetic noise)
            if len(self._per_frame_bg_stds) >= 1:
                sigma_med = float(np.median(self._per_frame_bg_stds))
                nframes = max(1, len(self._per_frame_bg_stds))
                global_stacked_noise_std = sigma_med / math.sqrt(nframes)
            else:
                global_stacked_noise_std = None

            tile_index = 0
            for (tile_y0, tile_x0, th, tw) in tiles:
                tile_index += 1
                if self._abort:
                    self._emit("Aborted by user.")
                    self.finished.emit(False, "Aborted", None, None)
                    return
                self._emit(f"[tile {tile_index}/{total_tiles}] origin=({tile_x0},{tile_y0}) size=({tw},{th})")

                # Accumulators for this tile: sum and counts
                tile_sum = np.zeros((th, tw), dtype=np.float64)
                tile_counts = np.zeros((th, tw), dtype=np.float64)

                # Build tile WCS (adjust CRPIX so reproject_interp maps to tile coordinates)
                tile_wcs = tile_wcs_from_mosaic(mosaic_wcs, tile_x0, tile_y0)

                # For each frame, reproject only to this tile and accumulate
                for i, (fn, hdr) in enumerate(frames_meta, start=1):
                    if self._abort:
                        self._emit("Aborted by user.")
                        self.finished.emit(False, "Aborted", None, None)
                        return
                    self._emit(f"[tile {tile_index}] Reprojecting frame {i}/{N}: {os.path.basename(fn)}")
                    try:
                        data, hdr_full = load_fits(fn)
                        data = normalize_fits_shape(data)
                        if data.ndim == 3 and data.shape[0] == 3:
                            data = np.mean(data.astype(np.float64), axis=0)
                        else:
                            data = data.astype(np.float64)

                        # Per-frame background normalization (S/N-safe)
                        if self.normalize_background:
                            try:
                                bg = float(np.nanmedian(sigma_clip(data, sigma=3.0)))
                            except Exception:
                                bg = float(np.nanmedian(data[np.isfinite(data)]))
                            data_proc = data - bg
                        else:
                            data_proc = data

                        # Reproject to tile using tile_wcs and shape_out=(th, tw)
                        try:
                            reprojected, footprint = reproject_interp((data_proc, WCS(hdr_full)), tile_wcs, shape_out=(th, tw), return_footprint=True)
                        except Exception as e:
                            self._emit(f"Reprojection failed for {fn} on tile: {e}; skipping frame for this tile")
                            continue

                        reprojected = np.asarray(reprojected, dtype=np.float64)
                        # footprint indicates where the frame contributed; treat footprint>0 as 1 contribution
                        footprint = np.asarray(footprint, dtype=np.float64)
                        footprint_mask = (footprint > 0.0).astype(np.float64)

                        # Apply mask threshold if requested (treat values <= mask_threshold as no contribution)
                        if self.mask_threshold != 0.0:
                            contrib_mask = (reprojected > self.mask_threshold).astype(np.float64)
                            footprint_mask *= contrib_mask

                        # Accumulate sum and counts
                        # For weighted combine we treat each contributing frame as weight=1 (flux-linear average)
                        tile_sum += np.where(np.isfinite(reprojected), reprojected * footprint_mask, 0.0)
                        tile_counts += footprint_mask

                    except Exception as e:
                        self._emit(f"Failed processing frame {fn} for tile: {e}; skipping")
                    # update progress roughly per frame within tile
                    self.progress.emit(int(100.0 * ((tile_index - 1) + (i / max(1, N))) / total_tiles))

                # After all frames processed for this tile, compute final tile values
                valid = tile_counts > 0
                tile_final = np.full((th, tw), np.nan, dtype=np.float64)
                tile_final[valid] = tile_sum[valid] / tile_counts[valid]

                # Mask low coverage if requested
                if self.mask_lowcov_flag:
                    lowcov_mask = tile_counts < self.lowcov_threshold
                    tile_final[lowcov_mask] = np.nan

                # Optionally add synthetic background noise (per-tile)
                if self.add_synthetic_bg:
                    try:
                        # Determine stacked noise std for this tile: use global estimate if available,
                        # otherwise estimate from tile_final clipped
                        if global_stacked_noise_std is not None:
                            stacked_noise_std = global_stacked_noise_std
                        else:
                            clipped_tile = sigma_clip(tile_final, sigma=3.0, maxiters=3)
                            stacked_noise_std = float(np.nanstd(clipped_tile.filled(np.nan)))
                            if not np.isfinite(stacked_noise_std) or stacked_noise_std <= 0:
                                stacked_noise_std = 0.0
                        if stacked_noise_std > 0:
                            rng = np.random.default_rng()
                            noise = rng.normal(loc=0.0, scale=stacked_noise_std, size=tile_final.shape)

                            # If matching noise to weight map, scale noise by 1/sqrt(W)
                            if self.match_noise_to_weight:
                                try:
                                    W = np.where(np.isfinite(tile_counts) & (tile_counts > 0), tile_counts, 1.0)
                                    scale = 1.0 / np.sqrt(W)
                                    noise = noise * scale
                                    self._emit("Matched synthetic noise to weight map for tile.")
                                except Exception as e:
                                    self._emit(f"Failed to match noise to weight map for tile: {e}")

                            # Add noise only where tile_final is finite (preserve NaNs)
                            finite_mask = np.isfinite(tile_final)
                            tile_final[finite_mask] = tile_final[finite_mask] + noise[finite_mask]
                            self._emit(f"Added synthetic background noise to tile (sigma ~ {stacked_noise_std:.4g}).")
                        else:
                            self._emit("Synthetic background noise not added for tile: estimated stacked noise <= 0.")
                    except Exception as e:
                        self._emit(f"Failed to add synthetic background noise for tile: {e}")

                # Convert tile_final and tile_counts into sums and counts to store in output
                # Our output primary stores accumulated sums; WEIGHTS stores counts
                # tile_sum currently holds the accumulated sums; tile_counts holds counts
                # But we must write the current accumulators into the output file (accumulate with existing)
                # We'll write tile_sum and tile_counts (both float64) into output using update function
                try:
                    # tile_sum and tile_counts are the accumulators we computed earlier
                    # Write them into the output (accumulation handled in _update_tile_in_output)
                    self._update_tile_in_output(tile_y0, tile_x0, tile_sum, tile_counts)
                    self._emit(f"Wrote tile {tile_index}/{total_tiles} to output.")
                except Exception as e:
                    self._emit(f"Failed to write tile to output: {e}")

            # After all tiles processed, convert accumulated sums/counts into final average image
            # Open output and compute final = sum / counts where counts>0, else NaN
            with fits.open(self.outpath, mode='update', memmap=True) as hdul:
                data = hdul[0].data.astype(np.float64)
                weights = hdul['WEIGHTS'].data.astype(np.float64)
                final = np.full_like(data, np.nan, dtype=np.float64)
                valid = weights > 0
                final[valid] = data[valid] / weights[valid]

                # Optionally mask low coverage globally (already applied per tile, but reapply)
                if self.mask_lowcov_flag:
                    final[weights < self.lowcov_threshold] = np.nan

                # Save final into primary HDU (overwrite sums with averaged final)
                hdul[0].data[:, :] = final.astype(np.float32)
                # Update header normalization if needed
                if self.write_full_plate and mosaic_wcs is not None:
                    try:
                        wcs_hdr = mosaic_wcs.to_header()
                        for k, v in wcs_hdr.items():
                            hdul[0].header[k] = v
                    except Exception:
                        pass
                hdul.flush()

            self._emit(f"Completed chunked stacking and wrote final averaged image to {self.outpath}")
            self.finished.emit(True, f"Wrote {self.outpath}", final, header_out)
            return

        except Exception as e:
            tb = traceback.format_exc()
            self.log.emit(f"Error: {e}\n{tb}")
            self.finished.emit(False, str(e), None, None)

# -------------------------
# GUI
# -------------------------
class StackerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FITS Stacker (dust-map optimized)")
        self._build_ui()
        self.resize(1200, 820)
        self.stack_result = None
        self.stack_header = None
        self.worker = None

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        grid = QGridLayout()
        central.setLayout(grid)

        grid.addWidget(QLabel("Input FITS directory:"), 0, 0)
        self.input_dir_edit = QLineEdit()
        grid.addWidget(self.input_dir_edit, 0, 1, 1, 3)
        btn_dir = QPushButton("Browse...")
        btn_dir.clicked.connect(self._browse_dir)
        grid.addWidget(btn_dir, 0, 4)

        grid.addWidget(QLabel("Output FITS file:"), 1, 0)
        self.output_edit = QLineEdit("stacked_output.fits")
        grid.addWidget(self.output_edit, 1, 1, 1, 3)
        btn_out = QPushButton("Browse...")
        btn_out.clicked.connect(self._browse_save)
        grid.addWidget(btn_out, 1, 4)

        grid.addWidget(QLabel("Stack method:"), 2, 0)
        self.method_combo = QComboBox()
        self.method_combo.addItems(["mean", "median", "sigma-clipped mean"])
        grid.addWidget(self.method_combo, 2, 1)

        grid.addWidget(QLabel("Sigma (for sigma-clipped):"), 2, 2)
        self.sigma_edit = QLineEdit("3.0")
        grid.addWidget(self.sigma_edit, 2, 3)

        self.reproject_chk = QCheckBox("Reproject to mosaic WCS (if available)")
        self.reproject_chk.setChecked(True)
        grid.addWidget(self.reproject_chk, 3, 0, 1, 3)

        self.weighted_chk = QCheckBox("Weighted combine (per-pixel average, flux-linear)")
        grid.addWidget(self.weighted_chk, 4, 0, 1, 4)

        # Background normalization checkbox
        self.norm_bg_chk = QCheckBox("Normalize background per frame (median subtraction)")
        self.norm_bg_chk.setChecked(True)
        grid.addWidget(self.norm_bg_chk, 5, 0, 1, 4)

        grid.addWidget(QLabel("Mask threshold (for weighted combine):"), 6, 0)
        self.mask_spin = QDoubleSpinBox()
        self.mask_spin.setRange(-1e12, 1e12)
        self.mask_spin.setDecimals(6)
        self.mask_spin.setValue(0.0)
        grid.addWidget(self.mask_spin, 6, 1)

        self.lowcov_chk = QCheckBox("Mask low-coverage pixels (weight < 2)")
        self.lowcov_chk.setChecked(True)
        grid.addWidget(self.lowcov_chk, 6, 2, 1, 2)

        # Option to write full plate-solve metadata into output header
        self.plate_chk = QCheckBox("Write full plate-solve metadata into output header")
        self.plate_chk.setChecked(True)
        grid.addWidget(self.plate_chk, 7, 0, 1, 4)

        grid.addWidget(QLabel("Pixel size (um) for focal length derivation (optional):"), 8, 0)
        self.pixsize_edit = QLineEdit("")
        grid.addWidget(self.pixsize_edit, 8, 1, 1, 2)

        # Add synthetic background noise checkbox (default unchecked)
        self.add_bg_noise_chk = QCheckBox("Add synthetic background noise after stacking (for StarNet/SyQon)")
        self.add_bg_noise_chk.setChecked(False)
        grid.addWidget(self.add_bg_noise_chk, 9, 0, 1, 4)

        # Match synthetic noise to weight map checkbox
        self.match_noise_weight_chk = QCheckBox("Match synthetic noise to weight map (physically correct)")
        self.match_noise_weight_chk.setChecked(False)
        grid.addWidget(self.match_noise_weight_chk, 10, 0, 1, 4)

        # Chunked processing controls
        self.chunked_chk = QCheckBox("Enable chunked (RAM-safe) processing (Option A)")
        self.chunked_chk.setChecked(True)
        grid.addWidget(self.chunked_chk, 11, 0, 1, 3)

        grid.addWidget(QLabel("Tile size (px):"), 11, 3)
        self.tile_spin = QSpinBox()
        self.tile_spin.setRange(64, 4096)
        self.tile_spin.setValue(512)
        grid.addWidget(self.tile_spin, 11, 4)

        # Run / Cancel / Preview / Plot buttons
        self.run_btn = QPushButton("Run Stack")
        self.run_btn.clicked.connect(self._on_run)
        grid.addWidget(self.run_btn, 12, 0)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._on_cancel)
        self.cancel_btn.setEnabled(False)
        grid.addWidget(self.cancel_btn, 12, 1)

        self.preview_btn = QPushButton("Preview Last Result")
        self.preview_btn.clicked.connect(self._on_preview)
        grid.addWidget(self.preview_btn, 12, 2)

        self.load_output_btn = QPushButton("Plot Output File")
        self.load_output_btn.clicked.connect(self._plot_output_file)
        grid.addWidget(self.load_output_btn, 12, 3)

        # Progress bar and log
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        grid.addWidget(self.progress, 13, 0, 1, 5)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        grid.addWidget(self.log_box, 14, 0, 1, 5)

        # Preview canvas
        self.canvas = PreviewCanvas()
        grid.addWidget(self.canvas, 15, 0, 6, 5)

    def _log(self, *args):
        self.log_box.append(" ".join(str(a) for a in args))
        QApplication.processEvents()

    def _browse_dir(self):
        dn = QFileDialog.getExistingDirectory(self, "Select input directory", "")
        if dn:
            self.input_dir_edit.setText(dn)

    def _browse_save(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Select output FITS", "", "FITS Files (*.fits *.fit);;All Files (*)")
        if fn:
            self.output_edit.setText(fn)

    def _on_run(self):
        try:
            input_dir = self.input_dir_edit.text().strip()
            if not input_dir:
                raise ValueError("Select input directory")
            pattern = os.path.join(input_dir, "*.fit*")
            files = sorted(glob.glob(pattern))
            if not files:
                raise FileNotFoundError(f"No FITS files found in {input_dir}")
            self._log(f"Found {len(files)} FITS files")

            reproject_flag = self.reproject_chk.isChecked()
            weighted_flag = self.weighted_chk.isChecked()
            mask_lowcov_flag = self.lowcov_chk.isChecked()
            normalize_bg_flag = self.norm_bg_chk.isChecked()
            write_plate_flag = self.plate_chk.isChecked()
            add_bg_noise_flag = self.add_bg_noise_chk.isChecked()
            match_noise_weight_flag = self.match_noise_weight_chk.isChecked()
            chunked_flag = self.chunked_chk.isChecked()
            tile_size = int(self.tile_spin.value())

            method_text = self.method_combo.currentText()
            if method_text == "mean":
                method = "mean"
            elif method_text == "median":
                method = "median"
            else:
                method = "sigma"

            sigma = float(self.sigma_edit.text().strip() or 3.0)
            mask_thresh = float(self.mask_spin.value())
            lowcov_threshold = 2

            outpath = self.output_edit.text().strip() or "stacked_output.fits"
            overwrite = True

            pixel_size_um = None
            ps_text = self.pixsize_edit.text().strip()
            if ps_text:
                try:
                    pixel_size_um = float(ps_text)
                except Exception:
                    pixel_size_um = None

            self.run_btn.setEnabled(False)
            self.cancel_btn.setEnabled(True)
            self._log("Starting worker thread for reprojection/stacking...")
            self.progress.setValue(0)

            self.worker = StackWorker(
                files, reproject_flag, method, sigma,
                weighted_flag, mask_thresh,
                mask_lowcov_flag, lowcov_threshold,
                outpath, overwrite,
                normalize_bg_flag, write_full_plate=write_plate_flag, pixel_size_um=pixel_size_um,
                add_synthetic_bg=add_bg_noise_flag,
                match_noise_to_weight=match_noise_weight_flag,
                chunked=chunked_flag, tile_size=tile_size
            )
            self.worker.log.connect(self._log)
            self.worker.progress.connect(self.progress.setValue)
            self.worker.finished.connect(self._on_finished)
            self.worker.start()

        except Exception as e:
            tb = traceback.format_exc()
            QMessageBox.critical(self, "Error", f"{e}\n\n{tb}")
            self._log("Error:", e)
            self.run_btn.setEnabled(True)
            self.cancel_btn.setEnabled(False)

    def _on_cancel(self):
        if self.worker is not None:
            self.worker.abort()
            self._log("Abort requested; waiting for worker to stop...")
            self.cancel_btn.setEnabled(False)

    def _on_finished(self, success, message, stack_result, header):
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress.setValue(100 if success else 0)
        if success:
            QMessageBox.information(self, "Done", message)
            self._log("Worker finished:", message)
            self.stack_result = stack_result
            self.stack_header = header
            if stack_result is not None:
                self.canvas.plot(stack_result, header)
        else:
            QMessageBox.critical(self, "Failed", message)
            self._log("Worker failed:", message)

    def _on_preview(self):
        if self.stack_result is None:
            QMessageBox.information(self, "No result", "There is no stacked result to preview. Run the stack first.")
            return
        self.canvas.plot(self.stack_result, self.stack_header)
        st = compute_stats(self.stack_result)
        self._log(f"Preview stats: min={st['min']:.6g} max={st['max']:.6g} mean={st['mean']:.6g} std={st['std']:.6g}")

    def _plot_output_file(self):
        outpath = self.output_edit.text().strip() or "stacked_output.fits"
        if not os.path.exists(outpath):
            fn, _ = QFileDialog.getOpenFileName(self, "Select output FITS to plot", "", "FITS Files (*.fits *.fit);;All Files (*)")
            if not fn:
                return
            outpath = fn
        try:
            data, hdr = load_fits(outpath)
            if data is None:
                raise ValueError("No data in primary HDU")
            data = normalize_fits_shape(data)
            if data.ndim == 3 and data.shape[0] == 3:
                data2 = np.mean(data.astype(np.float64), axis=0)
            elif data.ndim == 3:
                data2 = data[0].astype(np.float64)
            else:
                data2 = data.astype(np.float64)
            self.canvas.plot(data2, hdr)
            st = compute_stats(data2)
            self._log(f"Plotted output {os.path.basename(outpath)} stats: min={st['min']:.6g} max={st['max']:.6g} mean={st['mean']:.6g} std={st['std']:.6g}")
        except Exception as e:
            QMessageBox.critical(self, "Plot error", str(e))
            self._log("Plot error:", e)

def compute_stats(arr):
    a = np.asarray(arr).ravel()
    a = a[~np.isnan(a)]
    if a.size == 0:
        return {"min": np.nan, "max": np.nan, "mean": np.nan, "std": np.nan}
    return {"min": float(a.min()), "max": float(a.max()), "mean": float(a.mean()), "std": float(a.std())}

def main():
    app = QApplication(sys.argv)
    w = StackerWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
