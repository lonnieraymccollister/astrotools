#!/usr/bin/env python3
"""
apConsoleMakesDredImg.py
Reproject A_lambda magnitude maps to the stacked image WCS (2D),
compute multiplicative correction maps C = 10^(0.4*A),
apply per-channel multiplication to the linear stacked image,
and write outputs and diagnostics.

Edit filenames and channel_order below before running.
"""
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
# reproject_from_healpix may be available if reproject was installed with healpix support
try:
    from reproject import reproject_from_healpix
    _HAS_REPROJECT_HEALPIX = True
except Exception:
    reproject_from_healpix = None
    _HAS_REPROJECT_HEALPIX = False

import numpy as np
import sys
from copy import deepcopy

# -------------------------
# User-editable filenames
# -------------------------
#rescale ebv_scattered_map.A_B to the ebv_scattered_map.A_B plate sizes and copy stacked_plate_solved.fits to each using siril maxim Dl etc.
stacked_fn = "stacked_plate_solved.fits"   # linear, plate-solved stacked image (H,W,3) or (3,H,W)
A_b_fn = "ebv_scattered_map.A_B_2072_1411_out_with_old_header.fits"
A_g_fn = "ebv_scattered_map.A_G_2072_1411_out_with_old_header.fits"
A_r_fn = "ebv_scattered_map.A_R_2072_1411_out_with_old_header.fits"

out_dereddened = "stacked_dereddened_safe.fits"
out_Cb = "C_B.fits"
out_Cg = "C_G.fits"
out_Cr = "C_R.fits"
out_Ab_reproj = "A_B_reproj.fits"
out_Ag_reproj = "A_G_reproj.fits"
out_Ar_reproj = "A_R_reproj.fits"

# Set channel_order to "RGB" if your stacked image channels are R,G,B in that order.
# Set to "BGR" if channels are B,G,R. If None, the script will try to auto-detect common layouts.
channel_order = None  # "RGB", "BGR", or None to auto-detect

# -------------------------
# Helper functions
# -------------------------
def load_stack_and_header(fn):
    data = fits.getdata(fn)
    hdr = fits.getheader(fn)
    # If channel-first (3,H,W), move to channel-last (H,W,3)
    if data.ndim == 3 and data.shape[0] == 3:
        data = np.moveaxis(data, 0, 2)
    if data.ndim != 3 or data.shape[2] != 3:
        raise ValueError(f"Stacked image must be 3-channel RGB. Got shape {data.shape}")
    return data.astype(float), hdr

def make_2d_target_header(hdr, shape_out):
    """
    Build a 2D target header for reprojection.
    Uses WCS(hdr, naxis=2) to reduce any 3D core WCS + SIP/distortion to 2D.
    """
    w2 = WCS(hdr, naxis=2)
    target_hdr = w2.to_header()
    target_hdr['NAXIS'] = 2
    target_hdr['NAXIS1'] = shape_out[1]
    target_hdr['NAXIS2'] = shape_out[0]
    return target_hdr

def is_healpix_map(hdr):
    """Detect HEALPix-like headers (simple heuristic)."""
    keys = {k.upper(): hdr.get(k) for k in hdr.keys()}
    if isinstance(keys.get('PIXTYPE'), str) and keys.get('PIXTYPE').upper() == 'HEALPIX':
        return True
    if 'NSIDE' in keys or 'ORDERING' in keys:
        return True
    return False

def reproject_mag_to_target(mag_fn, target_hdr, shape_out):
    """
    Robust reprojection of a magnitude map to target_hdr (2D).
    Tries:
      1) reproject_interp if input has celestial WCS
      2) reproject_from_healpix if input is HEALPix and reproject has healpix support
      3) if input shape == shape_out, assume already on same pixel grid and return it
      4) otherwise raise a clear error with guidance
    """
    mag = fits.getdata(mag_fn)
    mag_hdr = fits.getheader(mag_fn)

    # Quick shape match: if shapes already match, return as-is (but warn)
    if mag.ndim == 2 and mag.shape == tuple(shape_out):
        print(f"Warning: {mag_fn} already matches target shape {shape_out}; using it directly.")
        return mag

    # Try to build a 2D WCS for the input map and check for celestial components
    has_cel = False
    try:
        w_in = WCS(mag_hdr, naxis=2)
        # If CTYPE keywords exist in header, assume celestial WCS present
        has_cel = any(k.startswith('CTYPE') for k in mag_hdr.keys())
    except Exception:
        has_cel = False

    # Case 1: input has celestial WCS -> use reproject_interp
    if has_cel:
        try:
            mag_reproj, footprint = reproject_interp((mag, mag_hdr), target_hdr, shape_out=shape_out)
            return mag_reproj
        except Exception as e:
            print(f"reproject_interp failed for {mag_fn}: {e}")

    # Case 2: HEALPix input -> try reproject_from_healpix
    try:
        if is_healpix_map(mag_hdr):
            if not _HAS_REPROJECT_HEALPIX:
                raise RuntimeError(
                    f"{mag_fn} appears to be HEALPix but reproject was not installed with HEALPix support.\n"
                    "Install with: pip install 'reproject[healpix]' and astropy-healpix."
                )
            print(f"Detected HEALPix map in {mag_fn}; attempting reproject_from_healpix...")
            mag_reproj, footprint = reproject_from_healpix(mag, mag_hdr, target_hdr, shape_out=shape_out)
            return mag_reproj
    except Exception as e:
        print(f"reproject_from_healpix failed for {mag_fn}: {e}")

    # Case 3: If input has no WCS but is same pixel scale/shape, use it (already handled above).
    # Otherwise, we cannot safely reproject: give a clear error with guidance.
    raise RuntimeError(
        f"Cannot reproject {mag_fn}: input FITS does not have a usable celestial WCS "
        f"and is not a HEALPix map or matching pixel grid.\n"
        "Options:\n"
        "  - Provide A_lambda FITS with a valid 2D celestial WCS (CTYPE/CRVAL/CRPIX/CD or PC keywords).\n"
        "  - If your map is HEALPix (Planck/Schlegel/Green), ensure the header contains PIXTYPE='HEALPIX' or NSIDE and install reproject with HEALPix support.\n"
        "  - If the map is already on the same pixel grid as the stacked image, save it with the same shape and a matching WCS header.\n"
        "  - Reproject the map to the stacked image WCS using an external tool (e.g., healpy, astropy_healpix) and supply the reprojected FITS."
    )

def make_C_from_A(A):
    C = np.where(np.isfinite(A), 10.0 ** (0.4 * A), 1.0)
    C[~np.isfinite(C)] = 1.0
    C[C <= 0] = 1.0
    return C

def detect_channel_order(img, hdr):
    """Try to detect channel order. Returns 'RGB' or 'BGR' or None."""
    hdr_keys = " ".join([k for k in hdr.keys() if isinstance(k, str)])
    if any(k in hdr_keys.upper() for k in ("BAYER", "BAYERPAT", "FILTER")):
        return None
    med = np.nanmedian(img.reshape(-1,3), axis=0)
    if med[0] >= med[2]:
        return "RGB"
    else:
        return "BGR"

# -------------------------
# Main processing
# -------------------------
if __name__ == "__main__":
    try:
        print("Loading stacked image:", stacked_fn)
        img, hdr = load_stack_and_header(stacked_fn)
        H, W, _ = img.shape
        print(f"Stack shape (H,W,3): {img.shape}")

        # determine channel order
        ch_order = channel_order
        if ch_order is None:
            print("Attempting to auto-detect channel order...")
            ch_order = detect_channel_order(img, hdr)
            if ch_order is None:
                print("Could not confidently auto-detect channel order. Defaulting to 'RGB'.")
                ch_order = "RGB"
            else:
                print("Auto-detected channel order:", ch_order)
        else:
            print("Using user-specified channel order:", ch_order)
        if ch_order not in ("RGB", "BGR"):
            raise ValueError("channel_order must be 'RGB' or 'BGR' or None")

        print("Building 2D target header from stacked image WCS...")
        target_hdr = make_2d_target_header(hdr, (H, W))

        # Reproject magnitude maps
        print("Reprojecting A_B...")
        Ab_reproj = reproject_mag_to_target(A_b_fn, target_hdr, (H, W))
        print("Reprojecting A_G...")
        Ag_reproj = reproject_mag_to_target(A_g_fn, target_hdr, (H, W))
        print("Reprojecting A_R...")
        Ar_reproj = reproject_mag_to_target(A_r_fn, target_hdr, (H, W))

        # Save reprojected magnitude maps for inspection
        fits.writeto(out_Ab_reproj, Ab_reproj.astype('float32'), target_hdr, overwrite=True)
        fits.writeto(out_Ag_reproj, Ag_reproj.astype('float32'), target_hdr, overwrite=True)
        fits.writeto(out_Ar_reproj, Ar_reproj.astype('float32'), target_hdr, overwrite=True)
        print("Wrote reprojected magnitude maps:", out_Ab_reproj, out_Ag_reproj, out_Ar_reproj)

        # Convert to multiplicative maps
        print("Converting magnitudes to multiplicative maps C = 10^(0.4*A)...")
        C_b = make_C_from_A(Ab_reproj)
        C_g = make_C_from_A(Ag_reproj)
        C_r = make_C_from_A(Ar_reproj)

        # Write C maps
        fits.writeto(out_Cb, C_b.astype('float32'), target_hdr, overwrite=True)
        fits.writeto(out_Cg, C_g.astype('float32'), target_hdr, overwrite=True)
        fits.writeto(out_Cr, C_r.astype('float32'), target_hdr, overwrite=True)
        print("Wrote multiplicative maps:", out_Cb, out_Cg, out_Cr)

        # Apply per-channel multiplication
        print("Applying per-channel multiplication...")
        if ch_order == "RGB":
            Rcorr = img[:, :, 0] * C_r
            Gcorr = img[:, :, 1] * C_g
            Bcorr = img[:, :, 2] * C_b
        else:  # BGR
            Bcorr = img[:, :, 0] * C_b
            Gcorr = img[:, :, 1] * C_g
            Rcorr = img[:, :, 2] * C_r

        out = np.stack([Rcorr, Gcorr, Bcorr], axis=2)

        # Write dereddened stacked image (channel-last)
        out_hdr = deepcopy(target_hdr)
        out_hdr['HISTORY'] = "Dereddened by apConsoleMakesDredImg.py"
        # If you prefer a 3D header for downstream tools, you can reconstruct a 3-plane header separately.
        fits.writeto(out_dereddened, out.astype('float32'), out_hdr, overwrite=True)
        print("Wrote dereddened stacked image:", out_dereddened)

        # Diagnostics
        print("Diagnostics:")
        print("  median(C_B, C_G, C_R):", np.nanmedian(C_b), np.nanmedian(C_g), np.nanmedian(C_r))
        print("  min/max(C_B):", np.nanmin(C_b), np.nanmax(C_b))
        print("  min/max(C_G):", np.nanmin(C_g), np.nanmax(C_g))
        print("  min/max(C_R):", np.nanmin(C_r), np.nanmax(C_r))

        # sample center pixel ratios
        y, x = H // 2, W // 2
        orig_px = img[y, x, :]
        corr_px = out[y, x, :]
        ratio = np.where(orig_px != 0, corr_px / orig_px, np.nan)
        print(f" sample pixel at center (y,x)=({y},{x}) orig -> corr -> ratio: {orig_px} -> {corr_px} -> {ratio}")

        print("Done. Inspect the reprojected A_*.fits and C_*.fits before stretching or running AI tools.")
    except Exception as e:
        print("ERROR:", str(e))
        sys.exit(1)