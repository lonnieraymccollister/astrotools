#!/usr/bin/env python3
"""
Debuggable RA/Dec filler for photometry CSVs.

Usage:
  python wcs_debug.py input_photometry.csv --out output_with_radec.csv --log debug.log
"""
import csv, os, argparse, traceback
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

def read_csv_rows(path):
    with open(path, newline="") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    return rows, fieldnames

def write_csv_rows(path, fieldnames, rows):
    with open(path, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

def compute_radec_from_fits(fits_path, x_pixel, y_pixel, log):
    """Try multiple HDUs and origins; return (ra,dec,info_str) or (None,None,reason)."""
    try:
        with fits.open(fits_path, memmap=False) as hdul:
            for idx, hdu in enumerate(hdul):
                hdr = getattr(hdu, "header", None)
                if hdr is None:
                    continue
                # build candidate WCS objects
                wcs_cands = []
                try:
                    wcs_cands.append(WCS(hdr, naxis=2))
                except Exception:
                    pass
                try:
                    w_full = WCS(hdr)
                    wcs_cands.append(w_full)
                    if getattr(w_full, "celestial", None) is not None:
                        wcs_cands.append(w_full.celestial)
                except Exception:
                    pass
                try:
                    wcs_cands.append(WCS(hdr, relax=True))
                except Exception:
                    pass

                for w in wcs_cands:
                    if w is None:
                        continue
                    try:
                        if not getattr(w, "has_celestial", False):
                            continue
                    except Exception:
                        continue
                    for origin in (0, 1):
                        try:
                            ra_arr, dec_arr = w.all_pix2world([x_pixel], [y_pixel], origin)
                            ra = float(ra_arr[0]); dec = float(dec_arr[0])
                            if np.isfinite(ra) and np.isfinite(dec):
                                info = f"HDU={idx} origin={origin} wcs_has_celestial={getattr(w,'has_celestial',None)}"
                                return ra, dec, info
                        except Exception as e:
                            # conversion failed for this candidate/origin; continue
                            continue
            return None, None, "no usable WCS found in any HDU"
    except Exception as e:
        return None, None, f"failed to open FITS: {e}"

def fill_radec_in_csv_debug(in_csv_path, out_csv_path=None, log_path=None):
    rows, fieldnames = read_csv_rows(in_csv_path)
    if out_csv_path is None:
        p = Path(in_csv_path)
        out_csv_path = str(p.with_name(p.stem + "_with_radec.csv"))
    # ensure columns exist
    if 'ra_deg' not in fieldnames:
        fieldnames = fieldnames + ['ra_deg']
    if 'dec_deg' not in fieldnames:
        fieldnames = fieldnames + ['dec_deg']

    # open debug log
    log_lines = []
    def log(s):
        log_lines.append(str(s))
        print(s)

    csv_dir = Path(in_csv_path).parent
    updated = 0
    for i, r in enumerate(rows):
        log(f"ROW {i}: start")
        # ensure fields exist for writing
        r.setdefault('ra_deg', '')
        r.setdefault('dec_deg', '')

        if r.get('ra_deg') not in (None, "", "nan") and r.get('dec_deg') not in (None, "", "nan"):
            log("  skipping: already has RA/Dec")
            continue

        orig = r.get('original_file', '').strip()
        try:
            x = float(r.get('x_pixel', ''))
            y = float(r.get('y_pixel', ''))
        except Exception:
            log(f"  invalid pixel coords x_pixel='{r.get('x_pixel')}' y_pixel='{r.get('y_pixel')}' - skipping")
            continue

        if not orig:
            log("  missing original_file - skipping")
            continue

        # resolve FITS path
        fits_path = orig
        if not os.path.exists(fits_path):
            candidate = csv_dir / orig
            if candidate.exists():
                fits_path = str(candidate)
                log(f"  resolved relative path -> {fits_path}")
            else:
                log(f"  FITS not found at '{orig}' or '{candidate}' - skipping")
                continue
        else:
            log(f"  found FITS at absolute path {fits_path}")

        # inspect fits info quick
        try:
            with fits.open(fits_path, memmap=False) as hdul:
                log(f"  FITS contains {len(hdul)} HDU(s); primary header keys: {list(hdul[0].header.keys())[:20]}")
        except Exception as e:
            log(f"  failed to open FITS for inspection: {e}")
            continue

        ra, dec, info = compute_radec_from_fits(fits_path, x, y, log)
        if ra is None:
            log(f"  compute_radec_from_fits failed: {info}")
            # leave blank
            continue

        r['ra_deg'] = f"{ra:.8f}"
        r['dec_deg'] = f"{dec:.8f}"
        updated += 1
        log(f"  wrote RA={r['ra_deg']} DEC={r['dec_deg']} ({info})")

    write_csv_rows(out_csv_path, fieldnames, rows)

    # write debug log if requested
    if log_path:
        try:
            with open(log_path, "w") as lf:
                for L in log_lines:
                    lf.write(str(L) + "\n")
        except Exception:
            pass

    return out_csv_path, updated

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Debug RA/Dec filler")
    ap.add_argument("csv", help="Input photometry CSV")
    ap.add_argument("--out", "-o", default=None, help="Output CSV path")
    ap.add_argument("--log", "-l", default=None, help="Write debug log to file")
    args = ap.parse_args()
    outpath, n = fill_radec_in_csv_debug(args.csv, args.out, args.log)
    print(f"Wrote {outpath} (filled RA/Dec for {n} rows)")