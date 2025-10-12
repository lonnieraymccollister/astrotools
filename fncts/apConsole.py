#!/usr/bin/env python3
"""
fetch_catalog_mags_console.py

Reads input CSV with 'ra','dec' (deg or sexagesimal strings), queries APASS or Gaia,
and prints results to the console (one line per input position).

Usage:
  python fetch_catalog_mags_console.py input.csv --catalog apass --radius 5.0
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
from tqdm import tqdm

# allow multiple neighbors so we can pick nearest
Vizier.ROW_LIMIT = 50

CATALOGS = {
    'apass': {
        'vizier_id': "II/336/apass9",
        'cols': ['RAJ2000', 'DEJ2000', 'Bmag', 'Vmag', 'gmag', 'rmag', 'imag']
    },
    'gaia': {
        'vizier_id': "I/355/gaiadr3",
        'cols': ['RA_ICRS', 'DE_ICRS', 'Gmag', 'BPmag', 'RPmag']
    }
}

def query_vizier_nearest(coord, catalog_key, radius_arcsec=3.0, max_retries=3, pause=0.2):
    info = CATALOGS[catalog_key]
    vizid = info['vizier_id']
    cols = info['cols']
    v = Vizier(columns=cols)
    v.ROW_LIMIT = 50
    last_exc = None
    for attempt in range(max_retries):
        try:
            res = v.query_region(coord, radius=radius_arcsec * u.arcsec, catalog=vizid)
            break
        except Exception as e:
            last_exc = e
            time.sleep(pause)
    else:
        raise RuntimeError(f"Vizier query failed after {max_retries} attempts: {last_exc}")

    if (res is None) or (len(res) == 0) or (len(res[0]) == 0):
        return None

    table = res[0]
    # build SkyCoord for entries
    cat_ra_col, cat_dec_col = cols[0], cols[1]
    ras = np.array(table[cat_ra_col])
    decs = np.array(table[cat_dec_col])
    try:
        cat_coords = SkyCoord(ras * u.deg, decs * u.deg, frame='icrs')
    except Exception:
        # fallback: treat values as strings and let astropy parse (rare)
        coords_list = [f"{r} {d}" for r, d in zip(ras, decs)]
        cat_coords = SkyCoord(coords_list, unit=(u.deg, u.deg), frame='icrs')

    seps = coord.separation(cat_coords).arcsec
    idx = int(np.nanargmin(seps))
    sep_best = float(seps[idx])

    row = table[idx]
    out = {'sep_arcsec': sep_best}
    # map mags
    for c in cols[2:]:
        try:
            val = row[c] if c in row.colnames else None
            out[c] = float(val) if (val is not None and val != '--') else np.nan
        except Exception:
            out[c] = np.nan
    # canonical cat coords
    out['cat_ra'] = float(row[cat_ra_col])
    out['cat_dec'] = float(row[cat_dec_col])
    return out

def parse_coord_pair(ra_in, dec_in):
    """
    Accept decimal degrees (numbers) or sexagesimal strings.
    Returns SkyCoord in ICRS (deg).
    """
    # numeric -> assume degrees
    try:
        ra_f = float(ra_in)
        dec_f = float(dec_in)
        return SkyCoord(ra_f * u.deg, dec_f * u.deg, frame='icrs')
    except Exception:
        # try to parse as sexagesimal string pair
        s = f"{ra_in} {dec_in}"
        # astropy will interpret RA as hourangle if unit specified; try autodetect by not specifying units
        return SkyCoord(s, frame='icrs')

def main():
    p = argparse.ArgumentParser(description="Query APASS or Gaia and print results to console")
    p.add_argument("input_csv", help="CSV with columns 'ra' and 'dec' (degrees or sexagesimal strings)")
    p.add_argument("--catalog", choices=['apass','gaia'], default='apass')
    p.add_argument("--radius", type=float, default=3.0, help="cone search radius in arcsec")
    p.add_argument("--sleep", type=float, default=0.08, help="sleep between queries (s)")
    p.add_argument("--max-retries", type=int, default=3)
    args = p.parse_args()

    inp = Path(args.input_csv)
    if not inp.exists():
        raise SystemExit(f"Input CSV not found: {inp}")

    df = pd.read_csv(inp, dtype=str)  # read as strings to support mixed formats
    if 'ra' not in df.columns or 'dec' not in df.columns:
        raise SystemExit("Input CSV must contain 'ra' and 'dec' columns")

    cat = args.catalog
    cols = CATALOGS[cat]['cols']

    # Print header
    if cat == 'apass':
        mag_labels = ['Bmag','gmag','rmag','imag','Vmag']
    else:
        mag_labels = ['Gmag','BPmag','RPmag']

    header = ["input_ra","input_dec","cat_ra","cat_dec","sep_arcsec"] + mag_labels
    print(",".join(header))

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Querying"):
        ra_in = row['ra'].strip()
        dec_in = row['dec'].strip()
        try:
            coord = parse_coord_pair(ra_in, dec_in)
        except Exception as e:
            # parsing failed
            out_vals = [ra_in, dec_in, "", "", "", *([""] * len(mag_labels))]
            print(",".join(map(str, out_vals)))
            continue

        try:
            res = query_vizier_nearest(coord, cat, radius_arcsec=args.radius,
                                       max_retries=args.max_retries, pause=args.sleep)
        except Exception as e:
            # query failed: print row with error note
            out_vals = [ra_in, dec_in, "QUERY_FAIL", "QUERY_FAIL", ""] + (["ERR"] * len(mag_labels))
            print(",".join(map(str, out_vals)))
            time.sleep(args.sleep)
            continue

        if res is None:
            out_vals = [ra_in, dec_in, "", "", ""] + ([""] * len(mag_labels))
            print(",".join(map(str, out_vals)))
        else:
            if cat == 'apass':
                mags = [res.get('Bmag', np.nan), res.get('gmag', np.nan), res.get('rmag', np.nan),
                        res.get('imag', np.nan), res.get('Vmag', np.nan)]
            else:
                mags = [res.get('Gmag', np.nan), res.get('BPmag', np.nan), res.get('RPmag', np.nan)]
                # pad to same number of columns for consistent printing
                if len(mags) < len(mag_labels):
                    mags += [np.nan] * (len(mag_labels) - len(mags))

            out_vals = [
                ra_in,
                dec_in,
                f"{res['cat_ra']:.8f}",
                f"{res['cat_dec']:.8f}",
                f"{res['sep_arcsec']:.3f}"
            ] + [("" if (m is None or (isinstance(m, float) and np.isnan(m))) else f"{m:.4f}") for m in mags]

            print(",".join(map(str, out_vals)))
        time.sleep(args.sleep)

if __name__ == "__main__":
    from tqdm import tqdm
    main()