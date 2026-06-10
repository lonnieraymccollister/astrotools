#!/usr/bin/env python3
"""
Query APASS or Gaia for positions in an input CSV and print results.

Input CSV must contain columns 'ra' and 'dec' (decimal degrees or sexagesimal strings).

Examples:
  # write to stdout (pipe or redirect in shell)
  python J:/work/zwork/fncts/apConsole.py J:/work/zwork/astk.csv --catalog gaia --radius 5.0 > J:/work/zwork/out.csv

  # write directly to a file using --output
  python J:/work/zwork/fncts/apConsole.py J:/work/zwork/astk.csv --catalog gaia --radius 5.0 --output J:/work/zwork/out.csv
"""
import argparse
import time
from pathlib import Path
import sys

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
        'cols': ['RA_ICRS', 'DE_ICRS', 'Gmag', 'BPmag', 'RPmag', 'Plx', 'e_Plx', 'RUWE']
    }
}

# common alternate names for parallax in various catalogs / tables
PARALLAX_CANDIDATES = ['Plx', 'parallax', 'parallax_value', 'parallax_mean', 'parallax_error']


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
    # map mags (we will keep missing mags as None)
    for c in cols[2:]:
        try:
            if c in row.colnames:
                val = row[c]
                if val is None or (isinstance(val, str) and val.strip() in ['', '--']):
                    out[c] = None
                else:
                    try:
                        out[c] = float(val)
                    except Exception:
                        out[c] = str(val)
            else:
                out[c] = None
        except Exception:
            out[c] = None

    # canonical cat coords
    try:
        out['cat_ra'] = float(row[cat_ra_col])
        out['cat_dec'] = float(row[cat_dec_col])
    except Exception:
        out['cat_ra'] = None
        out['cat_dec'] = None

    # Attempt to extract Gaia parallax and parallax_error if present under any common name
    if catalog_key == 'gaia':
        plx = None
        e_plx = None
        for pname in PARALLAX_CANDIDATES:
            if pname in row.colnames:
                try:
                    val = row[pname]
                    if val is None or (isinstance(val, str) and val.strip() in ['', '--']):
                        continue
                    plx = float(val)
                    # try to find an associated error column nearby (common name patterns)
                    for errname in ['e_' + pname, pname + '_error', 'parallax_error', 'e_parallax']:
                        if errname in row.colnames:
                            try:
                                e_plx = float(row[errname])
                                break
                            except Exception:
                                continue
                    break
                except Exception:
                    continue
        out['parallax'] = plx
        out['parallax_error'] = e_plx
        out['ruwe'] = float(row['RUWE']) if 'RUWE' in row.colnames and row['RUWE'] not in (None, '--') else None

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
        return SkyCoord(s, frame='icrs')


def fmt_empty_or(value, fmt="{:.4f}"):
    """
    Format numeric values using fmt; return empty string for None or NaN.
    Keep strings as-is.
    """
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    if isinstance(value, (int, float)):
        return fmt.format(value)
    return str(value)


def build_output_lines(results, catalog):
    """Convert list of result dicts into CSV lines (strings) with safe formatting."""

    def fmt_float(val, fmt="{:.6f}"):
        """Safely format floats; return empty string for anything non-numeric."""
        try:
            if val is None:
                return ""
            return fmt.format(float(val))
        except Exception:
            return ""

    # Column order
    if catalog == 'apass':
        mag_labels = ['Bmag', 'gmag', 'rmag', 'imag', 'Vmag']
    else:
        mag_labels = ['Gmag', 'BPmag', 'RPmag', 'parallax', 'parallax_error', 'ruwe']

    header = ["input_ra", "input_dec", "cat_ra", "cat_dec", "sep_arcsec"] + mag_labels
    lines = [",".join(header)]

    for item in results:
        if item is None:
            continue

        # Always preserve input coords as strings
        ra_in = item.get('input_ra', '')
        dec_in = item.get('input_dec', '')

        # Safe formatting for numeric fields
        cat_ra = fmt_float(item.get('cat_ra'), "{:.8f}")
        cat_dec = fmt_float(item.get('cat_dec'), "{:.8f}")
        sep = fmt_float(item.get('sep_arcsec'), "{:.3f}")

        # Magnitudes and other numeric fields
        mags_fmt = []
        for key in mag_labels:
            val = item.get(key)
            # Try numeric formatting first
            out = fmt_float(val)
            if out == "":
                # If not numeric, keep strings like "ERR" or leave blank
                if isinstance(val, str) and val not in ("", "None"):
                    mags_fmt.append(val)
                else:
                    mags_fmt.append("")
            else:
                mags_fmt.append(out)

        row = [ra_in, dec_in, cat_ra, cat_dec, sep] + mags_fmt
        lines.append(",".join(map(str, row)))

    return lines


def main(argv=None):
    p = argparse.ArgumentParser(description="Query APASS or Gaia and print results to console")
    p.add_argument("input_csv", help="CSV with columns 'ra' and 'dec' (degrees or sexagesimal strings)")
    p.add_argument("--catalog", choices=['apass', 'gaia'], default='apass')
    p.add_argument("--radius", type=float, default=3.0, help="cone search radius in arcsec")
    p.add_argument("--sleep", type=float, default=0.08, help="sleep between queries (s)")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--output", "-o", help="Optional output CSV path (writes file instead of stdout)")
    args = p.parse_args(argv)

    inp = Path(args.input_csv)
    if not inp.exists():
        raise SystemExit(f"Input CSV not found: {inp}")

    try:
        df = pd.read_csv(inp, dtype=str)  # read as strings to support mixed formats
    except Exception as e:
        raise SystemExit(f"Failed to read CSV {inp}: {e}")

    if 'ra' not in df.columns or 'dec' not in df.columns:
        raise SystemExit("Input CSV must contain 'ra' and 'dec' columns")

    catalog = args.catalog
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Querying", unit="pos"):
        ra_in = row['ra'].strip()
        dec_in = row['dec'].strip()
        entry = {'input_ra': ra_in, 'input_dec': dec_in}
        try:
            coord = parse_coord_pair(ra_in, dec_in)
        except Exception:
            # parsing failed -> write empty metrics for this row later
            # preserve input coords and produce blank fields
            entry.update({'cat_ra': None, 'cat_dec': None, 'sep_arcsec': None})
            # mag keys consistent with build_output_lines expectations
            if catalog == 'apass':
                for k in ['Bmag', 'gmag', 'rmag', 'imag', 'Vmag']:
                    entry[k] = None
            else:
                for k in ['Gmag', 'BPmag', 'RPmag', 'parallax', 'parallax_error', 'ruwe']:
                    entry[k] = None
            results.append(entry)
            continue

        try:
            res = query_vizier_nearest(coord, catalog, radius_arcsec=args.radius,
                                       max_retries=args.max_retries, pause=args.sleep)
        except Exception:
            # query failed: encode failure in the entry
            entry.update({'cat_ra': "QUERY_FAIL", 'cat_dec': "QUERY_FAIL", 'sep_arcsec': None})
            if catalog == 'apass':
                for k in ['Bmag', 'gmag', 'rmag', 'imag', 'Vmag']:
                    entry[k] = "ERR"
            else:
                for k in ['Gmag', 'BPmag', 'RPmag', 'parallax', 'parallax_error', 'ruwe']:
                    entry[k] = "ERR"
            results.append(entry)
            time.sleep(args.sleep)
            continue

        if res is None:
            # no match found
            entry.update({'cat_ra': None, 'cat_dec': None, 'sep_arcsec': None})
            if catalog == 'apass':
                for k in ['Bmag', 'gmag', 'rmag', 'imag', 'Vmag']:
                    entry[k] = None
            else:
                for k in ['Gmag', 'BPmag', 'RPmag', 'parallax', 'parallax_error', 'ruwe']:
                    entry[k] = None
        else:
            # Merge res dict into entry
            entry.update(res)
        results.append(entry)
        time.sleep(args.sleep)

    lines = build_output_lines(results, catalog)

    if args.output:
        outp = Path(args.output)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", newline="") as fh:
            fh.write("\n".join(lines))
        print(f"Wrote {len(lines)-1} results to {outp}", file=sys.stderr)
    else:
        # print to stdout
        sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()