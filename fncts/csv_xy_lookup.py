#!/usr/bin/env python3
"""
csv_xy_lookup.py

Usage:
  python csv_xy_lookup.py data.csv --x 123.4 --y 45.6 --fields "r-g" "g-b" [--tol 5.0] [--index X Y] [--closest N]

Notes:
  - CSV must contain two numeric coordinate columns (default names "x" and "y"
    unless you specify --index to set which columns to use).
  - Field names are matched fuzzily (case-insensitive, tolerant to minor typos).
  - Outputs the nearest-match row(s) and the requested field values.
"""
import argparse
import pandas as pd
import numpy as np
from difflib import get_close_matches
import sys

try:
    from scipy.spatial import cKDTree as KDTree
    _HAVE_KDTREE = True
except Exception:
    _HAVE_KDTREE = False

def fuzzy_col_match(cols, want, n=1, cutoff=0.6):
    """Return best-matching column name for want from list cols using difflib.
       If nothing matches above cutoff, return None."""
    cols_l = [c.lower() for c in cols]
    want_l = want.lower()
    # exact/substring quick checks
    if want in cols:
        return want
    if want_l in cols_l:
        return cols[cols_l.index(want_l)]
    # substring match
    for c, cl in zip(cols, cols_l):
        if want_l in cl:
            return c
    # difflib fallback
    matches = get_close_matches(want_l, cols_l, n=n, cutoff=cutoff)
    if not matches:
        return None
    return cols[cols_l.index(matches[0])]

def find_nearest_rows(df, xcol, ycol, x, y, k=1):
    pts = np.column_stack([df[xcol].astype(float).values, df[ycol].astype(float).values])
    target = np.array([x, y], dtype=float)
    if _HAVE_KDTREE:
        tree = KDTree(pts)
        dists, idx = tree.query(target, k=k)
        if k == 1:
            return np.atleast_1d(idx), np.atleast_1d(dists)
        return idx, dists
    # brute force
    d2 = np.sum((pts - target)**2, axis=1)
    idx = np.argsort(d2)[:k]
    dists = np.sqrt(d2[idx])
    return idx, dists

def parse_args():
    p = argparse.ArgumentParser(description="Lookup CSV by (x,y) and fuzzy-match field names")
    p.add_argument("csvfile", help="Input CSV file")
    p.add_argument("--x", type=float, required=True, help="X coordinate (numeric)")
    p.add_argument("--y", type=float, required=True, help="Y coordinate (numeric)")
    p.add_argument("--fields", nargs="+", required=True, help="Field names to lookup, e.g. 'r-g' 'g-b'")
    p.add_argument("--index", nargs=2, default=["x","y"], help="Column names for coordinates (default: x y)")
    p.add_argument("--closest", type=int, default=1, help="Return N closest rows (default 1)")
    p.add_argument("--tol", type=float, default=None, help="Optional distance tolerance (same units as x/y). If set, rows farther than tol are ignored")
    p.add_argument("--cutoff", type=float, default=0.6, help="Fuzzy match cutoff (0..1), higher is stricter")
    return p.parse_args()

def main():
    args = parse_args()
    try:
        df = pd.read_csv(args.csvfile)
    except Exception as e:
        sys.exit(f"Error reading CSV '{args.csvfile}': {e}")

    xcol, ycol = args.index
    if xcol not in df.columns or ycol not in df.columns:
        sys.exit(f"Coordinate columns not found in CSV. Available columns: {list(df.columns)}")

    # ensure numeric coordinates
    try:
        df[xcol] = pd.to_numeric(df[xcol], errors='coerce')
        df[ycol] = pd.to_numeric(df[ycol], errors='coerce')
    except Exception:
        pass
    valid_mask = df[xcol].notna() & df[ycol].notna()
    if not valid_mask.any():
        sys.exit("No valid numeric coordinates found in the CSV for the chosen index columns.")

    df = df.loc[valid_mask].reset_index(drop=True)

    # find nearest rows
    idxs, dists = find_nearest_rows(df, xcol, ycol, args.x, args.y, k=args.closest)
    # apply tolerance if provided
    if args.tol is not None:
        mask = dists <= args.tol
        if not mask.any():
            sys.exit(f"No rows within tolerance {args.tol}. Closest distance = {float(np.min(dists))}")
        idxs = idxs[mask]
        dists = dists[mask]

    # fuzzy-match requested fields to actual columns
    matched_fields = {}
    for f in args.fields:
        col = fuzzy_col_match(list(df.columns), f, cutoff=args.cutoff)
        if col is None:
            print(f"Warning: field '{f}' not found (no fuzzy match). Available columns: {list(df.columns)}", file=sys.stderr)
        matched_fields[f] = col

    # print results
    for i_row, dist in zip(np.atleast_1d(idxs), np.atleast_1d(dists)):
        row = df.iloc[i_row]
        coords = (float(row[xcol]), float(row[ycol]))
        print(f"\nRow index: {i_row}  distance: {dist:.6g}  {xcol}={coords[0]:.6g}  {ycol}={coords[1]:.6g}")
        for req_name, col in matched_fields.items():
            if col is None:
                print(f"  {req_name} -> NOT FOUND")
            else:
                val = row.get(col, "")
                print(f"  {req_name} (matched to column '{col}') = {val}")
    print()

if __name__ == "__main__":
    main()