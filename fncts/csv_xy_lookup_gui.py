#!/usr/bin/env python3
"""
csv_xy_lookup_gui.py

Simple PyQt6 GUI to:
 - open a CSV file
 - choose coordinate columns (dropdowns)
 - enter x, y values
 - enter requested field names (space-separated; fuzzy-matched)
 - set closest N and optional distance tolerance
 - run lookup and show results

Dependencies:
 - Python 3.8+
 - PyQt6
 - pandas (recommended)
 - scipy (optional; speeds nearest-neighbor search)
Install missing deps with pip (e.g., pip install PyQt6 pandas scipy)
"""
import sys
import math
from difflib import get_close_matches

from PyQt6.QtWidgets import (
    QApplication, QWidget, QGridLayout, QPushButton, QLabel, QComboBox,
    QLineEdit, QTextEdit, QFileDialog, QSpinBox, QDoubleSpinBox, QHBoxLayout,
    QMessageBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QTextCursor


# Try imports with graceful fallback
try:
    import pandas as pd
except Exception:
    pd = None

import numpy as np
try:
    from scipy.spatial import cKDTree as KDTree
    _HAVE_KDTREE = True
except Exception:
    _HAVE_KDTREE = False


def fuzzy_col_match(cols, want, cutoff=0.6):
    """Return best-matching column name for want from list cols using difflib.
       Returns None if no good match above cutoff."""
    if want is None or want == "":
        return None
    cols_l = [c.lower() for c in cols]
    want_l = want.lower()
    if want in cols:
        return want
    if want_l in cols_l:
        return cols[cols_l.index(want_l)]
    # substring
    for c, cl in zip(cols, cols_l):
        if want_l in cl:
            return c
    matches = get_close_matches(want_l, cols_l, n=1, cutoff=cutoff)
    if not matches:
        return None
    return cols[cols_l.index(matches[0])]


def find_nearest_rows(pts, target, k=1):
    """pts: (N,2) numpy array; target: (2,) array-like; returns (idxs, dists)"""
    pts = np.asarray(pts, dtype=float)
    target = np.asarray(target, dtype=float)
    if _HAVE_KDTREE:
        tree = KDTree(pts)
        dists, idx = tree.query(target, k=k)
        if k == 1:
            return np.atleast_1d(idx), np.atleast_1d(dists)
        return idx, dists
    # brute force
    d2 = np.sum((pts - target) ** 2, axis=1)
    idx_sorted = np.argsort(d2)[:k]
    dists = np.sqrt(d2[idx_sorted])
    return idx_sorted, dists


class CsvLookupWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CSV XY Lookup (fuzzy fields)")
        self.resize(780, 460)
        self.df = None
        self.columns = []

        layout = QGridLayout()
        row = 0

        btn_open = QPushButton("Open CSV")
        btn_open.clicked.connect(self.open_csv)
        layout.addWidget(btn_open, row, 0)

        self.lbl_file = QLabel("No file loaded")
        layout.addWidget(self.lbl_file, row, 1, 1, 4)
        row += 1

        layout.addWidget(QLabel("X column:"), row, 0)
        self.combo_x = QComboBox()
        layout.addWidget(self.combo_x, row, 1)

        layout.addWidget(QLabel("Y column:"), row, 2)
        self.combo_y = QComboBox()
        layout.addWidget(self.combo_y, row, 3)

        btn_refresh = QPushButton("Refresh columns")
        btn_refresh.clicked.connect(self.refresh_columns)
        layout.addWidget(btn_refresh, row, 4)
        row += 1

        layout.addWidget(QLabel("X value:"), row, 0)
        self.edit_x = QLineEdit()
        self.edit_x.setPlaceholderText("numeric x")
        layout.addWidget(self.edit_x, row, 1)

        layout.addWidget(QLabel("Y value:"), row, 2)
        self.edit_y = QLineEdit()
        self.edit_y.setPlaceholderText("numeric y")
        layout.addWidget(self.edit_y, row, 3)

        row += 1
        layout.addWidget(QLabel("Fields (space-separated, fuzzy)"), row, 0, 1, 2)
        self.edit_fields = QLineEdit()
        self.edit_fields.setPlaceholderText("e.g. r-g g-b RminusG")
        layout.addWidget(self.edit_fields, row, 2, 1, 3)
        row += 1

        layout.addWidget(QLabel("Closest N:"), row, 0)
        self.spin_k = QSpinBox()
        self.spin_k.setMinimum(1)
        self.spin_k.setValue(1)
        layout.addWidget(self.spin_k, row, 1)

        layout.addWidget(QLabel("Tolerance (optional)"), row, 2)
        self.spin_tol = QDoubleSpinBox()
        self.spin_tol.setMinimum(0.0)
        self.spin_tol.setDecimals(6)
        self.spin_tol.setValue(0.0)
        self.spin_tol.setSingleStep(0.1)
        layout.addWidget(self.spin_tol, row, 3)

        btn_run = QPushButton("Run lookup")
        btn_run.clicked.connect(self.run_lookup)
        layout.addWidget(btn_run, row, 4)
        row += 1

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        layout.addWidget(self.output, row, 0, 1, 5)
        row += 1

        # convenience row: show available columns
        layout.addWidget(QLabel("Available columns (click to copy name):"), row, 0, 1, 5)
        row += 1
        self.col_list = QTextEdit()
        self.col_list.setReadOnly(False)
        self.col_list.setMaximumHeight(120)
        layout.addWidget(self.col_list, row, 0, 1, 5)
        self.col_list.setAcceptRichText(False)

        self.setLayout(layout)

    def open_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV files (*.csv);;All files (*.*)")
        if not path:
            return
        try:
            if pd is not None:
                df = pd.read_csv(path, dtype=str)  # read as str to preserve everything, cast later
                # try to coerce numeric columns later
            else:
                # fallback: simple csv reader -> build DataFrame-like structure with numpy
                import csv
                with open(path, newline='', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                if not rows:
                    raise ValueError("Empty CSV")
                headers = rows[0]
                data = rows[1:]
                df = {h: [r[i] if i < len(r) else "" for r in data] for i, h in enumerate(headers)}
                import pandas as _pd_fallback
                df = _pd_fallback.DataFrame(df)
            self.df = df
            self.path = path
            self.columns = list(self.df.columns)
            self.lbl_file.setText(path)
            self.refresh_columns()
            self.col_list.setPlainText("\n".join(self.columns))
            self.output.append(f"Loaded {len(self.df)} rows, {len(self.columns)} columns.")
        except Exception as e:
            QMessageBox.critical(self, "Error reading CSV", str(e))

    def refresh_columns(self):
        self.combo_x.clear()
        self.combo_y.clear()
        if not self.columns:
            return
        self.combo_x.addItems(self.columns)
        self.combo_y.addItems(self.columns)
        # try to auto-select common names
        for pref in ("x", "ra", "lon", "px"):
            for c in self.columns:
                if c.lower() == pref:
                    self.combo_x.setCurrentText(c)
                    break
        for pref in ("y", "dec", "lat", "py"):
            for c in self.columns:
                if c.lower() == pref:
                    self.combo_y.setCurrentText(c)
                    break

    def run_lookup(self):
        self.output.clear()
        if self.df is None:
            QMessageBox.information(self, "No CSV", "Please load a CSV first.")
            return
        xcol = self.combo_x.currentText()
        ycol = self.combo_y.currentText()
        if not xcol or not ycol:
            QMessageBox.warning(self, "No columns", "Select X and Y columns.")
            return
        xstr = self.edit_x.text().strip()
        ystr = self.edit_y.text().strip()
        try:
            xval = float(xstr)
            yval = float(ystr)
        except Exception:
            QMessageBox.warning(self, "Bad coordinates", "X and Y must be numeric.")
            return

        k = int(self.spin_k.value())
        tol = float(self.spin_tol.value())
        fields_text = self.edit_fields.text().strip()
        if not fields_text:
            QMessageBox.warning(self, "No fields", "Enter one or more field names to lookup.")
            return
        requested = fields_text.split()

        # ensure numeric coordinate arrays
        try:
            self.df[xcol] = pd.to_numeric(self.df[xcol], errors='coerce')
            self.df[ycol] = pd.to_numeric(self.df[ycol], errors='coerce')
        except Exception:
            # fallback: try converting with numpy
            try:
                self.df[xcol] = self.df[xcol].astype(float)
                self.df[ycol] = self.df[ycol].astype(float)
            except Exception:
                QMessageBox.critical(self, "Conversion failed", "Cannot convert coordinate columns to numeric.")
                return

        valid = self.df[xcol].notna() & self.df[ycol].notna()
        if not valid.any():
            QMessageBox.critical(self, "No valid coords", "No numeric coordinates found in chosen columns.")
            return

        df_sub = self.df.loc[valid].reset_index(drop=False)  # keep original index as 'index' column
        coords = np.column_stack([df_sub[xcol].astype(float).values, df_sub[ycol].astype(float).values])
        idxs, dists = find_nearest_rows(coords, (xval, yval), k=k)

        # if tol > 0, filter by tolerance
        if tol > 0.0:
            mask = np.asarray(dists) <= tol
            if not mask.any():
                self.output.append(f"No rows within tolerance {tol:.6g}. Closest distance = {float(np.min(dists)):.6g}")
                return
            idxs = np.atleast_1d(idxs)[mask]
            dists = np.atleast_1d(dists)[mask]

        # match requested fields fuzzily to actual columns
        matched = {}
        for req in requested:
            col = fuzzy_col_match(list(self.df.columns), req, cutoff=0.5)
            matched[req] = col

        # print header
        self.output.append(f"Lookup at x={xval}, y={yval}  (using columns '{xcol}','{ycol}')")
        self.output.append(f"CSV rows considered: {len(df_sub)}   Returning {len(np.atleast_1d(idxs))} rows\n")

        for i, dist in zip(np.atleast_1d(idxs), np.atleast_1d(dists)):
            row = df_sub.iloc[int(i)]
            orig_idx = int(row['index']) if 'index' in row else i
            self.output.append(f"Row index: {orig_idx}  distance: {dist:.6g}")
            for req, col in matched.items():
                if col is None:
                    self.output.append(f"  {req} -> NOT FOUND")
                else:
                    val = row.get(col, "")
                    self.output.append(f"  {req} (matched to '{col}') = {val}")
            self.output.append("")  # blank line

        # scroll to top
        self.output.verticalScrollBar().setValue(0)

def main():
    app = QApplication(sys.argv)
    w = CsvLookupWidget()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()