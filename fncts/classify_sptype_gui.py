#!/usr/bin/env python3
"""
classify_sptype_gui.py

PyQt6 GUI to classify stars (OBAFGKM) from a Gaia CSV using Teff or BP-RP.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from PyQt6.QtWidgets import (
    QApplication, QWidget, QGridLayout, QLabel, QLineEdit, QPushButton, QFileDialog,
    QComboBox, QMessageBox, QTextEdit
)
from PyQt6.QtCore import Qt

def sptype_from_teff(teff):
    if np.isnan(teff):
        return ""
    t = float(teff)
    if t >= 30000:
        return "O"
    if t >= 10000:
        return "B"
    if t >= 7500:
        return "A"
    if t >= 6000:
        return "F"
    if t >= 5200:
        return "G"
    if t >= 3700:
        return "K"
    return "M"

def sptype_from_bp_rp(bp_rp):
    if np.isnan(bp_rp):
        return ""
    c = float(bp_rp)
    if c <= -0.3:
        return "O"
    if c <= 0.0:
        return "B"
    if c <= 0.3:
        return "A"
    if c <= 0.8:
        return "F"
    if c <= 1.2:
        return "G"
    if c <= 1.8:
        return "K"
    return "M"

class SptypeGui(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gaia Spectral Type Classifier")
        self.resize(760, 420)
        self._build_ui()

    def _build_ui(self):
        layout = QGridLayout()

        layout.addWidget(QLabel("Input CSV"), 0, 0)
        self.input_edit = QLineEdit()
        layout.addWidget(self.input_edit, 0, 1)
        btn_browse = QPushButton("Browse")
        btn_browse.clicked.connect(self.browse_input)
        layout.addWidget(btn_browse, 0, 2)

        layout.addWidget(QLabel("Output CSV"), 1, 0)
        self.output_edit = QLineEdit()
        layout.addWidget(self.output_edit, 1, 1)
        btn_out = QPushButton("Choose")
        btn_out.clicked.connect(self.choose_output)
        layout.addWidget(btn_out, 1, 2)

        layout.addWidget(QLabel("Method"), 2, 0)
        self.method_combo = QComboBox()
        self.method_combo.addItems(["auto (teff if present else color)", "teff", "color"])
        layout.addWidget(self.method_combo, 2, 1, 1, 2)

        layout.addWidget(QLabel("Teff column"), 3, 0)
        self.teff_col_edit = QLineEdit("teff_val")
        layout.addWidget(self.teff_col_edit, 3, 1, 1, 2)

        layout.addWidget(QLabel("BP-RP color column"), 4, 0)
        self.color_col_edit = QLineEdit("bp_rp")
        layout.addWidget(self.color_col_edit, 4, 1, 1, 2)

        layout.addWidget(QLabel("BPmag column (if no bp_rp)"), 5, 0)
        self.bpcol_edit = QLineEdit("phot_bp_mean_mag")
        layout.addWidget(self.bpcol_edit, 5, 1, 1, 2)

        layout.addWidget(QLabel("RPmag column (if no bp_rp)"), 6, 0)
        self.rpcol_edit = QLineEdit("phot_rp_mean_mag")
        layout.addWidget(self.rpcol_edit, 6, 1, 1, 2)

        btn_run = QPushButton("Classify and Save")
        btn_run.clicked.connect(self.run_classification)
        layout.addWidget(btn_run, 7, 0, 1, 3)

        layout.addWidget(QLabel("Status / Summary"), 8, 0)
        self.status_box = QTextEdit()
        self.status_box.setReadOnly(True)
        layout.addWidget(self.status_box, 9, 0, 4, 3)

        self.setLayout(layout)

    def browse_input(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV", str(Path.cwd()), "CSV files (*.csv);;All files (*)")
        if path:
            self.input_edit.setText(path)

    def choose_output(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", str(Path.cwd()), "CSV files (*.csv);;All files (*)")
        if path:
            self.output_edit.setText(path)

    def run_classification(self):
        in_path = self.input_edit.text().strip()
        out_path = self.output_edit.text().strip()
        if not in_path:
            QMessageBox.warning(self, "Missing input", "Choose an input CSV first")
            return
        if not out_path:
            QMessageBox.warning(self, "Missing output", "Choose an output CSV path first")
            return

        try:
            df = pd.read_csv(in_path)
        except Exception as e:
            QMessageBox.critical(self, "Read error", f"Could not read CSV: {e}")
            return

        teff_col_hint = self.teff_col_edit.text().strip()
        color_col_hint = self.color_col_edit.text().strip()
        bp_hint = self.bpcol_edit.text().strip()
        rp_hint = self.rpcol_edit.text().strip()
        method_choice = self.method_combo.currentText()

        # auto-detect columns if not present
        teff_col = teff_col_hint if teff_col_hint in df.columns else None
        color_col = color_col_hint if color_col_hint in df.columns else None

        if teff_col is None:
            for name in ("teff_val", "phot_teff_mean", "effective_temperature", "teff"):
                if name in df.columns:
                    teff_col = name
                    break

        if color_col is None:
            for name in ("bp_rp", "bp_rp_mean", "phot_bp_mean_mag_minus_phot_rp_mean_mag"):
                if name in df.columns:
                    color_col = name
                    break

        bp_col = bp_hint if bp_hint in df.columns else None
        rp_col = rp_hint if rp_hint in df.columns else None

        if bp_col is None:
            for name in ("phot_bp_mean_mag","BPmag","bp_mag","bp_mean_mag"):
                if name in df.columns:
                    bp_col = name
                    break
        if rp_col is None:
            for name in ("phot_rp_mean_mag","RPmag","rp_mag","rp_mean_mag"):
                if name in df.columns:
                    rp_col = name
                    break

        # decide method
        if method_choice.startswith("auto"):
            method = "teff" if teff_col is not None else "color"
        else:
            method = method_choice

        if method == "teff" and teff_col is None:
            QMessageBox.critical(self, "Column missing", "Teff method selected but no Teff column found")
            return
        if method == "color" and color_col is None and not (bp_col and rp_col):
            QMessageBox.critical(self, "Column missing", "Color method selected but no BP-RP column and no BP/RP mags found")
            return

        # If we will use color but bp_rp column not found, compute it from BP/RP mags if available
        if method == "color" and color_col is None and bp_col and rp_col:
            try:
                df["bp_rp"] = pd.to_numeric(df[bp_col], errors='coerce') - pd.to_numeric(df[rp_col], errors='coerce')
                color_col = "bp_rp"
                self.status_box.append(f"Computed bp_rp from {bp_col} - {rp_col}")
            except Exception as e:
                QMessageBox.critical(self, "Compute color", f"Failed to compute bp_rp: {e}")
                return

        # compute spectral_type
        if method == "teff":
            col = teff_col
            self.status_box.append(f"Using Teff column: {col}")
            df["spectral_type"] = df[col].apply(lambda x: sptype_from_teff(float(x)) if pd.notna(x) else "")
        else:
            col = color_col
            self.status_box.append(f"Using BP-RP column: {col}")
            df["spectral_type"] = df[col].apply(lambda x: sptype_from_bp_rp(float(x)) if pd.notna(x) else "")

        # summary
        counts = df["spectral_type"].value_counts(dropna=True).sort_index()
        self.status_box.append("Classification complete. Counts:")
        for k, v in counts.items():
            self.status_box.append(f"  {k}: {v}")
        self.status_box.append(f"Saving to: {out_path}")

        try:
            df.to_csv(out_path, index=False)
            self.status_box.append("Saved successfully.")
            QMessageBox.information(self, "Done", f"Saved {len(df)} rows to {out_path}")
        except Exception as e:
            QMessageBox.critical(self, "Save error", f"Could not save output CSV: {e}")

def main():
    app = QApplication(sys.argv)
    w = SptypeGui()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()