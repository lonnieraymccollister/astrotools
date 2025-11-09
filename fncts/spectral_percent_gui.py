#!/usr/bin/env python3
"""
spectral_percent_gui.py

PyQt6 GUI for spectral_percent_by_shells functionality.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from PyQt6.QtWidgets import (
    QApplication, QWidget, QGridLayout, QLabel, QLineEdit, QPushButton, QFileDialog,
    QComboBox, QMessageBox, QTextEdit, QSpinBox, QDoubleSpinBox
)
from PyQt6.QtCore import Qt

# reuse classification logic (same thresholds as CLI)
TYPES = list("OBAFGKM")

def sptype_from_teff(teff):
    if pd.isna(teff): return ""
    t = float(teff)
    if t >= 30000: return "O"
    if t >= 10000: return "B"
    if t >= 7500:  return "A"
    if t >= 6000:  return "F"
    if t >= 5200:  return "G"
    if t >= 3700:  return "K"
    return "M"

def sptype_from_bp_rp(bp_rp):
    if pd.isna(bp_rp): return ""
    c = float(bp_rp)
    if c <= -0.3: return "O"
    if c <= 0.0:  return "B"
    if c <= 0.3:  return "A"
    if c <= 0.8:  return "F"
    if c <= 1.2:  return "G"
    if c <= 1.8:  return "K"
    return "M"

def detect_and_build_color(df, color_hint, bp_hint, rp_hint):
    color_col = color_hint if color_hint in df.columns else None
    if color_col is None:
        for name in ("bp_rp","bp_rp_mean","phot_bp_mean_mag_minus_phot_rp_mean_mag"):
            if name in df.columns:
                color_col = name
                break
    bp_col = bp_hint if bp_hint in df.columns else None
    rp_col = rp_hint if rp_hint in df.columns else None
    if bp_col is None:
        for name in ("phot_bp_mean_mag","BPmag","bp_mag","bp_mean_mag"):
            if name in df.columns:
                bp_col = name; break
    if rp_col is None:
        for name in ("phot_rp_mean_mag","RPmag","rp_mag","rp_mean_mag"):
            if name in df.columns:
                rp_col = name; break
    if color_col is None and bp_col and rp_col:
        df["bp_rp"] = pd.to_numeric(df[bp_col], errors="coerce") - pd.to_numeric(df[rp_col], errors="coerce")
        color_col = "bp_rp"
    return color_col

def build_spectral_type_column(df, sptype_col_hint="spectral_type", teff_hint="teff_val", color_hint="bp_rp", bp_hint="phot_bp_mean_mag", rp_hint="phot_rp_mean_mag"):
    if sptype_col_hint in df.columns:
        s = df[sptype_col_hint].astype(str).str.strip().str.upper().str[:1]
        df["sptype"] = s.where(s.isin(TYPES), "")
        return df
    teff_col = teff_hint if teff_hint in df.columns else None
    if teff_col is None:
        for name in ("teff_val","phot_teff_mean","effective_temperature","teff"):
            if name in df.columns:
                teff_col = name; break
    color_col = detect_and_build_color(df, color_hint, bp_hint, rp_hint)
    if teff_col is not None:
        df["sptype"] = pd.to_numeric(df[teff_col], errors="coerce").apply(sptype_from_teff)
    elif color_col is not None:
        df["sptype"] = pd.to_numeric(df[color_col], errors="coerce").apply(sptype_from_bp_rp)
    else:
        df["sptype"] = ""
    return df

def parallax_to_distance(df, parallax_col="parallax"):
    if "distance_pc" in df.columns:
        df["distance_pc"] = pd.to_numeric(df["distance_pc"], errors="coerce")
        return df
    if parallax_col not in df.columns:
        raise ValueError("No parallax column and no distance_pc column found")
    p = pd.to_numeric(df[parallax_col], errors="coerce")
    df = df.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        df["distance_pc"] = np.where(p > 0, 1000.0 / p, np.nan)
    return df

def compute_shell_table(df, step, max_radius, return_counts=False):
    edges = np.arange(0.0, max_radius + step, step)
    shells = [(edges[i], edges[i+1]) for i in range(len(edges)-1)]
    df2 = df.replace([np.inf,-np.inf], np.nan).dropna(subset=["distance_pc"])
    out = pd.DataFrame(index=TYPES)
    for (r0, r1) in shells:
        mask = (df2["distance_pc"] > r0) & (df2["distance_pc"] <= r1)
        sub = df2[mask]
        N = len(sub)
        if N == 0:
            col = [0]*len(TYPES) if return_counts else [0.0]*len(TYPES)
        else:
            counts = sub["sptype"].value_counts()
            col = [int(counts.get(t, 0)) if return_counts else 100.0 * counts.get(t, 0) / N for t in TYPES]
        colname = f"{int(r0)}-{int(r1)}pc"
        out[colname] = col
    return out

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spectral-type by distance shells")
        self.resize(820, 540)
        self._build_ui()

    def _build_ui(self):
        layout = QGridLayout()

        layout.addWidget(QLabel("Input CSV"), 0, 0)
        self.input_edit = QLineEdit()
        layout.addWidget(self.input_edit, 0, 1)
        btn_browse = QPushButton("Browse")
        btn_browse.clicked.connect(self.browse_input)
        layout.addWidget(btn_browse, 0, 2)

        layout.addWidget(QLabel("Output CSV (optional)"), 1, 0)
        self.output_edit = QLineEdit()
        layout.addWidget(self.output_edit, 1, 1)
        btn_out = QPushButton("Choose")
        btn_out.clicked.connect(self.choose_output)
        layout.addWidget(btn_out, 1, 2)

        layout.addWidget(QLabel("Parallax column (mas)"), 2, 0)
        self.parallax_edit = QLineEdit("parallax")
        layout.addWidget(self.parallax_edit, 2, 1, 1, 2)

        layout.addWidget(QLabel("Step (pc)"), 3, 0)
        self.step_spin = QDoubleSpinBox()
        self.step_spin.setRange(1, 10000)
        self.step_spin.setValue(20.0)
        self.step_spin.setSingleStep(1.0)
        layout.addWidget(self.step_spin, 3, 1)

        layout.addWidget(QLabel("Max radius (pc)"), 3, 2)
        self.max_spin = QDoubleSpinBox()
        self.max_spin.setRange(1, 100000)
        self.max_spin.setValue(200.0)
        self.max_spin.setSingleStep(10.0)
        layout.addWidget(self.max_spin, 3, 3)

        layout.addWidget(QLabel("Output type"), 4, 0)
        self.output_combo = QComboBox()
        self.output_combo.addItems(["Percentages", "Counts"])
        layout.addWidget(self.output_combo, 4, 1)

        layout.addWidget(QLabel("Spectral_type column (if present)"), 5, 0)
        self.sptype_edit = QLineEdit("spectral_type")
        layout.addWidget(self.sptype_edit, 5, 1, 1, 3)

        layout.addWidget(QLabel("Teff column hint"), 6, 0)
        self.teff_edit = QLineEdit("teff_val")
        layout.addWidget(self.teff_edit, 6, 1, 1, 3)

        layout.addWidget(QLabel("BP-RP column hint"), 7, 0)
        self.color_edit = QLineEdit("bp_rp")
        layout.addWidget(self.color_edit, 7, 1, 1, 3)

        layout.addWidget(QLabel("BP mag column hint"), 8, 0)
        self.bp_edit = QLineEdit("phot_bp_mean_mag")
        layout.addWidget(self.bp_edit, 8, 1, 1, 3)

        layout.addWidget(QLabel("RP mag column hint"), 9, 0)
        self.rp_edit = QLineEdit("phot_rp_mean_mag")
        layout.addWidget(self.rp_edit, 9, 1, 1, 3)

        btn_run = QPushButton("Run")
        btn_run.clicked.connect(self.run)
        layout.addWidget(btn_run, 10, 0, 1, 4)

        layout.addWidget(QLabel("Output / Status"), 11, 0)
        self.status = QTextEdit()
        self.status.setReadOnly(True)
        layout.addWidget(self.status, 12, 0, 6, 4)

        self.setLayout(layout)

    def browse_input(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV", str(Path.cwd()), "CSV files (*.csv);;All files (*)")
        if path:
            self.input_edit.setText(path)

    def choose_output(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", str(Path.cwd()), "CSV files (*.csv);;All files (*)")
        if path:
            self.output_edit.setText(path)

    def run(self):
        in_path = self.input_edit.text().strip()
        if not in_path:
            QMessageBox.warning(self, "Missing input", "Choose an input CSV first")
            return
        try:
            df = pd.read_csv(in_path)
        except Exception as e:
            QMessageBox.critical(self, "Read error", f"Could not read CSV: {e}")
            return

        parallax_col = self.parallax_edit.text().strip()
        step = float(self.step_spin.value())
        max_r = float(self.max_spin.value())
        want_counts = (self.output_combo.currentText() == "Counts")

        sptype_hint = self.sptype_edit.text().strip()
        teff_hint = self.teff_edit.text().strip()
        color_hint = self.color_edit.text().strip()
        bp_hint = self.bp_edit.text().strip()
        rp_hint = self.rp_edit.text().strip()
        out_path = self.output_edit.text().strip() or None

        try:
            df = parallax_to_distance(df, parallax_col=parallax_col)
        except Exception as e:
            QMessageBox.critical(self, "Distance error", f"{e}")
            return

        df = build_spectral_type_column(df,
                                       sptype_col_hint=sptype_hint,
                                       teff_hint=teff_hint,
                                       color_hint=color_hint,
                                       bp_hint=bp_hint,
                                       rp_hint=rp_hint)
        table = compute_shell_table(df, step=step, max_radius=max_r, return_counts=want_counts)
        pd.set_option("display.float_format", lambda x: f"{x:10.6f}" if isinstance(x, float) else str(x))
        self.status.clear()
        self.status.append(f"Computed table ({'counts' if want_counts else 'percentages'}):\n")
        self.status.append(table.to_string())
        if out_path:
            try:
                Path(out_path).write_text(table.to_csv())
                self.status.append(f"\nSaved to {out_path}")
            except Exception as e:
                QMessageBox.critical(self, "Save error", f"Could not save output: {e}")

def parallax_to_distance(df, parallax_col="parallax"):
    if "distance_pc" in df.columns:
        df["distance_pc"] = pd.to_numeric(df["distance_pc"], errors="coerce")
        return df
    if parallax_col not in df.columns:
        raise ValueError("No parallax column and no distance_pc column found")
    p = pd.to_numeric(df[parallax_col], errors="coerce")
    df = df.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        df["distance_pc"] = np.where(p > 0, 1000.0 / p, np.nan)
    return df

def build_spectral_type_column(df, sptype_col_hint="spectral_type", teff_hint="teff_val", color_hint="bp_rp", bp_hint="phot_bp_mean_mag", rp_hint="phot_rp_mean_mag"):
    if sptype_col_hint in df.columns:
        s = df[sptype_col_hint].astype(str).str.strip().str.upper().str[:1]
        df["sptype"] = s.where(s.isin(TYPES), "")
        return df
    teff_col = teff_hint if teff_hint in df.columns else None
    if teff_col is None:
        for name in ("teff_val","phot_teff_mean","effective_temperature","teff"):
            if name in df.columns:
                teff_col = name; break
    color_col = detect_and_build_color(df, color_hint, bp_hint, rp_hint)
    if teff_col is not None:
        df["sptype"] = pd.to_numeric(df[teff_col], errors="coerce").apply(sptype_from_teff)
    elif color_col is not None:
        df["sptype"] = pd.to_numeric(df[color_col], errors="coerce").apply(sptype_from_bp_rp)
    else:
        df["sptype"] = ""
    return df

def detect_and_build_color(df, color_hint, bp_hint, rp_hint):
    color_col = color_hint if color_hint in df.columns else None
    if color_col is None:
        for name in ("bp_rp","bp_rp_mean","phot_bp_mean_mag_minus_phot_rp_mean_mag"):
            if name in df.columns:
                color_col = name
                break
    bp_col = bp_hint if bp_hint in df.columns else None
    rp_col = rp_hint if rp_hint in df.columns else None
    if bp_col is None:
        for name in ("phot_bp_mean_mag","BPmag","bp_mag","bp_mean_mag"):
            if name in df.columns:
                bp_col = name; break
    if rp_col is None:
        for name in ("phot_rp_mean_mag","RPmag","rp_mag","rp_mean_mag"):
            if name in df.columns:
                rp_col = name; break
    if color_col is None and bp_col and rp_col:
        df["bp_rp"] = pd.to_numeric(df[bp_col], errors="coerce") - pd.to_numeric(df[rp_col], errors="coerce")
        color_col = "bp_rp"
    return color_col

def compute_shell_table(df, step, max_radius, return_counts=False):
    edges = np.arange(0.0, max_radius + step, step)
    shells = [(edges[i], edges[i+1]) for i in range(len(edges)-1)]
    df2 = df.replace([np.inf,-np.inf], np.nan).dropna(subset=["distance_pc"])
    out = pd.DataFrame(index=TYPES)
    for (r0, r1) in shells:
        mask = (df2["distance_pc"] > r0) & (df2["distance_pc"] <= r1)
        sub = df2[mask]
        N = len(sub)
        if N == 0:
            col = [0]*len(TYPES) if return_counts else [0.0]*len(TYPES)
        else:
            counts = sub["sptype"].value_counts()
            col = [int(counts.get(t, 0)) if return_counts else 100.0 * counts.get(t, 0) / N for t in TYPES]
        colname = f"{int(r0)}-{int(r1)}pc"
        out[colname] = col
    return out

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
