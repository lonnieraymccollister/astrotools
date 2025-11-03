#!/usr/bin/env python3
"""
python ecolor_vs_distance_gui.py

PyQt6 GUI wrapper for plotting E(color) vs Distance with binned running median.
"""

import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from scipy.ndimage import uniform_filter1d
from astropy.coordinates import SkyCoord
import astropy.units as u

from PyQt6.QtWidgets import (
    QApplication, QWidget, QGridLayout, QLabel, QLineEdit, QPushButton, QFileDialog,
    QComboBox, QTextEdit, QProgressBar, QMessageBox, QCheckBox, QSpinBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject

# Worker signals
class WorkerSignals(QObject):
    status = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

def compute_and_plot(params, signals: WorkerSignals):
    try:
        csv_path = Path(params["csv"])
        ecol_col = params["ecol_col"]
        parallax_col = params["parallax_col"]
        dist_col = params["dist_col"]
        ra_col = params["ra_col"]
        dec_col = params["dec_col"]
        use_circle = params["use_circle"]
        center_ra = float(params["center_ra"])
        center_dec = float(params["center_dec"])
        radius_arcmin = float(params["radius_arcmin"])
        bin_count = int(params["bin_count"])
        min_stars = int(params["min_stars"])
        figout = params["figout"]

        signals.status.emit(f"Loading CSV: {csv_path.name}")
        df = pd.read_csv(csv_path)

        # check E(color) column
        if ecol_col not in df.columns:
            raise RuntimeError(f"Missing E(color) column: {ecol_col}")

        # compute distance_pc if necessary
        if dist_col in df.columns:
            df['distance_pc'] = df[dist_col].astype(float)
            signals.status.emit("Using provided distance column.")
        elif parallax_col in df.columns:
            par = pd.to_numeric(df[parallax_col], errors='coerce')
            # simple inversion for reasonable parallaxes
            mask_good = par > 0.5
            df = df[mask_good].copy()
            if df.empty:
                raise RuntimeError("No usable parallaxes after filtering (parallax > 0.5 mas).")
            df['distance_pc'] = 1000.0 / pd.to_numeric(df[parallax_col], errors='coerce')
            signals.status.emit(f"Computed distances from parallax for {len(df)} stars.")
        else:
            raise RuntimeError("Input must contain either a distance column or parallax column.")

        # footprint selection (circular)
        if use_circle:
            coords = SkyCoord(ra=df[ra_col].values * u.deg, dec=df[dec_col].values * u.deg)
            center = SkyCoord(center_ra * u.deg, center_dec * u.deg)
            sep = coords.separation(center).arcminute
            df = df[sep <= radius_arcmin].copy()
            signals.status.emit(f"Selected {len(df)} stars within {radius_arcmin} arcmin of ({center_ra},{center_dec})")

        # clean and cast
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[ecol_col, 'distance_pc'])
        df = df[(df['distance_pc'] > 0) & (df['distance_pc'] < 20000)]
        df[ecol_col] = df[ecol_col].astype(float)

        if len(df) == 0:
            raise RuntimeError("No stars after filtering.")

        signals.status.emit("Making scatter and running median plot...")

        # plotting (same logic as original script)
        plt.figure(figsize=(8,6))
        plt.scatter(df['distance_pc'], df[ecol_col], s=12, alpha=0.4, color='tab:blue', label='Stars')
        plt.xlabel("Distance (pc)")
        plt.ylabel(f"E(color)  ({ecol_col})")
        plt.title("E(color) vs Distance — Nebula footprint")

        dmin, dmax = df['distance_pc'].min(), df['distance_pc'].max()
        if dmax / dmin > 5:
            bins = np.logspace(np.log10(max(1,dmin)), np.log10(dmax), bin_count)
        else:
            bins = np.linspace(dmin, dmax, bin_count)

        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        med, _, _ = binned_statistic(df['distance_pc'], df[ecol_col], statistic='median', bins=bins)
        count, _, _ = binned_statistic(df['distance_pc'], df[ecol_col], statistic='count', bins=bins)
        pct16, _, _ = binned_statistic(df['distance_pc'], df[ecol_col],
                                       statistic=lambda x: np.percentile(x, 16) if len(x)>0 else np.nan, bins=bins)
        pct84, _, _ = binned_statistic(df['distance_pc'], df[ecol_col],
                                       statistic=lambda x: np.percentile(x, 84) if len(x)>0 else np.nan, bins=bins)

        mask_good = (count >= min_stars) & ~np.isnan(med)
        if mask_good.sum() > 0:
            plt.plot(bin_centers[mask_good], med[mask_good], color='red', lw=2, label='Binned median')
            plt.fill_between(bin_centers[mask_good], pct16[mask_good], pct84[mask_good], color='red', alpha=0.25, label='16-84%')
            if np.sum(mask_good) >= 5:
                smooth_med = uniform_filter1d(med[mask_good], size=2, mode='nearest')
                plt.plot(bin_centers[mask_good], smooth_med, color='darkred', ls='--', lw=1)

        plt.legend(loc='upper left')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(figout, dpi=200)
        plt.close()

        signals.status.emit(f"Saved plot to {figout}")
        signals.finished.emit(True, str(figout))

    except Exception as e:
        signals.finished.emit(False, str(e))


class EcolorGui(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("E(color) vs Distance — GUI")
        self.setMinimumSize(720, 520)
        self._build_ui()

    def _build_ui(self):
        layout = QGridLayout()
        row = 0

        layout.addWidget(QLabel("CSV file:"), row, 0)
        self.csv_edit = QLineEdit()
        layout.addWidget(self.csv_edit, row, 1)
        btn_csv = QPushButton("Browse")
        btn_csv.clicked.connect(self.browse_csv)
        layout.addWidget(btn_csv, row, 2)
        row += 1

        layout.addWidget(QLabel("E(color) column:"), row, 0)
        self.ecol_edit = QLineEdit("color_G-B")
        layout.addWidget(self.ecol_edit, row, 1)
        row += 1

        layout.addWidget(QLabel("Parallax column:"), row, 0)
        self.parallax_edit = QLineEdit("parallax")
        layout.addWidget(self.parallax_edit, row, 1)
        row += 1

        layout.addWidget(QLabel("Distance column (optional):"), row, 0)
        self.dist_edit = QLineEdit("distance_pc")
        layout.addWidget(self.dist_edit, row, 1)
        row += 1

        layout.addWidget(QLabel("RA column:"), row, 0)
        self.ra_edit = QLineEdit("ra_deg")
        layout.addWidget(self.ra_edit, row, 1)
        row += 1

        layout.addWidget(QLabel("Dec column:"), row, 0)
        self.dec_edit = QLineEdit("dec_deg")
        layout.addWidget(self.dec_edit, row, 1)
        row += 1

        # footprint controls
        self.use_circle_chk = QCheckBox("Use circular footprint")
        self.use_circle_chk.setChecked(True)
        layout.addWidget(self.use_circle_chk, row, 0)
        row += 1

        layout.addWidget(QLabel("Center RA (deg):"), row, 0)
        self.center_ra_edit = QLineEdit("315.0")
        layout.addWidget(self.center_ra_edit, row, 1)
        row += 1

        layout.addWidget(QLabel("Center Dec (deg):"), row, 0)
        self.center_dec_edit = QLineEdit("68.2")
        layout.addWidget(self.center_dec_edit, row, 1)
        row += 1

        layout.addWidget(QLabel("Radius (arcmin):"), row, 0)
        self.radius_edit = QLineEdit("15.0")
        layout.addWidget(self.radius_edit, row, 1)
        row += 1

        layout.addWidget(QLabel("Bin count:"), row, 0)
        self.bin_spin = QSpinBox()
        self.bin_spin.setRange(3, 200)
        self.bin_spin.setValue(30)
        layout.addWidget(self.bin_spin, row, 1)
        row += 1

        layout.addWidget(QLabel("Min stars per bin:"), row, 0)
        self.min_spin = QSpinBox()
        self.min_spin.setRange(1, 1000)
        self.min_spin.setValue(5)
        layout.addWidget(self.min_spin, row, 1)
        row += 1

        layout.addWidget(QLabel("Output figure filename:"), row, 0)
        self.fig_edit = QLineEdit("ecolor_vs_distance.png")
        layout.addWidget(self.fig_edit, row, 1)
        row += 1

        # run button and progress
        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self.run)
        layout.addWidget(self.run_btn, 0, 3, 2, 1)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # indeterminate while running
        self.progress.setVisible(False)
        layout.addWidget(self.progress, row, 0, 1, 3)
        row += 1

        # status log
        layout.addWidget(QLabel("Status / Log:"), row, 0)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log, row, 1, 4, 3)
        row += 4

        self.setLayout(layout)

    def browse_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV", str(Path.cwd()), "CSV files (*.csv);;All files (*)")
        if path:
            self.csv_edit.setText(path)

    def append_log(self, text):
        self.log.append(text)

    def run(self):
        csv_path = self.csv_edit.text().strip()
        if not csv_path:
            QMessageBox.warning(self, "Error", "Please select an input CSV.")
            return
        if not Path(csv_path).exists():
            QMessageBox.warning(self, "Error", "Input CSV not found.")
            return

        params = {
            "csv": csv_path,
            "ecol_col": self.ecol_edit.text().strip(),
            "parallax_col": self.parallax_edit.text().strip(),
            "dist_col": self.dist_edit.text().strip(),
            "ra_col": self.ra_edit.text().strip(),
            "dec_col": self.dec_edit.text().strip(),
            "use_circle": self.use_circle_chk.isChecked(),
            "center_ra": self.center_ra_edit.text().strip(),
            "center_dec": self.center_dec_edit.text().strip(),
            "radius_arcmin": self.radius_edit.text().strip(),
            "bin_count": self.bin_spin.value(),
            "min_stars": self.min_spin.value(),
            "figout": self.fig_edit.text().strip(),
        }

        self.run_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.append_log("Starting plot...")

        self.signals = WorkerSignals()
        self.signals.status.connect(lambda s: self.append_log(s))
        self.signals.finished.connect(self._finished)

        thread = threading.Thread(target=compute_and_plot, args=(params, self.signals), daemon=True)
        thread.start()

    def _finished(self, ok: bool, message: str):
        self.run_btn.setEnabled(True)
        self.progress.setVisible(False)
        if ok:
            self.append_log(f"Completed: {message}")
            QMessageBox.information(self, "Done", f"Plot saved: {message}")
            # open the saved file using system default viewer
            try:
                Path(message).resolve().open()  # simple attempt to open; may not work on all platforms
            except Exception:
                pass
        else:
            self.append_log(f"Error: {message}")
            QMessageBox.critical(self, "Error", message)

def main():
    app = QApplication(sys.argv)
    gui = EcolorGui()
    gui.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()