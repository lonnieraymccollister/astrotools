#!/usr/bin/env python3
"""
ebv_map_gui.py

PyQt6 GUI for interpolating scattered E(B-V)/E(color) sightlines to a 2D map with contours.
This version adds an optional Label column and a Show labels checkbox: non-blank labels
are drawn next to their stars only when the checkbox is enabled.
"""

import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from astropy.visualization import ImageNormalize, PercentileInterval, LinearStretch

from PyQt6.QtWidgets import (
    QApplication, QWidget, QGridLayout, QLabel, QLineEdit, QPushButton, QFileDialog,
    QTextEdit, QProgressBar, QSpinBox, QDoubleSpinBox, QCheckBox, QHBoxLayout, QVBoxLayout,
    QTableWidget, QTableWidgetItem, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject

# Worker signals
class WorkerSignals(QObject):
    status = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

def map_worker(params, signals: WorkerSignals):
    try:
        signals.status.emit("Loading CSV...")
        df = pd.read_csv(params["csv"])

        ra_col = params["ra_col"]
        dec_col = params["dec_col"]
        ebv_col = params["ebv_col"]

        if ra_col not in df.columns or dec_col not in df.columns or ebv_col not in df.columns:
            raise RuntimeError(f"CSV missing required columns: {ra_col}, {dec_col}, or {ebv_col}")

        ra = df[ra_col].values.astype(float)
        dec = df[dec_col].values.astype(float)
        ebv = df[ebv_col].values.astype(float)

        # build regular grid
        nside = int(params["nside"])
        signals.status.emit(f"Building grid {nside} x {nside} ...")
        ra_min, ra_max = ra.min(), ra.max()
        dec_min, dec_max = dec.min(), dec.max()
        grid_ra = np.linspace(ra_min, ra_max, nside)
        grid_dec = np.linspace(dec_max, dec_min, nside)  # descending for image
        RAg, DECg = np.meshgrid(grid_ra, grid_dec)

        signals.status.emit("Interpolating scattered data onto grid...")
        points = np.vstack([ra, dec]).T
        grid_ebv = griddata(points, ebv, (RAg, DECg), method=params["interp_method"])
        if np.any(np.isnan(grid_ebv)):
            # fill NaNs with nearest neighbor
            grid_near = griddata(points, ebv, (RAg, DECg), method="nearest")
            mask = np.isnan(grid_ebv)
            grid_ebv[mask] = grid_near[mask]

        # smoothing
        sigma = float(params["smooth_sigma"])
        if sigma > 0:
            signals.status.emit(f"Smoothing (sigma={sigma})...")
            grid_sm = gaussian_filter(grid_ebv, sigma=sigma)
        else:
            grid_sm = grid_ebv

        # plotting
        signals.status.emit("Rendering figure...")
        fig, ax = plt.subplots(1, 1, figsize=(9, 7))
        norm = ImageNormalize(grid_sm, interval=PercentileInterval(params["pct_interval"]), stretch=LinearStretch())
        im = ax.imshow(grid_sm, origin="upper", extent=(ra_min, ra_max, dec_min, dec_max),
                       cmap=params["cmap"], norm=norm, aspect="auto")
        cbar = plt.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label(params.get("cbar_label", "E(B-V) (mag)"))

        # contours: percentiles to actual values
        pmin = np.nanpercentile(grid_sm, params["contour_lo_pct"])
        pmax = np.nanpercentile(grid_sm, params["contour_hi_pct"])
        levels = np.linspace(pmin, pmax, params["ncontours"])
        cs = ax.contour(np.linspace(ra_min, ra_max, grid_sm.shape[1]),
                        np.linspace(dec_max, dec_min, grid_sm.shape[0]),
                        grid_sm, levels=levels, colors="white", linewidths=0.8)
        ax.clabel(cs, inline=1, fontsize=8, fmt="%.3f")

        # overlay input points (option) — with optional labels
        if params["plot_points"]:
            ax.scatter(ra, dec, s=params["point_size"], c="k", alpha=0.4, label="sightlines")
            # annotate labels if requested and enabled
            label_col = params.get("label_col", "")
            show_labels = bool(params.get("show_labels", True))
            if show_labels and label_col and label_col in df.columns:
                labels = df[label_col].astype(str).fillna("").values
                # offset scale (fraction of extent)
                dx_frac = 0.005
                dy_frac = 0.005
                dx = (ra_max - ra_min) * dx_frac if ra_max > ra_min else dx_frac
                dy = (dec_max - dec_min) * dy_frac if dec_max > dec_min else dy_frac
                for xi, yi, lab in zip(ra, dec, labels):
                    if str(lab).strip() != "":
                        ax.text(xi + dx, yi + dy, str(lab), fontsize=7, color="white",
                                bbox=dict(facecolor='black', alpha=0.5, pad=1, edgecolor='none'))
            ax.legend(loc="upper right")

        ax.set_xlabel("RA (deg)")
        ax.set_ylabel("Dec (deg)")
        ax.set_title(params.get("title", "Interpolated E(B-V) (scattered sightlines)"))
        plt.tight_layout()

        outpath = Path(params["outfig"])
        fig.savefig(str(outpath), dpi=params["dpi"])
        plt.close(fig)

        signals.status.emit(f"Saved map to {outpath}")
        signals.finished.emit(True, str(outpath))
    except Exception as e:
        signals.finished.emit(False, str(e))

class EbvMapGui(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("E(B-V) Map — GUI")
        self.setMinimumSize(900, 700)
        self._build_ui()

    def _build_ui(self):
        main = QVBoxLayout()
        grid = QGridLayout()

        row = 0
        grid.addWidget(QLabel("CSV file:"), row, 0)
        self.csv_edit = QLineEdit()
        grid.addWidget(self.csv_edit, row, 1, 1, 3)
        btn_csv = QPushButton("Browse")
        btn_csv.clicked.connect(self.browse_csv)
        grid.addWidget(btn_csv, row, 4)
        row += 1

        grid.addWidget(QLabel("RA column:"), row, 0)
        self.ra_edit = QLineEdit("ra_deg")
        grid.addWidget(self.ra_edit, row, 1)
        grid.addWidget(QLabel("Dec column:"), row, 2)
        self.dec_edit = QLineEdit("dec_deg")
        grid.addWidget(self.dec_edit, row, 3)
        row += 1

        grid.addWidget(QLabel("E(color) column:"), row, 0)
        self.ebv_edit = QLineEdit("color_G-B")
        grid.addWidget(self.ebv_edit, row, 1)

        # Label column (new)
        grid.addWidget(QLabel("Label column (optional):"), row, 2)
        self.label_edit = QLineEdit("label")
        grid.addWidget(self.label_edit, row, 3)
        row += 1

        grid.addWidget(QLabel("Show labels:"), row, 0)
        self.show_labels_chk = QCheckBox()
        self.show_labels_chk.setChecked(True)
        grid.addWidget(self.show_labels_chk, row, 1)

        grid.addWidget(QLabel("Interpolation:"), row, 2)
        self.interp_edit = QLineEdit("linear")
        grid.addWidget(self.interp_edit, row, 3)
        row += 1

        grid.addWidget(QLabel("Grid resolution (nside):"), row, 0)
        self.nside_spin = QSpinBox()
        self.nside_spin.setRange(50, 2000)
        self.nside_spin.setValue(300)
        grid.addWidget(self.nside_spin, row, 1)
        grid.addWidget(QLabel("Smooth sigma (pixels):"), row, 2)
        self.smooth_spin = QDoubleSpinBox()
        self.smooth_spin.setRange(0.0, 10.0)
        self.smooth_spin.setSingleStep(0.5)
        self.smooth_spin.setValue(1.0)
        grid.addWidget(self.smooth_spin, row, 3)
        row += 1

        grid.addWidget(QLabel("Contours percentiles low/high:"), row, 0)
        self.cont_lo_spin = QSpinBox(); self.cont_lo_spin.setRange(0,50); self.cont_lo_spin.setValue(10)
        grid.addWidget(self.cont_lo_spin, row, 1)
        self.cont_hi_spin = QSpinBox(); self.cont_hi_spin.setRange(50,100); self.cont_hi_spin.setValue(90)
        grid.addWidget(self.cont_hi_spin, row, 2)
        grid.addWidget(QLabel("Number of contours:"), row, 3)
        self.ncont_spin = QSpinBox(); self.ncont_spin.setRange(1,20); self.ncont_spin.setValue(7)
        grid.addWidget(self.ncont_spin, row, 4)
        row += 1

        grid.addWidget(QLabel("Plot input points:"), row, 0)
        self.plot_points_chk = QCheckBox(); self.plot_points_chk.setChecked(True)
        grid.addWidget(self.plot_points_chk, row, 1)
        grid.addWidget(QLabel("Point size:"), row, 2)
        self.point_size_spin = QSpinBox(); self.point_size_spin.setRange(1,50); self.point_size_spin.setValue(8)
        grid.addWidget(self.point_size_spin, row, 3)
        row += 1

        grid.addWidget(QLabel("Color map:"), row, 0)
        self.cmap_edit = QLineEdit("magma")
        grid.addWidget(self.cmap_edit, row, 1)
        grid.addWidget(QLabel("Percentile interval for stretch:"), row, 2)
        self.pct_interval_spin = QDoubleSpinBox(); self.pct_interval_spin.setRange(50, 100); self.pct_interval_spin.setValue(99.5)
        grid.addWidget(self.pct_interval_spin, row, 3)
        row += 1

        grid.addWidget(QLabel("Output figure filename:"), row, 0)
        self.out_edit = QLineEdit("ebv_scattered_map.png")
        grid.addWidget(self.out_edit, row, 1, 1, 3)
        btn_out = QPushButton("Browse")
        btn_out.clicked.connect(self.browse_out)
        grid.addWidget(btn_out, row, 4)
        row += 1

        # preview and run controls
        btn_preview = QPushButton("Load Preview (first 50 rows)")
        btn_preview.clicked.connect(self.load_preview)
        grid.addWidget(btn_preview, row, 0)
        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self.run)
        grid.addWidget(self.run_btn, row, 1)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        grid.addWidget(self.progress, row, 2, 1, 3)
        row += 1

        main.addLayout(grid)

        # preview table
        main.addWidget(QLabel("Preview (click header to autofill column names)"))
        self.preview_table = QTableWidget()
        main.addWidget(self.preview_table, stretch=1)

        # log
        main.addWidget(QLabel("Status / Log:"))
        self.log = QTextEdit(); self.log.setReadOnly(True)
        main.addWidget(self.log, stretch=1)

        self.setLayout(main)

        # enable drag-drop
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if not urls:
            return
        first = urls[0].toLocalFile()
        self.csv_edit.setText(first)
        self.append_log(f"Dropped file {first}")

    def browse_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV", str(Path.cwd()), "CSV files (*.csv);;All files (*)")
        if path:
            self.csv_edit.setText(path)

    def browse_out(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save figure as", str(Path.cwd()), "PNG files (*.png);;All files (*)")
        if path:
            self.out_edit.setText(path)

    def append_log(self, text):
        self.log.append(text)

    def load_preview(self):
        path = self.csv_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "No file", "Select a CSV file first.")
            return
        try:
            df = pd.read_csv(path, nrows=50)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed reading CSV: {e}")
            return
        self.preview_df = df
        self.preview_table.clear()
        self.preview_table.setColumnCount(len(df.columns))
        self.preview_table.setRowCount(len(df))
        self.preview_table.setHorizontalHeaderLabels(list(df.columns))
        for ci, col in enumerate(df.columns):
            for ri in range(len(df)):
                item = QTableWidgetItem(str(df.iloc[ri, ci]))
                item.setFlags(item.flags() ^ Qt.ItemFlag.ItemIsEditable)
                self.preview_table.setItem(ri, ci, item)
        self.append_log(f"Loaded preview ({len(df)} rows). Click a header cell to autofill RA/Dec/E(column/Label).")
        # connect header clicks
        self.preview_table.horizontalHeader().sectionClicked.connect(self.header_clicked)

    def header_clicked(self, index):
        header = self.preview_table.horizontalHeaderItem(index).text()
        # ask user where to map the header
        resp = QMessageBox.question(self, "Map column", f"Set '{header}' as RA column?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if resp == QMessageBox.StandardButton.Yes:
            self.ra_edit.setText(header); self.append_log(f"RA column set to {header}"); return
        resp = QMessageBox.question(self, "Map column", f"Set '{header}' as Dec column?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if resp == QMessageBox.StandardButton.Yes:
            self.dec_edit.setText(header); self.append_log(f"Dec column set to {header}"); return
        resp = QMessageBox.question(self, "Map column", f"Set '{header}' as E(color) column?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if resp == QMessageBox.StandardButton.Yes:
            self.ebv_edit.setText(header); self.append_log(f"E(color) column set to {header}"); return
        resp = QMessageBox.question(self, "Map column", f"Set '{header}' as Label column?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if resp == QMessageBox.StandardButton.Yes:
            self.label_edit.setText(header); self.append_log(f"Label column set to {header}"); return

    def run(self):
        csv_path = self.csv_edit.text().strip()
        if not csv_path:
            QMessageBox.warning(self, "No file", "Select a CSV file first.")
            return
        if not Path(csv_path).exists():
            QMessageBox.warning(self, "Missing", "CSV file not found.")
            return

        params = {
            "csv": csv_path,
            "ra_col": self.ra_edit.text().strip(),
            "dec_col": self.dec_edit.text().strip(),
            "ebv_col": self.ebv_edit.text().strip(),
            "label_col": self.label_edit.text().strip(),
            "show_labels": self.show_labels_chk.isChecked(),
            "nside": self.nside_spin.value(),
            "interp_method": self.interp_edit.text().strip(),
            "smooth_sigma": self.smooth_spin.value(),
            "contour_lo_pct": self.cont_lo_spin.value(),
            "contour_hi_pct": self.cont_hi_spin.value(),
            "ncontours": self.ncont_spin.value(),
            "plot_points": self.plot_points_chk.isChecked(),
            "point_size": self.point_size_spin.value(),
            "cmap": self.cmap_edit.text().strip(),
            "pct_interval": self.pct_interval_spin.value(),
            "outfig": self.out_edit.text().strip(),
            "title": "Interpolated E(B-V) (scattered sightlines)",
            "dpi": 200,
            "cbar_label": "E(B-V) (mag)"
        }

        self.run_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.append_log("Starting interpolation and plotting...")

        self.signals = WorkerSignals()
        self.signals.status.connect(lambda s: self.append_log(s))
        self.signals.finished.connect(self._finished)

        thread = threading.Thread(target=map_worker, args=(params, self.signals), daemon=True)
        thread.start()

    def _finished(self, ok: bool, message: str):
        self.run_btn.setEnabled(True)
        self.progress.setVisible(False)
        if ok:
            self.append_log(f"Completed: {message}")
            QMessageBox.information(self, "Done", f"Saved: {message}")
        else:
            self.append_log(f"Error: {message}")
            QMessageBox.critical(self, "Error", message)

def main():
    app = QApplication(sys.argv)
    gui = EbvMapGui()
    gui.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()