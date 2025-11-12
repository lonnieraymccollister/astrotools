#!/usr/bin/env python3
"""
ebv_map_gui.py

PyQt6 GUI for interpolating scattered E(B-V)/E(color) sightlines to a 2D map with contours.

Behavior:
 - Bottom x-axis shows decimal degrees.
 - Top x-axis shows RA in H:M by default (no seconds).
 - A checkbox enables H:M:S (seconds) when checked.
 - A spinbox controls the number of ticks used on both axes.
 - Two major plot lines are drawn at each axis extent.
 - Optional Label column and Show labels checkbox remain.
 - Place a highlighted star marker at a user-specified RA/Dec (degrees) and optional label.
 - New: checkbox to treat the marker RA/Dec entries as sexagesimal (RA: H:M:S, Dec: D:M:S)
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
from astropy.coordinates import SkyCoord
import astropy.units as u

from PyQt6.QtWidgets import (
    QApplication, QWidget, QGridLayout, QLabel, QLineEdit, QPushButton, QFileDialog,
    QTextEdit, QProgressBar, QSpinBox, QDoubleSpinBox, QCheckBox, QVBoxLayout,
    QTableWidget, QTableWidgetItem, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject

# Worker signals
class WorkerSignals(QObject):
    status = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

def _format_ra_hm_ticks(ticks_deg):
    sc = SkyCoord(ra=ticks_deg * u.deg, dec=np.zeros_like(ticks_deg) * u.deg, frame="icrs")
    hms = [sc.ra.to_string(unit=u.hour, sep=":", precision=0, pad=True) for _ in ticks_deg]
    return [s.rsplit(":", 1)[0] for s in hms]

def _format_ra_hms_ticks(ticks_deg):
    sc = SkyCoord(ra=ticks_deg * u.deg, dec=np.zeros_like(ticks_deg) * u.deg, frame="icrs")
    return [sc.ra.to_string(unit=u.hour, sep=":", precision=0, pad=True) for _ in ticks_deg]

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
        RAg, DECg = np.meshgrid(np.linspace(ra_min, ra_max, nside),
                                np.linspace(dec_max, dec_min, nside))  # descending for image

        signals.status.emit("Interpolating scattered data onto grid...")
        points = np.vstack([ra, dec]).T
        grid_ebv = griddata(points, ebv, (RAg, DECg), method=params["interp_method"])
        if np.any(np.isnan(grid_ebv)):
            grid_near = griddata(points, ebv, (RAg, DECg), method="nearest")
            mask = np.isnan(grid_ebv)
            grid_ebv[mask] = grid_near[mask]

        # smoothing
        sigma = float(params["smooth_sigma"])
        grid_sm = gaussian_filter(grid_ebv, sigma=sigma) if sigma > 0 else grid_ebv

        # plotting
        signals.status.emit("Rendering figure...")
        fig, ax = plt.subplots(1, 1, figsize=(9, 7))
        norm = ImageNormalize(grid_sm, interval=PercentileInterval(params["pct_interval"]), stretch=LinearStretch())
        im = ax.imshow(grid_sm, origin="upper", extent=(ra_min, ra_max, dec_min, dec_max),
                       cmap=params["cmap"], norm=norm, aspect="auto")
        cbar = plt.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label(params.get("cbar_label", "E(B-V) (mag)"))

        # contours
        pmin = np.nanpercentile(grid_sm, params["contour_lo_pct"])
        pmax = np.nanpercentile(grid_sm, params["contour_hi_pct"])
        levels = np.linspace(pmin, pmax, params["ncontours"])
        cs = ax.contour(np.linspace(ra_min, ra_max, grid_sm.shape[1]),
                        np.linspace(dec_max, dec_min, grid_sm.shape[0]),
                        grid_sm, levels=levels, colors="white", linewidths=0.8)
        ax.clabel(cs, inline=1, fontsize=8, fmt="%.3f")

        # overlay input points + labels
        if params["plot_points"]:
            ax.scatter(ra, dec, s=params["point_size"], c="k", alpha=0.4, label="sightlines")
            label_col = params.get("label_col", "")
            if params["show_labels"] and label_col and label_col in df.columns:
                labels = df[label_col].astype(str).fillna("").values
                dx_frac = 0.005
                dy_frac = 0.005
                dx = (ra_max - ra_min) * dx_frac if ra_max > ra_min else dx_frac
                dy = (dec_max - dec_min) * dy_frac if dec_max > dec_min else dy_frac
                for xi, yi, lab in zip(ra, dec, labels):
                    if str(lab).strip() != "":
                        ax.text(xi + dx, yi + dy, str(lab), fontsize=7, color="white",
                                bbox=dict(facecolor='black', alpha=0.5, pad=1, edgecolor='none'))
            ax.legend(loc="upper right")

        # optional placed marker (user-specified RA/Dec)
        if params.get("place_marker", False):
            mr = params.get("marker_ra", "").strip()
            md = params.get("marker_dec", "").strip()
            if mr != "" and md != "":
                try:
                    if params.get("marker_sexagesimal", False):
                        # parse RA as hours:minutes:seconds and Dec as degrees:minutes:seconds
                        try:
                            sc = SkyCoord(ra=mr, dec=md, frame="icrs")
                        except Exception:
                            sc = SkyCoord(mr, md, unit=(u.hourangle, u.deg), frame="icrs")
                        mra = sc.ra.deg
                        mdec = sc.dec.deg
                    else:
                        mra = float(mr)
                        mdec = float(md)

                    ax.scatter([mra], [mdec], s=max(80, params.get("point_size", 8) * 6),
                               marker="*", c="yellow", edgecolors="black", linewidths=0.6, zorder=10)
                    mlabel = params.get("marker_label", "").strip()
                    if mlabel != "":
                        dx = (ra_max - ra_min) * 0.01 if ra_max > ra_min else 0.01
                        dy = (dec_max - dec_min) * 0.01 if dec_max > dec_min else 0.01
                        ax.text(mra + dx, mdec + dy, mlabel, fontsize=9, color="yellow",
                                bbox=dict(facecolor='black', alpha=0.6, pad=1, edgecolor='none'), zorder=11)
                    signals.status.emit(f"Placed marker at RA={mra:.6f} deg, Dec={mdec:.6f} deg")
                except ValueError:
                    signals.status.emit("Invalid numeric marker RA/Dec (must be numeric degrees). Marker not placed.")
                except Exception as e:
                    signals.status.emit(f"Failed parsing sexagesimal marker coordinates: {e}. Marker not placed.")

        # two major plot lines per axis (at extents)
        ax.axvline(ra_min, color="cyan", linewidth=1.4, alpha=0.9)
        ax.axvline(ra_max, color="cyan", linewidth=1.4, alpha=0.9)
        ax.axhline(dec_min, color="cyan", linewidth=1.4, alpha=0.9)
        ax.axhline(dec_max, color="cyan", linewidth=1.4, alpha=0.9)

        # choose number of ticks (user-controlled)
        n_ticks = int(params.get("n_ticks", 4))
        if n_ticks < 2:
            n_ticks = 2
        xticks = np.linspace(ra_min, ra_max, n_ticks)

        # Bottom axis: decimal degrees
        ax.set_xticks(xticks)
        deg_labels = [f"{x:.3f}" for x in xticks]
        ax.set_xticklabels(deg_labels, rotation=0, fontsize=8)
        ax.set_xlabel("RA (deg)")

        # Top axis: H:M (default) or H:M:S if checkbox selected
        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        ax_top.set_xticks(xticks)
        use_hms = bool(params.get("use_hms", False))
        try:
            if use_hms:
                hms_labels = _format_ra_hms_ticks(xticks)  # H:M:S
                ax_top.set_xticklabels(hms_labels, rotation=0, fontsize=8)
                ax_top.set_xlabel("RA (H:M:S)")
            else:
                hm_labels = _format_ra_hm_ticks(xticks)  # H:M (no seconds)
                ax_top.set_xticklabels(hm_labels, rotation=0, fontsize=8)
                ax_top.set_xlabel("RA (H:M)")
        except Exception:
            def deg_to_hm(d):
                hours = (d / 15.0) % 24.0
                h = int(hours)
                m = int((hours - h) * 60.0)
                return f"{h:02d}:{m:02d}"
            def deg_to_hms(d):
                hours = (d / 15.0) % 24.0
                h = int(hours)
                m = int((hours - h) * 60.0)
                s = int(((hours - h) * 60.0 - m) * 60.0)
                return f"{h:02d}:{m:02d}:{s:02d}"
            if use_hms:
                ax_top.set_xticklabels([deg_to_hms(x) for x in xticks], rotation=0, fontsize=8)
                ax_top.set_xlabel("RA (H:M:S)")
            else:
                ax_top.set_xticklabels([deg_to_hm(x) for x in xticks], rotation=0, fontsize=8)
                ax_top.set_xlabel("RA (H:M)")

        ax_top.xaxis.set_ticks_position('top')
        ax_top.xaxis.set_label_position('top')
        ax_top.tick_params(axis='x', which='major', pad=6)

        ax.set_ylabel("Dec (deg)")
        ax.set_title(params.get("title", "Interpolated E(B-V) (scattered sightlines)"))
        ax.grid(which='major', color='gray', linestyle='--', linewidth=0.5, alpha=0.6)
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
        self.setMinimumSize(950, 780)
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

        grid.addWidget(QLabel("RA column (deg):"), row, 0)
        self.ra_edit = QLineEdit("ra_deg")
        grid.addWidget(self.ra_edit, row, 1)
        grid.addWidget(QLabel("Dec column (deg):"), row, 2)
        self.dec_edit = QLineEdit("dec_deg")
        grid.addWidget(self.dec_edit, row, 3)
        row += 1

        grid.addWidget(QLabel("E(color) column:"), row, 0)
        self.ebv_edit = QLineEdit("color_G-B")
        grid.addWidget(self.ebv_edit, row, 1)

        # Label column (optional)
        grid.addWidget(QLabel("Label column (optional):"), row, 2)
        self.label_edit = QLineEdit("label")
        grid.addWidget(self.label_edit, row, 3)
        row += 1

        grid.addWidget(QLabel("Show labels:"), row, 0)
        self.show_labels_chk = QCheckBox()
        self.show_labels_chk.setChecked(True)
        grid.addWidget(self.show_labels_chk, row, 1)

        # H:M:S checkbox and tick count
        grid.addWidget(QLabel("Show seconds in top RA?"), row, 2)
        self.hms_chk = QCheckBox()
        self.hms_chk.setChecked(False)  # default H:M (no seconds)
        grid.addWidget(self.hms_chk, row, 3)
        row += 1

        grid.addWidget(QLabel("Number of ticks (top & bottom):"), row, 0)
        self.nticks_spin = QSpinBox()
        self.nticks_spin.setRange(2, 12)
        self.nticks_spin.setValue(4)
        grid.addWidget(self.nticks_spin, row, 1)

        grid.addWidget(QLabel("Interpolation:"), row, 2)
        self.interp_edit = QLineEdit("linear")
        grid.addWidget(self.interp_edit, row, 3)
        row += 1

        # Place marker UI
        grid.addWidget(QLabel("Place star marker:"), row, 0)
        self.place_marker_chk = QCheckBox()
        self.place_marker_chk.setChecked(False)
        grid.addWidget(self.place_marker_chk, row, 1)

        grid.addWidget(QLabel("Marker RA (deg or H:M:S):"), row, 2)
        self.marker_ra_edit = QLineEdit("")   # user types RA in decimal degrees or H:M:S
        grid.addWidget(self.marker_ra_edit, row, 3)
        row += 1

        grid.addWidget(QLabel("Marker Dec (deg or D:M:S):"), row, 0)
        self.marker_dec_edit = QLineEdit("")  # user types Dec in decimal degrees or D:M:S
        grid.addWidget(self.marker_dec_edit, row, 1)

        grid.addWidget(QLabel("Marker coords are sexagesimal?"), row, 2)
        self.marker_sex_chk = QCheckBox()
        self.marker_sex_chk.setChecked(False)   # default: decimal degrees
        grid.addWidget(self.marker_sex_chk, row, 3)
        row += 1

        grid.addWidget(QLabel("Marker label (optional):"), row, 0)
        self.marker_label_edit = QLineEdit("")
        grid.addWidget(self.marker_label_edit, row, 1)
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
        self.preview_table.horizontalHeader().sectionClicked.connect(self.header_clicked)

    def header_clicked(self, index):
        header = self.preview_table.horizontalHeaderItem(index).text()
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
            "use_hms": self.hms_chk.isChecked(),
            "n_ticks": self.nticks_spin.value(),
            "place_marker": self.place_marker_chk.isChecked(),
            "marker_ra": self.marker_ra_edit.text().strip(),
            "marker_dec": self.marker_dec_edit.text().strip(),
            "marker_label": self.marker_label_edit.text().strip(),
            "marker_sexagesimal": self.marker_sex_chk.isChecked(),
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