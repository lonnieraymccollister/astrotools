#!/usr/bin/env python3
"""
ecolor_vs_distance_pyvista.py

Interactive PyVista viewer for E(color) vs distance and 3D point-cloud visualization.

Usage:
  python ecolor_vs_distance_pyvista.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from astropy.coordinates import SkyCoord
import astropy.units as u

from PyQt6.QtWidgets import (
    QApplication, QWidget, QGridLayout, QLabel, QLineEdit, QPushButton, QFileDialog,
    QComboBox, QSpinBox, QSlider, QCheckBox, QMessageBox, QHBoxLayout, QDoubleSpinBox
)
from PyQt6.QtCore import Qt

# PyVista Qt interactor
import pyvista as pv
from pyvistaqt import QtInteractor

# ---------------- utilities ----------------
def spherical_to_cartesian(ra_deg, dec_deg, r_pc):
    """
    Convert RA, Dec in degrees and distance in pc to Cartesian (x,y,z) in pc.
    """
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    cosd = np.cos(dec)
    x = r_pc * cosd * np.cos(ra)
    y = r_pc * cosd * np.sin(ra)
    z = r_pc * np.sin(dec)
    return np.vstack([x, y, z]).T

# ---- voxelization helpers ----
def voxelize_points_to_grid(points, values, grid_size=(64,64,64), method='mean'):
    xmin, ymin, zmin = np.min(points, axis=0)
    xmax, ymax, zmax = np.max(points, axis=0)
    nx, ny, nz = grid_size
    xe = np.linspace(xmin, xmax, nx+1)
    ye = np.linspace(ymin, ymax, ny+1)
    ze = np.linspace(zmin, zmax, nz+1)
    xc = 0.5*(xe[:-1] + xe[1:])
    yc = 0.5*(ye[:-1] + ye[1:])
    zc = 0.5*(ze[:-1] + ze[1:])
    grid_vals = np.full((nz, ny, nx), np.nan, dtype=float)

    ix = np.clip(np.digitize(points[:,0], xe) - 1, 0, nx-1)
    iy = np.clip(np.digitize(points[:,1], ye) - 1, 0, ny-1)
    iz = np.clip(np.digitize(points[:,2], ze) - 1, 0, nz-1)

    from collections import defaultdict
    accum = defaultdict(list)
    for k, (i,j,l) in enumerate(zip(ix, iy, iz)):
        accum[(l,j,i)].append(values[k])

    for (l,j,i), arr in accum.items():
        if method == 'median':
            grid_vals[l,j,i] = np.nanmedian(arr)
        else:
            grid_vals[l,j,i] = np.nanmean(arr)

    return grid_vals, xmin, xmax, ymin, ymax, zmin, zmax, (xc, yc, zc)


def make_image_grid_from_voxels(grid_vals, xmin, xmax, ymin, ymax, zmin, zmax):
    if grid_vals.ndim != 3:
        raise ValueError("grid_vals must be a 3D array with shape (nz, ny, nx)")
    nz, ny, nx = grid_vals.shape
    grid = pv.ImageData()
    grid.dimensions = np.array([nx + 1, ny + 1, nz + 1], dtype=int)
    grid.origin = (float(xmin), float(ymin), float(zmin))
    dx = (float(xmax) - float(xmin)) / max(1, nx)
    dy = (float(ymax) - float(ymin)) / max(1, ny)
    dz = (float(zmax) - float(zmin)) / max(1, nz)
    grid.spacing = (dx, dy, dz)
    cell_flat = np.nan_to_num(grid_vals.ravel(order='F'), nan=np.nan).astype(np.float32)
    grid.cell_data["ext"] = cell_flat
    grid = grid.cell_data_to_point_data()
    return grid

# ---------------- GUI / Viewer ----------------
class PyVistaEcolorViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("E(color) vs Distance — PyVista Viewer")
        self.resize(1200, 800)
        self._build_ui()
        self.points = None
        self.values = None
        self.mesh = None
        self.grid = None
        self.cloud_actor = None

    def _build_ui(self):
        layout = QGridLayout()
        ctrl = QWidget()
        vbox = QGridLayout()

        row = 0
        vbox.addWidget(QLabel("CSV file:"), row, 0)
        self.csv_edit = QLineEdit()
        vbox.addWidget(self.csv_edit, row, 1)
        btn = QPushButton("Browse")
        btn.clicked.connect(self.browse_csv)
        vbox.addWidget(btn, row, 2)
        row += 1

        vbox.addWidget(QLabel("Ecolor column:"), row, 0)
        self.ecol_edit = QLineEdit("color_G-B")
        vbox.addWidget(self.ecol_edit, row, 1, 1, 2)
        row += 1

        vbox.addWidget(QLabel("RA col:"), row, 0)
        self.ra_edit = QLineEdit("ra_deg")
        vbox.addWidget(self.ra_edit, row, 1)
        vbox.addWidget(QLabel("Dec col:"), row, 2)
        self.dec_edit = QLineEdit("dec_deg")
        vbox.addWidget(self.dec_edit, row, 3)
        row += 1

        vbox.addWidget(QLabel("Dist col (pc):"), row, 0)
        self.dist_edit = QLineEdit("distance_pc")
        vbox.addWidget(self.dist_edit, row, 1)
        vbox.addWidget(QLabel("Parallax col (alt):"), row, 2)
        self.parallax_edit = QLineEdit("parallax")
        vbox.addWidget(self.parallax_edit, row, 3)
        row += 1

        vbox.addWidget(QLabel("Subsample (%)"), row, 0)
        self.sub_spin = QSpinBox()
        self.sub_spin.setRange(1,100)
        self.sub_spin.setValue(20)
        vbox.addWidget(self.sub_spin, row, 1)
        vbox.addWidget(QLabel("Point size"), row, 2)
        self.psize_spin = QSpinBox()
        self.psize_spin.setRange(1,30)
        self.psize_spin.setValue(6)
        vbox.addWidget(self.psize_spin, row, 3)
        row += 1

        # Distance filter controls
        vbox.addWidget(QLabel("Min distance (pc)"), row, 0)
        self.min_dist_spin = QSpinBox()
        self.min_dist_spin.setRange(0, 20000)
        self.min_dist_spin.setValue(0)
        vbox.addWidget(self.min_dist_spin, row, 1)
        vbox.addWidget(QLabel("Max distance (pc)"), row, 2)
        self.max_dist_spin = QSpinBox()
        self.max_dist_spin.setRange(1, 20000)
        self.max_dist_spin.setValue(20000)
        vbox.addWidget(self.max_dist_spin, row, 3)
        row += 1

        vbox.addWidget(QLabel("Colormap"), row, 0)
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(["magma","viridis","plasma","inferno","cividis"])
        vbox.addWidget(self.cmap_combo, row, 1)
        vbox.addWidget(QLabel("Opacity"), row, 2)
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(1,100)
        self.opacity_slider.setValue(90)
        vbox.addWidget(self.opacity_slider, row, 3)
        row += 1

        self.voxel_chk = QCheckBox("Show voxel grid")
        vbox.addWidget(self.voxel_chk, row, 0, 1, 2)
        vbox.addWidget(QLabel("Grid size"), row, 2)
        self.grid_size_spin = QSpinBox()
        self.grid_size_spin.setRange(16,256)
        self.grid_size_spin.setValue(64)
        vbox.addWidget(self.grid_size_spin, row, 3)
        row += 1

        # Isosurface controls: threshold, single add, suggest, and multi-add
        vbox.addWidget(QLabel("Isosurface threshold"), row, 0)
        self.iso_edit = QLineEdit("0.2")
        vbox.addWidget(self.iso_edit, row, 1)

        self.iso_btn = QPushButton("Add isosurface")
        self.iso_btn.clicked.connect(self.add_isosurface)
        vbox.addWidget(self.iso_btn, row, 2)

        self.suggest_btn = QPushButton("Suggest threshold (90th pct)")
        self.suggest_btn.clicked.connect(self.suggest_threshold)
        vbox.addWidget(self.suggest_btn, row, 3)

        row += 1

        # Multi-isosurface row
        vbox.addWidget(QLabel("Multi-isosurf percents (comma)"), row, 0)
        self.multi_edit = QLineEdit("75,85,95")
        vbox.addWidget(self.multi_edit, row, 1)
        self.multi_btn = QPushButton("Add multiple isosurfaces")
        self.multi_btn.clicked.connect(self.add_multiple_isosurfaces)
        vbox.addWidget(self.multi_btn, row, 2, 1, 2)

        row += 1

        btn_load = QPushButton("Load & Show Point Cloud")
        btn_load.clicked.connect(self.load_and_plot)
        vbox.addWidget(btn_load, row, 0, 1, 4)
        row += 1

        # Save subset CSV button (uses Min/Max)
        btn_save_subset = QPushButton("Save subset CSV (Min/Max)")
        btn_save_subset.clicked.connect(self.save_subset_csv)
        vbox.addWidget(btn_save_subset, row, 0, 1, 4)
        row += 1

        btn_plot = QPushButton("Plot binned median E vs Distance")
        btn_plot.clicked.connect(self.plot_binned)
        vbox.addWidget(btn_plot, row, 0, 1, 4)
        row += 1

        btn_export = QPushButton("Export voxel to .npy")
        btn_export.clicked.connect(self.export_voxel)
        vbox.addWidget(btn_export, row, 0, 1, 4)
        row += 1

        ctrl.setLayout(vbox)
        layout.addWidget(ctrl, 0, 0)

        # right: pyvista interactor
        self.plotter_widget = QtInteractor(self)
        layout.addWidget(self.plotter_widget.interactor, 0, 1)

        self.setLayout(layout)

    def browse_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV", str(Path.cwd()), "CSV files (*.csv);;All files (*)")
        if path:
            self.csv_edit.setText(path)

    def load_and_plot(self):
        path = self.csv_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "Error", "Choose a CSV first")
            return
        df = pd.read_csv(path)
        ecol_col = self.ecol_edit.text().strip()
        ra_col = self.ra_edit.text().strip()
        dec_col = self.dec_edit.text().strip()
        dist_col = self.dist_edit.text().strip()
        parallax_col = self.parallax_edit.text().strip()

        # distance handling: prefer dist_col, else compute from parallax
        if dist_col in df.columns and df[dist_col].notna().any():
            df['distance_pc'] = pd.to_numeric(df[dist_col], errors='coerce')
        elif parallax_col in df.columns and df[parallax_col].notna().any():
            p = pd.to_numeric(df[parallax_col], errors='coerce')
            with np.errstate(divide='ignore', invalid='ignore'):
                df = df[p > 0.1].copy()
                df['distance_pc'] = 1000.0 / pd.to_numeric(df[parallax_col], errors='coerce')
        else:
            QMessageBox.critical(self, "Error", "CSV must contain either a distance column or a parallax column")
            return

        for col in (ecol_col, ra_col, dec_col, 'distance_pc'):
            if col not in df.columns:
                QMessageBox.critical(self, "Error", f"Missing column: {col}")
                return

        # drop NaNs and sensible range filter
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[ecol_col, ra_col, dec_col, 'distance_pc'])
        df = df[(df['distance_pc'] > 0) & (df['distance_pc'] < 20000)]
        if df.empty:
            QMessageBox.critical(self, "Error", "No valid rows after filtering")
            return

        # Apply distance-range filter from GUI
        try:
            min_d = float(self.min_dist_spin.value())
            max_d = float(self.max_dist_spin.value())
            df = df[(df['distance_pc'] >= min_d) & (df['distance_pc'] <= max_d)]
        except Exception:
            pass

        if df.empty:
            QMessageBox.information(self, "Filtered", "No stars remain after applying the distance filter.")
            return

        # convert to arrays
        ra = df[ra_col].astype(float).to_numpy()
        dec = df[dec_col].astype(float).to_numpy()
        dist = df['distance_pc'].astype(float).to_numpy()
        vals = df[ecol_col].astype(float).to_numpy()

        # convert spherical -> cartesian
        pts = spherical_to_cartesian(ra, dec, dist)

        # subsample for interactive speed
        frac = max(1, int(100.0 / max(1, self.sub_spin.value())))
        idx = np.arange(len(pts))
        if frac > 1:
            rng = np.random.default_rng(12345)
            keep = rng.choice(idx, size=max(1, len(idx)//frac), replace=False)
            pts_plot = pts[keep]
            vals_plot = vals[keep]
        else:
            pts_plot = pts
            vals_plot = vals

        self.points = pts_plot
        self.values = vals_plot

        # clear previous
        self.plotter_widget.clear()
        p = self.plotter_widget

        # create pyvista point cloud
        cloud = pv.PolyData(self.points)
        cloud["ext"] = self.values

        cmap = self.cmap_combo.currentText()
        point_size = self.psize_spin.value()
        opacity = self.opacity_slider.value() / 100.0

        # add sphere glyphs for nicer appearance if many points small; else use simple add_mesh
        if len(self.points) < 50000:
            glyph = cloud.glyph(scale=False, geom=pv.Sphere(radius=point_size*0.02))
            self.cloud_actor = p.add_mesh(glyph, scalars="ext", cmap=cmap, opacity=opacity, show_scalar_bar=True)
        else:
            self.cloud_actor = p.add_mesh(cloud, scalars="ext", point_size=point_size, render_points_as_spheres=True,
                                          cmap=cmap, opacity=opacity, show_scalar_bar=True)

        # add axes and a small sun marker at origin
        p.add_axes()
        p.add_mesh(pv.Sphere(radius=max(1.0, np.std(dist)*0.01), center=(0,0,0)), color='yellow')
        p.reset_camera()
        self.plotter_widget.update()

        # if voxel checkbox, build voxel grid
        if self.voxel_chk.isChecked():
            self._build_and_show_grid()

    def _build_and_show_grid(self):
        if self.points is None or self.values is None:
            return
        size = self.grid_size_spin.value()
        grid_vals, xmin, xmax, ymin, ymax, zmin, zmax, centers = voxelize_points_to_grid(
            self.points, self.values, grid_size=(size, size, size), method='mean'
        )
        grid = make_image_grid_from_voxels(grid_vals, xmin, xmax, ymin, ymax, zmin, zmax)
        self.grid = grid

        try:
            vmin = float(np.nanpercentile(self.values, 2))
            vmax = float(np.nanpercentile(self.values, 98))
            clim = (vmin, vmax)
        except Exception:
            clim = None

        try:
            self.plotter_widget.clear()
        except Exception:
            pass

        cloud = pv.PolyData(self.points)
        cloud["ext"] = self.values
        cmap = self.cmap_combo.currentText()
        point_size = self.psize_spin.value()
        opacity = self.opacity_slider.value() / 100.0
        if len(self.points) < 50000:
            glyph = cloud.glyph(scale=False, geom=pv.Sphere(radius=point_size*0.02))
            self.plotter_widget.add_mesh(glyph, scalars="ext", cmap=cmap, opacity=opacity, show_scalar_bar=True)
        else:
            self.plotter_widget.add_mesh(cloud, scalars="ext", point_size=point_size,
                                         render_points_as_spheres=True, cmap=cmap, opacity=opacity, show_scalar_bar=True)

        try:
            if clim is not None:
                self.plotter_widget.add_volume(grid, scalars="ext", cmap=cmap, opacity="sigmoid", clim=clim)
            else:
                self.plotter_widget.add_volume(grid, scalars="ext", cmap=cmap, opacity="sigmoid")
        except Exception:
            try:
                iso_val = np.nanmean(self.values)
                cont = grid.contour([iso_val], scalars="ext")
                self.plotter_widget.add_mesh(cont, color="white", opacity=0.6)
            except Exception as e:
                QMessageBox.warning(self, "Volume/Contour warning", f"Could not add volume or contour: {e}")

        self.plotter_widget.add_axes()
        self.plotter_widget.add_mesh(pv.Sphere(radius=max(1.0, np.nanstd(self.values)*0.01), center=(0,0,0)), color='yellow')
        self.plotter_widget.reset_camera()
        self.plotter_widget.update()

    def add_isosurface(self):
        if self.grid is None:
            QMessageBox.information(self, "Info", "Voxel grid not built. Check 'Show voxel grid' and click Load & Show.")
            return
        try:
            thresh = float(self.iso_edit.text().strip())
        except Exception:
            QMessageBox.critical(self, "Error", "Invalid isosurface threshold")
            return
        try:
            cont = self.grid.contour([thresh], scalars="ext")
            self.plotter_widget.add_mesh(cont, color="white", opacity=0.6)
            self.plotter_widget.update()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Isosurface failed: {e}")

    def suggest_threshold(self):
        if self.values is None or len(self.values) == 0:
            QMessageBox.information(self, "Suggest threshold", "No data loaded to compute percentiles.")
            return
        try:
            pct = 90.0
            suggested = float(np.nanpercentile(self.values, pct))
            self.iso_edit.setText(f"{suggested:.4g}")
            QMessageBox.information(self, "Suggest threshold", f"Suggested {pct}th percentile = {suggested:.4g}")
        except Exception as e:
            QMessageBox.warning(self, "Suggest threshold", f"Could not compute percentile: {e}")

    def _color_opacity_for_levels(self, n):
        cmap_name = self.cmap_combo.currentText()
        try:
            import matplotlib.cm as cm
            cmap = cm.get_cmap(cmap_name)
            colors = [cmap(i / max(1, (n - 1)))[:3] for i in range(n)]
        except Exception:
            colors = [(1.0, 1.0, 1.0) for _ in range(n)]
        opacities = np.linspace(0.4, 0.9, n)[::-1].tolist()
        return colors, opacities

    def add_multiple_isosurfaces(self):
        if self.grid is None:
            QMessageBox.information(self, "Info", "Voxel grid not built. Build it first (check 'Show voxel grid' and Load & Show).")
            return
        if self.values is None or len(self.values) == 0:
            QMessageBox.information(self, "Info", "No scalar values loaded.")
            return
        txt = self.multi_edit.text().strip()
        if not txt:
            QMessageBox.warning(self, "Multi-isosurfaces", "Enter comma-separated percentiles, e.g., 70,85,95")
            return
        try:
            parts = [float(p.strip()) for p in txt.split(",") if p.strip() != ""]
            parts = sorted(max(0.0, min(100.0, p)) for p in parts)
            if len(parts) == 0:
                raise ValueError("No valid percentiles provided")
        except Exception as e:
            QMessageBox.critical(self, "Multi-isosurfaces", f"Invalid percentiles: {e}")
            return

        vals = np.asarray(self.values)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            QMessageBox.critical(self, "Multi-isosurfaces", "No finite scalar values to compute thresholds.")
            return

        thresholds = [float(np.nanpercentile(vals, p)) for p in parts]
        colors, opacities = self._color_opacity_for_levels(len(thresholds))

        for thr, col, opa in zip(thresholds, colors, opacities):
            try:
                cont = self.grid.contour([thr], scalars="ext")
                self.plotter_widget.add_mesh(cont, color=col, opacity=float(opa), show_scalar_bar=False)
            except Exception as e:
                QMessageBox.warning(self, "Multi-isosurfaces", f"Contour at {thr:.4g} failed: {e}")

        self.plotter_widget.update()
        QMessageBox.information(self, "Multi-isosurfaces", f"Added {len(thresholds)} isosurfaces at percentiles: {parts}")

    def plot_binned(self):
        path = self.csv_edit.text().strip()
        if not path:
            return
        df = pd.read_csv(path)
        ecol_col = self.ecol_edit.text().strip()
        dist_col = self.dist_edit.text().strip()
        parallax_col = self.parallax_edit.text().strip()

        if dist_col in df.columns:
            df['distance_pc'] = pd.to_numeric(df[dist_col], errors='coerce')
        elif parallax_col in df.columns:
            p = pd.to_numeric(df[parallax_col], errors='coerce')
            df = df[p > 0.1].copy()
            df['distance_pc'] = 1000.0 / p
        else:
            QMessageBox.critical(self, "Error", "CSV missing distance or parallax column")
            return

        # apply GUI distance filter
        try:
            min_d = float(self.min_dist_spin.value())
            max_d = float(self.max_dist_spin.value())
            df = df[(df['distance_pc'] >= min_d) & (df['distance_pc'] <= max_d)]
        except Exception:
            pass

        if ecol_col not in df.columns:
            QMessageBox.critical(self, "Error", f"Missing E(color) column: {ecol_col}")
            return

        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[ecol_col, 'distance_pc'])
        df = df[(df['distance_pc'] > 0) & (df['distance_pc'] < 20000)]
        if df.empty:
            QMessageBox.critical(self, "Error", "No valid rows to plot")
            return

        bin_count = 30
        dmin, dmax = df['distance_pc'].min(), df['distance_pc'].max()
        if dmax / max(dmin,1) > 5:
            bins = np.logspace(np.log10(max(1,dmin)), np.log10(dmax), bin_count)
        else:
            bins = np.linspace(dmin, dmax, bin_count)

        bin_centers = 0.5*(bins[:-1] + bins[1:])
        med, _, _ = binned_statistic(df['distance_pc'], df[ecol_col], statistic='median', bins=bins)
        count, _, _ = binned_statistic(df['distance_pc'], df[ecol_col], statistic='count', bins=bins)
        pct16, _, _ = binned_statistic(df['distance_pc'], df[ecol_col],
                                       statistic=lambda x: np.percentile(x, 16) if len(x) > 0 else np.nan, bins=bins)
        pct84, _, _ = binned_statistic(df['distance_pc'], df[ecol_col],
                                       statistic=lambda x: np.percentile(x, 84) if len(x) > 0 else np.nan, bins=bins)

        mask_good = (count >= 5) & ~np.isnan(med)
        plt.figure(figsize=(8,6))
        plt.scatter(df['distance_pc'], df[ecol_col], s=6, alpha=0.3, color='tab:blue')
        if mask_good.sum()>0:
            plt.plot(bin_centers[mask_good], med[mask_good], color='red', lw=2)
            plt.fill_between(bin_centers[mask_good], pct16[mask_good], pct84[mask_good], color='red', alpha=0.25)
        plt.xscale('log' if dmax/dmin>5 else 'linear')
        plt.xlabel("Distance (pc)")
        plt.ylabel(f"E(color) ({ecol_col})")
        plt.title("Binned median E(color) vs Distance")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    def save_subset_csv(self):
        """
        Save a CSV of the original CSV rows that fall inside the Min/Max distance spinboxes.
        Uses the CSV currently entered in the CSV file box.
        """
        path = self.csv_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "Save subset", "Choose a CSV file first")
            return

        try:
            df = pd.read_csv(path)
        except Exception as e:
            QMessageBox.critical(self, "Save subset", f"Could not read CSV: {e}")
            return

        # compute distance_pc if needed (prefer dist col, else parallax)
        dist_col = self.dist_edit.text().strip()
        parallax_col = self.parallax_edit.text().strip()
        if dist_col in df.columns:
            df['distance_pc'] = pd.to_numeric(df[dist_col], errors='coerce')
        elif parallax_col in df.columns:
            p = pd.to_numeric(df[parallax_col], errors='coerce')
            df = df[p > 0.1].copy()
            df['distance_pc'] = 1000.0 / p
        else:
            QMessageBox.critical(self, "Save subset", "CSV missing distance or parallax column")
            return

        # drop invalid rows
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['distance_pc'])
        if df.empty:
            QMessageBox.information(self, "Save subset", "No valid distance values found")
            return

        # read min/max from GUI and filter
        try:
            min_d = float(self.min_dist_spin.value())
            max_d = float(self.max_dist_spin.value())
        except Exception:
            QMessageBox.warning(self, "Save subset", "Invalid Min/Max distance values")
            return

        df_sub = df[(df['distance_pc'] >= min_d) & (df['distance_pc'] <= max_d)]
        if df_sub.empty:
            QMessageBox.information(self, "Save subset", "No rows in the selected distance range")
            return

        out_path, _ = QFileDialog.getSaveFileName(self, "Save subset CSV", str(Path.cwd()), "CSV files (*.csv);;All files (*)")
        if not out_path:
            return
        try:
            df_sub.to_csv(out_path, index=False)
            QMessageBox.information(self, "Save subset", f"Saved {len(df_sub)} rows to:\n{out_path}")
        except Exception as e:
            QMessageBox.critical(self, "Save subset", f"Failed to save CSV: {e}")

    def export_voxel(self):
        if self.grid is None:
            QMessageBox.information(self, "Info", "No voxel grid to export")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save voxel scalars as .npy", str(Path.cwd()), "NumPy files (*.npy);;All files (*)")
        if not path:
            return
        if hasattr(self.grid, "point_arrays"):
            arr = self.grid.point_arrays.get("ext")
        else:
            arr = self.grid.point_data["ext"]
        np.save(path, arr)
        QMessageBox.information(self, "Saved", f"Saved voxel scalars to {path}")

# ---------------- main ----------------
def main():
    app = QApplication(sys.argv)
    win = PyVistaEcolorViewer()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()