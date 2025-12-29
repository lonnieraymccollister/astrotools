#!/usr/bin/env python3
"""
deskew_book_gui.py

PyQt6 GUI to load a color FITS, select four corner points on a preview,
warp the perspective to a rectangular front-facing view using OpenCV, and save
the result as a FITS file while preserving channel layout and header.

Enhancements:
 - Safe QImage creation and preview buffer lifetime handling
 - Scaled-to-fit preview with accurate click mapping
 - Buttons:
     * Expand to Edges (creates points 5..8 by projecting 1..4 to image edges)
     * Find Center (computes center of polygon 1..4 and labels as point 5; draws rays)
     * Select 6-9 (enter mode to pick four new points 6..9)
     * Warp Save 2 (warp using points 5..8) [if present]
     * Warp Save 3 (warp using points 6..9) [if present]
 - Visuals: points 1..4 red, points 5..8 orange, center point 5 orange filled, points 6..9 blue
"""
import sys
import os
import subprocess
import shutil
from pathlib import Path

import numpy as np
import cv2
from astropy.io import fits

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGridLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QCheckBox, QMessageBox, QSpinBox, QHBoxLayout
)
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QBrush
from PyQt6.QtCore import Qt, QPoint


# -------------------------
# Utilities
# -------------------------
def load_fits_preserve_header(path):
    """Load primary HDU data and header. Return (arr_yxc, header, layout_info).
    arr_yxc is (Y, X, 3) or (Y, X) expanded to 3 channels, dtype float64.
    layout_info: {'channel_first': bool, 'orig_shape': tuple}
    """
    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data
        header = hdul[0].header

    if data is None:
        raise ValueError("FITS contains no primary data")

    arr = np.array(data)
    layout_info = {"orig_shape": arr.shape, "channel_first": False}

    if arr.ndim == 3:
        if arr.shape[0] == 3:
            arr = np.transpose(arr, (1, 2, 0))
            layout_info["channel_first"] = True
        elif arr.shape[2] == 3:
            layout_info["channel_first"] = False
        else:
            if arr.shape[0] == 3:
                arr = np.transpose(arr, (1, 2, 0))
                layout_info["channel_first"] = True
            else:
                raise ValueError(f"Unsupported 3D FITS shape: {arr.shape}")
    elif arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
        layout_info["channel_first"] = False
    else:
        raise ValueError(f"Unsupported FITS dimensionality: {arr.shape}")

    arr = arr.astype(np.float64, copy=False)
    return arr, header, layout_info


def write_fits_preserve_layout(path, arr_yxc, header, layout_info):
    """Write arr_yxc back to FITS preserving original layout."""
    out = np.array(arr_yxc, copy=False)
    if layout_info.get("channel_first", False):
        if out.ndim == 3 and out.shape[2] == 3:
            out_write = np.transpose(out, (2, 0, 1))
        elif out.ndim == 2:
            out_write = out[np.newaxis, :, :]
        else:
            out_write = out
    else:
        out_write = out

    out_write = np.ascontiguousarray(out_write.astype(np.float32, copy=False))

    hdr = header.copy() if header is not None else fits.Header()
    try:
        hdr['BITPIX'] = -32
    except Exception:
        pass

    fits.writeto(path, out_write, header=hdr, overwrite=True)


def _check_normalized_0_1(arr, tol=1e-8):
    if arr is None or arr.size == 0:
        return False
    amin = float(np.nanmin(arr))
    amax = float(np.nanmax(arr))
    if np.isnan(amin) or np.isnan(amax):
        return False
    return (abs(amin - 0.0) <= tol) and (abs(amax - 1.0) <= tol)


def _launch_normalize_gui(filepath):
    try:
        script = Path("normalize_gui.py")
        if not script.exists():
            script = Path(__file__).resolve().parent / "normalize_gui.py"
        if not script.exists():
            return False
        subprocess.Popen([sys.executable, str(script), str(filepath)])
        return True
    except Exception:
        return False


def _maybe_launch_siril(output_path):
    try:
        siril_exe = shutil.which("siril")
        if siril_exe:
            subprocess.Popen([siril_exe, os.path.abspath(output_path)])
            return True, "Siril launched."
        else:
            return False, "Siril executable not found in PATH."
    except Exception as e:
        return False, f"Failed to launch Siril: {e}"


def _prepare_preview_array_for_qimage(arr):
    """
    Ensure arr is HxWx3 uint8, contiguous, and return (arr_uint8, bytes_per_line).
    Uses width*channels for bytes_per_line to avoid stride surprises.
    """
    if arr is None:
        raise ValueError("No array provided for preview")

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[..., :3]

    if np.issubdtype(arr.dtype, np.floating):
        a = np.nan_to_num(arr)
        mn = float(np.nanmin(a))
        mx = float(np.nanmax(a))
        if mx > mn:
            a = (a - mn) / (mx - mn) * 255.0
        else:
            a = np.zeros_like(a)
        arr8 = a.astype(np.uint8)
    else:
        arr8 = arr.astype(np.uint8)

    arr8 = np.ascontiguousarray(arr8)

    h, w = arr8.shape[:2]
    channels = arr8.shape[2] if arr8.ndim == 3 else 1
    bytes_per_line = int(w * channels)

    return arr8, bytes_per_line


# -------------------------
# Clickable preview widget (scaled-to-fit, maps clicks back to original coords)
# -------------------------
class ClickablePreview(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._pix_orig = None
        self._pix_display = None
        self._preview_buf = None

        # points
        self.points = []         # 1..4 (user-selected)
        self.center_point = None  # point 5 (center of polygon)
        self.edge_rays = []      # list of (ix,iy) intersections for rays through 1..4
        self.extra_points = []   # 6..9 (user-selected later)

        self.max_points = 4

        # geometry mapping
        self._scale = 1.0
        self._offset_x = 0
        self._offset_y = 0
        self._disp_w = 0
        self._disp_h = 0
        self._orig_w = 0
        self._orig_h = 0

        # selection mode: None or "extra" when selecting points 6..9
        self.select_mode = None

    def setPixmap(self, pixmap: QPixmap, buf=None):
        self._pix_orig = pixmap
        if buf is not None:
            self._preview_buf = buf

        if self._pix_orig is not None:
            self._orig_w = self._pix_orig.width()
            self._orig_h = self._pix_orig.height()
        else:
            self._orig_w = self._orig_h = 0

        self._update_display_pixmap()
        # do not clear center/extra automatically; keep them until user resets
        self.update()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._update_display_pixmap()

    def _update_display_pixmap(self):
        if self._pix_orig is None:
            super().setPixmap(QPixmap())
            self._pix_display = None
            self._scale = 1.0
            self._offset_x = self._offset_y = 0
            self._disp_w = self._disp_h = 0
            return

        label_w = max(1, self.width())
        label_h = max(1, self.height())
        pm_w = self._pix_orig.width()
        pm_h = self._pix_orig.height()

        self._scale = min(label_w / pm_w, label_h / pm_h)
        self._disp_w = int(round(pm_w * self._scale))
        self._disp_h = int(round(pm_h * self._scale))
        self._offset_x = (label_w - self._disp_w) // 2
        self._offset_y = (label_h - self._disp_h) // 2

        self._pix_display = self._pix_orig.scaled(self._disp_w, self._disp_h,
                                                  Qt.AspectRatioMode.KeepAspectRatio,
                                                  Qt.TransformationMode.SmoothTransformation)
        super().setPixmap(self._pix_display)

    def mousePressEvent(self, ev):
        if self._pix_display is None:
            return

        x = ev.position().x()
        y = ev.position().y()

        if not (self._offset_x <= x < self._offset_x + self._disp_w and
                self._offset_y <= y < self._offset_y + self._disp_h):
            return

        disp_x = x - self._offset_x
        disp_y = y - self._offset_y

        if self._scale > 0:
            img_x = int(round(disp_x / self._scale))
            img_y = int(round(disp_y / self._scale))
        else:
            img_x = int(disp_x)
            img_y = int(disp_y)

        img_x = max(0, min(self._orig_w - 1, img_x))
        img_y = max(0, min(self._orig_h - 1, img_y))

        if self.select_mode == "extra":
            # selecting points 6..9
            if len(self.extra_points) < 4:
                self.extra_points.append(QPoint(img_x, img_y))
            else:
                # reset extra selection
                self.extra_points = [QPoint(img_x, img_y)]
            # exit mode automatically when 4 selected
            if len(self.extra_points) >= 4:
                self.select_mode = None
        else:
            # selecting points 1..4
            if len(self.points) < self.max_points:
                self.points.append(QPoint(img_x, img_y))
            else:
                # reset selection and clear extras/center
                self.points = [QPoint(img_x, img_y)]
                self.center_point = None
                self.edge_rays = []
                self.extra_points = []
        self.update()

    def paintEvent(self, ev):
        super().paintEvent(ev)
        if self._pix_display is None:
            return

        painter = QPainter(self)

        # draw points 1..4 in red
        pen_red = QPen(QColor(255, 0, 0), 3)
        painter.setPen(pen_red)
        for i, pt in enumerate(self.points):
            dx = self._offset_x + int(round(pt.x() * self._scale))
            dy = self._offset_y + int(round(pt.y() * self._scale))
            painter.drawEllipse(dx - 5, dy - 5, 10, 10)
            painter.drawText(dx + 6, dy - 6, str(i + 1))

        # draw center point (5) and rays in orange if present
        if self.center_point is not None:
            pen_orange = QPen(QColor(255, 165, 0), 2)
            brush_orange = QBrush(QColor(255, 165, 0))
            painter.setPen(pen_orange)
            painter.setBrush(brush_orange)
            cdx = self._offset_x + int(round(self.center_point.x() * self._scale))
            cdy = self._offset_y + int(round(self.center_point.y() * self._scale))
            painter.drawEllipse(cdx - 6, cdy - 6, 12, 12)
            painter.drawText(cdx + 8, cdy - 8, "5")
            # draw rays to edges through points 1..4
            pen_ray = QPen(QColor(255, 165, 0), 1, Qt.PenStyle.DashLine)
            painter.setPen(pen_ray)
            for ray_pt in self.edge_rays:
                rx = self._offset_x + int(round(ray_pt[0] * self._scale))
                ry = self._offset_y + int(round(ray_pt[1] * self._scale))
                painter.drawLine(cdx, cdy, rx, ry)

        # draw expanded points 5..8 (if any) in orange (we keep compatibility with earlier expand_to_edges)
        # note: if center_point exists we already label it as 5; here extra edge points (if created by expand_to_edges) are drawn as 5..8
        if hasattr(self, "edge_points") and self.edge_points:
            pen_orange2 = QPen(QColor(255, 165, 0), 3)
            painter.setPen(pen_orange2)
            for j, pt in enumerate(self.edge_points):
                dx = self._offset_x + int(round(pt.x() * self._scale))
                dy = self._offset_y + int(round(pt.y() * self._scale))
                painter.drawEllipse(dx - 5, dy - 5, 10, 10)
                painter.drawText(dx + 6, dy - 6, str(5 + j))

        # draw extra points 6..9 in blue
        pen_blue = QPen(QColor(0, 120, 255), 3)
        painter.setPen(pen_blue)
        for k, pt in enumerate(self.extra_points):
            dx = self._offset_x + int(round(pt.x() * self._scale))
            dy = self._offset_y + int(round(pt.y() * self._scale))
            painter.drawEllipse(dx - 5, dy - 5, 10, 10)
            painter.drawText(dx + 6, dy - 6, str(6 + k))

        painter.end()


# -------------------------
# Main Window
# -------------------------
class DeskewWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Book Perspective Deskew (FITS)")
        self.resize(1100, 820)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QGridLayout(central)

        # Input file
        layout.addWidget(QLabel("Input FITS:"), 0, 0)
        self.input_edit = QLineEdit()
        layout.addWidget(self.input_edit, 0, 1)
        btn_in = QPushButton("Browse")
        btn_in.clicked.connect(self.browse_input)
        layout.addWidget(btn_in, 0, 2)

        # Output file
        layout.addWidget(QLabel("Output FITS:"), 1, 0)
        self.output_edit = QLineEdit()
        layout.addWidget(self.output_edit, 1, 1)
        btn_out = QPushButton("Browse")
        btn_out.clicked.connect(self.browse_output)
        layout.addWidget(btn_out, 1, 2)

        # Desired output width/height
        layout.addWidget(QLabel("Output width (px):"), 2, 0)
        self.out_w_spin = QSpinBox(); self.out_w_spin.setRange(100, 10000); self.out_w_spin.setValue(1200)
        layout.addWidget(self.out_w_spin, 2, 1)
        layout.addWidget(QLabel("Output height (px):"), 2, 2)
        self.out_h_spin = QSpinBox(); self.out_h_spin.setRange(100, 10000); self.out_h_spin.setValue(1600)
        layout.addWidget(self.out_h_spin, 2, 3)

        # Checkboxes
        self.chk_normalized = QCheckBox("Check input normalized to [0,1]")
        self.chk_normalized.setChecked(True)
        layout.addWidget(self.chk_normalized, 3, 0, 1, 2)
        self.chk_siril = QCheckBox("Offer to open result in Siril after save")
        self.chk_siril.setChecked(True)
        layout.addWidget(self.chk_siril, 3, 2, 1, 2)

        # Preview area
        self.preview = ClickablePreview()
        self.preview.setFixedSize(880, 620)
        layout.addWidget(self.preview, 4, 0, 1, 4)

        # Buttons row
        btn_row = QHBoxLayout()
        self.load_btn = QPushButton("Load & Show Preview")
        self.load_btn.clicked.connect(self.load_preview)
        btn_row.addWidget(self.load_btn)

        self.find_center_btn = QPushButton("Find Center (point 5)")
        self.find_center_btn.clicked.connect(self.find_center)
        btn_row.addWidget(self.find_center_btn)

        self.select_extra_btn = QPushButton("Select points 6..9")
        self.select_extra_btn.clicked.connect(self.start_select_extra)
        btn_row.addWidget(self.select_extra_btn)

        self.expand_btn = QPushButton("Expand to Edges (5..8)")
        self.expand_btn.clicked.connect(self.expand_to_edges)
        btn_row.addWidget(self.expand_btn)

        self.preview_warp_btn = QPushButton("Preview Warp (1..4)")
        self.preview_warp_btn.clicked.connect(self.preview_warp)
        btn_row.addWidget(self.preview_warp_btn)

        self.warp_btn = QPushButton("Warp & Save (1..4)")
        self.warp_btn.clicked.connect(self.warp_and_save)
        btn_row.addWidget(self.warp_btn)

        self.warp2_btn = QPushButton("Warp Save 2 (5..8)")
        self.warp2_btn.clicked.connect(self.warp_and_save2)
        btn_row.addWidget(self.warp2_btn)

        self.warp3_btn = QPushButton("Warp Save 3 (6..9)")
        self.warp3_btn.clicked.connect(self.warp_and_save3)
        btn_row.addWidget(self.warp3_btn)

        layout.addLayout(btn_row, 5, 0, 1, 4)

        self.info_label = QLabel("Click 4 corners on the preview: TL, TR, BR, BL. Click again to reset.")
        layout.addWidget(self.info_label, 6, 0, 1, 4)

        self.status = QLabel("")
        layout.addWidget(self.status, 7, 0, 1, 4)

        # internal state
        self.loaded_arr = None
        self.loaded_hdr = None
        self.layout_info = None
        self.display_uint8 = None
        self.last_warp_result = None
        self.last_warp_preview_buf = None

    def browse_input(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Open FITS", "", "FITS Files (*.fits *.fit);;All Files (*)")
        if fn:
            self.input_edit.setText(fn)

    def browse_output(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Save FITS", "", "FITS Files (*.fits *.fit);;All Files (*)")
        if fn:
            self.output_edit.setText(fn)

    def load_preview(self):
        path = self.input_edit.text().strip()
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "Input required", "Select an existing input FITS file first.")
            return
        try:
            arr, hdr, info = load_fits_preserve_header(path)
            self.loaded_arr = arr
            self.loaded_hdr = hdr
            self.layout_info = info

            if self.chk_normalized.isChecked():
                if not _check_normalized_0_1(self.loaded_arr):
                    launched = _launch_normalize_gui(path)
                    if launched:
                        QMessageBox.information(self, "Normalization required", "Input not normalized to [0,1]. Launched normalize_gui.py for the input file. Please normalize and re-run.")
                    else:
                        QMessageBox.information(self, "Normalization required", "Input not normalized to [0,1] and normalize_gui.py was not found.")
                    return

            img = np.nan_to_num(self.loaded_arr)
            mn = float(np.nanmin(img))
            mx = float(np.nanmax(img))
            if mx > mn:
                img8 = ((img - mn) / (mx - mn) * 255.0).astype(np.uint8)
            else:
                img8 = np.zeros_like(img, dtype=np.uint8)

            if img8.ndim == 2:
                img8 = np.stack([img8]*3, axis=-1)
            elif img8.shape[2] > 3:
                img8 = img8[:, :, :3]

            img8_prepared, bytes_per_line = _prepare_preview_array_for_qimage(img8)
            self.display_uint8 = img8_prepared

            h, w = img8_prepared.shape[:2]
            qimg = QImage(img8_prepared.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            qimg = qimg.copy()
            pix = QPixmap.fromImage(qimg)
            self.preview.setPixmap(pix, buf=img8_prepared)

            # reset selection state
            self.preview.points = []
            self.preview.center_point = None
            self.preview.edge_rays = []
            self.preview.edge_points = []
            self.preview.extra_points = []
            self.preview.select_mode = None

            self.status.setText(f"Loaded preview {w}x{h}")
            self.last_warp_result = None
            self.last_warp_preview_buf = None
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))

    def find_center(self):
        """
        Compute centroid of polygon defined by points 1..4 and set as center_point (5).
        Also compute ray intersections with image edges through each point 1..4 and store in edge_rays.
        """
        if self.loaded_arr is None:
            QMessageBox.warning(self, "No input", "Load an input FITS and preview first.")
            return
        pts = self.preview.points
        if len(pts) != 4:
            QMessageBox.warning(self, "Points required", "Click exactly 4 points first (TL, TR, BR, BL).")
            return

        # compute centroid (polygon centroid)
        xs = [p.x() for p in pts]
        ys = [p.y() for p in pts]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        self.preview.center_point = QPoint(int(round(cx)), int(round(cy)))

        # compute ray intersections with image border for each point
        h, w = self.loaded_arr.shape[:2]
        edge_pts = []
        rays = []
        for p in pts:
            px = float(p.x())
            py = float(p.y())
            vx = px - cx
            vy = py - cy
            if abs(vx) < 1e-9 and abs(vy) < 1e-9:
                # degenerate: use point itself
                ix, iy = px, py
                rays.append((ix, iy))
                edge_pts.append(QPoint(int(round(ix)), int(round(iy))))
                continue

            t_candidates = []
            # left x=0
            if vx != 0:
                t = (0.0 - cx) / vx
                if t > 0:
                    y_at = cy + vy * t
                    if 0.0 <= y_at <= (h - 1):
                        t_candidates.append(t)
            # right x=w-1
            if vx != 0:
                t = ((w - 1) - cx) / vx
                if t > 0:
                    y_at = cy + vy * t
                    if 0.0 <= y_at <= (h - 1):
                        t_candidates.append(t)
            # top y=0
            if vy != 0:
                t = (0.0 - cy) / vy
                if t > 0:
                    x_at = cx + vx * t
                    if 0.0 <= x_at <= (w - 1):
                        t_candidates.append(t)
            # bottom y=h-1
            if vy != 0:
                t = ((h - 1) - cy) / vy
                if t > 0:
                    x_at = cx + vx * t
                    if 0.0 <= x_at <= (w - 1):
                        t_candidates.append(t)

            if not t_candidates:
                # fallback: clamp to nearest edge
                if abs(vx) > abs(vy):
                    t_use = ((w - 1 - cx) / vx) if vx > 0 else ((0 - cx) / vx)
                else:
                    t_use = ((h - 1 - cy) / vy) if vy > 0 else ((0 - cy) / vy)
            else:
                t_use = min(t_candidates)

            ix = cx + vx * t_use
            iy = cy + vy * t_use
            ix = max(0, min(w - 1, ix))
            iy = max(0, min(h - 1, iy))
            rays.append((ix, iy))
            edge_pts.append(QPoint(int(round(ix)), int(round(iy))))

        self.preview.edge_rays = rays
        # store edge_points for compatibility/visualization (labels 5..8)
        self.preview.edge_points = edge_pts
        self.preview.update()
        self.status.setText("Center (point 5) computed and rays drawn to edges")

    def start_select_extra(self):
        """
        Enter mode to select points 6..9. Click four times on preview to set them.
        """
        if self.loaded_arr is None:
            QMessageBox.warning(self, "No input", "Load an input FITS and preview first.")
            return
        # clear previous extra points
        self.preview.extra_points = []
        self.preview.select_mode = "extra"
        self.preview.update()
        self.status.setText("Select points 6..9 by clicking on the preview (4 clicks)")

    def expand_to_edges(self):
        """
        Create points 5..8 by projecting points 1..4 from image center to edges (legacy behavior).
        """
        if self.loaded_arr is None:
            QMessageBox.warning(self, "No input", "Load an input FITS and preview first.")
            return
        pts = self.preview.points
        if len(pts) != 4:
            QMessageBox.warning(self, "Points required", "Click exactly 4 points first (TL, TR, BR, BL).")
            return

        h, w = self.loaded_arr.shape[:2]
        cx = (w - 1) / 2.0
        cy = (h - 1) / 2.0

        new_pts = []
        for p in pts:
            px = float(p.x())
            py = float(p.y())
            vx = px - cx
            vy = py - cy
            if abs(vx) < 1e-9 and abs(vy) < 1e-9:
                ix, iy = px, py
                new_pts.append(QPoint(int(round(ix)), int(round(iy))))
                continue

            t_candidates = []
            if vx != 0:
                t = (0.0 - cx) / vx
                if t > 0:
                    y_at = cy + vy * t
                    if 0.0 <= y_at <= (h - 1):
                        t_candidates.append(t)
                t = ((w - 1) - cx) / vx
                if t > 0:
                    y_at = cy + vy * t
                    if 0.0 <= y_at <= (h - 1):
                        t_candidates.append(t)
            if vy != 0:
                t = (0.0 - cy) / vy
                if t > 0:
                    x_at = cx + vx * t
                    if 0.0 <= x_at <= (w - 1):
                        t_candidates.append(t)
                t = ((h - 1) - cy) / vy
                if t > 0:
                    x_at = cx + vx * t
                    if 0.0 <= x_at <= (w - 1):
                        t_candidates.append(t)

            if not t_candidates:
                if abs(vx) > abs(vy):
                    t_use = ((w - 1 - cx) / vx) if vx > 0 else ((0 - cx) / vx)
                else:
                    t_use = ((h - 1 - cy) / vy) if vy > 0 else ((0 - cy) / vy)
            else:
                t_use = min(t_candidates)

            ix = cx + vx * t_use
            iy = cy + vy * t_use
            ix = max(0, min(w - 1, ix))
            iy = max(0, min(h - 1, iy))
            new_pts.append(QPoint(int(round(ix)), int(round(iy))))

        self.preview.edge_points = new_pts
        self.preview.update()
        self.status.setText("Expanded points to image edges (points 5..8 added)")

    def _compute_warp_from_points(self, src_points, out_w, out_h):
        if len(src_points) != 4:
            raise RuntimeError("Exactly 4 source points required")
        src = np.array([[p.x(), p.y()] if isinstance(p, QPoint) else [p[0], p[1]] for p in src_points], dtype=np.float32)
        dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        channels = []
        for c in range(self.loaded_arr.shape[2]):
            ch = np.ascontiguousarray(self.loaded_arr[:, :, c].astype(np.float32))
            warped = cv2.warpPerspective(ch, M, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            channels.append(warped)
        result = np.stack(channels, axis=-1)
        return result

    def preview_warp(self):
        if self.loaded_arr is None:
            QMessageBox.warning(self, "No input", "Load an input FITS and preview first.")
            return
        pts = self.preview.points
        if len(pts) != 4:
            QMessageBox.warning(self, "Points required", "Click exactly 4 points first (TL, TR, BR, BL).")
            return

        out_w = int(self.out_w_spin.value())
        out_h = int(self.out_h_spin.value())
        try:
            result = self._compute_warp_from_points(pts, out_w, out_h)
        except Exception as e:
            QMessageBox.critical(self, "Warp error", str(e))
            return

        img = np.nan_to_num(result)
        mn = float(np.nanmin(img))
        mx = float(np.nanmax(img))
        if mx > mn:
            img8 = ((img - mn) / (mx - mn) * 255.0).astype(np.uint8)
        else:
            img8 = np.zeros_like(img, dtype=np.uint8)
        if img8.ndim == 2:
            img8 = np.stack([img8]*3, axis=-1)
        elif img8.shape[2] > 3:
            img8 = img8[:, :, :3]

        img8_prepared, bytes_per_line = _prepare_preview_array_for_qimage(img8)
        qimg = QImage(img8_prepared.data, img8_prepared.shape[1], img8_prepared.shape[0], bytes_per_line, QImage.Format.Format_RGB888)
        qimg = qimg.copy()
        pix = QPixmap.fromImage(qimg)
        self.preview.setPixmap(pix, buf=img8_prepared)
        self.status.setText("Preview warp (points 1..4) shown in preview area")

    def warp_and_save(self):
        if self.loaded_arr is None:
            QMessageBox.warning(self, "No input", "Load an input FITS and preview first.")
            return
        pts = self.preview.points
        if len(pts) != 4:
            QMessageBox.warning(self, "Points required", "Click exactly 4 points on the preview in order TL, TR, BR, BL.")
            return

        out_w = int(self.out_w_spin.value())
        out_h = int(self.out_h_spin.value())
        try:
            result = self._compute_warp_from_points(pts, out_w, out_h)
        except Exception as e:
            QMessageBox.critical(self, "Warp error", str(e))
            return

        outpath = self.output_edit.text().strip()
        if not outpath:
            QMessageBox.warning(self, "Output required", "Choose an output filename first.")
            return
        try:
            write_fits_preserve_layout(outpath, result, self.loaded_hdr, self.layout_info)
            self.status.setText(f"Warp saved to {outpath}")
            if self.chk_siril.isChecked():
                launched, msg = _maybe_launch_siril(outpath)
                if launched:
                    QMessageBox.information(self, "Done", f"Warp saved and Siril launched.")
                else:
                    QMessageBox.information(self, "Done", f"Warp saved. {msg}")
            else:
                QMessageBox.information(self, "Done", f"Warp saved to {outpath}")
        except Exception as e:
            QMessageBox.critical(self, "Save error", str(e))

    def warp_and_save2(self):
        """
        Warp & save using points 5..8 (edge_points) if present.
        """
        if self.loaded_arr is None:
            QMessageBox.warning(self, "No input", "Load an input FITS and preview first.")
            return
        edge_pts = getattr(self.preview, "edge_points", None)
        if not edge_pts or len(edge_pts) != 4:
            QMessageBox.warning(self, "Points required", "Create expanded edge points first (Expand to Edges or Find Center).")
            return

        out_h, out_w = self.loaded_arr.shape[:2]
        try:
            result = self._compute_warp_from_points(edge_pts, out_w, out_h)
        except Exception as e:
            QMessageBox.critical(self, "Warp error", str(e))
            return

        outpath = self.output_edit.text().strip()
        if not outpath:
            QMessageBox.warning(self, "Output required", "Choose an output filename first.")
            return
        try:
            write_fits_preserve_layout(outpath, result, self.loaded_hdr, self.layout_info)
            self.status.setText(f"Warp (points 5..8) saved to {outpath}")
            if self.chk_siril.isChecked():
                launched, msg = _maybe_launch_siril(outpath)
                if launched:
                    QMessageBox.information(self, "Done", f"Warp saved and Siril launched.")
                else:
                    QMessageBox.information(self, "Done", f"Warp saved. {msg}")
            else:
                QMessageBox.information(self, "Done", f"Warp saved to {outpath}")
        except Exception as e:
            QMessageBox.critical(self, "Save error", str(e))

    def warp_and_save3(self):
        """
        Warp & save using points 6..9 (extra_points) if present.
        """
        if self.loaded_arr is None:
            QMessageBox.warning(self, "No input", "Load an input FITS and preview first.")
            return
        extra = self.preview.extra_points
        if len(extra) != 4:
            QMessageBox.warning(self, "Points required", "Select points 6..9 first (use Select points 6..9).")
            return

        out_h, out_w = self.loaded_arr.shape[:2]
        try:
            result = self._compute_warp_from_points(extra, out_w, out_h)
        except Exception as e:
            QMessageBox.critical(self, "Warp error", str(e))
            return

        outpath = self.output_edit.text().strip()
        if not outpath:
            QMessageBox.warning(self, "Output required", "Choose an output filename first.")
            return
        try:
            write_fits_preserve_layout(outpath, result, self.loaded_hdr, self.layout_info)
            self.status.setText(f"Warp (points 6..9) saved to {outpath}")
            if self.chk_siril.isChecked():
                launched, msg = _maybe_launch_siril(outpath)
                if launched:
                    QMessageBox.information(self, "Done", f"Warp saved and Siril launched.")
                else:
                    QMessageBox.information(self, "Done", f"Warp saved. {msg}")
            else:
                QMessageBox.information(self, "Done", f"Warp saved to {outpath}")
        except Exception as e:
            QMessageBox.critical(self, "Save error", str(e))


def main():
    app = QApplication(sys.argv)
    w = DeskewWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()