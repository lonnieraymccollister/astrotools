#!/usr/bin/env python3
"""
fits_crop_pyqt6_color.py

Upgraded from fits_crop_pyqt6.py to support color images.
Supports FITS with shape (H, W, 3) or (3, H, W). Keeps original
channel ordering when saving.
Dependencies:
    pip install astropy numpy opencv-python PyQt6
"""

import sys
import os
import json
import numpy as np
from astropy.io import fits
import cv2

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox, QGraphicsView,
    QGraphicsScene, QGraphicsPixmapItem, QGraphicsEllipseItem, QGraphicsLineItem,
    QPushButton, QHBoxLayout, QVBoxLayout, QWidget, QLabel
)
from PyQt6.QtGui import QPixmap, QImage, QPen, QColor, QBrush
from PyQt6.QtCore import Qt, QPointF, QRectF
from PyQt6.QtGui import QPainter

# ---------- Math helpers ----------
def unit_perp(v):
    perp = np.array([-v[1], v[0]], dtype=float)
    n = np.hypot(perp[0], perp[1])
    if n == 0:
        return np.array([0.0, 0.0])
    return perp / n

def compute_square_from_three(p1, p2, p3):
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    p3 = np.array(p3, dtype=float)
    v = p2 - p1
    s = np.hypot(v[0], v[1])
    if s == 0:
        raise ValueError("First two points are identical; side length zero.")
    u_perp = unit_perp(v)
    sign = np.sign(np.dot(p3 - p1, u_perp))
    if sign == 0:
        sign = 1.0
    c0 = p1
    c1 = p2
    c2 = p2 + sign * u_perp * s
    c3 = p1 + sign * u_perp * s
    corners = np.vstack([c0, c1, c2, c3])
    return corners

def crop_rotated_square_with_cv2(image, corners, out_size=None):
    """
    image: numpy array shape (H, W) or (H, W, C)
    corners: 4x2 array in image coordinates
    returns: cropped array (same dtype and channels), perspective matrix
    """
    src = np.array(corners, dtype=np.float32)
    side = np.hypot(*(src[1] - src[0]))
    if out_size is None:
        side_int = max(1, int(round(side)))
        dst = np.array([[0,0],[side_int-1,0],[side_int-1,side_int-1],[0,side_int-1]], dtype=np.float32)
        out_w, out_h = side_int, side_int
    else:
        out_w, out_h = out_size
        dst = np.array([[0,0],[out_w-1,0],[out_w-1,out_h-1],[0,out_h-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    # cv2.warpPerspective accepts multi-channel float32 arrays
    img32 = image.astype(np.float32)
    warped = cv2.warpPerspective(img32, M, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    # convert back to original dtype
    return warped.astype(image.dtype), M

# ---------- Graphics handle ----------
HANDLE_RADIUS = 6.0

class CornerHandle(QGraphicsEllipseItem):
    def __init__(self, x, y, index, parent=None):
        r = HANDLE_RADIUS
        super().__init__(-r, -r, 2*r, 2*r, parent)
        self.setPos(x, y)
        self.setBrush(QBrush(QColor(0, 255, 255)))
        self.setPen(QPen(QColor(0, 128, 128), 1))
        self.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.index = index

    def itemChange(self, change, value):
        return super().itemChange(change, value)

# ---------- Main App ----------
class FitsCropWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FITS Crop - PyQt6 (draggable corners) - Color support")
        self.resize(1000, 800)

        # State
        self.fits_path = None
        self.header = None
        self.image = None  # numpy array shape (H,W) or (H,W,C)
        self.orig_shape = None  # original data shape from FITS
        self.channel_layout = None  # 'channels_last' or 'channels_first' or None
        self.display_scale = 1.0  # scale from image to display
        self.points = []  # clicked points (image coords)
        self.corners = None  # 4x2 numpy
        self.crop_metadata = None
        self.last_cropped = None

        # UI
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHints(self.view.renderHints() | QPainter.RenderHint.Antialiasing)
        self.pixmap_item = None
        self.handle_items = []
        self.edge_items = []

        # Buttons
        btn_load = QPushButton("Load FITS")
        btn_load.clicked.connect(self.load_fits)
        btn_clear = QPushButton("Clear Points")
        btn_clear.clicked.connect(self.clear_points)
        btn_crop = QPushButton("Crop (use handles)")
        btn_crop.clicked.connect(self.crop_from_handles)
        btn_save = QPushButton("Save Crop + Metadata")
        btn_save.clicked.connect(self.save_crop_and_metadata)
        btn_load_meta = QPushButton("Load Metadata & Apply")
        btn_load_meta.clicked.connect(self.load_metadata_and_apply)

        info_label = QLabel("Click 3 points: p1, p2 (side), p3 (side selector). Then drag cyan handles to refine.")

        hbox = QHBoxLayout()
        for w in (btn_load, btn_clear, btn_crop, btn_save, btn_load_meta):
            hbox.addWidget(w)
        hbox.addStretch()
        hbox.addWidget(info_label)

        vbox = QVBoxLayout()
        vbox.addWidget(self.view)
        vbox.addLayout(hbox)

        container = QWidget()
        container.setLayout(vbox)
        self.setCentralWidget(container)

        # Mouse events on view
        self.view.viewport().installEventFilter(self)

    # ---------- FITS load/display ----------
    def load_fits(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select FITS file", "", "FITS files (*.fits *.fit *.fts);;All files (*)")
        if not path:
            return
        try:
            hdul = fits.open(path, memmap=False)
            data = None
            header = None
            for h in hdul:
                if h.data is not None and (h.data.ndim == 2 or h.data.ndim == 3):
                    data = h.data
                    header = h.header
                    break
            hdul.close()
            if data is None:
                QMessageBox.critical(self, "Error", "No 2D or 3D image found in FITS.")
                return
            self.fits_path = path
            self.header = header
            self.orig_shape = data.shape
            # Normalize channel layout to (H, W) or (H, W, C)
            if data.ndim == 2:
                img = np.array(data, copy=True)
                self.channel_layout = None
            else:
                # data.ndim == 3
                # common possibilities: (H, W, 3) or (3, H, W)
                if data.shape[0] == 3 and data.shape[1] != 3:
                    # assume (3, H, W) -> convert to (H, W, 3)
                    img = np.transpose(data, (1, 2, 0)).copy()
                    self.channel_layout = 'channels_first'
                elif data.shape[2] == 3:
                    img = np.array(data, copy=True)
                    self.channel_layout = 'channels_last'
                else:
                    # fallback: treat as multi-channel with channels last if possible
                    # try to detect small first axis as channels
                    if data.shape[0] <= 4:
                        img = np.transpose(data, (1, 2, 0)).copy()
                        self.channel_layout = 'channels_first'
                    else:
                        # treat as (H, W, C) anyway
                        img = np.array(data, copy=True)
                        self.channel_layout = 'channels_last'
            self.image = img
            self.points = []
            self.corners = None
            self.crop_metadata = None
            self.last_cropped = None
            self.update_display()
            self.setWindowTitle(f"FITS Crop - {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Error loading FITS", str(e))

    def update_display(self):
        self.scene.clear()
        self.handle_items = []
        self.edge_items = []
        if self.image is None:
            return
        img = self.image
        # Build display image (uint8) for QImage
        if img.ndim == 2:
            arr = img.astype(np.float64)
            vmin = np.percentile(arr, 1)
            vmax = np.percentile(arr, 99)
            if vmax <= vmin:
                vmax = arr.max() if arr.max() > vmin else vmin + 1.0
            disp = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
            disp8 = (disp * 255).astype(np.uint8)
            h, w = disp8.shape
            bytes_per_line = w
            qimg = QImage(disp8.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
        else:
            # color image (H, W, C) expected C==3
            arr = img.astype(np.float64)
            # compute vmin/vmax across all channels for consistent scaling
            vmin = np.percentile(arr, 1)
            vmax = np.percentile(arr, 99)
            if vmax <= vmin:
                vmax = arr.max() if arr.max() > vmin else vmin + 1.0
            disp = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
            disp8 = (disp * 255).astype(np.uint8)
            # ensure RGB order (FITS may be in any order but we normalized to H,W,3)
            if disp8.shape[2] == 3:
                rgb = disp8
            else:
                # if more channels, take first three
                rgb = disp8[:, :, :3]
            h, w, c = rgb.shape
            # QImage expects bytes in RGB888 order
            bytes_per_line = 3 * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.pixmap_item = QGraphicsPixmapItem(pix)
        self.scene.addItem(self.pixmap_item)
        self.view.setSceneRect(QRectF(0, 0, qimg.width(), qimg.height()))
        self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.display_scale = 1.0

        # draw existing points and corners if present
        for i, p in enumerate(self.points):
            circ = QGraphicsEllipseItem(p[0]-3, p[1]-3, 6, 6)
            circ.setBrush(QBrush(QColor(255,0,0)))
            circ.setPen(QPen(QColor(128,0,0)))
            self.scene.addItem(circ)
            txt = self.scene.addText(f"P{i+1}")
            txt.setDefaultTextColor(QColor(255,0,0))
            txt.setPos(p[0]+4, p[1]+4)

        if self.corners is not None:
            self._draw_square_and_handles(self.corners)

    # ---------- event filter to capture clicks ----------
    def eventFilter(self, source, event):
        if event.type() == event.Type.MouseButtonPress and source is self.view.viewport():
            if self.image is None:
                return super().eventFilter(source, event)
            pos = event.position()
            scene_pos = self.view.mapToScene(int(pos.x()), int(pos.y()))
            x, y = scene_pos.x(), scene_pos.y()
            if event.button() == Qt.MouseButton.LeftButton:
                if len(self.points) >= 3:
                    QMessageBox.information(self, "Points", "Already have 3 points. Clear points to select new ones.")
                else:
                    self.points.append((x, y))
                    if len(self.points) == 3:
                        try:
                            self.corners = compute_square_from_three(self.points[0], self.points[1], self.points[2])
                        except Exception as e:
                            QMessageBox.critical(self, "Error", str(e))
                            self.points = []
                            self.corners = None
                    self.update_display()
            return True
        # capture mouse release on scene to update corner positions if handles moved
        if event.type() == event.Type.MouseButtonRelease and source is self.view.viewport():
            # if handles exist, update internal corners from handle positions
            if self.handle_items and len(self.handle_items) == 4:
                corners = []
                for h in self.handle_items:
                    pos = h.scenePos()
                    corners.append([pos.x(), pos.y()])
                self.corners = np.array(corners, dtype=float)
                # redraw edges to match new positions
                # clear only edges and handles and redraw
                for it in self.edge_items:
                    self.scene.removeItem(it)
                self.edge_items = []
                # remove handles and re-add so they keep correct z-order
                for h in self.handle_items:
                    self.scene.removeItem(h)
                self.handle_items = []
                self._draw_square_and_handles(self.corners)
            return super().eventFilter(source, event)
        return super().eventFilter(source, event)

    # ---------- draw square and handles ----------
    def _draw_square_and_handles(self, corners):
        pen = QPen(QColor(0, 255, 255), 1.5)
        for i in range(4):
            p0 = corners[i]
            p1 = corners[(i+1)%4]
            line = QGraphicsLineItem(p0[0], p0[1], p1[0], p1[1])
            line.setPen(pen)
            self.scene.addItem(line)
            self.edge_items.append(line)
        for i in range(4):
            x, y = corners[i]
            h = CornerHandle(x, y, i)
            h.setZValue(10)
            self.scene.addItem(h)
            self.handle_items.append(h)
        self.scene.installEventFilter(self)

    # ---------- clear ----------
    def clear_points(self):
        self.points = []
        self.corners = None
        self.crop_metadata = None
        self.last_cropped = None
        self.update_display()

    # ---------- crop using current handle positions ----------
    def crop_from_handles(self):
        if self.image is None:
            QMessageBox.critical(self, "Error", "No FITS loaded.")
            return
        if not self.handle_items or len(self.handle_items) != 4:
            QMessageBox.critical(self, "Error", "Define square first by clicking three points.")
            return
        corners = []
        for h in self.handle_items:
            pos = h.scenePos()
            corners.append([pos.x(), pos.y()])
        corners = np.array(corners, dtype=float)
        try:
            cropped, M = crop_rotated_square_with_cv2(self.image, corners)
        except Exception as e:
            QMessageBox.critical(self, "Crop error", str(e))
            return
        self.last_cropped = cropped
        center = np.mean(corners, axis=0).tolist()
        side = float(np.hypot(*(corners[1] - corners[0])))
        v = corners[1] - corners[0]
        angle = float(np.arctan2(v[1], v[0]))
        self.crop_metadata = {
            "source_fits": os.path.basename(self.fits_path) if self.fits_path else None,
            "image_shape": list(self.orig_shape) if self.orig_shape is not None else list(self.image.shape),
            "corners": corners.tolist(),
            "center": center,
            "side_length": side,
            "angle_rad": angle,
            "perspective_matrix": M.tolist(),
            "dtype": str(self.image.dtype),
            "channel_layout": self.channel_layout
        }
        self._show_preview(cropped)

    def _show_preview(self, arr):
        # arr: (H,W) or (H,W,C)
        img = arr.astype(np.float64)
        if img.ndim == 2:
            vmin = np.percentile(img, 1)
            vmax = np.percentile(img, 99)
            if vmax <= vmin:
                vmax = img.max() if img.max() > vmin else vmin + 1.0
            disp = np.clip((img - vmin) / (vmax - vmin), 0.0, 1.0)
            disp8 = (disp * 255).astype(np.uint8)
            h, w = disp8.shape
            bytes_per_line = w
            qimg = QImage(disp8.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
        else:
            vmin = np.percentile(img, 1)
            vmax = np.percentile(img, 99)
            if vmax <= vmin:
                vmax = img.max() if img.max() > vmin else vmin + 1.0
            disp = np.clip((img - vmin) / (vmax - vmin), 0.0, 1.0)
            disp8 = (disp * 255).astype(np.uint8)
            if disp8.shape[2] == 3:
                rgb = disp8
            else:
                rgb = disp8[:, :, :3]
            h, w, c = rgb.shape
            bytes_per_line = 3 * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        win = QMainWindow(self)
        win.setWindowTitle("Cropped preview")
        lbl = QLabel()
        lbl.setPixmap(pix)
        win.setCentralWidget(lbl)
        win.resize(min(800, qimg.width()), min(800, qimg.height()))
        win.show()

    # ---------- save ----------
    def save_crop_and_metadata(self):
        if self.last_cropped is None:
            QMessageBox.critical(self, "Error", "No crop to save. Perform crop first.")
            return
        base, _ = QFileDialog.getSaveFileName(self, "Save cropped FITS as", "", "FITS (*.fits);;All files (*)")
        if not base:
            return
        try:
            out = self.last_cropped
            # convert back to original channel ordering if needed
            if self.channel_layout == 'channels_first':
                # original was (3, H, W)
                out_to_write = np.transpose(out, (2, 0, 1))
            else:
                out_to_write = out
            hdu = fits.PrimaryHDU(out_to_write.astype(self.image.dtype), header=self.header)
            hdu.header.add_history("Cropped with fits_crop_pyqt6_color.py")
            hdu.writeto(base, overwrite=True)
            meta_path = os.path.splitext(base)[0] + "_cropmeta.json"
            with open(meta_path, "w") as f:
                json.dump(self.crop_metadata, f, indent=2)
            QMessageBox.information(self, "Saved", f"Cropped FITS saved to:\n{base}\nMetadata saved to:\n{meta_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error saving", str(e))

    # ---------- load metadata and apply ----------
    def load_metadata_and_apply(self):
        meta_path, _ = QFileDialog.getOpenFileName(self, "Select crop metadata JSON", "", "JSON files (*.json);;All files (*)")
        if not meta_path:
            return
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load metadata: {e}")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Select FITS to apply crop to", "", "FITS files (*.fits *.fit *.fts);;All files (*)")
        if not path:
            return
        try:
            hdul = fits.open(path, memmap=False)
            data = None
            header = None
            for h in hdul:
                if h.data is not None and (h.data.ndim == 2 or h.data.ndim == 3):
                    data = h.data
                    header = h.header
                    break
            hdul.close()
            if data is None:
                QMessageBox.critical(self, "Error", "No 2D or 3D image found in FITS.")
                return
            # normalize to (H,W) or (H,W,3)
            if data.ndim == 2:
                img = np.array(data, copy=True)
                channel_layout = None
            else:
                if data.shape[0] == 3 and data.shape[1] != 3:
                    img = np.transpose(data, (1, 2, 0)).copy()
                    channel_layout = 'channels_first'
                else:
                    img = np.array(data, copy=True)
                    channel_layout = 'channels_last'
            corners = np.array(meta["corners"], dtype=float)
            cropped, M = crop_rotated_square_with_cv2(img, corners)
            base, _ = QFileDialog.getSaveFileName(self, "Save cropped FITS as", "", "FITS (*.fits);;All files (*)")
            if not base:
                return
            # convert back to original layout if needed
            if channel_layout == 'channels_first':
                out_to_write = np.transpose(cropped, (2, 0, 1))
            else:
                out_to_write = cropped
            hdu = fits.PrimaryHDU(out_to_write.astype(img.dtype), header=header)
            hdu.header.add_history(f"Cropped with fits_crop_pyqt6_color.py using metadata {os.path.basename(meta_path)}")
            hdu.writeto(base, overwrite=True)
            QMessageBox.information(self, "Saved", f"Cropped FITS saved to:\n{base}")
        except Exception as e:
            QMessageBox.critical(self, "Error applying metadata", str(e))

# ---------- run ----------
def main():
    app = QApplication(sys.argv)
    w = FitsCropWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()