#!/usr/bin/env python3
"""
align_jpg_gui.py
PyQt6 GUI to align one JPG to a reference JPG using feature matching (ORB or AKAZE).
Outputs a warped aligned image saved to disk.
"""
import sys
import os
import traceback
from pathlib import Path

import numpy as np
import cv2

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QFileDialog, QGridLayout, QComboBox, QTextEdit, QMessageBox, QDoubleSpinBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage

# ---------- utilities ----------
def ndarray_to_qpix(arr):
    if arr is None:
        return None
    a = np.asarray(arr)
    if a.ndim == 2:
        if a.dtype != np.uint8:
            mn, mx = np.nanmin(a), np.nanmax(a)
            if mx > mn:
                a8 = ((a - mn) / (mx - mn) * 255.0).astype(np.uint8)
            else:
                a8 = np.zeros_like(a, dtype=np.uint8)
        else:
            a8 = a
        h, w = a8.shape
        qimg = QImage(a8.data, w, h, w, QImage.Format.Format_Grayscale8)
        return QPixmap.fromImage(qimg)
    else:
        if a.shape[2] >= 3:
            rgb = cv2.cvtColor(a[:, :, :3], cv2.COLOR_BGR2RGB)
        else:
            rgb = a
        if rgb.dtype != np.uint8:
            mn, mx = np.nanmin(rgb), np.nanmax(rgb)
            if mx > mn:
                rgb8 = ((rgb - mn) / (mx - mn) * 255.0).astype(np.uint8)
            else:
                rgb8 = np.zeros_like(rgb, dtype=np.uint8)
        else:
            rgb8 = rgb
        h, w, ch = rgb8.shape
        bytes_per_line = ch * w
        fmt = QImage.Format.Format_RGB888 if ch == 3 else QImage.Format.Format_RGBA8888
        qimg = QImage(rgb8.data, w, h, bytes_per_line, fmt)
        return QPixmap.fromImage(qimg)

# ---------- worker ----------
class AlignWorker(QThread):
    log = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    preview = pyqtSignal(object)  # emits numpy array (aligned image) for preview

    def __init__(self, ref_path, src_path, out_path, method, keep_frac):
        super().__init__()
        self.ref_path = ref_path
        self.src_path = src_path
        self.out_path = out_path
        self.method = method
        self.keep_frac = float(keep_frac)

    def _emit(self, *args):
        self.log.emit(" ".join(str(a) for a in args))

    def run(self):
        try:
            self._emit("Reading images...")
            ref_color = cv2.imread(self.ref_path, cv2.IMREAD_COLOR)
            src_color = cv2.imread(self.src_path, cv2.IMREAD_COLOR)
            if ref_color is None:
                raise FileNotFoundError(f"Cannot read reference: {self.ref_path}")
            if src_color is None:
                raise FileNotFoundError(f"Cannot read source: {self.src_path}")

            ref = cv2.cvtColor(ref_color, cv2.COLOR_BGR2GRAY)
            src = cv2.cvtColor(src_color, cv2.COLOR_BGR2GRAY)
            h_ref, w_ref = ref.shape

            # choose detector
            if self.method == "ORB":
                detector = cv2.ORB_create(5000)
            elif self.method == "AKAZE":
                detector = cv2.AKAZE_create()
            else:
                detector = cv2.ORB_create(5000)

            self._emit("Detecting keypoints and computing descriptors...")
            kp1, d1 = detector.detectAndCompute(src, None)
            kp2, d2 = detector.detectAndCompute(ref, None)
            if d1 is None or d2 is None or len(kp1) < 4 or len(kp2) < 4:
                raise RuntimeError("Not enough keypoints/descriptors found for reliable alignment")

            # matcher selection
            if self.method == "ORB" or self.method == "AKAZE":
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            else:
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

            self._emit("Matching descriptors...")
            matches = matcher.match(d1, d2)
            if not matches:
                raise RuntimeError("No matches found between images")
            matches = sorted(matches, key=lambda x: x.distance)
            keep_n = max(4, int(len(matches) * max(0.01, min(1.0, self.keep_frac))))
            matches = matches[:keep_n]
            self._emit(f"Total matches: {len(matches)}, using top {keep_n}")

            # build point arrays
            pts_src = np.zeros((len(matches), 2), dtype=np.float32)
            pts_ref = np.zeros((len(matches), 2), dtype=np.float32)
            for i, m in enumerate(matches):
                pts_src[i, :] = kp1[m.queryIdx].pt
                pts_ref[i, :] = kp2[m.trainIdx].pt

            self._emit("Estimating homography with RANSAC...")
            H, mask = cv2.findHomography(pts_src, pts_ref, cv2.RANSAC, ransacReprojThreshold=5.0)
            if H is None:
                raise RuntimeError("Homography estimation failed")

            self._emit("Warping source image to reference frame...")
            warped = cv2.warpPerspective(src_color, H, (w_ref, h_ref), flags=cv2.INTER_LINEAR)

            # write result
            outdir = os.path.dirname(self.out_path)
            if outdir and not os.path.isdir(outdir):
                os.makedirs(outdir, exist_ok=True)
            ok = cv2.imwrite(self.out_path, warped)
            if not ok:
                raise IOError("cv2.imwrite failed to save output")

            self._emit(f"Wrote aligned image: {self.out_path}")
            # send preview (BGR) to main thread
            self.preview.emit(warped)
            self.finished.emit(True, "Alignment complete")
        except Exception as e:
            tb = traceback.format_exc()
            self.log.emit(f"Error: {e}\n{tb}")
            self.finished.emit(False, str(e))

# ---------- main window ----------
class AlignWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Align JPG to Reference")
        self._build_ui()
        self.resize(980, 560)
        self.worker = None

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        grid = QGridLayout()
        central.setLayout(grid)

        grid.addWidget(QLabel("Reference image (target):"), 0, 0)
        self.ref_edit = QLineEdit()
        grid.addWidget(self.ref_edit, 0, 1, 1, 3)
        btn_ref = QPushButton("Browse")
        btn_ref.clicked.connect(lambda: self._browse_file(self.ref_edit))
        grid.addWidget(btn_ref, 0, 4)

        grid.addWidget(QLabel("Image to align (source):"), 1, 0)
        self.src_edit = QLineEdit()
        grid.addWidget(self.src_edit, 1, 1, 1, 3)
        btn_src = QPushButton("Browse")
        btn_src.clicked.connect(lambda: self._browse_file(self.src_edit))
        grid.addWidget(btn_src, 1, 4)

        grid.addWidget(QLabel("Output filename:"), 2, 0)
        self.out_edit = QLineEdit("aligned_output.jpg")
        grid.addWidget(self.out_edit, 2, 1, 1, 3)
        btn_out = QPushButton("Browse Save")
        btn_out.clicked.connect(self._browse_save)
        grid.addWidget(btn_out, 2, 4)

        grid.addWidget(QLabel("Feature method:"), 3, 0)
        self.method_combo = QComboBox()
        self.method_combo.addItems(["ORB", "AKAZE"])
        grid.addWidget(self.method_combo, 3, 1)

        grid.addWidget(QLabel("Keep fraction of top matches (0.1–1.0):"), 3, 2)
        self.keep_spin = QDoubleSpinBox()
        self.keep_spin.setRange(0.1, 1.0)
        self.keep_spin.setSingleStep(0.05)
        self.keep_spin.setValue(0.9)
        grid.addWidget(self.keep_spin, 3, 3)

        self.run_btn = QPushButton("Run Alignment")
        self.run_btn.clicked.connect(self._on_run)
        grid.addWidget(self.run_btn, 4, 0)

        self.preview_ref = QLabel("Reference preview")
        self.preview_ref.setFixedSize(320, 240)
        self.preview_ref.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(self.preview_ref, 5, 0, 4, 1)

        self.preview_src = QLabel("Source preview")
        self.preview_src.setFixedSize(320, 240)
        self.preview_src.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(self.preview_src, 5, 1, 4, 1)

        self.preview_out = QLabel("Aligned preview")
        self.preview_out.setFixedSize(320, 240)
        self.preview_out.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(self.preview_out, 5, 2, 4, 1)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        grid.addWidget(self.log, 5, 3, 4, 2)

        # quick load buttons
        load_btn = QPushButton("Load Images for Preview")
        load_btn.clicked.connect(self._load_previews)
        grid.addWidget(load_btn, 4, 1)

        clear_btn = QPushButton("Clear Log")
        clear_btn.clicked.connect(lambda: self.log.clear())
        grid.addWidget(clear_btn, 4, 2)

    def _log(self, *args):
        self.log.append(" ".join(str(a) for a in args))
        QApplication.processEvents()

    def _browse_file(self, edit):
        fn, _ = QFileDialog.getOpenFileName(self, "Open image", "", "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)")
        if fn:
            edit.setText(fn)

    def _browse_save(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Save aligned image", "", "JPEG (*.jpg *.jpeg);;PNG (*.png);;All Files (*)")
        if fn:
            self.out_edit.setText(fn)

    def _load_previews(self):
        refp = self.ref_edit.text().strip()
        srcp = self.src_edit.text().strip()
        if refp and os.path.exists(refp):
            img = cv2.imread(refp, cv2.IMREAD_COLOR)
            pix = ndarray_to_qpix(img)
            if pix:
                self.preview_ref.setPixmap(pix.scaled(self.preview_ref.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            self._log(f"Loaded reference: {Path(refp).name}")
        else:
            self._log("Reference not found or path empty")
        if srcp and os.path.exists(srcp):
            img = cv2.imread(srcp, cv2.IMREAD_COLOR)
            pix = ndarray_to_qpix(img)
            if pix:
                self.preview_src.setPixmap(pix.scaled(self.preview_src.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            self._log(f"Loaded source: {Path(srcp).name}")
        else:
            self._log("Source not found or path empty")

    def _on_run(self):
        refp = self.ref_edit.text().strip()
        srcp = self.src_edit.text().strip()
        outp = self.out_edit.text().strip()
        if not refp or not os.path.exists(refp):
            QMessageBox.warning(self, "Missing reference", "Choose an existing reference image")
            return
        if not srcp or not os.path.exists(srcp):
            QMessageBox.warning(self, "Missing source", "Choose an existing source image")
            return
        if not outp:
            QMessageBox.warning(self, "Missing output", "Specify an output filename")
            return

        method = self.method_combo.currentText()
        keep_frac = self.keep_spin.value()

        # start worker
        self.run_btn.setEnabled(False)
        self.worker = AlignWorker(refp, srcp, outp, method, keep_frac)
        self.worker.log.connect(self._log)
        self.worker.preview.connect(self._show_aligned_preview)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()
        self._log("Alignment worker started...")

    def _show_aligned_preview(self, arr):
        pix = ndarray_to_qpix(arr)
        if pix:
            self.preview_out.setPixmap(pix.scaled(self.preview_out.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def _on_finished(self, success, message):
        self.run_btn.setEnabled(True)
        if success:
            QMessageBox.information(self, "Done", message)
        else:
            QMessageBox.critical(self, "Failed", message)
        self._log("Worker finished: " + message)

def main():
    app = QApplication(sys.argv)
    w = AlignWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()