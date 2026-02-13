#!/usr/bin/env python3
"""
siril_fft_notch_pyqt6_full.py

PyQt6 GUI for FFT notch filtering with:
- FilePicker for modulus/phase FITS
- FFTEditor with:
  - Auto-stretch FFT display
  - Gamma slider
  - FFT stats label
  - Notch deletion fixes
  - Center-protection toggle
  - File verification
  - Defensive wiring
  - All handlers implemented
"""

import sys
import os
import numpy as np
from astropy.io import fits
from scipy.fft import ifft2, ifftshift
from scipy.ndimage import gaussian_filter, label
from skimage.measure import regionprops

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QSlider, QFrame, QMessageBox
)
from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ----------------- Defaults -----------------
OUTPUT_IMAGE = "filtered_image.fits"
OUTPUT_MASK  = "fft_mask.fits"

AUTO_SMOOTH_SIGMA   = 3.0
AUTO_PERCENTILE     = 99.5
DEFAULT_RADIUS      = 6.0
DEFAULT_DEPTH       = 0.95
CENTER_MASK_RADIUS  = 10
# --------------------------------------------


# ---------- Helpers ----------

def load_modulus(path):
    data = fits.getdata(path)
    return np.array(data, dtype=np.float64), fits.getheader(path)

def load_phase(path):
    data = fits.getdata(path).astype(np.float64)
    return data, fits.getheader(path)

def normalize_phase(phase):
    pmin, pmax = np.nanmin(phase), np.nanmax(phase)
    if pmax <= 2*np.pi*1.1:
        return phase
    if pmax <= 360*1.1:
        return np.deg2rad(phase)
    rng = pmax - pmin
    if rng == 0:
        return phase
    return (phase - pmin) / rng * (2*np.pi)

def reconstruct_complex_fft(modulus_path, phase_path):
    mag, _ = load_modulus(modulus_path)
    ph, _  = load_phase(phase_path)
    ph_rad = normalize_phase(ph)
    return mag * np.exp(1j * ph_rad)

def log_magnitude(F):
    return np.log1p(np.abs(F))

def auto_detect_peaks(logmag, sigma=AUTO_SMOOTH_SIGMA,
                      percentile=AUTO_PERCENTILE,
                      center_mask_radius=CENTER_MASK_RADIUS):
    sm = gaussian_filter(logmag, sigma=sigma)
    th = np.percentile(sm, percentile)
    mask = sm > th

    h, w = mask.shape
    cy, cx = h//2, w//2
    r = center_mask_radius
    r = int(max(1, min(r, min(h, w)//4)))
    mask[max(0, cy-r):min(h, cy+r+1), max(0, cx-r):min(w, cx+r+1)] = False

    labeled, _ = label(mask)
    props = regionprops(labeled, intensity_image=sm)
    peaks = []
    for p in props:
        y0, x0 = p.centroid
        peaks.append((int(round(y0)), int(round(x0))))
    return peaks, sm

def build_notch_mask(shape, centers, radius, depth, gaussian=True):
    Y, X = np.indices(shape)
    mask = np.ones(shape, dtype=np.float64)
    for (py, px) in centers:
        d2 = (X - px)**2 + (Y - py)**2
        if gaussian:
            notch = 1.0 - depth * np.exp(-0.5 * d2 / (radius**2))
        else:
            notch = np.ones_like(mask)
            notch[d2 <= radius**2] = 1.0 - depth
        mask *= notch
    return 0.5 * (mask + mask[::-1, ::-1])

def apply_mask_and_inverse(F_complex, mask):
    F_filtered = F_complex * mask
    return np.real(ifft2(ifftshift(F_filtered)))

def stretch_for_display(img):
    img = np.asarray(img, dtype=np.float64)
    lo, hi = np.percentile(img, (0.5, 99.5))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(img)
    stretched = (img - lo) / (hi - lo)
    return np.clip(stretched, 0, 1)


# ---------- File picker ----------

class FilePicker(QWidget):
    """Initial form: two path boxes, two Browse buttons, Continue button."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Modulus and Phase FITS")
        self.mod_path = None
        self.phase_path = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        form = QVBoxLayout()

        # Modulus row
        row_mod = QHBoxLayout()
        lbl_mod = QLabel("Modulus FITS:")
        lbl_mod.setFixedWidth(110)
        self.txt_mod = QLabel("")
        self.txt_mod.setFrameStyle(QFrame.Shape.Panel | QFrame.Shadow.Sunken)
        self.txt_mod.setStyleSheet("background-color: #111; color: #ffd; padding:4px;")
        btn_mod = QPushButton("Browse…")
        btn_mod.clicked.connect(self.browse_modulus)
        row_mod.addWidget(lbl_mod)
        row_mod.addWidget(self.txt_mod, 1)
        row_mod.addWidget(btn_mod)
        form.addLayout(row_mod)

        # Phase row
        row_phase = QHBoxLayout()
        lbl_phase = QLabel("Phase FITS:")
        lbl_phase.setFixedWidth(110)
        self.txt_phase = QLabel("")
        self.txt_phase.setFrameStyle(QFrame.Shape.Panel | QFrame.Shadow.Sunken)
        self.txt_phase.setStyleSheet("background-color: #111; color: #ffd; padding:4px;")
        btn_phase = QPushButton("Browse…")
        btn_phase.clicked.connect(self.browse_phase)
        row_phase.addWidget(lbl_phase)
        row_phase.addWidget(self.txt_phase, 1)
        row_phase.addWidget(btn_phase)
        form.addLayout(row_phase)

        layout.addLayout(form)

        # Continue button centered (start disabled)
        self.btn_continue = QPushButton("Continue")
        self.btn_continue.setFixedWidth(160)
        self.btn_continue.setEnabled(False)
        self.btn_continue.clicked.connect(self.on_continue)
        h = QHBoxLayout()
        h.addStretch(1)
        h.addWidget(self.btn_continue)
        h.addStretch(1)
        layout.addLayout(h)

        self.setLayout(layout)
        self.resize(700, 180)

    def _update_continue_enabled(self):
        enabled = bool(self.mod_path) and bool(self.phase_path)
        self.btn_continue.setEnabled(enabled)

    def browse_modulus(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select modulus FITS", "", "FITS Files (*.fits *.fit *.fts)")
        if path:
            self.mod_path = path
            self.txt_mod.setText(path)
        self._update_continue_enabled()

    def browse_phase(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select phase FITS", "", "FITS Files (*.fits *.fit *.fts)")
        if path:
            self.phase_path = path
            self.txt_phase.setText(path)
        self._update_continue_enabled()

    def verify_files(self):
        """Return (ok, message). Verifies both files exist and are readable FITS."""
        if not self.mod_path or not self.phase_path:
            return False, "Modulus or phase path is empty."
        if not os.path.exists(self.mod_path):
            return False, f"Modulus file not found: {self.mod_path}"
        if not os.path.exists(self.phase_path):
            return False, f"Phase file not found: {self.phase_path}"

        try:
            fits.getdata(self.mod_path)
        except Exception as e:
            return False, f"Failed to read modulus FITS: {e}"
        try:
            fits.getdata(self.phase_path)
        except Exception as e:
            return False, f"Failed to read phase FITS: {e}"

        return True, "Files OK"

    def on_continue(self):
        ok, msg = self.verify_files()
        if not ok:
            QMessageBox.critical(self, "File verification failed", msg)
            return

        print("[DEBUG] modulus_path:", self.mod_path)
        print("[DEBUG] phase_path:", self.phase_path)

        try:
            self.editor = FFTEditor(self.mod_path, self.phase_path)
            self.editor.show()
            self.close()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open editor:\n{e}")


# ---------- FFT editor ----------

class FFTEditor(QWidget):
    def __init__(self, modulus_path, phase_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("FFT Notch Editor")

        if not os.path.exists(modulus_path) or not os.path.exists(phase_path):
            raise FileNotFoundError("Modulus or phase file not found before initializing editor.")

        self.modulus_path = modulus_path
        self.phase_path = phase_path

        self.F = None
        self.logmag = None
        self.img_orig = None
        self.img_prev = None
        self.peaks = []
        self.notches = []  # (py, px) or (py, px, enabled)
        self.mask = None

        self.radius = DEFAULT_RADIUS
        self.depth = DEFAULT_DEPTH
        self.gaussian = True
        self.center_protect_radius = 10
        self.center_protect_enabled = True

        self.fft_im = None
        self.gamma_value = 1.0
        self.lo_pct = 0.5
        self.hi_pct = 99.5

        self._build_ui()
        self._load_fft_and_init()

    # ---------- UI ----------

    def _build_ui(self):
        main = QVBoxLayout()
        top = QHBoxLayout()

        self.fig = Figure(figsize=(10, 5))
        self.canvas = FigureCanvas(self.fig)
        self.ax_fft = self.fig.add_subplot(1, 2, 1)
        self.ax_preview = self.fig.add_subplot(1, 2, 2)

        self.ax_fft.set_title("Log Magnitude (click to delete notch)")
        self.ax_preview.set_title("Preview (original)")

        top.addWidget(self.canvas, 1)
        main.addLayout(top)

        controls = QHBoxLayout()

        self.btn_auto = QPushButton("Auto Detect")
        self.btn_preview = QPushButton("Preview")
        self.btn_save = QPushButton("Save")
        self.btn_reset = QPushButton("Reset")
        self.btn_redet = QPushButton("Redetect")
        self.btn_toggle = QPushButton("Toggle Gaussian")

        self.btn_center_protect = QPushButton("Protect Center: ON")
        self.btn_center_protect.setCheckable(True)
        self.btn_center_protect.setChecked(True)
        self.btn_center_protect.clicked.connect(self.toggle_center_protect)
        controls.addWidget(self.btn_center_protect)

        def safe_connect(widget, signal_name, handler_name, fallback_message):
            handler = getattr(self, handler_name, None)
            if handler is None:
                def _missing():
                    QMessageBox.critical(self, "Handler missing", fallback_message)
                getattr(widget, signal_name).connect(_missing)
            else:
                getattr(widget, signal_name).connect(handler)

        safe_connect(self.btn_auto, "clicked", "on_auto", "Auto-detect handler not implemented.")
        safe_connect(self.btn_preview, "clicked", "on_preview", "Preview handler not implemented.")
        safe_connect(self.btn_save, "clicked", "on_save", "Save handler not implemented.")
        safe_connect(self.btn_reset, "clicked", "on_reset", "Reset handler not implemented.")
        safe_connect(self.btn_redet, "clicked", "on_redetect", "Redetect handler not implemented.")
        safe_connect(self.btn_toggle, "clicked", "on_toggle", "Toggle handler not implemented.")

        for b in (self.btn_auto, self.btn_preview, self.btn_save, self.btn_reset, self.btn_redet, self.btn_toggle):
            controls.addWidget(b)

        # Auto-stretch button
        self.btn_autostretch = QPushButton("Auto Stretch FFT")
        self.btn_autostretch.clicked.connect(
            lambda: self.update_fft_display(auto_stretch=True,
                                            lo_pct=self.lo_pct,
                                            hi_pct=self.hi_pct,
                                            gamma=self.gamma_value)
        )
        controls.addWidget(self.btn_autostretch)

        main.addLayout(controls)

        sliders = QHBoxLayout()
        lbl_rad = QLabel("Radius")
        self.sld_rad = QSlider(Qt.Orientation.Horizontal)
        self.sld_rad.setMinimum(1)
        self.sld_rad.setMaximum(100)
        self.sld_rad.setValue(int(self.radius))
        if getattr(self, "on_radius_change", None):
            self.sld_rad.valueChanged.connect(self.on_radius_change)
        else:
            self.sld_rad.valueChanged.connect(lambda v: None)

        lbl_dep = QLabel("Depth")
        self.sld_dep = QSlider(Qt.Orientation.Horizontal)
        self.sld_dep.setMinimum(0)
        self.sld_dep.setMaximum(100)
        self.sld_dep.setValue(int(self.depth * 100))
        if getattr(self, "on_depth_change", None):
            self.sld_dep.valueChanged.connect(self.on_depth_change)
        else:
            self.sld_dep.valueChanged.connect(lambda v: None)

        sliders.addWidget(lbl_rad)
        sliders.addWidget(self.sld_rad, 2)
        sliders.addSpacing(12)
        sliders.addWidget(lbl_dep)
        sliders.addWidget(self.sld_dep, 2)

        # Gamma slider
        self.lbl_gamma = QLabel("Gamma")
        self.sld_gamma = QSlider(Qt.Orientation.Horizontal)
        self.sld_gamma.setMinimum(30)   # 0.3
        self.sld_gamma.setMaximum(250)  # 2.5
        self.sld_gamma.setValue(100)    # 1.0
        self.sld_gamma.setFixedWidth(140)

        def _gamma_changed(v):
            self.gamma_value = v / 100.0
            self.update_fft_display(auto_stretch=True,
                                    lo_pct=self.lo_pct,
                                    hi_pct=self.hi_pct,
                                    gamma=self.gamma_value)

        self.sld_gamma.valueChanged.connect(_gamma_changed)

        sliders.addSpacing(12)
        sliders.addWidget(self.lbl_gamma)
        sliders.addWidget(self.sld_gamma, 1)

        main.addLayout(sliders)

        warn_noise = QVBoxLayout()
        self.lbl_warning = QLabel("")
        self.lbl_warning.setStyleSheet("color: red;")
        self.lbl_noise = QLabel("Noise: Unclassified")
        self.lbl_noise.setStyleSheet("color: goldenrod;")
        warn_noise.addWidget(self.lbl_warning)
        warn_noise.addWidget(self.lbl_noise)
        main.addLayout(warn_noise)

        # FFT stats label
        self.lbl_fft_stats = QLabel("FFT: min=.. max=.. pct=.. gamma=..")
        self.lbl_fft_stats.setStyleSheet("color: #ddd; background:#222; padding:4px;")
        main.addWidget(self.lbl_fft_stats)

        self.setLayout(main)
        self.resize(1100, 700)

        self.canvas.mpl_connect("button_press_event", self.on_click)

    # ---------- FFT loading ----------

    def _load_fft_and_init(self):
        try:
            print("[DEBUG] Loading FFT from:", self.modulus_path, self.phase_path)

            ph = fits.getdata(self.phase_path)
            mag = fits.getdata(self.modulus_path)

            if ph is None:
                raise ValueError("Phase data is empty.")
            if mag is None:
                raise ValueError("Modulus data is empty.")
            if ph.shape != mag.shape:
                raise ValueError(f"Phase shape {ph.shape} does not match modulus shape {mag.shape}.")
            if not np.isfinite(ph).all():
                raise ValueError("Phase contains NaN or infinite values.")

            pmin, pmax = float(np.nanmin(ph)), float(np.nanmax(ph))
            print(f"[DEBUG] Phase range: {pmin} .. {pmax}")

            self.F = reconstruct_complex_fft(self.modulus_path, self.phase_path)
            self.logmag = log_magnitude(self.F)
            self.peaks, _ = auto_detect_peaks(self.logmag)
            self.notches = list(self.peaks)
            self.rebuild_mask()
            self.img_orig = np.real(ifft2(ifftshift(self.F)))
            self.img_prev = self.img_orig.copy()
            self._draw_initial()
            self.update_warning()
            self.update_noise_text()
        except Exception as e:
            QMessageBox.critical(self, "Load error", f"Failed to load FFT: {e}")
            raise

    # ---------- Display helpers ----------

    def _clean_logmag(self, arr):
        a = np.array(arr, dtype=np.float64, copy=True)
        finite = np.isfinite(a)
        if not finite.any():
            return np.zeros_like(a)
        min_f = np.nanmin(a[finite])
        a[~finite] = min_f
        return a

    def _percentile_stretch(self, arr, lo_pct=0.5, hi_pct=99.5, clip=True):
        a = self._clean_logmag(arr)
        vmin = float(np.nanpercentile(a, lo_pct))
        vmax = float(np.nanpercentile(a, hi_pct))
        if vmax <= vmin:
            vmax = vmin + 1e-6
        stretched = (a - vmin) / (vmax - vmin)
        if clip:
            stretched = np.clip(stretched, 0.0, 1.0)
        return stretched, vmin, vmax

    def _apply_gamma(self, img, gamma=1.0):
        if gamma == 1.0:
            return img
        img = np.clip(img, 0.0, 1.0)
        return img ** (1.0 / gamma)

    def update_fft_display(self, auto_stretch=True, lo_pct=0.5, hi_pct=99.5, gamma=1.0):
        if self.logmag is None:
            return

        self.lo_pct = lo_pct
        self.hi_pct = hi_pct
        self.gamma_value = gamma

        lm = self._clean_logmag(self.logmag)
        stretched, vmin, vmax = self._percentile_stretch(lm, lo_pct, hi_pct)
        stretched = self._apply_gamma(stretched, gamma=gamma)

        if self.fft_im is None:
            self.fft_im = self.ax_fft.imshow(stretched, cmap="inferno", origin="lower", vmin=0.0, vmax=1.0)
        else:
            self.fft_im.set_data(stretched)
            self.fft_im.set_clim(0.0, 1.0)

        xs = [p[1] for p in self.notches]
        ys = [p[0] for p in self.notches]
        if getattr(self, "scatter", None) is None:
            self.scatter = self.ax_fft.scatter(xs, ys, s=60, facecolors="none", edgecolors="cyan", linewidths=1.2)
        else:
            self.scatter.set_offsets(np.c_[xs, ys])

        if self.lbl_fft_stats is not None:
            self.lbl_fft_stats.setText(
                f"FFT: min={vmin:.3g} max={vmax:.3g} pct={lo_pct}/{hi_pct} gamma={gamma:.2f}"
            )

        self.canvas.draw_idle()

    def _draw_initial(self):
        self.ax_fft.clear()
        self.ax_preview.clear()

        if self.fft_im is not None:
            try:
                self.fft_im.remove()
            except Exception:
                pass
            self.fft_im = None

        self.update_fft_display(auto_stretch=True,
                                lo_pct=self.lo_pct,
                                hi_pct=self.hi_pct,
                                gamma=self.gamma_value)

        xs = [p[1] for p in self.notches]
        ys = [p[0] for p in self.notches]
        self.scatter = self.ax_fft.scatter(xs, ys, s=60, facecolors="none", edgecolors="cyan", linewidths=1.2)

        self.ax_preview.imshow(stretch_for_display(self.img_prev), cmap="gray", origin="lower")
        self.ax_fft.set_title("Log Magnitude (click to delete notch)")
        self.ax_preview.set_title("Preview (original)")
        self.canvas.draw_idle()

    # ---------- Noise detection / labels ----------

    def detect_periodic_noise(self):
        if self.logmag is None:
            return False
        hi = np.percentile(self.logmag, 99.9)
        lo = np.percentile(self.logmag, 90)
        if hi <= 0 or (hi - lo) < 0.05 * hi:
            return False
        h, w = self.logmag.shape
        cy, cx = h//2, w//2
        Y, X = np.indices(self.logmag.shape)
        r = np.hypot(X - cx, Y - cy)
        mask = r > min(h, w) * 0.1
        bright = (self.logmag > hi * 0.9) & mask
        return np.sum(bright) >= 2

    def update_warning(self):
        if not self.detect_periodic_noise():
            self.lbl_warning.setText("No periodic noise detected — filtering not recommended.")
        else:
            self.lbl_warning.setText("")

    def update_noise_text(self):
        self.lbl_noise.setText("Noise: " + self.classify_noise())

    def classify_noise(self):
        if not self.notches:
            return "No periodic noise detected."
        h, w = self.logmag.shape
        cy, cx = h//2, w//2
        angles = []
        radii = []
        for (py, px) in self.notches:
            dy = py - cy
            dx = px - cx
            angle = np.degrees(np.arctan2(dy, dx))
            radius = np.hypot(dx, dy)
            angles.append(angle)
            radii.append(radius)
        angles = np.array(angles)
        radii = np.array(radii)
        msg = []
        if np.any(np.abs(angles) < 10) or np.any(np.abs(np.abs(angles)-180) < 10):
            msg.append("Horizontal banding")
        if np.any(np.abs(np.abs(angles)-90) < 10):
            msg.append("Vertical banding")
        if np.any(np.abs(np.abs(angles)-45) < 10) or np.any(np.abs(np.abs(angles)-135) < 10):
            msg.append("Diagonal banding")
        if len(self.notches) == 1:
            msg.append("Electronic interference")
        if len(self.notches) >= 6:
            msg.append("Grid / compression pattern")
        if not msg:
            return "Unclassified periodic noise"
        return ", ".join(msg)

    # ---------- Mask / overlay ----------

    def rebuild_mask(self):
        centers = []
        for n in self.notches:
            if len(n) >= 2:
                py, px = n[0], n[1]
                enabled = True if len(n) == 2 else bool(n[2])
                if enabled:
                    centers.append((py, px))
        if self.logmag is None:
            self.mask = np.ones((1,1), dtype=np.float64)
            return
        self.mask = build_notch_mask(self.logmag.shape, centers, self.radius, self.depth, self.gaussian)

    def refresh_overlay(self):
        if self.logmag is None or self.ax_fft is None:
            return
        xs = []
        ys = []
        colors = []
        for n in self.notches:
            if len(n) >= 2:
                py, px = n[0], n[1]
                xs.append(px)
                ys.append(py)
                if len(n) >= 3:
                    colors.append("cyan" if bool(n[2]) else "gray")
                else:
                    colors.append("cyan")
        try:
            if getattr(self, "scatter", None) is None:
                self.scatter = self.ax_fft.scatter(xs, ys, s=60, facecolors="none", edgecolors=colors, linewidths=1.2)
            else:
                if len(xs) == 0:
                    try:
                        self.scatter.remove()
                    except Exception:
                        pass
                    self.scatter = self.ax_fft.scatter([], [], s=60, facecolors="none", edgecolors="cyan", linewidths=1.2)
                else:
                    self.scatter.set_offsets(np.c_[xs, ys])
                    try:
                        self.scatter.set_edgecolors(colors)
                    except Exception:
                        pass
        except Exception:
            try:
                self.scatter.remove()
            except Exception:
                pass
            self.scatter = self.ax_fft.scatter(xs, ys, s=60, facecolors="none", edgecolors=colors, linewidths=1.2)
        self.canvas.draw_idle()

    # ---------- Callbacks ----------

    def on_click(self, event):
        if event.inaxes != self.ax_fft:
            return
        if event.xdata is None or event.ydata is None:
            return

        cx = float(event.xdata)
        cy = float(event.ydata)
        tol = max(6.0, float(self.radius))
        force_delete = getattr(event, "button", None) == 3

        indexed = []
        for i, n in enumerate(self.notches):
            if len(n) >= 2:
                py = float(n[0]); px = float(n[1])
                enabled = True if len(n) == 2 else bool(n[2])
                indexed.append((i, py, px, enabled))

        best_idx = None
        best_dist2 = float("inf")
        for i, py, px, enabled in indexed:
            if not enabled:
                continue
            d2 = (px - cx)**2 + (py - cy)**2
            if d2 < best_dist2:
                best_dist2 = d2
                best_idx = i

        if best_idx is not None and best_dist2 <= tol**2:
            h, w = self.logmag.shape
            cy_img, cx_img = h//2, w//2
            py, px = float(self.notches[best_idx][0]), float(self.notches[best_idx][1])
            center_dist = ((px - cx_img)**2 + (py - cy_img)**2)**0.5
            if center_dist <= self.center_protect_radius and not (force_delete or not self.center_protect_enabled):
                print("[INFO] Notch is inside center safe zone. Right-click to force delete or disable protection.")
                return
            removed = self.notches.pop(best_idx)
            print(f"[DEBUG] Deleted notch index {best_idx} coord {removed}")
            self.rebuild_mask()
            self.refresh_overlay()
            self.update_warning()
            self.update_noise_text()
            self.update_fft_display(auto_stretch=True,
                                    lo_pct=self.lo_pct,
                                    hi_pct=self.hi_pct,
                                    gamma=self.gamma_value)
            return

        self.notches.append((int(round(cy)), int(round(cx))))
        print(f"[DEBUG] Added notch at {(int(round(cy)), int(round(cx)))}")
        self.rebuild_mask()
        self.refresh_overlay()
        self.update_warning()
        self.update_noise_text()
        self.update_fft_display(auto_stretch=True,
                                lo_pct=self.lo_pct,
                                hi_pct=self.hi_pct,
                                gamma=self.gamma_value)

    def on_auto(self):
        try:
            print("[INFO] on_auto called")
            if getattr(self, "logmag", None) is None:
                QMessageBox.information(self, "Auto Detect", "FFT not loaded.")
                return
            self.peaks, _ = auto_detect_peaks(self.logmag)
            self.notches = list(self.peaks)
            self.rebuild_mask()
            self.refresh_overlay()
            self.update_warning()
            self.update_noise_text()
            self.update_fft_display(auto_stretch=True,
                                    lo_pct=self.lo_pct,
                                    hi_pct=self.hi_pct,
                                    gamma=self.gamma_value)
        except Exception as e:
            print("[ERROR] on_auto:", e)

    def on_preview(self):
        try:
            print("[INFO] on_preview called")
            if getattr(self, "logmag", None) is None:
                QMessageBox.information(self, "Preview", "FFT not loaded.")
                return
            if not self.detect_periodic_noise():
                QMessageBox.information(self, "Preview", "No periodic noise detected — preview shows original image.")
                self.img_prev = self.img_orig.copy()
                self.ax_preview.clear()
                self.ax_preview.imshow(stretch_for_display(self.img_prev), cmap="gray", origin="lower")
                self.ax_preview.set_title("Preview (no periodic noise)")
                self.canvas.draw_idle()
                return
            self.img_prev = apply_mask_and_inverse(self.F, self.mask)
            self.ax_preview.clear()
            self.ax_preview.imshow(stretch_for_display(self.img_prev), cmap="gray", origin="lower")
            self.ax_preview.set_title("Preview (filtered)")
            self.canvas.draw_idle()
        except Exception as e:
            print("[ERROR] on_preview:", e)

    def on_save(self):
        try:
            print("[INFO] on_save called")
            if getattr(self, "logmag", None) is None:
                QMessageBox.information(self, "Save", "FFT not loaded.")
                return
            if not self.detect_periodic_noise():
                img_out = self.img_orig.copy()
            else:
                img_out = apply_mask_and_inverse(self.F, self.mask)
            lo, hi = np.percentile(img_out, (0.5, 99.5))
            if hi <= lo:
                hi = lo + 1e-6
            img_norm = (img_out - lo) / (hi - lo)
            img_norm = np.clip(img_norm, 0, 1)
            fits.writeto(OUTPUT_IMAGE, img_norm.astype(np.float32), overwrite=True)
            fits.writeto(OUTPUT_MASK, self.mask.astype(np.float32), overwrite=True)
            QMessageBox.information(self, "Saved", f"Saved image: {OUTPUT_IMAGE}\nSaved mask: {OUTPUT_MASK}")
        except Exception as e:
            print("[ERROR] on_save:", e)

    def on_reset(self):
        try:
            print("[INFO] on_reset called")
            self.notches = list(getattr(self, "peaks", []))
            self.radius = DEFAULT_RADIUS
            self.depth = DEFAULT_DEPTH
            try:
                self.sld_rad.setValue(int(self.radius))
                self.sld_dep.setValue(int(self.depth * 100))
            except Exception:
                pass
            self.gaussian = True
            self.rebuild_mask()
            self.img_prev = getattr(self, "img_orig", None)
            if self.img_prev is not None:
                self.ax_preview.clear()
                self.ax_preview.imshow(stretch_for_display(self.img_prev), cmap="gray", origin="lower")
                self.ax_preview.set_title("Preview (original)")
            self.refresh_overlay()
            self.update_warning()
            self.update_noise_text()
            self.update_fft_display(auto_stretch=True,
                                    lo_pct=self.lo_pct,
                                    hi_pct=self.hi_pct,
                                    gamma=self.gamma_value)
        except Exception as e:
            print("[ERROR] on_reset:", e)

    def on_redetect(self):
        try:
            print("[INFO] on_redetect called")
            if getattr(self, "logmag", None) is None:
                QMessageBox.information(self, "Redetect", "FFT not loaded.")
                return
            pct = np.clip(AUTO_PERCENTILE - self.depth * 50.0, 90.0, 99.9)
            sigma = max(1.0, self.radius / 2.0)
            self.peaks, _ = auto_detect_peaks(self.logmag, sigma=sigma, percentile=pct)
            self.notches = list(self.peaks)
            self.rebuild_mask()
            self.refresh_overlay()
            self.update_warning()
            self.update_noise_text()
            self.update_fft_display(auto_stretch=True,
                                    lo_pct=self.lo_pct,
                                    hi_pct=self.hi_pct,
                                    gamma=self.gamma_value)
        except Exception as e:
            print("[ERROR] on_redetect:", e)

    def on_toggle(self):
        try:
            print("[INFO] on_toggle called")
            self.gaussian = not getattr(self, "gaussian", True)
            self.rebuild_mask()
        except Exception as e:
            print("[ERROR] on_toggle:", e)

    def on_radius_change(self, val):
        try:
            self.radius = float(self.sld_rad.value())
            self.rebuild_mask()
        except Exception as e:
            print("[ERROR] on_radius_change:", e)

    def on_depth_change(self, val):
        try:
            self.depth = float(self.sld_dep.value()) / 100.0
            self.rebuild_mask()
        except Exception as e:
            print("[ERROR] on_depth_change:", e)

    def toggle_center_protect(self):
        self.center_protect_enabled = not getattr(self, "center_protect_enabled", True)
        state = "ON" if self.center_protect_enabled else "OFF"
        self.btn_center_protect.setText(f"Protect Center: {state}")

    def verify_phase_loaded(self):
        if self.F is None:
            return False, "FFT not reconstructed (self.F is None)."
        if self.logmag is None:
            return False, "Log magnitude not computed."
        imag_norm = np.linalg.norm(np.imag(self.F))
        if imag_norm == 0:
            return True, "Phase appears to be zero (pure real FFT) — this may be valid."
        return True, "Phase loaded and FFT reconstructed."


# ---------- Main ----------

def main():
    app = QApplication(sys.argv)
    picker = FilePicker()
    picker.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()