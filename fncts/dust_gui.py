import matplotlib
matplotlib.use("Agg")

#!/usr/bin/env python3
import sys
import os
import csv
import traceback
from pathlib import Path
from functools import partial

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

from astropy.stats import sigma_clipped_stats, mad_std

# photutils / sep / fallback imports
try:
    from photutils.aperture import CircularAperture, aperture_photometry
    PHOTUTILS_HAS_APERTURE = True
except Exception:
    PHOTUTILS_HAS_APERTURE = False

try:
    import sep
    SEP_AVAILABLE = True
except Exception:
    SEP_AVAILABLE = False

# DAOStarFinder remains in photutils detection module if available
try:
    from photutils.detection import DAOStarFinder
except Exception:
    DAOStarFinder = None

import matplotlib
matplotlib.use("Agg")
from PyQt6 import QtWidgets, QtCore, QtGui
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# ---------------------
# Utility functions
# ---------------------
def read_fits_rgb(path):
    hdul = fits.open(path, memmap=False)
    if len(hdul) == 1:
        data = hdul[0].data
    else:
        try:
            data = np.array([hdu.data for hdu in hdul if hdu.data is not None])
        except Exception:
            data = hdul[0].data
    hdul.close()
    if data is None:
        raise ValueError("No image data found")
    if data.ndim == 2:
        data = np.stack([data, data, data], axis=0)
    if data.ndim == 3:
        if data.shape[0] == 3:
            return data
        if data.shape[-1] == 3:
            return np.transpose(data, (2, 0, 1))
    raise ValueError("Unsupported FITS channel layout: shape %s" % (data.shape,))

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def write_csv(path, rows, headers):
    with open(path, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(headers)
        writer.writerows(rows)

# ---------------------
# Aperture helpers (photutils / sep / mask fallback)
# ---------------------
def circular_aperture_sums_fast(image, x_arr, y_arr, r):
    x = np.asarray(x_arr, dtype=float)
    y = np.asarray(y_arr, dtype=float)
    if PHOTUTILS_HAS_APERTURE:
        positions = list(zip(x, y))
        apertures = CircularAperture(positions, r=r)
        phot = aperture_photometry(image, apertures)
        return np.array(phot['aperture_sum']), phot
    if SEP_AVAILABLE:
        img32 = image.astype(np.float32)
        fluxes = sep.sum_circle(img32, x.astype(np.float32), y.astype(np.float32), r)
        phot = {'aperture_sum': fluxes, 'x': x, 'y': y}
        return fluxes, phot
    h, w = image.shape
    yy, xx = np.mgrid[0:h, 0:w]
    sums = []
    for xi, yi in zip(x, y):
        mask = (xx - xi)**2 + (yy - yi)**2 <= r*r
        sums.append(image[mask].sum())
    fluxes = np.array(sums)
    phot = {'aperture_sum': fluxes, 'x': x, 'y': y}
    return fluxes, phot

# ---------------------
# Photometry and maps
# ---------------------
def detect_and_photometer(channel_data, fwhm=3.0, threshold_sigma=5.0, aperture_radius=4.0):
    mean, median, std = sigma_clipped_stats(channel_data, sigma=3.0)
    bkg_sigma = std if std > 0 else mad_std(channel_data)
    if DAOStarFinder is not None:
        daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold_sigma * bkg_sigma)
        sources = daofind(channel_data - median)
        if sources is None or len(sources) == 0:
            return Table(), None
        x = np.array(sources['xcentroid'], dtype=float)
        y = np.array(sources['ycentroid'], dtype=float)
        fluxes, phot_like = circular_aperture_sums_fast(channel_data - median, x, y, aperture_radius)
        t = Table()
        t['x'] = x
        t['y'] = y
        t['aperture_sum'] = fluxes
        return t, sources
    if SEP_AVAILABLE:
        img = channel_data.astype(np.float32)
        bkg = sep.Background(img)
        img_sub = img - bkg.back()
        objects = sep.extract(img_sub, threshold_sigma * bkg.rms(), err=bkg.rms())
        if objects is None or len(objects) == 0:
            return Table(), None
        x = objects['x']
        y = objects['y']
        fluxes, phot_like = circular_aperture_sums_fast(img_sub, x, y, aperture_radius)
        t = Table()
        t['x'] = x
        t['y'] = y
        t['aperture_sum'] = fluxes
        return t, objects
    h, w = channel_data.shape
    coords = []
    mean, median, std = sigma_clipped_stats(channel_data, sigma=3.0)
    bkg_sigma = std if std > 0 else mad_std(channel_data)
    thresh = median + threshold_sigma * bkg_sigma
    for iy in range(1, h-1):
        for ix in range(1, w-1):
            val = channel_data[iy, ix]
            if val > thresh and val > channel_data[iy-1,ix] and val > channel_data[iy+1,ix] and val > channel_data[iy,ix-1] and val > channel_data[iy,ix+1]:
                coords.append((ix, iy))
    if not coords:
        return Table(), None
    x = np.array([c[0] for c in coords], dtype=float)
    y = np.array([c[1] for c in coords], dtype=float)
    fluxes, phot_like = circular_aperture_sums_fast(channel_data - median, x, y, aperture_radius)
    t = Table()
    t['x'] = x
    t['y'] = y
    t['aperture_sum'] = fluxes
    return t, None

def compute_instrumental_magnitudes(phot_table, zero_point=25.0, flux_col='aperture_sum'):
    flux = np.array(phot_table[flux_col])
    with np.errstate(divide='ignore', invalid='ignore'):
        mag = zero_point - 2.5 * np.log10(np.where(flux>0, flux, np.nan))
    return mag

def compute_color_indices_from_fluxes(flux_r, flux_g, flux_b, zp=25.0):
    with np.errstate(divide='ignore', invalid='ignore'):
        mag_r = zp - 2.5*np.log10(np.where(flux_r>0, flux_r, np.nan))
        mag_g = zp - 2.5*np.log10(np.where(flux_g>0, flux_g, np.nan))
        mag_b = zp - 2.5*np.log10(np.where(flux_b>0, flux_b, np.nan))
    rg = mag_r - mag_g
    gb = mag_g - mag_b
    return rg, gb, (mag_r, mag_g, mag_b)

def make_reddening_map(image_shape, stars_xy, color_index, sigma_smooth=3.0):
    h, w = image_shape
    im = np.full((h, w), np.nan, dtype=float)
    weight = np.zeros((h, w), dtype=float)
    for (x, y), c in zip(stars_xy, color_index):
        ix, iy = int(round(y)), int(round(x))
        if 0 <= ix < h and 0 <= iy < w and np.isfinite(c):
            im[ix, iy] = c
            weight[ix, iy] = 1.0
    filled = np.nan_to_num(im, nan=0.0)
    smooth = gaussian_filter(filled, sigma=sigma_smooth)
    w_smooth = gaussian_filter(weight, sigma=sigma_smooth)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(w_smooth>0, smooth / w_smooth, np.nan)
    return result

# ---------------------
# Significance map utilities
# ---------------------
def mag_uncertainty_from_flux(flux, sky_rms, npix_aperture, gain=1.0):
    flux_e = flux * gain
    skyvar_e = (sky_rms**2) * gain
    var = flux_e + npix_aperture * skyvar_e
    var = np.where(var>0, var, np.nan)
    snr = flux_e / np.sqrt(var)
    with np.errstate(divide='ignore', invalid='ignore'):
        sigma_mag = 1.0857 / snr
    return sigma_mag, snr

def build_weighted_map(image_shape, stars_xy, values, sigmas, smoothing_sigma=3.0, min_weight=1e-8):
    h, w = image_shape
    val_img = np.zeros((h, w), dtype=float)
    weight_img = np.zeros((h, w), dtype=float)
    count_img = np.zeros((h, w), dtype=float)
    for (x, y), v, s in zip(stars_xy, values, sigmas):
        if not np.isfinite(v) or not np.isfinite(s) or s <= 0:
            continue
        ix, iy = int(round(y)), int(round(x))
        if 0 <= ix < h and 0 <= iy < w:
            w_i = 1.0 / (s*s)
            val_img[ix, iy] += v * w_i
            weight_img[ix, iy] += w_i
            count_img[ix, iy] += 1.0
    num_smooth = gaussian_filter(val_img, sigma=smoothing_sigma, mode='constant', truncate=4.0)
    w_smooth = gaussian_filter(weight_img, sigma=smoothing_sigma, mode='constant', truncate=4.0)
    n_smooth = gaussian_filter(count_img, sigma=smoothing_sigma, mode='constant', truncate=4.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        E_map = np.where(w_smooth > min_weight, num_smooth / w_smooth, np.nan)
        sigma_map = np.where(w_smooth > min_weight, 1.0 / np.sqrt(w_smooth), np.nan)
    return E_map, sigma_map, w_smooth, n_smooth

def make_significance_map(image_shape, stars_xy, E_values, perstar_sigmas,
                          smoothing_sigma=3.0, min_stars=3, min_weight=1e-8):
    E_map, sigma_map, weight_map, nmap = build_weighted_map(image_shape, stars_xy, E_values, perstar_sigmas,
                                                            smoothing_sigma=smoothing_sigma, min_weight=min_weight)
    with np.errstate(divide='ignore', invalid='ignore'):
        S_map = E_map / sigma_map
    mask = (nmap < min_stars) | (~np.isfinite(S_map))
    S_map_masked = np.ma.array(S_map, mask=mask)
    return E_map, sigma_map, S_map_masked, weight_map, nmap

# ---------------------
# Worker for batch processing (QThread)
# ---------------------
class BatchWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, str)
    finished_signal = QtCore.pyqtSignal(list)
    error_signal = QtCore.pyqtSignal(str)

    def __init__(self, paths, outdir, params):
        super().__init__()
        self.paths = paths
        self.outdir = outdir
        self.params = params
        self._stop = False

    def run(self):
        outputs = []
        try:
            for idx, p in enumerate(self.paths):
                if self._stop:
                    break
                self.progress.emit(int((idx/len(self.paths))*100), f"Processing {Path(p).name}")
                try:
                    out = self.process_one(p)
                    outputs.append(out)
                except Exception as e:
                    tb = traceback.format_exc()
                    self.progress.emit(int((idx/len(self.paths))*100), f"Error {Path(p).name}: {e}")
                    self.error_signal.emit(f"{p}: {e}\n{tb}")
                QtCore.QThread.msleep(200)
            self.progress.emit(100, "Done")
            self.finished_signal.emit(outputs)
        except Exception as e:
            self.error_signal.emit(str(e))

    def stop(self):
        self._stop = True

    def process_one(self, path):
        data = read_fits_rgb(path)
        mapping = self.params.get('channel_map', (0,1,2))
        rch = data[mapping[0]]
        gch = data[mapping[1]]
        bch = data[mapping[2]]
        phot_g, src_g = detect_and_photometer(gch,
                                              fwhm=self.params.get('fwhm', 3.0),
                                              threshold_sigma=self.params.get('threshold', 5.0),
                                              aperture_radius=self.params.get('aperture', 4.0))
        if len(phot_g) == 0:
            raise RuntimeError("No sources detected")
        positions_x = np.array(phot_g['x'])
        positions_y = np.array(phot_g['y'])
        flux_r, _ = circular_aperture_sums_fast(rch, positions_x, positions_y, self.params.get('aperture', 4.0))
        flux_b, _ = circular_aperture_sums_fast(bch, positions_x, positions_y, self.params.get('aperture', 4.0))
        flux_g, _ = circular_aperture_sums_fast(gch, positions_x, positions_y, self.params.get('aperture', 4.0))
        rg, gb, mags = compute_color_indices_from_fluxes(flux_r, flux_g, flux_b, zp=self.params.get('zp', 25.0))
        orig_path = str(Path(path).resolve())
        rows = []
        headers = ['original_file','x_pixel','y_pixel','flux_r','flux_g','flux_b','mag_r','mag_g','mag_b','color_R-G','color_G-B']
        for i in range(len(positions_x)):
            row = [orig_path,
                   positions_x[i], positions_y[i],
                   flux_r[i], flux_g[i], flux_b[i],
                   mags[0][i], mags[1][i], mags[2][i],
                   rg[i], gb[i]]
            rows.append(row)
        ensure_dir(self.outdir)
        base = Path(path).stem
        csv_out = os.path.join(self.outdir, f"{base}_photometry.csv")
        write_csv(csv_out, rows, headers)
        imshape = (rch.shape[0], rch.shape[1])
        stars_xy = list(zip(positions_x, positions_y))
        redmap = make_reddening_map(imshape, stars_xy, rg, sigma_smooth=self.params.get('smooth', 3.0))
        png_out = os.path.join(self.outdir, f"{base}_reddening.png")
        plt.figure(figsize=(6,6))
        plt.imshow(redmap, origin='lower', cmap='inferno')
        plt.colorbar(label='R-G color (mag)')
        plt.title(base + " R-G map")
        plt.savefig(png_out, dpi=150, bbox_inches='tight')
        plt.close()
        return {'original_file': orig_path, 'path': path, 'csv': csv_out, 'png': png_out, 'stars_xy': stars_xy, 'rg': rg, 'flux_r': flux_r, 'flux_g': flux_g, 'flux_b': flux_b}

# ---------------------
# Main GUI
# ---------------------
class DustGui(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RGB Dust Detection")
        self.resize(1000, 700)
        self._build_ui()
        self.worker = None

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        file_row = QtWidgets.QHBoxLayout()
        self.file_list = QtWidgets.QListWidget()
        file_buttons = QtWidgets.QVBoxLayout()
        add_btn = QtWidgets.QPushButton("Add Files")
        add_btn.clicked.connect(self.add_files)
        clear_btn = QtWidgets.QPushButton("Clear")
        clear_btn.clicked.connect(self.file_list.clear)
        file_buttons.addWidget(add_btn)
        file_buttons.addWidget(clear_btn)
        file_row.addWidget(self.file_list, stretch=3)
        file_row.addLayout(file_buttons, stretch=1)
        layout.addLayout(file_row)

        params_group = QtWidgets.QGroupBox("Parameters")
        pg = QtWidgets.QGridLayout()
        params_group.setLayout(pg)
        pg.addWidget(QtWidgets.QLabel("Channel map (R,G,B indices)"), 0, 0)
        self.channel_map_edit = QtWidgets.QLineEdit("0,1,2")
        pg.addWidget(self.channel_map_edit, 0, 1)
        pg.addWidget(QtWidgets.QLabel("Aperture radius (px)"), 1, 0)
        self.aperture_spin = QtWidgets.QDoubleSpinBox()
        self.aperture_spin.setRange(1.0, 50.0)
        self.aperture_spin.setValue(4.0)
        pg.addWidget(self.aperture_spin, 1, 1)
        pg.addWidget(QtWidgets.QLabel("Detection threshold (sigma)"), 2, 0)
        self.threshold_spin = QtWidgets.QDoubleSpinBox()
        self.threshold_spin.setRange(1.0, 20.0)
        self.threshold_spin.setValue(5.0)
        pg.addWidget(self.threshold_spin, 2, 1)
        pg.addWidget(QtWidgets.QLabel("FWHM (px)"), 3, 0)
        self.fwhm_spin = QtWidgets.QDoubleSpinBox()
        self.fwhm_spin.setRange(1.0, 50.0)
        self.fwhm_spin.setValue(3.0)
        pg.addWidget(self.fwhm_spin, 3, 1)
        pg.addWidget(QtWidgets.QLabel("Smoothing sigma (map)"), 4, 0)
        self.smooth_spin = QtWidgets.QDoubleSpinBox()
        self.smooth_spin.setRange(0.5, 20.0)
        self.smooth_spin.setValue(3.0)
        pg.addWidget(self.smooth_spin, 4, 1)
        pg.addWidget(QtWidgets.QLabel("Photometric zero point"), 5, 0)
        self.zp_spin = QtWidgets.QDoubleSpinBox()
        self.zp_spin.setRange(0.0, 40.0)
        self.zp_spin.setValue(25.0)
        pg.addWidget(self.zp_spin, 5, 1)
        layout.addWidget(params_group)

        out_row = QtWidgets.QHBoxLayout()
        out_row.addWidget(QtWidgets.QLabel("Output folder"))
        self.output_edit = QtWidgets.QLineEdit(str(Path.cwd()))
        out_row.addWidget(self.output_edit)
        out_btn = QtWidgets.QPushButton("Browse")
        out_btn.clicked.connect(self.browse_output)
        out_row.addWidget(out_btn)
        layout.addLayout(out_row)

        ctrl_row = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Start Batch")
        self.start_btn.clicked.connect(self.start_batch)
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_batch)
        self.signif_btn = QtWidgets.QPushButton("Compute Significance Map for Selected")
        self.signif_btn.clicked.connect(self.compute_significance_for_selected)
        ctrl_row.addWidget(self.start_btn)
        ctrl_row.addWidget(self.stop_btn)
        ctrl_row.addWidget(self.signif_btn)
        layout.addLayout(ctrl_row)

        self.progress_bar = QtWidgets.QProgressBar()
        layout.addWidget(self.progress_bar)
        self.log_text = QtWidgets.QPlainTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text, stretch=1)

    def add_files(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Select FITS files", str(Path.cwd()), "FITS files (*.fits *.fit *.fz);;All files (*)")
        for p in paths:
            self.file_list.addItem(p)

    def browse_output(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output folder", str(Path.cwd()))
        if d:
            self.output_edit.setText(d)

    def start_batch(self):
        files = [self.file_list.item(i).text() for i in range(self.file_list.count())]
        if not files:
            self.log("No files to process")
            return
        try:
            ch_text = self.channel_map_edit.text()
            mapping = tuple(int(x.strip()) for x in ch_text.split(","))
        except Exception:
            self.log("Channel map invalid, using default 0,1,2")
            mapping = (0,1,2)
        params = {
            'channel_map': mapping,
            'aperture': self.aperture_spin.value(),
            'threshold': self.threshold_spin.value(),
            'fwhm': self.fwhm_spin.value(),
            'smooth': self.smooth_spin.value(),
            'zp': self.zp_spin.value()
        }
        outdir = self.output_edit.text()
        ensure_dir(outdir)
        self.worker = BatchWorker(files, outdir, params)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.error_signal.connect(self.on_error)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.worker.start()
        self.log(f"Started batch of {len(files)} files")

    def stop_batch(self):
        if self.worker:
            self.worker.stop()
            self.log("Stop requested")
            self.stop_btn.setEnabled(False)

    def on_progress(self, pct, message):
        self.progress_bar.setValue(pct)
        self.log(f"{pct}% - {message}")

    def on_finished(self, outputs):
        self.log("Batch finished")
        for out in outputs:
            self.log(f"Produced: {out.get('csv')} and {out.get('png')} (from {out.get('original_file')})")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def on_error(self, msg):
        self.log("ERROR: " + msg)

    def log(self, text):
        self.log_text.appendPlainText(text)

    # ---------------------
    # Significance map GUI action
    # ---------------------
    def compute_significance_for_selected(self):
        item = self.file_list.currentItem()
        if item is None:
            self.log("Select a file in the list first")
            return
        path = item.text()
        try:
            data = read_fits_rgb(path)
        except Exception as e:
            self.log(f"Failed to read file: {e}")
            return
        try:
            mapping = tuple(int(x.strip()) for x in self.channel_map_edit.text().split(","))
        except Exception:
            mapping = (0,1,2)
        rch = data[mapping[0]]
        gch = data[mapping[1]]
        bch = data[mapping[2]]
        phot_g, src_g = detect_and_photometer(gch,
                                              fwhm=self.fwhm_spin.value(),
                                              threshold_sigma=self.threshold_spin.value(),
                                              aperture_radius=self.aperture_spin.value())
        if len(phot_g) == 0:
            self.log("No stars detected for significance map")
            return
        positions_x = np.array(phot_g['x'])
        positions_y = np.array(phot_g['y'])
        flux_r, _ = circular_aperture_sums_fast(rch, positions_x, positions_y, self.aperture_spin.value())
        flux_b, _ = circular_aperture_sums_fast(bch, positions_x, positions_y, self.aperture_spin.value())
        flux_g, _ = circular_aperture_sums_fast(gch, positions_x, positions_y, self.aperture_spin.value())
        rg, gb, mags = compute_color_indices_from_fluxes(flux_r, flux_g, flux_b, zp=self.zp_spin.value())
        # estimate sky rms from channel using sigma_clipped_stats on a binned background
        _, med_r, std_r = sigma_clipped_stats(rch, sigma=3.0)
        _, med_g, std_g = sigma_clipped_stats(gch, sigma=3.0)
        _, med_b, std_b = sigma_clipped_stats(bch, sigma=3.0)
        # approximate aperture pixel count
        ap_r = float(self.aperture_spin.value())
        npix_aperture = max(1.0, np.pi * (ap_r**2))
        # gain placeholder; user should set actual gain per camera for realistic uncertainties
        GAIN = 1.0
        sigma_r, snr_r = mag_uncertainty_from_flux(flux_r, std_r, npix_aperture, gain=GAIN)
        sigma_g, snr_g = mag_uncertainty_from_flux(flux_g, std_g, npix_aperture, gain=GAIN)
        sigma_rg = np.sqrt(np.nan_to_num(sigma_r**2) + np.nan_to_num(sigma_g**2))
        stars_xy = list(zip(positions_x, positions_y))
        imshape = (rch.shape[0], rch.shape[1])
        smoothing = self.smooth_spin.value()
        E_map, sigma_map, S_masked, weight_map, nmap = make_significance_map(imshape, stars_xy, rg, sigma_rg,
                                                                              smoothing_sigma=smoothing, min_stars=3)
        # plotting three panels
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        im0 = axes[0].imshow(E_map, origin='lower', cmap='plasma')
        axes[0].set_title('E(R-G) map (mag)')
        plt.colorbar(im0, ax=axes[0], fraction=0.046)
        im1 = axes[1].imshow(sigma_map, origin='lower', cmap='viridis')
        axes[1].set_title('Sigma map (mag)')
        plt.colorbar(im1, ax=axes[1], fraction=0.046)
        # compute robust vlim from unmasked significance values, fallback to 3.0
        sig_data = np.abs(S_masked.data)[~S_masked.mask]
        if sig_data.size == 0 or np.all(np.isnan(sig_data)):
            vlim = 3.0
        else:
            vlim = max(3.0, float(np.nanpercentile(sig_data, 95)))
        im2 = axes[2].imshow(S_masked, origin='lower', cmap='bwr', vmin=-vlim, vmax=vlim)
        axes[2].set_title('Significance S = E / sigma (masked)')
        plt.colorbar(im2, ax=axes[2], fraction=0.046)
        plt.suptitle(f"Significance map for {Path(path).name}")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
        # optionally save PNG next to output folder
        outdir = self.output_edit.text() or str(Path.cwd())
        base = Path(path).stem
        ensure_dir(outdir)
        png_out = os.path.join(outdir, f"{base}_significance.png")
        fig.savefig(png_out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        self.log(f"Saved significance figure to {png_out}")

# ---------------------
# Run
# ---------------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    gui = DustGui()
    gui.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()