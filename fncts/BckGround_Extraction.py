# batch_background_siril_match_patched.py
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter, binary_dilation
import matplotlib.pyplot as plt

# ---------- Defaults and constants ----------
DEFAULT_WORKDIR = os.path.expanduser("~")
DEFAULT_POLY_DEG = 2
DEFAULT_SAMPLES_PER_LINE = 30
DEFAULT_SIGMA_CLIP = 3.0
DEFAULT_GRID_TOL = 0.3
DEFAULT_SMOOTH_SIGMA = 0.5
MAX_FEEDBACK_ITERS = 8
MIN_ACCEPTED_CELLS = 40
MIN_ACCEPTED_PER_ROW = 2
MIN_ACCEPTED_PER_COL = 2
# ------------------------------------------------

# ---------- Utility functions ----------
def robust_sigma(data):
    med = np.median(data)
    mad = np.median(np.abs(data - med))
    return 1.4826 * mad

def star_mask_simple(data, threshold_sigma=5.0, dilation_radius=2):
    med = np.median(data)
    sigma = robust_sigma(data)
    if sigma <= 0:
        return np.zeros_like(data, dtype=bool)
    thr = med + threshold_sigma * sigma
    mask = data > thr
    neigh = np.zeros_like(mask, dtype=int)
    padded = np.pad(mask, 1, mode='constant', constant_values=0)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            neigh += padded[1+dy:1+dy+mask.shape[0], 1+dx:1+dx+mask.shape[1]]
    mask = mask & (neigh > 0)
    if dilation_radius > 0:
        structure = np.ones((2 * dilation_radius + 1, 2 * dilation_radius + 1), dtype=bool)
        mask = binary_dilation(mask, structure=structure)
    return mask

def build_grid_samples(data, samples_per_line, sigma_clip, grid_tol, star_mask=None, min_pixels_per_cell=10):
    ny, nx = data.shape
    xs = np.linspace(0, nx - 1, samples_per_line)
    ys = np.linspace(0, ny - 1, samples_per_line)
    cell_w = nx / samples_per_line
    cell_h = ny / samples_per_line

    xs_out, ys_out, zs_out = [], [], []
    mask_grid = np.zeros((samples_per_line, samples_per_line), dtype=bool)
    cell_counts = np.zeros_like(mask_grid, dtype=int)

    global_med = np.median(data)
    global_sig = robust_sigma(data)

    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            x0 = int(max(0, np.floor(i * cell_w)))
            x1 = int(min(nx, np.floor((i + 1) * cell_w)))
            y0 = int(max(0, np.floor(j * cell_h)))
            y1 = int(min(ny, np.floor((j + 1) * cell_h)))

            cell = data[y0:y1, x0:x1].ravel()
            if star_mask is not None:
                sm = star_mask[y0:y1, x0:x1].ravel()
                cell = cell[~sm]
            cell = cell[np.isfinite(cell)]
            cell_counts[j, i] = cell.size
            if cell.size < min_pixels_per_cell:
                continue

            med = np.median(cell)
            std = np.std(cell)
            if std <= 0:
                continue

            local_k = sigma_clip
            good = cell[(cell > med - local_k * std) & (cell < med + local_k * std)]
            frac_good = good.size / cell.size if cell.size > 0 else 0.0

            # brightness-aware: if cell is much brighter than global, be more tolerant
            if med > global_med + 2 * global_sig and frac_good < grid_tol:
                # keep slightly more pixels in bright regions
                frac_good = max(frac_good, grid_tol * 0.8)

            if good.size < 5 or frac_good < grid_tol:
                continue

            bg_val = np.median(good)
            xs_out.append(x)
            ys_out.append(y)
            zs_out.append(bg_val)
            mask_grid[j, i] = True

    if len(xs_out) == 0:
        raise RuntimeError("No grid cells survived sampling; relax parameters.")

    return np.array(xs_out), np.array(ys_out), np.array(zs_out), mask_grid, cell_counts

def fit_2d_poly(xs, ys, zs, degree):
    terms = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            terms.append((i, j))
    A = np.zeros((xs.size, len(terms)), dtype=float)
    for k, (i, j) in enumerate(terms):
        A[:, k] = (xs ** i) * (ys ** j)
    coeffs, _, _, _ = np.linalg.lstsq(A, zs, rcond=None)
    return coeffs, terms

def eval_2d_poly(coeffs, terms, shape):
    ny, nx = shape
    yy, xx = np.indices((ny, nx), dtype=float)
    bg = np.zeros((ny, nx), dtype=float)
    for c, (i, j) in zip(coeffs, terms):
        bg += c * (xx ** i) * (yy ** j)
    return bg

def recommend_samples_from_image_shape(nx, ny, target_pixels=100, s_min=4):
    total_pixels = max(1, nx * ny)
    s_est = int(np.floor(np.sqrt(total_pixels / max(1, target_pixels))))
    s_max = max(s_min, min(nx, ny) // 2)
    s = max(s_min, min(s_est, s_max))
    return int(s)
# ---------------------------------------

# --------------- Feedback loop core ----------------
def feedback_background_model(data, initial_params, debug_logger=None, siril_match=False):
    params = initial_params.copy()
    best = None
    diagnostics = []
    ny, nx = data.shape
    sigma_global = robust_sigma(data)

    starmask = star_mask_simple(data, threshold_sigma=5.0, dilation_radius=2)

    for it in range(MAX_FEEDBACK_ITERS):
        # clamp samples
        s_requested = int(params.get('samples', 30))
        s_min = 4
        s_max = max(s_min, min(ny, nx) // 2)
        s_clamped = int(max(s_min, min(s_requested, s_max)))
        params['samples'] = s_clamped

        if debug_logger:
            debug_logger(f"Iter {it}: nx={nx} ny={ny} samples_req={s_requested} samples_used={s_clamped}")

        try:
            xs, ys, zs, mask_grid, cell_counts = build_grid_samples(
                data,
                samples_per_line=params['samples'],
                sigma_clip=params['sigma_clip'],
                grid_tol=params['grid_tol'],
                star_mask=starmask
            )
        except Exception as e:
            params['sigma_clip'] = min(5.0, params['sigma_clip'] + 0.5)
            params['grid_tol'] = max(0.05, params['grid_tol'] - 0.05)
            if debug_logger:
                debug_logger(f"Iter {it}: sampling failed, relaxing -> sigma={params['sigma_clip']:.2f}, tol={params['grid_tol']:.2f}")
            continue

        accepted = mask_grid.sum()
        frac = accepted / mask_grid.size

        rows_accepted = mask_grid.sum(axis=1)
        cols_accepted = mask_grid.sum(axis=0)
        rows_ok = np.sum(rows_accepted >= MIN_ACCEPTED_PER_ROW)
        cols_ok = np.sum(cols_accepted >= MIN_ACCEPTED_PER_COL)

        coeffs, terms = fit_2d_poly(xs, ys, zs, degree=params['degree'])
        bg = eval_2d_poly(coeffs, terms, data.shape)
        if params['smooth_sigma'] > 0:
            bg = gaussian_filter(bg, params['smooth_sigma'])

        residual = data - bg
        res_med = np.median(residual[np.isfinite(residual)])
        res_std = np.std(residual[np.isfinite(residual)])
        res_skew = np.mean(((residual - res_med) ** 3)[np.isfinite(residual)]) / (res_std ** 3 + 1e-12)

        # histogram centering metrics
        neg_frac = np.mean(residual < 0)
        pos_frac = np.mean(residual > 0)
        balance = abs(neg_frac - pos_frac)

        # coarse variance
        try:
            s = params['samples']
            cell_w = float(nx) / s
            cell_h = float(ny) / s
            coarse_meds = []
            for j in range(s):
                y0 = int(round(j * cell_h))
                y1 = int(round((j + 1) * cell_h)) if j < s - 1 else ny
                for i in range(s):
                    x0 = int(round(i * cell_w))
                    x1 = int(round((i + 1) * cell_w)) if i < s - 1 else nx
                    block = residual[y0:y1, x0:x1]
                    block = block[np.isfinite(block)]
                    if block.size:
                        coarse_meds.append(np.median(block))
            if len(coarse_meds) > 0:
                coarse_var = float(np.nanvar(coarse_meds))
            else:
                coarse_var = float(np.nanvar(residual))
        except Exception:
            coarse_var = float(np.nanvar(residual))

        diag = {
            'iter': it,
            'accepted_cells': int(accepted),
            'accepted_frac': float(frac),
            'rows_ok': int(rows_ok),
            'cols_ok': int(cols_ok),
            'res_med': float(res_med),
            'res_std': float(res_std),
            'res_skew': float(res_skew),
            'neg_frac': float(neg_frac),
            'pos_frac': float(pos_frac),
            'balance': float(balance),
            'coarse_var': float(coarse_var),
            'params': params.copy()
        }
        diagnostics.append(diag)
        if debug_logger:
            debug_logger(
                f"Iter {it}: acc={accepted} frac={frac:.3f} rows_ok={rows_ok}/{params['samples']} "
                f"cols_ok={cols_ok}/{params['samples']} res_med={res_med:.3f} res_std={res_std:.2f} "
                f"skew={res_skew:.2f} balance={balance:.2f}"
            )

        enough_cells = accepted >= max(MIN_ACCEPTED_CELLS, int(0.2 * params['samples'] ** 2))
        coverage_ok = (rows_ok >= int(0.6 * params['samples'])) and (cols_ok >= int(0.6 * params['samples']))

        if siril_match:
            residual_ok = (
                abs(res_med) < 0.1 * sigma_global and
                abs(res_skew) < 0.5 and
                balance < 0.15 and
                res_std < 3.0 * sigma_global
            )
        else:
            residual_ok = (
                abs(res_skew) < 0.8 and
                res_std < max(3.0, 3.0 * sigma_global)
            )

        if enough_cells and coverage_ok and residual_ok:
            best = {'bg': bg, 'params': params.copy(), 'diag': diag}
            if debug_logger:
                debug_logger(f"Iter {it}: Accepting model.")
            break

        # parameter adjustments
        if not enough_cells:
            params['sigma_clip'] = min(5.0, params['sigma_clip'] + 0.5)
            params['grid_tol'] = max(0.05, params['grid_tol'] - 0.05)
            if debug_logger:
                debug_logger(f"Iter {it}: Too few cells -> sigma {params['sigma_clip']:.2f}, tol {params['grid_tol']:.2f}")
        else:
            # histogram centering
            if res_med > 0:  # background too low
                params['smooth_sigma'] = min(3.0, params['smooth_sigma'] + 0.2)
                params['sigma_clip'] = min(5.0, params['sigma_clip'] + 0.3)
                if params['degree'] < 3:
                    params['degree'] += 1
                if debug_logger:
                    debug_logger(f"Iter {it}: res_med>0 -> raise bg via smoothing/degree/clip")
            elif res_med < 0:  # background too high
                params['smooth_sigma'] = max(0.0, params['smooth_sigma'] - 0.2)
                params['sigma_clip'] = max(1.5, params['sigma_clip'] - 0.3)
                if params['degree'] > 1:
                    params['degree'] -= 1
                if debug_logger:
                    debug_logger(f"Iter {it}: res_med<0 -> lower bg via smoothing/degree/clip")

            if balance > 0.2:
                params['sigma_clip'] = min(5.0, params['sigma_clip'] + 0.5)
                params['grid_tol'] = max(0.05, params['grid_tol'] - 0.05)
                if debug_logger:
                    debug_logger(f"Iter {it}: histogram imbalance {balance:.2f} -> loosen clip/tol")

            if abs(res_skew) > 1.2:
                if res_skew > 0:
                    params['sigma_clip'] = max(1.5, params['sigma_clip'] - 0.3)
                else:
                    params['sigma_clip'] = min(5.0, params['sigma_clip'] + 0.3)
                if debug_logger:
                    debug_logger(f"Iter {it}: skew {res_skew:.2f} -> adjust sigma_clip {params['sigma_clip']:.2f}")

        params['sigma_clip'] = float(np.clip(params['sigma_clip'], 1.5, 5.0))
        params['grid_tol'] = float(np.clip(params['grid_tol'], 0.05, 0.8))
        params['smooth_sigma'] = float(np.clip(params['smooth_sigma'], 0.0, 3.0))
        params['degree'] = int(np.clip(params['degree'], 1, 4))

    if best is None:
        try:
            best = {'bg': bg, 'params': params.copy(), 'diag': diagnostics[-1] if diagnostics else {}}
        except Exception:
            raise RuntimeError("Feedback loop failed to produce a background model.")

    return best['bg'], best['params'], diagnostics, starmask, mask_grid

# ---------------- GUI and batch code ----------------
class BackgroundBatchGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Batch Background Extraction - Siril Match (Patched)")
        self.geometry("980x720")
        self.workdir = DEFAULT_WORKDIR
        self.output_dir = ""
        self.files = []
        self._stop_flag = threading.Event()
        self.last_bg = None
        self.last_residual = None
        self.last_mask_grid = None
        self.last_starmask = None
        self.last_diag = None
        self._last_suggestion = None
        self.create_widgets()
        self.refresh_list()

    def create_widgets(self):
        frm_top = tk.Frame(self)
        frm_top.pack(fill="x", padx=8, pady=6)

        tk.Label(frm_top, text="Work folder:").grid(row=0, column=0, sticky="w")
        self.dir_entry = tk.Entry(frm_top, width=64)
        self.dir_entry.insert(0, self.workdir)
        self.dir_entry.grid(row=0, column=1, padx=4)
        tk.Button(frm_top, text="Choose", command=self.choose_folder).grid(row=0, column=2)

        tk.Label(frm_top, text="Output folder (optional):").grid(row=1, column=0, sticky="w")
        self.out_entry = tk.Entry(frm_top, width=64)
        self.out_entry.grid(row=1, column=1, padx=4)
        tk.Button(frm_top, text="Choose", command=self.choose_output).grid(row=1, column=2)

        params = tk.Frame(self)
        params.pack(fill="x", padx=8, pady=6)

        tk.Label(params, text="Polynomial degree:").grid(row=0, column=0, sticky="e")
        self.poly_entry = tk.Entry(params, width=8)
        self.poly_entry.insert(0, str(DEFAULT_POLY_DEG))
        self.poly_entry.grid(row=0, column=1, padx=4)

        tk.Label(params, text="Samples per line:").grid(row=0, column=2, sticky="e")
        self.samples_entry = tk.Entry(params, width=8)
        self.samples_entry.insert(0, str(DEFAULT_SAMPLES_PER_LINE))
        self.samples_entry.grid(row=0, column=3, padx=4)

        tk.Label(params, text="Sigma-clip (k):").grid(row=0, column=4, sticky="e")
        self.sigma_entry = tk.Entry(params, width=8)
        self.sigma_entry.insert(0, str(DEFAULT_SIGMA_CLIP))
        self.sigma_entry.grid(row=0, column=5, padx=4)

        tk.Label(params, text="Grid tolerance (0-1):").grid(row=1, column=0, sticky="e")
        self.gridtol_entry = tk.Entry(params, width=8)
        self.gridtol_entry.insert(0, str(DEFAULT_GRID_TOL))
        self.gridtol_entry.grid(row=1, column=1, padx=4)

        tk.Label(params, text="Smooth sigma (px):").grid(row=1, column=2, sticky="e")
        self.smooth_entry = tk.Entry(params, width=8)
        self.smooth_entry.insert(0, str(DEFAULT_SMOOTH_SIGMA))
        self.smooth_entry.grid(row=1, column=3, padx=4)

        self.reco_label = tk.Label(params, text="Suggested samples: -")
        self.reco_label.grid(row=0, column=6, padx=8)
        tk.Button(params, text="Apply suggestion", command=self.apply_suggestion).grid(row=0, column=7, padx=4)

        self.save_bg_var = tk.BooleanVar(value=True)
        tk.Checkbutton(params, text="Save background map", variable=self.save_bg_var).grid(row=1, column=4, sticky="w")

        self.save_mask_var = tk.BooleanVar(value=True)
        tk.Checkbutton(params, text="Save grid mask", variable=self.save_mask_var).grid(row=1, column=5, sticky="w")

        self.siril_match_var = tk.BooleanVar(value=True)
        tk.Checkbutton(params, text="Siril‑Match Mode", variable=self.siril_match_var).grid(row=1, column=6, sticky="w")

        mid = tk.Frame(self)
        mid.pack(fill="both", expand=True, padx=8, pady=6)

        left = tk.Frame(mid)
        left.pack(side="left", fill="both", expand=True)
        tk.Label(left, text="FITS files in folder:").pack(anchor="w")
        self.listbox = tk.Listbox(left, selectmode="extended")
        self.listbox.pack(fill="both", expand=True, padx=2, pady=2)

        right = tk.Frame(mid, width=320)
        right.pack(side="right", fill="y")
        tk.Button(right, text="Refresh file list", width=28, command=self.refresh_list).pack(pady=4)
        tk.Button(right, text="Start batch", width=28, command=self.start_batch).pack(pady=4)
        tk.Button(right, text="Stop batch", width=28, command=self.stop_batch).pack(pady=4)
        tk.Button(right, text="Open work folder", width=28, command=self.open_workdir).pack(pady=4)
        tk.Button(right, text="Open output folder", width=28, command=self.open_outputdir).pack(pady=4)
        tk.Button(right, text="Show diagnostics (last file)", width=28, command=self.show_diagnostics).pack(pady=4)
        tk.Button(right, text="Show grid overlay (last file)", width=28, command=self.show_grid_overlay).pack(pady=4)
        tk.Button(right, text="Tuning tips", width=28, command=self.show_tuning_tips).pack(pady=4)

        tk.Label(self, text="Log:").pack(anchor="w", padx=8)
        self.log = scrolledtext.ScrolledText(self, height=12)
        self.log.pack(fill="both", expand=False, padx=8, pady=4)

    def log_write(self, text):
        self.log.insert(tk.END, text + "\n")
        self.log.see(tk.END)

    def choose_folder(self):
        d = filedialog.askdirectory(title="Choose working folder", initialdir=self.workdir)
        if d:
            self.workdir = d
            self.dir_entry.delete(0, tk.END)
            self.dir_entry.insert(0, d)
            self.refresh_list()

    def choose_output(self):
        d = filedialog.askdirectory(title="Choose output folder (optional)", initialdir=self.workdir)
        if d:
            self.output_dir = d
            self.out_entry.delete(0, tk.END)
            self.out_entry.insert(0, d)

    def open_workdir(self):
        os.startfile(self.workdir)

    def open_outputdir(self):
        out = self.out_entry.get().strip()
        if out:
            os.startfile(out)
        else:
            messagebox.showinfo("Output folder", "No output folder set.")

    def refresh_list(self):
        self.listbox.delete(0, tk.END)
        folder = self.dir_entry.get().strip() or self.workdir
        if not os.path.isdir(folder):
            self.log_write(f"Folder not found: {folder}")
            return
        self.workdir = folder
        fits_list = sorted([f for f in os.listdir(folder) if f.lower().endswith((".fit", ".fits"))])
        self.files = fits_list
        for f in fits_list:
            self.listbox.insert(tk.END, f)
        self.log_write(f"Found {len(fits_list)} FITS files in {folder}")

        if self.files:
            first = self.files[0]
            try:
                with fits.open(os.path.join(self.workdir, first), mode="readonly") as hd:
                    ny, nx = hd[0].data.shape
                self.update_suggestion_for_file(nx, ny)
            except Exception:
                self.reco_label.config(text="Suggested samples: -")
                self._last_suggestion = None

    def update_suggestion_for_file(self, nx, ny):
        try:
            target = 100
            s = recommend_samples_from_image_shape(nx, ny, target_pixels=target)
            self.reco_label.config(text=f"Suggested samples: {s} (target {target}px/cell)")
            self._last_suggestion = s
        except Exception:
            self.reco_label.config(text="Suggested samples: -")
            self._last_suggestion = None

    def apply_suggestion(self):
        s = getattr(self, "_last_suggestion", None)
        if s is None:
            messagebox.showinfo("Suggestion", "No suggestion available.")
            return
        self.samples_entry.delete(0, tk.END)
        self.samples_entry.insert(0, str(s))
        self.log_write(f"Applied suggested samples = {s}")

    def start_batch(self):
        if not self.listbox.size():
            messagebox.showwarning("No files", "No FITS files found to process.")
            return

        sel = self.listbox.curselection()
        if sel:
            files = [self.listbox.get(i) for i in sel]
        else:
            files = list(self.files)
        if not files:
            messagebox.showwarning("No files", "No files selected.")
            return

        try:
            degree = int(self.poly_entry.get().strip())
            samples = int(self.samples_entry.get().strip())
            sigma_clip = float(self.sigma_entry.get().strip())
            grid_tol = float(self.gridtol_entry.get().strip())
            smooth_sigma = float(self.smooth_entry.get().strip())
        except ValueError:
            messagebox.showerror("Parameters", "Check that all numeric parameters are valid.")
            return

        self._stop_flag.clear()
        t = threading.Thread(
            target=self._run_batch,
            args=(files, degree, samples, sigma_clip, grid_tol, smooth_sigma),
            daemon=True
        )
        t.start()

    def stop_batch(self):
        self._stop_flag.set()
        self.log_write("Stop requested; current file will finish then stop.")

    def _run_batch(self, files, degree, samples, sigma_clip, grid_tol, smooth_sigma):
        self.log_write(f"Starting batch: {len(files)} files")
        outdir = self.out_entry.get().strip() or ""
        if outdir:
            os.makedirs(outdir, exist_ok=True)

        for idx, fname in enumerate(files, start=1):
            if self._stop_flag.is_set():
                self.log_write("Batch stopped by user.")
                break

            self.log_write(f"[{idx}/{len(files)}] Processing {fname}")
            infile = os.path.join(self.workdir, fname)
            base, ext = os.path.splitext(fname)
            if outdir:
                outfile = os.path.join(outdir, base + "_bgsub" + ext)
                bgfile = os.path.join(outdir, base + "_bg" + ext)
                maskfile = os.path.join(outdir, base + "_mask" + ext)
                starmaskfile = os.path.join(outdir, base + "_starmask" + ext)
            else:
                outfile = os.path.join(self.workdir, base + "_bgsub" + ext)
                bgfile = os.path.join(self.workdir, base + "_bg" + ext)
                maskfile = os.path.join(self.workdir, base + "_mask" + ext)
                starmaskfile = os.path.join(self.workdir, base + "_starmask" + ext)

            try:
                with fits.open(infile, mode="readonly") as hdul:
                    data = hdul[0].data.astype(float)
                    header = hdul[0].header

                initial_params = {
                    'degree': degree,
                    'samples': samples,
                    'sigma_clip': sigma_clip,
                    'grid_tol': grid_tol,
                    'smooth_sigma': smooth_sigma
                }

                def dbglog(s): self.log_write(f"[{base}] {s}")

                ny, nx = data.shape
                self.log_write(f"[{base}] image size: nx={nx}, ny={ny}, requested samples={initial_params['samples']}")
                self.update_suggestion_for_file(nx, ny)

                bg, final_params, diagnostics, starmask, mask_grid = feedback_background_model(
                    data,
                    initial_params,
                    debug_logger=dbglog,
                    siril_match=self.siril_match_var.get()
                )

                corrected = data - bg

                self.last_bg = bg
                self.last_residual = corrected
                self.last_mask_grid = mask_grid
                self.last_starmask = starmask
                self.last_diag = diagnostics

                hdu_corr = fits.PrimaryHDU(corrected, header=header)
                fits.HDUList([hdu_corr]).writeto(outfile, overwrite=True)

                if self.save_bg_var.get():
                    hdu_bg = fits.PrimaryHDU(bg, header=header)
                    fits.HDUList([hdu_bg]).writeto(bgfile, overwrite=True)

                if self.save_mask_var.get():
                    hdu_mask = fits.PrimaryHDU(mask_grid.astype(np.uint8))
                    fits.HDUList([hdu_mask]).writeto(maskfile, overwrite=True)

                hdu_sm = fits.PrimaryHDU(starmask.astype(np.uint8))
                fits.HDUList([hdu_sm]).writeto(starmaskfile, overwrite=True)

                self.log_write(f"Saved: {outfile} (bg saved: {bgfile}) final_params: {final_params}")
            except Exception as e:
                self.log_write(f"Error processing {fname}: {e}")
                with open(os.path.join(self.workdir, "batch_bg_errors.log"), "a", encoding="utf-8") as ef:
                    ef.write(f"{fname}: {e}\n")

        self.log_write("Batch finished.")
        messagebox.showinfo("Batch complete", "Batch processing finished. Check log for details.")

    def show_diagnostics(self):
        if self.last_bg is None or self.last_residual is None or self.last_diag is None:
            messagebox.showinfo("Diagnostics", "No processed image in this session yet.")
            return

        diag = self.last_diag[-1] if isinstance(self.last_diag, list) else self.last_diag

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        im0 = axes[0].imshow(self.last_bg, origin="lower", cmap="magma")
        axes[0].set_title("Background model")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(self.last_residual, origin="lower", cmap="gray",
                             vmin=np.percentile(self.last_residual, 2),
                             vmax=np.percentile(self.last_residual, 98))
        axes[1].set_title("Residual (image - background)")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        # Robust histogram of finite residuals with annotations
        res = self.last_residual
        flat = res.ravel()
        mask = np.isfinite(res).ravel()
        vals = flat[mask]

        if vals.size == 0:
            axes[2].text(0.5, 0.5, "No finite residual pixels", ha="center", va="center")
        else:
            axes[2].hist(vals, bins=200, histtype="step", color='C0')
            med = np.median(vals)
            neg_frac = np.mean(vals < 0)
            pos_frac = np.mean(vals > 0)
            balance = abs(neg_frac - pos_frac)
            axes[2].axvline(med, color='C1', linestyle='--', label=f"median={med:.2f}")
            axes[2].legend()
            axes[2].set_xlabel("ADU")
            axes[2].set_ylabel("Count")
            # annotate with small text
            txt = f"median={med:.2f}\nneg_frac={neg_frac:.3f}\npos_frac={pos_frac:.3f}\nbalance={balance:.3f}"
            axes[2].text(0.95, 0.95, txt, transform=axes[2].transAxes, ha='right', va='top',
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=4))

            # log a short summary
            self.log_write(f"Diagnostics histogram: median={med:.2f}, neg_frac={neg_frac:.3f}, pos_frac={pos_frac:.3f}, balance={balance:.3f}")

        plt.suptitle(
            f"Last diagnostics: acc={diag.get('accepted_cells')} "
            f"frac={diag.get('accepted_frac'):.3f} res_med={diag.get('res_med'):.3f} "
            f"res_std={diag.get('res_std'):.2f} skew={diag.get('res_skew'):.2f} "
            f"balance={diag.get('balance'):.2f}"
        )
        plt.tight_layout()
        plt.show()

    def show_grid_overlay(self):
        if self.last_bg is None or self.last_mask_grid is None:
            messagebox.showinfo("Grid overlay", "No processed image in this session yet.")
            return

        mask = self.last_mask_grid
        starmask = self.last_starmask
        ny, nx = self.last_bg.shape
        samples = mask.shape[0]
        xs = np.linspace(0, nx - 1, samples)
        ys = np.linspace(0, ny - 1, samples)

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(self.last_bg, origin='lower', cmap='magma', alpha=0.6)
        for x in xs:
            ax.axvline(x, color='white', alpha=0.2)
        for y in ys:
            ax.axhline(y, color='white', alpha=0.2)
        cell_w = nx / samples
        cell_h = ny / samples
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                x0 = i * cell_w
                y0 = j * cell_h
                rect = plt.Rectangle((x0, y0), cell_w, cell_h, fill=False, linewidth=1.0)
                rect.set_edgecolor('lime' if mask[j, i] else 'red')
                ax.add_patch(rect)
        ax.imshow(starmask, origin='lower', cmap='Reds', alpha=0.25)
        ax.set_title("Grid overlay: green=accepted, red=rejected; star mask in red tint")
        plt.show()

    def show_tuning_tips(self):
        tips = [
            "1) Start with medium density: 20–30 samples per line for most images.",
            "2) Target ~100 pixels per cell as a default; increase target if heavy masking reduces usable pixels.",
            "3) If background looks too low: increase sigma-clip (looser), lower grid tolerance, or reduce density.",
            "4) If background misses gradients: increase samples, then increase polynomial degree cautiously.",
            "5) For dense grids: use stronger star masking (higher threshold_sigma or larger dilation).",
            "6) Use smoothing (Gaussian sigma 0.5–1.5) to avoid overfitting when grid is dense.",
            "7) Ensure minimum accepted cells and coverage; if many cells are rejected, the feedback loop will relax parameters.",
            "8) Inspect grid overlay and residual histogram after a run before batch-applying settings.",
            "9) Siril‑Match Mode aims for residual median ~0, symmetric histogram, and noise similar to global sigma.",
            "10) Use final_params from the log as starting defaults for similar future frames."
        ]
        win = tk.Toplevel(self)
        win.title("Tuning Tips")
        txt = scrolledtext.ScrolledText(win, width=80, height=20)
        txt.pack(padx=8, pady=8)
        txt.insert(tk.END, "\n".join(tips))
        txt.config(state="disabled")

if __name__ == "__main__":
    app = BackgroundBatchGUI()
    app.mainloop()
