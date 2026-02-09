import os
import subprocess
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

# ---------- Configuration defaults ----------
DEFAULT_SIRIL = r"C:\Program Files\Siril\bin\siril-cli.exe"
DEFAULT_WORKDIR = os.path.expanduser("~")
DEFAULT_CENTER = "05:35:25,+05:20:02"  # e.g. '05:35:25,+05:20:02' or leave blank
DEFAULT_RA = '05 35 25'   # kept for compatibility if you prefer RA/Dec fields
DEFAULT_DEC = '05 20 02.04'
DEFAULT_FOCAL = '3648.0'
DEFAULT_PX = '9.26'       # pixel size (units expected by your Siril build)
DEFAULT_RADIUS = '10'
DEFAULT_ORDER = '3'
# --------------------------------------------

class SirilBatchGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Siril Batch Plate Solve (platesolve)")
        self.geometry("880x600")
        self.workdir = DEFAULT_WORKDIR
        self.siril_exe = DEFAULT_SIRIL
        self.output_dir = ""
        self.create_widgets()
        self.files = []

    def create_widgets(self):
        frm_top = tk.Frame(self)
        frm_top.pack(fill="x", padx=8, pady=6)

        tk.Label(frm_top, text="Siril CLI:").grid(row=0, column=0, sticky="w")
        self.siril_entry = tk.Entry(frm_top, width=64)
        self.siril_entry.insert(0, self.siril_exe)
        self.siril_entry.grid(row=0, column=1, padx=4)
        tk.Button(frm_top, text="Browse", command=self.browse_siril).grid(row=0, column=2)

        tk.Label(frm_top, text="Work folder:").grid(row=1, column=0, sticky="w")
        self.dir_entry = tk.Entry(frm_top, width=64)
        self.dir_entry.insert(0, self.workdir)
        self.dir_entry.grid(row=1, column=1, padx=4)
        tk.Button(frm_top, text="Choose", command=self.choose_folder).grid(row=1, column=2)

        tk.Label(frm_top, text="Output folder (optional):").grid(row=2, column=0, sticky="w")
        self.out_entry = tk.Entry(frm_top, width=64)
        self.out_entry.grid(row=2, column=1, padx=4)
        tk.Button(frm_top, text="Choose", command=self.choose_output).grid(row=2, column=2)

        # parameters
        params = tk.Frame(self)
        params.pack(fill="x", padx=8, pady=6)
        tk.Label(params, text="Center coords (05:35:25,+05:20:02):").grid(row=0, column=0, sticky="e")
        self.center = tk.Entry(params, width=20); self.center.insert(0, DEFAULT_CENTER); self.center.grid(row=0, column=1, padx=4)
        tk.Label(params, text="Focal (mm):").grid(row=0, column=2, sticky="e")
        self.focal = tk.Entry(params, width=12); self.focal.insert(0, DEFAULT_FOCAL); self.focal.grid(row=0, column=3, padx=4)
        tk.Label(params, text="Pixel size (µm):").grid(row=0, column=4, sticky="e")
        self.px = tk.Entry(params, width=10); self.px.insert(0, DEFAULT_PX); self.px.grid(row=0, column=5, padx=4)

        tk.Label(params, text="Radius:").grid(row=1, column=0, sticky="e")
        self.radius = tk.Entry(params, width=10); self.radius.insert(0, DEFAULT_RADIUS); self.radius.grid(row=1, column=1, padx=4)
        tk.Label(params, text="Order:").grid(row=1, column=2, sticky="e")
        self.order = tk.Entry(params, width=10); self.order.insert(0, DEFAULT_ORDER); self.order.grid(row=1, column=3, padx=4)
        tk.Label(params, text="Force (overwrite WCS):").grid(row=1, column=4, sticky="e")
        self.force_var = tk.BooleanVar(value=False)
        tk.Checkbutton(params, variable=self.force_var).grid(row=1, column=5, sticky="w")

        # file list and controls
        mid = tk.Frame(self)
        mid.pack(fill="both", expand=True, padx=8, pady=6)

        left = tk.Frame(mid)
        left.pack(side="left", fill="both", expand=True)
        tk.Label(left, text="FITS files in folder:").pack(anchor="w")
        self.listbox = tk.Listbox(left, selectmode="extended")
        self.listbox.pack(fill="both", expand=True, padx=2, pady=2)

        right = tk.Frame(mid, width=260)
        right.pack(side="right", fill="y")
        tk.Button(right, text="Refresh file list", width=28, command=self.refresh_list).pack(pady=4)
        tk.Button(right, text="Start batch", width=28, command=self.start_batch).pack(pady=4)
        tk.Button(right, text="Stop batch", width=28, command=self.stop_batch).pack(pady=4)
        tk.Button(right, text="Open work folder", width=28, command=self.open_workdir).pack(pady=4)
        tk.Button(right, text="Open output folder", width=28, command=self.open_outputdir).pack(pady=4)

        # log area
        tk.Label(self, text="Log:").pack(anchor="w", padx=8)
        self.log = scrolledtext.ScrolledText(self, height=12)
        self.log.pack(fill="both", expand=False, padx=8, pady=4)

        self._stop_flag = threading.Event()
        self.refresh_list()

    # ---------- UI actions ----------
    def browse_siril(self):
        p = filedialog.askopenfilename(title="Select siril-cli executable", filetypes=[("exe","*.exe"),("All","*.*")])
        if p:
            self.siril_entry.delete(0, tk.END); self.siril_entry.insert(0, p)

    def choose_folder(self):
        d = filedialog.askdirectory(title="Choose working folder", initialdir=self.workdir)
        if d:
            self.workdir = d
            self.dir_entry.delete(0, tk.END); self.dir_entry.insert(0, d)
            self.refresh_list()

    def choose_output(self):
        d = filedialog.askdirectory(title="Choose output folder (optional)", initialdir=self.workdir)
        if d:
            self.output_dir = d
            self.out_entry.delete(0, tk.END); self.out_entry.insert(0, d)

    def open_workdir(self):
        os.startfile(self.workdir)

    def open_outputdir(self):
        if self.out_entry.get().strip():
            os.startfile(self.out_entry.get().strip())
        else:
            messagebox.showinfo("Output folder", "No output folder set.")

    def refresh_list(self):
        self.listbox.delete(0, tk.END)
        folder = self.dir_entry.get().strip() or self.workdir
        if not os.path.isdir(folder):
            self.log_write(f"Folder not found: {folder}")
            return
        self.workdir = folder
        fits = sorted([f for f in os.listdir(folder) if f.lower().endswith(".fit")])
        self.files = fits
        for f in fits:
            self.listbox.insert(tk.END, f)
        self.log_write(f"Found {len(fits)} FITS files in {folder}")

    def log_write(self, text):
        self.log.insert(tk.END, text + "\n")
        self.log.see(tk.END)

    # ---------- Batch processing ----------
    def start_batch(self):
        if not self.listbox.size():
            messagebox.showwarning("No files", "No FITS files found to process.")
            return
        siril = self.siril_entry.get().strip()
        if not siril or not os.path.isfile(siril):
            messagebox.showwarning("Siril not found", "Please set the path to siril-cli executable.")
            return
        sel = self.listbox.curselection()
        if sel:
            files = [self.listbox.get(i) for i in sel]
        else:
            files = list(self.files)
        if not files:
            messagebox.showwarning("No files", "No files selected.")
            return

        self._stop_flag.clear()
        t = threading.Thread(target=self._run_batch, args=(siril, self.workdir, files), daemon=True)
        t.start()

    def stop_batch(self):
        self._stop_flag.set()
        self.log_write("Stop requested; current file will finish then stop.")

    def _run_batch(self, siril_exe, workdir, files):
        self.log_write(f"Starting batch: {len(files)} files")
        for idx, fname in enumerate(files, start=1):
            if self._stop_flag.is_set():
                self.log_write("Batch stopped by user.")
                break
            self.log_write(f"[{idx}/{len(files)}] Processing {fname}")
            infile = os.path.join(workdir, fname)
            outdir = self.out_entry.get().strip() or ""
            if outdir:
                os.makedirs(outdir, exist_ok=True)
                outfile = os.path.join(outdir, fname)
            else:
                outfile = infile  # overwrite original

            # Build platesolve command line for the .ssf
            center_arg = self.center.get().strip()
            center_prefix = f'{center_arg} ' if center_arg else ''
            force_flag = '-force' if self.force_var.get() else ''
            platesolve_cmd = f'platesolve {center_prefix}-focal={self.focal.get()} -pixelsize={self.px.get()} -radius={self.radius.get()} -order={self.order.get()} {force_flag}'.strip()

            ssf_lines = [
                'requires 1.4',
                f'cd "{workdir.replace("\\\\","/")}"',
                f'load "{fname}"',
                platesolve_cmd,
                f'save "{outfile}"',
                'exit'
            ]
            temp_ssf = os.path.join(workdir, "temp.ssf")
            try:
                with open(temp_ssf, "w", encoding="utf-8") as fh:
                    fh.write("\n".join(ssf_lines) + "\n")
            except Exception as e:
                self.log_write(f"Failed to write temp.ssf: {e}")
                continue

            cmd = [siril_exe, "-s", temp_ssf]
            self.log_write("Running: " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
            try:
                proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, timeout=900)
                self.log_write(proc.stdout.strip())
                if proc.returncode != 0:
                    self.log_write(f"Error (returncode {proc.returncode}). Retrying with larger radius...")
                    # retry with larger radius
                    larger_radius = str(int(self.radius.get()) * 2)
                    platesolve_cmd_retry = f'platesolve {center_prefix}-focal={self.focal.get()} -pixelsize={self.px.get()} -radius={larger_radius} -order={self.order.get()} {force_flag}'.strip()
                    ssf_lines[3] = platesolve_cmd_retry
                    with open(temp_ssf, "w", encoding="utf-8") as fh:
                        fh.write("\n".join(ssf_lines) + "\n")
                    proc2 = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, timeout=900)
                    self.log_write(proc2.stdout.strip())
                    if proc2.returncode != 0:
                        self.log_write(f"Retry failed for {fname}. See stderr:\n{proc2.stderr.strip()}")
                        with open(os.path.join(workdir, "batch_errors.log"), "a", encoding="utf-8") as ef:
                            ef.write(f"{fname}: failed (return {proc2.returncode})\n")
                    else:
                        self.log_write(f"Retry succeeded for {fname}. Saved to {outfile}")
                else:
                    self.log_write(f"Solved and saved: {outfile}")
            except subprocess.TimeoutExpired:
                self.log_write(f"Timeout running Siril for {fname}")
                with open(os.path.join(workdir, "batch_errors.log"), "a", encoding="utf-8") as ef:
                    ef.write(f"{fname}: timeout\n")
            except Exception as e:
                self.log_write(f"Exception running Siril: {e}")
                with open(os.path.join(workdir, "batch_errors.log"), "a", encoding="utf-8") as ef:
                    ef.write(f"{fname}: exception {e}\n")
            finally:
                try:
                    if os.path.exists(temp_ssf):
                        os.remove(temp_ssf)
                except:
                    pass

        self.log_write("Batch finished.")
        messagebox.showinfo("Batch complete", "Batch processing finished. Check log for details.")

if __name__ == "__main__":
    app = SirilBatchGUI()
    app.mainloop()