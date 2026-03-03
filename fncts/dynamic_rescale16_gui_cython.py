#!/usr/bin/env python3
"""
DynamicRescale16 GUI with:
- Chunk mode using disk-backed memmaps
- Tiled final write (user adjustable tile size)
- Adaptive throttling (Windows API RAM monitor)
- Short temp directory inside out_dir/tmp/
- Short memmap filenames
- Reassemble tiled chunk output
- Windows FlushFileBuffers integration, fsync fallback, process memory logging,
  two-stage wait loop, and robust RAM governor with timeout and pause counter
- Additional cleanup: explicit memmap release, logfile close, and working-set trim
"""

import sys
import os
import re
import shutil
import time
import ctypes
from ctypes import wintypes
import gc
from pathlib import Path

import numpy as np
import cv2
from astropy.io import fits

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QComboBox, QFileDialog, QGridLayout, QTextEdit, QMessageBox,
    QSpinBox
)
from PyQt6.QtGui import QDoubleValidator
from PyQt6.QtCore import Qt

# -------------------------
# Windows RAM Reader (no psutil)
# -------------------------
class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]

# Persistent logfile (opened lazily)
LOGFILE_PATH = None
LOGFILE_HANDLE = None

def _open_logfile(out_dir="."):
    global LOGFILE_PATH, LOGFILE_HANDLE
    if LOGFILE_HANDLE is not None:
        return
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    LOGFILE_PATH = os.path.join(out_dir, "dynamic_rescale16_run.log")
    try:
        LOGFILE_HANDLE = open(LOGFILE_PATH, "a", buffering=1, encoding="utf-8")
    except Exception:
        LOGFILE_HANDLE = None

def _log_to_file(msg: str):
    global LOGFILE_HANDLE
    try:
        if LOGFILE_HANDLE is not None:
            LOGFILE_HANDLE.write(msg + "\n")
            LOGFILE_HANDLE.flush()
    except Exception:
        pass

def _close_logfile():
    global LOGFILE_HANDLE
    try:
        if LOGFILE_HANDLE:
            LOGFILE_HANDLE.close()
    except Exception:
        pass
    LOGFILE_HANDLE = None

def get_ram_percent():
    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    return float(stat.dwMemoryLoad)

# -------------------------
# Process private memory (Working Set) helper
# -------------------------
_psapi = ctypes.windll.psapi
_kernel32 = ctypes.windll.kernel32

class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ('cb', ctypes.c_ulong),
        ('PageFaultCount', ctypes.c_ulong),
        ('PeakWorkingSetSize', ctypes.c_size_t),
        ('WorkingSetSize', ctypes.c_size_t),
        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
        ('QuotaPagedPoolUsage', ctypes.c_size_t),
        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
        ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
        ('PagefileUsage', ctypes.c_size_t),
        ('PeakPagefileUsage', ctypes.c_size_t),
    ]

def get_process_private_mb():
    try:
        h = _kernel32.GetCurrentProcess()
        pmc = PROCESS_MEMORY_COUNTERS()
        pmc.cb = ctypes.sizeof(pmc)
        if _psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):
            return pmc.WorkingSetSize / (1024.0 * 1024.0)
    except Exception:
        pass
    return None

# -------------------------
# RAM governor + pause counter + timeout
# -------------------------
RAM_LIMIT = 80.0
RAM_PAUSE_COUNT = 0
WAIT_TIMEOUT = 60.0   # seconds to wait for RAM to drop after a flush (configurable)

def enforce_ram_limit(max_percent, log_fn):
    global RAM_PAUSE_COUNT
    used = get_ram_percent()
    if used <= max_percent:
        return
    RAM_PAUSE_COUNT += 1
    start = time.time()
    log_fn(f"[{time.strftime('%H:%M:%S')}] Pause #{RAM_PAUSE_COUNT}: RAM {used:.1f}% > {max_percent}%. Entering wait loop.")
    while True:
        time.sleep(0.25)
        used = get_ram_percent()
        elapsed = time.time() - start
        proc_mb = get_process_private_mb()
        if proc_mb is not None:
            log_fn(f"[{time.strftime('%H:%M:%S')}] Pause #{RAM_PAUSE_COUNT} check: system {used:.1f}%  proc {proc_mb:.1f} MB  elapsed {elapsed:.1f}s")
        else:
            log_fn(f"[{time.strftime('%H:%M:%S')}] Pause #{RAM_PAUSE_COUNT} check: system {used:.1f}%  elapsed {elapsed:.1f}s")
        if used <= max_percent:
            log_fn(f"[{time.strftime('%H:%M:%S')}] Resume #{RAM_PAUSE_COUNT}: RAM {used:.1f}% <= {max_percent}%. Waited {elapsed:.1f}s.")
            return
        if elapsed >= WAIT_TIMEOUT:
            log_fn(f"[{time.strftime('%H:%M:%S')}] Pause #{RAM_PAUSE_COUNT} timeout after {elapsed:.1f}s; continuing anyway (RAM {used:.1f}%).")
            return

# -------------------------
# Windows file flush helper (FlushFileBuffers) with fsync fallback
# -------------------------
GENERIC_WRITE = 0x40000000
OPEN_EXISTING = 3
FILE_SHARE_READ = 1
FILE_SHARE_WRITE = 2
FILE_ATTRIBUTE_NORMAL = 0x80

def _open_win_handle(path):
    CreateFileW = ctypes.windll.kernel32.CreateFileW
    CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                            wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    CreateFileW.restype = wintypes.HANDLE
    handle = CreateFileW(path, GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                         None, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, None)
    if handle == wintypes.HANDLE(-1).value:
        return None
    return handle

def _close_win_handle(handle):
    ctypes.windll.kernel32.CloseHandle(handle)

def flush_file_windows(path, log_fn=None):
    """Force Windows to flush file buffers for path. Returns True on success."""
    handle = _open_win_handle(path)
    if not handle:
        if log_fn:
            log_fn(f"[{time.strftime('%H:%M:%S')}] Could not open handle to flush {path}")
        return False
    ok = ctypes.windll.kernel32.FlushFileBuffers(handle)
    _close_win_handle(handle)
    if ok == 0:
        if log_fn:
            log_fn(f"[{time.strftime('%H:%M:%S')}] FlushFileBuffers failed for {path}")
        return False
    if log_fn:
        log_fn(f"[{time.strftime('%H:%M:%S')}] FlushFileBuffers succeeded for {path}")
    return True

# -------------------------
# Try importing Cython extension
# -------------------------
try:
    from warpaffinemaskrescale import warp_affine_mask_rescale
    CYTHON_AVAILABLE = True
    _IMPORT_ERR = None
except Exception as e:
    warp_affine_mask_rescale = None
    CYTHON_AVAILABLE = False
    _IMPORT_ERR = e

# -------------------------
# Logging helpers
# -------------------------
def log_disk_usage(path, log_fn):
    try:
        total, used, free = shutil.disk_usage(path)
        log_fn(f"Disk usage at {path}: Total={total/1e9:.2f}GB Used={used/1e9:.2f}GB Free={free/1e9:.2f}GB")
    except Exception as e:
        log_fn(f"Could not get disk usage for {path}: {e}")

def log_file_size(path, log_fn):
    try:
        if os.path.exists(path):
            size = os.path.getsize(path)
            log_fn(f"File {os.path.basename(path)}: {size/1e9:.2f}GB")
        else:
            log_fn(f"File {os.path.basename(path)} does not exist.")
    except Exception as e:
        log_fn(f"Could not get file size for {path}: {e}")

# -------------------------
# Pure-Python fallback block rescaler
# -------------------------
def block_rescale_float64(img64, out64, block_size):
    H, W = img64.shape
    bs = max(1, int(block_size))
    for y0 in range(0, H, bs):
        for x0 in range(0, W, bs):
            y1 = min(y0 + bs, H)
            x1 = min(x0 + bs, W)
            block = img64[y0:y1, x0:x1]  # view only
            bmin = np.nanmin(block)
            bmax = np.nanmax(block)
            if bmax > bmin:
                norm = (block - bmin) / (bmax - bmin)
            else:
                norm = np.zeros_like(block)
            out64[y0:y1, x0:x1] = norm * 65535.0

def rescale_blocks(img64, out64, block_size):
    if CYTHON_AVAILABLE and callable(warp_affine_mask_rescale):
        if img64.dtype != np.float64:
            img64 = img64.astype(np.float64)
        if out64.dtype != np.float64:
            out64 = out64.astype(np.float64)
        if not img64.flags['C_CONTIGUOUS']:
            img64 = np.ascontiguousarray(img64)
        if not out64.flags['C_CONTIGUOUS']:
            out64 = np.ascontiguousarray(out64)
        warp_affine_mask_rescale(img64, out64, int(block_size))
    else:
        block_rescale_float64(img64, out64, block_size)

# -------------------------
# FITS helpers
# -------------------------
def load_fits(file_path):
    with fits.open(file_path, memmap=True) as hdul:
        return hdul[0].data, hdul[0].header

# -------------------------
# Memmap helpers
# -------------------------
def _memmap_path(base_dir, stem, suffix):
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, f"{stem}_{suffix}.dat")

# -------------------------
# Tiled final write with adaptive throttling + flush + RAM checks
# -------------------------
def write_tiled_fits_from_binned(
    binned_path,
    header,
    full_height,
    full_width,
    out_dir,
    base_name,
    tile_h,
    tile_w,
    log_fn=print,
):
    log_fn(f"Writing tiled FITS outputs from {binned_path}")
    os.makedirs(out_dir, exist_ok=True)

    binned_mm = np.memmap(binned_path, dtype=np.float32, mode="r", shape=(full_height, full_width))

    sleep_time = 0.0
    ram_prev = get_ram_percent()

    tile_index = 0
    for y0 in range(0, full_height, tile_h):
        enforce_ram_limit(RAM_LIMIT, log_fn)
        y1 = min(y0 + tile_h, full_height)
        for x0 in range(0, full_width, tile_w):
            enforce_ram_limit(RAM_LIMIT, log_fn)
            x1 = min(x0 + tile_w, full_width)

            tile_view = binned_mm[y0:y1, x0:x1]
            if not tile_view.flags['C_CONTIGUOUS']:
                tile = np.ascontiguousarray(tile_view)
            else:
                tile = tile_view

            tile_hdr = header.copy()
            tile_hdr["NAXIS"] = 2
            tile_hdr["NAXIS1"] = int(x1 - x0)
            tile_hdr["NAXIS2"] = int(y1 - y0)
            tile_hdr["TILEY0"] = int(y0)
            tile_hdr["TILEY1"] = int(y1)
            tile_hdr["TILEX0"] = int(x0)
            tile_hdr["TILEX1"] = int(x1)

            outname = os.path.join(
                out_dir,
                f"{base_name}_tile_{tile_index:04d}_y{y0}_{y1}_x{x0}_{x1}.fits",
            )

            ram_before = get_ram_percent()
            proc_mb = get_process_private_mb()
            if proc_mb is not None:
                log_fn(f"[{time.strftime('%H:%M:%S')}] Tile {tile_index}: RAM before write {ram_before:.1f}%  proc {proc_mb:.1f} MB")
            else:
                log_fn(f"[{time.strftime('%H:%M:%S')}] Tile {tile_index}: RAM before write {ram_before:.1f}%")

            try:
                fits.writeto(outname, tile.astype(np.float32, copy=False), tile_hdr, overwrite=True)
                log_fn(f"[{time.strftime('%H:%M:%S')}] Tile {tile_index}: wrote file {outname}")
            except Exception as e:
                log_fn(f"[{time.strftime('%H:%M:%S')}] Tile {tile_index}: fits.writeto failed: {e}")
                enforce_ram_limit(RAM_LIMIT, log_fn)

            time.sleep(0.05)
            ram_after_write = get_ram_percent()
            proc_mb = get_process_private_mb()
            if proc_mb is not None:
                log_fn(f"[{time.strftime('%H:%M:%S')}] Tile {tile_index}: RAM after write {ram_after_write:.1f}%  proc {proc_mb:.1f} MB")
            else:
                log_fn(f"[{time.strftime('%H:%M:%S')}] Tile {tile_index}: RAM after write {ram_after_write:.1f}%")

            flushed = False
            try:
                flushed = flush_file_windows(outname, log_fn=log_fn)
            except Exception as e:
                log_fn(f"[{time.strftime('%H:%M:%S')}] Tile {tile_index}: FlushFileBuffers exception: {e}")

            if not flushed:
                try:
                    with open(outname, "rb+") as f:
                        f.flush()
                        os.fsync(f.fileno())
                    log_fn(f"[{time.strftime('%H:%M:%S')}] Tile {tile_index}: os.fsync succeeded for {outname}")
                except Exception as e:
                    log_fn(f"[{time.strftime('%H:%M:%S')}] Tile {tile_index}: os.fsync failed: {e}")

            time.sleep(0.05)
            ram_after_flush = get_ram_percent()
            proc_mb = get_process_private_mb()
            if proc_mb is not None:
                log_fn(f"[{time.strftime('%H:%M:%S')}] Tile {tile_index}: RAM after flush {ram_after_flush:.1f}%  proc {proc_mb:.1f} MB")
            else:
                log_fn(f"[{time.strftime('%H:%M:%S')}] Tile {tile_index}: RAM after flush {ram_after_flush:.1f}%")

            start = time.time()
            while get_ram_percent() > RAM_LIMIT and (time.time() - start) < 10.0:
                log_fn(f"[{time.strftime('%H:%M:%S')}] Tile {tile_index}: waiting (fast) now {get_ram_percent():.1f}%")
                time.sleep(0.25)
            while get_ram_percent() > RAM_LIMIT and (time.time() - start) < WAIT_TIMEOUT:
                log_fn(f"[{time.strftime('%H:%M:%S')}] Tile {tile_index}: waiting (slow) now {get_ram_percent():.1f}%")
                time.sleep(1.0)

            enforce_ram_limit(RAM_LIMIT, log_fn)

            try:
                del tile
                gc.collect()
            except Exception:
                pass

            ram_now = get_ram_percent()
            ram_delta = ram_now - ram_prev
            ram_prev = ram_now

            if ram_delta > 1.0:
                sleep_time = min(sleep_time + 0.02, 0.5)
            elif ram_delta < -0.5:
                sleep_time = max(sleep_time - 0.01, 0.0)

            if sleep_time > 0:
                time.sleep(sleep_time)

            tile_index += 1

    try:
        del binned_mm
        gc.collect()
    except Exception:
        pass

    log_fn(f"Finished writing {tile_index} tiled FITS files.")

# -------------------------
# Reassemble tiled chunk output
# -------------------------
def reassemble_tiled_chunk(
    tiles_dir,
    original_fits,
    output_fits,
    log_fn=print,
):
    log_fn(f"Reassembling tiled chunk output from {tiles_dir}")
    if not os.path.isdir(tiles_dir):
        raise ValueError("Tiles directory not found")

    orig_data, orig_header = load_fits(original_fits)
    if orig_data is None or orig_data.ndim != 2:
        raise ValueError("Original FITS must contain a 2D image")

    H, W = orig_data.shape
    final = np.zeros((H, W), dtype=np.float32)

    pattern = re.compile(
        r"_tile_(\d+)_y(\d+)_(\d+)_x(\d+)_(\d+)\.fits$", re.IGNORECASE
    )

    files = sorted(
        os.path.join(tiles_dir, f)
        for f in os.listdir(tiles_dir)
        if f.lower().endswith(".fits") and "_tile_" in f
    )
    if not files:
        raise ValueError("No tiled FITS files found in directory")

    for fpath in files:
        fname = os.path.basename(fpath)
        m = pattern.search(fname)
        if not m:
            log_fn(f"Skipping non-tile file: {fname}")
            continue
        tile_idx, y0, y1, x0, x1 = map(int, m.groups())
        with fits.open(fpath, memmap=True) as hdul:
            tile_data = hdul[0].data
        log_fn(f"Inserting tile {tile_idx:04d}: rows {y0}-{y1}, cols {x0}-{x1}")
        final[y0:y1, x0:x1] = tile_data

    fits.writeto(output_fits, final, orig_header, overwrite=True)
    log_fn(f"Reassembled tiled chunk saved to {output_fits}")

# -------------------------
# Chunk processing with streaming + tiled final write + adaptive throttling
# -------------------------
def process_entire_file(input_file, width_of_square, bin_value, gamma_value,
                        resize_factor, resize_div, tile_h, tile_w,
                        out_dir=".", log_fn=print):

    # ensure logfile is opened in the chosen out_dir
    _open_logfile(out_dir)
    log_fn(f"Processing entire file as chunk: {input_file}")
    _log_to_file(f"Processing entire file as chunk: {input_file}")

    with fits.open(input_file, memmap=True) as hdul:
        image_data = hdul[0].data
        header = hdul[0].header

    if image_data is None:
        log_fn("Input file contains no data.")
        _log_to_file("Input file contains no data.")
        return None
    if image_data.ndim != 2:
        raise ValueError("This processing expects a 2D image.")

    H, W = image_data.shape
    log_fn(f"Input shape: {H} x {W}")
    _log_to_file(f"Input shape: {H} x {W}")
    log_disk_usage(out_dir, log_fn)

    # Short temp directory inside out_dir/tmp
    tmp_dir = os.path.join(out_dir, "tmp")
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    # Short memmap filenames
    norm_path = _memmap_path(tmp_dir, "mm", "norm")
    resized_path = _memmap_path(tmp_dir, "mm", "resized")
    img64_path = _memmap_path(tmp_dir, "mm", "img64")
    out64_path = _memmap_path(tmp_dir, "mm", "out64")
    norm2_path = _memmap_path(tmp_dir, "mm", "norm2")
    binned_path = _memmap_path(tmp_dir, "mm", "binned")

    # -------------------------
    # Normalize
    # -------------------------
    minv = np.nanmin(image_data)
    maxv = np.nanmax(image_data)
    norm_mm = np.memmap(norm_path, dtype=np.uint16, mode="w+", shape=(H, W))

    if maxv == minv:
        norm_mm[:] = 0
    else:
        scale = 65535.0 / (maxv - minv)
        for y0 in range(H):
            enforce_ram_limit(RAM_LIMIT, log_fn)
            row = image_data[y0, :]
            norm_mm[y0, :] = np.clip((row - minv) * scale, 0, 65535).astype(np.uint16)
            if y0 % 5000 == 0:
                proc_mb = get_process_private_mb()
                log_fn(f"Normalizing row {y0}/{H}  proc={proc_mb:.1f} MB" if proc_mb is not None else f"Normalizing row {y0}/{H}")

    norm_mm.flush()
    del norm_mm
    gc.collect()

    # -------------------------
    # Resize (streaming)
    # -------------------------
    fx = (resize_factor / resize_div) if resize_div != 0 else 1.0
    if fx <= 0:
        fx = 1.0

    new_H = int(round(H * fx))
    new_W = int(round(W * fx))
    log_fn(f"Resized shape: {new_H} x {new_W}")
    _log_to_file(f"Resized shape: {new_H} x {new_W}")

    resized_mm = np.memmap(resized_path, dtype=np.uint16, mode="w+", shape=(new_H, new_W))
    norm_mm = np.memmap(norm_path, dtype=np.uint16, mode="r", shape=(H, W))

    strip = 512
    y_out = 0
    for y0 in range(0, H, strip):
        enforce_ram_limit(RAM_LIMIT, log_fn)
        y1 = min(y0 + strip, H)
        in_strip = norm_mm[y0:y1, :]  # view only

        out_strip = cv2.resize(
            in_strip,
            (new_W, int(round((y1 - y0) * fx))),
            interpolation=cv2.INTER_LANCZOS4
        )

        h_strip = out_strip.shape[0]
        resized_mm[y_out:y_out + h_strip, :] = out_strip
        y_out += h_strip

        if y0 % (strip * 5) == 0:
            proc_mb = get_process_private_mb()
            log_fn(f"Resized rows {y0}-{y1}  proc={proc_mb:.1f} MB" if proc_mb is not None else f"Resized rows {y0}-{y1}")

    resized_mm.flush()
    del norm_mm
    gc.collect()

    # -------------------------
    # Convert to float64 (streaming, no big copies)
    # -------------------------
    img64_mm = np.memmap(img64_path, dtype=np.float64, mode="w+", shape=(new_H, new_W))
    resized_mm = np.memmap(resized_path, dtype=np.uint16, mode="r", shape=(new_H, new_W))

    strip_conv = 256
    for y0 in range(0, new_H, strip_conv):
        enforce_ram_limit(RAM_LIMIT, log_fn)
        y1 = min(y0 + strip_conv, new_H)
        view = resized_mm[y0:y1, :]
        img64_mm[y0:y1, :] = view.astype(np.float64, copy=False)
        if y0 % (strip_conv * 5) == 0:
            proc_mb = get_process_private_mb()
            ram = get_ram_percent()
            log_fn(f"Converted rows {y0}-{y1}/{new_H}  RAM={ram:.1f}%  proc={proc_mb:.1f} MB" if proc_mb is not None else f"Converted rows {y0}-{y1}/{new_H}  RAM={ram:.1f}%")
        gc.collect()

    img64_mm.flush()
    del resized_mm
    gc.collect()

    # -------------------------
    # Block rescale
    # -------------------------
    out64_mm = np.memmap(out64_path, dtype=np.float64, mode="w+", shape=(new_H, new_W))
    log_fn("Running block rescale...")
    rescale_blocks(img64_mm, out64_mm, block_size=int(width_of_square))
    out64_mm.flush()
    del img64_mm
    gc.collect()

    # -------------------------
    # Gamma correction
    # -------------------------
    gamma = float(gamma_value)
    out64_mm = np.memmap(out64_path, dtype=np.float64, mode="r+", shape=(new_H, new_W))
    for y0 in range(new_H):
        if y0 % 2000 == 0:
            enforce_ram_limit(RAM_LIMIT, log_fn)
        row = out64_mm[y0, :]
        out64_mm[y0, :] = np.round(65535.0 * (row / 65535.0) ** gamma)
        if y0 % 5000 == 0:
            proc_mb = get_process_private_mb()
            log_fn(f"Gamma-corrected row {y0}/{new_H}  proc={proc_mb:.1f} MB" if proc_mb is not None else f"Gamma-corrected row {y0}/{new_H}")

    out64_mm.flush()

    # -------------------------
    # Normalize again
    # -------------------------
    norm2_mm = np.memmap(norm2_path, dtype=np.float64, mode="w+", shape=(new_H, new_W))
    for y0 in range(new_H):
        if y0 % 2000 == 0:
            enforce_ram_limit(RAM_LIMIT, log_fn)
        row = out64_mm[y0, :]
        norm2_mm[y0, :] = row / 6553500.0
        if y0 % 5000 == 0:
            proc_mb = get_process_private_mb()
            log_fn(f"Normalized row {y0}/{new_H}  proc={proc_mb:.1f} MB" if proc_mb is not None else f"Normalized row {y0}/{new_H}")

    norm2_mm.flush()
    del out64_mm
    gc.collect()

    # -------------------------
    # Binning with adaptive throttling and RAM governor
    # -------------------------
    bin_factor = max(1, int(bin_value))
    new_height = new_H // bin_factor
    new_width = new_W // bin_factor

    binned_mm = np.memmap(binned_path, dtype=np.float32, mode="w+", shape=(new_height, new_width))

    sleep_time = 0.0
    ram_prev = get_ram_percent()

    for by in range(new_height):
        enforce_ram_limit(RAM_LIMIT, log_fn)
        y0 = by * bin_factor
        y1 = y0 + bin_factor
        row_block = norm2_mm[y0:y1, :]

        for bx in range(new_width):
            enforce_ram_limit(RAM_LIMIT, log_fn)
            x0 = bx * bin_factor
            x1 = x0 + bin_factor
            block = row_block[:, x0:x1]
            binned_mm[by, bx] = np.sum(block, dtype=np.float64)

        gc.collect()

        ram_now = get_ram_percent()
        ram_delta = ram_now - ram_prev
        ram_prev = ram_now

        if ram_delta > 1.0:
            sleep_time = min(sleep_time + 0.02, 0.5)
        elif ram_delta < -0.5:
            sleep_time = max(sleep_time - 0.01, 0.0)

        if sleep_time > 0:
            time.sleep(sleep_time)

        if by % 100 == 0:
            proc_mb = get_process_private_mb()
            log_fn(f"Binned row {by}/{new_height}  RAM={ram_now:.1f}%  proc={proc_mb:.1f} MB" if proc_mb is not None else f"Binned row {by}/{new_height}  RAM={ram_now:.1f}%")

    binned_mm.flush()
    del norm2_mm
    gc.collect()

    # -------------------------
    # Tiled final write
    # -------------------------
    base = os.path.splitext(os.path.basename(input_file))[0]

    write_tiled_fits_from_binned(
        binned_path=binned_path,
        header=header,
        full_height=new_height,
        full_width=new_width,
        out_dir=out_dir,
        base_name=f"{base}_chunk_binned_gamma_corrected_drs",
        tile_h=tile_h,
        tile_w=tile_w,
        log_fn=log_fn,
    )

    log_disk_usage(out_dir, log_fn)

    # -------------------------
    # Cleanup temp directory and release resources
    # -------------------------
    try:
        log_fn(f"Cleaning up temp directory: {tmp_dir}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception as e:
        log_fn(f"Could not remove temp directory {tmp_dir}: {e}")

    # Explicitly release memmaps and large references, then GC
    try:
        # best-effort deletes; some names may not exist depending on code path
        for name in ("binned_mm", "img64_mm", "out64_mm", "norm_mm", "norm2_mm", "resized_mm"):
            try:
                if name in locals():
                    del locals()[name]
            except Exception:
                pass
    except Exception:
        pass
    gc.collect()

    # Close logfile handle so OS can release file resources
    _close_logfile()

    # Trim our process working set to release pages back to OS
    try:
        hproc = ctypes.windll.kernel32.GetCurrentProcess()
        # SetProcessWorkingSetSize(hProcess, -1, -1) requests trimming
        ctypes.windll.kernel32.SetProcessWorkingSetSize(hproc, -1, -1)
    except Exception:
        pass

    log_fn("Chunk processing complete (tiled output).")
    _log_to_file("Chunk processing complete (tiled output).")
    return None

# ============================================================
# GUI
# ============================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DynamicRescale16 GUI (tiled chunk output)")
        self._build_ui()
        self.resize(960, 700)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        grid = QGridLayout()
        central.setLayout(grid)

        # Mode selector
        grid.addWidget(QLabel("Mode:"), 0, 0)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "Split Tiles",
            "Process Tiles",
            "Process Chunk",
            "Reassemble Tiles",
            "Reassemble Tiled Chunk Output",
        ])
        self.mode_combo.currentIndexChanged.connect(self._update_mode)
        grid.addWidget(self.mode_combo, 0, 1, 1, 3)

        # Split inputs
        grid.addWidget(QLabel("Input FITS (split):"), 1, 0)
        self.split_input = QLineEdit()
        grid.addWidget(self.split_input, 1, 1, 1, 2)
        btn = QPushButton("Browse")
        btn.clicked.connect(lambda: self._pick_file(self.split_input, mode='open', filter="FITS Files (*.fits *.fit)"))
        grid.addWidget(btn, 1, 3)

        grid.addWidget(QLabel("Tile height:"), 2, 0)
        self.tile_h_split = QSpinBox()
        self.tile_h_split.setRange(16, 32768)
        self.tile_h_split.setValue(600)
        grid.addWidget(self.tile_h_split, 2, 1)

        grid.addWidget(QLabel("Tile width:"), 2, 2)
        self.tile_w_split = QSpinBox()
        self.tile_w_split.setRange(16, 32768)
        self.tile_w_split.setValue(600)
        grid.addWidget(self.tile_w_split, 2, 3)

        grid.addWidget(QLabel("Output tiles dir:"), 3, 0)
        self.tiles_dir = QLineEdit("tiles")
        grid.addWidget(self.tiles_dir, 3, 1, 1, 2)
        btn = QPushButton("Browse")
        btn.clicked.connect(lambda: self._pick_file(self.tiles_dir, mode='dir'))
        grid.addWidget(btn, 3, 3)

        # Process tiles
        grid.addWidget(QLabel("Tiles dir (process):"), 4, 0)
        self.proc_tiles_dir = QLineEdit("tiles")
        grid.addWidget(self.proc_tiles_dir, 4, 1, 1, 2)
        btn = QPushButton("Browse")
        btn.clicked.connect(lambda: self._pick_file(self.proc_tiles_dir, mode='dir'))
        grid.addWidget(btn, 4, 3)

        grid.addWidget(QLabel("Width of square:"), 5, 0)
        self.width_square = QSpinBox()
        self.width_square.setRange(1, 1024)
        self.width_square.setValue(5)
        grid.addWidget(self.width_square, 5, 1)

        grid.addWidget(QLabel("Bin value:"), 5, 2)
        self.bin_value = QSpinBox()
        self.bin_value.setRange(1, 1024)
        self.bin_value.setValue(25)
        grid.addWidget(self.bin_value, 5, 3)

        grid.addWidget(QLabel("Gamma:"), 6, 0)
        self.gamma_edit = QLineEdit("0.3981")
        self.gamma_edit.setValidator(QDoubleValidator(-10.0, 10.0, 6))
        grid.addWidget(self.gamma_edit, 6, 1)

        grid.addWidget(QLabel("Resize factor:"), 6, 2)
        self.resize_factor = QSpinBox()
        self.resize_factor.setRange(1, 1024)
        self.resize_factor.setValue(25)
        grid.addWidget(self.resize_factor, 6, 3)

        grid.addWidget(QLabel("Resize div:"), 7, 2)
        self.resize_div = QSpinBox()
        self.resize_div.setRange(1, 1024)
        self.resize_div.setValue(1)
        grid.addWidget(self.resize_div, 7, 3)

        # Chunk inputs
        grid.addWidget(QLabel("Input FITS (chunk):"), 8, 0)
        self.chunk_input = QLineEdit()
        grid.addWidget(self.chunk_input, 8, 1, 1, 2)
        btn = QPushButton("Browse")
        btn.clicked.connect(lambda: self._pick_file(self.chunk_input, mode='open', filter="FITS Files (*.fits *.fit)"))
        grid.addWidget(btn, 8, 3)

        grid.addWidget(QLabel("Chunk output dir:"), 9, 0)
        self.chunk_outdir = QLineEdit(".")
        grid.addWidget(self.chunk_outdir, 9, 1, 1, 2)
        btn = QPushButton("Browse")
        btn.clicked.connect(lambda: self._pick_file(self.chunk_outdir, mode='dir'))
        grid.addWidget(btn, 9, 3)

        # Tile size controls (ONLY for Process Chunk mode)
        grid.addWidget(QLabel("Tile height (chunk):"), 10, 0)
        self.tile_h_chunk = QSpinBox()
        self.tile_h_chunk.setRange(16, 32768)
        self.tile_h_chunk.setValue(500)
        grid.addWidget(self.tile_h_chunk, 10, 1)

        grid.addWidget(QLabel("Tile width (chunk):"), 10, 2)
        self.tile_w_chunk = QSpinBox()
        self.tile_w_chunk.setRange(16, 32768)
        self.tile_w_chunk.setValue(500)
        grid.addWidget(self.tile_w_chunk, 10, 3)

        # Reassemble tiles
        grid.addWidget(QLabel("Processed tiles dir (reassemble):"), 11, 0)
        self.rea_tiles_dir = QLineEdit("tiles")
        grid.addWidget(self.rea_tiles_dir, 11, 1, 1, 2)
        btn = QPushButton("Browse")
        btn.clicked.connect(lambda: self._pick_file(self.rea_tiles_dir, mode='dir'))
        grid.addWidget(btn, 11, 3)

        grid.addWidget(QLabel("Original FITS (for shape/header):"), 12, 0)
        self.orig_fits = QLineEdit()
        grid.addWidget(self.orig_fits, 12, 1, 1, 2)
        btn = QPushButton("Browse")
        btn.clicked.connect(lambda: self._pick_file(self.orig_fits, mode='open', filter="FITS Files (*.fits *.fit)"))
        grid.addWidget(btn, 12, 3)

        # Reassemble tiled chunk output
        grid.addWidget(QLabel("Tiled chunk dir:"), 13, 0)
        self.tiled_chunk_dir = QLineEdit(".")
        grid.addWidget(self.tiled_chunk_dir, 13, 1, 1, 2)
        btn = QPushButton("Browse")
        btn.clicked.connect(lambda: self._pick_file(self.tiled_chunk_dir, mode='dir'))
        grid.addWidget(btn, 13, 3)

        grid.addWidget(QLabel("Original FITS (chunk reassemble):"), 14, 0)
        self.chunk_orig_fits = QLineEdit()
        grid.addWidget(self.chunk_orig_fits, 14, 1, 1, 2)
        btn = QPushButton("Browse")
        btn.clicked.connect(lambda: self._pick_file(self.chunk_orig_fits, mode='open', filter="FITS Files (*.fits *.fit)"))
        grid.addWidget(btn, 14, 3)

        grid.addWidget(QLabel("Output FITS (chunk reassemble):"), 15, 0)
        self.chunk_rea_out = QLineEdit("reassembled_chunk.fits")
        grid.addWidget(self.chunk_rea_out, 15, 1, 1, 2)
        btn = QPushButton("Browse")
        btn.clicked.connect(lambda: self._pick_file(self.chunk_rea_out, mode='save', filter="FITS Files (*.fits *.fit)"))
        grid.addWidget(btn, 15, 3)

        # Run / clear / status
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self._on_run)
        grid.addWidget(self.run_button, 16, 0)

        self.clear_button = QPushButton("Clear Log")
        self.clear_button.clicked.connect(self._clear_log)
        grid.addWidget(self.clear_button, 16, 1)

        self.cython_label = QLabel(f"Cython available: {CYTHON_AVAILABLE}")
        grid.addWidget(self.cython_label, 16, 2, 1, 2)

        self.status = QTextEdit()
        self.status.setReadOnly(True)
        grid.addWidget(self.status, 17, 0, 8, 4)

        if not CYTHON_AVAILABLE and _IMPORT_ERR is not None:
            self._log(f"Import error for warpaffinemaskrescale: {_IMPORT_ERR}")

        self._update_mode(0)

    def _log(self, *args):
        text = " ".join(str(a) for a in args)
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {text}"
        try:
            self.status.append(line)
        except Exception:
            pass
        _log_to_file(line)
        QApplication.processEvents()

    def _clear_log(self):
        self.status.clear()

    def _pick_file(self, line_edit, mode='open', filter=None):
        if mode == 'open':
            fn, _ = QFileDialog.getOpenFileName(self, "Select file", "", filter or "All Files (*)")
            if fn:
                line_edit.setText(fn)
        elif mode == 'save':
            fn, _ = QFileDialog.getSaveFileName(self, "Select file", "", filter or "All Files (*)")
            if fn:
                line_edit.setText(fn)
        elif mode == 'dir':
            dn = QFileDialog.getExistingDirectory(self, "Select directory", "")
            if dn:
                line_edit.setText(dn)

    def _update_mode(self, idx):
        mode = self.mode_combo.currentText()

        split_enabled = (mode == "Split Tiles")
        proc_enabled = (mode == "Process Tiles")
        chunk_enabled = (mode == "Process Chunk")
        rea_tiles_enabled = (mode == "Reassemble Tiles")
        rea_chunk_enabled = (mode == "Reassemble Tiled Chunk Output")

        for w in (self.split_input, self.tile_h_split, self.tile_w_split, self.tiles_dir):
            w.setEnabled(split_enabled)

        for w in (self.proc_tiles_dir, self.width_square, self.bin_value,
                  self.gamma_edit, self.resize_factor, self.resize_div):
            w.setEnabled(proc_enabled)

        for w in (self.chunk_input, self.chunk_outdir, self.width_square,
                  self.bin_value, self.gamma_edit, self.resize_factor,
                  self.resize_div, self.tile_h_chunk, self.tile_w_chunk):
            w.setEnabled(chunk_enabled)

        for w in (self.rea_tiles_dir, self.orig_fits):
            w.setEnabled(rea_tiles_enabled)

        for w in (self.tiled_chunk_dir, self.chunk_orig_fits, self.chunk_rea_out):
            w.setEnabled(rea_chunk_enabled)

    def _split_tiles(self):
        in_path = self.split_input.text().strip()
        out_dir = self.tiles_dir.text().strip()
        tile_h = self.tile_h_split.value()
        tile_w = self.tile_w_split.value()
        if not in_path or not os.path.isfile(in_path):
            raise ValueError("Valid input FITS (split) required.")
        os.makedirs(out_dir, exist_ok=True)
        self._log(f"Splitting {in_path} into tiles {tile_h}x{tile_w} in {out_dir}")
        with fits.open(in_path, memmap=True) as hdul:
            data = hdul[0].data
            header = hdul[0].header
        H, W = data.shape
        idx = 0
        for y0 in range(0, H, tile_h):
            y1 = min(y0 + tile_h, H)
            for x0 in range(0, W, tile_w):
                x1 = min(x0 + tile_w, W)
                tile = np.ascontiguousarray(data[y0:y1, x0:x1])
                tile_hdr = header.copy()
                tile_hdr["NAXIS"] = 2
                tile_hdr["NAXIS1"] = int(x1 - x0)
                tile_hdr["NAXIS2"] = int(y1 - y0)
                tile_hdr["TILEY0"] = int(y0)
                tile_hdr["TILEY1"] = int(y1)
                tile_hdr["TILEX0"] = int(x0)
                tile_hdr["TILEX1"] = int(x1)
                outname = os.path.join(out_dir, f"tile_{idx:04d}_y{y0}_{y1}_x{x0}_{x1}.fits")
                fits.writeto(outname, tile.astype(np.float32), tile_hdr, overwrite=True)
                idx += 1
        self._log(f"Finished splitting into {idx} tiles.")

    def _process_tiles(self):
        tiles_dir = self.proc_tiles_dir.text().strip()
        if not tiles_dir or not os.path.isdir(tiles_dir):
            raise ValueError("Valid tiles dir required.")
        self._log("Process Tiles mode is not implemented in this patched script (use Process Chunk).")

    def _process_chunk(self):
        in_path = self.chunk_input.text().strip()
        out_dir = self.chunk_outdir.text().strip()
        if not in_path or not os.path.isfile(in_path):
            raise ValueError("Valid input FITS (chunk) required.")
        if not out_dir:
            out_dir = "."
        os.makedirs(out_dir, exist_ok=True)

        width_of_square = self.width_square.value()
        bin_value = self.bin_value.value()
        gamma_value = float(self.gamma_edit.text())
        resize_factor = self.resize_factor.value()
        resize_div = self.resize_div.value()
        tile_h = self.tile_h_chunk.value()
        tile_w = self.tile_w_chunk.value()

        # open logfile for this job
        _open_logfile(out_dir)
        self._log(f"Starting Process Chunk on {in_path}")
        process_entire_file(
            input_file=in_path,
            width_of_square=width_of_square,
            bin_value=bin_value,
            gamma_value=gamma_value,
            resize_factor=resize_factor,
            resize_div=resize_div,
            tile_h=tile_h,
            tile_w=tile_w,
            out_dir=out_dir,
            log_fn=self._log,
        )
        self._log("Process Chunk finished.")
        # close logfile after job
        _close_logfile()

    def _reassemble_tiles(self):
        tiles_dir = self.rea_tiles_dir.text().strip()
        orig_fits = self.orig_fits.text().strip()
        if not tiles_dir or not os.path.isdir(tiles_dir):
            raise ValueError("Valid processed tiles dir required.")
        if not orig_fits or not os.path.isfile(orig_fits):
            raise ValueError("Valid original FITS required.")
        out_fits = os.path.join(tiles_dir, "reassembled.fits")
        self._log(f"Reassembling tiles from {tiles_dir} into {out_fits}")
        reassemble_tiled_chunk(
            tiles_dir=tiles_dir,
            original_fits=orig_fits,
            output_fits=out_fits,
            log_fn=self._log,
        )
        self._log("Reassemble Tiles finished.")

    def _reassemble_tiled_chunk_output(self):
        tiles_dir = self.tiled_chunk_dir.text().strip()
        orig_fits = self.chunk_orig_fits.text().strip()
        out_fits = self.chunk_rea_out.text().strip()
        if not tiles_dir or not os.path.isdir(tiles_dir):
            raise ValueError("Valid tiled chunk dir required.")
        if not orig_fits or not os.path.isfile(orig_fits):
            raise ValueError("Valid original FITS (chunk reassemble) required.")
        if not out_fits:
            out_fits = "reassembled_chunk.fits"
        self._log(f"Reassembling tiled chunk from {tiles_dir} into {out_fits}")
        reassemble_tiled_chunk(
            tiles_dir=tiles_dir,
            original_fits=orig_fits,
            output_fits=out_fits,
            log_fn=self._log,
        )
        self._log("Reassemble Tiled Chunk Output finished.")

    def _on_run(self):
        try:
            mode = self.mode_combo.currentText()
            if mode == "Split Tiles":
                self._split_tiles()
            elif mode == "Process Tiles":
                self._process_tiles()
            elif mode == "Process Chunk":
                self._process_chunk()
            elif mode == "Reassemble Tiles":
                self._reassemble_tiles()
            elif mode == "Reassemble Tiled Chunk Output":
                self._reassemble_tiled_chunk_output()
            else:
                self._log(f"Unknown mode: {mode}")
        except Exception as e:
            self._log(f"Error: {e}")
            QMessageBox.critical(self, "Error", str(e))

def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    rc = app.exec()
    # ensure logfile closed on exit
    _close_logfile()
    sys.exit(rc)

if __name__ == "__main__":
    main()