#!/usr/bin/env python3
"""
video_gui.py
PyQt6 GUI to build MP4 and/or GIF from a sequence of image files (glob).
"""

import sys
import os
import glob
import traceback
from pathlib import Path

import cv2
import numpy as np
import imageio.v2 as imageio

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QFileDialog, QGridLayout, QComboBox, QTextEdit, QMessageBox, QSpinBox
)
from PyQt6.QtCore import Qt

class VideoBuilderWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Build MP4 / GIF from images")
        self._build_ui()
        self.resize(820, 360)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        grid = QGridLayout()
        central.setLayout(grid)

        # Input glob
        grid.addWidget(QLabel("Input glob (frames):"), 0, 0)
        self.glob_edit = QLineEdit("frames/*.jpg")
        grid.addWidget(self.glob_edit, 0, 1, 1, 3)
        btn_glob = QPushButton("Browse Folder")
        btn_glob.clicked.connect(self._browse_folder)
        grid.addWidget(btn_glob, 0, 4)

        # Output options
        grid.addWidget(QLabel("Output type:"), 1, 0)
        self.type_combo = QComboBox()
        self.type_combo.addItems(["MP4 only", "GIF only", "Both MP4+GIF"])
        grid.addWidget(self.type_combo, 1, 1)

        # MP4 filename
        grid.addWidget(QLabel("MP4 filename:"), 2, 0)
        self.mp4_edit = QLineEdit("output.mp4")
        grid.addWidget(self.mp4_edit, 2, 1, 1, 3)
        btn_mp4 = QPushButton("Browse")
        btn_mp4.clicked.connect(lambda: self._browse_save(self.mp4_edit, "MP4 Files (*.mp4);;All Files (*)"))
        grid.addWidget(btn_mp4, 2, 4)

        # GIF filename + width
        grid.addWidget(QLabel("GIF filename:"), 3, 0)
        self.gif_edit = QLineEdit("output.gif")
        grid.addWidget(self.gif_edit, 3, 1, 1, 2)
        btn_gif = QPushButton("Browse")
        btn_gif.clicked.connect(lambda: self._browse_save(self.gif_edit, "GIF Files (*.gif);;All Files (*)"))
        grid.addWidget(btn_gif, 3, 3)

        grid.addWidget(QLabel("GIF width (px, keep aspect):"), 3, 4)
        self.gif_width = QSpinBox(); self.gif_width.setRange(16,10000); self.gif_width.setValue(800)
        grid.addWidget(self.gif_width, 3, 5)

        # FPS
        grid.addWidget(QLabel("Frames per second:"), 4, 0)
        self.fps_spin = QSpinBox(); self.fps_spin.setRange(1,120); self.fps_spin.setValue(24)
        grid.addWidget(self.fps_spin, 4, 1)

        # Run / Clear
        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self._on_run)
        grid.addWidget(self.run_btn, 5, 0)

        self.clear_btn = QPushButton("Clear Log")
        self.clear_btn.clicked.connect(self._clear_log)
        grid.addWidget(self.clear_btn, 5, 1)

        # Log area
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        grid.addWidget(self.log, 6, 0, 6, 6)

    def _log(self, *args):
        self.log.append(" ".join(str(a) for a in args))
        QApplication.processEvents()

    def _clear_log(self):
        self.log.clear()

    def _browse_folder(self):
        dn = QFileDialog.getExistingDirectory(self, "Select folder to use for glob (will set glob to folder/*.ext)")
        if dn:
            # try to preserve extension if present in current glob
            cur = self.glob_edit.text().strip()
            ext = "*.*"
            if "." in cur:
                ext = "*" + Path(cur).suffix
            self.glob_edit.setText(os.path.join(dn, ext))

    def _browse_save(self, line_edit, filter_str):
        fn, _ = QFileDialog.getSaveFileName(self, "Select output file", "", filter_str)
        if fn:
            line_edit.setText(fn)

    def _collect_files(self, pattern):
        files = sorted(glob.glob(pattern))
        return files

    def _resize_for_gif(self, frame, target_width):
        h, w = frame.shape[:2]
        if w == target_width:
            return frame
        scale = target_width / float(w)
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(frame, (target_width, new_h), interpolation=cv2.INTER_AREA)
        return resized

    def _write_mp4(self, files, outpath, fps):
        if not files:
            raise RuntimeError("No input frames to write MP4")
        # read first to get size
        first = cv2.imread(files[0])
        if first is None:
            raise RuntimeError(f"Failed to read first frame {files[0]}")
        h, w = first.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(outpath, fourcc, float(fps), (w, h))
        if not writer.isOpened():
            raise RuntimeError("Failed to open VideoWriter for MP4")
        for fn in files:
            img = cv2.imread(fn)
            if img is None:
                self._log("Warning: skipping unreadable frame", fn)
                continue
            # ensure same size as first
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            # convert to BGR if grayscale (VideoWriter expects 3-channel color)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            writer.write(img)
        writer.release()

    def _write_gif(self, files, outpath, fps, gif_width):
        if not files:
            raise RuntimeError("No input frames to write GIF")
        duration = 1.0 / float(fps)
        frames = []
        for fn in files:
            img = cv2.imread(fn)
            if img is None:
                self._log("Warning: skipping unreadable frame", fn)
                continue
            # convert BGR->RGB for imageio
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # resize for gif width if requested
            if gif_width and gif_width > 0:
                img = self._resize_for_gif(img, gif_width)
            frames.append(img)
        if not frames:
            raise RuntimeError("No valid frames found for GIF")
        # imageio will use ffmpeg plugin if available for optimization; write animated gif
        imageio.mimsave(outpath, frames, duration=duration)

    def _on_run(self):
        pattern = self.glob_edit.text().strip()
        if not pattern:
            QMessageBox.warning(self, "Input required", "Enter an input glob pattern, e.g. frames/*.jpg")
            return
        files = self._collect_files(pattern)
        if not files:
            QMessageBox.critical(self, "No frames", f"No files matched pattern: {pattern}")
            return
        self._log(f"Found {len(files)} frames (first: {files[0]})")

        out_type = self.type_combo.currentIndex()  # 0 mp4,1 gif,2 both
        fps = int(self.fps_spin.value())
        try:
            if out_type in (0,2):
                mp4_path = self.mp4_edit.text().strip()
                if not mp4_path:
                    raise ValueError("MP4 filename required")
                self._log("Writing MP4 ->", mp4_path)
                self._write_mp4(files, mp4_path, fps)
                self._log("MP4 written:", mp4_path)

            if out_type in (1,2):
                gif_path = self.gif_edit.text().strip()
                if not gif_path:
                    raise ValueError("GIF filename required")
                gif_w = int(self.gif_width.value())
                self._log(f"Writing GIF -> {gif_path} (width={gif_w}px, fps={fps})")
                self._write_gif(files, gif_path, fps, gif_w)
                self._log("GIF written:", gif_path)

            QMessageBox.information(self, "Done", "Video/GIF creation completed successfully")
        except Exception as e:
            tb = traceback.format_exc()
            self._log("Error:", e)
            QMessageBox.critical(self, "Error", f"{e}\n\n{tb}")

def main():
    app = QApplication(sys.argv)
    w = VideoBuilderWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()