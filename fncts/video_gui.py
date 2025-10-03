#!/usr/bin/env python3
"""
video_gui_cv_only.py
PyQt6 GUI to build MP4 (and convert AVI fallback to GIF) from a sequence of image files (glob).

Behavior:
- Uses cv2.VideoWriter exclusively to stream frames to a video container (MP4 or AVI).
- If the code falls back to AVI or the user requested a .gif output, the program will
  convert the produced video to a true indexed-color GIF using imageio (ffmpeg) to read
  frames and Pillow to write the GIF with duration and loop metadata.

Requirements:
    pip install pyqt6 opencv-python imageio imageio-ffmpeg pillow
"""
import sys
import os
import glob
import traceback
import tempfile
from pathlib import Path

import cv2
import numpy as np
import imageio.v2 as imageio
from PIL import Image

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QFileDialog, QGridLayout, QComboBox, QTextEdit, QMessageBox, QSpinBox, QCheckBox
)
from PyQt6.QtCore import Qt

class VideoBuilderWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Build MP4 / GIF (cv2-only + optional conversion) from images")
        self._build_ui()
        self.resize(900, 460)
        # used to hold a user-requested .gif output path for conversion after video write
        self.last_requested_gif_path = None

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        grid = QGridLayout()
        central.setLayout(grid)

        # Input glob
        grid.addWidget(QLabel("Input glob (frames):"), 0, 0)
        self.glob_edit = QLineEdit("frames/*.jpg")
        grid.addWidget(self.glob_edit, 0, 1, 1, 4)
        btn_glob = QPushButton("Browse Folder")
        btn_glob.clicked.connect(self._browse_folder)
        grid.addWidget(btn_glob, 0, 5)

        # Output options
        grid.addWidget(QLabel("Output type:"), 1, 0)
        self.type_combo = QComboBox()
        self.type_combo.addItems(["MP4 only", "GIF only (cv2 VideoWriter + convert)", "Both MP4+GIF (cv2 VideoWriter + convert)"])
        grid.addWidget(self.type_combo, 1, 1, 1, 2)

        # MP4 filename
        grid.addWidget(QLabel("MP4 filename:"), 2, 0)
        self.mp4_edit = QLineEdit("output.mp4")
        grid.addWidget(self.mp4_edit, 2, 1, 1, 4)
        btn_mp4 = QPushButton("Browse")
        btn_mp4.clicked.connect(lambda: self._browse_save(self.mp4_edit, "MP4 Files (*.mp4);;All Files (*)"))
        grid.addWidget(btn_mp4, 2, 5)

        # GIF filename + width
        grid.addWidget(QLabel("GIF filename:"), 3, 0)
        self.gif_edit = QLineEdit("output.gif")
        grid.addWidget(self.gif_edit, 3, 1, 1, 3)
        btn_gif = QPushButton("Browse")
        btn_gif.clicked.connect(lambda: self._browse_save(self.gif_edit, "GIF Files (*.gif);;All Files (*)"))
        grid.addWidget(btn_gif, 3, 4)

        grid.addWidget(QLabel("GIF width (px, keep aspect):"), 3, 5)
        self.gif_width = QSpinBox(); self.gif_width.setRange(16,10000); self.gif_width.setValue(800)
        grid.addWidget(self.gif_width, 3, 6)

        # FPS
        grid.addWidget(QLabel("Frames per second:"), 4, 0)
        self.fps_spin = QSpinBox(); self.fps_spin.setRange(1,120); self.fps_spin.setValue(24)
        grid.addWidget(self.fps_spin, 4, 1)

        # Loop checkbox (informational; used during conversion to GIF)
        self.loop_info_chk = QCheckBox("Loop GIF forever (when converting to GIF)")
        self.loop_info_chk.setChecked(True)
        grid.addWidget(self.loop_info_chk, 4, 2, 1, 3)

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
        grid.addWidget(self.log, 6, 0, 6, 7)

    def _log(self, *args):
        self.log.append(" ".join(str(a) for a in args))
        QApplication.processEvents()

    def _clear_log(self):
        self.log.clear()

    def _browse_folder(self):
        dn = QFileDialog.getExistingDirectory(self, "Select folder to use for glob (will set glob to folder/*.ext)")
        if dn:
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

    def _open_first_valid_frame(self, files, target_width):
        for fn in files:
            img = cv2.imread(fn)
            if img is None:
                continue
            if target_width and target_width > 0:
                img = self._resize_for_gif(img, target_width)
            return img
        return None

    def _write_video_with_cv2(self, files, outpath, fps, target_width=None, container_fourcc="mp4v"):
        # Stream frames using cv2.VideoWriter only
        if not files:
            raise RuntimeError("No input frames to write video")
        first = self._open_first_valid_frame(files, target_width)
        if first is None:
            raise RuntimeError("Failed to read any input frame")
        h, w = first.shape[:2]
        # ensure channels -- VideoWriter expects 3-channel BGR
        if first.ndim == 2:
            first = cv2.cvtColor(first, cv2.COLOR_GRAY2BGR)
        # create writer
        fourcc = cv2.VideoWriter_fourcc(*container_fourcc)
        writer = cv2.VideoWriter(outpath, fourcc, float(fps), (w, h))
        used_outpath = outpath
        if not writer.isOpened():
            # try MJPG AVI fallback
            alt = outpath
            if not str(alt).lower().endswith(".avi"):
                alt = Path(outpath).with_suffix(".avi")
            alt = str(alt)
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(alt, fourcc, float(fps), (w, h))
            if not writer.isOpened():
                raise RuntimeError("cv2.VideoWriter failed to open writer for either requested container or AVI fallback")
            used_outpath = alt
            self._log("Fell back to", used_outpath)
        # write frames
        written = 0
        for fn in files:
            img = cv2.imread(fn)
            if img is None:
                self._log("Warning: skipping unreadable frame", fn)
                continue
            if target_width and target_width > 0:
                img = self._resize_for_gif(img, target_width)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            # ensure same size
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            writer.write(img)
            written += 1
        writer.release()
        self._log(f"Wrote {written} frames to {used_outpath}")
        return used_outpath

    def _write_mp4(self, files, outpath, fps):
        self._log("Writing MP4 using cv2.VideoWriter")
        return self._write_video_with_cv2(files, outpath, fps, target_width=None, container_fourcc="mp4v")

    def _convert_video_to_gif(self, video_path, out_gif, fps, loop_forever=True):
        """
        Read video_path (MP4/AVI) and write a true indexed-color GIF to out_gif.
        Uses imageio (ffmpeg) to read frames and Pillow to write GIF with duration and loop.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(video_path)
        self._log("Converting video to GIF:", video_path, "->", out_gif)
        reader = imageio.get_reader(video_path, format='ffmpeg')
        pil_frames = []
        for frame in reader:
            pil_frames.append(Image.fromarray(frame))
        reader.close()
        if not pil_frames:
            raise RuntimeError("No frames read from video for GIF conversion")
        duration_ms = int(round(1000.0 / float(fps)))
        loop_val = 0 if loop_forever else 1
        first, rest = pil_frames[0], pil_frames[1:]
        first.save(out_gif, save_all=True, append_images=rest,
                   duration=duration_ms, loop=loop_val, disposal=2)
        self._log(f"Converted video {video_path} -> GIF {out_gif} (duration={duration_ms}ms, loop={'infinite' if loop_val==0 else '1'})")

    def _write_gif_cv2_only(self, files, outpath, fps, gif_width):
        """
        Writes a video using cv2.VideoWriter and converts to GIF if needed.
        If user provided a .gif path, this function will write the video (MP4/AVI) then
        convert the produced video into a proper GIF using Pillow/imageio so delays/loop are preserved.
        """
        # pick container based on requested extension; prefer mp4v for .mp4, MJPG for .avi
        ext = Path(outpath).suffix.lower()
        if ext == ".mp4":
            fourcc = "mp4v"
        elif ext == ".avi":
            fourcc = "MJPG"
        else:
            # user requested .gif or other; write a temporary mp4 and convert
            fourcc = "mp4v"

        # remember user's requested gif path for conversion after writing video
        user_requested_gif_path = outpath if ext == ".gif" else None
        self.last_requested_gif_path = user_requested_gif_path

        self._log("Writing video via cv2 (will convert to GIF if user requested .gif or fallback to AVI occurred).")
        written_video_path = self._write_video_with_cv2(files, outpath, fps, target_width=gif_width, container_fourcc=fourcc)

        # If the user asked for a .gif, or if the writer fell back to AVI and user wants GIF conversion,
        # perform conversion to a true GIF file.
        should_convert = False
        target_gif = None
        if user_requested_gif_path:
            should_convert = True
            target_gif = user_requested_gif_path
        else:
            # if we wrote an AVI but the user had asked for GIF output in the UI (type_combo), convert
            requested_type = self.type_combo.currentIndex()
            # if user picked "GIF only" or "Both" and provided a gif filename, convert
            if requested_type in (1, 2):
                # prefer gif path provided in UI if it is a .gif
                gif_candidate = self.gif_edit.text().strip()
                if gif_candidate:
                    should_convert = True
                    target_gif = gif_candidate

        if should_convert and target_gif:
            try:
                loop_forever = bool(self.loop_info_chk.isChecked())
                self._convert_video_to_gif(written_video_path, target_gif, fps, loop_forever=loop_forever)
                self._log("GIF conversion completed:", target_gif)
            except Exception as e:
                self._log("GIF conversion failed:", e)
                # do not raise here; report but keep the produced video
        else:
            self._log("No GIF conversion requested or no valid target GIF path provided.")
        return written_video_path

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
                self._log(f"Writing GIF (cv2-only + convert if needed) -> {gif_path} (width={gif_w}px, fps={fps})")
                written = self._write_gif_cv2_only(files, gif_path, fps, gif_w)
                self._log("GIF-path written (video container or converted):", written)

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