#!/usr/bin/env python3
"""
filecount_gui.py
Simple PyQt6 GUI to count files in a directory matching a glob pattern.
"""

import sys
import os
import fnmatch
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QFileDialog, QGridLayout, QComboBox, QTextEdit, QCheckBox, QMessageBox, QHBoxLayout
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QClipboard

class FileCountWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("File Count")
        self._build_ui()
        self.resize(640, 260)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        grid = QGridLayout()
        central.setLayout(grid)

        # Directory row
        grid.addWidget(QLabel("Directory:"), 0, 0)
        self.dir_edit = QLineEdit()
        grid.addWidget(self.dir_edit, 0, 1, 1, 3)
        btn_dir = QPushButton("Browse")
        btn_dir.clicked.connect(self._browse_dir)
        grid.addWidget(btn_dir, 0, 4)

        # Pattern dropdown + editable entry
        grid.addWidget(QLabel("Pattern:"), 1, 0)
        self.pattern_combo = QComboBox()
        self.pattern_combo.setEditable(True)
        # common presets
        self.pattern_combo.addItems(["*.fits", "*.fit", "*.fts", "*.fits.gz", "*.png", "*.jpg", "*.tif", "*.*"])
        grid.addWidget(self.pattern_combo, 1, 1, 1, 3)
        btn_set = QPushButton("Use Selection")
        btn_set.clicked.connect(self._use_selected_pattern)
        grid.addWidget(btn_set, 1, 4)

        # Options row
        self.recursive_chk = QCheckBox("Include subdirectories (recursive)")
        self.recursive_chk.setChecked(False)
        grid.addWidget(self.recursive_chk, 2, 0, 1, 3)

        # Count / copy / clear buttons
        self.count_btn = QPushButton("Count Files")
        self.count_btn.clicked.connect(self._count_files)
        grid.addWidget(self.count_btn, 3, 0)

        self.copy_btn = QPushButton("Copy Result")
        self.copy_btn.clicked.connect(self._copy_result)
        grid.addWidget(self.copy_btn, 3, 1)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._clear_log)
        grid.addWidget(self.clear_btn, 3, 2)

        self.open_dir_btn = QPushButton("Open Dir in Explorer")
        self.open_dir_btn.clicked.connect(self._open_in_explorer)
        grid.addWidget(self.open_dir_btn, 3, 3, 1, 2)

        # Result / log area
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        grid.addWidget(self.log, 4, 0, 4, 5)

    def _browse_dir(self):
        dn = QFileDialog.getExistingDirectory(self, "Select directory", "")
        if dn:
            self.dir_edit.setText(dn)

    def _use_selected_pattern(self):
        # copy current combo text into editable combo (no-op mostly, but kept for UX)
        txt = self.pattern_combo.currentText()
        self.pattern_combo.setEditText(txt)

    def _count_files(self):
        d = self.dir_edit.text().strip()
        pattern = self.pattern_combo.currentText().strip()
        recursive = self.recursive_chk.isChecked()

        if not d:
            QMessageBox.warning(self, "Missing directory", "Please choose a directory to search.")
            return
        if not pattern:
            QMessageBox.warning(self, "Missing pattern", "Please enter a file pattern (e.g., *.fits).")
            return
        if not os.path.isdir(d):
            QMessageBox.critical(self, "Not a directory", f"Directory not found: {d}")
            return

        try:
            count, matched_paths = self._count_with_pattern(Path(d), pattern, recursive)
            self.log.append(f"Directory: {d}")
            self.log.append(f"Pattern: {pattern}  Recursive: {recursive}")
            self.log.append(f"File Count: {count}")
            if count and count <= 200:
                # show the matched filenames (limit to avoid huge logs)
                self.log.append("Matched files:")
                for p in matched_paths:
                    self.log.append(str(p))
            elif count > 200:
                self.log.append(f"Matched files list suppressed (count={count})")
            self.log.append("")  # blank line
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _count_with_pattern(self, base_path: Path, pattern: str, recursive: bool):
        matched = []
        if recursive:
            for root, dirs, files in os.walk(base_path):
                for fname in files:
                    if fnmatch.fnmatch(fname, pattern):
                        matched.append(Path(root) / fname)
        else:
            for fname in os.listdir(base_path):
                if fnmatch.fnmatch(fname, pattern):
                    matched.append(base_path / fname)
        return len(matched), matched

    def _copy_result(self):
        # copy last non-empty line or full log to clipboard
        text = self.log.toPlainText().strip()
        if not text:
            QMessageBox.information(self, "Nothing to copy", "Log is empty.")
            return
        # copy entire log
        cb = QApplication.clipboard()
        cb.setText(text, mode=QClipboard.Mode.Clipboard)
        # brief visual confirmation
        self.log.append("Copied result to clipboard.")

        # on some systems the clipboard contents may clear if no app holds it; keep for a moment
        QTimer.singleShot(1000, lambda: None)

    def _clear_log(self):
        self.log.clear()

    def _open_in_explorer(self):
        d = self.dir_edit.text().strip()
        if not d or not os.path.isdir(d):
            QMessageBox.information(self, "No directory", "Specify an existing directory first.")
            return
        p = Path(d)
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(p))
            elif sys.platform.startswith("darwin"):
                os.system(f'open "{p}"')
            else:
                os.system(f'xdg-open "{p}"')
        except Exception as e:
            QMessageBox.warning(self, "Open failed", f"Could not open directory: {e}")

def main():
    app = QApplication(sys.argv)
    w = FileCountWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()