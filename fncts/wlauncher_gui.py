#!/usr/bin/env python3
"""
launcher_gui.py
PyQt6 GUI replacement for your text-based menu.
Template-aware action mapping:
Each ACTION entry is:
    ("Label", (python_exe, script_path, arg1_source, arg2_source, ...))

argX_source can be:
    "input1", "input2", "output", "param"  -> taken from GUI fields
    any other string                        -> passed literally (e.g. "--out", "gaia", "5.0")
"""

import sys
import subprocess
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QComboBox,
    QGridLayout, QTextEdit, QFileDialog, QHBoxLayout, QMessageBox
)
from PyQt6.QtCore import QT_VERSION_STR, PYQT_VERSION_STR

# -----------------------
# Action mapping
# -----------------------
ACTIONS = {
    "Exit": ("exit",),

    "AffineTransform (exec file)": (sys.executable, "fncts/affine_transform.py"),
    "Crop": (sys.executable, "fncts/crop.py"),
    "Mask tool GUI (spawn)": (sys.executable, "fncts/mask_tool_gui.py"),
    "Remove NaN tool GUI (spawn)": (sys.executable, "fncts/remove_nan.py"),
    "Siril Batch Plate Solve (spawn)": (sys.executable, "fncts/batchPlatesolve.py"),
    "Fits Splitter (spawn)": (sys.executable, "fncts/fits_splitter.py"),
    "Align JPG GUI (spawn)": (sys.executable, "fncts/align_jpg_gui.py"),
    "Align & Stack (spawn)": (sys.executable, "fncts/align_and_stack_gui.py"),
    "Plot 3D GUI (spawn)": (sys.executable, "fncts/plot3d_gui.py"),
    "Rotational Gradient (spawn)": (sys.executable, "fncts/RotationalGradient.py"),
    "Dynamic Rescale (spawn)": (sys.executable, "fncts/dynamic_rescale16_gui_cython.py"),
    "File Count GUI (spawn)": (sys.executable, "fncts/filecount_gui.py"),
    "Resize GUI (spawn)": (sys.executable, "fncts/resize_gui.py"),
    "JPG Compress GUI (spawn)": (sys.executable, "fncts/jpgcomp_gui.py"),
    "Imghiststretch GUI (spawn)": (sys.executable, "fncts/imghiststretch_gui.py"),
    "Video GUI (spawn)": (sys.executable, "fncts/video_gui.py"),
    "Gamma GUI (spawn)": (sys.executable, "fncts/gamma_gui.py"),
    "Copy Old Header GUI (spawn)": (sys.executable, "fncts/cpy_old_hdr_gui.py"),
    "Hist Match GUI (spawn)": (sys.executable, "fncts/hist_match_gui.py"),
    "Distance GUI (spawn)": (sys.executable, "fncts/distance_gui.py"),
    "EdgeDetect GUI (spawn)": (sys.executable, "fncts/edgedetect_gui.py"),
    "BinImg GUI (spawn)": (sys.executable, "fncts/binimg_gui.py"),
    "WCS Plotter GUI (spawn)": (sys.executable, "fncts/fits_wcs_plotter_gui.py"),
    "Align Images By Dir (spawn)": (sys.executable, "fncts/align_imgs_gui_fallback.py"),
    "Combine LRGB GUI (spawn)": (sys.executable, "fncts/fits_lrgb_combine_gui.py"),
    "MaxDL ASTAP GUI (spawn)": (sys.executable, "fncts/mxdl_astap_gui.py"),
    "Centroid / Ratio GUI (spawn)": (sys.executable, "cent_ratio_gui.py"),
    "Combine Weighted GUI (spawn)": (sys.executable, "fncts/combine_weighted_gui.py"),
    "Pixel Math (spawn)": (sys.executable, "fncts/pixelmath.py"),
    "Color Tool (spawn)": (sys.executable, "fncts/color_tool.py"),
    "Image Filters (spawn)": (sys.executable, "fncts/image_filters.py"),
    "CLAHE GUI (spawn)": (sys.executable, "fncts/clahe_gui.py"),
    "Auto Stretch GUI (spawn)": (sys.executable, "fncts/autostr_gui.py"),
    "Rank_gui (spawn)": (sys.executable, "fncts/wrank.py"),
    "siril_fft_notch_gui (spawn)": (sys.executable, "fncts/siril_fft_notch_guia.py"),
    "Align Images (spawn)": (sys.executable, "fncts/align_imgs.py"),
    "Stacker GUI (spawn)": (sys.executable, "fncts/stacker_gui.py"),
    "Analyze FITS Roundness & Trails (spawn)": (sys.executable, "analyze_fits_roundness_trails.py"),
    "Normalize GUI (spawn)": (sys.executable, "fncts/normalize_gui.py"),
    "RaDec->2pt Angle GUI (spawn)": (sys.executable, "fncts/radectwoptang_gui.py"),
    "Dust reddening GUI (spawn)": (sys.executable, "fncts/dust_gui.py"),

    # Template-aware dust tools
    # apConsoleWcs.py csv --out out.csv
    "Dust WCS LkUp (spawn)": (
        sys.executable,
        "fncts/apConsoleWcs.py",
        "input1",
        "--out",
        "output"
    ),

    # apConsole.py csv --catalog gaia --radius 5.0 --out out.csv
    "Dust Gaia LkUp (spawn)": (
        sys.executable,
        "fncts/apConsole.py",
        "input1",
        "--catalog",
        "gaia",
        "--radius",
        "5.0",
        "--out",
        "output"
    ),

    "classify sptype GUI (spawn)": (sys.executable, "fncts/classify_sptype_gui.py"),
    "csv_xy_lookup GUI (spawn)": (sys.executable, "fncts/csv_xy_lookup_gui.py"),
    "spectral_percent GUI (spawn)": (sys.executable, "fncts/spectral_percent_gui.py"),
    "plot_hist&normality_gui (spawn)": (sys.executable, "fncts/plot_hist_with_normality_gui.py"),
    "ecolor_vs_distance_gui (spawn)": (sys.executable, "fncts/ecolor_vs_distance_gui.py"),
    "ecolor_vs_distance_pyvista.py (spawn)": (sys.executable, "fncts/ecolor_vs_distance_pyvista.py"),
    "ebv_map_gui (spawn)": (sys.executable, "fncts/ebv_map_gui.py"),
    "De-skew Img (spawn)": (sys.executable, "fncts/deskew_book_gui.py"),
    "PmVtr Img (spawn)": (sys.executable, "fncts/PmVtr.py"),
}

# Optional per-action hints
ACTION_HINTS = {
    "Align Images By Dir (spawn)": "Input1 = input folder; Input2 = pattern (e.g. *.fit*); Output = output folder",
    "Combine Weighted GUI (spawn)": "Input1 = input folder; Input2 = pattern (e.g. *.fit*); Output = out.fits",
    "Centroid / Ratio GUI (spawn)": "No inputs required; launches cent_ratio_gui.py",
    "AffineTransform (exec file)": "Executes the affine_transform.py script in fncts/",
    "Dust WCS LkUp (spawn)": "Input1 = input CSV; Output = output CSV; runs apConsoleWcs.py csv --out out.csv",
    "Dust Gaia LkUp (spawn)": "Input1 = input CSV; Output = output CSV; runs apConsole.py csv --catalog gaia --radius 5.0 --out out.csv",
    "Exit": "Close the launcher",
}

# -----------------------
# Launcher GUI
# -----------------------
class Launcher(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tools Launcher")
        self.resize(820, 520)
        self._build_ui()

    def _build_ui(self):
        layout = QGridLayout(self)

        # Version info
        layout.addWidget(QLabel(f"Qt: {QT_VERSION_STR}    PyQt: {PYQT_VERSION_STR}"), 0, 0, 1, 3)

        # Action selector
        layout.addWidget(QLabel("Choose action:"), 1, 0)
        self.action_combo = QComboBox()
        for label in ACTIONS.keys():
            self.action_combo.addItem(label)
        layout.addWidget(self.action_combo, 1, 1, 1, 2)

        # Input boxes
        layout.addWidget(QLabel("Input 1 (file/folder):"), 2, 0)
        self.input1 = QLineEdit()
        layout.addWidget(self.input1, 2, 1)
        btn1 = QPushButton("Browse")
        btn1.clicked.connect(lambda: self._browse_file(self.input1))
        layout.addWidget(btn1, 2, 2)

        layout.addWidget(QLabel("Input 2 / Pattern:"), 3, 0)
        self.input2 = QLineEdit()
        layout.addWidget(self.input2, 3, 1)
        btn2 = QPushButton("Browse folder")
        btn2.clicked.connect(lambda: self._browse_folder(self.input2))
        layout.addWidget(btn2, 3, 2)

        layout.addWidget(QLabel("Output / Save as:"), 4, 0)
        self.output = QLineEdit()
        layout.addWidget(self.output, 4, 1)
        btn3 = QPushButton("Save As")
        btn3.clicked.connect(lambda: self._browse_save(self.output))
        layout.addWidget(btn3, 4, 2)

        layout.addWidget(QLabel("Param:"), 5, 0)
        self.param = QLineEdit()
        layout.addWidget(self.param, 5, 1, 1, 2)

        # Run / Clear
        btn_run = QPushButton("Run Selected Action")
        btn_run.clicked.connect(self._on_run)
        btn_clear = QPushButton("Clear Log")
        btn_clear.clicked.connect(lambda: self.log.clear())
        h = QHBoxLayout()
        h.addWidget(btn_run)
        h.addWidget(btn_clear)
        layout.addLayout(h, 6, 0, 1, 3)

        # Files / Status
        layout.addWidget(QLabel("Files / Status:"), 7, 0)
        self.files_box = QTextEdit()
        self.files_box.setReadOnly(True)
        layout.addWidget(self.files_box, 8, 0, 1, 3)

        # Log
        layout.addWidget(QLabel("Log:"), 9, 0)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log, 10, 0, 1, 3)

        self.action_combo.currentTextChanged.connect(self._on_action_change)
        self._on_action_change(self.action_combo.currentText())

    def _log(self, *args):
        self.log.append(" ".join(str(a) for a in args))

    def _browse_file(self, lineedit):
        fn, _ = QFileDialog.getOpenFileName(self, "Select file", "", "All Files (*)")
        if fn:
            lineedit.setText(fn)

    def _browse_folder(self, lineedit):
        d = QFileDialog.getExistingDirectory(self, "Select folder")
        if d:
            lineedit.setText(d)

    def _browse_save(self, lineedit):
        fn, _ = QFileDialog.getSaveFileName(self, "Save as", "", "All Files (*)")
        if fn:
            lineedit.setText(fn)

    def _on_action_change(self, text):
        self.files_box.clear()
        hint = ACTION_HINTS.get(text, "")
        if hint:
            self.files_box.append(hint)
        else:
            self.files_box.append(
                "Select action and click Run. If the action spawns a script, "
                "ensure script path exists under 'fncts/'."
            )
        self._log(f"Selected: {text}")

    def _build_args_from_template(self, mapping):
        """
        Build argument list from a template-aware mapping:
        mapping = (pyexe, script_path, arg1_source, arg2_source, ...)
        """
        pyexe = mapping[0]
        script = mapping[1]
        script_path = Path(script)

        # allow scripts referenced relative to this file
        if not script_path.exists():
            alt = Path("fncts") / script_path.name
            if alt.exists():
                script_path = alt

        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")

        args = [pyexe, str(script_path)]

        # If mapping contains argument templates, use them
        if len(mapping) > 2:
            for item in mapping[2:]:
                if item == "input1":
                    args.append(self.input1.text().strip())
                elif item == "input2":
                    args.append(self.input2.text().strip())
                elif item == "output":
                    args.append(self.output.text().strip())
                elif item == "param":
                    args.append(self.param.text().strip())
                else:
                    # literal argument like "--out", "--catalog", "gaia", "5.0"
                    args.append(item)
        else:
            # fallback to old behavior
            if self.input1.text().strip():
                args.append(self.input1.text().strip())
            if self.input2.text().strip():
                args.append(self.input2.text().strip())
            if self.output.text().strip():
                args.append(self.output.text().strip())
            if self.param.text().strip():
                args.append(self.param.text().strip())

        return args

    def _validate_required_fields(self, action_label, mapping):
        """
        Simple validation: if template references input1/output/etc,
        ensure those fields are not empty.
        """
        missing = []
        if len(mapping) > 2:
            items = mapping[2:]
            if "input1" in items and not self.input1.text().strip():
                missing.append("Input 1")
            if "input2" in items and not self.input2.text().strip():
                missing.append("Input 2")
            if "output" in items and not self.output.text().strip():
                missing.append("Output")
            if "param" in items and not self.param.text().strip():
                missing.append("Param")

        if missing:
            QMessageBox.warning(
                self,
                "Missing input",
                f"The following fields are required for '{action_label}': "
                + ", ".join(missing),
            )
            return False
        return True

    def _on_run(self):
        action_label = self.action_combo.currentText()
        mapping = ACTIONS.get(action_label)
        if not mapping:
            QMessageBox.warning(self, "No action", "No action mapping found")
            return

        # Built-in exit
        if mapping[0] == "exit":
            QApplication.quit()
            return

        # Spawn command tuple
        if isinstance(mapping, tuple) and len(mapping) >= 2 and mapping[0] == sys.executable:
            if not self._validate_required_fields(action_label, mapping):
                return

            try:
                args = self._build_args_from_template(mapping)
            except FileNotFoundError as e:
                self._log(str(e))
                QMessageBox.critical(self, "Script not found", str(e))
                return
            except Exception as e:
                self._log(f"Error building args: {e}")
                QMessageBox.critical(self, "Argument error", str(e))
                return

            # Preview command
            self._log("Command:", " ".join(args))

            # Launch subprocess detached so GUI remains responsive
            try:
                subprocess.Popen(args)
                self._log(f"Spawned: {' '.join(args)}")
            except Exception as e:
                self._log(f"Failed to spawn: {e}")
                QMessageBox.critical(self, "Spawn error", str(e))
            return

        # Callable mapping (rare)
        if callable(mapping):
            inputs = {
                "input1": self.input1.text().strip(),
                "input2": self.input2.text().strip(),
                "output": self.output.text().strip(),
                "param": self.param.text().strip(),
            }
            try:
                mapping(inputs)
                self._log(f"Called action function for {action_label}")
            except Exception as e:
                self._log(f"Action function raised: {e}")
                QMessageBox.critical(self, "Action error", str(e))
            return

        QMessageBox.warning(self, "Unsupported action", f"Action mapping for '{action_label}' is not supported")

# -----------------------
# Main
# -----------------------
def main():
    app = QApplication(sys.argv)
    w = Launcher()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()