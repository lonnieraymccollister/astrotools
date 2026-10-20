import sys
import traceback
from pathlib import Path

import numpy as np
from astropy.io import fits
from photutils.segmentation import (
    detect_threshold,
    detect_sources,
    SourceCatalog,
)

from PyQt6.QtCore import QObject, pyqtSignal, QThread
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
    QDoubleSpinBox,
    QProgressBar,
    QSpinBox,
)

ROUNDNESS_THRESHOLD_DEFAULT = 0.5
NSIGMA_DEFAULT = 3.0
MIN_PIXELS_DEFAULT = 5


def analyze_fits_roundness(
    fits_path,
    roundness_threshold,
    nsigma,
    min_pixels,
):
    """
    Returns:
        total_count
        pass_count
        fail_count
        ratio (pass_count / fail_count)
    """

    with fits.open(fits_path, memmap=False) as hdul:
        data = hdul[0].data

    if data is None:
        return 0, 0, 0, 0.0

    data = np.asarray(data, dtype=np.float64)

    finite_mask = np.isfinite(data)

    if not finite_mask.any():
        return 0, 0, 0, 0.0

    threshold = detect_threshold(
        data,
        nsigma=nsigma,
        mask=~finite_mask,
    )

    segmentation = detect_sources(
        data,
        threshold,
        npixels=min_pixels,
    )

    if segmentation is None:
        return 0, 0, 0, 0.0

    catalog = SourceCatalog(data, segmentation)

    if len(catalog) == 0:
        return 0, 0, 0, 0.0

    table = catalog.to_table(
        columns=[
            "semimajor_sigma",
            "semiminor_sigma",
        ]
    )

    semimajor = np.asarray(
        table["semimajor_sigma"],
        dtype=np.float64,
    )

    semiminor = np.asarray(
        table["semiminor_sigma"],
        dtype=np.float64,
    )

    valid = (
        np.isfinite(semimajor)
        & np.isfinite(semiminor)
        & (semimajor > 0)
    )

    if not np.any(valid):
        return 0, 0, 0, 0.0

    roundness = semiminor[valid] / semimajor[valid]

    total_count = len(roundness)

    fail_count = int(
        np.sum(roundness <= roundness_threshold)
    )

    pass_count = total_count - fail_count

    if fail_count == 0:
        ratio = float("inf")
    else:
        ratio = pass_count / fail_count

    return (
        int(total_count),
        int(pass_count),
        int(fail_count),
        float(ratio),
    )


class Worker(QObject):

    finished = pyqtSignal()
    progress = pyqtSignal(int)
    log = pyqtSignal(str)

    def __init__(
        self,
        folder,
        roundness_threshold,
        nsigma,
        min_pixels,
    ):
        super().__init__()

        self.folder = Path(folder)
        self.roundness_threshold = roundness_threshold
        self.nsigma = nsigma
        self.min_pixels = min_pixels

    def run(self):

        try:

            fits_files = sorted(
                [
                    f
                    for f in self.folder.iterdir()
                    if f.suffix.lower()
                    in (".fit", ".fits")
                ]
            )

            total_files = len(fits_files)

            if total_files == 0:
                self.log.emit(
                    "No FITS files found."
                )
                self.finished.emit()
                return

            low_ratio_count = 0
            very_low_ratio_count = 0

            for index, fits_file in enumerate(fits_files):

                try:

                    (
                        stars,
                        passed,
                        failed,
                        ratio,
                    ) = analyze_fits_roundness(
                        fits_file,
                        self.roundness_threshold,
                        self.nsigma,
                        self.min_pixels,
                    )

                    ratio_str = (
                        f"{ratio:.3f}"
                        if np.isfinite(ratio)
                        else "INF"
                    )

                    self.log.emit(
                        f"{fits_file.name} | "
                        f"Stars={stars} | "
                        f"Pass={passed} | "
                        f"Fail={failed} | "
                        f"Ratio={ratio_str}"
                    )

                    if np.isfinite(ratio):

                        if ratio < 1.0:
                            low_ratio_count += 1

                        if ratio < 0.5:
                            very_low_ratio_count += 1

                except Exception as e:

                    self.log.emit(
                        f"{fits_file.name} | "
                        f"ERROR | {e}"
                    )

                    traceback.print_exc()

                pct = int(
                    ((index + 1) / total_files) * 100
                )

                self.progress.emit(pct)

            self.log.emit("")
            self.log.emit(
                "=============================="
            )
            self.log.emit(
                "Analysis Complete"
            )
            self.log.emit(
                f"Images Processed: {total_files}"
            )
            self.log.emit(
                f"Images With Ratio < 1.0 : {low_ratio_count}"
            )
            self.log.emit(
                f"Images With Ratio < 0.5 : {very_low_ratio_count}"
            )
            self.log.emit(
                "=============================="
            )

        except Exception as e:

            self.log.emit(
                f"Fatal Error: {e}"
            )

            traceback.print_exc()

        self.finished.emit()


class MainWindow(QMainWindow):

    def __init__(self):

        super().__init__()

        self.folder = ""

        self.setWindowTitle(
            "FITS Roundness Analyzer"
        )

        self.resize(900, 700)

        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)

        folder_layout = QHBoxLayout()

        self.folder_label = QLabel(
            "No folder selected"
        )

        self.btn_browse = QPushButton(
            "Select Folder"
        )

        self.btn_browse.clicked.connect(
            self.select_folder
        )

        folder_layout.addWidget(
            self.folder_label
        )

        folder_layout.addWidget(
            self.btn_browse
        )

        layout.addLayout(folder_layout)

        layout.addWidget(
            QLabel("Roundness Threshold")
        )

        self.roundness_spin = QDoubleSpinBox()
        self.roundness_spin.setDecimals(3)
        self.roundness_spin.setRange(
            0.01,
            1.0,
        )
        self.roundness_spin.setSingleStep(
            0.01
        )
        self.roundness_spin.setValue(
            ROUNDNESS_THRESHOLD_DEFAULT
        )

        layout.addWidget(
            self.roundness_spin
        )

        layout.addWidget(
            QLabel("Detection Sigma")
        )

        self.nsigma_spin = QDoubleSpinBox()
        self.nsigma_spin.setDecimals(1)
        self.nsigma_spin.setRange(
            0.5,
            20.0,
        )
        self.nsigma_spin.setSingleStep(
            0.5
        )
        self.nsigma_spin.setValue(
            NSIGMA_DEFAULT
        )

        layout.addWidget(
            self.nsigma_spin
        )

        layout.addWidget(
            QLabel("Minimum Pixels")
        )

        self.min_pixels_spin = QSpinBox()
        self.min_pixels_spin.setRange(
            1,
            1000,
        )
        self.min_pixels_spin.setValue(
            MIN_PIXELS_DEFAULT
        )

        layout.addWidget(
            self.min_pixels_spin
        )

        self.btn_start = QPushButton(
            "Analyze"
        )

        self.btn_start.clicked.connect(
            self.start_analysis
        )

        layout.addWidget(
            self.btn_start
        )

        self.progress = QProgressBar()
        self.progress.setValue(0)

        layout.addWidget(
            self.progress
        )

        self.log_window = QTextEdit()
        self.log_window.setReadOnly(True)

        layout.addWidget(
            self.log_window
        )

    def select_folder(self):

        folder = QFileDialog.getExistingDirectory(
            self,
            "Select FITS Folder",
        )

        if folder:
            self.folder = folder
            self.folder_label.setText(folder)

    def start_analysis(self):

        if not self.folder:

            QMessageBox.warning(
                self,
                "No Folder",
                "Please select a FITS folder first.",
            )

            return

        self.log_window.clear()
        self.progress.setValue(0)

        self.thread = QThread()

        self.worker = Worker(
            self.folder,
            self.roundness_spin.value(),
            self.nsigma_spin.value(),
            self.min_pixels_spin.value(),
        )

        self.worker.moveToThread(
            self.thread
        )

        self.thread.started.connect(
            self.worker.run
        )

        self.worker.log.connect(
            self.log_window.append
        )

        self.worker.progress.connect(
            self.progress.setValue
        )

        self.worker.finished.connect(
            self.thread.quit
        )

        self.worker.finished.connect(
            self.worker.deleteLater
        )

        self.thread.finished.connect(
            self.thread.deleteLater
        )

        self.thread.start()


if __name__ == "__main__":

    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())