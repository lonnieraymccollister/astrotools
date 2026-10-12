import sys
import numpy as np
import cv2

from astropy.io import fits

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


# ============================================================
# Histogram Window
# ============================================================

class HistogramCanvas(FigureCanvasQTAgg):

    def __init__(self):
        self.fig = Figure(figsize=(6, 3))
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.mean = None
        self.std = None
        self.p1 = None
        self.p99 = None
        self.fig.patch.set_facecolor("black")
        self.ax.set_facecolor("black")
        super().__init__(self.fig)

        self.mean = None
        self.std = None
        self.p1 = None
        self.p99 = None

        self.black_point = None
        self.white_point = None

        self.pixel_value = None

    def draw_histogram(self, data):

        self.ax.clear()

        d = data.astype(np.float64)

        dmin = np.min(d)
        dmax = np.max(d)

        if dmax > dmin:

            norm = (
                d - dmin
            ) / (
                dmax - dmin
            )

        else:

            norm = np.zeros_like(
                d,
                dtype=np.float64
            )

        flat = np.asarray(norm).ravel()

        if flat.size > 5_000_000:
            flat = flat[::8]

        elif flat.size > 1_000_000:
            flat = flat[::4]

        self.ax.hist(
            flat,
            bins=512,
            range=(0, 1),
            color="white",
            histtype="step"
        )
        if self.p1 is not None:

            self.ax.axvline(
                self.p1,
                color="gray",
                linewidth=1,
                linestyle="--"
            )

        if self.mean is not None and self.std is not None:

            mean_minus_sigma = self.mean - self.std

            self.ax.axvline(
                mean_minus_sigma,
                color="cyan",
                linewidth=2
            )

            self.ax.axvline(
                self.mean,
                color="green",
                linewidth=2
            )

        if self.p99 is not None:

            self.ax.axvline(
                self.p99,
                color="red",
                linewidth=1,
                linestyle="--"
            )
        if self.p1 is not None:

            self.ax.axvline(
                self.p1,
                color="gray",
                linewidth=1,
                linestyle="--"
            )

            self.ax.text(
                self.p1,
                self.ax.get_ylim()[1] * 0.95,
                "1%",
                color="gray",
                rotation=90
            )
            self.ax.text(
                mean_minus_sigma,
                self.ax.get_ylim()[1] * 0.95,
                "μ-1σ",
                color="cyan",
                rotation=90
            )
            self.ax.text(
                self.p99,
                self.ax.get_ylim()[1] * 0.95,
                "99%",
                color="red",
                rotation=90
            )
        self.ax.set_xlim(0, 1)

        self.ax.set_facecolor("black")

        self.ax.tick_params(
            colors="white"
        )

        #
        # Pixel Under Mouse
        #

        if self.pixel_value is not None:

            self.ax.axvline(
                self.pixel_value,
                color="magenta",
                linewidth=3
            )

            self.ax.text(
                self.pixel_value,
                self.ax.get_ylim()[1] * 0.75,
                f"{self.pixel_value:.3f}",
                color="magenta",
                rotation=90
            )

        self.draw()

        #
        # Pixel Under Mouse
        #

        if self.pixel_value is not None:

            self.ax.axvline(
                self.pixel_value,
                color="magenta",
                linewidth=2
            )

            self.ax.text(
                self.pixel_value,
                self.ax.get_ylim()[1] * 0.60,
                f"{self.pixel_value:.3f}",
                color="magenta",
                rotation=90
            )

class StretchWindow(QMainWindow):

    stretch_changed = pyqtSignal(float, float)

    def auto_sigma_stretch(self):

        if self.data is None:
            return

        d = self.data.astype(np.float64)

        dmin = np.min(d)
        dmax = np.max(d)

        if dmax <= dmin:
            return

        norm = (
            d - dmin
        ) / (
            dmax - dmin
        )

        mean = np.mean(norm)

        std = np.std(norm)

        black = max(
            0.0,
            mean - std
        )

        white = min(
            1.0,
            mean + 3.0 * std
        )

        self.black_edit.setText(
            f"{black:.6f}"
        )

        self.white_edit.setText(
            f"{white:.6f}"
        )

        self.gamma_edit.setText("1.0")

        self.apply_stretch()
    

    def __init__(self):

        super().__init__()

        self.gamma_edit = QLineEdit("1.0")
        self.setWindowTitle("Screen Stretch")

        self.data = None
        self.min_val = 0
        self.max_val = 1

        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)

        self.hist = HistogramCanvas()

        self.stats = QTextEdit()
        self.stats.setReadOnly(True)
        self.stats.setFixedHeight(120)

        self.black_edit = QLineEdit()
        self.gamma_edit = QLineEdit("1.0")
        self.white_edit = QLineEdit()

        apply_btn = QPushButton("Apply")
        reset_btn = QPushButton("Reset")
        auto_percentile_btn = QPushButton("Auto Percentile")
        auto_sigma_btn = QPushButton("Auto Sigma")

        apply_btn.clicked.connect(self.apply_stretch)
        reset_btn.clicked.connect(self.reset_stretch)
        auto_percentile_btn.clicked.connect(
            self.auto_stretch
        )

        auto_sigma_btn.clicked.connect(
            self.auto_sigma_stretch
)

        layout.addWidget(self.hist)

        layout.addWidget(QLabel("Black Point"))
        layout.addWidget(self.black_edit)

        layout.addWidget(QLabel("Gamma"))
        layout.addWidget(self.gamma_edit)

        layout.addWidget(QLabel("White Point"))
        layout.addWidget(self.white_edit)

        buttons = QHBoxLayout()

        buttons.addWidget(auto_sigma_btn)
        buttons.addWidget(auto_percentile_btn)
        buttons.addWidget(reset_btn)
        buttons.addWidget(apply_btn)

        layout.addLayout(buttons)

        layout.addWidget(self.stats)

    def set_image(self, data):

        print("shape =", data.shape)
        print("ndim =", data.ndim)

        self.data = data

        d = data.astype(np.float64)

        dmin = np.min(d)
        dmax = np.max(d)

        if dmax > dmin:

            norm = (
                d - dmin
            ) / (
                dmax - dmin
            )

        else:

            norm = np.zeros_like(
                d,
                dtype=np.float64
            )

        mean = np.mean(norm)

        std = np.std(norm)

        p1 = np.percentile(norm, 1)

        p99 = np.percentile(norm, 99)
        self.hist.mean = mean
        self.hist.std = std
        self.hist.p1 = p1
        self.hist.p99 = p99
        # Suggested initial values

        black = max(
            0.0,
            mean - std
        )

        white = min(
            1.0,
            mean + 3.0 * std
        )

        self.black_edit.setText(
            f"{black:.6f}"
        )

        self.white_edit.setText(
            f"{white:.6f}"
        )

        stats_text = (
            f"Datatype       {data.dtype}\n\n"
            f"Mean           {mean:.6f}\n\n"
            f"Std Dev        {std:.6f}\n\n"
            f"Mean - 1σ      {mean - std:.6f}\n\n"
            f"Mean + 1σ      {mean + std:.6f}\n\n"
            f"Mean + 2σ      {mean + 2*std:.6f}\n\n"
            f"Mean + 3σ      {mean + 3*std:.6f}\n\n"
            f"1% Percentile  {p1:.6f}\n\n"
            f"99% Percentile {p99:.6f}"
        )

        self.stats.setText(
            stats_text
        )

        self.hist.draw_histogram(
            data
        )

    def apply_stretch(self):

        try:
            black = float(self.black_edit.text())
            white = float(self.white_edit.text())

            if white <= black:
                return

            self.stretch_changed.emit(
                black,
                white
            )

        except Exception:
            pass

    def reset_stretch(self):

        self.black_edit.setText("0.0")
        self.white_edit.setText("1.0")

        self.apply_stretch()

    def auto_stretch(self):

        if self.data is None:
            return

        nz = self.data[self.data > 0]

        if len(nz) == 0:
            return

        black = np.percentile(nz, 0.1)
        white = np.percentile(nz, 99.9)

        self.black_edit.setText(f"{black:.6f}")
        self.white_edit.setText(f"{white:.6f}")

        self.apply_stretch()


# ============================================================
# Image Window
# ============================================================

class ImageWindow(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle("Image Viewer")

        self.data = None

        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)

        self.load_btn = QPushButton("Load Image")
        self.load_btn.clicked.connect(self.load_image)

        self.image_label = QLabel()
        self.image_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )
        self.image_label.setMouseTracking(True)
        self.image_label.mouseMoveEvent = self.mouse_move

        self.status_label = QLabel(
            "No image loaded"
        )

        layout.addWidget(self.load_btn)
        layout.addWidget(self.image_label)
        layout.addWidget(self.status_label)

        self.stretch_window = StretchWindow()
        self.stretch_window.show()

        self.stretch_window.stretch_changed.connect(
            self.update_display
        )

    def mouse_move(self, event):

        if self.data is None:
            return

        pixmap = self.image_label.pixmap()

        if pixmap is None:
            return

        label_w = self.image_label.width()
        label_h = self.image_label.height()

        img_h = self.data.shape[0]
        img_w = self.data.shape[1]

        x = int(event.position().x())
        y = int(event.position().y())

        ix = int(x * img_w / max(label_w,1))
        iy = int(y * img_h / max(label_h,1))

        ix = max(0, min(ix, img_w - 1))
        iy = max(0, min(iy, img_h - 1))

        dmin = np.min(self.data)
        dmax = np.max(self.data)

        if dmax <= dmin:
            return

        if self.data.ndim == 2:

            value = self.data[iy, ix]

            norm_value = (
                value - dmin
            ) / (
                dmax - dmin
            )

            self.status_label.setText(
                f"X={ix}  "
                f"Y={iy}  "
                f"Pixel={norm_value:.4f}"
            )

            self.stretch_window.hist.pixel_value = norm_value

            self.stretch_window.hist.draw_histogram(
                self.stretch_window.data
            )

        else:

            rgb = self.data[iy, ix]

            value = np.mean(rgb)

            norm_value = (
                value - dmin
            ) / (
                dmax - dmin
            )

            r = (
                rgb[0] - dmin
            ) / (
                dmax - dmin
            )

            g = (
                rgb[1] - dmin
            ) / (
                dmax - dmin
            )

            b = (
                rgb[2] - dmin
            ) / (
                dmax - dmin
            )

            self.status_label.setText(
                f"X={ix}  "
                f"Y={iy}  "
                f"R={r:.4f}  "
                f"G={g:.4f}  "
                f"B={b:.4f}  "
                f"N={norm_value:.4f}"
            )

            self.stretch_window.hist.pixel_value = norm_value

            self.stretch_window.hist.draw_histogram(
                self.stretch_window.data
            )

            self.stretch_window.hist.pixel_value = norm_value

            self.stretch_window.hist.draw_histogram(
                self.stretch_window.data
            )

        rgb = self.data[iy, ix]

    def load_image(self):

        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Images (*.fit *.fits *.fts *.png *.jpg *.jpeg *.tif *.tiff)"
        )

        if not file_name:
            return

        try:

            if file_name.lower().endswith(
                    (".fit", ".fits", ".fts")):

                data = fits.getdata(file_name)

                if data.ndim == 2:
                    pass

                elif data.ndim == 3 and data.shape[0] == 3:

                    data = np.transpose(
                        data,
                        (1, 2, 0)
                    )

                elif data.ndim == 3 and data.shape[2] == 3:
                    pass

                else:

                    raise ValueError(
                        f"Unsupported FITS shape: {data.shape}"
                    )

            else:

                data = cv2.imread(
                    file_name,
                    cv2.IMREAD_UNCHANGED
                )

                if data is None:
                    raise ValueError("Unable to read image")

                if data.ndim == 3:

                    data = cv2.cvtColor(
                        data,
                        cv2.COLOR_BGR2RGB
                    )

            #print("data exists:", 'data' in locals())
            #print("shape =", data.shape)

            print("dtype =", data.dtype)
            print("shape =", data.shape)

            print("min =", np.min(data))
            print("max =", np.max(data))

            nz = data[data > 0]

            print("min_nonzero =", np.min(nz))

            self.data = data.astype(np.float64)

            self.stretch_window.set_image(
                self.data
            )

            self.update_display(
                0.0,
                1.0
            )

        except Exception as e:

            self.status_label.setText(
                str(e)
            )
    def update_display(self,
                       black_point,
                       white_point):

        if self.data is None:
            return

        if white_point <= black_point:
            return

        dmin = np.min(self.data)
        dmax = np.max(self.data)

        if dmax <= dmin:
            return

        #
        # Normalize image to 0..1
        #

        img = (
            self.data - dmin
        ) / (
            dmax - dmin
        )

        #
        # Apply black / white points
        #

        img = np.clip(
            img,
            black_point,
            white_point
        )

        denom = max(
            white_point - black_point,
            1e-12
        )

        img = (
            img - black_point
        ) / denom

        img = np.clip(
            img,
            0,
            1
        )

        #
        # Gamma
        #

        try:

            gamma = float(
                self.stretch_window.gamma_edit.text()
            )

        except Exception:

            gamma = 1.0

        gamma = max(
            gamma,
            0.01
        )

        img = img ** gamma

        #
        # Convert for display
        #

        img = (
            img * 255
        ).astype(np.uint8)

        # Handle grayscale and RGB images

        if img.ndim == 2:

            h, w = img.shape

            img = np.ascontiguousarray(img)

            #print(img.shape)
            #print(img.dtype)

            qimg = QImage(
                img.tobytes(),
                w,
                h,
                w,
                QImage.Format.Format_Grayscale8
            )

        else:

            h, w, ch = img.shape

            img = np.ascontiguousarray(img)

            #print(img.shape)
            #print(img.dtype)

            qimg = QImage(
                img.tobytes(),
                w,
                h,
                3 * w,
                QImage.Format.Format_RGB888
            )

            qimg = QImage(
                img.tobytes(),
                w,
                h,
                3 * w,
                QImage.Format.Format_RGB888
            )

        pix = QPixmap.fromImage(qimg)

        self.image_label.setPixmap(
            pix.scaled(
                1200,
                900,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        )

        self.status_label.setText(
            f"Black={black_point:.2f}   "
            f"White={white_point:.2f}"
        )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    app = QApplication(sys.argv)

    image_window = ImageWindow()

    image_window.resize(1400, 1000)
    image_window.show()

    sys.exit(app.exec())