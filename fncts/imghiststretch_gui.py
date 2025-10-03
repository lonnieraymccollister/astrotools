import sys
import numpy as np
from scipy.special import erfinv
from astropy.io import fits
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QComboBox, QMessageBox, QFormLayout
)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ------------------------------
# Histogram specification functions
# ------------------------------
def rayleigh_specification(image, sigma):
    vals, inv_idx, counts = np.unique(image, return_inverse=True, return_counts=True)
    cdf = np.cumsum(counts).astype(np.float64) / counts.sum()
    epsilon = 1e-10
    safe_vals = np.clip(1 - cdf, epsilon, None)
    new_vals = sigma * np.sqrt(-2 * np.log(safe_vals))
    return new_vals[inv_idx].reshape(image.shape)

def gaussian_specification(image, mu, sigma):
    vals, inv_idx, counts = np.unique(image, return_inverse=True, return_counts=True)
    cdf = np.cumsum(counts).astype(np.float64) / counts.sum()
    epsilon = 1e-10
    safe_cdf = np.clip(cdf, epsilon, 1 - epsilon)
    new_vals = mu + sigma * np.sqrt(2) * erfinv(2 * safe_cdf - 1)
    return new_vals[inv_idx].reshape(image.shape)

def uniform_specification(image, lower, upper):
    vals, inv_idx, counts = np.unique(image, return_inverse=True, return_counts=True)
    cdf = np.cumsum(counts).astype(np.float64) / counts.sum()
    new_vals = lower + cdf * (upper - lower)
    return new_vals[inv_idx].reshape(image.shape)

def exponential_specification(image, lamb):
    vals, inv_idx, counts = np.unique(image, return_inverse=True, return_counts=True)
    cdf = np.cumsum(counts).astype(np.float64) / counts.sum()
    epsilon = 1e-10
    safe_vals = np.clip(1 - cdf, epsilon, None)
    new_vals = - (1.0 / lamb) * np.log(safe_vals)
    return new_vals[inv_idx].reshape(image.shape)

def lognormal_specification(image, mu, sigma_ln):
    vals, inv_idx, counts = np.unique(image, return_inverse=True, return_counts=True)
    cdf = np.cumsum(counts).astype(np.float64) / counts.sum()
    epsilon = 1e-10
    safe_cdf = np.clip(cdf, epsilon, 1 - epsilon)
    new_vals = np.exp(mu + sigma_ln * np.sqrt(2) * erfinv(2 * safe_cdf - 1))
    return new_vals[inv_idx].reshape(image.shape)

# ------------------------------
# Main GUI
# ------------------------------
class HistSpecGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Histogram Specification")
        self.image = None
        self.specified_image = None
        self.hdul_header = None

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()

        # File chooser
        file_layout = QHBoxLayout()
        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText("Select input FITS file...")
        btn_browse = QPushButton("Browse")
        btn_browse.clicked.connect(self.browse_file)
        file_layout.addWidget(self.file_edit)
        file_layout.addWidget(btn_browse)
        layout.addLayout(file_layout)

        # Distribution selector and parameter entries
        form = QFormLayout()
        self.dist_combo = QComboBox()
        self.dist_combo.addItems(["Rayleigh", "Gaussian", "Uniform", "Exponential", "Lognormal"])
        self.dist_combo.currentIndexChanged.connect(self.on_dist_change)
        form.addRow(QLabel("Distribution"), self.dist_combo)

        # Parameter widgets
        self.param_mu = QLineEdit()
        self.param_sigma = QLineEdit()
        self.param_lower = QLineEdit()
        self.param_upper = QLineEdit()
        self.param_lambda = QLineEdit()
        self.param_sigma_ln = QLineEdit()

        self.param_mu.setPlaceholderText("mu (used for Gaussian/lognormal)")
        self.param_sigma.setPlaceholderText("sigma (used for Gaussian)")
        self.param_lower.setPlaceholderText("lower (uniform)")
        self.param_upper.setPlaceholderText("upper (uniform)")
        self.param_lambda.setPlaceholderText("lambda (exponential)")
        self.param_sigma_ln.setPlaceholderText("sigma (lognormal)")

        form.addRow("Mu", self.param_mu)
        form.addRow("Sigma", self.param_sigma)
        form.addRow("Lower", self.param_lower)
        form.addRow("Upper", self.param_upper)
        form.addRow("Lambda", self.param_lambda)
        form.addRow("Sigma LN", self.param_sigma_ln)

        layout.addLayout(form)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_load = QPushButton("Load FITS")
        btn_load.clicked.connect(self.load_fits)
        btn_apply = QPushButton("Apply")
        btn_apply.clicked.connect(self.apply_specification)
        btn_show_hist = QPushButton("Show Histograms")
        btn_show_hist.clicked.connect(self.show_histograms)
        btn_save = QPushButton("Save Specified FITS")
        btn_save.clicked.connect(self.save_specified)

        btn_layout.addWidget(btn_load)
        btn_layout.addWidget(btn_apply)
        btn_layout.addWidget(btn_show_hist)
        btn_layout.addWidget(btn_save)
        layout.addLayout(btn_layout)

        # Status
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

        # Matplotlib canvas (for inline display if desired)
        self.fig = Figure(figsize=(6, 3))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)
        self.canvas.hide()  # hide by default; shown when plotting inline

        self.setLayout(layout)
        self.on_dist_change(0)  # initialize fields

    def browse_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open FITS", "", "FITS Files (*.fits *.fit);;All Files (*)")
        if path:
            self.file_edit.setText(path)

    def load_fits(self):
        path = self.file_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "No file", "Please choose a FITS file first.")
            return
        try:
            with fits.open(path) as hdul:
                data = hdul[0].data.astype(np.float64)
                header = hdul[0].header
            if data is None:
                raise ValueError("No image data found in primary HDU.")
            self.image = data
            self.hdul_header = header
            self.status_label.setText(f"Loaded {path} shape={self.image.shape}")
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))
            self.status_label.setText("Load failed")

    def on_dist_change(self, index):
        # Enable/disable parameter fields depending on distribution
        name = self.dist_combo.currentText().lower()
        # default disable all
        for w in [self.param_mu, self.param_sigma, self.param_lower, self.param_upper, self.param_lambda, self.param_sigma_ln]:
            w.setEnabled(False)
            w.clear()
        if name == "rayleigh":
            self.param_sigma.setEnabled(True)
            self.param_sigma.setPlaceholderText("sigma (default: std of image)")
        elif name == "gaussian":
            self.param_mu.setEnabled(True)
            self.param_sigma.setEnabled(True)
            self.param_mu.setPlaceholderText("mu (default: mean of image)")
            self.param_sigma.setPlaceholderText("sigma (default: std of image)")
        elif name == "uniform":
            self.param_lower.setEnabled(True)
            self.param_upper.setEnabled(True)
        elif name == "exponential":
            self.param_lambda.setEnabled(True)
            self.param_lambda.setPlaceholderText("lambda (positive)")
        elif name == "lognormal":
            self.param_mu.setEnabled(True)
            self.param_sigma_ln.setEnabled(True)
            self.param_mu.setPlaceholderText("mu (e.g., 0.0)")
            self.param_sigma_ln.setPlaceholderText("sigma (e.g., 0.5)")

    def apply_specification(self):
        if self.image is None:
            QMessageBox.warning(self, "No image", "Load a FITS image first.")
            return
        try:
            name = self.dist_combo.currentText().lower()
            if name == "rayleigh":
                sigma = self._get_float(self.param_sigma, default=np.std(self.image))
                self.specified_image = rayleigh_specification(self.image, sigma)
                out_name = "image_rayleigh_specified.fits"
            elif name == "gaussian":
                mu = self._get_float(self.param_mu, default=np.mean(self.image))
                sigma = self._get_float(self.param_sigma, default=np.std(self.image))
                self.specified_image = gaussian_specification(self.image, mu, sigma)
                out_name = "image_gaussian_specified.fits"
            elif name == "uniform":
                lower = self._get_float(self.param_lower, default=float(np.nanmin(self.image)))
                upper = self._get_float(self.param_upper, default=float(np.nanmax(self.image)))
                if upper <= lower:
                    raise ValueError("Upper must be greater than lower for uniform mapping.")
                self.specified_image = uniform_specification(self.image, lower, upper)
                out_name = "image_uniform_specified.fits"
            elif name == "exponential":
                lamb = self._get_float(self.param_lambda, default=0.1)
                if lamb <= 0:
                    raise ValueError("Lambda must be positive.")
                self.specified_image = exponential_specification(self.image, lamb)
                out_name = "image_exponential_specified.fits"
            elif name == "lognormal":
                mu = self._get_float(self.param_mu, default=0.0)
                sigma_ln = self._get_float(self.param_sigma_ln, default=0.5)
                if sigma_ln <= 0:
                    raise ValueError("Lognormal sigma must be positive.")
                self.specified_image = lognormal_specification(self.image, mu, sigma_ln)
                out_name = "image_lognormal_specified.fits"
            else:
                raise ValueError("Unknown distribution selected.")

            # Save automatically to working dir for convenience
            fits.writeto(out_name, self.specified_image, header=self.hdul_header, overwrite=True)
            self.status_label.setText(f"Applied {name}, saved {out_name}")
            QMessageBox.information(self, "Done", f"Specification applied and saved as {out_name}")
        except Exception as e:
            QMessageBox.critical(self, "Apply error", str(e))
            self.status_label.setText("Apply failed")

    def _get_float(self, widget, default=None):
        text = widget.text().strip()
        if text == "":
            if default is not None:
                return default
            raise ValueError(f"Missing parameter for {widget.placeholderText()}")
        return float(text)

    def show_histograms(self):
        if self.image is None:
            QMessageBox.warning(self, "No image", "Load a FITS image first.")
            return
        if self.specified_image is None:
            QMessageBox.warning(self, "No specified image", "Apply a specification first.")
            return

        # Prepare figure
        self.fig.clf()
        ax1 = self.fig.add_subplot(1, 2, 1)
        ax2 = self.fig.add_subplot(1, 2, 2)

        ax1.hist(self.image.ravel(), bins=256, color='blue', histtype='step')
        ax1.set_title("Original Image Histogram")
        ax1.set_xlabel("Intensity")
        ax1.set_ylabel("Frequency")

        ax2.hist(self.specified_image.ravel(), bins=256, color='red', histtype='step')
        ax2.set_title(f"Specified ({self.dist_combo.currentText()}) Histogram")
        ax2.set_xlabel("Intensity")
        ax2.set_ylabel("Frequency")

        self.canvas.draw()
        self.canvas.show()
        self.status_label.setText("Displayed histograms")

    def save_specified(self):
        if self.specified_image is None:
            QMessageBox.warning(self, "No specified image", "Apply a specification first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save specified FITS", "", "FITS Files (*.fits *.fit);;All Files (*)")
        if not path:
            return
        try:
            fits.writeto(path, self.specified_image, header=self.hdul_header, overwrite=True)
            self.status_label.setText(f"Saved specified image to {path}")
            QMessageBox.information(self, "Saved", f"Saved specified image to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Save error", str(e))
            self.status_label.setText("Save failed")

# ------------------------------
# Run
# ------------------------------
def main():
    app = QApplication(sys.argv)
    win = HistSpecGUI()
    win.resize(900, 600)
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()