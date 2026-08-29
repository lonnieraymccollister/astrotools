import csv
import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QFileDialog, QVBoxLayout, QHBoxLayout, QScrollArea,
    QDialog, QFormLayout, QLineEdit, QDialogButtonBox
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QSize
from skimage.metrics import structural_similarity as ssim


# ---------------- Manual Point Entry Dialog ----------------
class ManualPointDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Enter Points Manually")

        layout = QFormLayout(self)

        self.x1 = QLineEdit()
        self.y1 = QLineEdit()
        self.x2 = QLineEdit()
        self.y2 = QLineEdit()

        layout.addRow("Point 1 - X:", self.x1)
        layout.addRow("Point 1 - Y:", self.y1)
        layout.addRow("Point 2 - X:", self.x2)
        layout.addRow("Point 2 - Y:", self.y2)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout.addWidget(buttons)

    def get_points(self):
        return (
            int(self.x1.text()),
            int(self.y1.text()),
            int(self.x2.text()),
            int(self.y2.text())
        )


# ---------------- Main GUI ----------------
class SymmetryGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Symmetry Flip Tool")
        self.resize(1200, 900)

        self.points = []
        self.img = None
        self.img_path = None

        # --- Main Layout ---
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)

        # --- Buttons ---
        btn_layout = QHBoxLayout()

        self.load_btn = QPushButton("Load Image")
        self.load_btn.clicked.connect(self.load_image)
        btn_layout.addWidget(self.load_btn)

        self.run_btn = QPushButton("Run Symmetry Flip")
        self.run_btn.clicked.connect(self.run_symmetry)
        btn_layout.addWidget(self.run_btn)

        self.scan_btn = QPushButton("Auto Scan Symmetry Axes")
        self.scan_btn.clicked.connect(self.auto_scan_symmetry)
        btn_layout.addWidget(self.scan_btn)

        # NEW BUTTON — manual point entry
        self.manual_btn = QPushButton("Enter Points Manually")
        self.manual_btn.clicked.connect(self.enter_points_manually)
        btn_layout.addWidget(self.manual_btn)

        main_layout.addLayout(btn_layout)

        # --- Scrollable Image Area ---
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(False)
        main_layout.addWidget(self.scroll)

        self.label = QLabel("Load an image and click two points.")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("background-color: #222; color: white;")
        self.scroll.setWidget(self.label)

        self.setCentralWidget(main_widget)

    # ---------------- Manual Point Entry Handler ----------------
    def enter_points_manually(self):
        if self.img is None:
            print("Load an image first.")
            return

        dlg = ManualPointDialog()
        if dlg.exec() == QDialog.DialogCode.Accepted:
            x1, y1, x2, y2 = dlg.get_points()

            self.points = [(x1, y1), (x2, y2)]

            temp = self.img.copy()
            cv2.circle(temp, (x1, y1), 5, (0, 0, 255), -1)
            cv2.circle(temp, (x2, y2), 5, (0, 0, 255), -1)

            self.show_image(temp)

            print(f"Manual points set: ({x1}, {y1}) and ({x2}, {y2})")

    # ---------------- Symmetry Score ----------------
    def compute_symmetry_score(self, gray, sym_gray):
        mask = gray > np.percentile(gray, 20)

        score, ssim_map = ssim(
            gray,
            sym_gray,
            data_range=255,
            full=True
        )

        masked_score = float(np.mean(ssim_map[mask]))
        return masked_score, ssim_map

    # ---------------- Local Symmetry Map ----------------
    def local_point_symmetry(self, img, angle, center):
        h, w = img.shape[:2]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

        rot_angle = 90.0 - angle
        rot_mat = cv2.getRotationMatrix2D(tuple(center), rot_angle, 1.0)
        inv_rot = cv2.getRotationMatrix2D(tuple(center), -rot_angle, 1.0)

        rotated = cv2.warpAffine(gray, rot_mat, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT)

        mirrored = cv2.flip(rotated, 1)

        sym_local = cv2.warpAffine(mirrored, inv_rot, (w, h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT)

        diff = np.abs(gray - sym_local)
        diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        heat = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img, 0.7, heat, 0.3, 0)

        return heat, overlay, diff_norm

    # ---------------- Angle Search ----------------
    def search_angles(self, img, center, candidate_angles, p1=None, p2=None):
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        best_score = -1.0
        best_angle = None
        best_img = None

        for angle in candidate_angles:
            sym_img, rot_mat, inv_rot = self.make_symmetric(img, angle, center, w, h)

            if p1 is not None:
                p2_rot = np.dot(rot_mat[:, :2], p2) + rot_mat[:, 2]
                p2_flip = np.array([w - p2_rot[0], p2_rot[1]], dtype=np.float32)
                p2_final = np.dot(inv_rot[:, :2], p2_flip) + inv_rot[:, 2]

                delta = p1 - p2_final
                M = np.float32([[1, 0, delta[0]], [0, 1, delta[1]]])
                sym_img = cv2.warpAffine(sym_img, M, (w, h))

            sym_gray = cv2.cvtColor(sym_img, cv2.COLOR_BGR2GRAY)
            score, _ = self.compute_symmetry_score(gray, sym_gray)

            if score > best_score:
                best_score = score
                best_angle = angle
                best_img = sym_img

        return best_angle, best_score, best_img

    # ---------------- Axis Refinement ----------------
    def refine_axis(self, img, center, p1=None, p2=None):
        coarse_angles = np.arange(0, 180, 1.0)
        best_angle, _, _ = self.search_angles(img, center, coarse_angles, p1, p2)

        medium_angles = np.arange(best_angle - 2, best_angle + 2, 0.1)
        best_angle, _, _ = self.search_angles(img, center, medium_angles, p1, p2)

        fine_angles = np.arange(best_angle - 0.2, best_angle + 0.2, 0.01)
        return self.search_angles(img, center, fine_angles, p1, p2)

    # ---------------- Load Image ----------------
    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if not file_path:
            return

        self.img_path = file_path
        self.img = cv2.imread(file_path)
        if self.img is None:
            print("Failed to load image.")
            return

        self.points = []
        self.show_image(self.img)
        self.label.mousePressEvent = self.get_click

    # ---------------- Display Image ----------------
    def show_image(self, img):
        h, w, ch = img.shape
        bytes_per_line = ch * w
        qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
        pix = QPixmap.fromImage(qimg)

        self.label.setPixmap(pix)
        self.label.resize(QSize(w, h))

    # ---------------- Mouse Click Handler ----------------
    def get_click(self, event):
        if self.img is None:
            return

        x = int(event.position().x())
        y = int(event.position().y())
        self.points.append((x, y))

        temp = self.img.copy()
        for px, py in self.points:
            cv2.circle(temp, (px, py), 5, (0, 0, 255), -1)

        self.show_image(temp)

    # ---------------- Make Symmetric ----------------
    def make_symmetric(self, img, angle, center, w, h):
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, rot_mat, (w, h))
        flipped = cv2.flip(rotated, 1)
        inv_rot = cv2.getRotationMatrix2D(center, -angle, 1.0)
        final = cv2.warpAffine(flipped, inv_rot, (w, h))
        return final, rot_mat, inv_rot

    # ---------------- Error Heatmap ----------------
    def make_error_heatmap(self, img, aligned, cmap=cv2.COLORMAP_JET):
        diff = cv2.absdiff(img, aligned)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        heat = cv2.applyColorMap(norm.astype(np.uint8), cmap)
        overlay = cv2.addWeighted(img, 0.6, heat, 0.4, 0)
        return heat, overlay, norm

    # ---------------- Run Symmetry Flip ----------------
    def run_symmetry(self):
        if self.img is None:
            print("No image loaded.")
            return

        if len(self.points) != 2:
            print("You must click exactly two points.")
            return

        img = self.img.copy()
        p1 = np.array(self.points[0], dtype=np.float32)
        p2 = np.array(self.points[1], dtype=np.float32)

        angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) * 180.0 / np.pi
        center = np.array([(p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0], dtype=np.float32)

        h, w, _ = img.shape

        sym_img, rot_mat, inv_rot = self.make_symmetric(img, angle, center, w, h)

        p2_rot = np.dot(rot_mat[:, :2], p2) + rot_mat[:, 2]
        p2_flip = np.array([w - p2_rot[0], p2_rot[1]], dtype=np.float32)
        p2_final = np.dot(inv_rot[:, :2], p2_flip) + inv_rot[:, 2]

        delta = p1 - p2_final
        M = np.float32([[1, 0, delta[0]], [0, 1, delta[1]]])
        aligned = cv2.warpAffine(sym_img, M, (w, h))

        out_sym = self.img_path.rsplit(".", 1)[0] + "_symmetric_aligned.jpg"
        cv2.imwrite(out_sym, aligned)

        out_orig = self.img_path.rsplit(".", 1)[0] + "_original.jpg"
        cv2.imwrite(out_orig, img)

        combined = cv2.addWeighted(img, 0.5, aligned, 0.5, 0)
        out_comb = self.img_path.rsplit(".", 1)[0] + "_combined.jpg"
        cv2.imwrite(out_comb, combined)

        combined_axis = combined.copy()
        cv2.line(combined_axis,
                 (int(p1[0]), int(p1[1])),
                 (int(p2[0]), int(p2[1])),
                 (0, 255, 0), 2)

        out_axis = self.img_path.rsplit(".", 1)[0] + "_combined_axis.jpg"
        cv2.imwrite(out_axis, combined_axis)

        heat, overlay, _ = self.make_error_heatmap(img, aligned)
        out_heat = self.img_path.rsplit(".", 1)[0] + "_manual_error_heatmap.jpg"
        out_overlay = self.img_path.rsplit(".", 1)[0] + "_manual_error_overlay.jpg"
        cv2.imwrite(out_heat, heat)
        cv2.imwrite(out_overlay, overlay)

        print("Saved all output images.")

    # ---------------- Auto Scan ----------------
    def auto_scan_symmetry(self):
        if self.img is None:
            print("No image loaded.")
            return

        img = self.img.copy()
        h, w, _ = img.shape

        use_points = (len(self.points) == 2)
        if use_points:
            p1 = np.array(self.points[0], dtype=np.float32)
            p2 = np.array(self.points[1], dtype=np.float32)
            center = np.array([(p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0], dtype=np.float32)
            cx, cy = int(center[0]), int(center[1])
        else:
            cx, cy = w // 2, h // 2
            center = np.array([cx, cy], dtype=np.float32)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
        total_signal = float(np.sum(gray)) + 1e-12

        if use_points:
            best_angle, best_score, best_aligned = self.refine_axis(img, center, p1, p2)
        else:
            best_angle, best_score, best_aligned = self.refine_axis(img, center)

        if best_aligned is None:
            print("Auto-scan found no candidate.")
            return

        out_sym = self.img_path.rsplit(".", 1)[0] + "_autoscan_symmetric.jpg"
        cv2.imwrite(out_sym, best_aligned)

        combined = cv2.addWeighted(img, 0.5, best_aligned, 0.5, 0)
        out_comb = self.img_path.rsplit(".", 1)[0] + "_autoscan_combined.jpg"
        cv2.imwrite(out_comb, combined)

        length = int(max(w, h) * 1.5)
        dx = int(length * np.cos(np.radians(best_angle)))
        dy = int(length * np.sin(np.radians(best_angle)))

        combined_axis = combined.copy()
        cv2.line(combined_axis,
                 (cx - dx, cy - dy),
                 (cx + dx, cy + dy),
                 (0, 255, 0), 2)

        out_axis = self.img_path.rsplit(".", 1)[0] + "_autoscan_axis.jpg"
        cv2.imwrite(out_axis, combined_axis)

        heat, overlay, _ = self.make_error_heatmap(img, best_aligned)
        out_heat = self.img_path.rsplit(".", 1)[0] + "_autoscan_error_heatmap.jpg"
        out_overlay = self.img_path.rsplit(".", 1)[0] + "_autoscan_error_overlay.jpg"
        cv2.imwrite(out_heat, heat)
        cv2.imwrite(out_overlay, overlay)

        heat2, local_overlay, diff_norm = self.local_point_symmetry(img, best_angle, center)

        base = self.img_path.rsplit(".", 1)[0]
        cv2.imwrite(base + "_local_heatmap.jpg", heat2)
        cv2.imwrite(base + "_local_overlay.jpg", local_overlay)
        cv2.imwrite(base + "_local_diff.jpg", diff_norm)

        print(
            f"Auto-scan complete\n"
            f"Best angle = {best_angle:.4f}\n"
            f"Masked SSIM = {best_score:.5f}"
        )


# ---------------- Main ----------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SymmetryGUI()
    window.show()
    sys.exit(app.exec())
