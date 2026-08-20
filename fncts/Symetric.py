import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QFileDialog, QVBoxLayout, QHBoxLayout, QScrollArea
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QSize


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

    # ---------------- Load Image ----------------
    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if not file_path:
            return

        self.img_path = file_path
        self.img = cv2.imread(file_path)
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

        angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) * 180 / np.pi
        center = np.array([(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2], dtype=np.float32)

        h, w, _ = img.shape

        def make_symmetric(img, angle):
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img, rot_mat, (w, h))
            flipped = cv2.flip(rotated, 1)
            inv_rot = cv2.getRotationMatrix2D(center, -angle, 1.0)
            final = cv2.warpAffine(flipped, inv_rot, (w, h))
            return final, rot_mat, inv_rot

        # --- Symmetric image ---
        sym_img, rot_mat, inv_rot = make_symmetric(img, angle)

        # Transform p2 through the same matrices
        p2_rot = np.dot(rot_mat[:, :2], p2) + rot_mat[:, 2]
        p2_flip = np.array([w - p2_rot[0], p2_rot[1]], dtype=np.float32)
        p2_final = np.dot(inv_rot[:, :2], p2_flip) + inv_rot[:, 2]

        delta = p1 - p2_final
        M = np.float32([[1, 0, delta[0]], [0, 1, delta[1]]])
        aligned = cv2.warpAffine(sym_img, M, (w, h))

        # Save aligned symmetric image
        out_sym = self.img_path.rsplit(".", 1)[0] + "_symmetric_aligned.jpg"
        cv2.imwrite(out_sym, aligned)

        # Save original
        out_orig = self.img_path.rsplit(".", 1)[0] + "_original.jpg"
        cv2.imwrite(out_orig, img)

        # Combined
        combined = cv2.addWeighted(img, 0.5, aligned, 0.5, 0)
        out_comb = self.img_path.rsplit(".", 1)[0] + "_combined.jpg"
        cv2.imwrite(out_comb, combined)

        # Combined with axis
        combined_axis = combined.copy()
        cv2.line(combined_axis,
                 (int(p1[0]), int(p1[1])),
                 (int(p2[0]), int(p2[1])),
                 (0, 255, 0), 2)

        out_axis = self.img_path.rsplit(".", 1)[0] + "_combined_axis.jpg"
        cv2.imwrite(out_axis, combined_axis)

        print("Saved all output images.")


# ---------------- Main ----------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SymmetryGUI()
    window.show()
    sys.exit(app.exec())
