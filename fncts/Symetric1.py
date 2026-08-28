import csv
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

        self.scan_btn = QPushButton("Auto Scan Symmetry Axes")
        self.scan_btn.clicked.connect(self.auto_scan_symmetry)
        btn_layout.addWidget(self.scan_btn)

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

    # ---------------- Helper: make_symmetric ----------------
    def make_symmetric(self, img, angle, center, w, h):
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, rot_mat, (w, h))
        flipped = cv2.flip(rotated, 1)
        inv_rot = cv2.getRotationMatrix2D(center, -angle, 1.0)
        final = cv2.warpAffine(flipped, inv_rot, (w, h))
        return final, rot_mat, inv_rot

    # ---------------- Helper: create heatmap from error ----------------
    def make_error_heatmap(self, img, aligned, cmap=cv2.COLORMAP_JET):
        # Compute absolute difference and convert to grayscale
        diff = cv2.absdiff(img, aligned)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # Normalize to 0-255
        norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

        # Apply color map
        heat = cv2.applyColorMap(norm.astype(np.uint8), cmap)

        # Optionally overlay heatmap on original for visualization
        overlay = cv2.addWeighted(img, 0.6, heat, 0.4, 0)

        return heat, overlay, norm

    # ---------------- Run Symmetry Flip (manual 2-point) ----------------
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

        # Create symmetric image (rotate -> flip -> rotate back)
        sym_img, rot_mat, inv_rot = self.make_symmetric(img, angle, center, w, h)

        # Transform p2 through the same matrices to find where it lands
        p2_rot = np.dot(rot_mat[:, :2], p2) + rot_mat[:, 2]
        p2_flip = np.array([w - p2_rot[0], p2_rot[1]], dtype=np.float32)
        p2_final = np.dot(inv_rot[:, :2], p2_flip) + inv_rot[:, 2]

        # Translate so p2_final aligns with p1 (Maxim DL behavior)
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
    
        # Heatmap for this manual alignment (optional)
        heat, overlay, _ = self.make_error_heatmap(img, aligned)
        out_heat = self.img_path.rsplit(".", 1)[0] + "_manual_error_heatmap.jpg"
        out_overlay = self.img_path.rsplit(".", 1)[0] + "_manual_error_overlay.jpg"
        cv2.imwrite(out_heat, heat)
        cv2.imwrite(out_overlay, overlay)
    
        print("Saved all output images.")
    def auto_scan_symmetry(self):
        if self.img is None:
            print("No image loaded.")
            return
    
        img = self.img.copy()
        h, w, _ = img.shape
    
        # Determine center and whether to use clicked points
        use_points = (len(self.points) == 2)
        if use_points:
            p1 = np.array(self.points[0], dtype=np.float32)
            p2 = np.array(self.points[1], dtype=np.float32)
            center = np.array([(p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0], dtype=np.float32)
            cx, cy = int(center[0]), int(center[1])
        else:
            cx, cy = w // 2, h // 2
            center = np.array([cx, cy], dtype=np.float32)

        # Precompute grayscale and total signal for normalization
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
        total_signal = float(np.sum(gray)) + 1e-12
    
        angles = np.linspace(0, 180, num=180)
        results = []  # list of (angle, raw_distance)
    
        best_score = float("inf")
        best_angle = None
        best_aligned = None

        for angle in angles:
            # rotate -> flip -> rotate back
            sym_img, rot_mat, inv_rot = self.make_symmetric(img, angle, center, w, h)
    
            if use_points:
                # Transform p2 through the same matrices to find where it lands
                p2_rot = np.dot(rot_mat[:, :2], p2) + rot_mat[:, 2]
                p2_flip = np.array([w - p2_rot[0], p2_rot[1]], dtype=np.float32)
                p2_final = np.dot(inv_rot[:, :2], p2_flip) + inv_rot[:, 2]
    
                # Translate so p2_final aligns with p1
                delta = p1 - p2_final
                M = np.float32([[1, 0, delta[0]], [0, 1, delta[1]]])
                sym_aligned = cv2.warpAffine(sym_img, M, (w, h))
            else:
                # Align center point (fallback)
                c = np.array([cx, cy], dtype=np.float32)
                c_rot = np.dot(rot_mat[:, :2], c) + rot_mat[:, 2]
                c_flip = np.array([w - c_rot[0], c_rot[1]], dtype=np.float32)
                c_final = np.dot(inv_rot[:, :2], c_flip) + inv_rot[:, 2]
                delta = np.array([cx, cy], dtype=np.float32) - c_final
                M = np.float32([[1, 0, delta[0]], [0, 1, delta[1]]])
                sym_aligned = cv2.warpAffine(sym_img, M, (w, h))
    
            # compute raw distance on grayscale (sum absolute differences)
            sym_gray = cv2.cvtColor(sym_aligned, cv2.COLOR_BGR2GRAY).astype(np.float64)
            raw_D = float(np.sum(np.abs(gray - sym_gray)))
    
            results.append((float(angle), raw_D))
    
            if raw_D < best_score:
                best_score = raw_D
                best_angle = float(angle)
                best_aligned = sym_aligned
    
        if best_aligned is None:
            print("Auto-scan found no candidate.")
            return
    
        # Convert raw distances to normalized scores and percentiles
        Ds = np.array([d for _, d in results], dtype=np.float64)
        angles_arr = np.array([a for a, _ in results], dtype=np.float64)
    
        # Normalized score: 1 - (D / total_signal), clipped to [0,1]
        norm_scores = 1.0 - (Ds / (total_signal + 1e-12))
        norm_scores = np.clip(norm_scores, 0.0, 1.0)
    
        # Percentile: higher percentile = better symmetry (100 = best)
        ranks = Ds.argsort().argsort()  # 0..N-1 where 0 is best (smallest D)
        percentiles = 100.0 * (1.0 - ranks / (len(Ds) - 1.0)) if len(Ds) > 1 else np.array([100.0])
    
        # --- Save CSV of all angles and scores (raw, normalized, percentile) ---
        csv_all = self.img_path.rsplit(".", 1)[0] + "_autoscan_scores_with_norm.csv"
        try:
            with open(csv_all, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["angle_degrees", "raw_distance", "normalized_score", "percentile"])
                for ang, d, s, p in zip(angles_arr, Ds, norm_scores, percentiles):
                    writer.writerow([f"{ang:.6f}", f"{d:.6f}", f"{s:.6f}", f"{p:.2f}"])
            print(f"Saved full scores CSV: {csv_all}")
        except Exception as e:
            print(f"Failed to save CSV {csv_all}: {e}")
    
        # --- Save CSV of top N best axes ---
        top_n = 10
        order = np.argsort(Ds)
        csv_top = self.img_path.rsplit(".", 1)[0] + f"_autoscan_top{top_n}_scores.csv"
        try:
            with open(csv_top, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["rank", "angle_degrees", "raw_distance", "normalized_score", "percentile"])
                for i, idx in enumerate(order[:top_n], start=1):
                    writer.writerow([i,
                                     f"{angles_arr[idx]:.6f}",
                                     f"{Ds[idx]:.6f}",
                                     f"{norm_scores[idx]:.6f}",
                                     f"{percentiles[idx]:.2f}"])
            print(f"Saved top {top_n} scores CSV: {csv_top}")
        except Exception as e:
            print(f"Failed to save CSV {csv_top}: {e}")
    
        # Save best aligned symmetric image and outputs (same as before)
        out_sym = self.img_path.rsplit(".", 1)[0] + "_autoscan_symmetric.jpg"
        cv2.imwrite(out_sym, best_aligned)
    
        combined = cv2.addWeighted(img, 0.5, best_aligned, 0.5, 0)
        out_comb = self.img_path.rsplit(".", 1)[0] + "_autoscan_combined.jpg"
        cv2.imwrite(out_comb, combined)
    
        # Draw axis (through chosen center at best_angle)
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
    
        # Heatmap for autoscan best alignment
        heat, overlay, _ = self.make_error_heatmap(img, best_aligned)
        out_heat = self.img_path.rsplit(".", 1)[0] + "_autoscan_error_heatmap.jpg"
        out_overlay = self.img_path.rsplit(".", 1)[0] + "_autoscan_error_overlay.jpg"
        cv2.imwrite(out_heat, heat)
        cv2.imwrite(out_overlay, overlay)
    
        print(f"Auto-scan complete. Best angle = {best_angle:.2f} degrees. Best raw distance = {best_score:.2f}. Normalized = {1.0 - best_score/total_signal:.3f}")
            


# ---------------- Main ----------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SymmetryGUI()
    window.show()
    sys.exit(app.exec())
