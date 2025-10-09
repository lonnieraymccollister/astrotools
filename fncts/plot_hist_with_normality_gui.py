#!/usr/bin/env python3
"""
plot_normality_gui.py

Requires: PyQt6, pandas, numpy, matplotlib, scipy, (statsmodels optional)

Save and run:
    python plot_normality_gui.py
"""
import sys
from difflib import get_close_matches
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, shapiro, normaltest, anderson
import warnings

from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QComboBox, QLineEdit,
    QTextEdit, QGridLayout, QFileDialog, QSpinBox, QHBoxLayout, QMessageBox
)
from PyQt6.QtGui import QDoubleValidator, QIntValidator

try:
    import statsmodels.api as sm
    _HAVE_SM = True
except Exception:
    _HAVE_SM = False

def fuzzy_col_match(cols, want, cutoff=0.35):
    if want is None or want == "":
        return None
    cols_l = [c.lower() for c in cols]
    want_l = want.lower()
    if want in cols:
        return want
    if want_l in cols_l:
        return cols[cols_l.index(want_l)]
    for c, cl in zip(cols, cols_l):
        if want_l in cl or cl in want_l:
            return c
    matches = get_close_matches(want_l, cols_l, n=1, cutoff=cutoff)
    if matches:
        return cols[cols_l.index(matches[0])]
    return None

def run_normality_tests(values):
    results = {}
    # Shapiro-Wilk (note: n > 5000 may be skipped)
    try:
        if len(values) <= 5000:
            stat, p = shapiro(values)
            results['shapiro'] = (stat, p)
        else:
            results['shapiro'] = (np.nan, np.nan)
    except Exception:
        results['shapiro'] = (np.nan, np.nan)

    # D'Agostino's K^2
    try:
        stat, p = normaltest(values)
        results['dagostino_k2'] = (stat, p)
    except Exception:
        results['dagostino_k2'] = (np.nan, np.nan)

    # Anderson-Darling
    try:
        ad = anderson(values, dist='norm')
        results['anderson'] = ad
    except Exception:
        results['anderson'] = None

    return results

class NormalityGui(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Normality inspector")
        self.resize(920, 520)
        self.df = None
        self.values = None

        layout = QGridLayout()
        row = 0

        btn_open = QPushButton("Open CSV")
        btn_open.clicked.connect(self.open_csv)
        layout.addWidget(btn_open, row, 0)

        self.lbl_file = QLabel("No file loaded")
        layout.addWidget(self.lbl_file, row, 1, 1, 4)
        row += 1

        layout.addWidget(QLabel("Column:"), row, 0)
        self.combo_col = QComboBox()
        layout.addWidget(self.combo_col, row, 1)

        btn_refresh = QPushButton("Refresh cols")
        btn_refresh.clicked.connect(self.refresh_columns)
        layout.addWidget(btn_refresh, row, 2)

        btn_load_col = QPushButton("Load column")
        btn_load_col.clicked.connect(self.load_selected_column)
        layout.addWidget(btn_load_col, row, 3)

        row += 1
        # Stat entries
        layout.addWidget(QLabel("min_val"), row, 0)
        self.edit_min = QLineEdit()
        self.edit_min.setValidator(QDoubleValidator())
        layout.addWidget(self.edit_min, row, 1)

        layout.addWidget(QLabel("max_val"), row, 2)
        self.edit_max = QLineEdit()
        self.edit_max.setValidator(QDoubleValidator())
        layout.addWidget(self.edit_max, row, 3)

        row += 1
        layout.addWidget(QLabel("mean"), row, 0)
        self.edit_mean = QLineEdit()
        self.edit_mean.setValidator(QDoubleValidator())
        layout.addWidget(self.edit_mean, row, 1)

        layout.addWidget(QLabel("std"), row, 2)
        self.edit_std = QLineEdit()
        self.edit_std.setValidator(QDoubleValidator())
        layout.addWidget(self.edit_std, row, 3)

        row += 1
        layout.addWidget(QLabel("n"), row, 0)
        self.spin_n = QSpinBox()
        self.spin_n.setMinimum(1)
        self.spin_n.setMaximum(10_000_000)
        layout.addWidget(self.spin_n, row, 1)

        # action buttons
        btn_calc = QPushButton("Compute stats from CSV column")
        btn_calc.clicked.connect(self.compute_stats_from_loaded_values)
        layout.addWidget(btn_calc, row, 2, 1, 2)
        row += 1

        # bins and plot button row
        layout.addWidget(QLabel("bins"), row, 0)
        self.spin_bins = QSpinBox()
        self.spin_bins.setValue(40)
        self.spin_bins.setMinimum(5)
        self.spin_bins.setMaximum(200)
        layout.addWidget(self.spin_bins, row, 1)

        btn_plot = QPushButton("Plot (use boxes values)")
        btn_plot.clicked.connect(self.plot_using_entry_values)
        layout.addWidget(btn_plot, row, 2)

        btn_tests = QPushButton("Run normality tests")
        btn_tests.clicked.connect(self.run_tests_and_report)
        layout.addWidget(btn_tests, row, 3)

        row += 1
        # output console
        layout.addWidget(QLabel("Console output"), row, 0)
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        layout.addWidget(self.output, row, 1, 2, 3)
        row += 2

        # available columns display
        layout.addWidget(QLabel("Available columns (click refresh after load)"), row, 0, 1, 4)
        row += 1
        self.col_list = QTextEdit()
        self.col_list.setReadOnly(True)
        self.col_list.setMaximumHeight(120)
        layout.addWidget(self.col_list, row, 0, 1, 4)

        self.setLayout(layout)

    def open_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV files (*.csv);;All files (*.*)")
        if not path:
            return
        try:
            self.df = pd.read_csv(path, dtype=str)
            self.path = path
            self.lbl_file.setText(path)
            self.columns = list(self.df.columns)
            self.refresh_columns()
            self.col_list.setPlainText("\n".join(self.columns))
            self.output.append(f"Loaded file: {path}  ({len(self.df)} rows, {len(self.columns)} cols)")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def refresh_columns(self):
        self.combo_col.clear()
        if not hasattr(self, "columns") or not self.columns:
            return
        self.combo_col.addItems(self.columns)
        # try fuzzy select "r-g"
        pref = fuzzy_col_match(self.columns, "r-g", cutoff=0.3)
        if pref:
            self.combo_col.setCurrentText(pref)

    def load_selected_column(self):
        if self.df is None:
            QMessageBox.information(self, "No file", "Open a CSV first.")
            return
        col = self.combo_col.currentText()
        if not col:
            QMessageBox.warning(self, "No column", "Select a column.")
            return
        series = pd.to_numeric(self.df[col], errors='coerce').dropna().astype(float).values
        if series.size == 0:
            QMessageBox.warning(self, "No numeric data", f"Column {col} contains no numeric values.")
            return
        self.values = series
        self.output.append(f"Loaded column '{col}' with {len(series)} numeric values.")
        # auto-fill entries with computed stats
        self.fill_entries_from_values(series)

    def fill_entries_from_values(self, values):
        self.edit_min.setText(f"{float(np.nanmin(values)):.6f}")
        self.edit_max.setText(f"{float(np.nanmax(values)):.6f}")
        self.edit_mean.setText(f"{float(np.nanmean(values)):.6f}")
        self.edit_std.setText(f"{float(np.nanstd(values, ddof=1)):.6f}")
        self.spin_n.setValue(int(len(values)))

    def compute_stats_from_loaded_values(self):
        if self.values is None:
            QMessageBox.information(self, "No data", "Load a numeric column first.")
            return
        self.fill_entries_from_values(self.values)
        self.output.append("Entries updated from loaded values.")

    def run_tests_and_report(self):
        # use current loaded values if available; otherwise try to use entries to build a synthetic Gaussian
        if self.values is not None:
            vals = self.values
        else:
            QMessageBox.information(self, "No sample", "Load a column or use entries and press Plot to run tests on synthetic values.")
            return
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = run_normality_tests(vals)
        self.output.append(self.format_test_results(results))

    def format_test_results(self, results):
        lines = ["Normality test results:"]
        sh = results.get('shapiro')
        if sh and not np.isnan(sh[0]):
            lines.append(f" Shapiro-Wilk: stat = {sh[0]:.6g}, p = {sh[1]:.6g}  (recommended for n <= 5000)")
        else:
            lines.append(" Shapiro-Wilk: not run (n > 5000 or error)")

        k2 = results.get('dagostino_k2')
        if k2 and not np.isnan(k2[0]):
            lines.append(f" D'Agostino K^2: stat = {k2[0]:.6g}, p = {k2[1]:.6g}")
        else:
            lines.append(" D'Agostino K^2: not available")

        ad = results.get('anderson')
        if ad is not None:
            lines.append(f" Anderson-Darling: statistic = {ad.statistic:.6g}")
            cvs = ad.critical_values
            sigs = ad.significance_level
            for cv, sl in zip(cvs, sigs):
                note = "reject H0 (not normal)" if ad.statistic > cv else "fail to reject H0 (normal)"
                lines.append(f"  significance {sl}%: critical = {cv:.6g} -> {note}")
        else:
            lines.append(" Anderson-Darling: not available or error")
        return "\n".join(lines)

    def plot_using_entry_values(self):
        # read entries (allow user overrides)
        try:
            min_val = float(self.edit_min.text())
            max_val = float(self.edit_max.text())
            mean = float(self.edit_mean.text())
            std = float(self.edit_std.text())
            n = int(self.spin_n.value())
        except Exception as e:
            QMessageBox.warning(self, "Bad entries", f"Could not parse numeric entries: {e}")
            return

        # if actual sample present, plot empirical histogram + gaussian overlay
        if self.values is not None:
            values = self.values
        else:
            # create synthetic sample from Gaussian with given mean/std (not necessary but useful)
            values = np.random.normal(loc=mean, scale=std, size=n)

        bins = int(self.spin_bins.value())
        # build plot as in earlier script: histogram (density) and gaussian overlay + QQ
        data_mean = mean
        data_std = std
        x_min = min(np.min(values), data_mean - 4*data_std, min_val)
        x_max = max(np.max(values), data_mean + 4*data_std, max_val)
        x = np.linspace(x_min, x_max, 800)
        pdf = norm.pdf(x, loc=data_mean, scale=data_std)

        fig, (ax_hist, ax_qq) = plt.subplots(1, 2, figsize=(12,5), gridspec_kw={'width_ratios':[2,1]})
        ax_hist.hist(values, bins=bins, density=True, alpha=0.6, color='C0', edgecolor='k', linewidth=0.3)
        ax_hist.plot(x, pdf, color='C1', lw=2, label=f'Normal PDF μ={data_mean:.3f}, σ={data_std:.3f}')
        ax_hist.axvline(data_mean, color='k', linestyle='--', lw=1.3, label=f'mean={data_mean:.3f}')
        ax_hist.axvline(min_val, color='C3', linestyle='-', lw=1.2, label=f'min={min_val:.3f}')
        ax_hist.axvline(max_val, color='C4', linestyle='-', lw=1.2, label=f'max={max_val:.3f}')
        ax_hist.set_xlabel(self.combo_col.currentText() or "value"); ax_hist.set_ylabel('Density')
        ax_hist.set_title('Histogram with Normal overlay')
        ax_hist.legend(loc='upper right')

        stderr = data_std / np.sqrt(max(1, n))
        zmin = (min_val - data_mean) / data_std if data_std > 0 else np.nan
        zmax = (max_val - data_mean) / data_std if data_std > 0 else np.nan
        stats_txt = (f"n = {n}\nmin = {min_val:.6f}\nmax = {max_val:.6f}\n"
                     f"mean = {data_mean:.6f}\nstd = {data_std:.6f}\nstderr = {stderr:.6f}\n"
                     f"z(min) = {zmin:.3f}\nz(max) = {zmax:.3f}")
        ax_hist.text(0.02, 0.98, stats_txt, transform=ax_hist.transAxes, va='top', ha='left',
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85), fontsize=9)

        ax_qq.set_title('QQ-plot')
        if _HAVE_SM:
            sm.qqplot(values, line='s', ax=ax_qq)
        else:
            sorted_vals = np.sort(values)
            probs = (np.arange(1, len(sorted_vals)+1) - 0.5) / len(sorted_vals)
            theor_q = norm.ppf(probs, loc=data_mean, scale=data_std)
            ax_qq.scatter(theor_q, sorted_vals, s=6, color='k')
            mn = min(theor_q.min(), sorted_vals.min()); mx = max(theor_q.max(), sorted_vals.max())
            ax_qq.plot([mn,mx], [mn,mx], color='r', linestyle='--')
            ax_qq.set_xlabel('Theoretical quantiles'); ax_qq.set_ylabel('Sample quantiles')

        plt.tight_layout()
        plt.show()

        # run tests on actual loaded values when present
        if self.values is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results = run_normality_tests(self.values)
            self.output.append(self.format_test_results(results))
        else:
            self.output.append("Plotted synthetic Gaussian sample from the entries (no CSV column loaded).")

def main():
    app = QApplication(sys.argv)
    w = NormalityGui()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()