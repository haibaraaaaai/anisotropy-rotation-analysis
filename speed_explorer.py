"""
Speed Explorer GUI — lightweight viewer/reprocessor for Fourkas-processed data.

Loads fit_params.npz (from the Fourkas_processing notebook) + the original TDMS,
lets you interactively explore speed traces and recompute with different parameters.

Usage:
    python speed_explorer.py
"""

import sys
import os
from pathlib import Path

import numpy as np
from nptdms import TdmsFile

from PyQt6 import QtWidgets, QtCore
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


# ═════════════════════════════════════════════════════════════════════════════
# Reprocessing helpers (standalone — no notebook dependency)
# ═════════════════════════════════════════════════════════════════════════════

_ORIENTATIONS = ["90", "45", "135", "0"]

def _get_pol_ind(listpol):
    return [_ORIENTATIONS.index(p) for p in listpol]

def _anisotropy(c0, c90, c45, c135):
    dx = c0 + c90
    dy = c45 + c135
    ax = np.where(dx > 0, (c0 - c90) / dx, 0.0)
    ay = np.where(dy > 0, (c45 - c135) / dy, 0.0)
    return ax, ay

def _compute_speed_windowed(phi_trace, time_s, nperseg, overlap,
                            transform=False, theta_trace=None,
                            theta_axis_deg=0.0, phi_axis_deg=0.0):
    if transform:
        Th = np.radians(theta_axis_deg)
        Ph = np.radians(phi_axis_deg)
        cP, sP = np.cos(Ph), np.sin(Ph)
        cT, sT = np.cos(Th), np.sin(Th)
        R = np.array([[ cT*cP,  cT*sP, -sT],
                       [-sP,     cP,      0 ],
                       [ sT*cP,  sT*sP,  cT ]])
        th_rad = np.radians(theta_trace)
        dx = np.sin(th_rad) * np.cos(phi_trace)
        dy = np.sin(th_rad) * np.sin(phi_trace)
        dz = np.cos(th_rad)
        d_rot = R @ np.vstack([dx, dy, dz])
        phi_ax_raw = np.arctan2(d_rot[1], d_rot[0])
        phi_use = np.unwrap(phi_ax_raw)
    else:
        phi_use = phi_trace

    step = nperseg - overlap
    n_windows = max(0, (len(phi_use) - nperseg) // step + 1)
    speed_hz = np.empty(n_windows)
    speed_time = np.empty(n_windows)

    for w in range(n_windows):
        s = w * step
        e = s + nperseg
        t_win = time_s[s:e]
        p_win = phi_use[s:e]
        t_c = t_win - t_win[0]
        n = len(t_c)
        sum_t = np.sum(t_c)
        sum_p = np.sum(p_win)
        sum_tt = np.sum(t_c * t_c)
        sum_tp = np.sum(t_c * p_win)
        denom_lr = n * sum_tt - sum_t * sum_t
        if abs(denom_lr) > 1e-30:
            slope = (n * sum_tp - sum_t * sum_p) / denom_lr
        else:
            slope = 0.0
        speed_hz[w] = slope / (2.0 * np.pi)
        speed_time[w] = 0.5 * (t_win[0] + t_win[-1])

    return speed_hz, speed_time


# ═════════════════════════════════════════════════════════════════════════════
# Data container for one loaded file
# ═════════════════════════════════════════════════════════════════════════════

class LoadedFile:
    """Holds fit params and (optionally) reprocessed full-trace data."""

    def __init__(self, npz_path):
        self.npz_path = Path(npz_path)
        d = np.load(npz_path, allow_pickle=True)

        # Fit params
        self.Theta_axis = float(d["Theta_axis"])
        self.Phi_axis = float(d["Phi_axis"])
        self.Lambda = float(d["Lambda"])
        self.uncertainties = d["uncertainties"]

        # Backgrounds
        self.b0 = float(d["b0"])
        self.b90 = float(d["b90"])
        self.b45 = float(d["b45"])
        self.b135 = float(d["b135"])

        # Channel correction
        self.a = d["a"]   # gains
        self.o = d["o"]   # offsets
        self.matcor = d["matcor"]

        # Calibration
        self.NA = float(d["NA_calibrated"])
        self.nw = float(d["nw"])
        self.A_cal = float(d["A_cal"])
        self.B_cal = float(d["B_cal"])
        self.C_cal = float(d["C_cal"])
        self.R_sat = float(d["R_sat"])

        # Default speed trace
        self.speed_hz = d["speed_hz"]
        self.speed_time = d["speed_time"]
        self.speed_nperseg = int(d["speed_nperseg"])
        self.speed_overlap = int(d["speed_overlap"])
        self.speed_transform = bool(d["speed_transform"])

        # Data info
        self.freq = float(d["freq"])
        self.tdms_path = str(d["tdms_path"])

        # Full-trace (populated after reprocess)
        self.theta = None
        self.phi = None
        self.time_s = None

        # Name for display
        self.name = Path(self.tdms_path).stem

    def load_tdms_and_reprocess(self, progress_callback=None):
        """Load raw TDMS and recompute theta, phi from saved params."""
        tdms_obj = TdmsFile.open(self.tdms_path)
        group = tdms_obj.groups()[0]
        channels = group.channels()

        idx = _get_pol_ind(["0", "90", "45", "135"])
        c0_raw = np.asarray(channels[idx[0]][:], dtype=float)
        c90_raw = np.asarray(channels[idx[1]][:], dtype=float)
        c45_raw = np.asarray(channels[idx[2]][:], dtype=float)
        c135_raw = np.asarray(channels[idx[3]][:], dtype=float)

        try:
            time_inc = float(channels[0].properties["wf_increment"])
            time_off = float(channels[0].properties["wf_start_offset"])
        except KeyError:
            time_inc = 1.0 / self.freq
            time_off = 0.0

        datasize = len(c0_raw)
        self.time_s = time_off + time_inc * np.arange(datasize)

        if progress_callback:
            progress_callback("Applying corrections...")

        # Apply gain + offset + T-matrix
        raw_stack = np.vstack((c90_raw, c45_raw, c135_raw, c0_raw)).astype(float)
        for i in range(4):
            raw_stack[i] = self.o[i] + self.a[i] * raw_stack[i]
        tcor_stack = np.dot(self.matcor, raw_stack)
        del raw_stack

        _idx_fit = _get_pol_ind(["0", "90", "45", "135"])
        c0 = tcor_stack[_idx_fit[0]] - self.b0
        c90 = tcor_stack[_idx_fit[1]] - self.b90
        c45 = tcor_stack[_idx_fit[2]] - self.b45
        c135 = tcor_stack[_idx_fit[3]] - self.b135
        del tcor_stack

        if progress_callback:
            progress_callback("Computing angles...")

        ax, ay = _anisotropy(c0, c90, c45, c135)
        del c0, c90, c45, c135

        phi_raw = 0.5 * np.arctan2(ay, ax)
        self.phi = np.unwrap(phi_raw, period=np.pi)

        r_full = np.clip(np.sqrt(ax**2 + ay**2), 0.0, self.R_sat)
        denom = np.where(
            self.C_cal - r_full * self.B_cal > 0,
            self.C_cal - r_full * self.B_cal,
            np.nan,
        )
        sinsq = np.clip((r_full * self.A_cal) / denom, 0.0, 1.0)
        self.theta = np.degrees(np.arcsin(np.sqrt(sinsq)))

        if progress_callback:
            progress_callback("Done.")

    def compute_speed(self, nperseg, overlap, transform):
        """(Re)compute speed from the already-loaded phi/theta."""
        if self.phi is None:
            return
        self.speed_hz, self.speed_time = _compute_speed_windowed(
            self.phi, self.time_s, nperseg, overlap,
            transform=transform,
            theta_trace=self.theta,
            theta_axis_deg=self.Theta_axis,
            phi_axis_deg=self.Phi_axis,
        )
        self.speed_nperseg = nperseg
        self.speed_overlap = overlap
        self.speed_transform = transform


# ═════════════════════════════════════════════════════════════════════════════
# Worker thread for TDMS loading
# ═════════════════════════════════════════════════════════════════════════════

class ReprocessWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, loaded_file):
        super().__init__()
        self.lf = loaded_file

    def run(self):
        self.lf.load_tdms_and_reprocess(progress_callback=self.progress.emit)
        self.finished.emit()


# ═════════════════════════════════════════════════════════════════════════════
# Main GUI
# ═════════════════════════════════════════════════════════════════════════════

class SpeedExplorer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Speed Explorer")
        self.resize(1400, 900)

        self.files = {}          # name -> LoadedFile
        self._workers = []       # keep references to prevent GC

        self._build_ui()

    # ── UI construction ──────────────────────────────────────────────────
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)

        # ── Left panel: file list + params ───────────────────────────────
        left = QtWidgets.QVBoxLayout()

        # File list
        self.file_list = QtWidgets.QListWidget()
        self.file_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.file_list.itemSelectionChanged.connect(self._on_selection_changed)
        left.addWidget(QtWidgets.QLabel("Loaded files:"))
        left.addWidget(self.file_list)

        # Buttons
        btn_row = QtWidgets.QHBoxLayout()
        btn_load = QtWidgets.QPushButton("Load .npz")
        btn_load.clicked.connect(self._load_npz)
        btn_load_folder = QtWidgets.QPushButton("Load folder")
        btn_load_folder.clicked.connect(self._load_folder)
        btn_row.addWidget(btn_load)
        btn_row.addWidget(btn_load_folder)
        left.addLayout(btn_row)

        # Parameter info
        self.param_label = QtWidgets.QLabel("")
        self.param_label.setWordWrap(True)
        self.param_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        left.addWidget(self.param_label)

        left_widget = QtWidgets.QWidget()
        left_widget.setLayout(left)
        left_widget.setMaximumWidth(350)
        main_layout.addWidget(left_widget)

        # ── Right panel: plots + controls ────────────────────────────────
        right = QtWidgets.QVBoxLayout()

        # Speed controls
        ctrl = QtWidgets.QHBoxLayout()

        ctrl.addWidget(QtWidgets.QLabel("Window:"))
        self.spin_nperseg = QtWidgets.QSpinBox()
        self.spin_nperseg.setRange(100, 200000)
        self.spin_nperseg.setValue(8000)
        self.spin_nperseg.setSingleStep(1000)
        ctrl.addWidget(self.spin_nperseg)

        ctrl.addWidget(QtWidgets.QLabel("Overlap:"))
        self.spin_overlap = QtWidgets.QSpinBox()
        self.spin_overlap.setRange(0, 199000)
        self.spin_overlap.setValue(4000)
        self.spin_overlap.setSingleStep(500)
        ctrl.addWidget(self.spin_overlap)

        self.chk_transform = QtWidgets.QCheckBox("Axis-frame")
        self.chk_transform.setChecked(True)
        ctrl.addWidget(self.chk_transform)

        btn_recompute = QtWidgets.QPushButton("Recompute speed")
        btn_recompute.clicked.connect(self._recompute_speed)
        ctrl.addWidget(btn_recompute)

        btn_load_tdms = QtWidgets.QPushButton("Load TDMS (for reprocessing)")
        btn_load_tdms.clicked.connect(self._load_tdms_for_selected)
        ctrl.addWidget(btn_load_tdms)

        ctrl.addStretch()
        right.addLayout(ctrl)

        # Matplotlib figure with 4 subplots
        self.fig = Figure(figsize=(12, 10), dpi=100)
        self.fig.set_tight_layout(True)
        self.ax_speed = self.fig.add_subplot(4, 1, 1)
        self.ax_theta = self.fig.add_subplot(4, 1, 2, sharex=self.ax_speed)
        self.ax_phi = self.fig.add_subplot(4, 1, 3, sharex=self.ax_speed)
        self.ax_hist = self.fig.add_subplot(4, 1, 4)

        self.ax_speed.set_ylabel("Speed (Hz)")
        self.ax_speed.set_title("Speed")
        self.ax_speed.grid(alpha=0.3)

        self.ax_theta.set_ylabel("θ (deg)")
        self.ax_theta.set_title("θ")
        self.ax_theta.grid(alpha=0.3)

        self.ax_phi.set_ylabel("φ (rad)")
        self.ax_phi.set_xlabel("Time (s)")
        self.ax_phi.set_title("φ")
        self.ax_phi.grid(alpha=0.3)

        self.ax_hist.set_xlabel("Speed (Hz)")
        self.ax_hist.set_ylabel("Count")
        self.ax_hist.set_title("Speed histogram")
        self.ax_hist.grid(alpha=0.3)

        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        right.addWidget(self.toolbar)
        right.addWidget(self.canvas, stretch=1)

        main_layout.addLayout(right, stretch=1)

        # Status bar
        self.statusBar().showMessage("Ready — load a fit_params.npz to begin")

    # ── File loading ─────────────────────────────────────────────────────
    def _load_npz(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Open fit_params.npz", "", "NumPy files (*.npz)")
        for p in paths:
            self._add_file(p)

    def _load_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select folder with fit_params.npz files")
        if not folder:
            return
        for root, _dirs, files in os.walk(folder):
            for fn in files:
                if fn == "fit_params.npz":
                    self._add_file(os.path.join(root, fn))

    def _add_file(self, npz_path):
        try:
            lf = LoadedFile(npz_path)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Load error", f"Failed to load {npz_path}:\n{e}")
            return
        display_name = lf.name
        if display_name in self.files:
            display_name = f"{display_name} ({Path(npz_path).parent.name})"
        self.files[display_name] = lf
        self.file_list.addItem(display_name)
        self.statusBar().showMessage(f"Loaded {display_name}")
        if self.file_list.count() == 1:
            self.file_list.setCurrentRow(0)

    # ── Selection & display ──────────────────────────────────────────────
    def _on_selection_changed(self):
        self._update_plots()
        self._update_params()

    def _selected_files(self):
        return [self.files[item.text()] for item in self.file_list.selectedItems()]

    def _update_params(self):
        sel = self._selected_files()
        if not sel:
            self.param_label.setText("")
            return
        lines = []
        for lf in sel:
            lines.append(f"<b>{lf.name}</b>")
            lines.append(f"  Θ_axis = {lf.Theta_axis:.2f}° ± {lf.uncertainties[0]:.2f}°")
            lines.append(f"  Φ_axis = {lf.Phi_axis:.2f}° ± {lf.uncertainties[1]:.2f}°")
            lines.append(f"  Λ = {lf.Lambda:.2f}° ± {lf.uncertainties[2]:.2f}°")
            lines.append(f"  Freq = {lf.freq:.0f} Hz | NA = {lf.NA:.4f}")
            lines.append(f"  TDMS loaded: {'Yes' if lf.phi is not None else 'No'}")
            lines.append("")
        self.param_label.setText("<br>".join(lines))

    def _update_plots(self):
        sel = self._selected_files()
        for ax in [self.ax_speed, self.ax_theta, self.ax_phi, self.ax_hist]:
            ax.cla()

        self.ax_speed.set_ylabel("Speed (Hz)")
        self.ax_speed.set_title("Speed")
        self.ax_speed.grid(alpha=0.3)
        self.ax_theta.set_ylabel("θ (deg)")
        self.ax_theta.set_title("θ")
        self.ax_theta.grid(alpha=0.3)
        self.ax_phi.set_ylabel("φ (rad)")
        self.ax_phi.set_xlabel("Time (s)")
        self.ax_phi.set_title("φ")
        self.ax_phi.grid(alpha=0.3)
        self.ax_hist.set_xlabel("Speed (Hz)")
        self.ax_hist.set_ylabel("Count")
        self.ax_hist.set_title("Speed histogram")
        self.ax_hist.grid(alpha=0.3)

        all_speeds = []
        for i, lf in enumerate(sel):
            # Speed (always available from npz)
            self.ax_speed.plot(lf.speed_time, lf.speed_hz, lw=0.5, label=lf.name, alpha=0.8)
            all_speeds.append(lf.speed_hz)

            # θ and φ (only if TDMS loaded — decimate for display)
            if lf.theta is not None:
                dec = max(1, len(lf.theta) // 50000)
                self.ax_theta.plot(lf.time_s[::dec], lf.theta[::dec], lw=0.3, alpha=0.7)
                self.ax_phi.plot(lf.time_s[::dec], lf.phi[::dec], lw=0.3, alpha=0.7)

        if len(sel) > 1:
            self.ax_speed.legend(fontsize=7)

        # Histogram of all selected speeds
        if all_speeds:
            combined = np.concatenate(all_speeds)
            combined = combined[np.isfinite(combined)]
            if len(combined) > 0:
                lo, hi = np.percentile(combined, [1, 99])
                self.ax_hist.hist(combined, bins=200, range=(lo, hi),
                                  alpha=0.7, edgecolor='none')

        self.canvas.draw_idle()

    # ── TDMS loading / reprocessing ──────────────────────────────────────
    def _load_tdms_for_selected(self):
        sel = self._selected_files()
        if not sel:
            QtWidgets.QMessageBox.information(self, "No selection", "Select a file first.")
            return
        for lf in sel:
            if lf.phi is not None:
                continue  # already loaded
            if not Path(lf.tdms_path).exists():
                path, _ = QtWidgets.QFileDialog.getOpenFileName(
                    self, f"Locate TDMS for {lf.name}", "", "TDMS files (*.tdms)")
                if not path:
                    continue
                lf.tdms_path = path

            self.statusBar().showMessage(f"Loading TDMS for {lf.name}...")
            worker = ReprocessWorker(lf)
            worker.progress.connect(lambda msg, name=lf.name: self.statusBar().showMessage(f"{name}: {msg}"))
            worker.finished.connect(self._on_reprocess_done)
            self._workers.append(worker)
            worker.start()

    def _on_reprocess_done(self):
        self.statusBar().showMessage("TDMS loaded — you can now recompute speed")
        self._update_plots()
        self._update_params()

    def _recompute_speed(self):
        sel = self._selected_files()
        if not sel:
            return
        nperseg = self.spin_nperseg.value()
        overlap = self.spin_overlap.value()
        transform = self.chk_transform.isChecked()

        if overlap >= nperseg:
            QtWidgets.QMessageBox.warning(self, "Invalid", "Overlap must be less than window size.")
            return

        n_recomputed = 0
        for lf in sel:
            if lf.phi is None:
                continue
            lf.compute_speed(nperseg, overlap, transform)
            n_recomputed += 1

        if n_recomputed == 0:
            QtWidgets.QMessageBox.information(
                self, "No TDMS loaded",
                "Load the TDMS first (button above) to enable recomputation.")
            return

        self.statusBar().showMessage(
            f"Recomputed speed for {n_recomputed} file(s): "
            f"window={nperseg}, overlap={overlap}, "
            f"{'axis-frame' if transform else 'lab-frame'}")
        self._update_plots()


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    window = SpeedExplorer()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
