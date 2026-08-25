"""Main window — raw data viewer with anisotropy side panel."""

import json
import numpy as np
from pathlib import Path
from datetime import datetime

from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QFormLayout,
    QGroupBox, QPushButton, QLineEdit, QDoubleSpinBox, QSpinBox,
    QFileDialog, QLabel, QMessageBox, QApplication, QSplitter,
    QTabWidget,
)
from PyQt6.QtGui import QShortcut, QKeySequence

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from ..core.io import load_tdms, anisotropy_from_channels
from ..core.cycle import detect_cycle_anisotropy, subsample_ref_curve
from ..core.correction import (
    apply_full_trace_correction, extract_theta_phi,
    fourkas_ABC, build_matcor, compute_speed_windowed,
)
from .cycle_tab import CycleTab
from .correction_tab import CorrectionTab
from .trace_tab import TraceTab
from .speed_tab import SpeedTab



# ── Matplotlib canvas widget ─────────────────────────────────────────────────

class MplCanvas(FigureCanvas):
    def __init__(self, fig):
        super().__init__(fig)


# ── Main window ──────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Anisotropy Rotation Processing")
        self.resize(1500, 700)

        # Data state
        self._raw = None       # dict from load_tdms
        self._cycle = None     # (cyc_start, cyc_end) in decimated indices

        self._settings = QSettings("AnisotropyRotationGUI", "AnisotropyRotationProcessing")

        self._build_ui()
        self._connect_signals()
        self._define_shortcuts()

    # ── UI ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)

        self.tabs = QTabWidget()
        outer.addWidget(self.tabs)

        # ── Tab 0: Raw View ───────────────────────────────────────────────
        raw_tab = QWidget()
        raw_layout = QHBoxLayout(raw_tab)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        raw_layout.addWidget(splitter)
        self.tabs.addTab(raw_tab, "Raw View")

        # ── Left: controls ────────────────────────────────────────────────
        ctrl = QWidget()
        ctrl_layout = QVBoxLayout(ctrl)
        ctrl_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # File
        file_grp = QGroupBox("File")
        file_lay = QVBoxLayout()
        row = QHBoxLayout()
        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText("Select a .tdms file…")
        self.file_edit.setReadOnly(True)
        row.addWidget(self.file_edit)
        self.browse_btn = QPushButton("Browse…")
        row.addWidget(self.browse_btn)
        file_lay.addLayout(row)
        self.file_info = QLabel("")
        file_lay.addWidget(self.file_info)
        file_grp.setLayout(file_lay)
        ctrl_layout.addWidget(file_grp)

        # Window
        win_grp = QGroupBox("View Window")
        win_form = QFormLayout()
        self.win_start = QDoubleSpinBox()
        self.win_start.setRange(0, 1e7)
        self.win_start.setDecimals(3)
        self.win_start.setSuffix(" s")
        self.win_start.setValue(0)
        win_form.addRow("Start:", self.win_start)

        self.win_end = QDoubleSpinBox()
        self.win_end.setRange(0, 1e7)
        self.win_end.setDecimals(3)
        self.win_end.setSuffix(" s")
        self.win_end.setValue(1.0)
        win_form.addRow("End:", self.win_end)

        self.dec_view = QSpinBox()
        self.dec_view.setRange(1, 10000)
        self.dec_view.setValue(100)
        win_form.addRow("Decimation:", self.dec_view)

        self.apply_win_btn = QPushButton("Apply Window")
        self.apply_win_btn.setEnabled(False)
        win_form.addRow(self.apply_win_btn)

        win_grp.setLayout(win_form)
        ctrl_layout.addWidget(win_grp)

        nav_lbl = QLabel("Arrow keys: ←→ pan, ↑↓ resize window")
        nav_lbl.setWordWrap(True)
        nav_lbl.setStyleSheet("color: grey; font-size: 11px;")
        ctrl_layout.addWidget(nav_lbl)

        # Cycle detection
        cyc_grp = QGroupBox("Cycle Detection")
        cyc_lay = QVBoxLayout()
        self.detect_btn = QPushButton("Detect Cycle")
        self.detect_btn.setEnabled(False)
        cyc_lay.addWidget(self.detect_btn)
        self.cycle_info = QLabel("")
        self.cycle_info.setWordWrap(True)
        cyc_lay.addWidget(self.cycle_info)
        cyc_grp.setLayout(cyc_lay)
        ctrl_layout.addWidget(cyc_grp)

        # Status
        self.status = QLabel("Ready")
        self.status.setWordWrap(True)
        ctrl_layout.addWidget(self.status)
        ctrl_layout.addStretch()

        ctrl.setMinimumWidth(250)
        ctrl.setMaximumWidth(320)
        splitter.addWidget(ctrl)

        # ── Centre: raw channel traces ────────────────────────────────────
        centre = QWidget()
        centre_layout = QVBoxLayout(centre)
        centre_layout.setContentsMargins(0, 0, 0, 0)

        self.trace_fig = Figure(figsize=(10, 6), dpi=100)
        self.trace_canvas = MplCanvas(self.trace_fig)
        self.trace_toolbar = NavigationToolbar(self.trace_canvas, self)
        centre_layout.addWidget(self.trace_toolbar)
        centre_layout.addWidget(self.trace_canvas)
        splitter.addWidget(centre)

        # ── Right: anisotropy scatter ─────────────────────────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.anis_fig = Figure(figsize=(5, 5), dpi=100)
        self.anis_canvas = MplCanvas(self.anis_fig)
        self.anis_toolbar = NavigationToolbar(self.anis_canvas, self)
        right_layout.addWidget(self.anis_toolbar)
        right_layout.addWidget(self.anis_canvas)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 0)  # controls — fixed
        splitter.setStretchFactor(1, 3)  # traces — stretch
        splitter.setStretchFactor(2, 1)  # anisotropy — moderate

    # ── Signals ───────────────────────────────────────────────────────────

    def _connect_signals(self):
        self.browse_btn.clicked.connect(self._browse)
        self.apply_win_btn.clicked.connect(self._update_plots)
        self.detect_btn.clicked.connect(self._detect_cycle)
        self.win_start.editingFinished.connect(self._update_plots)
        self.win_end.editingFinished.connect(self._update_plots)
        self.dec_view.editingFinished.connect(self._update_plots)

    # ── File loading ──────────────────────────────────────────────────────

    def _browse(self):
        last_dir = self._settings.value("last_dir", "")
        path, _ = QFileDialog.getOpenFileName(
            self, "Select TDMS file", last_dir,
            "TDMS files (*.tdms);;All files (*)")
        if not path:
            return
        self.file_edit.setText(path)
        self._settings.setValue("last_dir", str(Path(path).parent))
        self.status.setText(f"Loading {Path(path).name}…")
        QApplication.processEvents()

        try:
            self._raw = load_tdms(path)
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))
            self.status.setText(f"Error: {e}")
            return

        r = self._raw
        dur = r["time_s"][-1] - r["time_s"][0]
        self.file_info.setText(
            f"{r['datasize']:,} pts | {r['freq']:.0f} Hz | {dur:.1f} s")

        # Set sensible window defaults
        self.win_start.setRange(r["time_s"][0], r["time_s"][-1])
        self.win_end.setRange(r["time_s"][0], r["time_s"][-1])
        self.win_start.setValue(r["time_s"][0])
        self.win_end.setValue(min(r["time_s"][0] + 1.0, r["time_s"][-1]))

        self.apply_win_btn.setEnabled(True)
        self.detect_btn.setEnabled(True)
        self._cycle = None
        self.cycle_info.setText("")

        # Close all tabs except the Raw tab (index 0)
        while self.tabs.count() > 1:
            self.tabs.removeTab(self.tabs.count() - 1)

        self.status.setText("File loaded.")

        self._update_plots()

    # ── Plotting ──────────────────────────────────────────────────────────

    def _get_window_slice(self):
        """Return (index_start, index_end, decimation) for current window."""
        r = self._raw
        t0 = self.win_start.value()
        t1 = self.win_end.value()
        dec = self.dec_view.value()
        i0 = max(0, int((t0 - r["time_off"]) / r["time_inc"]))
        i1 = min(r["datasize"], int((t1 - r["time_off"]) / r["time_inc"]))
        i1 = max(i0 + 1, i1)
        return i0, i1, dec

    def _update_plots(self):
        if self._raw is None:
            return
        r = self._raw
        i0, i1, dec = self._get_window_slice()

        t = r["time_s"][i0:i1:dec]
        c0 = r["c0_raw"][i0:i1:dec]
        c90 = r["c90_raw"][i0:i1:dec]
        c45 = r["c45_raw"][i0:i1:dec]
        c135 = r["c135_raw"][i0:i1:dec]

        # ── Raw channel traces ────────────────────────────────────────────
        self.trace_fig.clear()
        axes = self.trace_fig.subplots(4, 1, sharex=True)
        channels = [(c0, "c0 (0°)", "tab:blue"),
                     (c90, "c90 (90°)", "tab:orange"),
                     (c45, "c45 (45°)", "tab:green"),
                     (c135, "c135 (135°)", "tab:red")]
        for ax, (data, label, color) in zip(axes, channels):
            ax.plot(t, data, lw=0.5, color=color)
            ax.set_ylabel(label, fontsize=9)
            ax.grid(alpha=0.25)
            ax.tick_params(labelsize=8)
        axes[-1].set_xlabel("Time (s)")
        axes[0].set_title(
            f"{Path(self.file_edit.text()).name}  "
            f"[{self.win_start.value():.3f} – {self.win_end.value():.3f} s]  "
            f"dec={dec}", fontsize=10)

        # Overlay cycle region if detected
        if self._cycle is not None:
            cs, ce = self._cycle  # indices in decimated window coords
            # Convert to time
            t_cs = r["time_s"][i0 + cs * dec] if i0 + cs * dec < r["datasize"] else t[0]
            t_ce = r["time_s"][min(i0 + ce * dec, r["datasize"] - 1)]
            for ax in axes:
                ax.axvspan(t_cs, t_ce, alpha=0.15, color="tab:purple",
                           label="cycle")

        self.trace_fig.tight_layout()
        self.trace_canvas.draw_idle()

        # ── Anisotropy scatter ────────────────────────────────────────────
        ax_a, ay_a = anisotropy_from_channels(c0, c90, c45, c135)

        self.anis_fig.clear()
        ax = self.anis_fig.add_subplot(111)
        ax.scatter(ax_a, ay_a, s=1, alpha=0.3, c="tab:blue", rasterized=True)

        # Overlay cycle points if detected
        if self._cycle is not None:
            cs, ce = self._cycle
            ax_cyc = ax_a[cs:ce]
            ay_cyc = ay_a[cs:ce]
            ax.scatter(ax_cyc, ay_cyc, s=2, alpha=0.5, c="tab:purple",
                       rasterized=True, label="cycle")
            ax.legend(fontsize=8)

        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("anis_x", fontsize=9)
        ax.set_ylabel("anis_y", fontsize=9)
        ax.set_title("Anisotropy (raw)", fontsize=10)
        ax.grid(alpha=0.25)
        self.anis_fig.tight_layout()
        self.anis_canvas.draw_idle()

        n_pts = len(t)
        self.status.setText(
            f"Showing {n_pts:,} pts  (raw: {i1 - i0:,}, dec {dec})")

    # ── Cycle detection ───────────────────────────────────────────────────

    def _detect_cycle(self):
        if self._raw is None:
            return
        r = self._raw
        i0, i1, dec = self._get_window_slice()

        c0 = r["c0_raw"][i0:i1:dec]
        c90 = r["c90_raw"][i0:i1:dec]
        c45 = r["c45_raw"][i0:i1:dec]
        c135 = r["c135_raw"][i0:i1:dec]

        ax_a, ay_a = anisotropy_from_channels(c0, c90, c45, c135)

        self.status.setText("Detecting cycle…")
        QApplication.processEvents()

        try:
            cyc_start, cyc_end = detect_cycle_anisotropy(ax_a, ay_a)
        except Exception as e:
            QMessageBox.critical(self, "Cycle detection error", str(e))
            self.status.setText(f"Error: {e}")
            return

        self._cycle = (cyc_start, cyc_end)
        n_cyc = cyc_end - cyc_start

        # Convert to time for display
        t_start = r["time_s"][i0 + cyc_start * dec]
        t_end = r["time_s"][min(i0 + cyc_end * dec, r["datasize"] - 1)]
        self.cycle_info.setText(
            f"Cycle: [{cyc_start}, {cyc_end}]\n"
            f"{n_cyc} pts ({t_start:.4f} – {t_end:.4f} s)")
        self.status.setText("Cycle detected.")
        self._update_plots()

        # Open cycle tab with full-resolution anisotropy from current window
        self._open_cycle_tab(i0, i1, cyc_start, cyc_end, dec)

    def _open_cycle_tab(self, i0, i1, cyc_start, cyc_end, dec):
        """Create (or replace) the Cycle tab with full-resolution data."""
        r = self._raw
        # Store window offset for later raw-channel lookup
        self._cycle_window_i0 = i0
        # Full-resolution channels from the viewed window
        c0 = r["c0_raw"][i0:i1]
        c90 = r["c90_raw"][i0:i1]
        c45 = r["c45_raw"][i0:i1]
        c135 = r["c135_raw"][i0:i1]
        ax_full, ay_full = anisotropy_from_channels(c0, c90, c45, c135)

        # Convert decimated cycle indices to full-resolution indices
        fs = cyc_start * dec
        fe = cyc_end * dec

        # Remove old cycle tab if present
        for idx in range(self.tabs.count() - 1, 0, -1):
            if self.tabs.tabText(idx) == "Cycle":
                self.tabs.removeTab(idx)

        tab = CycleTab(ax_full, ay_full, fs, fe, parent=self)
        tab.confirmed.connect(self._cycle_confirmed)
        self.tabs.addTab(tab, "Cycle")
        self.tabs.setCurrentWidget(tab)

    def _cycle_confirmed(self, start, end, ref_result):
        """Called when user locks in cycle selection."""
        self._confirmed_cycle = {
            "start": start,
            "end": end,
            "ref": ref_result,
        }
        self.status.setText(
            f"Cycle confirmed: [{start}, {end}]  "
            f"({end - start} pts)")
        self._open_correction_tab(start, end, ref_result)

    def _open_correction_tab(self, start, end, ref_result):
        """Open the Correction & Fitting tab with confirmed cycle data."""
        r = self._raw
        i0 = getattr(self, '_cycle_window_i0', 0)

        # Raw channels for the confirmed cycle range
        raw_start = i0 + start
        raw_end = i0 + end
        c0_cyc = r["c0_raw"][raw_start:raw_end]
        c90_cyc = r["c90_raw"][raw_start:raw_end]
        c45_cyc = r["c45_raw"][raw_start:raw_end]
        c135_cyc = r["c135_raw"][raw_start:raw_end]

        matched_idx = ref_result[0]
        sx_ref = ref_result[2]
        sy_ref = ref_result[3]

        # Remove old correction tab if present
        for idx in range(self.tabs.count() - 1, 0, -1):
            if self.tabs.tabText(idx) == "Correction":
                self.tabs.removeTab(idx)

        tab = CorrectionTab(
            c0_cyc, c90_cyc, c45_cyc, c135_cyc,
            matched_idx, sx_ref, sy_ref,
            parent=self)
        tab.confirmed.connect(self._correction_confirmed)
        self.tabs.addTab(tab, "Correction")
        self.tabs.setCurrentWidget(tab)

    def _correction_confirmed(self, result):
        """Called when user confirms fit values in the Correction tab."""
        self._confirmed_fit = result
        self.status.setText("Computing full-trace angles & speed…")
        QApplication.processEvents()
        self._open_result_tabs(result)

    def _open_result_tabs(self, result):
        """Open angle / speed viewer tabs after fit confirmation."""
        r = self._raw
        rotation_frame = result.get("rotation_frame", False)

        # Full-trace correction
        a = np.array(result["a"])
        o = np.array(result["o"])
        matcor, _ = build_matcor()
        backgrounds = {
            "b0": result["b0"], "b90": result["b90"],
            "b45": result["b45"], "b135": result["b135"],
        }
        c0, c90, c45, c135 = apply_full_trace_correction(
            r["c0_raw"], r["c90_raw"], r["c45_raw"], r["c135_raw"],
            a, o, matcor, backgrounds)

        from ..core.io import anisotropy_from_channels as _anis
        ax, ay = _anis(c0, c90, c45, c135)

        A = result.get("A")
        B = result.get("B")
        C = result.get("C")
        if A is None or B is None or C is None:
            NA = result["NA"]
            n = result["n"]
            alpha = np.arcsin(NA / n)
            A, B, C = fourkas_ABC(alpha)
        theta, phi = extract_theta_phi(ax, ay, A, B, C)

        time_s = r["time_s"]
        freq = r["freq"]

        if rotation_frame:
            # 2 tabs: rotation-frame angle + speed
            _, _, phi_rot = compute_speed_windowed(
                phi, time_s, 2000, 1000,
                transform=True, theta_trace=theta,
                theta_axis_deg=result["Theta"],
                phi_axis_deg=result["Phi"])

            tab_angle = TraceTab(time_s, phi_rot,
                                "φ' (rad)", "Rotation angle (axis frame)",
                                parent=self)
            self.tabs.addTab(tab_angle, "φ' (rot)")

            tab_speed = SpeedTab(
                time_s, phi_rot, freq,
                default_nperseg=2000, default_overlap=1000,
                transform=False,  # phi_rot is already in axis frame
                parent=self)
            self.tabs.addTab(tab_speed, "Speed (rot)")
            self.tabs.setCurrentWidget(tab_angle)
        else:
            # 3 tabs: phi, theta, speed
            tab_phi = TraceTab(time_s, phi, "φ (rad)", "φ (lab frame)",
                               parent=self)
            self.tabs.addTab(tab_phi, "φ (lab)")

            tab_theta = TraceTab(time_s, theta, "θ (°)", "θ (lab frame)",
                                 parent=self)
            self.tabs.addTab(tab_theta, "θ (lab)")

            tab_speed = SpeedTab(
                time_s, phi, freq,
                default_nperseg=2000, default_overlap=1000,
                transform=False,
                theta_trace=theta,
                theta_axis_deg=result["Theta"],
                phi_axis_deg=result["Phi"],
                parent=self)
            self.tabs.addTab(tab_speed, "Speed")
            self.tabs.setCurrentWidget(tab_phi)

        # Export fit metadata / params for downstream reuse.
        try:
            self._export_fit_outputs(result, time_s, theta, phi, freq)
        except Exception as exc:
            self.status.setText(
                f"Fit confirmed but export failed: {exc}")
            return

        self.status.setText(
            f"Fit confirmed: Θ={result['Theta']:.2f}° "
            f"Φ={result['Phi']:.2f}° Λ={result['Lambda']:.2f}°  — "
            f"{'rotation-frame' if rotation_frame else 'lab-frame'} tabs open")

    def _build_output_dir(self, tdms_path):
        """Create output directory mirroring notebook convention.

        Replaces the first `data` segment with `data_analysed` and strips
        the file suffix from the tail path.
        """
        tdms_path = Path(tdms_path)
        parts = tdms_path.parts
        try:
            data_idx = list(parts).index("data")
        except ValueError:
            data_idx = max(0, len(parts) - 2)
        out_dir = (Path(*parts[:data_idx]) / "data_analysed"
                   / Path(*parts[data_idx + 1:]).with_suffix(""))
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir, data_idx

    def _export_fit_outputs(self, result, time_s, theta, phi, freq):
        """Export fit metadata and compact fit params for GUI reuse."""
        tdms_path = Path(self.file_edit.text())
        out_dir, data_idx = self._build_output_dir(tdms_path)

        calibration_mode = result.get("calibration_mode", "legacy")
        na_in = result.get("NA_in", None)
        na_out = result.get("NA_out", None)

        # Build one default speed trace exactly like GUI defaults.
        speed_nperseg = 2000
        speed_overlap = 1000
        speed_transform = bool(result.get("rotation_frame", False))
        speed_hz, speed_time, _ = compute_speed_windowed(
            phi,
            time_s,
            speed_nperseg,
            speed_overlap,
            transform=speed_transform,
            theta_trace=theta,
            theta_axis_deg=result["Theta"],
            phi_axis_deg=result["Phi"],
        )

        metadata = {
            "filename": str(Path(*tdms_path.parts[data_idx + 1:])),
            "fitted_axis_angle_deg": {
                "Theta_axis": float(result["Theta"]),
                "Phi_axis": float(result["Phi"]),
            },
            "fitted_half_cone_angle_deg": {
                "Lambda": float(result["Lambda"]),
            },
            "fitted_background": {
                "dc": float(result["dc"]),
                "fa": float(result["fa"]),
                "fb": float(result["fb"]),
                "b0": float(result["b0"]),
                "b90": float(result["b90"]),
                "b45": float(result["b45"]),
                "b135": float(result["b135"]),
            },
            "fitted_gain": {
                "a_90": float(result["a"][0]),
                "a_45": float(result["a"][1]),
                "a_135": float(result["a"][2]),
                "a_0": float(result["a"][3]),
            },
            "fitted_offset": {
                "o_90": float(result["o"][0]),
                "o_45": float(result["o"][1]),
                "o_135": float(result["o"][2]),
                "o_0": float(result["o"][3]),
            },
            "optics": {
                "calibration_mode": calibration_mode,
                "NA": float(result.get("NA", np.nan)),
                "n": float(result.get("n", np.nan)),
                "NA_in": (None if na_in is None else float(na_in)),
                "NA_out": (None if na_out is None else float(na_out)),
                "A": float(result["A"]),
                "B": float(result["B"]),
                "C": float(result["C"]),
                "R_sat": float(result["R_sat"]),
            },
            "speed_calculation": {
                "nperseg": speed_nperseg,
                "overlap": speed_overlap,
                "transform": speed_transform,
            },
            "analysis_timestamp": datetime.now().isoformat(),
        }

        json_path = out_dir / "metadata.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        npz_path = out_dir / "fit_params.npz"
        np.savez(
            npz_path,
            Theta_axis=float(result["Theta"]),
            Phi_axis=float(result["Phi"]),
            Lambda=float(result["Lambda"]),
            dc=float(result["dc"]),
            fa=float(result["fa"]),
            fb=float(result["fb"]),
            b0=float(result["b0"]),
            b90=float(result["b90"]),
            b45=float(result["b45"]),
            b135=float(result["b135"]),
            a=np.array(result["a"], dtype=float),
            o=np.array(result["o"], dtype=float),
            A=float(result["A"]),
            B=float(result["B"]),
            C=float(result["C"]),
            R_sat=float(result["R_sat"]),
            calibration_mode=np.array(calibration_mode),
            NA=float(result.get("NA", np.nan)),
            n=float(result.get("n", np.nan)),
            NA_in=(np.nan if na_in is None else float(na_in)),
            NA_out=(np.nan if na_out is None else float(na_out)),
            speed_hz=speed_hz,
            speed_time=speed_time,
            speed_nperseg=int(speed_nperseg),
            speed_overlap=int(speed_overlap),
            speed_transform=bool(speed_transform),
            freq=float(freq),
            tdms_path=str(tdms_path),
        )

    # ── Keyboard shortcuts (QShortcut — works regardless of focus) ─────

    def _define_shortcuts(self):
        QShortcut(QKeySequence("Right"), self).activated.connect(self._on_right)
        QShortcut(QKeySequence("Left"), self).activated.connect(self._on_left)
        QShortcut(QKeySequence("Up"), self).activated.connect(self._on_up)
        QShortcut(QKeySequence("Down"), self).activated.connect(self._on_down)

    def _active_cycle_tab(self):
        w = self.tabs.currentWidget()
        if isinstance(w, CycleTab):
            return w
        return None

    def _active_trace_tab(self):
        w = self.tabs.currentWidget()
        if isinstance(w, TraceTab):
            return w
        return None

    def _on_right(self):
        ct = self._active_cycle_tab()
        if ct:
            ct._nudge_start(+1)
            return
        tt = self._active_trace_tab()
        if tt:
            tt.nav_pan(+1)
            return
        self._pan_right()

    def _on_left(self):
        ct = self._active_cycle_tab()
        if ct:
            ct._nudge_start(-1)
            return
        tt = self._active_trace_tab()
        if tt:
            tt.nav_pan(-1)
            return
        self._pan_left()

    def _on_up(self):
        ct = self._active_cycle_tab()
        if ct:
            ct._nudge_end(+1)
            return
        tt = self._active_trace_tab()
        if tt:
            tt.nav_zoom(1.5)
            return
        self._zoom_out()

    def _on_down(self):
        ct = self._active_cycle_tab()
        if ct:
            ct._nudge_end(-1)
            return
        tt = self._active_trace_tab()
        if tt:
            tt.nav_zoom(0.667)
            return
        self._zoom_in()

    def _nav_params(self):
        t0 = self.win_start.value()
        t1 = self.win_end.value()
        span = t1 - t0
        t_min = self._raw["time_s"][0]
        t_max = self._raw["time_s"][-1]
        return t0, t1, span, t_min, t_max

    def _pan_right(self):
        if self._raw is None:
            return
        t0, t1, span, t_min, t_max = self._nav_params()
        shift = span * 0.25
        new_start = min(t0 + shift, t_max - span)
        new_start = max(new_start, t_min)
        self.win_start.setValue(new_start)
        self.win_end.setValue(new_start + span)
        self._cycle = None
        self._update_plots()

    def _pan_left(self):
        if self._raw is None:
            return
        t0, t1, span, t_min, t_max = self._nav_params()
        shift = span * 0.25
        new_start = max(t0 - shift, t_min)
        self.win_start.setValue(new_start)
        self.win_end.setValue(new_start + span)
        self._cycle = None
        self._update_plots()

    def _zoom_out(self):
        if self._raw is None:
            return
        t0, t1, span, t_min, t_max = self._nav_params()
        new_span = min(span * 1.5, t_max - t_min)
        centre = (t0 + t1) / 2
        new_start = max(centre - new_span / 2, t_min)
        new_end = min(new_start + new_span, t_max)
        new_start = max(new_end - new_span, t_min)
        self.win_start.setValue(new_start)
        self.win_end.setValue(new_end)
        self._cycle = None
        self._update_plots()

    def _zoom_in(self):
        if self._raw is None:
            return
        t0, t1, span, t_min, t_max = self._nav_params()
        new_span = max(span * 0.667, self._raw["time_inc"] * 100)
        centre = (t0 + t1) / 2
        new_start = max(centre - new_span / 2, t_min)
        new_end = min(new_start + new_span, t_max)
        self.win_start.setValue(new_start)
        self.win_end.setValue(new_end)
        self._cycle = None
        self._update_plots()
