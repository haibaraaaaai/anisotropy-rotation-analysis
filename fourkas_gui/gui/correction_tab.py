"""Correction & fitting tab — channel correction, T-matrix, shape fit,
manual parameter tuning with overlay, and confirmation."""

import numpy as np

from PyQt6.QtCore import Qt, pyqtSignal, QEvent
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QFormLayout,
    QGroupBox, QPushButton, QDoubleSpinBox, QLabel,
    QSplitter, QMessageBox, QApplication, QScrollArea,
    QCheckBox,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from ..core.correction import (
    build_matcor, find_channel_gain, compute_offset,
    apply_correction, run_shape_fit, fourkas_template,
    fourkas_ABC, fourkas_ABC_annular,
)
from ..core.io import anisotropy_from_channels, get_pol_ind


def _dspin(value, lo, hi, step, decimals):
    """Helper to create a QDoubleSpinBox with mouse-wheel disabled."""
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setValue(value)
    s.setSingleStep(step)
    s.setDecimals(decimals)
    s.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    s.installEventFilter(_WheelFilter.instance())
    return s


_ARROW_KEYS = {
    Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down,
}


class _WheelFilter:
    """Singleton event filter that blocks wheel *and* arrow-key events on spinboxes."""
    _inst = None

    @classmethod
    def instance(cls):
        if cls._inst is None:
            from PyQt6.QtCore import QObject
            cls._inst = _WheelFilterObj()
        return cls._inst


class _WheelFilterObj(QWidget):
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Wheel:
            return True  # swallow
        if event.type() == QEvent.Type.KeyPress and event.key() in _ARROW_KEYS:
            return True  # swallow
        return False


class CorrectionTab(QWidget):
    """Tab for channel correction and shape fitting on a confirmed cycle."""

    confirmed = pyqtSignal(object)  # emits full result dict

    def __init__(self, c0_cyc, c90_cyc, c45_cyc, c135_cyc,
                 matched_idx, sx_ref, sy_ref, parent=None):
        super().__init__(parent)
        self._c0 = c0_cyc
        self._c90 = c90_cyc
        self._c45 = c45_cyc
        self._c135 = c135_cyc
        self._matched_idx = matched_idx
        self._sx_ref = sx_ref
        self._sy_ref = sy_ref

        self._matcor, self._matcorb = build_matcor()
        self._a = None
        self._o = None
        self._tcor_sub = None
        self._fit_result = None         # last auto-fit result dict
        self._auto_fit_params = None    # snapshot of auto-fit spinbox values

        self._build_ui()
        self._connect_signals()

    # ── UI ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        outer = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(splitter)

        # ── Left panel (scrollable) ───────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        ctrl = QWidget()
        ctrl_layout = QVBoxLayout(ctrl)
        ctrl_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # ── Info ──────────────────────────────────────────────────────────
        info_grp = QGroupBox("Cycle Info")
        info_form = QFormLayout()
        n_raw = len(self._c0)
        n_sub = len(self._matched_idx)
        self.info_label = QLabel(
            f"Cycle: {n_raw:,} raw points\n"
            f"Subsampled: {n_sub} points")
        self.info_label.setWordWrap(True)
        info_form.addRow(self.info_label)
        info_grp.setLayout(info_form)
        ctrl_layout.addWidget(info_grp)

        # ── Auto correction ──────────────────────────────────────────────
        cor_grp = QGroupBox("Channel Correction")
        cor_lay = QVBoxLayout()
        self.compute_cor_btn = QPushButton("Compute Gain && T-Correction")
        cor_lay.addWidget(self.compute_cor_btn)
        self.cor_info = QLabel("")
        self.cor_info.setWordWrap(True)
        cor_lay.addWidget(self.cor_info)
        cor_grp.setLayout(cor_lay)
        ctrl_layout.addWidget(cor_grp)

        # ── Manual a / o ─────────────────────────────────────────────────
        ao_grp = QGroupBox("Manual Gain / Offset")
        ao_form = QFormLayout()

        self.a90_spin = _dspin(1.0, 0.0, 5.0, 0.001, 4)
        self.a45_spin = _dspin(1.0, 0.0, 5.0, 0.001, 4)
        self.a135_spin = _dspin(1.0, 0.0, 5.0, 0.001, 4)
        ao_form.addRow("a_90:", self.a90_spin)
        ao_form.addRow("a_45:", self.a45_spin)
        ao_form.addRow("a_135:", self.a135_spin)

        self.o90_spin = _dspin(0.0, -1e6, 1e6, 0.1, 2)
        self.o45_spin = _dspin(0.0, -1e6, 1e6, 0.1, 2)
        self.o135_spin = _dspin(0.0, -1e6, 1e6, 0.1, 2)
        self.o0_spin = _dspin(0.0, -1e6, 1e6, 0.1, 2)
        ao_form.addRow("o_90:", self.o90_spin)
        ao_form.addRow("o_45:", self.o45_spin)
        ao_form.addRow("o_135:", self.o135_spin)
        ao_form.addRow("o_0:", self.o0_spin)

        self.apply_ao_btn = QPushButton("Apply Manual a / o")
        ao_form.addRow(self.apply_ao_btn)

        ao_grp.setLayout(ao_form)
        ctrl_layout.addWidget(ao_grp)

        # ── Auto shape fit ────────────────────────────────────────────────
        fit_grp = QGroupBox("Auto Shape Fit")
        fit_form = QFormLayout()

        self.na_spin = _dspin(1.30, 0.1, 2.0, 0.00001, 5)
        fit_form.addRow("NA:", self.na_spin)

        self.n_spin = _dspin(1.33, 1.0, 2.0, 0.01, 2)
        fit_form.addRow("n:", self.n_spin)

        self.hole_chk = QCheckBox("Use hole correction (annular)")
        self.hole_chk.setChecked(False)
        fit_form.addRow(self.hole_chk)

        self.na_out_spin = _dspin(1.30, 0.1, 2.0, 0.00001, 5)
        fit_form.addRow("NA_out:", self.na_out_spin)

        self.na_in_spin = _dspin(0.39, 0.0, 2.0, 0.00001, 5)
        fit_form.addRow("NA_in:", self.na_in_spin)

        self.run_fit_btn = QPushButton("Run Shape Fit")
        self.run_fit_btn.setEnabled(False)
        self.run_fit_btn.setStyleSheet(
            "QPushButton { background-color: #FF9800; color: white; "
            "font-weight: bold; padding: 6px; }")
        fit_form.addRow(self.run_fit_btn)

        fit_grp.setLayout(fit_form)
        ctrl_layout.addWidget(fit_grp)

        # ── Manual fit ────────────────────────────────────────────────────
        mfit_grp = QGroupBox("Manual Fit (overlay)")
        mfit_form = QFormLayout()

        self.theta_spin = _dspin(45.0, 0.0, 90.0, 0.1, 2)
        self.phi_spin = _dspin(0.0, -180.0, 180.0, 0.1, 2)
        self.lam_spin = _dspin(45.0, 0.0, 90.0, 0.1, 2)
        mfit_form.addRow("Θ (axis polar):", self.theta_spin)
        mfit_form.addRow("Φ (axis azimuth):", self.phi_spin)
        mfit_form.addRow("Λ (cone half):", self.lam_spin)

        self.dc_spin = _dspin(0.0, -1e6, 1e6, 0.001, 4)
        self.fa_spin = _dspin(0.0, -1.0, 1.0, 0.001, 4)
        self.fb_spin = _dspin(0.0, -1.0, 1.0, 0.001, 4)
        mfit_form.addRow("δc:", self.dc_spin)
        mfit_form.addRow("fa:", self.fa_spin)
        mfit_form.addRow("fb:", self.fb_spin)

        btn_row = QHBoxLayout()
        self.set_manual_btn = QPushButton("Set Manual Fit")
        self.set_manual_btn.setEnabled(False)
        self.set_manual_btn.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; "
            "font-weight: bold; padding: 4px; }")
        btn_row.addWidget(self.set_manual_btn)

        self.revert_btn = QPushButton("Revert to Auto")
        self.revert_btn.setEnabled(False)
        btn_row.addWidget(self.revert_btn)

        mfit_form.addRow(btn_row)
        mfit_grp.setLayout(mfit_form)
        ctrl_layout.addWidget(mfit_grp)

        # ── Confirm ──────────────────────────────────────────────────────
        self.rot_frame_chk = QCheckBox("Speed in rotation frame")
        self.rot_frame_chk.setChecked(False)
        ctrl_layout.addWidget(self.rot_frame_chk)

        self.confirm_btn = QPushButton("Confirm Fit && Compute Angles/Speed")
        self.confirm_btn.setEnabled(False)
        self.confirm_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 6px; }")
        ctrl_layout.addWidget(self.confirm_btn)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        ctrl_layout.addWidget(self.status)
        ctrl_layout.addStretch()

        scroll.setWidget(ctrl)
        scroll.setMinimumWidth(290)
        scroll.setMaximumWidth(360)
        splitter.addWidget(scroll)

        # ── Centre: plot ──────────────────────────────────────────────────
        centre = QWidget()
        centre_layout = QVBoxLayout(centre)
        centre_layout.setContentsMargins(0, 0, 0, 0)

        self.fig = Figure(figsize=(10, 10), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        centre_layout.addWidget(self.toolbar)
        centre_layout.addWidget(self.canvas)
        splitter.addWidget(centre)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self._plot_raw()
        self._apply_hole_ui_state()

    # ── Signals ───────────────────────────────────────────────────────────

    def _connect_signals(self):
        self.compute_cor_btn.clicked.connect(self._compute_correction)
        self.apply_ao_btn.clicked.connect(self._apply_manual_ao)
        self.run_fit_btn.clicked.connect(self._run_fit)
        self.confirm_btn.clicked.connect(self._confirm)
        self.set_manual_btn.clicked.connect(self._set_manual_fit)
        self.revert_btn.clicked.connect(self._revert_to_auto)
        self.hole_chk.toggled.connect(self._on_calibration_mode_changed)
        self.na_spin.valueChanged.connect(self._on_calibration_mode_changed)
        self.n_spin.valueChanged.connect(self._on_calibration_mode_changed)
        self.na_in_spin.valueChanged.connect(self._on_calibration_mode_changed)
        self.na_out_spin.valueChanged.connect(self._on_calibration_mode_changed)

        # Manual fit spinboxes → live overlay update
        for spin in (self.theta_spin, self.phi_spin, self.lam_spin,
                     self.dc_spin, self.fa_spin, self.fb_spin):
            spin.valueChanged.connect(self._on_manual_fit_changed)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _recompute_tcor_sub(self):
        idx = self._matched_idx
        self._tcor_sub = apply_correction(
            self._c0[idx], self._c90[idx],
            self._c45[idx], self._c135[idx],
            self._a, self._o, self._matcor)

    def _update_ao_spinboxes(self):
        for spin, val in zip(
                [self.a90_spin, self.a45_spin, self.a135_spin],
                [self._a[0], self._a[1], self._a[2]]):
            spin.blockSignals(True)
            spin.setValue(val)
            spin.blockSignals(False)
        for spin, val in zip(
                [self.o90_spin, self.o45_spin, self.o135_spin, self.o0_spin],
                [self._o[0], self._o[1], self._o[2], self._o[3]]):
            spin.blockSignals(True)
            spin.setValue(val)
            spin.blockSignals(False)

    def _update_fit_spinboxes(self, result, block=True):
        for spin, key in [(self.theta_spin, "Theta"),
                          (self.phi_spin, "Phi"),
                          (self.lam_spin, "Lambda"),
                          (self.dc_spin, "dc"),
                          (self.fa_spin, "fa"),
                          (self.fb_spin, "fb")]:
            if block:
                spin.blockSignals(True)
            spin.setValue(result[key])
            if block:
                spin.blockSignals(False)

    def _snapshot_spinbox_params(self):
        """Return dict of current manual-fit spinbox values."""
        return {
            "Theta": self.theta_spin.value(),
            "Phi": self.phi_spin.value(),
            "Lambda": self.lam_spin.value(),
            "dc": self.dc_spin.value(),
            "fa": self.fa_spin.value(),
            "fb": self.fb_spin.value(),
        }

    def _apply_hole_ui_state(self):
        use_hole = self.hole_chk.isChecked()
        self.na_in_spin.setEnabled(use_hole)
        self.na_out_spin.setEnabled(use_hole)

    def _current_fourkas_coeffs(self):
        """Compute current Fourkas coefficients from selected calibration mode."""
        n = self.n_spin.value()
        if self.hole_chk.isChecked():
            na_in = self.na_in_spin.value()
            na_out = self.na_out_spin.value()
            A, B, C = fourkas_ABC_annular(na_in, na_out, n_water=n)
            mode = "annular"
        else:
            NA = self.na_spin.value()
            if NA >= n:
                raise ValueError("NA must be smaller than n.")
            alpha = np.arcsin(NA / n)
            A, B, C = fourkas_ABC(alpha)
            mode = "legacy"
            na_in = None
            na_out = None
        return A, B, C, mode, na_in, na_out

    def _on_calibration_mode_changed(self, _value=None):
        self._apply_hole_ui_state()
        if self._tcor_sub is None or self._auto_fit_params is None:
            return
        try:
            self._plot_with_overlay()
        except ValueError as exc:
            self.status.setText(f"Calibration settings invalid: {exc}")

    def _compute_model(self, params):
        """Compute (ax_model, ay_model) from param dict."""
        A, B, C, _, _, _ = self._current_fourkas_coeffs()
        Theta = np.radians(params["Theta"])
        Phi = np.radians(params["Phi"])
        Lam = np.radians(params["Lambda"])
        return fourkas_template(Theta, Phi, Lam, A, B, C)

    def _compute_data(self, params):
        """Compute corrected anisotropy data using param backgrounds."""
        dc = params["dc"]
        fa = params["fa"]
        fb = params["fb"]
        da, db = fa * dc, fb * dc
        b0 = (dc + da) / 2
        b90 = (dc - da) / 2
        b45 = (dc + db) / 2
        b135 = (dc - db) / 2

        idx0 = get_pol_ind(["0"])[0]
        idx90 = get_pol_ind(["90"])[0]
        idx45 = get_pol_ind(["45"])[0]
        idx135 = get_pol_ind(["135"])[0]

        c0_cor = self._tcor_sub[idx0] - b0
        c90_cor = self._tcor_sub[idx90] - b90
        c45_cor = self._tcor_sub[idx45] - b45
        c135_cor = self._tcor_sub[idx135] - b135
        return anisotropy_from_channels(c0_cor, c90_cor, c45_cor, c135_cor)

    def _current_result_dict(self, params):
        """Build a full result dict from param snapshot."""
        A, B, C, mode, na_in, na_out = self._current_fourkas_coeffs()
        R_sat = C / (A + B)
        dc, fa, fb = params["dc"], params["fa"], params["fb"]
        da, db = fa * dc, fb * dc
        ax_data, ay_data = self._compute_data(params)
        ax_model, ay_model = self._compute_model(params)
        return {
            "ax_data": ax_data, "ay_data": ay_data,
            "ax_model": ax_model, "ay_model": ay_model,
            "R_sat": R_sat, "A": A, "B": B, "C": C,
            "Theta": params["Theta"], "Phi": params["Phi"],
            "Lambda": params["Lambda"],
            "dc": dc, "fa": fa, "fb": fb,
            "b0": (dc + da) / 2, "b90": (dc - da) / 2,
            "b45": (dc + db) / 2, "b135": (dc - db) / 2,
            "calibration_mode": mode,
            "NA": self.na_spin.value(),
            "n": self.n_spin.value(),
            "NA_in": na_in,
            "NA_out": na_out,
        }

    # ── Compute auto correction ───────────────────────────────────────────

    def _compute_correction(self):
        self.status.setText("Computing channel gain…")
        QApplication.processEvents()

        c0, c90, c45, c135 = self._c0, self._c90, self._c45, self._c135
        self._a = find_channel_gain(c0, c90, c45, c135, self._matcorb)
        self._o = compute_offset(
            self._a[3] * c0, self._a[0] * c90,
            self._a[1] * c45, self._a[2] * c135)

        self._recompute_tcor_sub()
        self._update_ao_spinboxes()

        self.cor_info.setText(
            f"a_90={self._a[0]:.4f}  a_45={self._a[1]:.4f}  "
            f"a_135={self._a[2]:.4f}\n"
            f"o: {self._o[0]:.2f}, {self._o[1]:.2f}, "
            f"{self._o[2]:.2f}, {self._o[3]:.2f}")

        self.run_fit_btn.setEnabled(True)
        self.confirm_btn.setEnabled(True)
        self.status.setText("Correction computed.")
        self._plot_corrected()

    # ── Apply manual a / o ────────────────────────────────────────────────

    def _apply_manual_ao(self):
        self._a = np.array([
            self.a90_spin.value(),
            self.a45_spin.value(),
            self.a135_spin.value(),
            1.0,
        ])
        self._o = np.array([
            self.o90_spin.value(),
            self.o45_spin.value(),
            self.o135_spin.value(),
            self.o0_spin.value(),
        ])
        self._recompute_tcor_sub()

        self.cor_info.setText(
            f"a_90={self._a[0]:.4f}  a_45={self._a[1]:.4f}  "
            f"a_135={self._a[2]:.4f}  (manual)\n"
            f"o: {self._o[0]:.2f}, {self._o[1]:.2f}, "
            f"{self._o[2]:.2f}, {self._o[3]:.2f}")

        self.run_fit_btn.setEnabled(True)
        self.confirm_btn.setEnabled(True)
        self.status.setText("Manual a/o applied.")
        # Redraw with current fit spinbox values
        if self._auto_fit_params is not None:
            self._plot_with_overlay()
        else:
            self._plot_corrected()

    # ── Auto shape fit ────────────────────────────────────────────────────

    def _run_fit(self):
        if self._tcor_sub is None:
            return

        NA = self.na_spin.value()
        n = self.n_spin.value()

        self.status.setText("Running shape fit…")
        self.run_fit_btn.setEnabled(False)
        QApplication.processEvents()

        try:
            A, B, C, _, _, _ = self._current_fourkas_coeffs()
            result = run_shape_fit(
                self._tcor_sub,
                NA,
                n,
                A_override=A,
                B_override=B,
                C_override=C,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Shape fit error", str(exc))
            self.status.setText(f"Error: {exc}")
            self.run_fit_btn.setEnabled(True)
            return

        self._fit_result = result
        self.run_fit_btn.setEnabled(True)

        # Store auto-fit snapshot
        self._auto_fit_params = {
            "Theta": result["Theta"], "Phi": result["Phi"],
            "Lambda": result["Lambda"],
            "dc": result["dc"], "fa": result["fa"], "fb": result["fb"],
        }

        # Set spinboxes to auto-fit values
        self._update_fit_spinboxes(result)
        self.set_manual_btn.setEnabled(True)
        self.revert_btn.setEnabled(True)
        self.status.setText(
            f"Fit: Θ={result['Theta']:.2f}° Φ={result['Phi']:.2f}° "
            f"Λ={result['Lambda']:.2f}°  cost={result['cost']:.6f}")

        # Plot auto-fit (no overlay yet since spinboxes == auto)
        self._plot_with_overlay()

    # ── Manual fit live overlay ───────────────────────────────────────────

    def _on_manual_fit_changed(self, _value=None):
        if self._tcor_sub is None:
            return
        self._plot_with_overlay()

    def _set_manual_fit(self):
        """Accept current spinbox values as the active fit."""
        self._auto_fit_params = self._snapshot_spinbox_params()
        self.status.setText("Manual fit values set as active.")
        self._plot_with_overlay()

    def _revert_to_auto(self):
        """Revert spinboxes to last auto-fit values."""
        if self._fit_result is None:
            return
        self._auto_fit_params = {
            "Theta": self._fit_result["Theta"],
            "Phi": self._fit_result["Phi"],
            "Lambda": self._fit_result["Lambda"],
            "dc": self._fit_result["dc"],
            "fa": self._fit_result["fa"],
            "fb": self._fit_result["fb"],
        }
        self._update_fit_spinboxes(self._fit_result)
        self.status.setText("Reverted to auto-fit values.")
        self._plot_with_overlay()

    # ── Confirm ───────────────────────────────────────────────────────────

    def _confirm(self):
        params = self._auto_fit_params
        if params is None:
            params = self._snapshot_spinbox_params()
        if self._tcor_sub is None:
            return
        info = self._current_result_dict(params)
        info["a"] = self._a.tolist() if self._a is not None else None
        info["o"] = self._o.tolist() if self._o is not None else None
        info["rotation_frame"] = self.rot_frame_chk.isChecked()
        self.confirmed.emit(info)
        self.status.setText("Fit values confirmed — computing angles/speed…")

    # ── Plots ─────────────────────────────────────────────────────────────

    def _plot_raw(self):
        self.fig.clear()
        ax_data, ay_data = anisotropy_from_channels(
            self._c0, self._c90, self._c45, self._c135)
        ax = self.fig.add_subplot(111)
        ax.scatter(ax_data, ay_data, s=0.5, alpha=0.15, c="tab:grey",
                   rasterized=True, label="all cycle")
        idx = self._matched_idx
        ax.scatter(ax_data[idx], ay_data[idx], s=15, c="tab:purple",
                   marker="x", alpha=0.8, label=f"{len(idx)} sub-pts")
        sx_c = np.append(self._sx_ref, self._sx_ref[0])
        sy_c = np.append(self._sy_ref, self._sy_ref[0])
        ax.plot(sx_c, sy_c, 'r-', lw=1, alpha=0.5, label="ref curve")
        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("anis_x"); ax.set_ylabel("anis_y")
        ax.set_title("Raw anisotropy (uncorrected)")
        ax.legend(fontsize=8); ax.grid(alpha=0.25)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _plot_corrected(self):
        if self._tcor_sub is None:
            return
        self.fig.clear()
        idx0 = get_pol_ind(["0"])[0]
        idx90 = get_pol_ind(["90"])[0]
        idx45 = get_pol_ind(["45"])[0]
        idx135 = get_pol_ind(["135"])[0]
        ax_cor, ay_cor = anisotropy_from_channels(
            self._tcor_sub[idx0], self._tcor_sub[idx90],
            self._tcor_sub[idx45], self._tcor_sub[idx135])
        ax = self.fig.add_subplot(111)
        ax.scatter(ax_cor, ay_cor, s=30, c="tab:blue", marker="o",
                   alpha=0.8, label="corrected sub-pts")
        sx_c = np.append(self._sx_ref, self._sx_ref[0])
        sy_c = np.append(self._sy_ref, self._sy_ref[0])
        ax.plot(sx_c, sy_c, 'r-', lw=1, alpha=0.4, label="ref curve")
        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("anis_x"); ax.set_ylabel("anis_y")
        ax.set_title("T-corrected anisotropy (subsampled)")
        ax.legend(fontsize=8); ax.grid(alpha=0.25)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _plot_with_overlay(self):
        """Plot auto-fit + manual-fit overlay from spinbox values."""
        if self._tcor_sub is None or self._auto_fit_params is None:
            return

        auto = self._auto_fit_params
        manual = self._snapshot_spinbox_params()

        self.fig.clear()
        ax = self.fig.add_subplot(111)

        # Auto-fit data & model (solid, authoritative)
        ax_auto_d, ay_auto_d = self._compute_data(auto)
        ax_auto_m, ay_auto_m = self._compute_model(auto)

        ax.scatter(ax_auto_d, ay_auto_d,
                   s=30, c="tab:blue", marker="o", alpha=0.8,
                   label="data (auto)")
        ax_mc = np.append(ax_auto_m, ax_auto_m[0])
        ay_mc = np.append(ay_auto_m, ay_auto_m[0])
        ax.plot(ax_mc, ay_mc, 'g-', lw=2, alpha=0.8,
                label=(f"auto: Θ={auto['Theta']:.1f}° "
                       f"Φ={auto['Phi']:.1f}° "
                       f"Λ={auto['Lambda']:.1f}°"))

        # Manual overlay (only if different from auto)
        is_same = all(
            abs(manual[k] - auto[k]) < 1e-6
            for k in ("Theta", "Phi", "Lambda", "dc", "fa", "fb"))

        if not is_same:
            ax_man_d, ay_man_d = self._compute_data(manual)
            ax_man_m, ay_man_m = self._compute_model(manual)

            ax.scatter(ax_man_d, ay_man_d,
                       s=20, c="tab:orange", marker="x", alpha=0.6,
                       label="data (manual)")
            ax_mmc = np.append(ax_man_m, ax_man_m[0])
            ay_mmc = np.append(ay_man_m, ay_man_m[0])
            ax.fill(ax_mmc, ay_mmc, color="orange", alpha=0.10)
            ax.plot(ax_mmc, ay_mmc, '--', color="tab:orange", lw=1.5,
                    alpha=0.7,
                    label=(f"manual: Θ={manual['Theta']:.1f}° "
                           f"Φ={manual['Phi']:.1f}° "
                           f"Λ={manual['Lambda']:.1f}°"))

        # R_sat circle
        A, B, C, mode, _, _ = self._current_fourkas_coeffs()
        R = C / (A + B)
        t = np.linspace(0, 2 * np.pi, 200)
        ax.plot(R * np.cos(t), R * np.sin(t),
            'k--', lw=0.5, alpha=0.3,
            label=f"R_sat={R:.3f} ({mode})")

        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("anis_x"); ax.set_ylabel("anis_y")
        ax.set_title("Shape Fit" + (" (manual overlay)" if not is_same else ""))
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.25)
        self.fig.tight_layout()
        self.canvas.draw_idle()
