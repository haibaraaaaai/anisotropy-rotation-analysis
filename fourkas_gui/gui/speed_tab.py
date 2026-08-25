"""Speed viewer tab — extends TraceTab with window-size / overlap controls,
a recompute button, and integrated step fitting on the *displayed* data."""

import numpy as np

from PyQt6.QtWidgets import (
    QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox,
    QPushButton, QApplication, QCheckBox,
)

from .trace_tab import TraceTab, _no_wheel_spin
from ..core.correction import compute_speed_windowed
from ..core.steps import run_step_fit, reconstruct_step_trace


class SpeedTab(TraceTab):
    """Tab for viewing windowed rotation speed with adjustable parameters
    and optional piecewise-constant step fitting."""

    def __init__(self, time_s, phi_trace, freq,
                 default_nperseg=2000, default_overlap=1000,
                 transform=False, theta_trace=None,
                 theta_axis_deg=None, phi_axis_deg=None,
                 parent=None):
        self._phi_trace = phi_trace
        self._full_time_s = time_s
        self._freq = freq
        self._transform = transform
        self._theta_trace = theta_trace
        self._theta_axis_deg = theta_axis_deg
        self._phi_axis_deg = phi_axis_deg
        self._default_nperseg = default_nperseg
        self._default_overlap = default_overlap

        # Speed data (filled on first recompute)
        self._speed_hz = np.array([0.0])
        self._speed_time = np.array([0.0])

        # Step fit state — fit is on a sub-range of the displayed data
        self._step_time = None   # time array the fit was computed on
        self._step_vals = None   # piecewise-constant values (same length)

        title = "Speed (rotation frame)" if transform else "Speed (lab frame)"
        super().__init__(time_s, np.zeros_like(time_s), "Speed (Hz)", title,
                         parent=parent)

        # Override the dummy trace — recompute with real params
        self._recompute_speed()

    # ── Extra controls (injected into left panel) ─────────────────────────

    def _add_extra_controls(self, layout):
        # ── Speed parameters ──────────────────────────────────────────────
        grp = QGroupBox("Speed Parameters")
        form = QFormLayout()

        self.nperseg_spin = QSpinBox()
        self.nperseg_spin.setRange(10, 1000000)
        self.nperseg_spin.setValue(self._default_nperseg)
        _no_wheel_spin(self.nperseg_spin)
        form.addRow("Window size:", self.nperseg_spin)

        self.overlap_spin = QSpinBox()
        self.overlap_spin.setRange(0, 999999)
        self.overlap_spin.setValue(self._default_overlap)
        _no_wheel_spin(self.overlap_spin)
        form.addRow("Overlap:", self.overlap_spin)

        self.recompute_btn = QPushButton("Recompute Speed")
        self.recompute_btn.setStyleSheet(
            "QPushButton { background-color: #FF9800; color: white; "
            "font-weight: bold; padding: 6px; }")
        form.addRow(self.recompute_btn)

        grp.setLayout(form)
        layout.addWidget(grp)

        self.recompute_btn.clicked.connect(self._recompute_speed)

        # ── Step fitting ──────────────────────────────────────────────────
        step_grp = QGroupBox("Step Fit")
        step_form = QFormLayout()

        self.step_chk = QCheckBox("Show step fit")
        self.step_chk.setChecked(False)
        step_form.addRow(self.step_chk)

        # Start / End time for step fitting region
        self.step_start_spin = QDoubleSpinBox()
        self.step_start_spin.setDecimals(3)
        self.step_start_spin.setSingleStep(0.1)
        self.step_start_spin.setSuffix(" s")
        self.step_start_spin.setToolTip(
            "Start time of the region to fit steps on")
        _no_wheel_spin(self.step_start_spin)
        step_form.addRow("Fit start:", self.step_start_spin)

        self.step_end_spin = QDoubleSpinBox()
        self.step_end_spin.setDecimals(3)
        self.step_end_spin.setSingleStep(0.1)
        self.step_end_spin.setSuffix(" s")
        self.step_end_spin.setToolTip(
            "End time of the region to fit steps on")
        _no_wheel_spin(self.step_end_spin)
        step_form.addRow("Fit end:", self.step_end_spin)

        self.step_use_view_btn = QPushButton("Use current view")
        self.step_use_view_btn.setToolTip(
            "Set step fit range to the current view window")
        step_form.addRow(self.step_use_view_btn)

        self.penalty_spin = QDoubleSpinBox()
        self.penalty_spin.setRange(0.001, 10000.0)
        self.penalty_spin.setDecimals(3)
        self.penalty_spin.setSingleStep(0.1)
        self.penalty_spin.setValue(1.0)
        self.penalty_spin.setToolTip(
            "Ruptures penalty — lower = more change points")
        _no_wheel_spin(self.penalty_spin)
        step_form.addRow("Penalty:", self.penalty_spin)

        self.step_min_size_spin = QSpinBox()
        self.step_min_size_spin.setRange(2, 100000)
        self.step_min_size_spin.setValue(5)
        self.step_min_size_spin.setToolTip(
            "Minimum segment length (displayed points)")
        _no_wheel_spin(self.step_min_size_spin)
        step_form.addRow("Min segment:", self.step_min_size_spin)

        self.step_min_hz_spin = QDoubleSpinBox()
        self.step_min_hz_spin.setRange(0.0, 10000.0)
        self.step_min_hz_spin.setDecimals(1)
        self.step_min_hz_spin.setSingleStep(0.5)
        self.step_min_hz_spin.setValue(0.0)
        self.step_min_hz_spin.setSuffix(" Hz")
        self.step_min_hz_spin.setToolTip(
            "Steps smaller than this (Hz) are merged. 0 = no merging.")
        _no_wheel_spin(self.step_min_hz_spin)
        step_form.addRow("Min step:", self.step_min_hz_spin)

        self.step_fit_btn = QPushButton("Run Step Fit")
        self.step_fit_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 6px; }")
        step_form.addRow(self.step_fit_btn)

        step_grp.setLayout(step_form)
        layout.addWidget(step_grp)

        self.step_fit_btn.clicked.connect(self._run_step_fit)
        self.step_chk.toggled.connect(self._update_plot)
        self.step_use_view_btn.clicked.connect(self._sync_step_range_to_view)

    # ── Sync step fit range with current view ─────────────────────────────

    def _sync_step_range_to_view(self):
        """Copy the current view window into the step fit start/end."""
        self.step_start_spin.setValue(self.win_start.value())
        self.step_end_spin.setValue(self.win_end.value())

    def _update_step_range(self):
        """Update step fit spinbox ranges to match speed data bounds."""
        if len(self._speed_time) > 0:
            t0 = float(self._speed_time[0])
            t_max = float(self._speed_time[-1])
            self.step_start_spin.setRange(t0, t_max)
            self.step_end_spin.setRange(t0, t_max)

    # ── Invalidate step fit when view parameters change ───────────────────

    def _invalidate_steps(self):
        """Clear step fit results (called when displayed data changes)."""
        self._step_time = None
        self._step_vals = None
        if hasattr(self, 'step_chk'):
            self.step_chk.setChecked(False)

    def _connect_signals(self):
        """Extend base to invalidate steps when decimation/smoothing change."""
        super()._connect_signals()
        # Decimation / smoothing changes invalidate step fit
        self.dec_spin.editingFinished.connect(self._invalidate_steps)
        self.smooth_chk.toggled.connect(self._invalidate_steps)
        self.smooth_win.editingFinished.connect(self._invalidate_steps)

    # ── Get displayed data for a given time range ─────────────────────────

    def _get_range_data(self, t0, t1):
        """Return (time, values) for a time range, with current dec+smooth."""
        from scipy.ndimage import uniform_filter1d
        dec = self.dec_spin.value()
        mask = (self._time_s >= t0) & (self._time_s <= t1)
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return np.array([0.0]), np.array([0.0])
        t_out = self._time_s[idx[::dec]]
        y_out = self._trace[idx[::dec]]
        if self.smooth_chk.isChecked() and len(y_out) > 1:
            win = min(self.smooth_win.value(), len(y_out))
            if win >= 2:
                y_out = uniform_filter1d(y_out, size=win)
        return t_out, y_out

    # ── Recompute speed ───────────────────────────────────────────────────

    def _recompute_speed(self):
        nperseg = self.nperseg_spin.value()
        overlap = self.overlap_spin.value()
        if overlap >= nperseg:
            overlap = nperseg - 1
            self.overlap_spin.setValue(overlap)

        # Remember current view settings
        prev_start = self.win_start.value()
        prev_end = self.win_end.value()
        prev_dec = self.dec_spin.value()

        self.status.setText("Computing speed…")
        QApplication.processEvents()

        self._speed_hz, self._speed_time, _ = compute_speed_windowed(
            self._phi_trace, self._full_time_s,
            nperseg, overlap,
            transform=self._transform,
            theta_trace=self._theta_trace,
            theta_axis_deg=self._theta_axis_deg,
            phi_axis_deg=self._phi_axis_deg,
        )

        # Replace the base-class trace data
        self._time_s = self._speed_time
        self._trace = self._speed_hz

        # Invalidate step fit (speed data changed)
        self._invalidate_steps()

        # Update spinbox ranges and restore previous view
        if len(self._speed_time) > 0:
            t0 = float(self._speed_time[0])
            t_max = float(self._speed_time[-1])
            self.win_start.setRange(t0, t_max)
            self.win_end.setRange(t0, t_max)
            self.win_start.setValue(np.clip(prev_start, t0, t_max))
            self.win_end.setValue(np.clip(prev_end, t0, t_max))
            self.dec_spin.setValue(prev_dec)
            self._update_step_range()

        self._update_plot()
        self.status.setText(
            f"Speed computed: {len(self._speed_hz):,} windows  "
            f"(nperseg={nperseg}, overlap={overlap})")

    # ── Step fitting on a sub-range of displayed data ─────────────────────

    def _run_step_fit(self):
        fit_t0 = self.step_start_spin.value()
        fit_t1 = self.step_end_spin.value()
        if fit_t1 <= fit_t0:
            self.status.setText("Step fit: start must be < end.")
            return

        # Get data in the fit range with current dec + smoothing
        t_fit, y_fit = self._get_range_data(fit_t0, fit_t1)

        if len(y_fit) < 3:
            self.status.setText("Not enough data in fit range for step fit.")
            return

        self.status.setText("Running step fit…")
        QApplication.processEvents()

        penalty = self.penalty_spin.value()
        min_size = self.step_min_size_spin.value()
        min_hz = self.step_min_hz_spin.value()
        min_step = min_hz if min_hz > 0 else None

        boundaries, levels = run_step_fit(
            y_fit, penalty=penalty,
            min_size=min_size, min_step=min_step)

        # Build piecewise-constant array aligned to t_fit
        step_vals = reconstruct_step_trace(len(y_fit), boundaries, levels)

        self._step_time = t_fit
        self._step_vals = step_vals

        n_segs = len(levels)
        n_pts = len(y_fit)
        if n_segs > 1:
            step_sizes = np.abs(np.diff(levels))
            self.status.setText(
                f"Step fit: {n_segs} segments on {n_pts:,} pts "
                f"[{fit_t0:.3f}–{fit_t1:.3f} s], "
                f"mean step = {step_sizes.mean():.2f} Hz  "
                f"(dec={self.dec_spin.value()}"
                f"{', smoothed' if self.smooth_chk.isChecked() else ''})")
        else:
            self.status.setText(f"Step fit: {n_segs} segment(s)")

        # Show overlay automatically
        self.step_chk.setChecked(True)
        self._update_plot()

    # ── Plot override ─────────────────────────────────────────────────────

    def _update_plot(self):
        t, y = self._get_window_data()
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.plot(t, y, lw=0.5, color="tab:blue", alpha=0.6)

        # Overlay step fit if enabled and available
        if (self._step_vals is not None
                and hasattr(self, 'step_chk')
                and self.step_chk.isChecked()):
            # Only draw the portion of the step fit visible in current view
            view_t0 = self.win_start.value()
            view_t1 = self.win_end.value()
            mask = ((self._step_time >= view_t0)
                    & (self._step_time <= view_t1))
            if np.any(mask):
                ax.plot(self._step_time[mask], self._step_vals[mask],
                        lw=1.5, color="tab:red", alpha=0.9,
                        label="step fit", drawstyle="steps-post")
                ax.legend(fontsize=8, loc="upper right")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel(self._ylabel)
        smooth_str = (f"  smooth={self.smooth_win.value()}"
                      if self.smooth_chk.isChecked() else "")
        ax.set_title(f"{self._title}  [{self.win_start.value():.3f} – "
                     f"{self.win_end.value():.3f} s]  "
                     f"dec={self.dec_spin.value()}{smooth_str}")
        ax.grid(alpha=0.25)
        self.fig.tight_layout()
        self.canvas.draw_idle()
        # Keep step-fit status if present
        if not ("Step fit" in self.status.text()
                or "Speed computed" in self.status.text()):
            self.status.setText(f"{len(t):,} pts displayed")
