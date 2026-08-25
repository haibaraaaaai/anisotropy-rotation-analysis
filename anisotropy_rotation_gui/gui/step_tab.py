"""Step fitting tab — overlays piecewise-constant step fit on a trace."""

import numpy as np

from PyQt6.QtWidgets import (
    QGroupBox, QFormLayout, QDoubleSpinBox, QSpinBox,
    QPushButton, QApplication,
)

from .trace_tab import TraceTab, _no_wheel_spin
from ..core.steps import run_step_fit, reconstruct_step_trace


class StepTab(TraceTab):
    """Tab for viewing a trace with an overlaid step fit.

    The step fit uses ruptures KernelCPD (linear kernel) change-point
    detection with adjustable penalty, min segment size, and min step
    size.  After fitting, the piecewise-constant result is drawn on
    top of the raw trace.
    """

    def __init__(self, time_s, trace, ylabel, title,
                 default_penalty=0.1, default_min_size=5,
                 default_min_deg=10.0, parent=None):
        self._full_trace = trace
        self._full_time = time_s
        self._step_trace = None       # piecewise-constant reconstruction
        self._step_boundaries = None  # index array
        self._step_levels = None      # level per segment
        self._default_penalty = default_penalty
        self._default_min_size = default_min_size
        self._default_min_deg = default_min_deg

        super().__init__(time_s, trace, ylabel, title, parent=parent)

    # ── Extra controls (injected into left panel) ─────────────────────────

    def _add_extra_controls(self, layout):
        grp = QGroupBox("Step Fit")
        form = QFormLayout()

        self.penalty_spin = QDoubleSpinBox()
        self.penalty_spin.setRange(0.001, 1000.0)
        self.penalty_spin.setDecimals(3)
        self.penalty_spin.setSingleStep(0.01)
        self.penalty_spin.setValue(self._default_penalty)
        self.penalty_spin.setToolTip(
            "Ruptures penalty — lower = more change points")
        _no_wheel_spin(self.penalty_spin)
        form.addRow("Penalty:", self.penalty_spin)

        self.min_size_spin = QSpinBox()
        self.min_size_spin.setRange(2, 1000000)
        self.min_size_spin.setValue(self._default_min_size)
        self.min_size_spin.setToolTip(
            "Minimum segment length (samples)")
        _no_wheel_spin(self.min_size_spin)
        form.addRow("Min segment:", self.min_size_spin)

        self.min_deg_spin = QDoubleSpinBox()
        self.min_deg_spin.setRange(0.0, 360.0)
        self.min_deg_spin.setDecimals(1)
        self.min_deg_spin.setSingleStep(1.0)
        self.min_deg_spin.setValue(self._default_min_deg)
        self.min_deg_spin.setSuffix("°")
        self.min_deg_spin.setToolTip(
            "Steps smaller than this are merged")
        _no_wheel_spin(self.min_deg_spin)
        form.addRow("Min step:", self.min_deg_spin)

        self.fit_btn = QPushButton("Run Step Fit")
        self.fit_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 6px; }")
        form.addRow(self.fit_btn)

        grp.setLayout(form)
        layout.addWidget(grp)

        self.fit_btn.clicked.connect(self._run_fit)

    # ── Fit ───────────────────────────────────────────────────────────────

    def _run_fit(self):
        self.status.setText("Running step fit…")
        QApplication.processEvents()

        penalty = self.penalty_spin.value()
        min_size = self.min_size_spin.value()
        min_deg = self.min_deg_spin.value()

        boundaries, levels = run_step_fit(
            self._full_trace, penalty=penalty,
            min_size=min_size, min_step=min_deg * np.pi / 180.0)

        self._step_boundaries = boundaries
        self._step_levels = levels
        self._step_trace = reconstruct_step_trace(
            len(self._full_trace), boundaries, levels)

        n_steps = len(levels)
        step_sizes_deg = np.abs(np.diff(levels)) * 180.0 / np.pi
        self.status.setText(
            f"Step fit done: {n_steps} segments, "
            f"mean step = {step_sizes_deg.mean():.1f}°" if len(step_sizes_deg) > 0
            else f"Step fit done: {n_steps} segment(s)")
        self._update_plot()

    # ── Plot override ─────────────────────────────────────────────────────

    def _update_plot(self):
        t, y = self._get_window_data()
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.plot(t, y, lw=0.5, color="tab:blue", alpha=0.6)

        # Overlay step fit if available
        if self._step_trace is not None:
            t0 = self.win_start.value()
            t1 = self.win_end.value()
            dec = self.dec_spin.value()
            mask = (self._time_s >= t0) & (self._time_s <= t1)
            idx = np.where(mask)[0]
            if len(idx) > 0:
                t_step = self._time_s[idx[::dec]]
                y_step = self._step_trace[idx[::dec]]
                ax.plot(t_step, y_step, lw=1.5, color="tab:red",
                        alpha=0.9, label="step fit")
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
        self.status.setText(
            self.status.text() if "Step fit" in self.status.text()
            else f"{len(t):,} pts displayed")
