"""Trace viewer tab — generic 1-D trace with window/decimation controls."""

import numpy as np
from scipy.ndimage import uniform_filter1d

from PyQt6.QtCore import Qt, QEvent
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QFormLayout,
    QGroupBox, QDoubleSpinBox, QSpinBox, QLabel,
    QPushButton, QSplitter, QCheckBox,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


_ARROW_KEYS = {
    Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down,
}


class _WheelFilter(QWidget):
    """Singleton event filter that blocks wheel *and* arrow-key events."""
    _inst = None

    @classmethod
    def instance(cls):
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Wheel:
            return True
        if event.type() == QEvent.Type.KeyPress and event.key() in _ARROW_KEYS:
            return True
        return False


def _no_wheel_spin(spin):
    """Configure a spinbox to ignore mouse wheel and not steal arrow-key focus."""
    spin.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    spin.installEventFilter(_WheelFilter.instance())
    return spin


class TraceTab(QWidget):
    """Tab for viewing a 1-D trace with start/end/decimation controls."""

    def __init__(self, time_s, trace, ylabel, title, parent=None):
        """
        Parameters
        ----------
        time_s : 1-D array — time axis (seconds)
        trace  : 1-D array — data to plot (same length as time_s)
        ylabel : str — Y-axis label
        title  : str — plot title / tab label
        """
        super().__init__(parent)
        self._time_s = time_s
        self._trace = trace
        self._ylabel = ylabel
        self._title = title

        self._build_ui()
        self._connect_signals()
        self._update_plot()

    # ── UI ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        outer = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(splitter)

        # ── Left: controls ────────────────────────────────────────────────
        ctrl = QWidget()
        ctrl_layout = QVBoxLayout(ctrl)
        ctrl_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        win_grp = QGroupBox("View Window")
        win_form = QFormLayout()

        t_min, t_max = float(self._time_s[0]), float(self._time_s[-1])

        self.win_start = QDoubleSpinBox()
        self.win_start.setRange(t_min, t_max)
        self.win_start.setDecimals(3)
        self.win_start.setSuffix(" s")
        self.win_start.setValue(t_min)
        _no_wheel_spin(self.win_start)
        win_form.addRow("Start:", self.win_start)

        self.win_end = QDoubleSpinBox()
        self.win_end.setRange(t_min, t_max)
        self.win_end.setDecimals(3)
        self.win_end.setSuffix(" s")
        self.win_end.setValue(min(t_min + 10.0, t_max))
        _no_wheel_spin(self.win_end)
        win_form.addRow("End:", self.win_end)

        self.dec_spin = QSpinBox()
        self.dec_spin.setRange(1, 10000)
        self.dec_spin.setValue(100)
        _no_wheel_spin(self.dec_spin)
        win_form.addRow("Decimation:", self.dec_spin)

        self.apply_btn = QPushButton("Apply")
        win_form.addRow(self.apply_btn)

        win_grp.setLayout(win_form)
        ctrl_layout.addWidget(win_grp)

        # ── Smoothing ────────────────────────────────────────────────
        smooth_grp = QGroupBox("Smoothing")
        smooth_form = QFormLayout()

        self.smooth_chk = QCheckBox("Enable")
        self.smooth_chk.setChecked(False)
        smooth_form.addRow(self.smooth_chk)

        self.smooth_win = QSpinBox()
        self.smooth_win.setRange(3, 100000)
        self.smooth_win.setValue(101)
        self.smooth_win.setToolTip("Uniform moving-average window (samples)")
        _no_wheel_spin(self.smooth_win)
        smooth_form.addRow("Window:", self.smooth_win)

        smooth_grp.setLayout(smooth_form)
        ctrl_layout.addWidget(smooth_grp)

        nav_lbl = QLabel("← → pan  ↑ ↓ zoom")
        nav_lbl.setStyleSheet("color: grey; font-size: 11px;")
        ctrl_layout.addWidget(nav_lbl)

        self._add_extra_controls(ctrl_layout)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        ctrl_layout.addWidget(self.status)
        ctrl_layout.addStretch()

        ctrl.setMinimumWidth(220)
        ctrl.setMaximumWidth(300)
        splitter.addWidget(ctrl)

        # ── Centre: plot ──────────────────────────────────────────────────
        centre = QWidget()
        centre_layout = QVBoxLayout(centre)
        centre_layout.setContentsMargins(0, 0, 0, 0)

        self.fig = Figure(figsize=(12, 5), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        centre_layout.addWidget(self.toolbar)
        centre_layout.addWidget(self.canvas)
        splitter.addWidget(centre)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

    def _add_extra_controls(self, layout):
        """Override in subclass to add more controls."""
        pass

    # ── Signals ───────────────────────────────────────────────────────────

    def _connect_signals(self):
        self.apply_btn.clicked.connect(self._update_plot)
        self.win_start.editingFinished.connect(self._update_plot)
        self.win_end.editingFinished.connect(self._update_plot)
        self.dec_spin.editingFinished.connect(self._update_plot)
        self.smooth_chk.toggled.connect(self._update_plot)
        self.smooth_win.editingFinished.connect(self._update_plot)

    # ── Plotting ──────────────────────────────────────────────────────────

    def _get_window_data(self):
        """Return (time_slice, trace_slice) for current window+decimation."""
        t0 = self.win_start.value()
        t1 = self.win_end.value()
        dec = self.dec_spin.value()
        mask = (self._time_s >= t0) & (self._time_s <= t1)
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return np.array([0]), np.array([0])
        t_out = self._time_s[idx[::dec]]
        y_out = self._trace[idx[::dec]]
        if self.smooth_chk.isChecked() and len(y_out) > 1:
            win = min(self.smooth_win.value(), len(y_out))
            if win >= 2:
                y_out = uniform_filter1d(y_out, size=win)
        return t_out, y_out

    def _update_plot(self):
        t, y = self._get_window_data()
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.plot(t, y, lw=0.5, color="tab:blue")
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
        self.status.setText(f"{len(t):,} pts displayed")

    # ── Arrow-key navigation (called by MainWindow) ──────────────────────

    def nav_pan(self, direction):
        """Pan left (-1) or right (+1)."""
        t0 = self.win_start.value()
        t1 = self.win_end.value()
        span = t1 - t0
        t_min = float(self._time_s[0])
        t_max = float(self._time_s[-1])
        shift = span * 0.25 * direction
        new_start = np.clip(t0 + shift, t_min, t_max - span)
        self.win_start.setValue(new_start)
        self.win_end.setValue(new_start + span)
        self._update_plot()

    def nav_zoom(self, factor):
        """Zoom in (factor<1) or out (factor>1)."""
        t0 = self.win_start.value()
        t1 = self.win_end.value()
        span = t1 - t0
        t_min = float(self._time_s[0])
        t_max = float(self._time_s[-1])
        dt_min = (self._time_s[1] - self._time_s[0]) * 100 if len(self._time_s) > 1 else 0.001
        new_span = np.clip(span * factor, dt_min, t_max - t_min)
        centre = (t0 + t1) / 2
        new_start = np.clip(centre - new_span / 2, t_min, t_max - new_span)
        new_end = min(new_start + new_span, t_max)
        self.win_start.setValue(new_start)
        self.win_end.setValue(new_end)
        self._update_plot()
