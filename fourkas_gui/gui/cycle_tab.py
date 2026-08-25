"""Cycle refinement tab — adjust start/end, auto-monotonic refit,
point inspector, and manual point assignment."""

import numpy as np

from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QFormLayout,
    QGroupBox, QPushButton, QSpinBox, QLabel, QApplication,
    QSplitter, QMessageBox, QDialog, QDialogButtonBox,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from ..core.cycle import subsample_ref_curve, is_monotonic, enforce_monotonic


class CycleTab(QWidget):
    """Tab for refining cycle selection and computing reference curve."""

    confirmed = pyqtSignal(int, int, object)  # start, end, ref_result

    def __init__(self, ax_full, ay_full, cyc_start, cyc_end, parent=None):
        super().__init__(parent)
        self._ax = ax_full
        self._ay = ay_full
        self._n = len(ax_full)
        self._ref_result = None   # subsample_ref_curve output tuple
        self._final_idx = None    # final matched indices (after auto-refit or manual)
        self._final_dist = None

        self._build_ui(cyc_start, cyc_end)
        self._connect_signals()
        self._update_plot()
        QTimer.singleShot(0, self._clear_spinbox_focus)

    def _clear_spinbox_focus(self):
        self.canvas.setFocus()

    # ── UI ────────────────────────────────────────────────────────────────

    def _build_ui(self, cyc_start, cyc_end):
        outer = QHBoxLayout(self)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(splitter)

        # ── Left panel: controls ──────────────────────────────────────────
        ctrl = QWidget()
        ctrl_layout = QVBoxLayout(ctrl)
        ctrl_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Cycle range
        range_grp = QGroupBox("Cycle Range")
        range_form = QFormLayout()

        self.start_spin = QSpinBox()
        self.start_spin.setRange(0, max(0, self._n - 2))
        self.start_spin.setValue(cyc_start)
        range_form.addRow("Start index:", self.start_spin)

        self.end_spin = QSpinBox()
        self.end_spin.setRange(1, self._n)
        self.end_spin.setValue(cyc_end)
        range_form.addRow("End index:", self.end_spin)

        self.n_pts_label = QLabel(f"{cyc_end - cyc_start} pts")
        range_form.addRow("Cycle length:", self.n_pts_label)

        range_grp.setLayout(range_form)
        ctrl_layout.addWidget(range_grp)

        self.apply_range_btn = QPushButton("Apply Range")
        ctrl_layout.addWidget(self.apply_range_btn)

        nav_lbl = QLabel("Arrow keys:\n\u2190\u2192 adjust start \u00b1step\n\u2191\u2193 adjust end \u00b1step")
        nav_lbl.setWordWrap(True)
        nav_lbl.setStyleSheet("color: grey; font-size: 11px;")
        ctrl_layout.addWidget(nav_lbl)

        step_form = QFormLayout()
        self.step_spin = QSpinBox()
        self.step_spin.setRange(1, 100000)
        self.step_spin.setValue(100)
        step_form.addRow("Arrow step:", self.step_spin)
        ctrl_layout.addLayout(step_form)

        # Reference curve parameters
        ref_grp = QGroupBox("Reference Curve")
        ref_form = QFormLayout()

        self.nsub_spin = QSpinBox()
        self.nsub_spin.setRange(5, 3600)
        self.nsub_spin.setValue(40)
        ref_form.addRow("N sub-points:", self.nsub_spin)

        self.sg_win_spin = QSpinBox()
        self.sg_win_spin.setRange(5, 10001)
        self.sg_win_spin.setValue(201)
        self.sg_win_spin.setSingleStep(2)
        ref_form.addRow("Savgol window:", self.sg_win_spin)

        self.sg_order_spin = QSpinBox()
        self.sg_order_spin.setRange(1, 9)
        self.sg_order_spin.setValue(3)
        ref_form.addRow("Savgol order:", self.sg_order_spin)

        self.compute_ref_btn = QPushButton("Compute Ref Curve")
        ref_form.addRow(self.compute_ref_btn)

        self.ref_info = QLabel("")
        self.ref_info.setWordWrap(True)
        ref_form.addRow(self.ref_info)

        ref_grp.setLayout(ref_form)
        ctrl_layout.addWidget(ref_grp)

        # Inspect / Manual
        tool_grp = QGroupBox("Tools")
        tool_lay = QVBoxLayout()

        self.inspect_btn = QPushButton("Inspect Points")
        self.inspect_btn.setEnabled(False)
        tool_lay.addWidget(self.inspect_btn)

        self.manual_btn = QPushButton("Manual Assignment")
        self.manual_btn.setEnabled(False)
        tool_lay.addWidget(self.manual_btn)

        tool_grp.setLayout(tool_lay)
        ctrl_layout.addWidget(tool_grp)

        # Confirm
        self.confirm_btn = QPushButton("Confirm Cycle")
        self.confirm_btn.setEnabled(False)
        self.confirm_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 6px; }")
        ctrl_layout.addWidget(self.confirm_btn)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        ctrl_layout.addWidget(self.status)
        ctrl_layout.addStretch()

        ctrl.setMinimumWidth(260)
        ctrl.setMaximumWidth(330)
        splitter.addWidget(ctrl)

        # ── Centre: anisotropy scatter ────────────────────────────────────
        centre = QWidget()
        centre_layout = QVBoxLayout(centre)
        centre_layout.setContentsMargins(0, 0, 0, 0)

        self.fig = Figure(figsize=(8, 8), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        centre_layout.addWidget(self.toolbar)
        centre_layout.addWidget(self.canvas)
        splitter.addWidget(centre)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

    # ── Signals ───────────────────────────────────────────────────────────

    def _connect_signals(self):
        self.apply_range_btn.clicked.connect(self._on_range_changed)
        self.compute_ref_btn.clicked.connect(self._compute_ref)
        self.inspect_btn.clicked.connect(self._inspect_points)
        self.manual_btn.clicked.connect(self._manual_assignment)
        self.confirm_btn.clicked.connect(self._confirm)

    # ── Public nudge methods (called by MainWindow shortcuts) ─────────

    def _nudge_start(self, direction):
        step = self.step_spin.value()
        new = self.start_spin.value() + direction * step
        new = max(0, min(new, self.end_spin.value() - 1))
        self.start_spin.setValue(new)
        self._on_range_changed()

    def _nudge_end(self, direction):
        step = self.step_spin.value()
        new = self.end_spin.value() + direction * step
        new = max(self.start_spin.value() + 1, min(new, self._n))
        self.end_spin.setValue(new)
        self._on_range_changed()

    # ── Range changed ─────────────────────────────────────────────────────

    def _on_range_changed(self):
        self._ref_result = None
        self._final_idx = None
        self._final_dist = None
        self.confirm_btn.setEnabled(False)
        self.inspect_btn.setEnabled(False)
        self.manual_btn.setEnabled(False)
        self.ref_info.setText("")
        s = self.start_spin.value()
        e = self.end_spin.value()
        self.n_pts_label.setText(f"{e - s} pts")
        self._update_plot()

    # ── Main plot ─────────────────────────────────────────────────────────

    def _update_plot(self):
        s = self.start_spin.value()
        e = self.end_spin.value()

        self.fig.clear()
        ax = self.fig.add_subplot(111)

        # All points (background)
        ax.scatter(self._ax, self._ay, s=0.5, alpha=0.15, c="tab:grey",
                   rasterized=True, label="all", zorder=1)

        # Cycle points (highlighted)
        ax.scatter(self._ax[s:e], self._ay[s:e], s=2, alpha=0.6,
                   c="tab:purple", rasterized=True, label="cycle", zorder=2)

        # Reference curve + matched points
        if self._ref_result is not None:
            _, _, sx_ref, sy_ref, _, _, _ = self._ref_result
            # Close the loop visually
            sx_closed = np.append(sx_ref, sx_ref[0])
            sy_closed = np.append(sy_ref, sy_ref[0])
            ax.plot(sx_closed, sy_closed, 'r-', lw=1.5, alpha=0.8,
                    label="ref curve", zorder=3)

            idx = self._final_idx if self._final_idx is not None else self._ref_result[0]
            ax_cyc = self._ax[s:e]
            ay_cyc = self._ay[s:e]
            valid = idx < len(ax_cyc)
            ax.scatter(ax_cyc[idx[valid]], ay_cyc[idx[valid]],
                       s=20, alpha=0.8, c="tab:red", marker="x",
                       rasterized=True, label="matched", zorder=4)

        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("anis_x")
        ax.set_ylabel("anis_y")
        ax.set_title(f"Cycle [{s}, {e}]  ({e - s} pts)", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.25)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    # ── Compute reference curve + auto monotonic refit ────────────────────

    def _compute_ref(self):
        s = self.start_spin.value()
        e = self.end_spin.value()
        if e - s < 20:
            QMessageBox.warning(self, "Too few points",
                                "Need at least 20 points for reference curve.")
            return

        ax_cyc = self._ax[s:e]
        ay_cyc = self._ay[s:e]

        n_sub = self.nsub_spin.value()
        sg_win = self.sg_win_spin.value()
        sg_order = self.sg_order_spin.value()

        self.status.setText("Computing reference curve\u2026")
        QApplication.processEvents()

        try:
            result = subsample_ref_curve(
                ax_cyc, ay_cyc,
                n_sub=n_sub,
                savgol_window=sg_win,
                savgol_order=sg_order,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Ref curve error", str(exc))
            self.status.setText(f"Error: {exc}")
            return

        self._ref_result = result
        matched_idx = result[0]
        matched_dist = result[1]
        sx_ref, sy_ref = result[2], result[3]
        sg_win_used = result[6]

        # Check monotonicity and auto-refit if needed
        mono, n_violations = is_monotonic(matched_idx)
        info_lines = [f"SG window: {sg_win_used}"]

        if not mono:
            info_lines.append(f"Non-monotonic ({n_violations} violations)")
            info_lines.append("Auto-refitting\u2026")
            self.ref_info.setText("\n".join(info_lines))
            QApplication.processEvents()

            try:
                idx_new, dist_new, n_anchors, n_fixed, n_iters = enforce_monotonic(
                    matched_idx, ax_cyc, ay_cyc, sx_ref, sy_ref)
            except Exception as exc:
                QMessageBox.critical(self, "Monotonic refit error", str(exc))
                self.status.setText(f"Error: {exc}")
                return

            self._final_idx = idx_new
            self._final_dist = dist_new

            mono2, n_viol2 = is_monotonic(idx_new)
            info_lines.append(f"After refit: {'monotonic \u2713' if mono2 else f'{n_viol2} violations remain'}")
            info_lines.append(f"Anchors: {n_anchors}, re-matched: {n_fixed} ({n_iters} iter)")
        else:
            self._final_idx = matched_idx
            self._final_dist = matched_dist
            info_lines.append("Monotonic \u2713")

        n_unique = len(np.unique(self._final_idx))
        info_lines.append(f"Unique pts: {n_unique}/{n_sub}")
        info_lines.append(f"Mean dist: {np.mean(self._final_dist):.4f}")

        self.ref_info.setText("\n".join(info_lines))
        self.confirm_btn.setEnabled(True)
        self.inspect_btn.setEnabled(True)
        self.manual_btn.setEnabled(True)
        self.status.setText("Reference curve computed.")
        self._update_plot()

    # ── Point Inspector ───────────────────────────────────────────────────

    def _inspect_points(self):
        if self._final_idx is None:
            return
        s = self.start_spin.value()
        e = self.end_spin.value()
        dlg = PointInspectorDialog(
            self._ax[s:e], self._ay[s:e],
            self._final_idx,
            self._ref_result[2], self._ref_result[3],  # sx_ref, sy_ref
            parent=self,
        )
        dlg.exec()

    # ── Manual Assignment ─────────────────────────────────────────────────

    def _manual_assignment(self):
        if self._ref_result is None:
            return
        s = self.start_spin.value()
        e = self.end_spin.value()
        dlg = ManualAssignDialog(
            self._ax[s:e], self._ay[s:e],
            self._final_idx.copy() if self._final_idx is not None else self._ref_result[0].copy(),
            self._ref_result[2], self._ref_result[3],
            parent=self,
        )
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._final_idx = dlg.get_indices()
            # Recompute distances
            ax_cyc = self._ax[s:e]
            ay_cyc = self._ay[s:e]
            sx_ref, sy_ref = self._ref_result[2], self._ref_result[3]
            raw_pts = np.column_stack([ax_cyc[self._final_idx], ay_cyc[self._final_idx]])
            ref_pts = np.column_stack([sx_ref, sy_ref])
            self._final_dist = np.linalg.norm(raw_pts - ref_pts, axis=1)

            n_unique = len(np.unique(self._final_idx))
            self.ref_info.setText(
                self.ref_info.text() + f"\nManual: {n_unique} unique pts")
            self.status.setText("Manual assignment applied.")
            self._update_plot()

    # ── Confirm ───────────────────────────────────────────────────────────

    def _confirm(self):
        if self._final_idx is None:
            return
        s = self.start_spin.value()
        e = self.end_spin.value()
        # Build result tuple in same format as subsample_ref_curve
        # but with final (possibly refit/manual) indices
        ref = (
            self._final_idx,
            self._final_dist,
            self._ref_result[2],  # sx_ref
            self._ref_result[3],  # sy_ref
            self._ref_result[4],  # ax_sg
            self._ref_result[5],  # ay_sg
            self._ref_result[6],  # sg_win
        )
        self.confirmed.emit(s, e, ref)
        self.confirm_btn.setEnabled(False)
        self.confirm_btn.setText("\u2713 Confirmed")
        self.status.setText(f"Cycle [{s}, {e}] locked in.")


# ══════════════════════════════════════════════════════════════════════════════
# Point Inspector Dialog
# ══════════════════════════════════════════════════════════════════════════════

class PointInspectorDialog(QDialog):
    """Step through matched points one by one, ordered by raw index."""

    def __init__(self, ax_cyc, ay_cyc, matched_idx, sx_ref, sy_ref, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Point Inspector")
        self.resize(700, 650)

        self._ax = ax_cyc
        self._ay = ay_cyc
        self._sx = sx_ref
        self._sy = sy_ref
        self._n_sub = len(matched_idx)

        # Sort by raw index for inspection
        self._order = np.argsort(matched_idx)
        self._sorted_idx = matched_idx[self._order]
        self._current = 0

        lay = QVBoxLayout(self)

        # Info
        self.info_label = QLabel()
        lay.addWidget(self.info_label)

        # Plot
        self.fig = Figure(figsize=(6, 6), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        lay.addWidget(self.canvas)

        # Navigation
        nav = QHBoxLayout()
        self.prev_btn = QPushButton("\u2190 Prev")
        self.next_btn = QPushButton("Next \u2192")
        self.pos_spin = QSpinBox()
        self.pos_spin.setRange(1, self._n_sub)
        self.pos_spin.setValue(1)
        self.pos_spin.setPrefix("Point ")
        nav.addWidget(self.prev_btn)
        nav.addWidget(self.pos_spin)
        nav.addWidget(self.next_btn)
        lay.addLayout(nav)

        # Close
        close_btn = QPushButton("Close")
        lay.addWidget(close_btn)

        self.prev_btn.clicked.connect(self._prev)
        self.next_btn.clicked.connect(self._next)
        self.pos_spin.valueChanged.connect(self._goto)
        close_btn.clicked.connect(self.accept)

        self._draw()

    def _prev(self):
        if self._current > 0:
            self._current -= 1
            self.pos_spin.setValue(self._current + 1)

    def _next(self):
        if self._current < self._n_sub - 1:
            self._current += 1
            self.pos_spin.setValue(self._current + 1)

    def _goto(self, val):
        self._current = val - 1
        self._draw()

    def _draw(self):
        k = self._current
        ref_k = self._order[k]       # which ref point this is
        raw_k = self._sorted_idx[k]  # which raw index

        self.info_label.setText(
            f"Point {k + 1}/{self._n_sub}  |  "
            f"Ref index: {ref_k}  |  Raw index: {raw_k}")

        self.fig.clear()
        ax = self.fig.add_subplot(111)

        # All cycle points
        ax.scatter(self._ax, self._ay, s=1, alpha=0.2, c="tab:grey",
                   rasterized=True, zorder=1)
        # Ref curve
        ax.plot(self._sx, self._sy, 'r-', lw=1, alpha=0.5, zorder=2)

        # All matched points so far (up to current)
        prev_raw = self._sorted_idx[:k + 1]
        ax.scatter(self._ax[prev_raw], self._ay[prev_raw],
                   s=15, alpha=0.5, c="tab:blue", marker="o", zorder=3)

        # Current point highlighted
        ax.scatter(self._ax[raw_k], self._ay[raw_k],
                   s=80, c="tab:red", marker="*", zorder=5,
                   edgecolors="black", linewidths=0.5)

        # Corresponding ref point
        ax.scatter(self._sx[ref_k], self._sy[ref_k],
                   s=40, c="tab:green", marker="D", zorder=4,
                   edgecolors="black", linewidths=0.5, label="ref pt")

        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect("equal")
        ax.set_title(f"Point {k + 1}: ref[{ref_k}] \u2192 raw[{raw_k}]")
        ax.grid(alpha=0.25)
        self.fig.tight_layout()
        self.canvas.draw_idle()

        self.prev_btn.setEnabled(k > 0)
        self.next_btn.setEnabled(k < self._n_sub - 1)


# ══════════════════════════════════════════════════════════════════════════════
# Manual Assignment Dialog
# ══════════════════════════════════════════════════════════════════════════════

class ManualAssignDialog(QDialog):
    """Manually assign each ref point to a raw index."""

    def __init__(self, ax_cyc, ay_cyc, initial_idx, sx_ref, sy_ref, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manual Point Assignment")
        self.resize(800, 700)

        self._ax = ax_cyc
        self._ay = ay_cyc
        self._sx = sx_ref
        self._sy = sy_ref
        self._n_sub = len(initial_idx)
        self._n_raw = len(ax_cyc)
        self._idx = initial_idx.copy()
        self._current = 0

        lay = QVBoxLayout(self)

        self.info_label = QLabel()
        lay.addWidget(self.info_label)

        # Plot
        self.fig = Figure(figsize=(7, 7), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        lay.addWidget(self.canvas)

        # Controls
        ctrl = QHBoxLayout()
        self.prev_btn = QPushButton("\u2190 Prev")
        self.next_btn = QPushButton("Next \u2192")

        self.idx_spin = QSpinBox()
        self.idx_spin.setRange(0, self._n_raw - 1)
        self.idx_spin.setValue(self._idx[0])
        self.idx_spin.setPrefix("Raw idx: ")

        self.set_btn = QPushButton("Set && Next")
        self.set_btn.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; font-weight: bold; }")

        ctrl.addWidget(self.prev_btn)
        ctrl.addWidget(self.idx_spin)
        ctrl.addWidget(self.set_btn)
        ctrl.addWidget(self.next_btn)
        lay.addLayout(ctrl)

        # Accept / Cancel
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        lay.addWidget(btns)

        self.prev_btn.clicked.connect(self._prev)
        self.next_btn.clicked.connect(self._next)
        self.set_btn.clicked.connect(self._set_and_next)
        self.idx_spin.valueChanged.connect(self._draw)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

        self._draw()

    def _prev(self):
        if self._current > 0:
            self._current -= 1
            self.idx_spin.setValue(self._idx[self._current])
            self._draw()

    def _next(self):
        if self._current < self._n_sub - 1:
            self._current += 1
            self.idx_spin.setValue(self._idx[self._current])
            self._draw()

    def _set_and_next(self):
        self._idx[self._current] = self.idx_spin.value()
        if self._current < self._n_sub - 1:
            self._current += 1
            self.idx_spin.setValue(self._idx[self._current])
        self._draw()

    def _draw(self, _value=None):
        k = self._current
        raw_k = self._idx[k]          # committed value
        preview_k = self.idx_spin.value()  # spinbox value (may differ)

        if preview_k != raw_k:
            self.info_label.setText(
                f"Ref point {k + 1}/{self._n_sub}  |  "
                f"Committed: {raw_k}  |  Preview: {preview_k}  |  "
                f"Press 'Set & Next' to apply")
        else:
            self.info_label.setText(
                f"Ref point {k + 1}/{self._n_sub}  |  "
                f"Raw index: {raw_k}  |  "
                f"Use arrows/spinbox then 'Set & Next'")

        self.fig.clear()
        ax = self.fig.add_subplot(111)

        # Cycle points
        ax.scatter(self._ax, self._ay, s=1, alpha=0.2, c="tab:grey",
                   rasterized=True, zorder=1)

        # Ref curve
        ax.plot(self._sx, self._sy, 'r-', lw=1, alpha=0.4, zorder=2)
        # Highlight current ref point
        ax.scatter(self._sx[k], self._sy[k], s=80, c="tab:green",
                   marker="D", zorder=5, edgecolors="black", linewidths=0.5,
                   label=f"ref[{k}]")

        # Already-assigned points
        for j in range(self._n_sub):
            if j == k:
                continue
            color = "tab:blue" if j < k else "lightblue"
            ax.scatter(self._ax[self._idx[j]], self._ay[self._idx[j]],
                       s=10, alpha=0.4, c=color, zorder=3)

        # Committed assignment (dimmed if preview differs)
        committed_alpha = 0.3 if preview_k != raw_k else 0.8
        ax.scatter(self._ax[raw_k], self._ay[raw_k],
                   s=60, c="tab:red", marker="*", zorder=6,
                   edgecolors="black", linewidths=0.5,
                   alpha=committed_alpha,
                   label=f"set[{raw_k}]")

        # Preview point (only when different from committed)
        if preview_k != raw_k:
            ax.scatter(self._ax[preview_k], self._ay[preview_k],
                       s=120, c="tab:orange", marker="*", zorder=7,
                       edgecolors="black", linewidths=0.5,
                       label=f"preview[{preview_k}]")
            # Line from ref to preview
            ax.plot([self._sx[k], self._ax[preview_k]],
                    [self._sy[k], self._ay[preview_k]],
                    color="tab:orange", ls="--", lw=1, alpha=0.7, zorder=4)
        else:
            # Line from ref to committed
            ax.plot([self._sx[k], self._ax[raw_k]],
                    [self._sy[k], self._ay[raw_k]],
                    'k--', lw=0.8, alpha=0.5, zorder=4)

        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect("equal")
        title = f"Assign ref[{k}] \u2192 raw[{raw_k}]"
        if preview_k != raw_k:
            title += f"  (preview: {preview_k})"
        ax.set_title(title)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.25)
        self.fig.tight_layout()
        self.canvas.draw_idle()

        self.prev_btn.setEnabled(k > 0)
        self.next_btn.setEnabled(k < self._n_sub - 1)

    def get_indices(self):
        return self._idx
