# editor/logs_panel.py
from __future__ import annotations

import json

from PySide6.QtCore import Qt, Slot, Signal
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QListWidget, QPushButton, QTextEdit,
)
from PySide6.QtWidgets import QDockWidget  # プラグイン用

from backend import log_utils
from editor.plugin_core import EditorPlugin
from editor.viewport_1d import MatplotlibCanvas


class LogsPage(QWidget):
    runSelected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        main_layout = QHBoxLayout(self)

        # 左：run 一覧
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("Runs"))
        self.run_list = QListWidget()
        self.run_list.currentItemChanged.connect(self.on_run_selected)
        self.run_list.setMaximumWidth(185)
        left_layout.addWidget(self.run_list)

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_runs)
        left_layout.addWidget(self.refresh_button)

        # 右：Config / Eval / Loss
        right_layout = QVBoxLayout()

        right_layout.addWidget(QLabel("Config"))
        self.config_text = QTextEdit()
        self.config_text.setReadOnly(True)
        right_layout.addWidget(self.config_text, 2)

        right_layout.addWidget(QLabel("Eval"))
        self.eval_label = QLabel("(no eval)")
        right_layout.addWidget(self.eval_label)

        right_layout.addWidget(QLabel("Loss curves"))
        self.canvas = MatplotlibCanvas(self, width=5, height=3, dpi=100)
        right_layout.addWidget(self.canvas, 4)

        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 2)

        self.refresh_runs()

    def refresh_runs(self):
        self.run_list.clear()
        self._runs = log_utils.list_runs()
        if not self._runs:
            self.run_list.addItem("(no runs)")
            self.run_list.setEnabled(False)
            return

        self.run_list.setEnabled(True)
        for info in self._runs:
            self.run_list.addItem(info.run_id)

    @Slot("QListWidgetItem*", "QListWidgetItem*")
    def on_run_selected(self, current, previous):
        if current is None:
            return
        run_id = current.text()
        if run_id == "(no runs)":
            return

        # Config / Summary
        summary = log_utils.load_summary(run_id)
        if summary is not None:
            self.config_text.setPlainText(
                json.dumps(summary, indent=2, ensure_ascii=False)
            )
        else:
            cfg = log_utils.load_config(run_id)
            if cfg is None:
                self.config_text.setPlainText("(no config.json)")
            else:
                self.config_text.setPlainText(
                    json.dumps(cfg, indent=2, ensure_ascii=False)
                )

        # Eval / metrics
        ev = log_utils.load_eval(run_id)
        if ev is None:
            if summary is not None:
                metrics = summary.get("results", {}).get("metrics", {})
                l2 = metrics.get("L2_error", None)
                linf = metrics.get("Linf_error", None)
                if l2 is not None and linf is not None:
                    self.eval_label.setText(f"L2 = {l2:.3e},  L∞ = {linf:.3e}")
                else:
                    self.eval_label.setText("No eval.json")
            else:
                self.eval_label.setText("No eval.json")
        else:
            l2 = ev.get("L2_error", None)
            linf = ev.get("Linf_error", None)
            if l2 is not None and linf is not None:
                self.eval_label.setText(f"L2 = {l2:.3e},  L∞ = {linf:.3e}")
            else:
                self.eval_label.setText(json.dumps(ev, ensure_ascii=False))

        # Loss 曲線
        rows = log_utils.load_loss_csv(run_id)
        ax = self.canvas.axes
        ax.clear()

        if not rows:
            ax.text(0.5, 0.5, "No CSV log", ha="center", va="center")
        else:
            epochs = [int(r["epoch"]) for r in rows]

            def get_series(col_name: str, fallback: str | None = None):
                if col_name in rows[0]:
                    return [float(r[col_name]) for r in rows]
                if fallback and fallback in rows[0]:
                    return [float(r[fallback]) for r in rows]
                return None

            loss_total = get_series("loss_total", "loss")
            loss_pde = get_series("loss_pde")
            loss_ic = get_series("loss_ic")
            loss_bc = get_series("loss_bc")

            line_width = getattr(log_utils, "LINE_WIDTH", 1.5)

            if loss_total is not None:
                ax.plot(epochs, loss_total, label="loss_total",
                        linewidth=line_width)
            if loss_pde is not None:
                ax.plot(epochs, loss_pde, label="loss_pde",
                        linewidth=line_width)
            if loss_ic is not None:
                ax.plot(epochs, loss_ic, label="loss_ic",
                        linewidth=line_width)
            if loss_bc is not None:
                ax.plot(epochs, loss_bc, label="loss_bc",
                        linewidth=line_width)

            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            ax.set_yscale("log")
            ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0))

        self.canvas.fig.subplots_adjust(bottom=0.18)
        self.canvas.apply_dark_style()
        self.canvas.draw()

        self.runSelected.emit(run_id)


class LogsPlugin(EditorPlugin):
    plugin_id = "logs"
    display_name = "Logs"

    def create_dock(self, main_window):
        from PySide6.QtCore import Qt

        widget = LogsPage(main_window)
        if hasattr(main_window, "on_log_run_selected"):
            widget.runSelected.connect(main_window.on_log_run_selected)

        dock = QDockWidget(self.display_name, main_window)
        dock.setWidget(widget)
        dock.setObjectName(self.plugin_id)
        dock.setAllowedAreas(
            Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea | Qt.BottomDockWidgetArea
        )
        main_window.addDockWidget(Qt.RightDockWidgetArea, dock)
        dock.hide()

        if hasattr(main_window, "view_menu"):
            main_window.view_menu.addAction(dock.toggleViewAction())

        return dock
