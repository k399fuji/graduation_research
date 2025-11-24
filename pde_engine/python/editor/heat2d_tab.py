# editor/heat2d_tab.py
from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QHBoxLayout,
    QLabel, QDoubleSpinBox, QSpinBox, QComboBox,
    QLineEdit, QPushButton, QTextEdit, QApplication,
)

from backend.heat2d_backend import Heat2DConfig, Heat2DResult, run_heat2d
try:
    import heat2d_cpp
except ImportError:
    heat2d_cpp = None


class Heat2DCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

        self.im = None
        self.cbar = None
        self.apply_dark_style()

    def apply_dark_style(self):
        self.fig.patch.set_facecolor("#2b2b2b")
        self.ax.set_facecolor("#2b2b2b")

        for spine in self.ax.spines.values():
            spine.set_color("white")
        self.ax.tick_params(axis="both", colors="white")
        self.ax.xaxis.label.set_color("white")
        self.ax.yaxis.label.set_color("white")
        self.ax.title.set_color("white")

        if self.cbar is not None:
            self.cbar.outline.set_edgecolor("white")
            self.cbar.ax.tick_params(colors="white")
            for label in self.cbar.ax.get_yticklabels():
                label.set_color("white")

    def plot_solution(self, x, y, u):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        u = np.asarray(u, dtype=float)
        if u.ndim == 1:
            u = u.reshape(len(y), len(x))

        self.ax.clear()
        self.apply_dark_style()

        extent = [x.min(), x.max(), y.min(), y.max()]
        self.im = self.ax.imshow(
            u,
            origin="lower",
            extent=extent,
            aspect="equal",
        )
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_title("Heat2D: u(x, y, T_final)")

        if self.cbar is not None:
            self.cbar.remove()
            self.cbar = None
        self.cbar = self.fig.colorbar(self.im, ax=self.ax)
        self.apply_dark_style()
        self.draw_idle()


class Heat2DTab(QWidget):
    """
    C++ heat2d_cpp.solve_heat_2d を呼んで 2D ヒートマップを描画するタブ。
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        main_layout = QVBoxLayout(self)

        form_layout = QFormLayout()

        self.spin_Lx = QDoubleSpinBox()
        self.spin_Lx.setRange(0.1, 10.0)
        self.spin_Lx.setSingleStep(0.1)
        self.spin_Lx.setValue(1.0)

        self.spin_Ly = QDoubleSpinBox()
        self.spin_Ly.setRange(0.1, 10.0)
        self.spin_Ly.setSingleStep(0.1)
        self.spin_Ly.setValue(1.0)

        self.spin_Nx = QSpinBox()
        self.spin_Nx.setRange(10, 301)
        self.spin_Nx.setSingleStep(10)
        self.spin_Nx.setValue(101)

        self.spin_Ny = QSpinBox()
        self.spin_Ny.setRange(10, 301)
        self.spin_Ny.setSingleStep(10)
        self.spin_Ny.setValue(101)

        self.spin_alpha = QDoubleSpinBox()
        self.spin_alpha.setDecimals(5)
        self.spin_alpha.setRange(1e-5, 1.0)
        self.spin_alpha.setSingleStep(0.001)
        self.spin_alpha.setValue(0.01)

        self.spin_dt = QDoubleSpinBox()
        self.spin_dt.setDecimals(6)
        self.spin_dt.setRange(1e-6, 1.0)
        self.spin_dt.setSingleStep(1e-4)
        self.spin_dt.setValue(1e-4)

        self.spin_Tfinal = QDoubleSpinBox()
        self.spin_Tfinal.setRange(0.001, 10.0)
        self.spin_Tfinal.setSingleStep(0.1)
        self.spin_Tfinal.setValue(0.1)

        self.combo_ic_type = QComboBox()
        self.combo_ic_type.addItems(["gaussian", "sinexy"])

        self.spin_gaussian_kx = QDoubleSpinBox()
        self.spin_gaussian_kx.setRange(1.0, 500.0)
        self.spin_gaussian_kx.setSingleStep(10.0)
        self.spin_gaussian_kx.setValue(100.0)

        self.spin_gaussian_ky = QDoubleSpinBox()
        self.spin_gaussian_ky.setRange(1.0, 500.0)
        self.spin_gaussian_ky.setSingleStep(10.0)
        self.spin_gaussian_ky.setValue(100.0)

        self.edit_tag = QLineEdit("heat2d")

        form_layout.addRow("Nx (grid in x)", self.spin_Nx)
        form_layout.addRow("Ny (grid in y)", self.spin_Ny)
        form_layout.addRow("Lx (length in x)", self.spin_Lx)
        form_layout.addRow("Ly (length in y)", self.spin_Ly)
        form_layout.addRow("alpha (diffusivity)", self.spin_alpha)
        form_layout.addRow("dt (time step)", self.spin_dt)
        form_layout.addRow("T_final", self.spin_Tfinal)
        form_layout.addRow("IC type", self.combo_ic_type)
        form_layout.addRow("Gaussian kx", self.spin_gaussian_kx)
        form_layout.addRow("Gaussian ky", self.spin_gaussian_ky)
        form_layout.addRow("tag", self.edit_tag)

        main_layout.addLayout(form_layout)

        button_layout = QHBoxLayout()
        self.run_button = QPushButton("Run 2D Heat Simulation")
        self.run_button.clicked.connect(self.on_run_clicked)
        button_layout.addWidget(self.run_button)

        self.run_id_label = QLabel("run_id: (not run yet)")
        button_layout.addWidget(self.run_id_label)

        self.info_label = QLabel("Ready.")
        button_layout.addWidget(self.info_label)

        button_layout.addStretch()
        main_layout.addLayout(button_layout)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        main_layout.addWidget(QLabel("Log:"))
        main_layout.addWidget(self.log_text)

        self.canvas = Heat2DCanvas(self, width=5, height=4, dpi=100)
        main_layout.addWidget(self.canvas, 1)

    def append_log(self, text: str):
        self.log_text.append(text)

    @Slot()
    def on_run_clicked(self):
        self.info_label.setText("Running 2D Heat simulation...")
        self.run_button.setEnabled(False)
        QApplication.processEvents()

        cfg = Heat2DConfig(
            Nx_cpp=int(self.spin_Nx.value()),
            Ny_cpp=int(self.spin_Ny.value()),
            Lx=float(self.spin_Lx.value()),
            Ly=float(self.spin_Ly.value()),
            alpha=float(self.spin_alpha.value()),
            dt_cpp=float(self.spin_dt.value()),
            T_final=float(self.spin_Tfinal.value()),
            ic_type=self.combo_ic_type.currentText().lower(),
            gaussian_kx=float(self.spin_gaussian_kx.value()),
            gaussian_ky=float(self.spin_gaussian_ky.value()),
            tag=self.edit_tag.text().strip() or "heat2d",
        )

        self.append_log(f"Config: {cfg}")

        try:
            result: Heat2DResult = run_heat2d(cfg)
        except Exception as e:
            self.append_log(f"Error: {e!r}")
            self.info_label.setText("Error during Heat2D simulation.")
            self.run_button.setEnabled(True)
            return

        self.canvas.plot_solution(result.x, result.y, result.u)
        self.run_id_label.setText(f"run_id: {result.run_id}")
        self.info_label.setText("Done.")

        if result.config_json_path is not None:
            self.append_log(f"Config JSON: {result.config_json_path}")
        if result.eval_json_path is not None:
            self.append_log(f"Eval JSON: {result.eval_json_path}")
        if result.summary_json_path is not None:
            self.append_log(f"Summary JSON: {result.summary_json_path}")

        self.run_button.setEnabled(True)
