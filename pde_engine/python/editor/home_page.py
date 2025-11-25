# editor/home_page.py
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton,
)


class HomePage(QWidget):
    openPinnRequested = Signal()
    openHeat2DRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)

        title = QLabel("PDE Simulation & PINN Playground")
        title.setAlignment(Qt.AlignCenter)
        font = title.font()
        font.setPointSize(font.pointSize() + 4)
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title)

        layout.addSpacing(10)

        desc = QLabel(
            "左のボタンから機能を選んでください。\n"
            "- PINN Editor: 1D Heat / Wave の PINN 学習・可視化\n"
            "- 2D Heat Simulation: 2次元熱方程式の数値シミュレーション"
        )
        desc.setAlignment(Qt.AlignCenter)
        desc.setWordWrap(True)
        layout.addWidget(desc)

        layout.addSpacing(20)

        btn_layout = QVBoxLayout()
        self.btn_open_pinn = QPushButton("Open PINN Editor")
        self.btn_open_pinn.setObjectName("btnOpenPinn")
        self.btn_open_pinn.clicked.connect(self.openPinnRequested)
        btn_layout.addWidget(self.btn_open_pinn)

        self.btn_open_heat2d = QPushButton("Open 2D Heat Simulation")
        self.btn_open_heat2d.setObjectName("btnOpenHeat2D")
        self.btn_open_heat2d.clicked.connect(self.openHeat2DRequested)
        btn_layout.addWidget(self.btn_open_heat2d)

        layout.addLayout(btn_layout)
        layout.addStretch()
