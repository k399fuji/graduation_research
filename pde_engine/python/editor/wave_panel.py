from __future__ import annotations

from PySide6.QtWidgets import (
    QFormLayout,
    QDoubleSpinBox,
    QSpinBox,
    QLineEdit,
    QComboBox,
)


from backend.wave_pinn_backend import WavePINNConfig
from editor.workers import WavePINNWorker, BasePINNWorker
from editor.pinn_base import BasePINNPage

class WavePINNPage(BasePINNPage):
    experiment_name = "Wave"

    def build_form(self, form_layout: QFormLayout) -> None:
        # PDE params
        self.spin_L = QDoubleSpinBox()
        self.spin_L.setRange(0.1, 10.0)
        self.spin_L.setValue(1.0)
        self.spin_L.setSingleStep(0.1)

        self.spin_c = QDoubleSpinBox()
        self.spin_c.setRange(0.1, 10.0)
        self.spin_c.setValue(1.0)
        self.spin_c.setSingleStep(0.1)

        self.spin_T = QDoubleSpinBox()
        self.spin_T.setRange(0.01, 10.0)
        self.spin_T.setValue(1.0)
        self.spin_T.setSingleStep(0.1)

        # PINN params
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(100, 100000)
        self.spin_epochs.setValue(5000)
        self.spin_epochs.setSingleStep(100)

        self.spin_hidden = QSpinBox()
        self.spin_hidden.setRange(8, 512)
        self.spin_hidden.setValue(64)
        self.spin_hidden.setSingleStep(8)

        self.spin_layers = QSpinBox()
        self.spin_layers.setRange(2, 10)
        self.spin_layers.setValue(4)

        self.spin_Nr = QSpinBox()
        self.spin_Nr.setRange(100, 50000)
        self.spin_Nr.setValue(2000)
        self.spin_Nr.setSingleStep(100)

        self.spin_Nic_u = QSpinBox()
        self.spin_Nic_u.setRange(10, 10000)
        self.spin_Nic_u.setValue(200)
        self.spin_Nic_u.setSingleStep(10)

        self.spin_Nic_ut = QSpinBox()
        self.spin_Nic_ut.setRange(10, 10000)
        self.spin_Nic_ut.setValue(200)
        self.spin_Nic_ut.setSingleStep(10)

        self.spin_Nbc = QSpinBox()
        self.spin_Nbc.setRange(10, 10000)
        self.spin_Nbc.setValue(200)
        self.spin_Nbc.setSingleStep(10)

        # IC / weights
        self.combo_ic_type = QComboBox()
        self.combo_ic_type.addItems(["gaussian", "sine", "twopeaks"])

        self.spin_gaussian_k = QDoubleSpinBox()
        self.spin_gaussian_k.setRange(1.0, 500.0)
        self.spin_gaussian_k.setValue(100.0)
        self.spin_gaussian_k.setSingleStep(10.0)

        self.edit_tag = QLineEdit("qt")

        self.spin_w_pde = QDoubleSpinBox()
        self.spin_w_pde.setRange(0.0, 100.0)
        self.spin_w_pde.setSingleStep(0.1)
        self.spin_w_pde.setValue(1.0)

        self.spin_w_ic_u = QDoubleSpinBox()
        self.spin_w_ic_u.setRange(0.0, 100.0)
        self.spin_w_ic_u.setSingleStep(0.1)
        self.spin_w_ic_u.setValue(1.0)

        self.spin_w_ic_ut = QDoubleSpinBox()
        self.spin_w_ic_ut.setRange(0.0, 100.0)
        self.spin_w_ic_ut.setSingleStep(0.1)
        self.spin_w_ic_ut.setValue(1.0)

        self.spin_w_bc = QDoubleSpinBox()
        self.spin_w_bc.setRange(0.0, 100.0)
        self.spin_w_bc.setSingleStep(0.1)
        self.spin_w_bc.setValue(1.0)

        # layout
        form_layout.addRow("IC type", self.combo_ic_type)
        form_layout.addRow("Gaussian k", self.spin_gaussian_k)

        form_layout.addRow("L (domain length)", self.spin_L)
        form_layout.addRow("c (wave speed)", self.spin_c)
        form_layout.addRow("T_final", self.spin_T)

        form_layout.addRow("epochs", self.spin_epochs)
        form_layout.addRow("hidden_dim", self.spin_hidden)
        form_layout.addRow("num_layers", self.spin_layers)

        form_layout.addRow("N_r (residual points)", self.spin_Nr)
        form_layout.addRow("N_ic_u (IC u points)", self.spin_Nic_u)
        form_layout.addRow("N_ic_ut (IC ut points)", self.spin_Nic_ut)
        form_layout.addRow("N_bc (BC points)", self.spin_Nbc)

        form_layout.addRow("w_pde", self.spin_w_pde)
        form_layout.addRow("w_ic_u", self.spin_w_ic_u)
        form_layout.addRow("w_ic_ut", self.spin_w_ic_ut)
        form_layout.addRow("w_bc", self.spin_w_bc)

        form_layout.addRow("tag", self.edit_tag)

    def get_config(self) -> WavePINNConfig:
        return WavePINNConfig(
            L=float(self.spin_L.value()),
            c=float(self.spin_c.value()),
            T_final=float(self.spin_T.value()),
            epochs=int(self.spin_epochs.value()),
            hidden_dim=int(self.spin_hidden.value()),
            num_layers=int(self.spin_layers.value()),
            N_r=int(self.spin_Nr.value()),
            N_ic_u=int(self.spin_Nic_u.value()),
            N_ic_ut=int(self.spin_Nic_ut.value()),
            N_bc=int(self.spin_Nbc.value()),
            w_pde=float(self.spin_w_pde.value()),
            w_ic_u=float(self.spin_w_ic_u.value()),
            w_ic_ut=float(self.spin_w_ic_ut.value()),
            w_bc=float(self.spin_w_bc.value()),
            ic_type=self.combo_ic_type.currentText(),
            gaussian_k=float(self.spin_gaussian_k.value()),
            tag=self.edit_tag.text().strip() or "qt",
        )

    def create_worker(self, cfg: WavePINNConfig) -> BasePINNWorker:
        return WavePINNWorker(cfg, self)

class WavePINNPanel(WavePINNPage):
    """
    Wave PINN 用のパネルクラス。

    こちらも今は WavePINNPage の薄いラッパ。
    将来的にファイル分割・機能拡張するときのための名前。
    """
    pass

__all__ = ["WavePINNPage", "WavePINNPanel"]