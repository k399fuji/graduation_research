from __future__ import annotations

from PySide6.QtWidgets import (
    QFormLayout,
    QDoubleSpinBox,
    QSpinBox,
    QLineEdit,
    QComboBox,
)


from backend.heat_pinn_backend import HeatPINNConfig
from editor.pinn_base import BasePINNPage
from editor.workers import HeatPINNWorker, BasePINNWorker


class HeatPINNPage(BasePINNPage):
    experiment_name = "Heat"

    def build_form(self, form_layout: QFormLayout) -> None:
        # ==== PDE params ====
        self.spin_L = QDoubleSpinBox()
        self.spin_L.setRange(0.1, 10.0)
        self.spin_L.setValue(1.0)
        self.spin_L.setSingleStep(0.1)

        self.spin_alpha = QDoubleSpinBox()
        self.spin_alpha.setDecimals(5)
        self.spin_alpha.setRange(1e-5, 1.0)
        self.spin_alpha.setValue(0.01)
        self.spin_alpha.setSingleStep(0.001)

        self.spin_T = QDoubleSpinBox()
        self.spin_T.setRange(0.01, 10.0)
        self.spin_T.setValue(0.4)
        self.spin_T.setSingleStep(0.1)

        # ==== PINN params ====
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
        self.spin_Nr.setValue(1000)
        self.spin_Nr.setSingleStep(100)

        self.spin_Nic = QSpinBox()
        self.spin_Nic.setRange(10, 10000)
        self.spin_Nic.setValue(200)
        self.spin_Nic.setSingleStep(10)

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

        self.spin_w_pde = QDoubleSpinBox()
        self.spin_w_pde.setRange(0.0, 100.0)
        self.spin_w_pde.setSingleStep(0.1)
        self.spin_w_pde.setValue(1.0)

        self.spin_w_ic = QDoubleSpinBox()
        self.spin_w_ic.setRange(0.0, 100.0)
        self.spin_w_ic.setSingleStep(0.1)
        self.spin_w_ic.setValue(1.0)

        self.spin_w_bc = QDoubleSpinBox()
        self.spin_w_bc.setRange(0.0, 100.0)
        self.spin_w_bc.setSingleStep(0.1)
        self.spin_w_bc.setValue(1.0)

        self.edit_tag = QLineEdit("qt")

        # ---- layout ----
        form_layout.addRow("IC type", self.combo_ic_type)
        form_layout.addRow("Gaussian k", self.spin_gaussian_k)
        form_layout.addRow("L (domain length)", self.spin_L)
        form_layout.addRow("alpha (diffusivity)", self.spin_alpha)
        form_layout.addRow("T_final", self.spin_T)
        form_layout.addRow("epochs", self.spin_epochs)
        form_layout.addRow("hidden_dim", self.spin_hidden)
        form_layout.addRow("num_layers", self.spin_layers)
        form_layout.addRow("N_r (residual points)", self.spin_Nr)
        form_layout.addRow("N_ic (IC points)", self.spin_Nic)
        form_layout.addRow("N_bc (BC points)", self.spin_Nbc)
        form_layout.addRow("w_pde (PDE weight)", self.spin_w_pde)
        form_layout.addRow("w_ic (IC weight)", self.spin_w_ic)
        form_layout.addRow("w_bc (BC weight)", self.spin_w_bc)
        form_layout.addRow("tag", self.edit_tag)

    def get_config(self) -> HeatPINNConfig:
        return HeatPINNConfig(
            L=float(self.spin_L.value()),
            alpha=float(self.spin_alpha.value()),
            T_final=float(self.spin_T.value()),
            epochs=int(self.spin_epochs.value()),
            hidden_dim=int(self.spin_hidden.value()),
            num_layers=int(self.spin_layers.value()),
            N_r=int(self.spin_Nr.value()),
            N_ic=int(self.spin_Nic.value()),
            N_bc=int(self.spin_Nbc.value()),
            w_pde=float(self.spin_w_pde.value()),
            w_ic=float(self.spin_w_ic.value()),
            w_bc=float(self.spin_w_bc.value()),
            ic_type=self.combo_ic_type.currentText(),
            gaussian_k=float(self.spin_gaussian_k.value()),
            tag=self.edit_tag.text().strip() or "qt",
        )

    def create_worker(self, cfg: HeatPINNConfig) -> BasePINNWorker:
        return HeatPINNWorker(cfg, self)

class HeatPINNPanel(HeatPINNPage):
    """
    Heat PINN 用のパネルクラス。

    現時点では HeatPINNPage をそのまま継承した薄いラッパ。
    後で heat_panel.py などに分離したり、
    追加のメソッドを足したりするときの足場にする。
    """
    pass

__all__ = ["HeatPINNPage", "HeatPINNPanel"]
