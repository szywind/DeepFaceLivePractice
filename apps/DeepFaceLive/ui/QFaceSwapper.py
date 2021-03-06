from pathlib import Path

from localization import L
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from resources.gfx import QXImageDB
from xlib import qt as lib_qt

from ..backend import FaceSwapper
from .widgets.QBackendPanel import QBackendPanel
from .widgets.QCheckBoxCSWFlag import QCheckBoxCSWFlag
from .widgets.QComboBoxCSWDynamicSingleSwitch import \
    QComboBoxCSWDynamicSingleSwitch
from .widgets.QErrorCSWError import QErrorCSWError
from .widgets.QLabelPopupInfo import QLabelPopupInfo
from .widgets.QLabelPopupInfoCSWInfoLabel import QLabelPopupInfoCSWInfoLabel
from .widgets.QProgressBarCSWProgress import QProgressBarCSWProgress
from .widgets.QSliderCSWNumber import QSliderCSWNumber
from .widgets.QSpinBoxCSWNumber import QSpinBoxCSWNumber


class QFaceSwapper(QBackendPanel):
    def __init__(self, backend : FaceSwapper, dfm_models_path : Path):
        self._dfm_models_path = dfm_models_path

        cs = backend.get_control_sheet()

        btn_open_folder = self.btn_open_folder = lib_qt.QXPushButton(image = QXImageDB.eye_outline('light gray'), tooltip_text='Reveal in Explorer', released=self._btn_open_folder_released, fixed_size=(24,22) )

        q_model_label = QLabelPopupInfo(label=L('@QFaceSwapper.model'), popup_info_text=L('@QFaceSwapper.help.model') )
        q_model       = QComboBoxCSWDynamicSingleSwitch(cs.model, reflect_state_widgets=[q_model_label, btn_open_folder])

        q_model_dl_error = self._q_model_dl_error = QErrorCSWError(cs.model_dl_error)
        q_model_dl_progress = self._q_model_dl_progress = QProgressBarCSWProgress(cs.model_dl_progress)

        q_model_info_label = self._q_model_info_label = QLabelPopupInfoCSWInfoLabel(cs.model_info_label)

        q_device_label  = QLabelPopupInfo(label=L('@QFaceSwapper.device'), popup_info_text=L('@QFaceSwapper.help.device') )
        q_device        = QComboBoxCSWDynamicSingleSwitch(cs.device, reflect_state_widgets=[q_device_label])

        q_face_id_label  = QLabelPopupInfo(label=L('@QFaceSwapper.face_id'), popup_info_text=L('@QFaceSwapper.help.face_id') )
        q_face_id        = QSpinBoxCSWNumber(cs.face_id, reflect_state_widgets=[q_face_id_label])

        q_morph_factor_label = QLabelPopupInfo(label=L('@QFaceSwapper.morph_factor'), popup_info_text=L('@QFaceSwapper.help.morph_factor') )
        q_morph_factor       = QSliderCSWNumber(cs.morph_factor, reflect_state_widgets=[q_morph_factor_label])

        q_sharpen_amount_label = QLabelPopupInfo(label=L('@QFaceSwapper.presharpen_amount'), popup_info_text=L('@QFaceSwapper.help.presharpen_amount') )
        q_sharpen_amount       = QSliderCSWNumber(cs.presharpen_amount, reflect_state_widgets=[q_sharpen_amount_label])

        q_pre_gamma_label = QLabelPopupInfo(label=L('@QFaceSwapper.pregamma'), popup_info_text=L('@QFaceSwapper.help.pregamma') )

        q_pre_gamma_red   = QSpinBoxCSWNumber(cs.pre_gamma_red, reflect_state_widgets=[q_pre_gamma_label])
        q_pre_gamma_green = QSpinBoxCSWNumber(cs.pre_gamma_green )
        q_pre_gamma_blue  = QSpinBoxCSWNumber(cs.pre_gamma_blue)

        q_two_pass_label = QLabelPopupInfo(label=L('@QFaceSwapper.two_pass'), popup_info_text=L('@QFaceSwapper.help.two_pass') )
        q_two_pass       = QCheckBoxCSWFlag(cs.two_pass, reflect_state_widgets=[q_two_pass_label])

        grid_l = lib_qt.QXGridLayout(spacing=5)
        row = 0
        grid_l.addWidget(q_model_label, row, 0, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter  )
        grid_l.addLayout(lib_qt.QXHBoxLayout([q_model, 2, btn_open_folder, 2, q_model_info_label]), row, 1, alignment=Qt.AlignmentFlag.AlignLeft )
        row += 1
        grid_l.addWidget(q_model_dl_progress, row, 0, 1, 2 )
        row += 1
        grid_l.addWidget(q_model_dl_error, row, 0, 1, 2 )
        row += 1
        grid_l.addWidget(q_device_label, row, 0, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter  )
        grid_l.addWidget(q_device, row, 1, alignment=Qt.AlignmentFlag.AlignLeft )
        row += 1
        grid_l.addWidget(q_face_id_label, row, 0, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter  )
        grid_l.addWidget(q_face_id, row, 1, alignment=Qt.AlignmentFlag.AlignLeft )
        row += 1
        grid_l.addWidget(q_morph_factor_label, row, 0, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter  )
        grid_l.addWidget(q_morph_factor, row, 1)
        row += 1
        grid_l.addWidget(q_sharpen_amount_label, row, 0, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter  )
        grid_l.addWidget(q_sharpen_amount, row, 1)
        row += 1

        grid_l.addWidget( q_pre_gamma_label, row, 0, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter  )
        grid_l.addLayout( lib_qt.QXHBoxLayout([q_pre_gamma_red, q_pre_gamma_green, q_pre_gamma_blue ]), row, 1)
        row += 1

        grid_l.addWidget(q_two_pass_label, row, 0, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter  )
        grid_l.addWidget(q_two_pass, row, 1)
        row += 1

        super().__init__(backend, L('@QFaceSwapper.module_title'),
                         layout=lib_qt.QXVBoxLayout([grid_l]) )


    def _btn_open_folder_released(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile( str(self._dfm_models_path) ))
