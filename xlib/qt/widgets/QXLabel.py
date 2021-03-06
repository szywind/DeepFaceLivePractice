from typing import Union, Any

from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *

from ..gui import QXImage
from ._part_QXWidget import _part_QXWidget


class QXLabel(QLabel, _part_QXWidget):
    def __init__(self, text = None,
                       color = None,
                       image : QXImage = None,
                       movie = None,
                       word_wrap = False, scaled_contents = False,

                       font=None, tooltip_text=None,
                       size_policy=None,
                       minimum_size=None, minimum_width=None, minimum_height=None,
                       maximum_size=None, maximum_width=None, maximum_height=None,
                       fixed_size=None, fixed_width=None, fixed_height=None,
                       hided=False, enabled=True
                       ):

        super().__init__()
        self._default_pal = QPalette( self.palette() )

        if text is not None:
            self.setText(text)
        if movie is not None:
            self.setMovie(movie)
        if image is not None:
            self.setPixmap(image.as_QXPixmap())
        if word_wrap:
            self.setWordWrap(True)

        self.setScaledContents(scaled_contents)
        self.set_color(color)

        _part_QXWidget.__init__(self,   font=font, tooltip_text=tooltip_text,
                                        size_policy=size_policy,
                                        minimum_size=minimum_size, minimum_width=minimum_width, minimum_height=minimum_height,
                                        maximum_size=maximum_size, maximum_width=maximum_width, maximum_height=maximum_height,
                                        fixed_size=fixed_size, fixed_width=fixed_width, fixed_height=fixed_height,
                                        hided=hided, enabled=enabled )

    def _update_color(self):
        if self._color is not None:
            pal = QPalette(self._default_pal)
            pal.setColor( QPalette.ColorRole.WindowText, self._color )
            self.setPalette(pal)
        else:
            self.setPalette(self._default_pal)


    def set_color(self, color : Union[Any,None] ):
        self._color = QColor(color) if color is not None else None
        self._update_color()

    def changeEvent(self, ev : QEvent):
        super().changeEvent(ev)

        if ev.type() == QEvent.Type.EnabledChange:
            self._update_color()


    def focusInEvent(self, ev : QFocusEvent):
        super().focusInEvent(ev)
        _part_QXWidget.focusInEvent(self, ev)

    def resizeEvent(self, ev : QResizeEvent):
        super().resizeEvent(ev)
        _part_QXWidget.resizeEvent(self, ev)
