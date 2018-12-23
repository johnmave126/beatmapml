import numpy as np
from typing import Tuple

__all__ = [
    'RenderContext'
]


class RenderContext:
    def __init__(self,
                 canvas_size: Tuple[int, int],
                 field_box: Tuple[int, int, int, int]) -> None:
        self._canvas_w, self._canvas_h = canvas_size
        (self._field_left,
            self._field_top,
            self._field_right,
            self._field_bottom) = field_box

        self._lin_y, self._lin_x = np.ogrid[0:self._canvas_w, 0:self._canvas_h]

    def create_mask(self, x, y, radius):
        canvas_x = (1 - x) * self._field_left + x * self._field_right
        canvas_y = (1 - y) * self._field_top + y * self._field_bottom
        lin_x = self._lin_y - canvas_x
        lin_y = self._lin_x - canvas_y
        return lin_x * lin_x + lin_y * lin_y <= radius * radius
