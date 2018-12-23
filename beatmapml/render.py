import numpy as np
import cv2
import math
from typing import Tuple

PropotionPosition = Tuple[float, float]

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

        self._top_left = np.array([self._field_left, self._field_top])
        self._bottom_right = np.array([self._field_right, self._field_bottom])

    def create_note_mask(self,
                         x: float,
                         y: float,
                         radius: float) -> np.ndarray:
        """Create mask of a note

            Args:
                x (float): The x coordinate of center of the note.
                y (float): The y coordinate of center of the note.
                radius (float): The radius of the note.

            Returns:
                The mask of the note
        """
        pos = np.array([x, y])
        canvas_pos = (1 - pos) * self._top_left + pos * self._bottom_right
        canvas_pos = (canvas_pos * 8).astype(np.int16)
        radius = int(radius * 8)
        mask = np.zeros((self._canvas_w, self._canvas_h), dtype=np.uint8)
        return cv2.circle(mask,
                          tuple(np.flip(canvas_pos)),
                          radius,
                          color=1,
                          thickness=-1,
                          shift=3).astype(np.bool)

    def create_strip_mask(self,
                          start_pos: PropotionPosition,
                          end_pos: PropotionPosition,
                          width: float) -> np.ndarray:
        """Create mask of a rectangular strip

            Args:
                start_pos (PropotionPosition): The (x, y) coordinate of start
                    of the strip.
                end_pos (PropotionPosition): The (x, y) coordinate of end of
                    the strip.
                width (float): The width of the strip.

            Returns:
                The mask of the strip
        """
        start_pos = np.array(start_pos)
        end_pos = np.array(end_pos)
        start_pos = (1 - start_pos) * self._top_left + \
            start_pos * self._bottom_right
        end_pos = (1 - end_pos) * self._top_left + end_pos * self._bottom_right
        mask = np.zeros((self._canvas_w, self._canvas_h), dtype=np.uint8)
        start_pos = (start_pos * 8).astype(np.int16)
        end_pos = (end_pos * 8).astype(np.int16)
        width = int(width)
        return cv2.line(mask,
                        tuple(np.flip(start_pos)),
                        tuple(np.flip(end_pos)),
                        color=1,
                        thickness=width,
                        shift=3).astype(np.bool)
