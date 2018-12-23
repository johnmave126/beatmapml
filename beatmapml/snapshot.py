from slider import Beatmap
from slider.beatmap import Circle, Slider
from slider.mod import ar_to_ms
from typing import Tuple
import numpy as np
import math
import itertools
import heapq

from .parameter_convert import calc_cs_propotion
from .render import RenderContext


MAX_PLAYFIELD = (512, 384)

CanvasSize = Tuple[int, int]
BoundingBox = Tuple[int, int, int, int]


def make_snapshots(beatmap: Beatmap,
                   target_width: int,
                   capture_rate: int) -> np.ndarray:
    """Make snapshots of a beatmap

    Args:
        beatmap (Beatmap): The beatmap to process.
        target_width (int): The pixel width of desired output.
        capture_rate (int): The capture rate of the snapshots in Hz

    Returns:
        Snapshots of the beatmap. A numpy array of size
        target_width x floor(target_width * 16 / 9)
        x 2 x (length_of_beatmap x capture_rate)
    """
    processor = SnapshotProcessor(beatmap, target_width, capture_rate)
    return processor.make_snapshots()


class SnapshotProcessor():
    """Internal class to take snapshots of a beatmap"""

    def __init__(self,
                 beatmap: Beatmap,
                 target_width: int,
                 capture_rate: int) -> None:
        self._beatmap = beatmap
        self._canvas_size, self._field_box = SnapshotProcessor.calc_dimension(
            target_width)
        self._interval = 1000 / capture_rate
        self._radius = self.get_radius()
        self._render_ctx = RenderContext(
            self._canvas_size, self._field_box)
        self._lookahead = ar_to_ms(self._beatmap.approach_rate)
        self._hitcircles = [
            o for o in self._beatmap.hit_objects if isinstance(o, Circle)]
        self._sliders = [
            o for o in self._beatmap.hit_objects if isinstance(o, Slider)]

        self._hitcircles = sorted(
            self._hitcircles, key=lambda circle: circle.time)
        for circle in self._hitcircles:
            x_prop = circle.position[0] / MAX_PLAYFIELD[0]
            y_prop = circle.position[1] / MAX_PLAYFIELD[1]
            circle.mask = self._render_ctx.create_mask(
                x_prop, y_prop, self._radius)

        self._sliders = sorted(
            self._sliders, key=lambda slider: slider.time)
        last_hitcircle_time = (
            self._hitcircles[-1].time.total_seconds() * 1000
            if len(self._hitcircles) > 0 else 0)
        last_slider_time = (
            max((x.end_time.total_seconds() * 1000 for x in self._sliders))
            if len(self._sliders) > 0 else 0)
        self._end_time = max(last_hitcircle_time, last_slider_time)

    def make_snapshots(self) -> np.ndarray:
        tick = 0
        circle_start = 0
        circle_end = 0
        slider_start = 0
        slider_counter = itertools.count()
        slider_pool = []
        num_slice = math.ceil(self._end_time / self._interval)
        snapshot = np.zeros((num_slice,) + self._canvas_size + (2,))
        circle_count = np.zeros(self._canvas_size)
        circle_count_rc = np.zeros(self._canvas_size)
        for snapshot_idx in range(num_slice):
            while (circle_end > circle_start and
                   self._hitcircles[circle_start].time.total_seconds() * 1000 <
                   tick):
                circle = self._hitcircles[circle_start]
                circle_count[circle.mask] -= 1
                circle_count_rc[circle.mask] = 1 / \
                    (circle_count[circle.mask] + 1e-8)
                circle_start += 1
            while len(slider_pool) > 0 and slider_pool[0][0] < tick:
                heapq.heappop(slider_pool)

            while (circle_end < len(self._hitcircles) and
                   self._hitcircles[circle_end].time.total_seconds() * 1000 <
                   tick + self._lookahead):
                circle = self._hitcircles[circle_end]
                circle_count[circle.mask] += 1
                circle_count_rc[circle.mask] = 1 / circle_count[circle.mask]
                circle_end += 1
            while (slider_start < len(self._sliders) and
                   self._sliders[slider_start].time.total_seconds() * 1000 <
                   tick + self._lookahead):
                slider = self._sliders[slider_start]
                heapq.heappush(
                    slider_pool,
                    (slider.time.total_seconds(),
                     next(slider_counter),
                     slider))
                slider_start += 1

            if len(slider_pool) == 0 and circle_end == circle_start:
                tick += self._interval
                continue

            circle_canvas_raw = np.zeros(self._canvas_size)
            for i in range(circle_start, circle_end):
                circle = self._hitcircles[i]
                progress = (1 - (circle.time.total_seconds() * 1000 - tick) /
                            self._lookahead)
                circle_canvas_raw[circle.mask] += progress
            self.stamp_raw_canvas(snapshot[snapshot_idx, ..., 0],
                                  circle_canvas_raw, circle_count_rc)
            """
            slider_canvas_raw = np.zeros(self._canvas_size + (2,))
            # TODO: add slider handling
            self.stamp_raw_canvas(snapshot[snapshot_idx, ..., 1],
                                  slider_canvas_raw[..., 0])
            """
            tick += self._interval
        return snapshot

    @staticmethod
    def calc_dimension(target_width: int) -> Tuple[CanvasSize,
                                                   BoundingBox]:
        """Calculate the target dimensions

        Args:
            target_width (int): The desired output width.

        Returns:
            (canvas_size, field_box) where
            canvas_size = (width, height): The size of all viewable portion
            field_box = (left, top, right, bottom): The coordinate of
                        bounding box relative to canvas
        """
        MAX_CS_RADIUS_RATIO = calc_cs_propotion(2)
        MAX_CS_RADIUS = math.floor(
            target_width * MAX_CS_RADIUS_RATIO / (1 + 2 * MAX_CS_RADIUS_RATIO))
        field_width = target_width - 2 * MAX_CS_RADIUS
        field_height = math.floor(field_width * 3 / 4)
        target_height = field_height + 2 * MAX_CS_RADIUS
        return (
            (target_width, target_height),
            (MAX_CS_RADIUS,
             MAX_CS_RADIUS,
             MAX_CS_RADIUS + field_width,
             MAX_CS_RADIUS + field_height))

    def get_radius(self) -> float:
        radius_ratio = calc_cs_propotion(self._beatmap.circle_size)
        return (self._field_box[2] - self._field_box[0]) * radius_ratio

    def create_raw_canvas(self) -> np.ndarray:
        return np.zeros(self._canvas_size + (2,))

    def stamp_raw_canvas(self,
                         canvas: np.ndarray,
                         raw_canvas: np.ndarray,
                         count_rc: np.ndarray) -> None:
        np.multiply(raw_canvas, count_rc, out=canvas)
