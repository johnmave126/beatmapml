from slider import Beatmap
from slider.beatmap import Circle, Slider
from slider.mod import ar_to_ms
from slider.position import Point as TickPoint
from typing import Tuple, List
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
            x_prop = circle.position.x / MAX_PLAYFIELD[0]
            y_prop = circle.position.y / MAX_PLAYFIELD[1]
            circle.mask = self._render_ctx.create_note_mask(
                x_prop, y_prop, self._radius)
            circle.time_ms = circle.time.total_seconds() * 1000

        self._sliders = sorted(
            self._sliders, key=lambda slider: slider.time)
        strip_counter = itertools.count()
        for slider in self._sliders:
            slider.time_ms = slider.time.total_seconds() * 1000
            slider.end_time_ms = slider.end_time.total_seconds() * 1000
            slider.strips = self.make_strips(slider, strip_counter)

        last_hitcircle_time = (
            self._hitcircles[-1].time_ms
            if len(self._hitcircles) > 0 else 0)
        last_slider_time = (
            max((x.end_time_ms for x in self._sliders))
            if len(self._sliders) > 0 else 0)
        self._end_time = max(last_hitcircle_time, last_slider_time)

    def make_snapshots(self) -> np.ndarray:
        tick = 0

        circle_start = 0
        circle_end = 0

        slider_start = 0
        strip_pool = []

        num_slice = math.floor(self._end_time / self._interval) + 2

        snapshot = np.zeros((num_slice,) + self._canvas_size + (2,))

        circle_count = np.zeros(self._canvas_size)
        circle_count_rc = np.zeros(self._canvas_size)

        strip_count = np.zeros(self._canvas_size)
        strip_count_rc = np.zeros(self._canvas_size)
        for snapshot_idx in range(num_slice):
            circle_start, circle_end = self.update_circle_pool(
                tick, circle_start, circle_end, circle_count, circle_count_rc)

            slider_start = self.update_slider_pool(
                tick, slider_start, strip_pool, strip_count, strip_count_rc)

            if len(strip_pool) == 0 and circle_end == circle_start:
                tick += self._interval
                continue

            circle_canvas_raw = np.zeros(self._canvas_size)
            for i in range(circle_start, circle_end):
                circle = self._hitcircles[i]
                progress = (1 - (circle.time_ms - tick) /
                            self._lookahead)
                circle_canvas_raw[circle.mask] += progress
            self.stamp_raw_canvas(snapshot[snapshot_idx, ..., 0],
                                  circle_canvas_raw, circle_count_rc)

            slider_canvas_raw = np.zeros(self._canvas_size)
            for (end_ms, _, mask, start_ms) in strip_pool:
                progress = (1 - (end_ms - tick) /
                            (end_ms - start_ms + self._lookahead))
                slider_canvas_raw[mask] += progress
            self.stamp_raw_canvas(snapshot[snapshot_idx, ..., 1],
                                  slider_canvas_raw, strip_count_rc)

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

    def make_strips(self,
                    slider: Slider,
                    counter: itertools.count):
        slider_start = slider.time_ms
        slider_end = slider.end_time_ms
        beats_per_ms = slider.num_beats / (slider_end - slider_start)
        tick_interval = self._interval / 4
        slider.tick_rate = beats_per_ms * tick_interval
        tick_points = slider.tick_points
        samples = [(p.x / MAX_PLAYFIELD[0],
                    p.y / MAX_PLAYFIELD[1],
                    p.offset.total_seconds() * 1000)
                   for p in itertools.chain([TickPoint(slider.position.x,
                                                       slider.position.y,
                                                       slider.time)],
                                            tick_points,
                                            [tick_points[-1]])]
        return [
            self.make_strip(s[2], (s[0], s[1]), e[2],
                            (e[0], e[1]), counter) + (slider_start,)
            for (s, e) in zip(samples[:-1], samples[1:])
        ]

    def make_strip(self, start, start_pos, end, end_pos, counter):
        time_ms = (start + end) / 2
        mask = self._render_ctx.create_strip_mask(start_pos,
                                                  end_pos,
                                                  2 * self._radius)
        return (time_ms, next(counter), mask)

    def update_circle_pool(self,
                           tick,
                           start,
                           end,
                           circle_count,
                           circle_count_rc):

        count_mask = np.zeros(self._canvas_size, dtype=np.bool)
        has_change = False

        while (end < len(self._hitcircles) and
               self._hitcircles[end].time_ms <
               tick + self._lookahead):
            circle = self._hitcircles[end]
            circle_count[circle.mask] += 1
            count_mask |= circle.mask
            end += 1
            has_change = True

        while (end > start and
               self._hitcircles[start].time_ms <
               tick):
            circle = self._hitcircles[start]
            circle_count[circle.mask] -= 1
            count_mask |= circle.mask
            start += 1
            has_change = True

        if has_change:
            circle_count_rc[count_mask] = 1 / (circle_count[count_mask] + 1e-8)

        return start, end

    def update_slider_pool(self,
                           tick,
                           start,
                           strip_pool,
                           strip_count,
                           strip_count_rc):
        count_mask = np.zeros(self._canvas_size, dtype=np.bool)
        has_change = False
        while (start < len(self._sliders) and
               self._sliders[start].time_ms <
               tick + self._lookahead):
            slider = self._sliders[start]
            for strip in slider.strips:
                heapq.heappush(strip_pool, strip)
                strip_count[strip[2]] += 1
                count_mask |= strip[2]
            start += 1
            has_change = True

        while len(strip_pool) > 0 and strip_pool[0][0] < tick:
            (_, _, mask, _) = heapq.heappop(strip_pool)
            strip_count[mask] -= 1
            count_mask |= mask
            has_change = True

        if has_change:
            strip_count_rc[count_mask] = 1 / (strip_count[count_mask] + 1e-8)

        return start
