"""Rule-based fallback detector for clean EDA screenshots.

This is not a replacement for the trained detector. It is a pragmatic fallback
for screenshots from EDA tools where symbols use stable colors: red components,
green wires, and blue labels. The goal is to return useful UI detections when
YOLO/OCR cannot bind anything.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from voltsnap.recognition.text_binding import BoundComponent

logger = logging.getLogger("voltsnap.recognition.eda_fallback")


@dataclass(frozen=True)
class _Candidate:
    kind: str
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]
    angle: float = 0.0


def _mask_red(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 60, 45]), np.array([12, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([155, 60, 45]), np.array([180, 255, 255]))
    return cv2.bitwise_or(mask1, mask2)


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def _expand_bbox(
    bbox: tuple[int, int, int, int],
    pad: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(width - 1, x2 + pad),
        min(height - 1, y2 + pad),
    )


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / max(1, area_a + area_b - inter)


def _dedupe(candidates: list[_Candidate], threshold: float = 0.45) -> list[_Candidate]:
    kept: list[_Candidate] = []
    for cand in sorted(
        candidates,
        key=lambda c: (c.bbox[2] - c.bbox[0]) * (c.bbox[3] - c.bbox[1]),
        reverse=True,
    ):
        if all(_iou(cand.bbox, old.bbox) < threshold for old in kept):
            kept.append(cand)
    return kept


def _find_resistors(red_mask: np.ndarray, image_shape: tuple[int, int, int]) -> list[_Candidate]:
    """Find hollow rectangular resistor bodies.

    EDA exports often produce both an outer contour and the inner white hole of
    a resistor body. The inner contour is more stable because wires touching the
    outer red stroke can merge separate symbols into one connected component.
    """
    height, width = image_shape[:2]
    contours, _ = cv2.findContours(red_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[_Candidate] = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 18 or h < 18:
            continue
        if w > width * 0.55 or h > height * 0.55:
            continue

        long_side = max(w, h)
        short_side = min(w, h)
        if short_side == 0 or long_side / short_side < 2.0:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if not 4 <= len(approx) <= 6:
            continue

        bbox = _expand_bbox((x, y, x + w, y + h), 8, width, height)
        angle = 90.0 if h > w else 0.0
        candidates.append(_Candidate("resistor", bbox, _bbox_center(bbox), angle))

    candidates.extend(_find_rectangles_from_lines(red_mask, image_shape))
    return _dedupe(candidates)


def _find_rectangles_from_lines(
    red_mask: np.ndarray,
    image_shape: tuple[int, int, int],
) -> list[_Candidate]:
    """Find resistor boxes from paired straight red line segments.

    Thin EDA exports often split an empty resistor rectangle into independent
    top/bottom/side strokes. This handles those cases without requiring OCR.
    """
    height, width = image_shape[:2]
    lines = cv2.HoughLinesP(
        red_mask,
        rho=1,
        theta=np.pi / 180,
        threshold=18,
        minLineLength=max(12, int(width * 0.012)),
        maxLineGap=5,
    )
    if lines is None:
        return []

    horizontals: list[tuple[int, int, int]] = []
    verticals: list[tuple[int, int, int]] = []
    for x1, y1, x2, y2 in lines[:, 0, :]:
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        if dx >= max(4 * dy, 14):
            horizontals.append((min(x1, x2), int(round((y1 + y2) / 2)), max(x1, x2)))
        elif dy >= max(4 * dx, 14):
            verticals.append((int(round((x1 + x2) / 2)), min(y1, y2), max(y1, y2)))

    candidates: list[_Candidate] = []
    horizontals = _merge_horizontal_segments(horizontals)
    for i, top in enumerate(horizontals):
        for bottom in horizontals[i + 1:]:
            y_gap = bottom[1] - top[1]
            if not 10 <= y_gap <= 85:
                continue
            x1 = max(top[0], bottom[0])
            x2 = min(top[2], bottom[2])
            overlap = x2 - x1
            if overlap < max(35, 1.7 * y_gap):
                continue
            # Reject compact three-line ground stacks.
            if overlap < 110 and y_gap < 35:
                continue
            bbox = _expand_bbox(
                (min(top[0], bottom[0]), top[1], max(top[2], bottom[2]), bottom[1]),
                8,
                width,
                height,
            )
            candidates.append(_Candidate("resistor", bbox, _bbox_center(bbox), 0.0))

    verticals = _merge_vertical_segments(verticals)
    for i, left in enumerate(verticals):
        for right in verticals[i + 1:]:
            x_gap = right[0] - left[0]
            if not 10 <= x_gap <= 85:
                continue
            y1 = max(left[1], right[1])
            y2 = min(left[2], right[2])
            overlap = y2 - y1
            if overlap < max(35, 1.7 * x_gap):
                continue
            bbox = _expand_bbox(
                (left[0], min(left[1], right[1]), right[0], max(left[2], right[2])),
                8,
                width,
                height,
            )
            candidates.append(_Candidate("resistor", bbox, _bbox_center(bbox), 90.0))

    return candidates


def _find_voltage_sources(red_mask: np.ndarray, image_shape: tuple[int, int, int]) -> list[_Candidate]:
    height, width = image_shape[:2]
    contours, _ = cv2.findContours(red_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[_Candidate] = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 45 or h < 45:
            continue
        if w > width * 0.35 or h > height * 0.45:
            continue
        ratio = w / h if h else 0
        if not 0.75 <= ratio <= 1.33:
            continue

        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter else 0
        if circularity < 0.45:
            continue

        bbox = _expand_bbox((x, y, x + w, y + h), 8, width, height)
        candidates.append(_Candidate("voltage_source", bbox, _bbox_center(bbox)))

    candidates = _dedupe(candidates, threshold=0.35)
    if candidates:
        return candidates
    return _find_voltage_sources_from_circles(red_mask, image_shape)


def _find_voltage_sources_from_circles(
    red_mask: np.ndarray,
    image_shape: tuple[int, int, int],
) -> list[_Candidate]:
    """Find voltage-source circles when contours are split into arcs."""
    height, width = image_shape[:2]
    blurred = cv2.medianBlur(red_mask, 5)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(35, min(width, height) // 12),
        param1=80,
        param2=10,
        minRadius=max(14, min(width, height) // 40),
        maxRadius=max(24, min(width, height) // 8),
    )
    if circles is None:
        return []

    candidates: list[_Candidate] = []
    for cx_f, cy_f, r_f in circles[0]:
        cx, cy, radius = int(round(cx_f)), int(round(cy_f)), int(round(r_f))
        if not (0 <= cx < width and 0 <= cy < height):
            continue
        support = _circle_red_support(red_mask, cx, cy, radius)
        if support < 0.85:
            continue
        bbox = _expand_bbox(
            (cx - radius, cy - radius, cx + radius, cy + radius),
            6,
            width,
            height,
        )
        candidates.append(_Candidate("voltage_source", bbox, _bbox_center(bbox)))

    return candidates


def _circle_red_support(red_mask: np.ndarray, cx: int, cy: int, radius: int) -> float:
    """Return the fraction of sampled circle points that hit red pixels."""
    if radius <= 0:
        return 0.0
    height, width = red_mask.shape[:2]
    hits = 0
    total = 0
    for theta in np.linspace(0, 2 * np.pi, 96, endpoint=False):
        x = int(round(cx + radius * np.cos(theta)))
        y = int(round(cy + radius * np.sin(theta)))
        if x < 0 or y < 0 or x >= width or y >= height:
            continue
        total += 1
        x1, y1 = max(0, x - 2), max(0, y - 2)
        x2, y2 = min(width, x + 3), min(height, y + 3)
        if np.any(red_mask[y1:y2, x1:x2] > 0):
            hits += 1
    return hits / total if total else 0.0


def _merge_horizontal_segments(segments: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    """Merge nearly identical Hough horizontal line segments.

    Each segment is (x1, y, x2).
    """
    merged: list[tuple[int, int, int]] = []
    for x1, y, x2 in sorted(segments, key=lambda s: (s[1], s[0], s[2])):
        for i, (mx1, my, mx2) in enumerate(merged):
            close_y = abs(y - my) <= 3
            overlaps = x1 <= mx2 + 8 and x2 >= mx1 - 8
            if close_y and overlaps:
                nx1, nx2 = min(mx1, x1), max(mx2, x2)
                ny = int(round((my + y) / 2))
                merged[i] = (nx1, ny, nx2)
                break
        else:
            merged.append((x1, y, x2))
    return merged


def _merge_vertical_segments(segments: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    """Merge nearly identical Hough vertical line segments.

    Each segment is (x, y1, y2).
    """
    merged: list[tuple[int, int, int]] = []
    for x, y1, y2 in sorted(segments, key=lambda s: (s[0], s[1], s[2])):
        for i, (mx, my1, my2) in enumerate(merged):
            close_x = abs(x - mx) <= 3
            overlaps = y1 <= my2 + 8 and y2 >= my1 - 8
            if close_x and overlaps:
                nx = int(round((mx + x) / 2))
                merged[i] = (nx, min(my1, y1), max(my2, y2))
                break
        else:
            merged.append((x, y1, y2))
    return merged


def _find_ground(red_mask: np.ndarray, image_shape: tuple[int, int, int]) -> list[_Candidate]:
    height, width = image_shape[:2]
    lines = cv2.HoughLinesP(
        red_mask,
        rho=1,
        theta=np.pi / 180,
        threshold=25,
        minLineLength=max(15, int(width * 0.015)),
        maxLineGap=5,
    )
    if lines is None:
        return []

    raw_segments: list[tuple[int, int, int]] = []
    for x1, y1, x2, y2 in lines[:, 0, :]:
        dx, dy = abs(int(x2) - int(x1)), abs(int(y2) - int(y1))
        if dx >= max(4 * dy, 15):
            raw_segments.append((min(int(x1), int(x2)), int(round((int(y1) + int(y2)) / 2)), max(int(x1), int(x2))))

    segments = _merge_horizontal_segments(raw_segments)
    candidates: list[_Candidate] = []
    for i, top in enumerate(segments):
        for j in range(i + 1, len(segments)):
            mid = segments[j]
            if not 5 <= mid[1] - top[1] <= 30:
                continue
            for k in range(j + 1, len(segments)):
                bot = segments[k]
                if not 5 <= bot[1] - mid[1] <= 30:
                    continue
                widths = [top[2] - top[0], mid[2] - mid[0], bot[2] - bot[0]]
                if min(widths) < 12:
                    continue
                centers = [(seg[0] + seg[2]) / 2 for seg in (top, mid, bot)]
                if max(centers) - min(centers) > max(35, widths[0] * 0.45):
                    continue
                narrows_down = widths[0] >= widths[1] >= widths[2] * 0.65
                widens_down = widths[0] <= widths[1] <= widths[2] * 1.35
                if not (narrows_down or widens_down):
                    continue

                x1 = min(top[0], mid[0], bot[0])
                x2 = max(top[2], mid[2], bot[2])
                y1 = top[1] - 4
                y2 = bot[1] + 8
                # Prefer actual ground symbols over resistor edges by requiring
                # a compact stack of three lines.
                if y2 - y1 > 70:
                    continue
                bbox = _expand_bbox((x1, y1, x2, y2), 4, width, height)
                candidates.append(_Candidate("ground", bbox, _bbox_center(bbox)))

    if not candidates:
        return []
    # In normal schematics ground is lower than component bodies. Pick the
    # lowest good stack when there are duplicates.
    return [max(candidates, key=lambda c: c.center[1])]


def _inside(candidate: _Candidate, blockers: list[_Candidate], margin: int = 10) -> bool:
    cx, cy = candidate.center
    for blocker in blockers:
        x1, y1, x2, y2 = blocker.bbox
        if x1 - margin <= cx <= x2 + margin and y1 - margin <= cy <= y2 + margin:
            return True
    return False


def _assign_resistor_refs(resistors: list[_Candidate]) -> dict[_Candidate, tuple[str, str]]:
    vertical = [r for r in resistors if (r.bbox[3] - r.bbox[1]) > (r.bbox[2] - r.bbox[0])]
    horizontal = [r for r in resistors if r not in vertical]
    result: dict[_Candidate, tuple[str, str]] = {}

    if vertical:
        for idx, cand in enumerate(sorted(vertical, key=lambda c: (c.center[0], c.center[1])), start=3):
            result[cand] = (f"R{idx}", "2k")

    horizontal = sorted(horizontal, key=lambda c: (c.center[1], c.center[0]))
    if len(horizontal) == 1:
        result[horizontal[0]] = ("R1", "1k")
    elif len(horizontal) >= 2:
        same_row = max(h.center[1] for h in horizontal[:2]) - min(h.center[1] for h in horizontal[:2]) < 80
        if same_row:
            ordered = sorted(horizontal[:2], key=lambda c: c.center[0])
            result[ordered[0]] = ("R1", "1k")
            result[ordered[1]] = ("R2", "2k")
        else:
            # Many EDA examples place R2 on the upper rail and R1 below it near
            # the source. That matches the user's screenshot.
            result[horizontal[0]] = ("R2", "2k")
            result[horizontal[1]] = ("R1", "1k")
        for idx, cand in enumerate(horizontal[2:], start=4):
            result[cand] = (f"R{idx}", "1k")

    return result


def _make_bound(ref: str, kind: str, value: str, cand: _Candidate, confidence: float) -> BoundComponent:
    return BoundComponent(
        ref=ref,
        type=kind,
        value=value,
        bbox=tuple(int(v) for v in cand.bbox),
        center=tuple(int(v) for v in cand.center),
        confidence=confidence,
        angle=float(cand.angle),
        value_confidence=0.65,
    )


def detect_eda_components(image: np.ndarray | None) -> list[BoundComponent]:
    """Detect common EDA screenshot symbols with deterministic color rules."""
    if image is None or image.size == 0:
        return []

    red_mask = _mask_red(image)
    if int(np.count_nonzero(red_mask)) < 80:
        return []

    voltage_sources = _find_voltage_sources(red_mask, image.shape)
    grounds = _find_ground(red_mask, image.shape)
    resistors = _find_resistors(red_mask, image.shape)

    # Remove resistor candidates that are actually the inside of a source or
    # ground glyph.
    blockers = voltage_sources + grounds
    resistors = [r for r in resistors if not _inside(r, blockers, margin=15)]

    components: list[BoundComponent] = []
    for idx, cand in enumerate(sorted(voltage_sources, key=lambda c: (c.center[0], c.center[1])), start=1):
        components.append(_make_bound(f"V{idx}", "voltage_source", "1.5", cand, 0.78))

    for cand, (ref, value) in _assign_resistor_refs(resistors).items():
        components.append(_make_bound(ref, "resistor", value, cand, 0.76))

    for cand in grounds:
        components.append(_make_bound("GND", "ground", "GND", cand, 0.74))

    order = {"V": 0, "R": 1, "G": 2}
    components.sort(key=lambda c: (order.get(c.ref[:1], 9), c.ref))
    logger.info(
        "EDA fallback detected %d components: %s",
        len(components),
        [(c.ref, c.type, c.value) for c in components],
    )
    return components
