"""Wire extraction and pin-to-net inference for EDA screenshots."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from voltsnap.models import ComponentInfo, PinInfo


@dataclass
class WireTopology:
    """Detected electrical topology from visible wires."""

    pin_to_net: dict[str, int] = field(default_factory=dict)
    net_pins: dict[int, list[str]] = field(default_factory=dict)
    connections: list[dict] = field(default_factory=list)
    wire_pixel_count: int = 0


def detect_wire_topology(
    image: np.ndarray | None,
    components: list[ComponentInfo],
    bound_components: list[Any],
    snap_radius: int = 96,
) -> WireTopology:
    """Detect colored/dark EDA wires and snap component pins onto wire nets."""
    if image is None or image.size == 0 or not components:
        return WireTopology()

    wire_mask = extract_wire_mask(image)
    wire_pixel_count = int(np.count_nonzero(wire_mask))
    if wire_pixel_count < 20:
        return WireTopology(wire_pixel_count=wire_pixel_count)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        wire_mask, connectivity=8
    )
    valid_labels = _valid_wire_labels(stats, num_labels)
    if not valid_labels:
        return WireTopology(wire_pixel_count=wire_pixel_count)

    bound_by_ref = {_obj_get(b, "ref", ""): b for b in bound_components if _obj_get(b, "ref", "")}
    component_by_ref = {c.ref: c for c in components if c.ref}
    pin_by_name: dict[str, PinInfo] = {
        pin.name: pin for comp in components for pin in comp.pins
    }

    raw_pin_labels: dict[str, int] = {}
    for pin in pin_by_name.values():
        label = _nearest_wire_label(
            labels,
            wire_mask,
            valid_labels,
            pin.pixel_position,
            snap_radius,
        )
        if label is not None:
            raw_pin_labels[pin.name] = label

    if not raw_pin_labels:
        return WireTopology(wire_pixel_count=wire_pixel_count)

    parent = {label: label for label in valid_labels}
    for comp in components:
        if comp.type != "ground":
            continue
        labels_to_merge = {
            raw_pin_labels[pin.name]
            for pin in comp.pins
            if pin.name in raw_pin_labels
        }
        bound = bound_by_ref.get(comp.ref)
        if bound is not None:
            labels_to_merge.update(
                _wire_labels_near_bbox(labels, wire_mask, valid_labels, _obj_get(bound, "bbox"), pad=snap_radius)
            )
        if labels_to_merge:
            label = sorted(labels_to_merge)[0]
            for pin in comp.pins:
                raw_pin_labels.setdefault(pin.name, label)
        _union_all(parent, labels_to_merge)

    root_to_net: dict[int, int] = {}
    pin_to_net: dict[str, int] = {}
    net_pins: dict[int, list[str]] = {}
    for pin_name, label in sorted(raw_pin_labels.items()):
        root = _find(parent, label)
        if root not in root_to_net:
            root_to_net[root] = len(root_to_net) + 1
        net_id = root_to_net[root]
        pin_to_net[pin_name] = net_id
        net_pins.setdefault(net_id, []).append(pin_name)

    connections = _connections_from_nets(net_pins, pin_by_name, component_by_ref, bound_by_ref)
    return WireTopology(
        pin_to_net=pin_to_net,
        net_pins=net_pins,
        connections=connections,
        wire_pixel_count=wire_pixel_count,
    )


def extract_wire_mask(image: np.ndarray) -> np.ndarray:
    """Return a binary mask of likely schematic wires."""
    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        bgr = image

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(
        hsv,
        np.array([35, 35, 35], dtype=np.uint8),
        np.array([95, 255, 255], dtype=np.uint8),
    )
    green_lines = _extract_orthogonal_lines(green_mask)
    if int(np.count_nonzero(green_lines)) >= 20:
        return green_lines

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    sat = hsv[:, :, 1]
    dark_mask = cv2.inRange(gray, 0, 120)
    low_sat = cv2.inRange(sat, 0, 140)
    dark_mask = cv2.bitwise_and(dark_mask, low_sat)
    return _extract_orthogonal_lines(dark_mask)


def _extract_orthogonal_lines(mask: np.ndarray) -> np.ndarray:
    mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    if mask.size == 0:
        return mask

    h, w = mask.shape[:2]
    kernel_len = max(9, min(h, w) // 80)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))

    horizontal = cv2.morphologyEx(mask, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(mask, cv2.MORPH_OPEN, vertical_kernel)
    lines = cv2.bitwise_or(horizontal, vertical)
    lines = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    lines = cv2.dilate(lines, np.ones((3, 3), np.uint8), iterations=1)
    return lines


def _valid_wire_labels(stats: np.ndarray, num_labels: int) -> set[int]:
    valid: set[int] = set()
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 12:
            continue
        if max(w, h) < 12:
            continue
        if x < 0 or y < 0:
            continue
        valid.add(label)
    return valid


def _nearest_wire_label(
    labels: np.ndarray,
    mask: np.ndarray,
    valid_labels: set[int],
    point: tuple[int, int],
    radius: int,
) -> int | None:
    px, py = int(point[0]), int(point[1])
    h, w = labels.shape[:2]
    x1, x2 = max(0, px - radius), min(w, px + radius + 1)
    y1, y2 = max(0, py - radius), min(h, py + radius + 1)
    if x1 >= x2 or y1 >= y2:
        return None

    label_window = labels[y1:y2, x1:x2]
    mask_window = mask[y1:y2, x1:x2]
    ys, xs = np.where(mask_window > 0)
    if len(xs) == 0:
        return None

    candidate_labels = label_window[ys, xs]
    valid_mask = np.isin(candidate_labels, list(valid_labels))
    if not np.any(valid_mask):
        return None

    xs = xs[valid_mask]
    ys = ys[valid_mask]
    candidate_labels = candidate_labels[valid_mask]
    dx = xs.astype(np.float64) - (px - x1)
    dy = ys.astype(np.float64) - (py - y1)
    dist_sq = dx * dx + dy * dy
    best_idx = int(np.argmin(dist_sq))
    if dist_sq[best_idx] > radius * radius:
        return None
    return int(candidate_labels[best_idx])


def _wire_labels_near_bbox(
    labels: np.ndarray,
    mask: np.ndarray,
    valid_labels: set[int],
    bbox: tuple[int, int, int, int] | None,
    pad: int,
) -> set[int]:
    if bbox is None:
        return set()
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = labels.shape[:2]
    x1, x2 = max(0, x1 - pad), min(w, x2 + pad + 1)
    y1, y2 = max(0, y1 - pad), min(h, y2 + pad + 1)
    if x1 >= x2 or y1 >= y2:
        return set()
    window_labels = labels[y1:y2, x1:x2]
    window_mask = mask[y1:y2, x1:x2]
    found = set(int(v) for v in np.unique(window_labels[window_mask > 0]))
    return found & valid_labels


def _connections_from_nets(
    net_pins: dict[int, list[str]],
    pin_by_name: dict[str, PinInfo],
    component_by_ref: dict[str, ComponentInfo],
    bound_by_ref: dict[str, Any],
) -> list[dict]:
    connections: list[dict] = []
    seen: set[tuple[tuple[str, str], tuple[str, str]]] = set()

    for net_id, pin_names in sorted(net_pins.items()):
        pins = [pin_by_name[name] for name in pin_names if name in pin_by_name]
        pins = [pin for pin in pins if pin.component_ref in component_by_ref]
        if len(pins) < 2:
            continue

        ground_pins = [
            pin for pin in pins
            if component_by_ref[pin.component_ref].type == "ground"
            or pin.component_ref.upper().startswith("GND")
        ]
        if ground_pins:
            hub = ground_pins[0]
            targets = [pin for pin in pins if pin.component_ref != hub.component_ref]
        else:
            pins = sorted(pins, key=lambda p: (p.pixel_position[1], p.pixel_position[0], p.name))
            hub = pins[0]
            targets = pins[1:]

        for pin in targets:
            if pin.component_ref == hub.component_ref:
                continue
            conn = _make_connection(pin, hub, net_id, bound_by_ref)
            key = tuple(sorted([
                (conn["start_ref"], conn.get("start_pin_name", conn["start_pin"])),
                (conn["end_ref"], conn.get("end_pin_name", conn["end_pin"])),
            ]))
            if key in seen:
                continue
            seen.add(key)
            connections.append(conn)

    return connections


def _make_connection(
    start_pin: PinInfo,
    end_pin: PinInfo,
    net_id: int,
    bound_by_ref: dict[str, Any],
) -> dict:
    return {
        "start_ref": start_pin.component_ref,
        "end_ref": end_pin.component_ref,
        "start_pin": _pin_side(start_pin, bound_by_ref.get(start_pin.component_ref)),
        "end_pin": _pin_side(end_pin, bound_by_ref.get(end_pin.component_ref)),
        "start_pin_name": start_pin.name,
        "end_pin_name": end_pin.name,
        "net_id": net_id,
        "source": "wire_detection",
    }


def _pin_side(pin: PinInfo, bound: Any | None) -> str:
    bbox = _obj_get(bound, "bbox") if bound is not None else None
    if not bbox:
        return "right" if pin.name.endswith("pin2") else "left"

    x1, y1, x2, y2 = [float(v) for v in bbox]
    px, py = pin.pixel_position
    distances = {
        "left": abs(px - x1),
        "right": abs(px - x2),
        "top": abs(py - y1),
        "bottom": abs(py - y2),
    }
    return min(distances, key=distances.get)


def _union_all(parent: dict[int, int], labels_to_merge: set[int]) -> None:
    labels = sorted(label for label in labels_to_merge if label in parent)
    if len(labels) < 2:
        return
    first = labels[0]
    for label in labels[1:]:
        _union(parent, first, label)


def _find(parent: dict[int, int], label: int) -> int:
    root = label
    while parent[root] != root:
        root = parent[root]
    while parent[label] != label:
        next_label = parent[label]
        parent[label] = root
        label = next_label
    return root


def _union(parent: dict[int, int], a: int, b: int) -> None:
    ra = _find(parent, a)
    rb = _find(parent, b)
    if ra != rb:
        parent[rb] = ra


def _obj_get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)
