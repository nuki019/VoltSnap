"""VoltSnap 核心数据结构定义"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


@dataclass
class PinInfo:
    """元件引脚信息"""
    name: str                      # 引脚名，如 "R1_pin1"
    component_ref: str             # 所属元件，如 "R1"
    position: Tuple[float, float]  # schemdraw 坐标系中的 (x, y)
    pixel_position: Tuple[int, int] = (0, 0)  # 像素坐标 (x, y)
    net_id: int | None = None      # 所属电气节点 ID


@dataclass
class ComponentInfo:
    """元件信息"""
    ref: str                       # "R1", "V1", "C1"
    type: str                      # "resistor", "voltage_source", "capacitor"
    value: str                     # "1k", "5", "10uF"
    pins: List[PinInfo] = field(default_factory=list)
    bbox_center: Tuple[float, float] = (0.0, 0.0)
    angle: float = 0.0             # 旋转角度（度）


@dataclass
class CircuitSpec:
    """电路规格：定义元件和连接关系"""
    name: str
    components: List[ComponentInfo] = field(default_factory=list)
    expected_netlist: str = ""     # 预期的 SPICE 网表（用于验证）


@dataclass
class RenderResult:
    """schemdraw 渲染结果"""
    image_path: str
    components: List[ComponentInfo] = field(default_factory=list)
    image_size: Tuple[int, int] = (0, 0)  # (width, height)


@dataclass
class PreprocessResult:
    """图像预处理结果"""
    gray: np.ndarray               # 灰度图 (H, W), uint8
    binary: np.ndarray             # 二值化图 (H, W), uint8, 0 或 255
    original: np.ndarray           # 原图 (H, W, 3), BGR


@dataclass
class TopologyResult:
    """拓扑重建结果"""
    skeleton: np.ndarray           # 骨架化图 (H, W), uint8
    labels: np.ndarray             # 连通域标签图 (H, W), int32
    num_nets: int                  # 电气节点（Net）数量
    pin_to_net: dict[str, int] = field(default_factory=dict)   # {"R1_pin1": 1, ...}
    net_pins: dict[int, list[str]] = field(default_factory=dict)  # {1: ["R1_pin1", ...]}


@dataclass
class SimulationResult:
    """仿真结果"""
    success: bool
    node_voltages: dict[str, float] = field(default_factory=dict)
    branch_currents: dict[str, float] = field(default_factory=dict)
    raw_output: str = ""
    error_message: str | None = None
