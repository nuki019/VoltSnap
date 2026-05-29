"""电路模板定义与随机化生成"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List

from voltsnap.models import CircuitSpec, ComponentInfo, PinInfo

# ── 标准元件值序列 (E12) ──────────────────────────────────────────────
_E12 = [1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.8, 8.2]
_RESISTOR_VALUES = [f"{v}k" for v in _E12] + [f"{int(v*1000)}" for v in _E12[:6]]
_CAPACITOR_VALUES = [f"{v}u" for v in _E12] + [f"{v*1000}p" for v in _E12[:4]]
_VOLTAGE_VALUES = ["1", "2", "3", "3.3", "5", "6", "9", "10", "12", "15"]

# ── 元件工厂 ──────────────────────────────────────────────────────────


def _make_resistor(ref: str, value: str) -> ComponentInfo:
    return ComponentInfo(
        ref=ref,
        type="resistor",
        value=value,
        pins=[
            PinInfo(name=f"{ref}_pin1", component_ref=ref, position=(0, 0)),
            PinInfo(name=f"{ref}_pin2", component_ref=ref, position=(0, 0)),
        ],
    )


def _make_capacitor(ref: str, value: str) -> ComponentInfo:
    return ComponentInfo(
        ref=ref,
        type="capacitor",
        value=value,
        pins=[
            PinInfo(name=f"{ref}_pin1", component_ref=ref, position=(0, 0)),
            PinInfo(name=f"{ref}_pin2", component_ref=ref, position=(0, 0)),
        ],
    )


def _make_inductor(ref: str, value: str) -> ComponentInfo:
    return ComponentInfo(
        ref=ref,
        type="inductor",
        value=value,
        pins=[
            PinInfo(name=f"{ref}_pin1", component_ref=ref, position=(0, 0)),
            PinInfo(name=f"{ref}_pin2", component_ref=ref, position=(0, 0)),
        ],
    )


def _make_voltage_source(ref: str, value: str) -> ComponentInfo:
    return ComponentInfo(
        ref=ref,
        type="voltage_source",
        value=value,
        pins=[
            PinInfo(name=f"{ref}_pin1", component_ref=ref, position=(0, 0)),
            PinInfo(name=f"{ref}_pin2", component_ref=ref, position=(0, 0)),
        ],
    )


def _make_current_source(ref: str, value: str) -> ComponentInfo:
    return ComponentInfo(
        ref=ref,
        type="current_source",
        value=value,
        pins=[
            PinInfo(name=f"{ref}_pin1", component_ref=ref, position=(0, 0)),
            PinInfo(name=f"{ref}_pin2", component_ref=ref, position=(0, 0)),
        ],
    )


# ── 阶段 0 固定演示电路 ──────────────────────────────────────────────


def get_demo_circuits() -> list[CircuitSpec]:
    """返回阶段 0 的所有演示电路规格"""
    return [
        _resistor_divider(),
        _parallel_resistors(),
        _rc_circuit(),
    ]


def _resistor_divider() -> CircuitSpec:
    return CircuitSpec(
        name="resistor_divider",
        components=[
            _make_voltage_source("V1", "5"),
            _make_resistor("R1", "1k"),
            _make_resistor("R2", "2k"),
        ],
        expected_netlist=(
            "* VoltSnap Auto-Generated Netlist\n"
            "V1 N1 0 DC 5\n"
            "R1 N1 N2 1k\n"
            "R2 N2 0 2k\n"
            ".op\n"
            ".end\n"
        ),
    )


def _parallel_resistors() -> CircuitSpec:
    return CircuitSpec(
        name="parallel_resistors",
        components=[
            _make_voltage_source("V1", "10"),
            _make_resistor("R1", "1k"),
            _make_resistor("R2", "2k"),
        ],
        expected_netlist=(
            "* VoltSnap Auto-Generated Netlist\n"
            "V1 N1 0 DC 10\n"
            "R1 N1 0 1k\n"
            "R2 N1 0 2k\n"
            ".op\n"
            ".end\n"
        ),
    )


def _rc_circuit() -> CircuitSpec:
    return CircuitSpec(
        name="rc_circuit",
        components=[
            _make_voltage_source("V1", "5"),
            _make_resistor("R1", "1k"),
            _make_capacitor("C1", "10u"),
        ],
        expected_netlist=(
            "* VoltSnap Auto-Generated Netlist\n"
            "V1 N1 0 DC 5\n"
            "R1 N1 N2 1k\n"
            "C1 N2 0 10u\n"
            ".op\n"
            ".end\n"
        ),
    )


# ── 随机电路生成器 ────────────────────────────────────────────────────

@dataclass
class RandomCircuitConfig:
    """随机电路生成配置"""
    resistor_range: tuple[int, int] = (1, 4)      # 电阻数量范围
    capacitor_range: tuple[int, int] = (0, 2)      # 电容数量范围
    inductor_range: tuple[int, int] = (0, 1)        # 电感数量范围
    source_voltage_range: tuple[float, float] = (1.0, 24.0)
    topology_types: list[str] | None = None  # None = 全部


class RandomCircuitGenerator:
    """
    随机电路生成器。

    基于课程知识点的模板，随机化元件参数和拓扑变体。
    支持的拓扑类型：
    - series_resistors:    串联电阻链
    - parallel_resistors:  并联电阻
    - resistor_divider:    分压器
    - rc_series:           RC 串联
    - rc_parallel:         RC 并联
    - rl_series:           RL 串联
    - rlc_series:          RLC 串联
    - two_mesh:            双网孔电路
    """

    TOPOLOGY_TYPES = [
        "series_resistors",
        "parallel_resistors",
        "resistor_divider",
        "rc_series",
        "rc_parallel",
        "rl_series",
        "rlc_series",
        "two_mesh",
    ]

    def __init__(self, config: RandomCircuitConfig | None = None, seed: int | None = None):
        self.config = config or RandomCircuitConfig()
        self._rng = random.Random(seed)

    def generate(self, topology_type: str | None = None) -> CircuitSpec:
        """生成一个随机电路"""
        if topology_type is None:
            types = self.config.topology_types or self.TOPOLOGY_TYPES
            topology_type = self._rng.choice(types)

        method = getattr(self, f"_gen_{topology_type}")
        circuit = method()
        return circuit

    def generate_batch(self, count: int) -> list[CircuitSpec]:
        """批量生成随机电路"""
        circuits = []
        types = self.config.topology_types or self.TOPOLOGY_TYPES
        for i in range(count):
            topology_type = types[i % len(types)]
            circuit = self.generate(topology_type)
            circuit.name = f"{topology_type}_{i:05d}"
            circuits.append(circuit)
        return circuits

    # ── 参数随机化 ─────────────────────────────────────────────────────

    def _rand_resistor_value(self) -> str:
        return self._rng.choice(_RESISTOR_VALUES)

    def _rand_capacitor_value(self) -> str:
        return self._rng.choice(_CAPACITOR_VALUES)

    def _rand_inductor_value(self) -> str:
        v = self._rng.choice(_E12)
        return f"{v}m"

    def _rand_voltage_value(self) -> str:
        return self._rng.choice(_VOLTAGE_VALUES)

    # ── 拓扑生成方法 ───────────────────────────────────────────────────

    def _gen_series_resistors(self) -> CircuitSpec:
        """串联电阻: V1 → R1 → R2 → ... → GND"""
        n = self._rng.randint(*self.config.resistor_range)
        n = max(2, n)
        v_val = self._rand_voltage_value()
        comps = [_make_voltage_source("V1", v_val)]
        r_vals = [self._rand_resistor_value() for _ in range(n)]
        for i in range(n):
            comps.append(_make_resistor(f"R{i+1}", r_vals[i]))

        netlist = self._build_series_netlist("V1", comps[1:], v_val)
        return CircuitSpec(name="series_resistors", components=comps, expected_netlist=netlist)

    def _gen_parallel_resistors(self) -> CircuitSpec:
        """并联电阻: V1 → [R1 ‖ R2 ‖ ...] → GND"""
        n = self._rng.randint(2, 4)
        v_val = self._rand_voltage_value()
        comps = [_make_voltage_source("V1", v_val)]
        r_vals = [self._rand_resistor_value() for _ in range(n)]
        for i in range(n):
            comps.append(_make_resistor(f"R{i+1}", r_vals[i]))

        lines = [f"* VoltSnap: parallel_resistors"]
        lines.append(f"V1 N1 0 DC {v_val}")
        for i in range(n):
            lines.append(f"R{i+1} N1 0 {r_vals[i]}")
        lines.append(".op")
        lines.append(".end")
        return CircuitSpec(name="parallel_resistors", components=comps, expected_netlist="\n".join(lines) + "\n")

    def _gen_resistor_divider(self) -> CircuitSpec:
        """分压器: V1 → R1 → R2 → GND"""
        v_val = self._rand_voltage_value()
        r1_val = self._rand_resistor_value()
        r2_val = self._rand_resistor_value()
        comps = [
            _make_voltage_source("V1", v_val),
            _make_resistor("R1", r1_val),
            _make_resistor("R2", r2_val),
        ]
        netlist = (
            f"* VoltSnap: resistor_divider\n"
            f"V1 N1 0 DC {v_val}\n"
            f"R1 N1 N2 {r1_val}\n"
            f"R2 N2 0 {r2_val}\n"
            f".op\n"
            f".end\n"
        )
        return CircuitSpec(name="resistor_divider", components=comps, expected_netlist=netlist)

    def _gen_rc_series(self) -> CircuitSpec:
        """RC 串联: V1 → R1 → C1 → GND"""
        v_val = self._rand_voltage_value()
        r_val = self._rand_resistor_value()
        c_val = self._rand_capacitor_value()
        comps = [
            _make_voltage_source("V1", v_val),
            _make_resistor("R1", r_val),
            _make_capacitor("C1", c_val),
        ]
        netlist = (
            f"* VoltSnap: rc_series\n"
            f"V1 N1 0 DC {v_val}\n"
            f"R1 N1 N2 {r_val}\n"
            f"C1 N2 0 {c_val}\n"
            f".op\n"
            f".end\n"
        )
        return CircuitSpec(name="rc_series", components=comps, expected_netlist=netlist)

    def _gen_rc_parallel(self) -> CircuitSpec:
        """RC 并联: V1 → R1, C1 并联到 GND"""
        v_val = self._rand_voltage_value()
        r_val = self._rand_resistor_value()
        c_val = self._rand_capacitor_value()
        comps = [
            _make_voltage_source("V1", v_val),
            _make_resistor("R1", r_val),
            _make_capacitor("C1", c_val),
        ]
        netlist = (
            f"* VoltSnap: rc_parallel\n"
            f"V1 N1 0 DC {v_val}\n"
            f"R1 N1 0 {r_val}\n"
            f"C1 N1 0 {c_val}\n"
            f".op\n"
            f".end\n"
        )
        return CircuitSpec(name="rc_parallel", components=comps, expected_netlist=netlist)

    def _gen_rl_series(self) -> CircuitSpec:
        """RL 串联: V1 → R1 → L1 → GND"""
        v_val = self._rand_voltage_value()
        r_val = self._rand_resistor_value()
        l_val = self._rand_inductor_value()
        comps = [
            _make_voltage_source("V1", v_val),
            _make_resistor("R1", r_val),
            _make_inductor("L1", l_val),
        ]
        netlist = (
            f"* VoltSnap: rl_series\n"
            f"V1 N1 0 DC {v_val}\n"
            f"R1 N1 N2 {r_val}\n"
            f"L1 N2 0 {l_val}\n"
            f".op\n"
            f".end\n"
        )
        return CircuitSpec(name="rl_series", components=comps, expected_netlist=netlist)

    def _gen_rlc_series(self) -> CircuitSpec:
        """RLC 串联: V1 → R1 → L1 → C1 → GND"""
        v_val = self._rand_voltage_value()
        r_val = self._rand_resistor_value()
        l_val = self._rand_inductor_value()
        c_val = self._rand_capacitor_value()
        comps = [
            _make_voltage_source("V1", v_val),
            _make_resistor("R1", r_val),
            _make_inductor("L1", l_val),
            _make_capacitor("C1", c_val),
        ]
        netlist = (
            f"* VoltSnap: rlc_series\n"
            f"V1 N1 0 DC {v_val}\n"
            f"R1 N1 N2 {r_val}\n"
            f"L1 N2 N3 {l_val}\n"
            f"C1 N3 0 {c_val}\n"
            f".op\n"
            f".end\n"
        )
        return CircuitSpec(name="rlc_series", components=comps, expected_netlist=netlist)

    def _gen_two_mesh(self) -> CircuitSpec:
        """双网孔电路: V1 → R1 → [R2 ‖ R3] → R4 → GND"""
        v_val = self._rand_voltage_value()
        r_vals = [self._rand_resistor_value() for _ in range(4)]
        comps = [
            _make_voltage_source("V1", v_val),
            _make_resistor("R1", r_vals[0]),
            _make_resistor("R2", r_vals[1]),
            _make_resistor("R3", r_vals[2]),
            _make_resistor("R4", r_vals[3]),
        ]
        netlist = (
            f"* VoltSnap: two_mesh\n"
            f"V1 N1 0 DC {v_val}\n"
            f"R1 N1 N2 {r_vals[0]}\n"
            f"R2 N2 N3 {r_vals[1]}\n"
            f"R3 N2 N3 {r_vals[2]}\n"
            f"R4 N3 0 {r_vals[3]}\n"
            f".op\n"
            f".end\n"
        )
        return CircuitSpec(name="two_mesh", components=comps, expected_netlist=netlist)

    # ── 辅助方法 ───────────────────────────────────────────────────────

    def _build_series_netlist(self, source_ref: str, resistors: list[ComponentInfo], v_val: str) -> str:
        lines = [f"* VoltSnap: series_resistors"]
        lines.append(f"{source_ref} N1 0 DC {v_val}")
        for i, r in enumerate(resistors):
            n1 = f"N{i+1}"
            n2 = f"N{i+2}" if i < len(resistors) - 1 else "0"
            lines.append(f"{r.ref} {n1} {n2} {r.value}")
        lines.append(".op")
        lines.append(".end")
        return "\n".join(lines) + "\n"
