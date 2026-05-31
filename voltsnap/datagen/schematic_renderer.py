"""schemdraw 电路图渲染与坐标提取"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")  # 无头后端，不弹窗

import schemdraw
import schemdraw.elements as elm

from voltsnap.models import CircuitSpec, ComponentInfo, PinInfo, RenderResult

logger = logging.getLogger("voltsnap.datagen.renderer")


class SchematicRenderer:
    """用 schemdraw 渲染电路图并提取元件像素坐标"""

    def __init__(self, dpi: int = 150, inches_per_unit: float = 3.0):
        self.dpi = dpi
        self.inches_per_unit = inches_per_unit

    def render(self, circuit: CircuitSpec, output_path: str) -> RenderResult:
        """
        绘制电路图，保存 PNG，提取元件引脚像素坐标。

        阶段 0 的电路模板使用固定的绘制顺序，通过 schemdraw 元素的
        .start/.end 属性提取坐标，再转换为像素坐标。
        """
        logger.info("Rendering circuit: %s", circuit.name)

        d = schemdraw.Drawing()
        elements: list[tuple[ComponentInfo, object]] = []

        # 按电路名称选择绘制逻辑
        draw_map = {
            "resistor_divider": self._draw_resistor_divider,
            "parallel_resistors": self._draw_parallel_resistors,
            "rc_circuit": self._draw_rc_circuit,
            "diode_rectifier": self._draw_diode_rectifier,
            "op_amp_inverting": self._draw_op_amp_inverting,
            # 随机拓扑（名称可能带编号后缀，用 startswith 匹配）
        }

        draw_fn = draw_map.get(circuit.name)
        if draw_fn:
            elements = draw_fn(d, circuit)
        elif circuit.name.startswith("series_resistors"):
            elements = self._draw_series_resistors(d, circuit)
        elif circuit.name.startswith("rc_series"):
            elements = self._draw_rc_series(d, circuit)
        elif circuit.name.startswith("rc_parallel"):
            elements = self._draw_rc_parallel(d, circuit)
        elif circuit.name.startswith("rl_series"):
            elements = self._draw_rl_series(d, circuit)
        elif circuit.name.startswith("rlc_series"):
            elements = self._draw_rlc_series(d, circuit)
        elif circuit.name.startswith("two_mesh"):
            elements = self._draw_two_mesh(d, circuit)
        elif circuit.name.startswith("diode_rectifier") or circuit.name.startswith("diode_series"):
            elements = self._draw_diode_rectifier(d, circuit)
        elif circuit.name.startswith("op_amp_inverting"):
            elements = self._draw_op_amp_inverting(d, circuit)
        else:
            # 通用回退：按元件类型顺序绘制
            elements = self._draw_generic(d, circuit)

        # 保存图片（这会创建 Figure 对象）
        d.save(output_path, dpi=self.dpi)
        logger.info("Saved image: %s", output_path)

        # 获取图片尺寸（像素）：d.fig 是 schemdraw Figure 包装器，d.fig.fig 是真实 matplotlib Figure
        mpl_fig = d.fig.fig
        w, h = int(mpl_fig.get_figwidth() * self.dpi), int(mpl_fig.get_figheight() * self.dpi)

        # 提取坐标
        self._extract_positions(d, elements, img_height=h)

        # 更新 circuit.components 的引脚坐标
        updated_components = [comp for comp, _ in elements]

        return RenderResult(
            image_path=output_path,
            components=updated_components,
            image_size=(w, h),
        )

    def _draw_resistor_divider(
        self, d: schemdraw.Drawing, circuit: CircuitSpec
    ) -> list[tuple[ComponentInfo, object]]:
        """绘制串联分压电路: V1(5V) -> R1(1k) -> R2(2k) -> GND"""
        comps = {c.ref: c for c in circuit.components}

        v1_elm = d.add(elm.SourceV().label(f"V1\n{comps['V1'].value}V").down())
        r1_elm = d.add(elm.Resistor().right().label(f"R1\n{comps['R1'].value}"))
        r2_elm = d.add(elm.Resistor().label(f"R2\n{comps['R2'].value}"))
        d.add(elm.Ground())
        # 用导线闭合回路（从 R2 底部回到 V1 负极）
        d.add(elm.Line().left().tox(v1_elm.start).color("black"))

        return [
            (comps["V1"], v1_elm),
            (comps["R1"], r1_elm),
            (comps["R2"], r2_elm),
        ]

    def _draw_parallel_resistors(
        self, d: schemdraw.Drawing, circuit: CircuitSpec
    ) -> list[tuple[ComponentInfo, object]]:
        """绘制并联电阻电路: V1(10V) -> [R1(1k) || R2(2k)] -> GND"""
        comps = {c.ref: c for c in circuit.components}
        resistors = sorted(
            [c for c in circuit.components if c.type == "resistor"],
            key=lambda c: c.ref,
        )

        v1_elm = d.add(elm.SourceV().label(f"V1\n{comps['V1'].value}V").down())

        # 上支路 R1
        r1_elm = d.add(elm.Resistor().right().label(f"{resistors[0].ref}\n{resistors[0].value}"))
        d.add(elm.Ground())

        # 下支路 R2（向下偏移后从 V1 底部开始）
        d.push()
        d.add(elm.Line().at(v1_elm.end).down().length(2))  # 偏移
        r2_elm = d.add(elm.Resistor().right().label(f"{resistors[1].ref}\n{resistors[1].value}"))
        d.add(elm.Ground())
        d.pop()

        return [
            (comps["V1"], v1_elm),
            (resistors[0], r1_elm),
            (resistors[1], r2_elm),
        ]

    def _draw_rc_circuit(
        self, d: schemdraw.Drawing, circuit: CircuitSpec
    ) -> list[tuple[ComponentInfo, object]]:
        """绘制 RC 电路: V1(5V) -> R1(1k) -> C1(10uF) -> GND"""
        comps = {c.ref: c for c in circuit.components}

        v1_elm = d.add(elm.SourceV().label(f"V1\n{comps['V1'].value}V").down())
        r1_elm = d.add(elm.Resistor().right().label(f"R1\n{comps['R1'].value}"))
        c1_elm = d.add(elm.Capacitor().label(f"C1\n{comps['C1'].value}F"))
        d.add(elm.Ground())
        d.add(elm.Line().left().tox(v1_elm.start).color("black"))

        return [
            (comps["V1"], v1_elm),
            (comps["R1"], r1_elm),
            (comps["C1"], c1_elm),
        ]

    # ── 随机拓扑绘制方法 ──────────────────────────────────────────────

    def _draw_series_resistors(
        self, d: schemdraw.Drawing, circuit: CircuitSpec
    ) -> list[tuple[ComponentInfo, object]]:
        """串联电阻链: V1 → R1 → R2 → ... → GND"""
        comps = {c.ref: c for c in circuit.components}
        resistors = [c for c in circuit.components if c.type == "resistor"]
        resistors.sort(key=lambda c: c.ref)

        v1 = comps["V1"]
        v1_elm = d.add(elm.SourceV().label(f"V1\n{v1.value}V").down())
        result = [(v1, v1_elm)]

        for r in resistors:
            r_elm = d.add(elm.Resistor().right().label(f"{r.ref}\n{r.value}"))
            result.append((r, r_elm))

        d.add(elm.Ground())
        d.add(elm.Line().left().tox(v1_elm.start).color("black"))
        return result

    def _draw_rc_series(
        self, d: schemdraw.Drawing, circuit: CircuitSpec
    ) -> list[tuple[ComponentInfo, object]]:
        """RC 串联: V1 → R1 → C1 → GND"""
        comps = {c.ref: c for c in circuit.components}
        v1 = comps["V1"]
        r1 = next(c for c in circuit.components if c.type == "resistor")
        c1 = next(c for c in circuit.components if c.type == "capacitor")

        v1_elm = d.add(elm.SourceV().label(f"V1\n{v1.value}V").down())
        r1_elm = d.add(elm.Resistor().right().label(f"{r1.ref}\n{r1.value}"))
        c1_elm = d.add(elm.Capacitor().label(f"{c1.ref}\n{c1.value}F"))
        d.add(elm.Ground())
        d.add(elm.Line().left().tox(v1_elm.start).color("black"))

        return [(v1, v1_elm), (r1, r1_elm), (c1, c1_elm)]

    def _draw_rc_parallel(
        self, d: schemdraw.Drawing, circuit: CircuitSpec
    ) -> list[tuple[ComponentInfo, object]]:
        """RC 并联: V1 → [R1 ‖ C1] → GND"""
        comps = {c.ref: c for c in circuit.components}
        v1 = comps["V1"]
        r1 = next(c for c in circuit.components if c.type == "resistor")
        c1 = next(c for c in circuit.components if c.type == "capacitor")

        v1_elm = d.add(elm.SourceV().label(f"V1\n{v1.value}V").down())
        r1_elm = d.add(elm.Resistor().right().label(f"{r1.ref}\n{r1.value}"))
        d.add(elm.Ground())
        d.push()
        d.add(elm.Line().at(v1_elm.end).down().length(2))  # 偏移
        c1_elm = d.add(elm.Capacitor().right().label(f"{c1.ref}\n{c1.value}F"))
        d.add(elm.Ground())
        d.pop()

        return [(v1, v1_elm), (r1, r1_elm), (c1, c1_elm)]

    def _draw_rl_series(
        self, d: schemdraw.Drawing, circuit: CircuitSpec
    ) -> list[tuple[ComponentInfo, object]]:
        """RL 串联: V1 → R1 → L1 → GND"""
        comps = {c.ref: c for c in circuit.components}
        v1 = comps["V1"]
        r1 = next(c for c in circuit.components if c.type == "resistor")
        l1 = next(c for c in circuit.components if c.type == "inductor")

        v1_elm = d.add(elm.SourceV().label(f"V1\n{v1.value}V").down())
        r1_elm = d.add(elm.Resistor().right().label(f"{r1.ref}\n{r1.value}"))
        l1_elm = d.add(elm.Inductor().label(f"{l1.ref}\n{l1.value}H"))
        d.add(elm.Ground())
        d.add(elm.Line().left().tox(v1_elm.start).color("black"))

        return [(v1, v1_elm), (r1, r1_elm), (l1, l1_elm)]

    def _draw_rlc_series(
        self, d: schemdraw.Drawing, circuit: CircuitSpec
    ) -> list[tuple[ComponentInfo, object]]:
        """RLC 串联: V1 → R1 → L1 → C1 → GND"""
        comps = {c.ref: c for c in circuit.components}
        v1 = comps["V1"]
        r1 = next(c for c in circuit.components if c.type == "resistor")
        l1 = next(c for c in circuit.components if c.type == "inductor")
        c1 = next(c for c in circuit.components if c.type == "capacitor")

        v1_elm = d.add(elm.SourceV().label(f"V1\n{v1.value}V").down())
        r1_elm = d.add(elm.Resistor().right().label(f"{r1.ref}\n{r1.value}"))
        l1_elm = d.add(elm.Inductor().label(f"{l1.ref}\n{l1.value}H"))
        c1_elm = d.add(elm.Capacitor().label(f"{c1.ref}\n{c1.value}F"))
        d.add(elm.Ground())
        d.add(elm.Line().left().tox(v1_elm.start).color("black"))

        return [(v1, v1_elm), (r1, r1_elm), (l1, l1_elm), (c1, c1_elm)]

    def _draw_two_mesh(
        self, d: schemdraw.Drawing, circuit: CircuitSpec
    ) -> list[tuple[ComponentInfo, object]]:
        """双网孔电路: V1 → R1 → [R2 ‖ R3] → R4 → GND"""
        comps = {c.ref: c for c in circuit.components}
        resistors = sorted(
            [c for c in circuit.components if c.type == "resistor"],
            key=lambda c: c.ref,
        )

        v1 = comps["V1"]
        v1_elm = d.add(elm.SourceV().label(f"V1\n{v1.value}V").down())
        r1_elm = d.add(elm.Resistor().right().label(f"{resistors[0].ref}\n{resistors[0].value}"))

        # R2 上支路
        r2_elm = d.add(elm.Resistor().right().label(f"{resistors[1].ref}\n{resistors[1].value}"))
        r4_elm = d.add(elm.Resistor().right().label(f"{resistors[3].ref}\n{resistors[3].value}"))
        d.add(elm.Ground())

        # R3 下支路（从 R1 右端开始）
        d.push()
        d.add(elm.Resistor().at(r1_elm.end).right().label(f"{resistors[2].ref}\n{resistors[2].value}"))
        r3_elm = d.elements[-1]
        d.add(elm.Line().tox(r4_elm.start))
        d.pop()

        d.add(elm.Line().left().tox(v1_elm.start).color("black"))

        return [(v1, v1_elm), (resistors[0], r1_elm), (resistors[1], r2_elm),
                (resistors[2], r3_elm), (resistors[3], r4_elm)]

    def _draw_diode_rectifier(
        self, d: schemdraw.Drawing, circuit: CircuitSpec
    ) -> list[tuple[ComponentInfo, object]]:
        """半波整流电路: V1 → D1 → R1 → GND"""
        comps = {c.ref: c for c in circuit.components}
        v1 = comps["V1"]
        d1 = next(c for c in circuit.components if c.type == "diode")
        r1 = next(c for c in circuit.components if c.type == "resistor")

        v1_elm = d.add(elm.SourceV().label(f"V1\n{v1.value}V").down())
        d1_elm = d.add(elm.Diode().right().label(f"{d1.ref}"))
        r1_elm = d.add(elm.Resistor().label(f"{r1.ref}\n{r1.value}"))
        d.add(elm.Ground())
        d.add(elm.Line().left().tox(v1_elm.start).color("black"))

        return [(v1, v1_elm), (d1, d1_elm), (r1, r1_elm)]

    def _draw_op_amp_inverting(
        self, d: schemdraw.Drawing, circuit: CircuitSpec
    ) -> list[tuple[ComponentInfo, object]]:
        """反相放大器: V1 → R1 → (-) U1 (+) → GND, Rf 反馈"""
        comps = {c.ref: c for c in circuit.components}
        v1 = comps["V1"]
        r1 = next(c for c in circuit.components
                   if c.type == "resistor" and c.ref == "R1")
        rf = next(c for c in circuit.components
                  if c.type == "resistor" and c.ref == "Rf")
        u1 = next(c for c in circuit.components if c.type == "op_amp")

        # 电压源
        v1_elm = d.add(elm.SourceV().label(f"V1\n{v1.value}V").down())

        # R1 从 V1 底部向右到运放反相输入
        r1_elm = d.add(elm.Resistor().right().label(f"R1\n{r1.value}"))

        # 运放：in2(反相) 接 R1 右端，in1(同相) 接 GND
        op_elm = d.add(elm.Opamp().anchor("in2").label("U1", loc="center"))

        # 同相输入端接地
        d.add(elm.Ground().at(op_elm.in1))

        # Rf 反馈：从运放输出到反相输入节点
        rf_elm = d.add(elm.Resistor().at(op_elm.out).left().to(r1_elm.end).label(f"Rf\n{rf.value}", loc="bottom"))

        # 闭合回路：从运放输出向右再向下接地，再回到 V1 负极
        d.add(elm.Line().at(op_elm.out).right().length(1))
        d.add(elm.Ground())

        return [(v1, v1_elm), (r1, r1_elm), (rf, rf_elm), (u1, op_elm)]

    def _draw_generic(
        self, d: schemdraw.Drawing, circuit: CircuitSpec
    ) -> list[tuple[ComponentInfo, object]]:
        """通用回退绘制：按元件类型顺序绘制串联链"""
        comps = {c.ref: c for c in circuit.components}
        source = next((c for c in circuit.components if c.type == "voltage_source"), None)
        others = [c for c in circuit.components if c.type != "voltage_source"]
        others.sort(key=lambda c: c.ref)

        result = []
        if source:
            v_elm = d.add(elm.SourceV().label(f"V1\n{source.value}V").down())
            result.append((source, v_elm))

        elm_map = {
            "resistor": elm.Resistor,
            "capacitor": elm.Capacitor,
            "inductor": elm.Inductor,
            "diode": elm.Diode,
        }
        label_suffix = {
            "resistor": "",
            "capacitor": "F",
            "inductor": "H",
            "diode": "",
        }

        has_op_amp = False
        for comp in others:
            if comp.type == "op_amp":
                op_elm = d.add(elm.Opamp().anchor("in2").label(comp.ref, loc="center"))
                result.append((comp, op_elm))
                has_op_amp = True
                continue
            e_cls = elm_map.get(comp.type, elm.Resistor)
            suffix = label_suffix.get(comp.type, "")
            e = d.add(e_cls().right().label(f"{comp.ref}\n{comp.value}{suffix}"))
            result.append((comp, e))

        d.add(elm.Ground())
        if result:
            first = result[0][1]
            if hasattr(first, "start"):
                d.add(elm.Line().left().tox(first.start).color("black"))

        return result

    def _extract_positions(
        self,
        d: schemdraw.Drawing,
        elements: list[tuple[ComponentInfo, object]],
        img_height: int,
    ) -> None:
        """
        从 schemdraw 元素中提取引脚坐标，转换为像素坐标。

        schemdraw 坐标系：原点任意，Y 轴向上，单位是 schemdraw units。
        d.get_bbox() 返回绘图区域边界。
        d.fig.inches_per_unit 给出每个 unit 对应的英寸数。
        像素坐标系：原点在图片左上角，Y 轴向下。

        转换公式（减去 bbox 偏移，翻转 Y 轴）：
          pixel_x = (x - bbox.xmin) * inches_per_unit * dpi
          pixel_y = img_height - (y - bbox.ymin) * inches_per_unit * dpi
        """
        inches_per_unit = d.fig.inches_per_unit
        bbox = d.get_bbox()

        for comp, elm_obj in elements:
            center = elm_obj.center

            def to_pixel(pt):
                px = int((pt[0] - bbox.xmin) * inches_per_unit * self.dpi)
                py = int(img_height - (pt[1] - bbox.ymin) * inches_per_unit * self.dpi)
                return (px, py)

            comp.bbox_center = to_pixel(center)
            try:
                comp.angle = elm_obj.theta()
            except Exception:
                comp.angle = 0.0

            # Opamp 等多引脚元件：通过 anchors 映射
            if hasattr(elm_obj, 'anchors') and 'in1' in elm_obj.anchors:
                anchor_map = {
                    0: 'in1',   # 非反相输入
                    1: 'in2',   # 反相输入
                    2: 'out',   # 输出
                }
                for idx, anchor_name in anchor_map.items():
                    if idx < len(comp.pins):
                        pt = getattr(elm_obj, anchor_name)
                        comp.pins[idx].position = (float(pt[0]), float(pt[1]))
                        comp.pins[idx].pixel_position = to_pixel(pt)
                # 用 in1 和 out 更新 bbox_center
                try:
                    in1_pt = elm_obj.in1
                    out_pt = elm_obj.out
                    mid = ((in1_pt[0] + out_pt[0]) / 2, (in1_pt[1] + out_pt[1]) / 2)
                    comp.bbox_center = to_pixel(mid)
                except Exception:
                    pass
            else:
                # 二端元件：start/end
                start = elm_obj.start
                end = elm_obj.end
                if len(comp.pins) >= 2:
                    comp.pins[0].position = (float(start[0]), float(start[1]))
                    comp.pins[0].pixel_position = to_pixel(start)
                    comp.pins[1].position = (float(end[0]), float(end[1]))
                    comp.pins[1].pixel_position = to_pixel(end)
