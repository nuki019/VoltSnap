"""识别管线 — 端到端识别流程"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from voltsnap.models import ComponentInfo, PinInfo

logger = logging.getLogger("voltsnap.recognition.pipeline")


@dataclass
class RecognitionResult:
    """识别管线输出"""
    image_path: str
    components: list[ComponentInfo]
    bound_components: list[dict]
    ocr_results: list[dict]
    detections: list[dict]
    pin_to_net: dict[str, int]
    netlist: str
    success: bool = True
    error_message: str | None = None


class RecognitionPipeline:
    """
    端到端识别管线。

    流程：图像 → 元件检测 → OCR → 文字绑定 → 拓扑重建 → 网表生成
    """

    def __init__(
        self,
        model_path: str | None = None,
        use_gpu: bool = True,
    ):
        """
        Parameters
        ----------
        model_path : str | None
            YOLO OBB 模型路径。None 使用默认预训练模型。
        use_gpu : bool
            是否使用 GPU 推理。
        """
        self._model_path = model_path
        self._use_gpu = use_gpu
        self._detector = None
        self._ocr_parser = None
        self._binder = None

    def _init_modules(self):
        """延迟初始化各模块"""
        if self._detector is not None:
            return

        from voltsnap.recognition.detector import ComponentDetector
        from voltsnap.recognition.ocr_parser import OCRParser
        from voltsnap.recognition.text_binding import TextBinder

        self._detector = ComponentDetector(model_path=self._model_path)
        self._ocr_parser = OCRParser()
        self._binder = TextBinder()

    def process(
        self,
        image_path: str | Path,
        conf_threshold: float = 0.25,
    ) -> RecognitionResult:
        """
        识别单张电路图。

        Parameters
        ----------
        image_path : str | Path
            输入图像路径。
        conf_threshold : float
            检测置信度阈值。

        Returns
        -------
        RecognitionResult
            识别结果。
        """
        self._init_modules()
        image_path = Path(image_path)

        if not image_path.exists():
            return RecognitionResult(
                image_path=str(image_path),
                components=[],
                bound_components=[],
                ocr_results=[],
                detections=[],
                pin_to_net={},
                netlist="",
                success=False,
                error_message=f"Image not found: {image_path}",
            )

        image = cv2.imread(str(image_path))
        if image is None:
            return RecognitionResult(
                image_path=str(image_path),
                components=[],
                bound_components=[],
                ocr_results=[],
                detections=[],
                pin_to_net={},
                netlist="",
                success=False,
                error_message=f"Failed to load image: {image_path}",
            )

        try:
            return self._process_image(image, str(image_path), conf_threshold)
        except Exception as e:
            logger.error("Recognition failed: %s", e, exc_info=True)
            return RecognitionResult(
                image_path=str(image_path),
                components=[],
                bound_components=[],
                ocr_results=[],
                detections=[],
                pin_to_net={},
                netlist="",
                success=False,
                error_message=str(e),
            )

    def _process_image(
        self,
        image: np.ndarray,
        image_path: str,
        conf_threshold: float,
    ) -> RecognitionResult:
        """处理单张图像"""
        # 1. 元件检测
        detections = self._detector.detect(image, conf_threshold=conf_threshold)
        det_dicts = [
            {
                "class_name": d.class_name,
                "class_id": d.class_id,
                "bbox": d.bbox,
                "center": d.center,
                "confidence": d.confidence,
                "angle": d.angle,
            }
            for d in detections
        ]

        # 2. OCR 识别（使用合成标注的 OCR 结果）
        # 阶段 2 MVP：从标注文件中读取已知 OCR 结果
        ocr_results = self._extract_ocr_from_annotations(image_path)
        parsed_ocr = self._ocr_parser.parse(ocr_results)

        # 3. 文字-元件绑定
        bound = self._binder.bind(
            det_dicts,
            [vars(r) for r in parsed_ocr],
        )

        # 4. 转换为 ComponentInfo
        components = []
        for b in bound:
            if not b.ref:
                continue
            pins = self._infer_pins(b)
            components.append(ComponentInfo(
                ref=b.ref,
                type=b.type,
                value=b.value or "1k",
                pins=pins,
            ))

        # 5. 生成网表
        pin_to_net, netlist = self._generate_netlist(components, bound)

        return RecognitionResult(
            image_path=image_path,
            components=components,
            bound_components=[vars(b) for b in bound],
            ocr_results=[vars(r) for r in parsed_ocr],
            detections=det_dicts,
            pin_to_net=pin_to_net,
            netlist=netlist,
        )

    def _extract_ocr_from_annotations(self, image_path: str) -> list[dict]:
        """
        从标注文件中提取 OCR 结果。

        阶段 2 MVP：直接读取标注中的元件信息，模拟 OCR 输出。
        后续替换为真实 OCR 模型。
        """
        ann_path = Path(image_path).parent / "annotation.json"
        if not ann_path.exists():
            return []

        import json
        with open(ann_path, encoding="utf-8") as f:
            ann = json.load(f)

        ocr_results = []
        for comp in ann.get("components", []):
            # 模拟 OCR 输出元件编号
            ocr_results.append({
                "text": comp["ref"],
                "bbox": [0, 0, 50, 20],
                "confidence": 0.95,
                "is_ref": True,
                "normalized": comp["ref"].upper(),
                "component_type": comp["type"],
            })
            # 模拟 OCR 输出参数值
            ocr_results.append({
                "text": comp["value"],
                "bbox": [0, 25, 50, 45],
                "confidence": 0.90,
                "is_value": True,
                "normalized": comp["value"].lower(),
                "component_type": comp["type"],
            })

        return ocr_results

    def _infer_pins(self, bound) -> list[PinInfo]:
        """从绑定结果推断引脚位置"""
        cx, cy = bound.center
        x1, y1, x2, y2 = bound.bbox
        w = x2 - x1
        h = y2 - y1

        angle_rad = np.radians(bound.angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        # 二端元件：两个引脚在两端
        hw = max(w, h) / 2
        pin1_x = int(cx - hw * cos_a)
        pin1_y = int(cy - hw * sin_a)
        pin2_x = int(cx + hw * cos_a)
        pin2_y = int(cy + hw * sin_a)

        return [
            PinInfo(name=f"{bound.ref}_pin1", component_ref=bound.ref, position=(0, 0),
                    pixel_position=(pin1_x, pin1_y)),
            PinInfo(name=f"{bound.ref}_pin2", component_ref=bound.ref, position=(0, 0),
                    pixel_position=(pin2_x, pin2_y)),
        ]

    def _generate_netlist(
        self,
        components: list[ComponentInfo],
        bound_components: list[dict],
    ) -> tuple[dict[str, int], str]:
        """从检测结果生成网表"""
        if not components:
            return {}, ""

        # 简单拓扑：按引脚位置聚类形成 Net
        all_pins = []
        for comp in components:
            for pin in comp.pins:
                all_pins.append(pin)

        # 基于像素距离聚类
        pin_to_net: dict[str, int] = {}
        net_centers: list[tuple[int, int]] = []
        next_net_id = 1

        for pin in all_pins:
            px, py = pin.pixel_position
            assigned = False

            for net_id, (nx, ny) in enumerate(net_centers, 1):
                dist = ((px - nx) ** 2 + (py - ny) ** 2) ** 0.5
                if dist < 30:  # 30px 聚类半径
                    pin_to_net[pin.name] = net_id
                    # 更新 Net 中心
                    net_centers[net_id - 1] = (
                        (nx + px) // 2,
                        (ny + py) // 2,
                    )
                    assigned = True
                    break

            if not assigned:
                pin_to_net[pin.name] = next_net_id
                net_centers.append((px, py))
                next_net_id += 1

        # 生成 SPICE 网表
        netlist = self._components_to_spice(components, pin_to_net)

        return pin_to_net, netlist

    @staticmethod
    def _components_to_spice(
        components: list[ComponentInfo],
        pin_to_net: dict[str, int],
    ) -> str:
        """生成 SPICE 网表字符串"""
        lines = ["* VoltSnap Recognition Netlist"]

        # 找 GND net（最常出现的 net）
        net_counts: dict[int, int] = {}
        for net_id in pin_to_net.values():
            net_counts[net_id] = net_counts.get(net_id, 0) + 1
        gnd_net = max(net_counts, key=net_counts.get) if net_counts else 0

        def _node(pin_name: str) -> str:
            net_id = pin_to_net.get(pin_name, 0)
            return "0" if net_id == gnd_net else f"N{net_id}"

        has_diode = False
        has_op_amp = False
        has_bjt_npn = False
        has_bjt_pnp = False
        has_nmos = False
        has_pmos = False

        for comp in components:
            if len(comp.pins) < 2:
                continue

            n1 = _node(comp.pins[0].name)
            n2 = _node(comp.pins[1].name)

            prefix_map = {
                "resistor": "R",
                "capacitor": "C",
                "inductor": "L",
                "voltage_source": "V",
                "current_source": "I",
                "diode": "D",
                "op_amp": "X",
                "led": "D",
                "switch": "S",
                "npn_transistor": "Q",
                "pnp_transistor": "Q",
                "nmos": "M",
                "pmos": "M",
            }
            prefix = prefix_map.get(comp.type, "R")

            if comp.type == "voltage_source":
                lines.append(f"{prefix}{comp.ref} {n1} {n2} DC {comp.value}")
            elif comp.type == "current_source":
                lines.append(f"{prefix}{comp.ref} {n1} {n2} DC {comp.value}")
            elif comp.type == "diode":
                lines.append(f"{prefix}{comp.ref} {n1} {n2} DMOD")
                has_diode = True
            elif comp.type == "led":
                lines.append(f"{prefix}{comp.ref} {n1} {n2} LEDMOD")
                has_diode = True  # LED 模型追加在 diode 之后
            elif comp.type == "op_amp":
                # 运放使用子电路引用，3 引脚: in+, in-, out
                has_op_amp = True
                n3 = _node(comp.pins[2].name) if len(comp.pins) >= 3 else "NC"
                lines.append(f"{prefix}{comp.ref} {n1} {n2} {n3} OPAMP")
            elif comp.type == "ground":
                # 接地符号不生成 SPICE 行（通过 net 0 隐式处理）
                pass
            elif comp.type == "switch":
                lines.append(f"{prefix}{comp.ref} {n1} {n2} SWMOD")
            elif comp.type == "npn_transistor":
                # BJT: collector, base, emitter
                n3 = _node(comp.pins[2].name) if len(comp.pins) >= 3 else "0"
                lines.append(f"{prefix}{comp.ref} {n1} {n2} {n3} NPNMOD")
                has_bjt_npn = True
            elif comp.type == "pnp_transistor":
                n3 = _node(comp.pins[2].name) if len(comp.pins) >= 3 else "0"
                lines.append(f"{prefix}{comp.ref} {n1} {n2} {n3} PNPMOD")
                has_bjt_pnp = True
            elif comp.type == "nmos":
                # MOSFET: drain, gate, source, body(=source)
                n3 = _node(comp.pins[2].name) if len(comp.pins) >= 3 else "0"
                n4 = n3  # body = source
                if len(comp.pins) >= 4:
                    n4 = _node(comp.pins[3].name)
                lines.append(f"{prefix}{comp.ref} {n1} {n2} {n3} {n4} NMOSMOD")
                has_nmos = True
            elif comp.type == "pmos":
                n3 = _node(comp.pins[2].name) if len(comp.pins) >= 3 else "0"
                n4 = n3
                if len(comp.pins) >= 4:
                    n4 = _node(comp.pins[3].name)
                lines.append(f"{prefix}{comp.ref} {n1} {n2} {n3} {n4} PMOSMOD")
                has_pmos = True
            else:
                lines.append(f"{prefix}{comp.ref} {n1} {n2} {comp.value}")

        # 追加必需的模型/子电路定义
        if has_diode:
            lines.append(".model DMOD D")
            lines.append(".model LEDMOD D(Is=1e-20 N=2 Rs=0.1 BV=5 IBV=10u)")
        if has_op_amp:
            lines.append(".subckt OPAMP in+ in- out")
            lines.append("E1 out 0 in+ in- 100k")
            lines.append(".ends OPAMP")
        if has_bjt_npn:
            lines.append(".model NPNMOD NPN(Is=1e-14 Bf=100)")
        if has_bjt_pnp:
            lines.append(".model PNPMOD PNP(Is=1e-14 Bf=100)")
        if has_nmos:
            lines.append(".model NMOSMOD NMOS(Vto=1 Kp=2e-4)")
        if has_pmos:
            lines.append(".model PMOSMOD PMOS(Vto=-1 Kp=1e-4)")

        lines.append(".op")
        lines.append(".end")
        return "\n".join(lines) + "\n"
