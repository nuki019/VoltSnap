"""阶段 2 识别模块测试"""
import json
from pathlib import Path

import numpy as np
import pytest

from voltsnap.recognition.annotation_converter import AnnotationConverter, COMPONENT_CLASSES
from voltsnap.recognition.ocr_parser import OCRParser, standardize_value
from voltsnap.recognition.pipeline import RecognitionPipeline
from voltsnap.recognition.text_binding import TextBinder
from voltsnap.models import ComponentInfo, PinInfo


# ── 标注转换器测试 ────────────────────────────────────────────────────


class TestAnnotationConverter:
    """YOLO OBB 标注格式转换测试"""

    @pytest.fixture
    def sample_annotation(self):
        return {
            "sample_id": "test_000001",
            "image_size": [640, 480],
            "components": [
                {"ref": "V1", "type": "voltage_source", "value": "5"},
                {"ref": "R1", "type": "resistor", "value": "1k"},
                {"ref": "R2", "type": "resistor", "value": "2k"},
            ],
            "pin_positions": {
                "V1_pin1": [100, 50],
                "V1_pin2": [100, 200],
                "R1_pin1": [100, 200],
                "R1_pin2": [300, 200],
                "R2_pin1": [300, 200],
                "R2_pin2": [300, 400],
            },
        }

    def test_convert_sample(self, sample_annotation):
        converter = AnnotationConverter()
        detections = converter.convert_sample(sample_annotation)

        assert len(detections) == 3
        for det in detections:
            assert 0 <= det.cx <= 1
            assert 0 <= det.cy <= 1
            assert det.w > 0
            assert det.h > 0
            assert det.class_id in COMPONENT_CLASSES.values()

    def test_class_mapping(self, sample_annotation):
        converter = AnnotationConverter()
        detections = converter.convert_sample(sample_annotation)

        type_to_class = {d.class_id for d in detections}
        assert COMPONENT_CLASSES["resistor"] in type_to_class
        assert COMPONENT_CLASSES["voltage_source"] in type_to_class

    def test_diode_class_id(self):
        assert COMPONENT_CLASSES["diode"] == 5

    def test_op_amp_class_id(self):
        assert COMPONENT_CLASSES["op_amp"] == 6

    def test_new_component_class_ids(self):
        """新增元件类型在 COMPONENT_CLASSES 中有定义"""
        assert COMPONENT_CLASSES["ground"] == 7
        assert COMPONENT_CLASSES["switch"] == 8
        assert COMPONENT_CLASSES["led"] == 9
        assert COMPONENT_CLASSES["npn_transistor"] == 10
        assert COMPONENT_CLASSES["pnp_transistor"] == 11
        assert COMPONENT_CLASSES["nmos"] == 12
        assert COMPONENT_CLASSES["pmos"] == 13

    def test_convert_diode_annotation(self):
        """二极管标注转换：2 引脚生成有效 OBB"""
        annotation = {
            "sample_id": "diode_001",
            "image_size": [640, 480],
            "components": [
                {"ref": "V1", "type": "voltage_source", "value": "5"},
                {"ref": "D1", "type": "diode", "value": "D"},
            ],
            "pin_positions": {
                "V1_pin1": [100, 50],
                "V1_pin2": [100, 200],
                "D1_anode": [200, 200],
                "D1_cathode": [350, 200],
            },
        }
        converter = AnnotationConverter()
        detections = converter.convert_sample(annotation)
        assert len(detections) == 2
        diode_det = [d for d in detections if d.class_id == COMPONENT_CLASSES["diode"]]
        assert len(diode_det) == 1
        assert 0 <= diode_det[0].cx <= 1
        assert 0 <= diode_det[0].cy <= 1

    def test_convert_op_amp_annotation(self):
        """运放标注转换：使用前两个引脚生成 OBB"""
        annotation = {
            "sample_id": "opamp_001",
            "image_size": [640, 480],
            "components": [
                {"ref": "U1", "type": "op_amp", "value": "OP"},
            ],
            "pin_positions": {
                "U1_in+": [200, 150],
                "U1_in-": [200, 250],
                "U1_out": [400, 200],
            },
        }
        converter = AnnotationConverter()
        detections = converter.convert_sample(annotation)
        assert len(detections) == 1
        assert detections[0].class_id == COMPONENT_CLASSES["op_amp"]

    def test_yolo_obb_line_format(self, sample_annotation):
        converter = AnnotationConverter()
        detections = converter.convert_sample(sample_annotation)

        for det in detections:
            line = converter.to_yolo_obb_line(det)
            parts = line.split()
            assert len(parts) == 9  # class_id + 8 coords
            assert parts[0].isdigit()
            for coord in parts[1:]:
                val = float(coord)
                assert 0 <= val <= 1

    def test_convert_and_save(self, sample_annotation, tmp_path):
        ann_path = tmp_path / "annotation.json"
        ann_path.write_text(json.dumps(sample_annotation), encoding="utf-8")

        out_path = tmp_path / "labels" / "test.txt"
        converter = AnnotationConverter()
        count = converter.convert_and_save(ann_path, out_path)

        assert count == 3
        assert out_path.exists()
        lines = out_path.read_text().strip().split("\n")
        assert len(lines) == 3


# ── OCR 解析器测试 ────────────────────────────────────────────────────


class TestOCRParser:
    """电学参数解析测试"""

    def test_parse_resistor_value(self):
        parser = OCRParser()
        results = parser.parse([
            {"text": "10k", "bbox": [0, 0, 50, 20], "confidence": 0.9},
        ])
        assert len(results) == 1
        assert results[0].is_value
        assert results[0].normalized == "10k"
        assert results[0].component_type == "resistor"

    def test_parse_capacitor_value(self):
        parser = OCRParser()
        results = parser.parse([
            {"text": "10uF", "bbox": [0, 0, 50, 20], "confidence": 0.9},
        ])
        assert len(results) == 1
        assert results[0].is_value
        assert results[0].normalized == "10uf"
        assert results[0].component_type == "capacitor"

    def test_parse_ref(self):
        parser = OCRParser()
        results = parser.parse([
            {"text": "R1", "bbox": [0, 0, 50, 20], "confidence": 0.9},
        ])
        assert len(results) == 1
        assert results[0].is_ref
        assert results[0].normalized == "R1"
        assert results[0].component_type == "resistor"

    def test_parse_voltage(self):
        parser = OCRParser()
        results = parser.parse([
            {"text": "5V", "bbox": [0, 0, 50, 20], "confidence": 0.9},
        ])
        assert len(results) == 1
        assert results[0].is_value
        assert "v" in results[0].normalized

    def test_standardize_4k7(self):
        assert standardize_value("4k7") == "4.7k"

    def test_standardize_micro(self):
        assert standardize_value("10μF") == "10uf"

    def test_standardize_mega(self):
        assert standardize_value("1MΩ") == "1meg"

    def test_parse_ref_ground(self):
        parser = OCRParser()
        results = parser.parse([
            {"text": "G1", "bbox": [0, 0, 50, 20], "confidence": 0.9},
        ])
        assert len(results) == 1
        assert results[0].is_ref
        assert results[0].component_type == "ground"

    def test_parse_ref_switch(self):
        parser = OCRParser()
        results = parser.parse([
            {"text": "S1", "bbox": [0, 0, 50, 20], "confidence": 0.9},
        ])
        assert len(results) == 1
        assert results[0].is_ref
        assert results[0].component_type == "switch"

    def test_parse_ref_led(self):
        parser = OCRParser()
        results = parser.parse([
            {"text": "LED1", "bbox": [0, 0, 50, 20], "confidence": 0.9},
        ])
        assert len(results) == 1
        assert results[0].is_ref
        assert results[0].component_type == "led"

    def test_parse_ref_npn(self):
        parser = OCRParser()
        results = parser.parse([
            {"text": "Q1", "bbox": [0, 0, 50, 20], "confidence": 0.9},
        ])
        assert len(results) == 1
        assert results[0].is_ref
        assert results[0].component_type == "npn_transistor"

    def test_parse_ref_nmos(self):
        parser = OCRParser()
        results = parser.parse([
            {"text": "M1", "bbox": [0, 0, 50, 20], "confidence": 0.9},
        ])
        assert len(results) == 1
        assert results[0].is_ref
        assert results[0].component_type == "nmos"


# ── 文字绑定测试 ──────────────────────────────────────────────────────


class TestTextBinder:
    """匈牙利算法绑定测试"""

    def test_bind_refs_and_values(self):
        binder = TextBinder()

        detections = [
            {
                "class_name": "resistor",
                "bbox": [100, 100, 200, 130],
                "center": [150, 115],
                "confidence": 0.9,
                "angle": 0.0,
            },
            {
                "class_name": "resistor",
                "bbox": [300, 100, 400, 130],
                "center": [350, 115],
                "confidence": 0.85,
                "angle": 0.0,
            },
        ]

        ocr_results = [
            {"text": "R1", "bbox": [110, 80, 140, 100], "is_ref": True,
             "is_value": False, "normalized": "R1", "component_type": "resistor"},
            {"text": "R2", "bbox": [310, 80, 340, 100], "is_ref": True,
             "is_value": False, "normalized": "R2", "component_type": "resistor"},
            {"text": "10k", "bbox": [130, 135, 170, 155], "is_ref": False,
             "is_value": True, "normalized": "10k", "component_type": "resistor"},
            {"text": "20k", "bbox": [330, 135, 370, 155], "is_ref": False,
             "is_value": True, "normalized": "20k", "component_type": "resistor"},
        ]

        bound = binder.bind(detections, ocr_results)

        assert len(bound) == 2
        # 应该正确绑定到最近的元件
        refs = {b.ref for b in bound}
        assert "R1" in refs or "R2" in refs

    def test_bind_empty(self):
        binder = TextBinder()
        bound = binder.bind([], [])
        assert bound == []

    def test_bind_no_ocr(self):
        binder = TextBinder()
        detections = [
            {"class_name": "resistor", "bbox": [100, 100, 200, 130],
             "center": [150, 115], "confidence": 0.9, "angle": 0.0},
        ]
        bound = binder.bind(detections, [])
        assert len(bound) == 1
        assert bound[0].ref == ""


# ── 集成测试（标注转换 + 绑定） ──────────────────────────────────────


class TestRecognitionIntegration:
    """识别模块集成测试"""

    def test_full_annotation_to_binding(self, tmp_path):
        """从标注到绑定的完整流程"""
        # 创建模拟标注
        annotation = {
            "sample_id": "integ_000001",
            "image_size": [640, 480],
            "components": [
                {"ref": "V1", "type": "voltage_source", "value": "5"},
                {"ref": "R1", "type": "resistor", "value": "1k"},
            ],
            "pin_positions": {
                "V1_pin1": [100, 50],
                "V1_pin2": [100, 200],
                "R1_pin1": [100, 200],
                "R1_pin2": [300, 200],
            },
        }

        # 转换标注
        converter = AnnotationConverter()
        detections = converter.convert_sample(annotation)
        assert len(detections) == 2

        # 模拟 OCR 输出
        ocr_results = [
            {"text": "V1", "bbox": [80, 30, 120, 50], "confidence": 0.95,
             "is_ref": True, "is_value": False, "normalized": "V1",
             "component_type": "voltage_source"},
            {"text": "R1", "bbox": [180, 180, 220, 200], "confidence": 0.90,
             "is_ref": True, "is_value": False, "normalized": "R1",
             "component_type": "resistor"},
            {"text": "5V", "bbox": [80, 60, 110, 80], "confidence": 0.85,
             "is_ref": False, "is_value": True, "normalized": "5v",
             "component_type": "voltage_source"},
            {"text": "1k", "bbox": [180, 205, 210, 225], "confidence": 0.80,
             "is_ref": False, "is_value": True, "normalized": "1k",
             "component_type": "resistor"},
        ]

        # 绑定
        binder = TextBinder()
        det_dicts = [
            {"class_name": "voltage_source", "bbox": [80, 30, 120, 200],
             "center": [100, 115], "confidence": 0.9, "angle": -90.0},
            {"class_name": "resistor", "bbox": [100, 180, 300, 220],
             "center": [200, 200], "confidence": 0.85, "angle": 0.0},
        ]
        bound = binder.bind(det_dicts, ocr_results)

        assert len(bound) == 2
        # 至少有一个成功绑定了编号
        has_ref = any(b.ref for b in bound)
        assert has_ref


# ── RecognitionPipeline 网表生成测试 ─────────────────────────────────


class TestComponentsToSpice:
    """RecognitionPipeline._components_to_spice 测试"""

    def test_diode_netlist_includes_model(self):
        """二极管网表应包含 .model DMOD D"""
        components = [
            ComponentInfo(
                ref="V1", type="voltage_source", value="5",
                pins=[
                    PinInfo(name="V1_pin1", component_ref="V1", position=(0, 0)),
                    PinInfo(name="V1_pin2", component_ref="V1", position=(0, 0)),
                ],
            ),
            ComponentInfo(
                ref="D1", type="diode", value="D",
                pins=[
                    PinInfo(name="D1_anode", component_ref="D1", position=(0, 0)),
                    PinInfo(name="D1_cathode", component_ref="D1", position=(0, 0)),
                ],
            ),
            ComponentInfo(
                ref="R1", type="resistor", value="1k",
                pins=[
                    PinInfo(name="R1_pin1", component_ref="R1", position=(0, 0)),
                    PinInfo(name="R1_pin2", component_ref="R1", position=(0, 0)),
                ],
            ),
        ]
        pin_to_net = {
            "V1_pin1": 1, "V1_pin2": 2,
            "D1_anode": 1, "D1_cathode": 3,
            "R1_pin1": 3, "R1_pin2": 2,
        }
        netlist = RecognitionPipeline._components_to_spice(components, pin_to_net)
        assert ".model DMOD D" in netlist
        assert "D1" in netlist
        assert "DMOD" in netlist

    def test_op_amp_netlist_includes_subcircuit(self):
        """运放网表应包含 .subckt OPAMP 定义"""
        components = [
            ComponentInfo(
                ref="V1", type="voltage_source", value="1",
                pins=[
                    PinInfo(name="V1_pin1", component_ref="V1", position=(0, 0)),
                    PinInfo(name="V1_pin2", component_ref="V1", position=(0, 0)),
                ],
            ),
            ComponentInfo(
                ref="R1", type="resistor", value="10k",
                pins=[
                    PinInfo(name="R1_pin1", component_ref="R1", position=(0, 0)),
                    PinInfo(name="R1_pin2", component_ref="R1", position=(0, 0)),
                ],
            ),
            ComponentInfo(
                ref="U1", type="op_amp", value="OP",
                pins=[
                    PinInfo(name="U1_in+", component_ref="U1", position=(0, 0)),
                    PinInfo(name="U1_in-", component_ref="U1", position=(0, 0)),
                    PinInfo(name="U1_out", component_ref="U1", position=(0, 0)),
                ],
            ),
        ]
        pin_to_net = {
            "V1_pin1": 1, "V1_pin2": 2,
            "R1_pin1": 1, "R1_pin2": 3,
            "U1_in+": 2, "U1_in-": 3, "U1_out": 4,
        }
        netlist = RecognitionPipeline._components_to_spice(components, pin_to_net)
        assert ".subckt OPAMP in+ in- out" in netlist
        assert ".ends OPAMP" in netlist
        assert "XU1" in netlist
        assert "OPAMP" in netlist

    def test_diode_op_amp_no_exception(self):
        """混合元件网表生成不抛异常"""
        components = [
            ComponentInfo(
                ref="V1", type="voltage_source", value="5",
                pins=[
                    PinInfo(name="V1_pin1", component_ref="V1", position=(0, 0)),
                    PinInfo(name="V1_pin2", component_ref="V1", position=(0, 0)),
                ],
            ),
            ComponentInfo(
                ref="D1", type="diode", value="D",
                pins=[
                    PinInfo(name="D1_anode", component_ref="D1", position=(0, 0)),
                    PinInfo(name="D1_cathode", component_ref="D1", position=(0, 0)),
                ],
            ),
            ComponentInfo(
                ref="R1", type="resistor", value="1k",
                pins=[
                    PinInfo(name="R1_pin1", component_ref="R1", position=(0, 0)),
                    PinInfo(name="R1_pin2", component_ref="R1", position=(0, 0)),
                ],
            ),
            ComponentInfo(
                ref="U1", type="op_amp", value="OP",
                pins=[
                    PinInfo(name="U1_in+", component_ref="U1", position=(0, 0)),
                    PinInfo(name="U1_in-", component_ref="U1", position=(0, 0)),
                    PinInfo(name="U1_out", component_ref="U1", position=(0, 0)),
                ],
            ),
        ]
        pin_to_net = {
            "V1_pin1": 1, "V1_pin2": 2,
            "D1_anode": 1, "D1_cathode": 3,
            "R1_pin1": 3, "R1_pin2": 2,
            "U1_in+": 2, "U1_in-": 3, "U1_out": 4,
        }
        netlist = RecognitionPipeline._components_to_spice(components, pin_to_net)
        # 应同时包含二极管模型和运放子电路
        assert ".model DMOD D" in netlist
        assert ".subckt OPAMP" in netlist
        assert ".ends OPAMP" in netlist

    def test_only_resistor_no_model_defs(self):
        """纯电阻网表不包含额外模型定义"""
        components = [
            ComponentInfo(
                ref="R1", type="resistor", value="1k",
                pins=[
                    PinInfo(name="R1_pin1", component_ref="R1", position=(0, 0)),
                    PinInfo(name="R1_pin2", component_ref="R1", position=(0, 0)),
                ],
            ),
        ]
        pin_to_net = {"R1_pin1": 1, "R1_pin2": 2}
        netlist = RecognitionPipeline._components_to_spice(components, pin_to_net)
        assert ".model" not in netlist
        assert ".subckt" not in netlist

    def test_led_netlist_includes_model(self):
        """LED 网表应包含 LEDMOD 模型"""
        components = [
            ComponentInfo(
                ref="V1", type="voltage_source", value="5",
                pins=[
                    PinInfo(name="V1_pin1", component_ref="V1", position=(0, 0)),
                    PinInfo(name="V1_pin2", component_ref="V1", position=(0, 0)),
                ],
            ),
            ComponentInfo(
                ref="D1", type="led", value="LED",
                pins=[
                    PinInfo(name="D1_anode", component_ref="D1", position=(0, 0)),
                    PinInfo(name="D1_cathode", component_ref="D1", position=(0, 0)),
                ],
            ),
        ]
        pin_to_net = {"V1_pin1": 1, "V1_pin2": 2, "D1_anode": 1, "D1_cathode": 3}
        netlist = RecognitionPipeline._components_to_spice(components, pin_to_net)
        assert "D1" in netlist
        assert "LEDMOD" in netlist
        assert ".model LEDMOD D" in netlist

    def test_npn_transistor_netlist(self):
        """NPN 三极管网表应包含 Q 前缀和 NPNMOD 模型"""
        components = [
            ComponentInfo(
                ref="Q1", type="npn_transistor", value="NPN",
                pins=[
                    PinInfo(name="Q1_c", component_ref="Q1", position=(0, 0)),
                    PinInfo(name="Q1_b", component_ref="Q1", position=(0, 0)),
                    PinInfo(name="Q1_e", component_ref="Q1", position=(0, 0)),
                ],
            ),
        ]
        pin_to_net = {"Q1_c": 1, "Q1_b": 2, "Q1_e": 3}
        netlist = RecognitionPipeline._components_to_spice(components, pin_to_net)
        assert "Q1" in netlist
        assert "NPNMOD" in netlist
        assert ".model NPNMOD NPN" in netlist

    def test_pnp_transistor_netlist(self):
        """PNP 三极管网表应包含 PNPMOD 模型"""
        components = [
            ComponentInfo(
                ref="Q2", type="pnp_transistor", value="PNP",
                pins=[
                    PinInfo(name="Q2_c", component_ref="Q2", position=(0, 0)),
                    PinInfo(name="Q2_b", component_ref="Q2", position=(0, 0)),
                    PinInfo(name="Q2_e", component_ref="Q2", position=(0, 0)),
                ],
            ),
        ]
        pin_to_net = {"Q2_c": 1, "Q2_b": 2, "Q2_e": 3}
        netlist = RecognitionPipeline._components_to_spice(components, pin_to_net)
        assert "Q2" in netlist
        assert "PNPMOD" in netlist
        assert ".model PNPMOD PNP" in netlist

    def test_nmos_netlist(self):
        """NMOS 网表应包含 M 前缀和 NMOSMOD 模型"""
        components = [
            ComponentInfo(
                ref="M1", type="nmos", value="NMOS",
                pins=[
                    PinInfo(name="M1_d", component_ref="M1", position=(0, 0)),
                    PinInfo(name="M1_g", component_ref="M1", position=(0, 0)),
                    PinInfo(name="M1_s", component_ref="M1", position=(0, 0)),
                ],
            ),
        ]
        pin_to_net = {"M1_d": 1, "M1_g": 2, "M1_s": 3}
        netlist = RecognitionPipeline._components_to_spice(components, pin_to_net)
        assert "M1" in netlist
        assert "NMOSMOD" in netlist
        assert ".model NMOSMOD NMOS" in netlist

    def test_pmos_netlist(self):
        """PMOS 网表应包含 PMOSMOD 模型"""
        components = [
            ComponentInfo(
                ref="M2", type="pmos", value="PMOS",
                pins=[
                    PinInfo(name="M2_d", component_ref="M2", position=(0, 0)),
                    PinInfo(name="M2_g", component_ref="M2", position=(0, 0)),
                    PinInfo(name="M2_s", component_ref="M2", position=(0, 0)),
                ],
            ),
        ]
        pin_to_net = {"M2_d": 1, "M2_g": 2, "M2_s": 3}
        netlist = RecognitionPipeline._components_to_spice(components, pin_to_net)
        assert "M2" in netlist
        assert "PMOSMOD" in netlist
        assert ".model PMOSMOD PMOS" in netlist

    def test_ground_no_spice_line(self):
        """接地符号不生成 SPICE 行"""
        components = [
            ComponentInfo(
                ref="G1", type="ground", value="GND",
                pins=[
                    PinInfo(name="G1_pin1", component_ref="G1", position=(0, 0)),
                    PinInfo(name="G1_pin2", component_ref="G1", position=(0, 0)),
                ],
            ),
        ]
        pin_to_net = {"G1_pin1": 1, "G1_pin2": 2}
        netlist = RecognitionPipeline._components_to_spice(components, pin_to_net)
        assert "G1" not in netlist

    def test_switch_netlist(self):
        """开关网表应包含 S 前缀和 SWMOD"""
        components = [
            ComponentInfo(
                ref="S1", type="switch", value="SW",
                pins=[
                    PinInfo(name="S1_pin1", component_ref="S1", position=(0, 0)),
                    PinInfo(name="S1_pin2", component_ref="S1", position=(0, 0)),
                ],
            ),
        ]
        pin_to_net = {"S1_pin1": 1, "S1_pin2": 2}
        netlist = RecognitionPipeline._components_to_spice(components, pin_to_net)
        assert "S1" in netlist
        assert "SWMOD" in netlist

    def test_new_types_no_exception(self):
        """所有新类型混合网表不抛异常"""
        components = [
            ComponentInfo(ref="V1", type="voltage_source", value="5",
                          pins=[PinInfo(name="V1_p1", component_ref="V1", position=(0, 0)),
                                PinInfo(name="V1_p2", component_ref="V1", position=(0, 0))]),
            ComponentInfo(ref="R1", type="resistor", value="1k",
                          pins=[PinInfo(name="R1_p1", component_ref="R1", position=(0, 0)),
                                PinInfo(name="R1_p2", component_ref="R1", position=(0, 0))]),
            ComponentInfo(ref="D1", type="led", value="LED",
                          pins=[PinInfo(name="D1_p1", component_ref="D1", position=(0, 0)),
                                PinInfo(name="D1_p2", component_ref="D1", position=(0, 0))]),
            ComponentInfo(ref="Q1", type="npn_transistor", value="NPN",
                          pins=[PinInfo(name="Q1_c", component_ref="Q1", position=(0, 0)),
                                PinInfo(name="Q1_b", component_ref="Q1", position=(0, 0)),
                                PinInfo(name="Q1_e", component_ref="Q1", position=(0, 0))]),
            ComponentInfo(ref="M1", type="nmos", value="NMOS",
                          pins=[PinInfo(name="M1_d", component_ref="M1", position=(0, 0)),
                                PinInfo(name="M1_g", component_ref="M1", position=(0, 0)),
                                PinInfo(name="M1_s", component_ref="M1", position=(0, 0))]),
            ComponentInfo(ref="S1", type="switch", value="SW",
                          pins=[PinInfo(name="S1_p1", component_ref="S1", position=(0, 0)),
                                PinInfo(name="S1_p2", component_ref="S1", position=(0, 0))]),
        ]
        pin_to_net = {
            "V1_p1": 1, "V1_p2": 2, "R1_p1": 1, "R1_p2": 3,
            "D1_p1": 1, "D1_p2": 4, "Q1_c": 3, "Q1_b": 2, "Q1_e": 5,
            "M1_d": 4, "M1_g": 2, "M1_s": 5, "S1_p1": 3, "S1_p2": 5,
        }
        netlist = RecognitionPipeline._components_to_spice(components, pin_to_net)
        assert "LEDMOD" in netlist
        assert "NPNMOD" in netlist
        assert "NMOSMOD" in netlist
        assert "SWMOD" in netlist
