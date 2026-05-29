"""阶段 2 识别模块测试"""
import json
from pathlib import Path

import numpy as np
import pytest

from voltsnap.recognition.annotation_converter import AnnotationConverter, COMPONENT_CLASSES
from voltsnap.recognition.ocr_parser import OCRParser, standardize_value
from voltsnap.recognition.text_binding import TextBinder


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
