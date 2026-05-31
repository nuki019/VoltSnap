"""EDA 截图 fallback 和 Unicode 路径读取测试"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from voltsnap.utils import imread_unicode, imwrite_unicode


# ── 辅助函数 ────────────────────────────────────────────────────────


def _create_synthetic_eda_image() -> np.ndarray:
    """
    生成一张模拟 EDA 截图，包含:
    - V1=1.5V (红色圆形, 左侧)
    - R1=1kΩ (红色横向矩形, 上方)
    - R2=2kΩ (红色横向矩形, 右上方)
    - R3=2kΩ (红色竖向矩形, 右侧)
    - GND (接地符号, 底部)
    - 蓝色文字标签和值
    - 绿色导线
    """
    # 创建白色背景
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255

    # 颜色定义 (BGR)
    red = (0, 0, 255)
    blue = (255, 100, 0)
    green = (0, 180, 0)
    dark_red = (0, 0, 200)

    # === 红色元件 ===

    # V1: 电压源 (圆形) - 左侧
    cv2.circle(img, (150, 300), 30, red, 2)
    cv2.putText(img, "+", (140, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, red, 2)
    cv2.putText(img, "-", (140, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.6, red, 2)

    # R1: 电阻 (横向矩形) - 上方
    cv2.rectangle(img, (280, 150), (420, 175), red, 2)

    # R2: 电阻 (横向矩形) - 右上方
    cv2.rectangle(img, (500, 150), (640, 175), red, 2)

    # R3: 电阻 (竖向矩形) - 右侧
    cv2.rectangle(img, (620, 250), (645, 400), red, 2)

    # GND: 接地符号 (三条横线) - 底部，画大一些确保面积 > 200
    cv2.line(img, (130, 440), (170, 440), red, 3)
    cv2.line(img, (120, 455), (180, 455), red, 3)
    cv2.line(img, (110, 470), (190, 470), red, 3)
    # 竖线连接
    cv2.line(img, (150, 410), (150, 440), red, 3)

    # === 蓝色文字 ===

    # V1 标签和值
    cv2.putText(img, "V1", (120, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.5, blue, 1)
    cv2.putText(img, "1.5V", (170, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.45, blue, 1)

    # R1 标签和值
    cv2.putText(img, "R1", (320, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, blue, 1)
    cv2.putText(img, "1k", (330, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.45, blue, 1)

    # R2 标签和值
    cv2.putText(img, "R2", (540, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, blue, 1)
    cv2.putText(img, "2k", (550, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.45, blue, 1)

    # R3 标签和值
    cv2.putText(img, "R3", (650, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.5, blue, 1)
    cv2.putText(img, "2k", (650, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.45, blue, 1)

    # GND 标签
    cv2.putText(img, "GND", (115, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.5, blue, 1)

    # === 绿色导线 ===

    # V1 上端 -> R1 左端
    cv2.line(img, (150, 270), (150, 160), green, 2)
    cv2.line(img, (150, 160), (280, 160), green, 2)

    # R1 右端 -> R2 左端
    cv2.line(img, (420, 160), (500, 160), green, 2)

    # R2 右端 -> R3 上端
    cv2.line(img, (640, 160), (632, 160), green, 2)
    cv2.line(img, (632, 160), (632, 250), green, 2)

    # V1 下端 -> GND
    cv2.line(img, (150, 330), (150, 410), green, 2)

    # R3 下端 -> GND (通过底部导线)
    cv2.line(img, (632, 400), (632, 510), green, 2)
    cv2.line(img, (632, 510), (150, 510), green, 2)
    cv2.line(img, (150, 510), (150, 470), green, 2)

    return img


# ── Unicode 路径读取测试 ────────────────────────────────────────────


class TestImreadUnicode:
    """Unicode 安全图片读取测试"""

    def test_read_normal_path(self, tmp_path):
        """普通 ASCII 路径读取"""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        path = tmp_path / "test.png"
        imwrite_unicode(path, img)

        result = imread_unicode(path)
        assert result is not None
        assert result.shape == (100, 100, 3)

    def test_read_chinese_path(self, tmp_path):
        """中文路径读取"""
        img = np.zeros((50, 80, 3), dtype=np.uint8)
        chinese_dir = tmp_path / "测试目录"
        chinese_dir.mkdir()
        path = chinese_dir / "截图文件.png"
        imwrite_unicode(path, img)

        result = imread_unicode(path)
        assert result is not None
        assert result.shape == (50, 80, 3)

    def test_read_chinese_filename(self, tmp_path):
        """中文文件名读取"""
        img = np.ones((60, 40, 3), dtype=np.uint8) * 128
        path = tmp_path / "屏幕截图 2026-05-29.png"
        imwrite_unicode(path, img)

        result = imread_unicode(path)
        assert result is not None
        assert result.shape == (60, 40, 3)

    def test_read_nonexistent(self, tmp_path):
        """不存在的文件返回 None"""
        path = tmp_path / "不存在.png"
        result = imread_unicode(path)
        assert result is None

    def test_read_with_imread_color_flag(self, tmp_path):
        """IMREAD_COLOR 标志"""
        img = np.zeros((30, 30, 3), dtype=np.uint8)
        img[:, :, 0] = 255  # 蓝色
        path = tmp_path / "flag_test.png"
        cv2.imwrite(str(path), img)

        result = imread_unicode(path, cv2.IMREAD_COLOR)
        assert result is not None
        assert result.shape == (30, 30, 3)

    def test_read_with_imread_grayscale_flag(self, tmp_path):
        """IMREAD_GRAYSCALE 标志"""
        img = np.zeros((30, 30, 3), dtype=np.uint8)
        path = tmp_path / "gray_test.png"
        cv2.imwrite(str(path), img)

        result = imread_unicode(path, cv2.IMREAD_GRAYSCALE)
        assert result is not None
        assert len(result.shape) == 2  # 灰度图是 2D


# ── EDA Fallback 检测器测试 ────────────────────────────────────────


class TestEDAFallback:
    """EDA 截图 fallback 检测测试"""

    def test_detect_synthetic_eda_components(self):
        """合成 EDA 图应检测出 V1/R1/R2/R3/GND"""
        from voltsnap.recognition.eda_fallback import detect_eda_components

        image = _create_synthetic_eda_image()
        components = detect_eda_components(image)

        # 应检测到至少 5 个元件
        assert len(components) >= 4, f"Expected >=4 components, got {len(components)}: {[(c.ref, c.type) for c in components]}"

        refs = {c.ref for c in components}
        types = {c.type for c in components}

        # 应包含电压源
        assert "voltage_source" in types, f"No voltage_source found, types: {types}"

        # 应包含电阻（至少一个）
        assert "resistor" in types, f"No resistor found, types: {types}"

        # 应包含接地
        assert "ground" in types, f"No ground found, types: {types}"

    def test_detect_voltage_source_value(self):
        """电压源值应被检测"""
        from voltsnap.recognition.eda_fallback import detect_eda_components

        image = _create_synthetic_eda_image()
        components = detect_eda_components(image)

        vs_comps = [c for c in components if c.type == "voltage_source"]
        assert len(vs_comps) >= 1, "No voltage source detected"
        # 值应该非空
        assert vs_comps[0].value, "Voltage source value is empty"

    def test_detect_resistor_values(self):
        """电阻值应被检测"""
        from voltsnap.recognition.eda_fallback import detect_eda_components

        image = _create_synthetic_eda_image()
        components = detect_eda_components(image)

        res_comps = [c for c in components if c.type == "resistor"]
        assert len(res_comps) >= 2, f"Expected >=2 resistors, got {len(res_comps)}"

    def test_components_have_valid_bbox(self):
        """所有元件应有有效 bbox"""
        from voltsnap.recognition.eda_fallback import detect_eda_components

        image = _create_synthetic_eda_image()
        components = detect_eda_components(image)

        for comp in components:
            x1, y1, x2, y2 = comp.bbox
            assert x2 > x1, f"Invalid bbox width for {comp.ref}: {comp.bbox}"
            assert y2 > y1, f"Invalid bbox height for {comp.ref}: {comp.bbox}"
            assert comp.center[0] > 0, f"Invalid center x for {comp.ref}"
            assert comp.center[1] > 0, f"Invalid center y for {comp.ref}"

    def test_empty_image_returns_empty(self):
        """空白图像应返回空列表"""
        from voltsnap.recognition.eda_fallback import detect_eda_components

        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        components = detect_eda_components(img)
        assert components == []

    def test_none_image_returns_empty(self):
        """None 图像应返回空列表"""
        from voltsnap.recognition.eda_fallback import detect_eda_components

        components = detect_eda_components(None)
        assert components == []


# ── 集成测试：RecognitionPipeline + EDA fallback ────────────────────


class TestPipelineFallbackIntegration:
    """RecognitionPipeline EDA fallback 集成测试"""

    def test_pipeline_with_synthetic_eda(self, tmp_path):
        """Pipeline 应通过 EDA fallback 识别合成图"""
        from voltsnap.recognition.pipeline import RecognitionPipeline

        image = _create_synthetic_eda_image()
        image_path = tmp_path / "synthetic_eda.png"
        cv2.imwrite(str(image_path), image)

        # 不使用 GPU，跳过 YOLO（因为没有模型文件）
        pipeline = RecognitionPipeline(use_gpu=False)
        result = pipeline.process(image_path)

        assert result.success, f"Pipeline failed: {result.error_message}"
        # 应该通过 fallback 检测到元件
        assert len(result.components) >= 1, \
            f"Expected components from fallback, got {len(result.components)}"
        # 网表不应为空
        assert result.netlist, "Netlist is empty"

    def test_pipeline_infers_wire_connections(self, tmp_path):
        from voltsnap.recognition.pipeline import RecognitionPipeline

        image = _create_synthetic_eda_image()
        image_path = tmp_path / "wire_topology.png"
        cv2.imwrite(str(image_path), image)

        pipeline = RecognitionPipeline(use_gpu=False)
        result = pipeline.process(image_path)

        assert result.success, f"Pipeline failed: {result.error_message}"
        assert result.connections, "Expected visible EDA wires to produce connections"
        assert result.pin_to_net["V1_pin1"] == result.pin_to_net["R1_pin1"]
        assert result.pin_to_net["R1_pin2"] == result.pin_to_net["R2_pin1"]
        assert result.pin_to_net["R2_pin2"] == result.pin_to_net["R3_pin1"]
        assert result.pin_to_net["V1_pin2"] == result.pin_to_net["GND_pin1"]
        assert result.pin_to_net["R3_pin2"] == result.pin_to_net["GND_pin1"]

        pairs = {(c["start_ref"], c["end_ref"]) for c in result.connections}
        assert {("V1", "R1"), ("R1", "R2"), ("R3", "R2")} <= pairs

    def test_pipeline_with_chinese_path(self, tmp_path):
        """Pipeline 应处理中文路径"""
        from voltsnap.recognition.pipeline import RecognitionPipeline

        image = _create_synthetic_eda_image()
        chinese_dir = tmp_path / "测试"
        chinese_dir.mkdir()
        image_path = chinese_dir / "电路图.png"
        cv2.imwrite(str(image_path), image)

        pipeline = RecognitionPipeline(use_gpu=False)
        result = pipeline.process(image_path)

        assert result.success, f"Pipeline failed on Chinese path: {result.error_message}"

    def test_pipeline_detections_for_ui(self, tmp_path):
        """Pipeline 输出的 detections 应可用于 UI 画框"""
        from voltsnap.recognition.pipeline import RecognitionPipeline

        image = _create_synthetic_eda_image()
        image_path = tmp_path / "ui_test.png"
        cv2.imwrite(str(image_path), image)

        pipeline = RecognitionPipeline(use_gpu=False)
        result = pipeline.process(image_path)

        assert result.success
        # detections 应有 bbox 和 class_name
        for det in result.detections:
            assert "bbox" in det
            assert "class_name" in det
            assert len(det["bbox"]) == 4

    def test_pipeline_bound_components_for_table(self, tmp_path):
        """Pipeline 输出的 bound_components 应可用于表格"""
        from voltsnap.recognition.pipeline import RecognitionPipeline

        image = _create_synthetic_eda_image()
        image_path = tmp_path / "table_test.png"
        cv2.imwrite(str(image_path), image)

        pipeline = RecognitionPipeline(use_gpu=False)
        result = pipeline.process(image_path)

        assert result.success
        for bc in result.bound_components:
            assert "ref" in bc
            assert "type" in bc
            assert "value" in bc
            assert "center" in bc
            assert "confidence" in bc
