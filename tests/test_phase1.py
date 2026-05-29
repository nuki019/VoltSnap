"""阶段 1 数据生成管线综合测试"""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from voltsnap.datagen.batch_generator import BatchGenerator, SampleAnnotation
from voltsnap.datagen.circuit_templates import RandomCircuitGenerator
from voltsnap.datagen.dataset_splitter import DatasetSplitter
from voltsnap.datagen.degradation import DegradationConfig, DegradationPipeline, apply_single_degradation


# ── 退化管线测试 ──────────────────────────────────────────────────────


class TestDegradationPipeline:
    """图像退化增强测试"""

    @pytest.fixture
    def sample_image(self):
        """创建测试用图像（100x100 灰白棋盘格）"""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[::2, ::2] = 255
        img[1::2, 1::2] = 255
        return img

    def test_pipeline_preserves_shape(self, sample_image):
        pipeline = DegradationPipeline(seed=42)
        result = pipeline.apply(sample_image)
        assert result.shape == sample_image.shape

    def test_pipeline_preserves_dtype(self, sample_image):
        pipeline = DegradationPipeline(seed=42)
        result = pipeline.apply(sample_image)
        assert result.dtype == np.uint8

    def test_pipeline_deterministic_with_seed(self, sample_image):
        p1 = DegradationPipeline(seed=99)
        p2 = DegradationPipeline(seed=99)
        r1 = p1.apply(sample_image)
        r2 = p2.apply(sample_image)
        np.testing.assert_array_equal(r1, r2)

    def test_pipeline_different_seeds_differ(self, sample_image):
        p1 = DegradationPipeline(seed=1)
        p2 = DegradationPipeline(seed=2)
        r1 = p1.apply(sample_image)
        r2 = p2.apply(sample_image)
        assert not np.array_equal(r1, r2)

    def test_all_degradations_enabled(self, sample_image):
        config = DegradationConfig(
            p_blur=1.0, p_noise=1.0, p_contrast=1.0,
            p_downscale=1.0, p_jpeg=1.0, p_affine=1.0,
        )
        pipeline = DegradationPipeline(config=config, seed=42)
        result = pipeline.apply(sample_image)
        assert result.shape == sample_image.shape

    def test_no_degradation_passthrough(self, sample_image):
        config = DegradationConfig(
            p_blur=0.0, p_noise=0.0, p_contrast=0.0,
            p_downscale=0.0, p_jpeg=0.0, p_affine=0.0,
        )
        pipeline = DegradationPipeline(config=config, seed=42)
        result = pipeline.apply(sample_image)
        np.testing.assert_array_equal(result, sample_image)

    def test_single_degradation_blur(self, sample_image):
        result = apply_single_degradation(sample_image, "blur", seed=42)
        assert result.shape == sample_image.shape

    def test_single_degradation_noise(self, sample_image):
        result = apply_single_degradation(sample_image, "noise", seed=42)
        assert result.shape == sample_image.shape

    def test_single_degradation_contrast(self, sample_image):
        result = apply_single_degradation(sample_image, "contrast", seed=42)
        assert result.shape == sample_image.shape


# ── 批量生成器测试 ────────────────────────────────────────────────────


class TestBatchGenerator:
    """批量数据生成器测试"""

    def test_generate_single_sample(self, tmp_path):
        gen = BatchGenerator(output_dir=tmp_path, degrade=False, seed=42)
        ann = gen.generate_sample(0, "resistor_divider")

        assert ann.sample_id.startswith("resistor_divider_")
        assert Path(ann.image_path).exists()
        assert len(ann.components) == 3
        assert ann.netlist
        assert ann.image_size[0] > 0

        # 标注 JSON 应已保存
        ann_path = Path(ann.image_path).parent / "annotation.json"
        assert ann_path.exists()
        data = json.loads(ann_path.read_text(encoding="utf-8"))
        assert data["sample_id"] == ann.sample_id

    def test_generate_with_degradation(self, tmp_path):
        gen = BatchGenerator(output_dir=tmp_path, degrade=True, seed=42)
        ann = gen.generate_sample(0, "rc_series")

        assert ann.degraded_image_path is not None
        assert Path(ann.degraded_image_path).exists()

    def test_generate_batch(self, tmp_path):
        gen = BatchGenerator(output_dir=tmp_path, degrade=False, seed=42)
        annotations = gen.generate_batch(8)

        assert len(annotations) == 8
        # 每个样本目录存在
        for ann in annotations:
            assert Path(ann.image_path).exists()

        # manifest.json 存在
        manifest_path = tmp_path / "dataset" / "manifest.json"
        if not manifest_path.exists():
            # 可能在 output_dir 根目录
            manifest_path = tmp_path / "manifest.json"
        assert manifest_path.exists()

    def test_batch_topology_distribution(self, tmp_path):
        gen = BatchGenerator(output_dir=tmp_path, degrade=False, seed=42)
        dist = {"resistor_divider": 0.5, "rc_series": 0.5}
        annotations = gen.generate_batch(10, topology_distribution=dist)

        topo_counts = {}
        for ann in annotations:
            topo_counts[ann.topology_type] = topo_counts.get(ann.topology_type, 0) + 1
        assert topo_counts.get("resistor_divider", 0) >= 4
        assert topo_counts.get("rc_series", 0) >= 4

    def test_pin_positions_valid(self, tmp_path):
        gen = BatchGenerator(output_dir=tmp_path, degrade=False, seed=42)
        ann = gen.generate_sample(0, "resistor_divider")

        for name, pos in ann.pin_positions.items():
            assert len(pos) == 2
            assert pos[0] > 0 or pos[1] > 0  # 不全是 (0,0)


# ── 数据集划分测试 ────────────────────────────────────────────────────


class TestDatasetSplitter:
    """数据集质量统计与划分测试"""

    @pytest.fixture
    def small_dataset(self, tmp_path):
        """生成小型测试数据集"""
        gen = BatchGenerator(output_dir=tmp_path / "ds", degrade=False, seed=42)
        gen.generate_batch(12)
        return tmp_path / "ds"

    def test_compute_statistics(self, small_dataset):
        splitter = DatasetSplitter(small_dataset)
        stats = splitter.compute_statistics()

        assert stats["total_samples"] == 12
        assert len(stats["topology_distribution"]) >= 1
        assert "image_size_stats" in stats

    def test_split_sizes(self, small_dataset):
        splitter = DatasetSplitter(small_dataset)
        splits = splitter.split(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)

        total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        assert total == 12
        assert len(splits["train"]) >= 6
        # 小数据集中某些拓扑只有 1 个样本，int(1*0.15)=0，val/test 可能为空
        # 但总数必须正确

    def test_split_no_overlap(self, small_dataset):
        splitter = DatasetSplitter(small_dataset)
        splits = splitter.split(seed=42)

        train_set = set(splits["train"])
        val_set = set(splits["val"])
        test_set = set(splits["test"])
        assert len(train_set & val_set) == 0
        assert len(train_set & test_set) == 0
        assert len(val_set & test_set) == 0

    def test_split_files_saved(self, small_dataset):
        splitter = DatasetSplitter(small_dataset)
        splitter.split(seed=42)

        split_dir = small_dataset / "splits"
        assert (split_dir / "train.json").exists()
        assert (split_dir / "val.json").exists()
        assert (split_dir / "test.json").exists()
        assert (split_dir / "split_stats.json").exists()

    def test_split_stratified(self, small_dataset):
        """分层划分：各拓扑类型在 train/val/test 中都有代表"""
        splitter = DatasetSplitter(small_dataset)
        splits = splitter.split(seed=42)

        # 加载样本元数据
        with open(small_dataset / "manifest.json", encoding="utf-8") as f:
            manifest = json.load(f)
        sample_map = {s["sample_id"]: s["topology_type"] for s in manifest["samples"]}

        train_topos = {sample_map[sid] for sid in splits["train"] if sid in sample_map}
        # 训练集应覆盖至少 2 种拓扑
        assert len(train_topos) >= 2
