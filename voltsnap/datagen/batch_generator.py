"""批量数据生成器 — 渲染随机电路 + 退化增强 + 标注导出"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from voltsnap.config import Config
from voltsnap.datagen.circuit_templates import RandomCircuitConfig, RandomCircuitGenerator
from voltsnap.datagen.degradation import DegradationConfig, DegradationPipeline
from voltsnap.datagen.schematic_renderer import SchematicRenderer
from voltsnap.datagen.netlist_generator import NetlistGenerator
from voltsnap.models import CircuitSpec

logger = logging.getLogger("voltsnap.datagen.batch")


@dataclass
class SampleAnnotation:
    """单个样本的标注信息"""
    sample_id: str
    topology_type: str
    components: list[dict]
    netlist: str
    image_path: str
    degraded_image_path: str | None
    binary_path: str | None
    skeleton_path: str | None
    pin_positions: dict[str, list[float]]
    image_size: list[int]


class BatchGenerator:
    """
    批量数据生成器。

    流程：随机电路 → schemdraw 渲染 → 图像退化 → 标注导出
    """

    def __init__(
        self,
        output_dir: str | Path | None = None,
        render_dpi: int = 150,
        inches_per_unit: float = 3.0,
        circuit_config: RandomCircuitConfig | None = None,
        degradation_config: DegradationConfig | None = None,
        degrade: bool = True,
        seed: int | None = None,
    ):
        self.output_dir = Path(output_dir) if output_dir else Config.GENERATED_DIR / "dataset"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.renderer = SchematicRenderer(dpi=render_dpi, inches_per_unit=inches_per_unit)
        self.circuit_gen = RandomCircuitGenerator(config=circuit_config, seed=seed)
        self.degradation = DegradationPipeline(config=degradation_config, seed=seed)
        self.netlist_gen = NetlistGenerator()
        self.degrade = degrade
        self._seed = seed

    def generate_sample(self, index: int, topology_type: str | None = None) -> SampleAnnotation:
        """
        生成单个训练样本。

        Parameters
        ----------
        index : int
            样本编号（用于命名）。
        topology_type : str | None
            指定拓扑类型，None 则随机选择。

        Returns
        -------
        SampleAnnotation
            包含图片路径、标注信息的标注对象。
        """
        # 生成随机电路
        circuit = self.circuit_gen.generate(topology_type)
        sample_id = f"{circuit.name}_{index:06d}"
        sample_dir = self.output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        # 渲染电路图
        image_path = str(sample_dir / "circuit.png")
        render_result = self.renderer.render(circuit, image_path)

        # 图像退化
        degraded_path: str | None = None
        if self.degrade:
            image = cv2.imread(image_path)
            if image is not None:
                degraded = self.degradation.apply(image)
                degraded_path = str(sample_dir / "degraded.png")
                cv2.imwrite(degraded_path, degraded)

        # 提取引脚坐标
        pin_positions: dict[str, list[float]] = {}
        for comp in render_result.components:
            for pin in comp.pins:
                pin_positions[pin.name] = list(pin.pixel_position)

        # 组装标注
        annotation = SampleAnnotation(
            sample_id=sample_id,
            topology_type=circuit.name,
            components=[
                {"ref": c.ref, "type": c.type, "value": c.value}
                for c in circuit.components
            ],
            netlist=circuit.expected_netlist or "",
            image_path=image_path,
            degraded_image_path=degraded_path,
            binary_path=None,
            skeleton_path=None,
            pin_positions=pin_positions,
            image_size=list(render_result.image_size),
        )

        # 保存标注 JSON
        ann_path = sample_dir / "annotation.json"
        ann_path.write_text(json.dumps(asdict(annotation), indent=2, ensure_ascii=False), encoding="utf-8")

        return annotation

    def generate_batch(
        self,
        count: int,
        topology_distribution: dict[str, float] | None = None,
    ) -> list[SampleAnnotation]:
        """
        批量生成训练样本。

        Parameters
        ----------
        count : int
            总样本数。
        topology_distribution : dict[str, float] | None
            各拓扑类型的比例，None 则均匀分布。
            例: {"resistor_divider": 0.3, "two_mesh": 0.7}

        Returns
        -------
        list[SampleAnnotation]
            所有样本的标注列表。
        """
        annotations: list[SampleAnnotation] = []
        types = self.circuit_gen.TOPOLOGY_TYPES

        if topology_distribution:
            # 按比例分配
            type_list: list[str | None] = []
            for topo, ratio in topology_distribution.items():
                type_list.extend([topo] * max(1, int(ratio * count)))
            while len(type_list) < count:
                type_list.append(None)
            type_list = type_list[:count]
        else:
            type_list = [None] * count

        start_time = time.time()
        failed = 0

        for i, topo in enumerate(type_list):
            try:
                ann = self.generate_sample(i, topo)
                annotations.append(ann)
            except Exception as e:
                logger.warning("Sample %d failed: %s", i, e)
                failed += 1

            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (count - i - 1) / rate
                logger.info(
                    "Progress: %d/%d (%.1f/s, ETA %.0fs, failed %d)",
                    i + 1, count, rate, eta, failed,
                )

        elapsed = time.time() - start_time
        logger.info(
            "Batch complete: %d samples in %.1fs (%.1f/s, %d failed)",
            len(annotations), elapsed, len(annotations) / max(elapsed, 0.001), failed,
        )

        # 保存汇总标注
        manifest = {
            "total": len(annotations),
            "failed": failed,
            "elapsed_seconds": elapsed,
            "seed": self._seed,
            "degrade": self.degrade,
            "samples": [asdict(a) for a in annotations],
        }
        manifest_path = self.output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Manifest saved: %s", manifest_path)

        return annotations
