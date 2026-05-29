"""数据集质量统计与 train/val/test 划分"""
from __future__ import annotations

import json
import logging
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("voltsnap.datagen.splitter")


class DatasetSplitter:
    """
    数据集质量统计与划分工具。

    功能：
    1. 扫描 manifest.json，统计样本分布
    2. 检测低质量样本（损坏图片、空白图、尺寸异常）
    3. 按拓扑类型分层划分 train / val / test
    """

    def __init__(self, dataset_dir: str | Path):
        self.dataset_dir = Path(dataset_dir)
        self.manifest_path = self.dataset_dir / "manifest.json"
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"manifest.json not found: {self.manifest_path}")
        with open(self.manifest_path, encoding="utf-8") as f:
            self.manifest = json.load(f)
        self.samples: list[dict] = self.manifest.get("samples", [])

    # ── 质量统计 ──────────────────────────────────────────────────────

    def compute_statistics(self) -> dict:
        """
        计算数据集质量统计。

        Returns
        -------
        dict
            包含拓扑分布、图像质量指标、异常样本列表。
        """
        topo_dist = Counter()
        image_sizes: list[tuple[int, int]] = []
        corrupt_samples: list[str] = []
        blank_samples: list[str] = []
        tiny_samples: list[str] = []

        for sample in self.samples:
            sid = sample["sample_id"]
            topo_dist[sample["topology_type"]] += 1

            # 检查图片是否存在且可读
            img_path = sample.get("image_path", "")
            if not img_path or not Path(img_path).exists():
                corrupt_samples.append(sid)
                continue

            img = cv2.imread(img_path)
            if img is None:
                corrupt_samples.append(sid)
                continue

            h, w = img.shape[:2]
            image_sizes.append((w, h))

            # 空白检测：标准差 < 5 认为是纯色图
            if np.std(img) < 5:
                blank_samples.append(sid)
                continue

            # 尺寸异常：< 100px
            if w < 100 or h < 100:
                tiny_samples.append(sid)

        # 尺寸统计
        if image_sizes:
            widths = [s[0] for s in image_sizes]
            heights = [s[1] for s in image_sizes]
            size_stats = {
                "min_width": min(widths),
                "max_width": max(widths),
                "avg_width": sum(widths) / len(widths),
                "min_height": min(heights),
                "max_height": max(heights),
                "avg_height": sum(heights) / len(heights),
            }
        else:
            size_stats = {}

        stats = {
            "total_samples": len(self.samples),
            "topology_distribution": dict(topo_dist),
            "image_size_stats": size_stats,
            "corrupt_samples": corrupt_samples,
            "blank_samples": blank_samples,
            "tiny_samples": tiny_samples,
            "quality_issues_count": len(corrupt_samples) + len(blank_samples) + len(tiny_samples),
        }

        logger.info("Statistics: %d samples, %d quality issues", stats["total_samples"], stats["quality_issues_count"])
        return stats

    # ── 分层划分 ──────────────────────────────────────────────────────

    def split(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        copy_files: bool = False,
    ) -> dict[str, list[str]]:
        """
        按拓扑类型分层划分 train / val / test。

        Parameters
        ----------
        train_ratio, val_ratio, test_ratio : float
            划分比例，总和应为 1.0。
        seed : int
            随机种子，保证可复现。
        copy_files : bool
            是否复制图片文件到 split 目录。False 则只生成索引 JSON。

        Returns
        -------
        dict[str, list[str]]
            {"train": [...], "val": [...], "test": [...]}
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为 1.0"

        # 按拓扑类型分组
        groups: dict[str, list[dict]] = {}
        for sample in self.samples:
            topo = sample["topology_type"]
            groups.setdefault(topo, []).append(sample)

        rng = np.random.RandomState(seed)
        splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}

        for topo, samples in groups.items():
            rng.shuffle(samples)
            n = len(samples)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)

            splits["train"].extend(s["sample_id"] for s in samples[:n_train])
            splits["val"].extend(s["sample_id"] for s in samples[n_train:n_train + n_val])
            splits["test"].extend(s["sample_id"] for s in samples[n_train + n_val:])

        # 打乱各 split 内部顺序
        for key in splits:
            rng.shuffle(splits[key])

        # 保存索引 JSON
        split_dir = self.dataset_dir / "splits"
        split_dir.mkdir(parents=True, exist_ok=True)
        for split_name, ids in splits.items():
            index_path = split_dir / f"{split_name}.json"
            index_path.write_text(json.dumps(ids, indent=2), encoding="utf-8")
            logger.info("Split '%s': %d samples → %s", split_name, len(ids), index_path)

        # 可选：复制文件到 split 目录
        if copy_files:
            self._copy_split_files(splits, split_dir)

        # 保存划分统计
        split_stats = {
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "seed": seed,
            "train_count": len(splits["train"]),
            "val_count": len(splits["val"]),
            "test_count": len(splits["test"]),
            "topology_distribution": {
                topo: len(samples) for topo, samples in groups.items()
            },
        }
        stats_path = split_dir / "split_stats.json"
        stats_path.write_text(json.dumps(split_stats, indent=2, ensure_ascii=False), encoding="utf-8")

        return splits

    def _copy_split_files(self, splits: dict[str, list[str]], split_dir: Path) -> None:
        """复制样本文件到各 split 目录"""
        sample_map = {s["sample_id"]: s for s in self.samples}
        for split_name, ids in splits.items():
            dest_dir = split_dir / split_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for sid in ids:
                sample = sample_map.get(sid)
                if not sample:
                    continue
                src_dir = self.dataset_dir / sid
                if src_dir.exists():
                    shutil.copytree(src_dir, dest_dir / sid, dirs_exist_ok=True)
            logger.info("Copied %d samples to %s", len(ids), dest_dir)
