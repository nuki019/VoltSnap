"""YOLO OBB 元件检测器训练脚本

用法:
    # 1. 先生成训练数据
    python scripts/train_detector.py --generate --samples 5000

    # 2. 转换标注格式
    python scripts/train_detector.py --convert

    # 3. 训练模型
    python scripts/train_detector.py --train --epochs 100 --batch 16

    # 4. 测试推理
    python scripts/train_detector.py --detect path/to/image.png
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("train_detector")

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "dataset"
YOLO_DIR = PROJECT_ROOT / "data" / "yolo_obb"


def generate_training_data(samples: int, seed: int = 42):
    """阶段 1：生成训练数据"""
    from voltsnap.datagen.batch_generator import BatchGenerator

    logger.info("Generating %d training samples...", samples)
    gen = BatchGenerator(
        output_dir=DATASET_DIR,
        degrade=True,
        seed=seed,
    )
    annotations = gen.generate_batch(samples)
    logger.info("Generated %d samples in %s", len(annotations), DATASET_DIR)
    return annotations


def convert_annotations():
    """阶段 2：转换标注为 YOLO OBB 格式"""
    from voltsnap.recognition.annotation_converter import AnnotationConverter

    converter = AnnotationConverter()

    # 转换所有样本
    stats = converter.convert_dataset(DATASET_DIR)
    logger.info("Conversion: %s", stats)

    # 生成 YOLO 目录结构
    prepare_yolo_directory()

    return stats


def prepare_yolo_directory():
    """准备 YOLO 训练目录结构"""
    manifest_path = DATASET_DIR / "manifest.json"
    if not manifest_path.exists():
        logger.error("manifest.json not found in %s", DATASET_DIR)
        return

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    labels_dir = DATASET_DIR / "labels"

    # 读取 split 索引
    train_ids, val_ids = _load_splits()

    # 创建 YOLO 目录
    for split in ["train", "val"]:
        (YOLO_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 复制/链接文件
    sample_map = {s["sample_id"]: s for s in manifest["samples"]}

    for split_name, ids in [("train", train_ids), ("val", val_ids)]:
        for sid in ids:
            sample = sample_map.get(sid)
            if not sample:
                continue

            # 复制图片（优先用退化图）
            src_img = Path(sample.get("degraded_image_path") or sample.get("image_path", ""))
            if src_img.exists():
                dst_img = YOLO_DIR / "images" / split_name / f"{sid}.png"
                if not dst_img.exists():
                    shutil.copy2(src_img, dst_img)

            # 复制标签
            src_lbl = labels_dir / f"{sid}.txt"
            if src_lbl.exists():
                dst_lbl = YOLO_DIR / "labels" / split_name / f"{sid}.txt"
                if not dst_lbl.exists():
                    shutil.copy2(src_lbl, dst_lbl)

    # 生成 data.yaml
    from voltsnap.recognition.detector import create_data_yaml
    create_data_yaml(YOLO_DIR, YOLO_DIR / "data.yaml")

    logger.info("YOLO directory prepared: %s", YOLO_DIR)


def _load_splits() -> tuple[list[str], list[str]]:
    """加载 train/val 划分"""
    split_dir = DATASET_DIR / "splits"
    if not (split_dir / "train.json").exists():
        # 没有预定义划分，按 80/20 随机划分
        with open(DATASET_DIR / "manifest.json", encoding="utf-8") as f:
            manifest = json.load(f)
        ids = [s["sample_id"] for s in manifest["samples"]]
        import random
        random.seed(42)
        random.shuffle(ids)
        n = int(len(ids) * 0.8)
        return ids[:n], ids[n:]

    with open(split_dir / "train.json", encoding="utf-8") as f:
        train_ids = json.load(f)
    with open(split_dir / "val.json", encoding="utf-8") as f:
        val_ids = json.load(f)
    return train_ids, val_ids


def train_model(epochs: int, batch_size: int, model_size: str, device: str):
    """阶段 3：训练 YOLO OBB 模型"""
    from voltsnap.recognition.detector import YOLOTrainer

    data_yaml = YOLO_DIR / "data.yaml"
    if not data_yaml.exists():
        logger.error("data.yaml not found. Run --convert first.")
        return

    trainer = YOLOTrainer(data_yaml)
    best_path = trainer.train(
        model_size=model_size,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        project=str(PROJECT_ROOT / "runs" / "obb"),
        name="voltsnap_detector",
    )
    logger.info("Training complete. Best model: %s", best_path)
    return best_path


def test_detection(image_path: str, model_path: str | None = None):
    """阶段 4：测试检测"""
    from voltsnap.recognition.detector import ComponentDetector
    import cv2

    if model_path is None:
        # 查找最新训练的模型
        runs_dir = PROJECT_ROOT / "runs" / "obb" / "voltsnap_detector"
        best = runs_dir / "weights" / "best.pt"
        if best.exists():
            model_path = str(best)
        else:
            logger.warning("No trained model found, using pretrained yolov8n-obb")

    detector = ComponentDetector(model_path=model_path)
    detections = detector.detect_from_path(image_path)

    logger.info("Detected %d components in %s:", len(detections), image_path)
    for d in detections:
        logger.info("  %s (%.2f): center=%s, angle=%.1f°",
                     d.class_name, d.confidence, d.center, d.angle)

    return detections


def main():
    parser = argparse.ArgumentParser(description="VoltSnap 元件检测器训练工具")
    parser.add_argument("--generate", action="store_true", help="生成训练数据")
    parser.add_argument("--samples", type=int, default=1000, help="生成样本数")
    parser.add_argument("--convert", action="store_true", help="转换标注格式")
    parser.add_argument("--train", action="store_true", help="训练模型")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch", type=int, default=16, help="批大小")
    parser.add_argument("--model-size", default="n", choices=["n", "s", "m", "l", "x"], help="模型大小")
    parser.add_argument("--device", default="0", help="设备 (0=GPU, cpu=CPU)")
    parser.add_argument("--detect", type=str, help="检测图片路径")
    parser.add_argument("--model", type=str, help="模型路径")

    args = parser.parse_args()

    if args.generate:
        generate_training_data(args.samples)

    if args.convert:
        convert_annotations()

    if args.train:
        train_model(args.epochs, args.batch, args.model_size, args.device)

    if args.detect:
        test_detection(args.detect, args.model)

    if not any([args.generate, args.convert, args.train, args.detect]):
        parser.print_help()


if __name__ == "__main__":
    main()
