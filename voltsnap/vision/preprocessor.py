"""图像预处理：灰度化、二值化"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from voltsnap.models import PreprocessResult

logger = logging.getLogger("voltsnap.vision.preprocessor")


class ImagePreprocessor:
    """将电路图转为灰度图和二值化图"""

    def __init__(self, binary_threshold: int = 128):
        self.binary_threshold = binary_threshold

    def process(self, image_path: str) -> PreprocessResult:
        """从文件路径读取并处理"""
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        return self._process_array(img)

    def process_from_array(self, image: np.ndarray) -> PreprocessResult:
        """从 numpy 数组直接处理"""
        return self._process_array(image)

    def _process_array(self, img: np.ndarray) -> PreprocessResult:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # schemdraw 输出白底黑线，THRESH_BINARY_INV 反转为前景=255
        _, binary = cv2.threshold(
            gray, self.binary_threshold, 255, cv2.THRESH_BINARY_INV
        )
        logger.info(
            "Preprocessed: shape=%s, foreground_pixels=%d",
            img.shape[:2],
            int(np.sum(binary > 0)),
        )
        return PreprocessResult(gray=gray, binary=binary, original=img)
