"""图像预处理：灰度化、二值化、纸面扫描增强"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from voltsnap.models import PreprocessResult
from voltsnap.utils import imread_unicode

logger = logging.getLogger("voltsnap.vision.preprocessor")


class ImagePreprocessor:
    """将电路图转为灰度图和二值化图，支持纸面扫描增强"""

    def __init__(
        self,
        binary_threshold: int = 128,
        enable_paper_enhance: bool = True,
    ):
        self.binary_threshold = binary_threshold
        self.enable_paper_enhance = enable_paper_enhance

    def process(self, image_path: str) -> PreprocessResult:
        """从文件路径读取并处理"""
        img = imread_unicode(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        return self._process_array(img)

    def process_from_array(self, image: np.ndarray) -> PreprocessResult:
        """从 numpy 数组直接处理"""
        return self._process_array(image)

    def _process_array(self, img: np.ndarray) -> PreprocessResult:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if self.enable_paper_enhance:
            gray = self._correct_uneven_lighting(gray)
            gray = self._remove_paper_texture(gray)

        # 自适应二值化，比全局阈值更适合纸面扫描
        binary = self._adaptive_binarize(gray)

        logger.info(
            "Preprocessed: shape=%s, foreground_pixels=%d",
            img.shape[:2],
            int(np.sum(binary > 0)),
        )
        return PreprocessResult(gray=gray, binary=binary, original=img)

    def _correct_uneven_lighting(self, gray: np.ndarray) -> np.ndarray:
        """校正明暗不均：用大核高斯模糊估计背景亮度，然后减去"""
        h, w = gray.shape
        # 核大小约为图像尺寸的 1/4，必须为奇数
        ksize = max(31, (min(h, w) // 4) | 1)
        background = cv2.GaussianBlur(gray, (ksize, ksize), 0)
        # 减去背景并归一化
        corrected = cv2.subtract(background, gray)
        corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)
        return corrected

    def _remove_paper_texture(self, gray: np.ndarray) -> np.ndarray:
        """去除纸张纹理：中值滤波 + 轻度双边滤波保边去噪"""
        # 中值滤波去椒盐噪点/纸张颗粒
        denoised = cv2.medianBlur(gray, 3)
        # 双边滤波：保边去纹理，d=5 控制邻域直径
        denoised = cv2.bilateralFilter(denoised, d=5, sigmaColor=50, sigmaSpace=50)
        return denoised

    def _adaptive_binarize(self, gray: np.ndarray) -> np.ndarray:
        """自适应二值化：结合全局阈值和自适应阈值"""
        # 全局 Otsu 阈值
        otsu_val, otsu_binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        # 自适应阈值（处理局部明暗差异）
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
            blockSize=15, C=10,
        )
        # 取两者的交集（前景像素在两种方法中都被认为是前景）
        binary = cv2.bitwise_and(otsu_binary, adaptive)
        # 如果交集太少（阈值过严），回退到 Otsu
        if np.sum(binary > 0) < 100:
            binary = otsu_binary
        return binary


class MoireRemover:
    """摩尔纹/网纹去除器"""

    @staticmethod
    def remove_moire(gray: np.ndarray) -> np.ndarray:
        """
        使用频域滤波去除摩尔纹。

        摩尔纹表现为频域中的周期性尖峰，通过陷波滤波器抑制。
        """
        h, w = gray.shape
        # DFT
        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft, axes=[0, 1])

        # 构建陷波滤波器：抑制高频周期性分量
        mask = np.ones((h, w, 2), np.float32)
        cy, cx = h // 2, w // 2

        # 在几个关键频率位置创建陷波带阻
        for radius_frac in [0.15, 0.25, 0.35]:
            r = int(min(h, w) * radius_frac)
            for angle_offset in [0, 45, 90, 135]:
                for sign in [1, -1]:
                    angle_rad = np.radians(angle_offset * sign)
                    nx = int(cx + r * np.cos(angle_rad))
                    ny = int(cy + r * np.sin(angle_rad))
                    # 创建圆形陷波
                    Y, X = np.ogrid[:h, :w]
                    dist = np.sqrt((X - nx) ** 2 + (Y - ny) ** 2)
                    notch_radius = max(5, min(h, w) * 0.03)
                    mask[dist < notch_radius] = 0.0

        # 应用滤波器
        filtered = dft_shift * mask
        # IDFT
        f_ishift = np.fft.ifftshift(filtered, axes=[0, 1])
        result = cv2.idft(f_ishift, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result


class ShadowRemover:
    """阴影去除器"""

    @staticmethod
    def remove_shadow(gray: np.ndarray) -> np.ndarray:
        """
        去除纸面轻微阴影。

        使用形态学闭运算估计光照背景，然后归一化。
        """
        # 用大核闭运算估计背景光照
        kernel_size = max(31, min(gray.shape) // 8)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        # 归一化：background / gray，消除光照不均
        # 避免除零
        background = np.maximum(background, 1)
        normalized = (gray.astype(np.float32) / background.astype(np.float32)) * 255
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)
        return normalized
