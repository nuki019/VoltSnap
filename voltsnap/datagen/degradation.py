"""图像退化增强管线 — 模拟拍照场景的图像质量下降"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class DegradationConfig:
    """退化参数配置"""
    # 高斯模糊：核大小必须为奇数
    blur_kernel_range: tuple[int, int] = (3, 9)
    blur_sigma_range: tuple[float, float] = (0.5, 3.0)

    # 高斯噪声
    noise_sigma_range: tuple[float, float] = (5.0, 30.0)

    # 对比度 + 亮度抖动
    contrast_range: tuple[float, float] = (0.6, 1.4)   # alpha
    brightness_range: tuple[int, int] = (-30, 30)       # beta

    # 分辨率下采样然后恢复
    downscale_range: tuple[float, float] = (0.3, 0.8)

    # JPEG 压缩伪影
    jpeg_quality_range: tuple[int, int] = (30, 80)

    # 仿射变换（轻微旋转 + 缩放）
    rotation_range: tuple[float, float] = (-5.0, 5.0)   # 度
    scale_range: tuple[float, float] = (0.9, 1.1)

    # 各增强的启用概率
    p_blur: float = 0.5
    p_noise: float = 0.5
    p_contrast: float = 0.5
    p_downscale: float = 0.3
    p_jpeg: float = 0.3
    p_affine: float = 0.3


class DegradationPipeline:
    """
    图像退化增强管线。

    按随机顺序组合多种退化效果，模拟手机拍照场景：
    - 高斯模糊（手抖 / 对焦不准）
    - 高斯噪声（低光照）
    - 对比度 / 亮度抖动（曝光不均）
    - 分辨率下采样（远距离拍摄）
    - JPEG 压缩伪影
    - 仿射变换（轻微视角变化）
    """

    def __init__(self, config: DegradationConfig | None = None, seed: int | None = None):
        self.config = config or DegradationConfig()
        self._rng = random.Random(seed)
        self._np_rng = np.random.RandomState(seed)

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        对输入图像应用随机退化组合。

        Parameters
        ----------
        image : np.ndarray
            输入图像 (BGR, uint8)。

        Returns
        -------
        np.ndarray
            退化后的图像 (BGR, uint8)，尺寸与输入相同。
        """
        result = image.copy()
        h, w = result.shape[:2]

        # 仿射变换（最先应用，改变几何形状）
        if self._rng.random() < self.config.p_affine:
            result = self._apply_affine(result)

        # 高斯模糊
        if self._rng.random() < self.config.p_blur:
            result = self._apply_blur(result)

        # 高斯噪声
        if self._rng.random() < self.config.p_noise:
            result = self._apply_noise(result)

        # 对比度 + 亮度
        if self._rng.random() < self.config.p_contrast:
            result = self._apply_contrast(result)

        # 分辨率下采样
        if self._rng.random() < self.config.p_downscale:
            result = self._apply_downscale(result, h, w)

        # JPEG 压缩（最后应用，模拟最终保存）
        if self._rng.random() < self.config.p_jpeg:
            result = self._apply_jpeg(result)

        return result

    # ── 单一退化效果 ──────────────────────────────────────────────────

    def _apply_blur(self, img: np.ndarray) -> np.ndarray:
        k = self._rng.randrange(
            self.config.blur_kernel_range[0],
            self.config.blur_kernel_range[1] + 1,
            2,  # 保证奇数
        )
        sigma = self._rng.uniform(*self.config.blur_sigma_range)
        return cv2.GaussianBlur(img, (k, k), sigma)

    def _apply_noise(self, img: np.ndarray) -> np.ndarray:
        sigma = self._rng.uniform(*self.config.noise_sigma_range)
        noise = self._np_rng.normal(0, sigma, img.shape).astype(np.float32)
        noisy = np.clip(img.astype(np.float32) + noise, 0, 255)
        return noisy.astype(np.uint8)

    def _apply_contrast(self, img: np.ndarray) -> np.ndarray:
        alpha = self._rng.uniform(*self.config.contrast_range)
        beta = self._rng.randint(*self.config.brightness_range)
        return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    def _apply_downscale(self, img: np.ndarray, orig_h: int, orig_w: int) -> np.ndarray:
        scale = self._rng.uniform(*self.config.downscale_range)
        small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    def _apply_jpeg(self, img: np.ndarray) -> np.ndarray:
        quality = self._rng.randint(*self.config.jpeg_quality_range)
        encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
        _, buf = cv2.imencode(".jpg", img, encode_param)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

    def _apply_affine(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        cx, cy = w / 2, h / 2

        angle = self._rng.uniform(*self.config.rotation_range)
        scale = self._rng.uniform(*self.config.scale_range)

        M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
        return cv2.warpAffine(
            img, M, (w, h),
            borderMode=cv2.BORDER_REPLICATE,
        )


def apply_single_degradation(image: np.ndarray, kind: str, seed: int | None = None) -> np.ndarray:
    """
    便捷函数：只应用一种退化效果。

    Parameters
    ----------
    image : np.ndarray
        输入图像 (BGR, uint8)。
    kind : str
        退化类型: blur, noise, contrast, downscale, jpeg, affine。
    seed : int | None
        随机种子。

    Returns
    -------
    np.ndarray
        退化后的图像。
    """
    pipeline = DegradationPipeline(
        config=DegradationConfig(
            p_blur=1.0 if kind == "blur" else 0.0,
            p_noise=1.0 if kind == "noise" else 0.0,
            p_contrast=1.0 if kind == "contrast" else 0.0,
            p_downscale=1.0 if kind == "downscale" else 0.0,
            p_jpeg=1.0 if kind == "jpeg" else 0.0,
            p_affine=1.0 if kind == "affine" else 0.0,
        ),
        seed=seed,
    )
    return pipeline.apply(image)
