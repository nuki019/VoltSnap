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

    # 纸张纹理噪声
    paper_texture_sigma_range: tuple[float, float] = (3.0, 15.0)

    # 不均匀光照
    uneven_light_strength_range: tuple[float, float] = (0.3, 0.7)

    # 摩尔纹（周期性波纹）
    moire_frequency_range: tuple[float, float] = (0.02, 0.08)
    moire_amplitude_range: tuple[float, float] = (10.0, 40.0)

    # 阴影
    shadow_strength_range: tuple[float, float] = (0.5, 0.85)
    shadow_radius_range: tuple[float, float] = (0.3, 0.7)  # 占图像比例

    # 各增强的启用概率
    p_blur: float = 0.5
    p_noise: float = 0.5
    p_contrast: float = 0.5
    p_downscale: float = 0.3
    p_jpeg: float = 0.3
    p_affine: float = 0.3
    p_paper_texture: float = 0.4
    p_uneven_light: float = 0.3
    p_moire: float = 0.2
    p_shadow: float = 0.3


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

        # 不均匀光照（模拟纸面光照不均）
        if self._rng.random() < self.config.p_uneven_light:
            result = self._apply_uneven_lighting(result)

        # 阴影（模拟手持拍摄遮挡）
        if self._rng.random() < self.config.p_shadow:
            result = self._apply_shadow(result)

        # 纸张纹理
        if self._rng.random() < self.config.p_paper_texture:
            result = self._apply_paper_texture(result)

        # 摩尔纹
        if self._rng.random() < self.config.p_moire:
            result = self._apply_moire(result)

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

    def _apply_paper_texture(self, img: np.ndarray) -> np.ndarray:
        """模拟纸张纹理噪声（细微颗粒感）"""
        sigma = self._rng.uniform(*self.config.paper_texture_sigma_range)
        # 纹理噪声：低幅度高频噪声叠加
        noise = self._np_rng.normal(0, sigma, img.shape[:2]).astype(np.float32)
        # 轻微模糊使纹理更自然
        k = self._rng.choice([3, 5])
        noise = cv2.GaussianBlur(noise, (k, k), 0)
        # 叠加到各通道
        if img.ndim == 3:
            noise = noise[:, :, np.newaxis]
        textured = np.clip(img.astype(np.float32) + noise, 0, 255)
        return textured.astype(np.uint8)

    def _apply_uneven_lighting(self, img: np.ndarray) -> np.ndarray:
        """模拟不均匀光照（纸面明暗渐变）"""
        h, w = img.shape[:2]
        strength = self._rng.uniform(*self.config.uneven_light_strength_range)

        # 创建渐变掩码：从随机方向的亮到暗
        mask = np.zeros((h, w), dtype=np.float32)
        # 随机选择渐变方向
        angle = self._rng.uniform(0, 2 * np.pi)
        Y, X = np.mgrid[:h, :w]
        # 归一化坐标
        Xn = (X / w - 0.5) * 2
        Yn = (Y / h - 0.5) * 2
        gradient = Xn * np.cos(angle) + Yn * np.sin(angle)
        # 归一化到 [0, 1]
        gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min() + 1e-6)
        # 应用强度
        mask = 1.0 - strength * gradient

        if img.ndim == 3:
            mask = mask[:, :, np.newaxis]
        result = np.clip(img.astype(np.float32) * mask, 0, 255)
        return result.astype(np.uint8)

    def _apply_moire(self, img: np.ndarray) -> np.ndarray:
        """模拟摩尔纹/网纹（周期性波纹干扰）"""
        h, w = img.shape[:2]
        freq = self._rng.uniform(*self.config.moire_frequency_range)
        amplitude = self._rng.uniform(*self.config.moire_amplitude_range)

        # 创建两个不同方向的正弦波叠加
        angle1 = self._rng.uniform(0, np.pi)
        angle2 = angle1 + self._rng.uniform(np.pi / 6, np.pi / 3)

        Y, X = np.mgrid[:h, :w]
        wave1 = np.sin(2 * np.pi * freq * (X * np.cos(angle1) + Y * np.sin(angle1)))
        wave2 = np.sin(2 * np.pi * freq * 0.7 * (X * np.cos(angle2) + Y * np.sin(angle2)))

        moire_pattern = amplitude * (wave1 + wave2) / 2

        if img.ndim == 3:
            moire_pattern = moire_pattern[:, :, np.newaxis]

        result = np.clip(img.astype(np.float32) + moire_pattern, 0, 255)
        return result.astype(np.uint8)

    def _apply_shadow(self, img: np.ndarray) -> np.ndarray:
        """模拟手持拍摄时的轻微阴影遮挡"""
        h, w = img.shape[:2]
        strength = self._rng.uniform(*self.config.shadow_strength_range)
        radius_frac = self._rng.uniform(*self.config.shadow_radius_range)

        # 在随机位置创建圆形阴影
        cx = self._rng.randint(int(w * 0.2), int(w * 0.8))
        cy = self._rng.randint(int(h * 0.2), int(h * 0.8))
        radius = int(min(h, w) * radius_frac)

        Y, X = np.mgrid[:h, :w]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        # 平滑的圆形阴影衰减
        shadow_mask = np.clip(dist / radius, 0, 1)
        shadow_mask = shadow_mask ** 2  # 更平滑的衰减
        # 反转：阴影中心最暗
        shadow_factor = strength + (1 - strength) * shadow_mask

        if img.ndim == 3:
            shadow_factor = shadow_factor[:, :, np.newaxis]

        result = np.clip(img.astype(np.float32) * shadow_factor, 0, 255)
        return result.astype(np.uint8)


def apply_single_degradation(image: np.ndarray, kind: str, seed: int | None = None) -> np.ndarray:
    """
    便捷函数：只应用一种退化效果。

    Parameters
    ----------
    image : np.ndarray
        输入图像 (BGR, uint8)。
    kind : str
        退化类型: blur, noise, contrast, downscale, jpeg, affine,
                  paper_texture, uneven_light, moire, shadow。
    seed : int | None
        随机种子。

    Returns
    -------
    np.ndarray
        退化后的图像。
    """
    config = DegradationConfig(
        p_blur=1.0 if kind == "blur" else 0.0,
        p_noise=1.0 if kind == "noise" else 0.0,
        p_contrast=1.0 if kind == "contrast" else 0.0,
        p_downscale=1.0 if kind == "downscale" else 0.0,
        p_jpeg=1.0 if kind == "jpeg" else 0.0,
        p_affine=1.0 if kind == "affine" else 0.0,
        p_paper_texture=1.0 if kind == "paper_texture" else 0.0,
        p_uneven_light=1.0 if kind == "uneven_light" else 0.0,
        p_moire=1.0 if kind == "moire" else 0.0,
        p_shadow=1.0 if kind == "shadow" else 0.0,
    )
    pipeline = DegradationPipeline(config=config, seed=seed)
    return pipeline.apply(image)
