"""VoltSnap 通用工具函数"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread_unicode(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """
    读取图片，支持 Unicode / 中文路径。

    cv2.imread 在 Windows 上不支持非 ASCII 路径，
    使用 Python open + cv2.imdecode 替代。

    Parameters
    ----------
    path : str | Path
        图片文件路径。
    flags : int
        cv2 读取标志，默认 IMREAD_COLOR。

    Returns
    -------
    np.ndarray | None
        读取成功返回图像数组，失败返回 None。
    """
    try:
        with open(str(path), "rb") as f:
            data = f.read()
    except (OSError, FileNotFoundError):
        return None
    if not data:
        return None
    buf = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(buf, flags)


def imwrite_unicode(path: str | Path, image: np.ndarray) -> bool:
    """
    写入图片，支持 Unicode / 中文路径。

    cv2.imwrite 在 Windows 上不支持非 ASCII 路径，
    使用 cv2.imencode + Python open 替代。

    Parameters
    ----------
    path : str | Path
        输出文件路径。
    image : np.ndarray
        图像数组。

    Returns
    -------
    bool
        写入成功返回 True。
    """
    ext = Path(path).suffix or ".png"
    success, buf = cv2.imencode(ext, image)
    if not success:
        return False
    try:
        with open(str(path), "wb") as f:
            f.write(buf.tobytes())
        return True
    except OSError:
        return False
