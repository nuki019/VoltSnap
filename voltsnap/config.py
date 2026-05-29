"""VoltSnap 全局配置"""
from __future__ import annotations

import os
from pathlib import Path


class Config:
    PROJECT_ROOT = Path(__file__).parent.parent

    DATA_DIR = PROJECT_ROOT / "data"
    SAMPLES_DIR = DATA_DIR / "samples"
    GENERATED_DIR = DATA_DIR / "generated"
    ANNOTATIONS_DIR = DATA_DIR / "annotations"

    NGSPICE_PATH = os.environ.get("NGSPICE_PATH", r"C:\tools\ngspice\Spice64\bin\ngspice.exe")

    RENDER_DPI = 150
    INCHES_PER_UNIT = 3.0

    PIN_SEARCH_RADIUS = 15       # 像素
    MIN_COMPONENT_AREA = 50      # 过滤噪点的最小连通域面积

    SIM_TIMEOUT = 10             # 秒

    @classmethod
    def ensure_dirs(cls) -> None:
        cls.GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        cls.ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
