"""OCR 识别与电学参数解析"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger("voltsnap.recognition.ocr")

# 电学单位标准化映射
_UNIT_MAP = {
    # 电阻
    "ohm": "", "ω": "", "Ω": "",
    "kohm": "k", "kω": "k", "kΩ": "k",
    "meg": "meg", "mω": "meg", "mΩ": "meg",
    # 电容
    "farad": "f", "ff": "f",
    "uf": "uf", "μf": "uf", "µf": "uf",
    "nf": "nf", "pf": "pf",
    # 电感
    "henry": "h", "mh": "mh", "uh": "uh", "μh": "uh",
    # 电压/电流
    "volt": "v", "mv": "mv", "kv": "kv",
    "amp": "a", "ma": "ma", "ua": "ua", "μa": "ua",
}

# 参数值正则：匹配类似 10k, 4.7uF, 0.1u, 100n, 5V, 1mA 等
_VALUE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*"
    r"([kKmMμµuUnNpPfFvV][aA]?[hHfFvVΩω]?)",
    re.IGNORECASE,
)

# 元件编号正则：R1, C2, L3, V1, I1, D1, U1, G1, S1, LED1, Q1, M1 等
_REF_RE = re.compile(
    r"^[RrLlCcVvIiDdUuGgSsMm]\d+$|^[Ll][Ee][Dd]\d+$|^[Qq]\d+$",
)


@dataclass
class OCRResult:
    """OCR 单条识别结果"""
    text: str            # 原始文本
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float    # 识别置信度
    # 解析后的结构化信息
    is_value: bool = False      # 是否是参数值
    is_ref: bool = False        # 是否是元件编号
    normalized: str = ""        # 标准化后的文本
    component_type: str = ""    # 推断的元件类型


class OCRParser:
    """
    OCR 识别结果解析器。

    功能：
    1. 从 OCR 原始文本中提取电学参数
    2. 标准化单位格式
    3. 区分元件编号和参数值
    4. 推断元件类型
    """

    def parse(self, ocr_results: list[dict]) -> list[OCRResult]:
        """
        解析 OCR 结果列表。

        Parameters
        ----------
        ocr_results : list[dict]
            OCR 原始结果，每个 dict 至少包含:
            - text: str
            - bbox: [x1, y1, x2, y2]
            - confidence: float

        Returns
        -------
        list[OCRResult]
            解析后的结构化结果。
        """
        parsed = []
        for item in ocr_results:
            result = self._parse_single(item)
            if result:
                parsed.append(result)
        return parsed

    def _parse_single(self, item: dict) -> OCRResult | None:
        """解析单条 OCR 结果"""
        text = item.get("text", "").strip()
        if not text or len(text) > 20:
            return None

        bbox = tuple(item.get("bbox", [0, 0, 0, 0]))
        conf = item.get("confidence", 0.0)

        result = OCRResult(
            text=text,
            bbox=bbox,
            confidence=conf,
        )

        # 尝试匹配元件编号
        if _REF_RE.match(text):
            result.is_ref = True
            result.normalized = text.upper()
            result.component_type = self._ref_to_type(text)
            return result

        # 尝试匹配参数值
        parsed_value = self._parse_value(text)
        if parsed_value:
            result.is_value = True
            result.normalized = parsed_value
            result.component_type = self._infer_type_from_value(parsed_value)
            return result

        # 都不是，保留原文本
        result.normalized = text
        return result

    def _parse_value(self, text: str) -> str | None:
        """解析并标准化电学参数值"""
        # 清理文本
        clean = text.strip().lower()
        clean = clean.replace(" ", "")

        # 匹配数值+单位
        m = _VALUE_RE.search(clean)
        if not m:
            return None

        number = m.group(1)
        unit_raw = m.group(2).lower()

        # 标准化单位
        unit = _UNIT_MAP.get(unit_raw, unit_raw)

        # 格式化
        if unit:
            return f"{number}{unit}"
        return number

    def _ref_to_type(self, ref: str) -> str:
        """从元件编号推断类型"""
        upper = ref.upper()
        if upper.startswith("LED"):
            return "led"
        prefix = ref[0].upper()
        type_map = {
            "R": "resistor",
            "C": "capacitor",
            "L": "inductor",
            "V": "voltage_source",
            "I": "current_source",
            "D": "diode",
            "U": "op_amp",
            "G": "ground",
            "S": "switch",
            "Q": "npn_transistor",
            "M": "nmos",
        }
        return type_map.get(prefix, "unknown")

    def _infer_type_from_value(self, value: str) -> str:
        """从参数值推断元件类型"""
        v = value.lower()
        # 电阻单位
        if any(u in v for u in ["k", "meg", "ohm"]):
            return "resistor"
        if re.match(r"^\d+(?:\.\d+)?$", v):
            return "resistor"  # 纯数字默认为电阻值

        # 电容单位
        if any(u in v for u in ["uf", "nf", "pf", "f"]):
            return "capacitor"

        # 电感单位
        if any(u in v for u in ["mh", "uh", "h"]):
            return "inductor"

        # 电压
        if "v" in v:
            return "voltage_source"

        # 电流
        if any(u in v for u in ["ma", "ua", "a"]):
            return "current_source"

        return "unknown"


def standardize_value(raw: str) -> str:
    """
    标准化电学参数值。

    Examples
    --------
    >>> standardize_value("4k7")
    '4.7k'
    >>> standardize_value("10μF")
    '10uf'
    >>> standardize_value("1MΩ")
    '1meg'
    """
    clean = raw.strip().lower().replace(" ", "")

    # 处理 "4k7" 格式（k/m 作为小数点）
    m = re.match(r"^(\d+)([km])(\d+)$", clean)
    if m:
        integer, sep, decimal = m.groups()
        unit = "k" if sep == "k" else "meg"
        return f"{integer}.{decimal}{unit}"

    # 处理标准格式
    m = _VALUE_RE.match(clean)
    if m:
        number = m.group(1)
        unit_raw = m.group(2).lower()
        unit = _UNIT_MAP.get(unit_raw, unit_raw)
        if unit:
            return f"{number}{unit}"
        return number

    return clean
