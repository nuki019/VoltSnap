"""ngspice subprocess 调用与输出解析"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path

from voltsnap.models import SimulationResult

logger = logging.getLogger("voltsnap.simulation.ngspice")

# 正则匹配 ngspice .op 输出
# 格式 1（详细模式）: V(N1): 5 voltage
# 格式 2（简洁模式）: n1                5.000000e+00
_VOLTAGE_RE = re.compile(
    r"(?:[Vv]\((\w+)\)|^\s*(\w+))\s*[:\s]\s*([-\d.eE+]+)"
)
# 电流格式: v1#branch: -1.66667e-03  或  v1#branch    -1.66667e-03
_CURRENT_RE = re.compile(
    r"(v\d+)#branch[:\s]+\s*([-\d.eE+]+)"
)


class NgspiceRunner:
    """通过 subprocess 调用 ngspice 执行 .op 仿真"""

    def __init__(self, ngspice_path: str = "ngspice", timeout: int = 10):
        self.ngspice_path = ngspice_path
        self.timeout = timeout

    def run(self, netlist_text: str) -> SimulationResult:
        """
        将网表写入临时文件，调用 ngspice -b 执行，解析输出。
        """
        # 创建临时 .cir 文件
        cir_fd, cir_path = tempfile.mkstemp(suffix=".cir")
        log_fd, log_path = tempfile.mkstemp(suffix=".log")
        os.close(cir_fd)
        os.close(log_fd)

        try:
            # 写入网表
            Path(cir_path).write_text(netlist_text, encoding="utf-8")

            # 调用 ngspice
            cmd = [self.ngspice_path, "-b", "-o", log_path, cir_path]
            logger.info("Running: %s", " ".join(cmd))

            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            # 读取输出日志
            log_text = Path(log_path).read_text(encoding="utf-8", errors="replace")

            if proc.returncode != 0 and not log_text.strip():
                return SimulationResult(
                    success=False,
                    raw_output=proc.stderr,
                    error_message=f"ngspice exited with code {proc.returncode}",
                )

            return self._parse_output(log_text)

        except subprocess.TimeoutExpired:
            return SimulationResult(
                success=False,
                error_message=f"ngspice timed out after {self.timeout}s",
            )
        except FileNotFoundError:
            return SimulationResult(
                success=False,
                error_message=f"ngspice not found at: {self.ngspice_path}",
            )
        finally:
            # 清理临时文件
            for p in (cir_path, log_path):
                try:
                    os.unlink(p)
                except OSError:
                    pass

    def _parse_output(self, output_text: str) -> SimulationResult:
        """解析 ngspice .op 输出，提取节点电压和支路电流"""
        node_voltages: dict[str, float] = {}
        branch_currents: dict[str, float] = {}

        lines = output_text.splitlines()

        # 找到数据区段的起始位置（"Node Voltage" 表头之后）
        start_idx = 0
        for i, line in enumerate(lines):
            if "Operating Point" in line:
                start_idx = i + 1
                break
            if "Node" in line and "Voltage" in line:
                start_idx = i + 1
                # 跳过分隔线
                while start_idx < len(lines) and set(lines[start_idx].strip()) <= {"-", "\t", " "}:
                    start_idx += 1
                break

        # 解析电压和电流数据，遇到模型参数段停止
        for line in lines[start_idx:]:
            stripped = line.strip()
            # 遇到模型参数段则停止
            if any(stripped.startswith(kw) for kw in ("Resistor models", "Resistor:", "Vsource:", "Capacitor:", "Total analysis")):
                break

            # 匹配节点电压
            m = _VOLTAGE_RE.search(line)
            if m:
                node_name = m.group(1) or m.group(2)
                try:
                    value = float(m.group(3))
                except ValueError:
                    continue
                if node_name and not node_name.isdigit():
                    node_voltages[node_name.upper()] = value
                continue

            # 匹配支路电流
            m = _CURRENT_RE.search(line)
            if m:
                branch_name = m.group(1)
                try:
                    value = float(m.group(2))
                except ValueError:
                    continue
                branch_currents[branch_name.upper()] = value

        success = len(node_voltages) > 0
        error_msg = None if success else "No voltage data found in output"

        if not success:
            logger.warning("Parse failed. Raw output:\n%s", output_text)

        return SimulationResult(
            success=success,
            node_voltages=node_voltages,
            branch_currents=branch_currents,
            raw_output=output_text,
            error_message=error_msg,
        )
