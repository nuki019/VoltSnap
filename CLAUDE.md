# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

VoltSnap 是一个 Python 桌面应用，拍照识别大学基础电路图并进行 SPICE 仿真。核心流程：图像输入 → 元件检测 → 拓扑重建 → 网表生成 → ngspice 仿真。

## 常用命令

```bash
# 环境搭建（conda 环境名：voltsnap，Python 3.11）
scripts\setup_conda_env.bat

# 运行全部测试
conda activate voltsnap && pytest tests/ -v

# 运行单个测试文件
pytest tests/test_vision.py -v

# 验证环境依赖
python tools/verify_env.py

# 启动 Demo 管线（定义 → 渲染 → 预处理 → 拓扑 → 网表 → 仿真）
python -m voltsnap.pipeline.demo_pipeline

# 启动 GUI
python -m voltsnap.gui.app

# YOLO 检测器训练（多步骤）
python scripts/train_detector.py --generate --samples 5000
python scripts/train_detector.py --convert
python scripts/train_detector.py --train --epochs 100 --batch 16
python scripts/train_detector.py --detect path/to/image.png
```

## 架构

分层管线架构，各模块职责单一：

```
voltsnap/
├── models.py          # 核心数据结构（dataclass）：PinInfo, ComponentInfo, CircuitSpec, RenderResult, SimulationResult 等
├── config.py          # 全局配置（路径、阈值）
├── datagen/           # Layer 1 - 数据生成：电路模板、schemdraw 渲染、图像退化、批量生成、网表生成
├── vision/            # Layer 2 - 图像处理：灰度化、二值化、骨架化（Zhang-Suen）、连通分量、引脚-网络映射
├── recognition/       # Layer 3 - 识别：YOLO OBB 检测、OCR 解析、匈牙利算法文本-元件绑定、端到端识别管线
├── simulation/        # Layer 4 - 仿真：ngspice 子进程调用、输出解析、网表校验（悬空引脚/接地检测）
├── gui/               # Layer 5 - GUI：PyQt6 三面板布局（图像/元件表/仿真结果），QThread 异步架构
└── pipeline/          # 管线编排：demo_pipeline.py 串联全部层级
```

### 关键设计决策

- **元件检测**：YOLOv8 OBB（有向边界框），5 类元件（电阻/电容/电感/电压源/电流源）
- **拓扑重建**：骨架化 + 连通分量分析，引脚通过搜索半径吸附到最近网络。已知限制：闭环电路会被坍缩为单连通分量
- **文本绑定**：scipy 匈牙利算法最优匹配（scipy 不可用时降级为贪心匹配）
- **仿真**：通过 subprocess 调用 ngspice -b，正则解析输出。ngspice 需外部安装，默认路径 `C:\tools\ngspice\Spice64\bin\ngspice.exe`
- **OCR**：当前为模拟实现（从 annotation.json 读取），尚未接入真实 OCR 模型

### 可选依赖（惰性加载）

- `ultralytics` — YOLO 训练/推理
- `scipy` — 匈牙利算法
- `PyQt6` / `pyqtgraph` — GUI 和交互波形显示
- ngspice 外部二进制 — 仿真

## 开发注意事项

- `opencv-contrib-python` 必须是 contrib 版本（需要 `cv2.ximgproc.thinning`）
- 测试中需要 ngspice 的用例会自动 `pytest.skip`
- `.gitignore` 排除了 `*.pt` 模型文件
- matplotlib 在 schemdraw 渲染中使用 `Agg` 非交互后端
- 8 种电路拓扑类型：series_resistors, parallel_resistors, resistor_divider, rc_series, rc_parallel, rl_series, rlc_series, two_mesh
