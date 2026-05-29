# VoltSnap

电路图拍照识别与仿真软件。拍摄大学基础电路图，自动识别元件、导线拓扑和参数，生成 SPICE 网表并完成仿真。

## 环境搭建

### 方式一：一键脚本

```bash
scripts\setup_conda_env.bat
```

### 方式二：手动

```bash
conda create -n voltsnap python=3.11 -y
conda activate voltsnap
pip install schemdraw "opencv-contrib-python>=4.8" numpy matplotlib networkx pyyaml pytest
```

### ngspice 安装

参见 [tools/install_ngspice.md](tools/install_ngspice.md)。

### 验证

```bash
python tools/verify_env.py
```

## 运行测试

```bash
conda activate voltsnap
pytest tests/ -v
```

## 运行 Demo

```bash
python -m voltsnap.pipeline.demo_pipeline
```
