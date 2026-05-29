# ngspice Windows 安装指南

## 下载

从 https://ngspice.sourceforge.io 下载 Windows 64-bit 版本（推荐 ngspice-42 或更新）。

选择 "DLL and EXE" 包（包含 `ngspice.exe` 和 `ngspice.dll`）。

## 安装

1. 解压到固定路径，例如 `C:\tools\ngspice\`
2. 将 `C:\tools\ngspice\bin\` 加入系统 PATH：
   - 右键"此电脑" → 属性 → 高级系统设置 → 环境变量
   - 在"系统变量"中找到 `Path`，编辑，添加 `C:\tools\ngspice\bin\`

## 验证

```bash
ngspice --version
```

应输出版本号，如 `ngspice-42`。

## 环境变量（可选）

如果不想修改 PATH，可以设置 `NGSPICE_PATH` 环境变量指向 `ngspice.exe` 的完整路径：

```bash
set NGSPICE_PATH=C:\tools\ngspice\bin\ngspice.exe
```
