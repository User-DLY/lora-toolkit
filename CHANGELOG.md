# Changelog

本项目的所有重要变更都会记录在此文件。

格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

---

## [0.0.2] - 2026-10-31

### 新增

- **窗口控件操作工具** `win_ctrl.py`
  - 基于 Windows UI Automation（`uiautomation` 库）
  - 按控件名定位，使用物理像素坐标，规避截图坐标与 DPI 缩放不一致的问题
  - 功能：列出窗口控件、按名称点击、按名称输入、读取控件值、按文字搜索控件
  - 子命令：`screen` / `windows` / `list` / `click` / `type` / `read` / `find`

- **依赖** 新增 `uiautomation>=2.0`、`comtypes>=1.2`

### 说明

Windows 的高 DPI 环境下，截图坐标（逻辑像素）与控件实际坐标（物理像素）存在缩放比例差异，
按截图像素点击会累积偏差。`win_ctrl.py` 通过 UI Automation 直接获取控件包围盒，
不依赖任何坐标换算。

---
## [0.0.1] - 2026-10-31

### 新增

- **命令行入口** `train_cli.py`
  - 30 项可配置参数，覆盖数据、超参数、检查点、评估、监控、启动六类
  - 支持 GNU 风格（`--xxx`）、短选项（`-xx`）、DOS 风格（`/xxx`）三种书写方式
  - `/?`、`/h`、`/help`、`/usage` 均映射至帮助信息

- **进程管理** `proc_mgr.py`
  - PID 登记表（`run/pids.json`）
  - 按名称查询、精确停止，避免 `taskkill /IM python.exe` 的全局误杀
  - 已终止进程的登记清理

- **独立进程启动** `launch_detached.py`
  - 使用 `DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP` 标志
  - 父进程退出不影响子进程
  - 启动时自动登记 PID

- **自动恢复** `auto_resume.py`
  - 通过注册表 `HKCU\Software\Microsoft\Windows\CurrentVersion\Run` 触发
  - 三条件自毁：训练完成、尝试次数超限、状态文件缺失
  - 自毁时同步清理注册表项与计数器

- **进度页面** `progress_server.py`
  - 本地 HTTP 服务，默认端口 8088
  - 实时显示步数、损失、验证损失、显存、内存、预计剩余时间
  - 损失曲线可视化

- **训练主程序** `trainer.py`
  - LoRA 微调，支持 fp16 + GradScaler
  - 双槽原子检查点 + 历史归档两级结构
  - 五路监控（显存、内存、损失、速度、硬上限），全部可配置
  - 训练完成后主动清理自动恢复注册表项

### 变更

- 检查点由单槽改为双槽 + 历史归档
- 监控阈值由源码常量改为配置文件 + 命令行参数
- 上下文长度由单一常量改为 `ctx_map.json` 单一数据源驱动

### 修复

- 监控模块中不存在的变量引用（`_loss` 应为 `step_loss`）导致的 `NameError`
- `global` 声明位置错误导致的 `SyntaxError`
- 配置文件 UTF-8 BOM 导致的 JSON 解析失败（统一使用 `utf-8-sig`）
- 训练完成后自动恢复注册表项残留（需等待下次登录才清理）

### 已知问题

- `TRAIN_NO_GC=1` 在 8 GB 显卡上会触发内存交换，实测训练速度降至 1/3.2
- 验证集与训练集不同源，`val_loss` 仅可用于趋势观察
- 系统级异常重启无法由软件侧消除

---

## 版本号约定

```
0.0.1  初始可用版本
0.0.2  修复类更新（不改变接口）
0.1.0  新增功能（向后兼容）
1.0.0  接口稳定，可用于生产
```
