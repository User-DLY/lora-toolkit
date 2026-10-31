# LoRA Toolkit

**Version 0.0.1**

Windows 环境下低显存 GPU（8 GB 级）的 LoRA 微调工具链。面向本地 Agent 模型的训练、恢复与监控场景。

---

## 1. 项目定位

本项目解决在消费级显卡上运行 LoRA 微调时的四类工程问题：

| 问题 | 现象 | 本项目方案 |
|---|---|---|
| 进程生命周期 | 启动器退出导致训练中断 | `DETACHED_PROCESS` 独立进程组 |
| 训练中断恢复 | 系统异常重启后训练进度丢失 | 注册表 Run + 三条件自动恢复 |
| 进程误杀 | `taskkill /IM python.exe` 波及其它进程 | PID 登记表 + 精确停止 |
| 参数硬编码 | 超参数变更需修改源码 | 配置文件驱动 + 完整 CLI |

---

## 2. 系统要求

| 项目 | 要求 |
|---|---|
| 操作系统 | Windows 10 / 11 |
| Python | 3.10 及以上（开发环境为 3.12） |
| PyTorch | 2.1 及以上（ROCm 或 CUDA 后端） |
| 显存 | 建议 8 GB 及以上 |
| 浏览器 | Edge（默认）或 Chrome |

---

## 3. 安装

```powershell
git clone <repository-url>
cd lora-toolkit
pip install -r requirements.txt
```

---

## 4. 快速开始

```powershell
# 使用默认配置启动训练
python train_cli.py

# 查看完整参数
python train_cli.py --help

# 查看受管进程状态
python proc_mgr.py list

# 停止训练进程（不影响进度页）
python proc_mgr.py stop train
```

训练启动后将自动打开进度页面（`http://127.0.0.1:8088`）。

---

## 5. 模块说明

| 模块 | 职责 |
|---|---|
| `train_cli.py` | 命令行入口，参数解析与训练调度 |
| `proc_mgr.py` | 进程登记、状态查询、精确停止 |
| `launch_detached.py` | 以 `DETACHED_PROCESS` 启动子进程并登记 PID |
| `auto_resume.py` | 登录时检查训练状态，未完成则恢复，满足终止条件则自毁 |
| `progress_server.py` | 本地训练进度监控页面 |
| `trainer.py` | LoRA 训练主程序 |
| `chat_template.jinja` | 训练用对话模板 |

---

## 6. 命令行参数

### 6.1 数据

| 参数 | 短选项 | 默认值 | 说明 |
|---|---|---|---|
| `--data` | `-d` | `data-v1.5/train_v113_train.jsonl` | 训练数据（JSONL） |
| `--cache` | `-c` | 自动推导 | 编码缓存（PKL） |
| `--eval-cache` | — | 内置 | 验证集缓存 |
| `--maxlen` | `-ml` | `1536` | 最大序列长度 |

### 6.2 训练超参数

| 参数 | 短选项 | 默认值 | 说明 |
|---|---|---|---|
| `--epochs` | `-e` | `2` | 训练轮数 |
| `--lr` | `-lr` | `5e-5` | 学习率 |
| `--r` | `-r` | `16` | LoRA rank |
| `--alpha` | `-a` | `32` | LoRA alpha |
| `--batch` | `-b` | `1` | 批大小 |
| `--accum` | `-ac` | `8` | 梯度累积步数 |
| `--ce-chunk` | `-cc` | `512` | 交叉熵分块大小 |

### 6.3 检查点

| 参数 | 短选项 | 默认值 | 说明 |
|---|---|---|---|
| `--ckpt-every` | `-ce` | `5` | 保存间隔（步） |
| `--ckpt-keep` | `-ck` | `3` | 历史检查点保留数量（0 表示全部保留） |
| `--ckpt-dir` | — | 同 `--out` | 检查点目录 |
| `--out` | `-o` | `out-v113` | 适配器输出目录 |

检查点采用两级存储结构：

```
out-v113/
├── ckpt_a.pt              # 双槽热备份（用于断点恢复，原子写入）
├── ckpt_b.pt
└── ckpt-history/
    ├── step_0220.pt       # 历史归档（按 --ckpt-keep 滚动保留）
    ├── step_0225.pt
    └── step_0230.pt
```

### 6.4 评估

| 参数 | 短选项 | 默认值 | 说明 |
|---|---|---|---|
| `--eval-every` | `-ee` | `100` | 评估间隔（步），0 表示禁用 |

### 6.5 监控与恢复

| 参数 | 短选项 | 默认值 | 说明 |
|---|---|---|---|
| `--wd-vram-alloc` | `-wa` | `6.2` | GPU 显存张量阈值（GB） |
| `--wd-vram-free` | `-wf` | `0.25` | 驱动报告剩余显存阈值（GB） |
| `--wd-vram-times` | `-wt` | `3` | 显存超限连续触发次数上限 |
| `--wd-mem-warn` | `-mw` | `1.5` | 系统内存告警阈值（GB，仅告警） |
| `--wd-loss-max` | `-lm` | `20.0` | 损失绝对值阈值 |
| `--wd-loss-ratio` | `-lq` | `3.0` | 相对前 5 步均值的倍数阈值 |
| `--wd-loss-times` | `-lt` | `2` | 损失异常连续触发次数上限 |
| `--wd-slow-sec` | `-ss` | `60.0` | 单步耗时阈值（秒） |
| `--wd-slow-times` | `-st` | `3` | 慢步连续触发次数上限 |
| `--vram-limit` | `-vl` | `6.6` | 显存张量硬上限（GB） |
| `--mem-abort` | `-ma` | `93.0` | 系统内存占用率上限（%） |

监控采用**连续触发机制**：单次超限仅记录告警，连续 N 次达到阈值才中止训练。该设计用于规避 GPU 驱动在训练满载时的显存读数异常。

### 6.6 启动与恢复

| 参数 | 短选项 | 默认值 | 说明 |
|---|---|---|---|
| `--resume` / `--no-resume` | — | 启用 | 是否启用断点恢复 |
| `--preload-adapter` | `-pa` | 空 | 预加载适配器目录（增量训练） |
| `--max-attempts` | `-mx` | `5` | 自动恢复最大尝试次数 |
| `--self-destruct` / `--no-self-destruct` | — | 启用 | 训练完成后是否清理注册表项 |
| `--detached` / `--no-detached` | — | 启用 | 是否以独立进程启动 |
| `--gpu-warmup` | `-gw` | `3` | 启动前 GPU 预热秒数（0 表示跳过） |

### 6.7 进度页面

| 参数 | 短选项 | 默认值 | 说明 |
|---|---|---|---|
| `--port` | `-p` | `8088` | 进度页面端口 |
| `--open-page` / `--no-open-page` | — | 启用 | 是否自动打开进度页面 |
| `--browser` | `-br` | `edge` | 浏览器选择：`edge` / `chrome` / `none` |
| `--log` | — | 自动生成 | 训练日志路径 |
| `--quiet` | `-q` | 关闭 | 减少标准输出 |

---

## 7. 参数书写风格

支持三种等效写法：

```powershell
# GNU 风格
python train_cli.py --ckpt-every 3 --ckpt-keep 5

# 短选项
python train_cli.py -ce 3 -ck 5

# DOS 风格（Windows 习惯）
python train_cli.py /ce 3 /ck 5
python train_cli.py /help
python train_cli.py /?
```

DOS 风格转换规则：

| 写法 | 等效参数 |
|---|---|
| `/?`、`/h`、`/help`、`/usage` | `--help` |
| `/ce 3` | `--ckpt-every 3` |
| `/wa=6.5` | `--wd-vram-alloc 6.5` |
| `/wd-vram-alloc 6.5` | `--wd-vram-alloc 6.5` |
| `/no-resume` | `--no-resume` |

---

## 8. 进程隔离设计

### 8.1 问题背景

Windows 环境下，通过 `Start-Process` 或 `cmd /c start` 启动的子进程会绑定到父进程的 Job Object，父会话结束时子进程被一并终止。

### 8.2 解决方案

`launch_detached.py` 使用 `DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP` 标志启动子进程：

```python
flags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
subprocess.Popen([...], creationflags=flags, close_fds=True)
```

### 8.3 PID 登记

每个受管进程启动后登记至 `run/pids.json`：

```json
{
  "train":    {"pid": 10720, "started": "2026-10-31 12:35:21"},
  "progress": {"pid": 3040,  "started": "2026-10-31 12:19:29"}
}
```

`proc_mgr.py stop <name>` 仅终止指定 PID，不影响其它 Python 进程。

### 8.4 自动恢复的终止条件

`auto_resume.py` 在以下任一条件满足时删除注册表项并退出：

1. 训练已完成（`step >= total` 且 `aborted == false`）
2. 尝试次数达到 `--max-attempts` 上限
3. `status.json` 不存在（输出目录被删除或路径变更）

此外，训练正常完成时，`trainer.py` 会主动清理注册表项，无需等待下次登录。

---

## 9. 性能参考

测试环境：8 GB 显存消费级 GPU，1.7B 参数模型，960 条样本，`maxlen=1536`。

| 配置 | 单步耗时 | 240 步总耗时 |
|---|---|---|
| 无预热 | 22.4 s | 89 分钟 |
| GPU 预热 3 秒 | 15.5 s | 62 分钟 |
| 预热 + 稳态 | 14.0 s | 56 分钟 |

优化建议：

- **GPU 预热**：启动前进行 3–5 秒 GPU 压力测试，可使风扇提前进入工作状态，避免因热降频导致首步延迟。
- **评估频率**：`--eval-every` 不宜过小。验证集评估（259 条）单次约 3 分钟，每 50 步一次的评估开销约占总时长 25%。
- **检查点间隔**：`--ckpt-every` 需权衡恢复粒度与 I/O 开销。

---

## 10. 已知限制

1. **`TRAIN_NO_GC=1` 不建议启用**
   实测显存占用由 3.5 GB 增至 10.45 GB，在 8 GB 显卡上触发内存交换，训练速度下降至原来的 1/3.2。源码中的「可获得 25% 速度提升」注释与实际不符。

2. **验证集与训练集不同源**
   `val_loss` 绝对值不可跨实验比较，仅可用于观察同一训练内的收敛趋势。

3. **系统级异常重启无法由软件侧消除**
   训练期间 GPU 满载导致机箱温度升高，散热不足可能触发存储或内存类异常关机。可采取的硬件侧措施包括限制 GPU 功耗、提高风扇转速、改善机箱风道；软件侧仅能做到「重启后自动恢复」与「减少进度损失」。

---

## 11. 版本历史

| 版本 | 日期 | 内容 |
|---|---|---|
| **0.0.1** | 2026-10-31 | 初始版本。命令行入口、进程管理、独立启动、自动恢复、进度页面、监控参数化。 |

---

## 12. 许可证

MIT License

---

## 13. 窗口控件操作（Windows UI Automation）

win_ctrl.py 基于 Windows UI Automation 提供按控件名的精确操作，
不依赖截图坐标。

### 13.1 背景

高 DPI 环境下，截图坐标（逻辑像素）与控件实际坐标（物理像素）存在缩放差异。
以本机为例：

| 项 | 分辨率 |
|---|---|
| 物理屏幕 | 2560 × 1440 |
| 系统报告逻辑像素 | 2048 × 1152（DPI 125%） |
| 常规截图 | 1366 × 768 |

按截图坐标点击会累积偏差。UI Automation 直接返回控件包围盒（物理像素），无换算误差。

### 13.2 子命令

`powershell
# 屏幕信息
python win_ctrl.py screen

# 列出所有顶层窗口
python win_ctrl.py windows

# 列出指定窗口内所有可点击控件
python win_ctrl.py list "窗口标题片段"

# 按控件名点击
python win_ctrl.py click "窗口标题片段" "按钮名"

# 按控件名输入文本
python win_ctrl.py type "窗口标题片段" "输入框名" "要输入的文字"

# 按控件名读取值
python win_ctrl.py read "窗口标题片段" "控件名"

# 搜索名称包含指定文字的控件
python win_ctrl.py find "窗口标题片段" "关键词"
`

### 13.3 示例

`powershell
# 列出 Edge 中所有按钮
python win_ctrl.py list "Microsoft Edge"

# 点击 Edge 的刷新按钮
python win_ctrl.py click "Microsoft Edge" "刷新"
`

### 13.4 依赖

`powershell
pip install uiautomation comtypes
`

