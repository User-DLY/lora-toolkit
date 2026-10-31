# -*- coding: utf-8 -*-
"""训练自动恢复：开机时检查，未完成就继续，完成/超限就自毁

用户要求（2026-10-31）：
  「计划任务要有限度，完了就自己踢出启动部队」
  → 任务不能永久存在，必须有明确的终止条件

自毁条件（三个，任一满足就删自己）：
  ① 训练已完成（step >= total 且 aborted == false）
  ② 尝试次数超过 MAX_ATTEMPTS（默认 5 次，防无限重启循环）
  ③ 找不到 status.json（说明产物目录被删/路径错了，任务无效）

工作流程：
  1. 读 out-v113/status.json
  2. 判断是否满足自毁条件
  3. 不满足 → 用 launch_detached 启动训练（如果还没在跑）
  4. 满足 → schtasks /Delete 删掉自己，退出
"""
import json
import os
import subprocess
import sys
import time

# 路径由脚本位置推导
_HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.environ.get("LORA_PYTHON_DIR") or _HERE
TRAINER = os.path.join(TRAIN, "trainer.py")
LAUNCHER = os.path.join(TRAIN, "..", "..", "..", "..", "..", "D", "x")  # 占位，下面会覆盖
LAUNCHER = os.path.join(_HERE, "launch_detached.py")
STATUS = os.path.join(WORK, "out", "status.json")
COUNTER = os.path.join(WORK, ".auto_resume_count")
LOG = os.path.join(WORK, "auto-resume.log")
TASK_NAME = "LoraTrainResume"

_HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get("LORA_WORK_DIR") or _HERE
MAX_ATTEMPTS = 5


def w(msg):
    line = "[%s] %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass
    print(line.rstrip())


def self_destruct(reason):
    """自毁：删掉注册表 Run 项（不需要提权）"""
    w("自毁：%s" % reason)
    try:
        import winreg
        k = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                           r"Software\Microsoft\Windows\CurrentVersion\Run",
                           0, winreg.KEY_SET_VALUE)
        winreg.DeleteValue(k, TASK_NAME)
        winreg.CloseKey(k)
        w("注册表 Run 项 %s 已删除" % TASK_NAME)
    except FileNotFoundError:
        w("注册表 Run 项本来就不存在")
    except Exception as e:
        w("删除注册表项失败: %r" % (e,))
    # 顺手清掉计数器
    try:
        if os.path.isfile(COUNTER):
            os.remove(COUNTER)
    except Exception:
        pass
    sys.exit(0)


def is_training_running():
    """检查训练进程是否在跑"""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object { $_.CommandLine -match 'train_v14' } | "
             "Measure-Object | Select-Object -ExpandProperty Count"],
            capture_output=True, timeout=30, text=True, encoding="utf-8", errors="replace")
        return out.stdout.strip().isdigit() and int(out.stdout.strip()) > 0
    except Exception:
        return False


def main():
    w("=" * 50)
    w("auto_resume 启动检查")

    # 条件 ③：status.json 不存在 → 任务无效
    if not os.path.isfile(STATUS):
        self_destruct("找不到 status.json（产物目录可能被删）")

    # 读 status
    try:
        with open(STATUS, encoding="utf-8-sig") as f:
            st = json.load(f)
    except Exception as e:
        self_destruct("status.json 读取失败: %r" % (e,))

    step = int(st.get("step", 0))
    total = int(st.get("total", 0))
    aborted = bool(st.get("aborted", False))
    w("status: step=%d/%d aborted=%s" % (step, total, aborted))

    # 条件 ①：已完成
    if total > 0 and step >= total and not aborted:
        self_destruct("训练已完成（%d/%d）" % (step, total))

    # 条件 ②：尝试次数超限
    attempts = 0
    if os.path.isfile(COUNTER):
        try:
            attempts = int(open(COUNTER).read().strip())
        except Exception:
            attempts = 0
    if attempts >= MAX_ATTEMPTS:
        self_destruct("尝试次数 %d >= %d（防无限重启）" % (attempts, MAX_ATTEMPTS))

    # 已在跑 → 不重复启动
    if is_training_running():
        w("训练已在跑，本次不启动（下次开机再查）")
        sys.exit(0)

    # 启动
    attempts += 1
    try:
        with open(COUNTER, "w") as f:
            f.write(str(attempts))
    except Exception:
        pass

    w("启动训练（第 %d/%d 次尝试，从 step %d 继续）" % (attempts, MAX_ATTEMPTS, step))
    try:
        subprocess.run([sys.executable, "-u", LAUNCHER, TRAINER,
                        os.path.join(WORK, "auto-resume-run.log")],
                       capture_output=True, timeout=120,
                       cwd=os.path.dirname(LAUNCHER))
        w("已启动")
    except Exception as e:
        w("启动失败: %r" % (e,))


if __name__ == "__main__":
    main()
