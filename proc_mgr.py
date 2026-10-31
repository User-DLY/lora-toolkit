# -*- coding: utf-8 -*-
"""进程管理器：登记 / 查询 / 精确停止（互不牵连）

用户要求（2026-10-31）：
  「不要 1 个死了，其他全都死了」
  → 每个进程独立启动、独立登记、独立停止
  → 绝不用 taskkill /IM python.exe（会杀掉所有 python）

三个受管进程：
  train     训练进程（trainer.py）
  progress  进度页（progress_server.py）
  auto      自动恢复（auto_resume.py，通常由注册表 Run 拉起）

用法：
  python proc_mgr.py list                     # 列出所有受管进程 + 存活状态
  python proc_mgr.py stop train               # 只停训练
  python proc_mgr.py stop progress            # 只停进度页
  python proc_mgr.py stop all                 # 全停（逐个精确杀）
  python proc_mgr.py clean                    # 清掉已死的登记
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PIDFILE = os.path.join(HERE, "run", "pids.json")
NAMES = ("train", "progress", "auto")


def _load():
    os.makedirs(os.path.dirname(PIDFILE), exist_ok=True)
    if not os.path.isfile(PIDFILE):
        return {}
    try:
        with open(PIDFILE, encoding="utf-8-sig") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _save(d):
    os.makedirs(os.path.dirname(PIDFILE), exist_ok=True)
    with open(PIDFILE, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)


def alive(pid):
    """检查 PID 是否还活着"""
    if not pid:
        return False
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "if (Get-Process -Id %d -ErrorAction SilentlyContinue) { '1' } else { '0' }" % int(pid)],
            capture_output=True, text=True, timeout=15,
            encoding="utf-8", errors="replace")
        return out.stdout.strip() == "1"
    except Exception:
        return False


def register(name, pid):
    """登记一个进程"""
    if name not in NAMES:
        print("未知进程名: %s（可选 %s）" % (name, "/".join(NAMES)))
        return False
    d = _load()
    d[name] = {"pid": int(pid), "ts": time.time(),
               "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    _save(d)
    return True


def unregister(name):
    d = _load()
    if name in d:
        del d[name]
        _save(d)


def stop_one(name, pid):
    """只杀这一个 PID（不会牵连其他 python）"""
    if not pid:
        return False
    try:
        r = subprocess.run(["taskkill", "/F", "/PID", str(int(pid))],
                           capture_output=True, text=True, timeout=20,
                           encoding="utf-8", errors="replace")
        ok = "SUCCESS" in (r.stdout or "") or "成功" in (r.stdout or "")
        if not ok and alive(pid):
            # 尝试 tree kill（只对这个 PID 的子树）
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(int(pid))],
                           capture_output=True, text=True, timeout=20)
        print("  [stop] %s (PID %d): %s" % (name, int(pid), "已停" if not alive(pid) else "仍在"))
        return not alive(pid)
    except Exception as e:
        print("  [stop] %s (PID %d) 失败: %r" % (name, int(pid), e))
        return False


def cmd_list():
    d = _load()
    if not d:
        print("没有登记的进程")
        return
    print("%-10s %8s %-20s %s" % ("名称", "PID", "启动时间", "状态"))
    print("-" * 58)
    any_alive = False
    for name in NAMES:
        if name not in d:
            continue
        info = d[name]
        pid = info.get("pid")
        a = alive(pid)
        any_alive = any_alive or a
        print("%-10s %8s %-20s %s" % (name, pid, info.get("started", "?"),
                                      "运行中" if a else "已停"))
    print()
    if not any_alive:
        print("（全部已停）")


def cmd_stop(targets):
    d = _load()
    if not d:
        print("没有登记的进程")
        return
    if "all" in targets:
        targets = [n for n in NAMES if n in d]
    for name in targets:
        if name not in d:
            print("  %s: 未登记" % name)
            continue
        pid = d[name].get("pid")
        if alive(pid):
            stop_one(name, pid)
        else:
            print("  [stop] %s (PID %s): 本来就已停" % (name, pid))
        unregister(name)
    print()
    print("已处理: %s" % ", ".join(targets))


def cmd_clean():
    d = _load()
    dead = [n for n in list(d.keys()) if not alive(d[n].get("pid"))]
    for n in dead:
        del d[n]
    _save(d)
    print("清掉 %d 条已死登记: %s" % (len(dead), ", ".join(dead) if dead else "无"))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cmd = sys.argv[1].lower()
    if cmd == "list":
        cmd_list()
    elif cmd == "stop":
        cmd_stop(sys.argv[2:])
    elif cmd == "clean":
        cmd_clean()
    elif cmd == "register":
        # 内部用：python proc_mgr.py register <name> <pid>
        if len(sys.argv) >= 4:
            register(sys.argv[2], int(sys.argv[3]))
    elif cmd == "unregister":
        if len(sys.argv) >= 3:
            unregister(sys.argv[2])
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
