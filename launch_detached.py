# -*- coding: utf-8 -*-
"""训练启动器：用 DETACHED_PROCESS 让训练进程彻底脱离父进程

背景（2026-10-31 实测）：
  用 cmd /c start /B 或 Start-Process 启动训练，
  父 PowerShell 会话一结束，训练进程就被带走（job object 绑定）。
  三次崩溃都是这个原因：step 4 / 186 / 200，每次都在跑新命令的时候。
  err 日志无异常、aborted: false、系统事件无 GPU 重置 → 纯外部杀进程。

解法：
  subprocess.Popen(..., creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP)
  DETACHED_PROCESS (0x8) 让子进程不继承父进程的控制台/job
  CREATE_NEW_PROCESS_GROUP (0x200) 让它有独立的进程组

用法：
  python launch_detached.py <脚本路径> <日志路径>
  例：python launch_detached.py trainer.py D:\\...\\tv113f-run.log
"""
import os
import subprocess
import sys

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_NO_WINDOW = 0x08000000


def main():
    if len(sys.argv) < 3:
        print("用法: python launch_detached.py <脚本> <日志>")
        sys.exit(1)

    script = os.path.abspath(sys.argv[1])
    logpath = os.path.abspath(sys.argv[2])
    errpath = logpath + ".err"

    if not os.path.isfile(script):
        print("找不到脚本: %s" % script)
        sys.exit(1)

    py = sys.executable
    workdir = os.path.dirname(script)

    # 打开日志文件（子进程要自己持有句柄）
    fout = open(logpath, "w", encoding="utf-8", errors="replace")
    ferr = open(errpath, "w", encoding="utf-8", errors="replace")

    flags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW

    p = subprocess.Popen(
        [py, "-u", script],
        cwd=workdir,
        stdout=fout,
        stderr=ferr,
        stdin=subprocess.DEVNULL,
        creationflags=flags,
        close_fds=True,
    )

    # ★登记 PID（供 proc_mgr.py 精确停止，避免 taskkill /IM python.exe 全杀）
    try:
        import json as _json, time as _time
        _piddir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run")
        os.makedirs(_piddir, exist_ok=True)
        _pidfile = os.path.join(_piddir, "pids.json")
        _d = {}
        if os.path.isfile(_pidfile):
            try:
                with open(_pidfile, encoding="utf-8-sig") as _f:
                    _d = _json.load(_f) or {}
            except Exception:
                _d = {}
        _base = os.path.basename(script).lower()
        if "train_v14" in _base:
            _name = "train"
        elif "progress_server" in _base:
            _name = "progress"
        elif "auto_resume" in _base:
            _name = "auto"
        else:
            _name = _base.replace(".py", "")
        _d[_name] = {"pid": p.pid, "ts": _time.time(),
                     "started": _time.strftime("%Y-%m-%d %H:%M:%S"),
                     "script": script}
        with open(_pidfile, "w", encoding="utf-8") as _f:
            _json.dump(_d, _f, ensure_ascii=False, indent=2)
        print("  登记: %s -> run/pids.json" % _name)
    except Exception as _e:
        print("  登记失败: %r" % (_e,))

    print("已启动（完全脱离父进程）")
    print("  PID:  %d" % p.pid)
    print("  脚本: %s" % script)
    print("  日志: %s" % logpath)
    print("  错误: %s" % errpath)
    print()
    print("验证: 关闭这个 PowerShell 会话后，进程应该还在跑")


if __name__ == "__main__":
    main()
