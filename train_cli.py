# -*- coding: utf-8 -*-
"""LoRA 训练启动器（CLI 工具，参数齐全）

用法：
  python train_cli.py                          # 用默认配置启动
  python train_cli.py --help                   # 看全部参数
  python train_cli.py --ckpt-keep 5 --browser edge
  python train_cli.py --preload-adapter out-v112 --epochs 1

设计原则：
  ① 参数 → 生成 config JSON → trainer.py 读取（不改源码常量）
  ② 默认 DETACHED 启动（不被父进程带走）
  ③ 自动开进度页（默认 Edge）
  ④ 自动恢复（注册表 Run + 自毁）
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PYDIR = r"C:\Users\13802\AppData\Local\Programs\Python\Python312"
TRAINER = os.path.join(PYDIR, "trainer.py")
LAUNCHER = os.path.join(HERE, "launch_detached.py")
PROGRESS = os.path.join(HERE, "progress_server.py")
DATA_DIR = os.path.join(HERE, "data-v1.5")
CONFIG_DIR = os.path.join(HERE, "run-configs")
REG_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
REG_NAME = "LoraTrainResume"

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_NO_WINDOW = 0x08000000

BROWSERS = {
    "edge": [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
             r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"],
    "chrome": [r"C:\Users\13802\AppData\Local\Google\Chrome\Application\chrome.exe",
               r"C:\Program Files\Google\Chrome\Application\chrome.exe"],
}


# ══════════════════════════════════════════════════════════════
# DOS 风格参数支持（2026-10-31）
#   /help  /h  /?  → --help
#   /wa 6.5        → --wd-vram-alloc 6.5
#   /ce 3          → --ckpt-every 3
#   也兼容 --xxx / -xx 原样透传
# ══════════════════════════════════════════════════════════════
SHORT_TO_LONG = {
    "d": "--data", "c": "--cache", "ml": "--maxlen",
    "e": "--epochs", "lr": "--lr", "r": "--r", "a": "--alpha",
    "b": "--batch", "ac": "--accum", "cc": "--ce-chunk",
    "ce": "--ckpt-every", "ck": "--ckpt-keep", "o": "--out",
    "ee": "--eval-every", "pa": "--preload-adapter", "mx": "--max-attempts",
    "gw": "--gpu-warmup", "p": "--port", "br": "--browser",
    "wa": "--wd-vram-alloc", "wf": "--wd-vram-free", "wt": "--wd-vram-times",
    "mw": "--wd-mem-warn", "lm": "--wd-loss-max", "lq": "--wd-loss-ratio",
    "lt": "--wd-loss-times", "ss": "--wd-slow-sec", "st": "--wd-slow-times",
    "vl": "--vram-limit", "ma": "--mem-abort",
    "q": "--quiet",
}
HELP_ALIASES = {"?", "h", "help", "usage", "-?", "/?"}


def normalize_argv(argv):
    """把 DOS 风格 /xxx 和常见拼写转成 argparse 认识的 --xxx

    支持的写法：
      /help  /h  /?  /usage      → --help
      /wa 6.5  /wa=6.5           → --wd-vram-alloc 6.5
      /ce 3                       → --ckpt-every 3
      /data my.jsonl              → --data my.jsonl
      /no-resume                  → --no-resume
      /no-resume=1                → --no-resume
      --xxx / -xx 原样透传
    """
    out = [argv[0]]
    for a in argv[1:]:
        # 已经是 - / -- 开头 → 透传
        if a.startswith("-"):
            low = a.lstrip("-").lower()
            if low in HELP_ALIASES:
                out.append("--help")
            else:
                out.append(a)
            continue

        # /xxx 形式
        if a.startswith("/") and len(a) > 1 and not a.startswith("//"):
            body = a[1:]
            # 拆 key=value
            if "=" in body:
                key, val = body.split("=", 1)
                tail = [val]
            else:
                key, tail = body, []

            k = key.lower()
            # /?  /h  /help  /usage
            if k in HELP_ALIASES:
                out.append("--help")
                continue
            # /no-xxx → --no-xxx
            if k.startswith("no-") or k.startswith("no_"):
                out.append("--" + k.replace("_", "-"))
                continue
            # 短名命中 → 长名
            if k in SHORT_TO_LONG:
                out.append(SHORT_TO_LONG[k])
                out.extend(tail)
                continue
            # 长名（/data → --data，/wd-vram-alloc → --wd-vram-alloc）
            out.append("--" + k.replace("_", "-"))
            out.extend(tail)
            continue

        # 普通值
        out.append(a)
    return out


def build_parser():
    p = argparse.ArgumentParser(
        prog="train_cli.py",
        description="LoRA 训练启动器（参数齐全，自动恢复，自动开进度页）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python train_cli.py
  python train_cli.py -ce 3 -ck 5              # 还原点：每3步存、留5个
  python train_cli.py -d my.jsonl -o out-v120 -e 1
  python train_cli.py -br chrome -p 8090       # 换 Chrome + 换端口
  python train_cli.py --no-open-page --no-detached
  python train_cli.py -pa out-v113 -e 1 -lr 3e-5   # 续训
  python train_cli.py -wa 6.5 -wf 0.3 -wt 5    # 看门狗放宽

DOS 风格（Windows 习惯，/ 开头也能用）：
  python train_cli.py /help          # 显示帮助（同 --help）
  python train_cli.py /?             # 同上
  python train_cli.py /ce 3 /ck 5    # 还原点：每3步存、留5个
  python train_cli.py /br chrome     # 换 Chrome
  python train_cli.py /wa 6.5 /wt 5  # 看门狗放宽
        """)

    g = p.add_argument_group("数据")
    g.add_argument("-d", "--data", default=os.path.join(DATA_DIR, "train_v113_train.jsonl"),
                   help="训练数据 jsonl（默认 data-v1.5/train_v113_train.jsonl）")
    g.add_argument("-c", "--cache", default=None,
                   help="cache pkl（默认按 data 名推导：train_X.jsonl → cache_X.pkl）")
    g.add_argument("--eval-cache", default=None,
                   help="验证集 cache（默认用脚本内置的）")
    g.add_argument("-ml", "--maxlen", type=int, default=1536,
                   help="最大序列长度（默认 1536）")

    g = p.add_argument_group("训练")
    g.add_argument("-e", "--epochs", type=int, default=2, help="epoch 数（默认 2）")
    g.add_argument("-lr", "--lr", type=float, default=5e-5, help="学习率（默认 5e-5）")
    g.add_argument("-r", "--r", type=int, default=16, help="LoRA rank（默认 16）")
    g.add_argument("-a", "--alpha", type=int, default=32, help="LoRA alpha（默认 32）")
    g.add_argument("-b", "--batch", type=int, default=1, help="每步样本数（默认 1）")
    g.add_argument("-ac", "--accum", type=int, default=8, help="梯度累积（默认 8）")
    g.add_argument("-cc", "--ce-chunk", type=int, default=512, help="CE 分块（默认 512）")

    g = p.add_argument_group("还原点（checkpoint）")
    g.add_argument("-ce", "--ckpt-every", type=int, default=5,
                   help="每 N 步存一次（默认 5；崩溃最多丢 N 步）")
    g.add_argument("-ck", "--ckpt-keep", type=int, default=3,
                   help="保留几个还原点（默认 3；超出滚动删最旧；0=全留）")
    g.add_argument("--ckpt-dir", default=None,
                   help="还原点目录（默认 = --out 目录）")
    g.add_argument("-o", "--out", default=os.path.join(HERE, "out-v113"),
                   help="adapter 输出目录（默认 out-v113）")

    g = p.add_argument_group("评估")
    g.add_argument("-ee", "--eval-every", type=int, default=100,
                   help="每 N 步评估（默认 100；0=不评估）")

    g = p.add_argument_group("恢复与续训")
    g.add_argument("--resume", dest="resume", action="store_true", default=True,
                   help="自动从 ckpt 恢复（默认开）")
    g.add_argument("--no-resume", dest="resume", action="store_false",
                   help="不自动恢复，从头训")
    g.add_argument("-pa", "--preload-adapter", default=None,
                   help="预加载 adapter 目录（在已有 adapter 上续训）")
    g.add_argument("-mx", "--max-attempts", type=int, default=5,
                   help="自动恢复最多几次（默认 5；超过自毁）")
    g.add_argument("--self-destruct", dest="self_destruct", action="store_true", default=True,
                   help="完成后删注册表 Run 项（默认开）")
    g.add_argument("--no-self-destruct", dest="self_destruct", action="store_false",
                   help="完成后保留注册表 Run 项")

    g = p.add_argument_group("启动")
    g.add_argument("--detached", dest="detached", action="store_true", default=True,
                   help="用 DETACHED_PROCESS 启动（默认开；父进程退出不影响）")
    g.add_argument("--no-detached", dest="detached", action="store_false",
                   help="前台启动（能看到实时输出）")
    g.add_argument("-gw", "--gpu-warmup", type=int, default=3,
                   help="开训前 GPU 压力测试秒数（默认 3；0=跳过）")

    g = p.add_argument_group("进度页")
    g.add_argument("-p", "--port", type=int, default=8088, help="进度页端口（默认 8088）")
    g.add_argument("--open-page", dest="open_page", action="store_true", default=True,
                   help="启动后自动打开进度页（默认开）")
    g.add_argument("--no-open-page", dest="open_page", action="store_false",
                   help="不自动打开进度页")
    g.add_argument("-br", "--browser", choices=["edge", "chrome", "none"], default="edge",
                   help="用哪个浏览器开进度页（默认 edge）")

    g = p.add_argument_group("看门狗（Watchdog）")
    g.add_argument("-wa", "--wd-vram-alloc", type=float, default=6.2,
                   help="① torch 张量阈值 GB（默认 6.2）")
    g.add_argument("-wf", "--wd-vram-free", type=float, default=0.25,
                   help="① 驱动报的剩余阈值 GB（默认 0.25）")
    g.add_argument("-wt", "--wd-vram-times", type=int, default=3,
                   help="① 连续触发几次才 abort（默认 3）")
    g.add_argument("-mw", "--wd-mem-warn", type=float, default=1.5,
                   help="② 内存只剩 N GB 时警告（默认 1.5，只警告不 abort）")
    g.add_argument("-lm", "--wd-loss-max", type=float, default=20.0,
                   help="③ loss 绝对值阈值（默认 20）")
    g.add_argument("-lq", "--wd-loss-ratio", type=float, default=3.0,
                   help="③ 比前 5 步均值高几倍（默认 3）")
    g.add_argument("-lt", "--wd-loss-times", type=int, default=2,
                   help="③ 连续触发几次才 abort（默认 2）")
    g.add_argument("-ss", "--wd-slow-sec", type=float, default=60.0,
                   help="④ 单步超过 N 秒算慢（默认 60）")
    g.add_argument("-st", "--wd-slow-times", type=int, default=3,
                   help="④ 连续几次慢才 abort（默认 3）")
    g.add_argument("-vl", "--vram-limit", type=float, default=6.6,
                   help="⑤ torch 张量硬上限 GB（默认 6.6）")
    g.add_argument("-ma", "--mem-abort", type=float, default=93.0,
                   help="⑤ 整机内存百分比上限（默认 93）")

    g = p.add_argument_group("日志")
    g.add_argument("--log", default=None,
                   help="训练日志路径（默认 run-<timestamp>.log）")
    g.add_argument("--quiet", action="store_true", help="少打印")

    return p


def log(msg, quiet=False):
    if not quiet:
        print(msg)


def find_browser(name):
    if name == "none":
        return None
    for cand in BROWSERS.get(name, []):
        if os.path.isfile(cand):
            return cand
    return None


def gpu_warmup(seconds, quiet=False):
    if seconds <= 0:
        return
    code = (
        "import torch,time\n"
        "a=torch.randn(1024,1024,device='cuda',dtype=torch.float16)\n"
        "b=torch.randn(1024,1024,device='cuda',dtype=torch.float16)\n"
        "torch.cuda.synchronize()\n"
        "t0=time.time();n=0\n"
        "while time.time()-t0<%d:\n"
        "    c=torch.matmul(a,b);torch.cuda.synchronize();n+=1\n"
        "print('warmup %.1fs %d matmul'%%(time.time()-t0,n))\n"
        "del a,b,c;torch.cuda.empty_cache()\n" % seconds
    )
    tmp = os.path.join(HERE, "_warmup.py")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(code)
    try:
        r = subprocess.run([sys.executable, "-u", tmp],
                           capture_output=True, text=True, timeout=60,
                           encoding="utf-8", errors="replace")
        if not quiet and r.stdout.strip():
            log("  [GPU] " + r.stdout.strip())
    except Exception as e:
        log("  [GPU] warmup failed: %r" % (e,), quiet)
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass


def start_progress_server(port, quiet=False):
    """启动进度页（DETACHED），返回是否成功"""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object { $_.CommandLine -match 'progress_server' } | "
             "Measure-Object | Select-Object -ExpandProperty Count"],
            capture_output=True, text=True, timeout=20,
            encoding="utf-8", errors="replace")
        if out.stdout.strip().isdigit() and int(out.stdout.strip()) > 0:
            log("  [进度页] 已在跑", quiet)
            return True
    except Exception:
        pass

    # 用 launcher 起（DETACHED）
    logpath = os.path.join(HERE, "srv-progress.log")
    try:
        subprocess.run([sys.executable, "-u", LAUNCHER, PROGRESS, logpath],
                       capture_output=True, timeout=60,
                       cwd=os.path.dirname(LAUNCHER))
        time.sleep(4)
        log("  [进度页] 已启动 http://127.0.0.1:%d" % port, quiet)
        return True
    except Exception as e:
        log("  [进度页] 启动失败: %r" % (e,), quiet)
        return False


def open_page(browser_name, port, quiet=False):
    exe = find_browser(browser_name)
    if not exe:
        log("  [浏览器] 找不到 %s，跳过" % browser_name, quiet)
        return
    url = "http://127.0.0.1:%d/?t=%s" % (port, time.strftime("%H%M%S"))
    try:
        subprocess.Popen([exe, "--new-window", url],
                         creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP)
        log("  [浏览器] 已用 %s 打开 %s" % (browser_name, url), quiet)
    except Exception as e:
        log("  [浏览器] 打开失败: %r" % (e,), quiet)


def write_config(args):
    os.makedirs(CONFIG_DIR, exist_ok=True)
    cfgs = {
        "data": args.data,
        "cache": args.cache,
        "eval_cache": args.eval_cache,
        "maxlen": args.maxlen,
        "epochs": args.epochs,
        "lr": args.lr,
        "r": args.r,
        "alpha": args.alpha,
        "batch": args.batch,
        "accum": args.accum,
        "ce_chunk": args.ce_chunk,
        "ckpt_every": args.ckpt_every,
        "ckpt_keep": args.ckpt_keep,
        "ckpt_dir": args.ckpt_dir,
        "out": args.out,
        "eval_every": args.eval_every,
        "resume": args.resume,
        "preload_adapter": args.preload_adapter,
        "wd_vram_alloc": args.wd_vram_alloc,
        "wd_vram_free": args.wd_vram_free,
        "wd_vram_times": args.wd_vram_times,
        "wd_mem_warn": args.wd_mem_warn,
        "wd_loss_max": args.wd_loss_max,
        "wd_loss_ratio": args.wd_loss_ratio,
        "wd_loss_times": args.wd_loss_times,
        "wd_slow_sec": args.wd_slow_sec,
        "wd_slow_times": args.wd_slow_times,
        "vram_limit": args.vram_limit,
        "mem_abort": args.mem_abort,
    }
    path = os.path.join(CONFIG_DIR, "_active.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfgs, f, ensure_ascii=False, indent=2)
    return path


def setup_registry(script_path, quiet=False):
    """写注册表 Run（自动恢复）"""
    try:
        import winreg
        k = winreg.OpenKey(winreg.HKEY_CURRENT_USER, REG_KEY, 0, winreg.KEY_SET_VALUE)
        cmd = '"%s" -u "%s"' % (sys.executable, script_path)
        winreg.SetValueEx(k, REG_NAME, 0, winreg.REG_SZ, cmd)
        winreg.CloseKey(k)
        log("  [自动恢复] 已写注册表 Run 项", quiet)
        return True
    except Exception as e:
        log("  [自动恢复] 写注册表失败: %r" % (e,), quiet)
        return False


def start_training(args, logpath):
    if args.detached:
        import json as _json
        # 让 auto_resume 也指向这个脚本
        r = subprocess.run([sys.executable, "-u", LAUNCHER, TRAINER, logpath],
                           capture_output=True, text=True, timeout=120,
                           encoding="utf-8", errors="replace",
                           cwd=os.path.dirname(LAUNCHER))
        if r.stdout.strip():
            for line in r.stdout.strip().split("\n"):
                log("  " + line, args.quiet)
        return True
    else:
        r = subprocess.run([sys.executable, "-u", TRAINER],
                           cwd=os.path.dirname(TRAINER))
        return r.returncode == 0


def main():
    p = build_parser()
    args = p.parse_args(normalize_argv(sys.argv))

    if not os.path.isfile(TRAINER):
        log("找不到训练脚本: %s" % TRAINER)
        sys.exit(1)

    ts = time.strftime("%Y%m%d_%H%M%S")
    logpath = args.log or os.path.join(HERE, "run-%s.log" % ts)

    log("=" * 62)
    log("  LoRA 训练启动器")
    log("=" * 62)
    log("  数据:      %s" % os.path.basename(args.data))
    log("  输出:      %s" % os.path.basename(args.out))
    log("  还原点:    每 %d 步 / 留 %d 个 / 目录 %s" % (
        args.ckpt_every, args.ckpt_keep, args.ckpt_dir or "(同输出)"))
    log("  评估:      每 %d 步" % args.eval_every if args.eval_every else "  评估:      关闭")
    log("  maxlen:    %d   epochs: %d   lr: %g   r/alpha: %d/%d" % (
        args.maxlen, args.epochs, args.lr, args.r, args.alpha))
    log("  恢复:      %s（最多 %d 次）" % ("开" if args.resume else "关", args.max_attempts))
    log("  看门狗:    VRAM %.1f/%.2f GB x%d | MEM警告 %.1f GB | loss %.0fx%.1f x%d | 慢 %.0fs x%d" % (
        args.wd_vram_alloc, args.wd_vram_free, args.wd_vram_times,
        args.wd_mem_warn, args.wd_loss_max, args.wd_loss_ratio, args.wd_loss_times,
        args.wd_slow_sec, args.wd_slow_times))
    log("  启动:      %s" % ("DETACHED" if args.detached else "前台"))
    log("  进度页:    %s" % ("http://127.0.0.1:%d" % args.port if args.open_page else "不开"))
    log("  日志:      %s" % logpath)
    log("")

    # 1. GPU 预热
    if args.gpu_warmup > 0:
        log("  [1/4] GPU 预热 %d 秒..." % args.gpu_warmup)
        gpu_warmup(args.gpu_warmup, args.quiet)

    # 2. 写配置
    cfg = write_config(args)
    log("  [2/4] 配置写入 %s" % os.path.basename(cfg), args.quiet)

    # 3. 进度页
    if args.open_page:
        log("  [3/4] 启动进度页...")
        start_progress_server(args.port, args.quiet)

    # 4. 启动训练
    log("  [4/4] 启动训练...")
    ok = start_training(args, logpath)

    # 5. 自动打开进度页
    if args.open_page and args.browser != "none":
        time.sleep(3)
        open_page(args.browser, args.port, args.quiet)

    # 6. 注册表自动恢复
    if args.resume:
        setup_registry(os.path.join(HERE, "auto_resume.py"), args.quiet)

    log("")
    log("=" * 62)
    if ok:
        log("  已启动。看进度: http://127.0.0.1:%d" % args.port)
        log("  停止: taskkill /F /IM python.exe（会杀掉所有 python）")
        log("  或:   找 PID 后 taskkill /F /PID <pid>")
    else:
        log("  启动失败，看日志: %s" % logpath)
    log("=" * 62)


if __name__ == "__main__":
    main()
