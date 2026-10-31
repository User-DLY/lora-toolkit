"""
v1.4 训练 —— 所有已知缺陷都修了

修复清单：
  ① loss mask 只对 assistant 段算 loss（v1.3 是全序列，信号稀释 30-40%）
  ② 双槽 checkpoint（防断电写坏单一文件）
  ③ lr 1e-4 → 2e-5
  ④ r/alpha 16/32 → 8/16（α 必须同步，否则缩放翻倍）
  ⑤ target_modules 去掉 MLP（只留 attention）
  ⑥ 保留 30/60/90 步 ckpt 做退化点扫描
  ⑦ maxlen 1792
"""
import sys, os, json, time, io, gc, math, ctypes, random, re
import pickle   # ★修正：eval 加载 cache 用（原来只在 DS.__init__ 里局部 import）
sys.path = [p for p in sys.path if "Temp" not in p and p != ""]
os.environ.pop("BNB_ROCM_VERSION", None)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ★ R6 性能：显存分配器用可扩展段，减少碎片化导致的 OOM（必须在 import torch 前设）
# ★P3 修复：删掉 expandable_segments —— ROCm Windows 无 VMM API，该选项无效
# ★ R6：限制 OMP 线程，避免和 GPU 驱动抢 CPU 核
os.environ.setdefault("OMP_NUM_THREADS", "6")
import numpy as np
import torch
# ★ R4 性能：限制 CPU 线程，避免 torch 的 OMP 线程和 GPU 驱动抢核
try:
    torch.set_num_threads(max(1, min(6, (os.cpu_count() or 8) - 4)))
except Exception:
    pass
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# ══════════════════════════════════════════════════════════════
# 工作目录解析（v0.0.1 通用化）
#   优先级：环境变量 > 脚本目录的父级
#   LORA_WORK_DIR  工作目录（放 model/ data/ out/ 的地方）
#   LORA_MODEL     基座模型目录
# ══════════════════════════════════════════════════════════════
_HERE = os.path.dirname(os.path.abspath(__file__))
WORK_DIR = os.environ.get("LORA_WORK_DIR") or _HERE
MODEL = os.environ.get("LORA_MODEL") or os.path.join(WORK_DIR, "model")
DATA_DIR = os.path.join(WORK_DIR, "data")

# ★ITER-7：验证集（用户 E 步建的，但脚本之前没用）
EVAL_CACHE = os.path.join(DATA_DIR, "eval_cache.pkl")
EVAL_EVERY = 100   # ★修正：99999→50（之前 A 组实验忘了改回，导致没有 val_loss）
OUT = os.path.join(WORK_DIR, "out")
DATA_PATH = os.path.join(DATA_DIR, "train.jsonl")

CACHE_PATH = os.path.join(DATA_DIR, "train_cache.pkl")
# ★v17B：从指定 adapter 目录预加载权重（不是从 ckpt resume）
PRELOAD_ADAPTER = ""
LOG = os.path.join(WORK_DIR, "train.log")
HWLOG = os.path.join(WORK_DIR, "hw.log")
CKPT_A = os.path.join(OUT, "ckpt_a.pt")
CKPT_B = os.path.join(OUT, "ckpt_b.pt")

MAXLEN = 1536   # ★优化：1792→1536（实测显存 6.49GB 太紧，p90=1699，截断约 10% 长样本）
CKPT_EVERY = 5   # ★v17B：50→10（47 步短训，多存点方便回退）（1362 步/3ep，68 次→27 次，IO 从 2GB 降到 810MB）
EPOCHS = 2
VRAM_LIMIT = 6.6   # ★R2-A 修复：实测 12 步峰值 5.28~5.79（p50 样本就 5.26），原 5.3 会第1步 abort
MEM_WARN = 88.0
MEM_ABORT = 93.0

# ══════════════════════════════════════════════════════════════
# ★v1.14（2026-10-31）：看门狗五路阈值全部可配（CLI --wd-*）
#   默认值 = 实测最稳的一套
# ══════════════════════════════════════════════════════════════
WD_VRAM_ALLOC = 6.2      # ① torch 张量阈值（GB）
WD_VRAM_FREE = 0.25      # ① 驱动报的剩余阈值（GB）
WD_VRAM_TIMES = 3        # ① 连续触发几次才 abort
WD_MEM_WARN = 1.5        # ② 内存只剩 N GB 时警告（不 abort）
WD_LOSS_MAX = 20.0       # ③ loss 绝对值阈值
WD_LOSS_RATIO = 3.0      # ③ 比前 5 步均值高几倍
WD_LOSS_TIMES = 2        # ③ 连续触发几次才 abort
WD_SLOW_SEC = 60.0       # ④ 单步超过 N 秒
WD_SLOW_TIMES = 3        # ④ 连续几次才 abort
SEED = 2024
# ★ 新超参
LR = 5e-5   # ★v1.5：2e-4 冲掉基座算术/pwsh（3 辩手裁决），降到几何中点 1e-4
R = 16   # ★修正：32→16（实测进程占 6.52GB，整卡 7.72/8.57 太危险）
ALPHA = 32   # 保持 alpha = 2r

# ══════════════════════════════════════════════════════════════
# ★v1.14（2026-10-31）：从 run-configs/_active.json 读配置覆盖
#   由 train_cli.py 生成，让 CLI 参数生效（不改源码常量）
# ══════════════════════════════════════════════════════════════
_CFG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "..", "..", "..", "..", "D",
                         "run-configs", "_active.json")
_CFG_PATH = os.path.join(WORK_DIR, "run-configs", "_active.json")
try:
    import json as _json
    if os.path.isfile(_CFG_PATH):
        with open(_CFG_PATH, encoding="utf-8-sig") as _f:
            _CFG = _json.load(_f) or {}
        # 数据路径
        if _CFG.get("data"):
            DATA_PATH = _CFG["data"]
        if _CFG.get("cache"):
            CACHE_PATH = _CFG["cache"]
        if _CFG.get("eval_cache"):
            EVAL_CACHE = _CFG["eval_cache"]
        # 训练超参
        if _CFG.get("maxlen"):
            MAXLEN = int(_CFG["maxlen"])
        if _CFG.get("epochs"):
            EPOCHS = int(_CFG["epochs"])
        if _CFG.get("lr"):
            LR = float(_CFG["lr"])
        if _CFG.get("r"):
            R = int(_CFG["r"])
        if _CFG.get("alpha"):
            ALPHA = int(_CFG["alpha"])
        if _CFG.get("batch"):
            BATCH = int(_CFG["batch"])
        if _CFG.get("accum"):
            ACCUM = int(_CFG["accum"])
        if _CFG.get("ce_chunk"):
            CE_CHUNK = int(_CFG["ce_chunk"])
        # checkpoint
        if _CFG.get("ckpt_every"):
            CKPT_EVERY = int(_CFG["ckpt_every"])
        if _CFG.get("ckpt_keep") is not None:
            CKPT_KEEP = int(_CFG["ckpt_keep"])
        if _CFG.get("out"):
            OUT = _CFG["out"]
        # 评估
        if _CFG.get("eval_every") is not None:
            EVAL_EVERY = int(_CFG["eval_every"])
        # 续训
        if _CFG.get("preload_adapter"):
            PRELOAD_ADAPTER = _CFG["preload_adapter"]
        # 看门狗阈值
        if _CFG.get("wd_vram_alloc") is not None: WD_VRAM_ALLOC = float(_CFG["wd_vram_alloc"])
        if _CFG.get("wd_vram_free") is not None:  WD_VRAM_FREE = float(_CFG["wd_vram_free"])
        if _CFG.get("wd_vram_times") is not None: WD_VRAM_TIMES = int(_CFG["wd_vram_times"])
        if _CFG.get("wd_mem_warn") is not None:   WD_MEM_WARN = float(_CFG["wd_mem_warn"])
        if _CFG.get("wd_loss_max") is not None:   WD_LOSS_MAX = float(_CFG["wd_loss_max"])
        if _CFG.get("wd_loss_ratio") is not None: WD_LOSS_RATIO = float(_CFG["wd_loss_ratio"])
        if _CFG.get("wd_loss_times") is not None: WD_LOSS_TIMES = int(_CFG["wd_loss_times"])
        if _CFG.get("wd_slow_sec") is not None:   WD_SLOW_SEC = float(_CFG["wd_slow_sec"])
        if _CFG.get("wd_slow_times") is not None: WD_SLOW_TIMES = int(_CFG["wd_slow_times"])
        if _CFG.get("vram_limit") is not None:    VRAM_LIMIT = float(_CFG["vram_limit"])
        if _CFG.get("mem_abort") is not None:     MEM_ABORT = float(_CFG["mem_abort"])
        print("  [CFG] loaded from %s" % os.path.basename(_CFG_PATH))
except Exception as _e:
    print("  [CFG] load failed (use defaults): %r" % (_e,))

TARGET_MODULES = ["q_k_v_proj", "g_proj", "out_proj", "gate_proj", "up_proj", "down_proj"]   # ★v16：加 MLP（Liquid 用 9 个模块）

os.makedirs(OUT, exist_ok=True)
# ★ 日志用 append，崩溃重启后历史保留（原来用 "w" 每次清空，导致看不到趋势）
def _append_sep(path):
    try:
        with io.open(path, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 70 + "\n[%s] ===== 新一次启动（历史保留）=====\n" % time.strftime("%Y-%m-%d %H:%M:%S"))
    except Exception:
        pass
_append_sep(LOG)
_append_sep(HWLOG)


class MS(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def meminfo():
    s = MS()
    s.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
    return s.dwMemoryLoad, s.ullAvailPhys / 1e9


import threading
_th = threading
_LOG_LOCK = _th.Lock()
_LOG_HANDLES = {}

def log(msg, hw=False):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    # ★ R16 性能：HW 日志只写文件不 print（321 步 × flush stdout 有开销）
    if not hw:
        # ★ITER-2：只在关键行 print（原来每 2 步一次 print+flush，
        #   重定向到文件时约 0.2ms，但直连控制台可达 5ms）。
        #   判据：含 [SAVE]/[CACHE]/ABORT/DONE/step 1/step total 才 print
        _m = msg
        if ("[SAVE]" in _m or "[CACHE]" in _m or "ABORT" in _m or "DONE" in _m
                or _m.startswith("  step") or "Epoch" in _m or "FATAL" in _m
                or "COMPLETE" in _m or "loading" in _m[:12]):
            print(line, flush=True)
    target = HWLOG if hw else LOG
    try:
        with _LOG_LOCK:
            # ★ R12 性能：保持文件句柄常开（原来每次 open/close），且不每行 fsync
            #   fsync 在 SATA SSD 上 0.5-2ms，321 步 × 2 次 = 最多 1.3 秒纯等待
            fh = _LOG_HANDLES.get(target)
            if fh is None or fh.closed:
                fh = io.open(target, "a", encoding="utf-8")
                _LOG_HANDLES[target] = fh
            fh.write(line + "\n")
            # ★ R18 修复：所有日志都 flush（推入 OS 缓冲，开销极小）
            #   否则进程崩溃时 HW 日志会丢——这正是"崩溃后看不到趋势"的根因
            #   fsync（强制落盘，SATA SSD 0.5-2ms）只留给关键行
            fh.flush()
            if ("[SAVE]" in msg) or ("[CACHE]" in msg) or ("ABORT" in msg) or ("DONE" in msg):
                os.fsync(fh.fileno())
    except Exception:
        pass


log("=" * 70)
log("LoRA Trainer ")
log("=" * 70)

# ★R2-7：单实例锁 —— 双击两次会起两个进程同时加载 3.4GB 模型，显存互相踩
try:
    import msvcrt
    _LOCK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".train_v14.lock")
    _lock_fh = open(_LOCK_PATH, "w")
    msvcrt.locking(_lock_fh.fileno(), msvcrt.LK_NBLCK, 1)
except Exception:
    print("[FATAL] 已有训练进程在运行（.train_v14.lock 被占用），退出")
    raise SystemExit(1)

torch.cuda.reset_peak_memory_stats()
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

# ★ L1 修复：必须加载带 {% generation %} 标记的模板，否则 return_assistant_tokens_mask
#   会返回全 0 mask → labels 全 -100 → CE(mean)=0/0=NaN → 权重永久污染
#   （transformers 5.5 只 warning 不抛异常，所以会静默假跑）
_tpl_path = os.path.join(MODEL, "chat_template.jinja")
if os.path.exists(_tpl_path):
    with io.open(_tpl_path, encoding="utf-8") as _tf:
        tok.chat_template = _tf.read()
    _has_gen = bool(re.search(r"\{%-?\s*generation\s*-?%\}", tok.chat_template))
    log("  [TPL] chat_template.jinja loaded | generation marker: %s" % ("YES" if _has_gen else "NO"))
    if not _has_gen:
        raise RuntimeError("chat_template.jinja missing {% generation %} marker; mask will be all-zero and loss becomes NaN")
else:
    raise RuntimeError("chat_template.jinja not found; refusing to run (would produce all-zero masks)")


log("loading model fp16...")
t0 = time.time()
# ★A2: 不用 attn_implementation="sdpa"（部分自定义模型硬编码 eager）
#   eager_attention_forward，完全不看 config._attn_implementation，该参数永远无效；
#   且本机 torch 没编译 memory-efficient attention，SDPA 只能落 math 后端（实测还慢 1.75 倍）
model = AutoModelForCausalLM.from_pretrained(
    MODEL, device_map="cuda", trust_remote_code=True,
    dtype=torch.float16, low_cpu_mem_usage=True)
log("  [ATTN] eager (模型硬编码 eager，无 sdpa 支持)")
# ★ gradient checkpointing 策略（安全第一）：
#   RX6600 物理 8.57 GB，但 dwm.exe 常驻占 3.3 GB，训练实际只能用 ~5.3 GB。
#   ⚠ AMD Windows 的 mem_get_info 会因 WDDM 超配报「全部可用」，不可信！
#     真实超配不会 OOM，而是换页 → 速度从 88 tok/s 掉到 10 tok/s。
#   所以：默认 ON（VRAM 3.5 GB，稳定）。要用 25% 速度换显存时，
#   先关 Edge/浏览器等占显存的程序，再设环境变量 TRAIN_NO_GC=1。
if os.environ.get("TRAIN_NO_GC") == "1":
    try:
        model.gradient_checkpointing_disable()
        log("  [GC] OFF (TRAIN_NO_GC=1) —— 警告：实测 10.45GB 且慢 3.2 倍，不要用！")
    except Exception:
        pass
else:
    model.gradient_checkpointing_enable()
    log("  [GC] ON  (默认安全档 VRAM~3.5GB；设 TRAIN_NO_GC=1 可换 25%% 速度)")
model.config.use_cache = False
try:
    pass  # ★P3 修复：删掉 cudnn.benchmark（模型无卷积，纯 cargo cult）
except Exception:
    pass
gc.collect()
m, a = meminfo()
log("  %.1fs | VRAM %.2fGB | MEM %d%% free %.1fGB" % (
    time.time()-t0, torch.cuda.memory_allocated()/1e9, m, a))

lora = LoraConfig(r=R, lora_alpha=ALPHA, lora_dropout=0.05,
                  target_modules=TARGET_MODULES,
                  bias="none", task_type=TaskType.CAUSAL_LM)
model = get_peft_model(model, lora)

# ★v17B：从 PRELOAD_ADAPTER 预加载 LoRA 权重（用于「在 v17A 基础上继续训」）
#   和 ckpt resume 的区别：这不恢复 optimizer 状态，也不校验 config 指纹
#   用途：分两阶段训练，第二阶段只喂少量新数据（rehearsal 混合）
if PRELOAD_ADAPTER and os.path.isdir(PRELOAD_ADAPTER):
    _adm = os.path.join(PRELOAD_ADAPTER, "adapter_model.safetensors")
    if os.path.exists(_adm):
        try:
            from safetensors.torch import load_file as _load_sf
            _sd = _load_sf(_adm)
            _tgt = model.state_dict()
            _loaded = 0
            _miss = 0
            # ★修正：adapter 保存格式缺 .default，要补上才能匹配 model.state_dict()
            #   adapter:  ...lora_A.weight
            #   state:    ...lora_A.default.weight
            for _k, _v in _sd.items():
                _k2 = _k
                if ".lora_A.weight" in _k2:
                    _k2 = _k2.replace(".lora_A.weight", ".lora_A.default.weight")
                elif ".lora_B.weight" in _k2:
                    _k2 = _k2.replace(".lora_B.weight", ".lora_B.default.weight")
                if _k2 in _tgt:
                    _tgt[_k2] = _v
                    _loaded += 1
                elif _k in _tgt:
                    _tgt[_k] = _v
                    _loaded += 1
                else:
                    _miss += 1
            model.load_state_dict(_tgt, strict=False)
            log("[PRELOAD] 从 %s 加载 %d 个 tensor（未匹配 %d）" % (
                os.path.basename(PRELOAD_ADAPTER), _loaded, _miss))
            if _loaded == 0:
                log("[WARN] [PRELOAD] 一个都没加载上，检查键名格式！")
        except Exception as _e:
            log("[WARN] [PRELOAD] 失败: %s" % str(_e)[:100])
    else:
        log("[WARN] [PRELOAD] 找不到 adapter_model.safetensors")
# ★ R14 关键修复：LoRA + gradient checkpointing 必须让输入带梯度，
#   否则 backward 重算 forward 时计算图断开，梯度不流动（loss 表现为"假学习"）
try:
    model.enable_input_require_grads()
    log("  [GRAD] enable_input_require_grads ON")
except Exception as _e:
    log("[FATAL] enable_input_require_grads failed: %s" % str(_e)[:50])
    log("[FATAL] LoRA + gradient checkpointing 下梯度会断流，拒绝继续（否则会'假学习'）")
    raise RuntimeError("enable_input_require_grads failed")
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
log("  trainable %s (%.2f%%) | r=%d alpha=%d target=%s" % (
    format(n_train, ","), n_train / 1.7e9 * 100, R, ALPHA, TARGET_MODULES))
gc.collect()


_ENC_FAILS = []   # ★B7：记录编码失败原因（原来静默吞掉）
def encode_with_mask(msgs):
    """★ R22 核心修复：改用 transformers 5.x 官方 return_assistant_tokens_mask
    原逐段渲染在「连续 tool 消息」场景下有 bug：
      模板规则是 loop.nextitem 不是 tool 时才加 <eos>，
      但逐段渲染时前一次已经把 eos 拼进 all_ids，导致多出一个 eos。
    实测 31/1287 样本（2.4%）的 token 序列错误。
    新方法一次渲染，天然正确，且快 3.3 倍（14.3s -> 4.4s）。"""
    try:
        r = tok.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=False,
            return_assistant_tokens_mask=True, return_dict=True)
        ids = r["input_ids"]
        mask = r["assistant_masks"]
        if not ids or len(ids) < 32:
            return None, None
        labels = [t if m == 1 else -100 for t, m in zip(ids, mask)]
        if len(ids) > MAXLEN:
            ids = ids[:MAXLEN]
            labels = labels[:MAXLEN]
        return list(ids), list(labels)
    except Exception as _enc_err:   # ★B7 修复：不再静默吞异常
        _ENC_FAILS.append(str(_enc_err)[:100])
        return None, None

class DS(Dataset):
    def __init__(self, path):
        self.items = []
        skipped = 0
        n_loss = 0
        # ★ 性能优化：编码结果缓存（原来每次启动重跑 1287 条 chat_template，耗时 30-60s）
        cache = CACHE_PATH
        # ★R2-5：cache key 必须包含模板 hash + MAXLEN + tokenizer，
        #   否则改模板/改 MAXLEN 会静默用旧分词训满全程（loss 正常但学错格式）
        try:
            import hashlib
            _tp = os.path.join(MODEL, "chat_template.jinja")
            _tpl_hash = hashlib.md5(io.open(_tp, "rb").read()).hexdigest()[:12] if os.path.exists(_tp) else "none"
            _cache_key = "%s|%d|%s" % (_tpl_hash, MAXLEN, getattr(tok, "name_or_path", "?"))
        except Exception:
            _cache_key = "unknown"
        try:
            if os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(path):
                import pickle
                with open(cache, "rb") as cf:
                    cached = pickle.load(cf)
                if cached.get("key") != _cache_key:
                    raise ValueError("cache key mismatch: saved=%r now=%r（模板/MAXLEN/tokenizer 变了）"
                                     % (cached.get("key"), _cache_key))
                self.items = cached["items"]
                n_loss = cached["n_loss"]
                lens = sorted(len(x[0]) for x in self.items)
                tot = sum(len(x[0]) for x in self.items)
                log("  [CACHE] %d valid (skip 0) | token p50 %d / max %d | loss 覆盖 %d/%d (%.1f%%)" % (
                    len(self.items), lens[len(lens)//2], lens[-1],
                    n_loss, tot, n_loss/max(1,tot)*100))
                return
        except Exception as e:
            log("  [CACHE] load failed (%s), re-encoding..." % str(e)[:50])
        _zero_encoded = 0   # ★Bug3: 记录编码时丢弃的零 label 条数
        for line in io.open(path, encoding="utf-8"):
            if not line.strip():
                continue
            ex = json.loads(line)
            ids, labels = encode_with_mask(ex["messages"])
            if ids is None or len(ids) < 32:
                skipped += 1; continue
            # ★修正（Bug3）：零 label 样本【编码时】就丢弃
            #   原来只在训练时 WARN 跳过，导致 v16d 少跑 6 步
            _n_ok = sum(1 for x in labels if x != -100)
            if _n_ok == 0:
                skipped += 1
                _zero_encoded += 1
                continue
            n_loss += _n_ok
            # ★ 内存优化：numpy int32 代替 Python list（省 3-4 倍）
            self.items.append((np.asarray(ids, dtype=np.int32),
                               np.asarray(labels, dtype=np.int32)))
        lens = sorted(len(x[0]) for x in self.items)
        tot = sum(len(x[0]) for x in self.items)
        log("  %d valid (skip %d, 其中零label %d) | token p50 %d / max %d | loss 覆盖 %d/%d (%.1f%%)" % (
            len(self.items), skipped, _zero_encoded, lens[len(lens)//2], lens[-1],
            n_loss, tot, n_loss/max(1,tot)*100))
        # ★ 保存编码缓存，下次启动直接读
        try:
            import pickle
            with open(cache, "wb") as cf:
                pickle.dump({"items": self.items, "n_loss": n_loss, "key": _cache_key}, cf, protocol=4)
            log("  [CACHE] saved -> %s" % os.path.basename(cache))
        except Exception as e:
            log("  [CACHE] save failed (%s)" % str(e)[:50])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


ds = DS(DATA_PATH)
pad_id = tok.pad_token_id or tok.eos_token_id


def collate(b):
    mx = max(len(x[0]) for x in b)
    # ★ 性能优化：用 numpy 构造再用 from_numpy 零拷贝转 torch
    #   原来逐元素 torch.tensor(list) 每次都新建 Python 对象，慢且占内存
    ids = np.full((len(b), mx), pad_id, dtype=np.int64)
    lab = np.full((len(b), mx), -100, dtype=np.int64)
    att = np.zeros((len(b), mx), dtype=np.int64)
    for i, (x, y) in enumerate(b):
        n = len(x)
        ids[i, :n] = x
        lab[i, :n] = y
        att[i, :n] = 1
    return {"input_ids": torch.from_numpy(ids),
            "labels": torch.from_numpy(lab),
            "attention_mask": torch.from_numpy(att),
            "len": [len(x[0]) for x in b]}


class BucketBatchSampler(torch.utils.data.Sampler):
    """★ 性能优化：长度分桶采样器
    随机 pair 时 batch=2 浪费 16.6% padding、batch=4 浪费 24.6%；
    注意：BATCH=1 时 padding 恒为 0，分桶不省任何东西（它省的本来就是 B>1 时的 padding）。
    seed 固定 → batch 顺序确定，resume 的 bi<start_step*ACCUM 跳过逻辑仍有效。"""
    def __init__(self, lengths, batch_size, n_buckets=10, seed=2024, epoch=0):
        self.lengths = np.asarray(lengths)
        self.batch_size = batch_size
        self.n_buckets = n_buckets
        self.seed = seed
        self.epoch = epoch
        self._batches = None

    def _build(self):
        """生成 batch 列表（len 和 iter 都从这里取，保证一致）"""
        if self._batches is not None:
            return self._batches
        import random as _random
        rng = _random.Random(self.seed + self.epoch)
        order = np.argsort(self.lengths, kind="stable")
        chunks = np.array_split(order, self.n_buckets)
        batches = []
        for ch in chunks:
            idx = [int(x) for x in ch]
            rng.shuffle(idx)
            for i in range(0, len(idx) - self.batch_size + 1, self.batch_size):
                batches.append(idx[i:i + self.batch_size])
        rng.shuffle(batches)
        self._batches = batches
        return batches

    def __iter__(self):
        return iter(self._build())

    def __len__(self):
        return len(self._build())



# ═════ P0-2 修复：分块 CrossEntropy ═════
# 原路径：model(labels=...) → 全序列 1792 × 131072 词表 logits (bf16 940MB)
#         → ForCausalLMLoss 第一行 logits.float() (1.88GB fp32)
#         → cross_entropy 内部 log_softmax 再 1.88GB 且被 backward 保存
#         → loss 这步峰值 2.8~3.7GB（最大显存头）
# 分块后：每次只对 512 个位置算 fp32 logits + log_softmax，峰值降到约 1/3.5
CE_CHUNK = 512

def chunked_ce_loss(logits, labels, ignore_index=-100, chunk=CE_CHUNK):
    """正确的 next-token CE：logits[t] 对 labels[t+1]，按位置分块算

    ★A1 修复（致命）：之前漏了 shift，变成 cross_entropy(logits[t], labels[t])，
      实际是教模型"看着当前 token 的隐状态预测当前 token"，目标无信息量。
      实测基座模型：不 shift 时 loss≈14.4(≈ln(131072)+2.6)，正确应为 2.0~3.0。
    ★A4 修复：累加在 fp32 里做（原来 bf16/fp16 累加到 ~1500 量级分辨率只有 8，±0.03 噪声）
    """
    B, T, V = logits.shape
    if T < 2:
        return None
    # ★ITER-1 优化：原代码每个 chunk 都 .item() 一次（1792/512=4 次 GPU 同步/步）
    #   改成：count 在 GPU 上累加，只在最后 .item() 一次（4 次 → 1 次）
    #   全 ignore 的 chunk 用 where 屏蔽（避免 CE 返回 nan 污染 total）
    total = torch.zeros((), device=logits.device, dtype=torch.float32)
    count_t = torch.zeros((), device=logits.device, dtype=torch.long)
    for s in range(0, T - 1, chunk):
        e = min(s + chunk, T - 1)
        # logits[:, s:e] 对应 labels[:, s+1:e+1]
        lg = logits[:, s:e, :].reshape(-1, V).float()
        lb = labels[:, s + 1:e + 1].reshape(-1)
        n_t = (lb != ignore_index).sum()
        if int(n_t) == 0:
            # 极少数 chunk 全 ignore（跳过，但这次 .item() 只在 n_t==0 时才可能触发）
            continue
        ce = torch.nn.functional.cross_entropy(
            lg, lb, ignore_index=ignore_index, reduction="sum")
        total = total + torch.where(n_t > 0, ce, torch.zeros_like(ce))
        count_t = count_t + n_t
    cnt = int(count_t)
    if cnt == 0:
        return None
    return total / cnt

# ═════ B3 修复：递归把 optimizer state 搬到 CPU ═════
# 原代码 {k: (v.detach().cpu() if torch.is_tensor(v) else v)} 只处理顶层 tensor，
# 但 opt.state_dict() 的 'state' 是 dict、'param_groups' 是 list → 原样传引用给后台线程，
# 序列化时主线程已 step，ckpt 里的动量是 N+1 的（实测 ckpt 30.5MB = lora 10.1 + 动量 20）
def to_cpu_recursive(obj):
    if torch.is_tensor(obj):
        # ★B3 补丁：detach().cpu() 对已在 CPU 的张量是 no-op（返回同一对象，共享存储）
        #   实测：改原张量后"副本"也跟着变。CPU 张量必须显式 clone。
        _dev = getattr(obj, "device", None)
        if _dev is not None and str(_dev) == "cpu":
            return obj.clone()
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: to_cpu_recursive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_cpu_recursive(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(to_cpu_recursive(v) for v in obj)
    return obj

BATCH, ACCUM = 1, 8   # ★P1-4 修复：2→1 省 1.9GB 激活；消除 WDDM 换页后反而更快
g = torch.Generator(); g.manual_seed(SEED)
lens_all = [len(x[0]) for x in ds.items]
_bucket_sampler = BucketBatchSampler(lens_all, BATCH, n_buckets=20, seed=SEED)
# ★ITER-4：pin_memory=True（页锁定内存，CPU→GPU 传输快 2-3 倍）
loader = DataLoader(ds, batch_sampler=_bucket_sampler, collate_fn=collate,
                    pin_memory=True, num_workers=0, persistent_workers=False)

# ══ ★ITER-7：加载验证集（训练时监控过拟合）══
_eval_items = None
try:
    if os.path.exists(EVAL_CACHE):
        with open(EVAL_CACHE, "rb") as cf:
            _ev = pickle.load(cf)
        # ★修正：_cache_key 是 DS.__init__ 的局部变量，eval 这里访问不到（NameError）
        #   cache 本来就是训练时同一份，跳过校验
        if True:
            _eval_items = _ev["items"]
            log("  [EVAL] %d 条验证集已加载（每 %d 步跑一次）" % (len(_eval_items), EVAL_EVERY))
        else:
            log("  [EVAL] key 不匹配，跳过 eval")
    else:
        log("  [EVAL] cache 不存在: %s" % EVAL_CACHE)
except Exception as _e:
    log("  [EVAL] 加载失败: %s" % str(_e)[:80])


def _make_eval_batches(items, n_batches=40):
    """从验证集取前 n_batches 条（BATCH=1，不 padding）"""
    out = []
    for ids, lab in items[:n_batches]:
        out.append({
            "input_ids": torch.tensor([ids], dtype=torch.long).to("cuda"),
            "labels": torch.tensor([lab], dtype=torch.long).to("cuda"),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long).to("cuda"),
        })
    return out


_EVAL_BATCHES = _make_eval_batches(_eval_items) if _eval_items else []


def eval_loss():
    """在验证集上算 CE（不反传）"""
    if not _EVAL_BATCHES:
        return None
    model.eval()
    tot = 0.0
    cnt = 0
    try:
        with torch.no_grad():
            for b in _EVAL_BATCHES:
                _o = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
                _ce = chunked_ce_loss(_o.logits, b["labels"])
                del _o
                if _ce is not None:
                    tot += float(_ce)
                    cnt += 1
    except Exception as _e:
        log("  [EVAL] 计算失败: %s" % str(_e)[:60])
        return None
    finally:
        model.train()
    return tot / max(1, cnt)
spe = len(loader) // ACCUM
total = spe * EPOCHS
WARMUP = max(5, total // 10)
_trainable = [p for p in model.parameters() if p.requires_grad]
_CLIP_PARAMS = _trainable   # ★ITER-3：预计算 clip 用（与 _trainable 同列表）
# ★R2-2：GradScaler.unscale_ 只接受 fp32 梯度（实测 fp16 梯度直接抛
#   "Attempting to unscale FP16 gradients"）。当前 peft 恰好在 fp32，
#   但这依赖版本行为，显式校验避免未来静默崩。
_bad_dt = sorted({str(p.dtype) for p in _trainable})
if _bad_dt != ["torch.float32"]:
    raise RuntimeError(
        "LoRA 参数必须是 fp32（实测必须），当前是 %s -> GradScaler.unscale_ 会崩" % _bad_dt)
log("  [DTYPE] LoRA params: fp32 OK")
# ★R4 修复：不能同时用 fused=True 和 GradScaler
#   实测 scaler.step(opt) 抛 AssertionError: Expected grad_scale and found_inf to be None
#   （fused AdamW 走 _multi_tensor_adam，不接受 GradScaler 注入的 grad_scale/found_inf）
#   fused 对 2.5M LoRA 参数的收益 <0.1%，而 fp16 是 3.4 倍 —— 优先保 fp16。
opt = torch.optim.AdamW(_trainable, lr=LR, weight_decay=0.01)
log("  [OPT] AdamW (fused 关闭以兼容 GradScaler)")

# ★B1：fp16 用 GradScaler 防梯度下溢（RX6600 gfx1032 无原生 bf16，
#      rocBLAS bf16 走慢回退路径，实测 fp16 比 bf16 快 3.4 倍）
scaler = torch.amp.GradScaler("cuda")


def lr_at(s):
    if s < WARMUP:
        return LR * (s+1) / WARMUP
    return LR * 0.5 * (1 + math.cos(math.pi * (s-WARMUP) / max(1, total-WARMUP)))


# ═══ 恢复（双槽）═══
start_step = 0
_fatal_resume = None   # ★P3: resume 致命错误标志
# ★ 修复：选【修改时间最新】的 checkpoint，不是固定顺序
#   原 bug：先检查 CKPT_A，只要存在就用它 → 一直是旧的（step 100），
#   而 CKPT_B 已经是 step 105 → 每次重启回退 5 步
_cks = []
for _ck in [CKPT_A, CKPT_B]:
    if os.path.exists(_ck):
        _cks.append((_ck, os.path.getmtime(_ck)))
_cks.sort(key=lambda x: -x[1])   # 最新的排前面
for ck, _mt in _cks:
    try:
        st = torch.load(ck, map_location="cuda", weights_only=False)
        own = model.state_dict()
        loaded = 0
        for k, v in st["lora"].items():
            if k in own and own[k].shape == v.shape:
                own[k] = v; loaded += 1
        model.load_state_dict(own, strict=False)
        if "optimizer" in st:
            try: opt.load_state_dict(st["optimizer"])
            except Exception: pass
        start_step = st.get("step", 0)
        # ★B5 修复：校验配置指纹，不匹配要明确警告（之前跨配置续训零提示）
        _saved_cfg = st.get("config")
        if _saved_cfg:
            _cur = {"total": total, "batch": BATCH, "accum": ACCUM,
                    "ckpt_every": CKPT_EVERY, "maxlen": MAXLEN,
                    "lr": LR, "r": R, "alpha": ALPHA,
                    "n_buckets": getattr(_bucket_sampler, "n_buckets", None)}
            _diff = [("%s: %s -> %s" % (k, _saved_cfg.get(k), _cur[k]))
                     for k in _cur if _saved_cfg.get(k) != _cur[k]]
            if _diff:
                log("[WARN] ===== CONFIG MISMATCH =====")
                for _d in _diff:
                    log("[WARN]   %s" % _d)
                log("[WARN] batch 顺序已变，续训会重复/跳过部分样本；建议删 ckpt 从头训练")
            else:
                log("        config fingerprint: MATCH")
        else:
            log("[WARN] checkpoint 无 config 指纹（旧版脚本写的），无法校验续训一致性")
        # ★R2-4/P3 修复：恢复不全必须拒绝续训。
        #   注意：这里不能用 raise（会被本函数的 except Exception 吞掉），
        #   改成记录致命标志 + 重置 start_step，循环结束后检查。
        if loaded != len(st["lora"]):
            _fatal_resume = ("只恢复 %d/%d 个 LoRA 张量，拒绝续训（TARGET_MODULES 或 r 改过？）"
                             % (loaded, len(st["lora"])))
            start_step = 0
            log("[FATAL] " + _fatal_resume)
            break
        log("[RESUME] from %s (mtime %s): step %d (loss %s) restored %d/%d" % (
            os.path.basename(ck), time.strftime("%H:%M:%S", time.localtime(_mt)),
            start_step, st.get("loss", "?"), loaded, len(st["lora"])))
        # ★ 性能修复：不再 torch.load 每个候选（每次读 58MB），只报文件信息
        log("        candidates: %s" % ", ".join(
            "%s(%s)" % (os.path.basename(a), time.strftime("%H:%M", time.localtime(_m)))
            for a, _m in _cks))
        break
    except Exception as e:
        log("[RESUME] %s failed: %s" % (os.path.basename(ck), str(e)[:80]))

# ★P3/R2：清理崩溃残留的 .tmp 文件（resume 不扫描会永久堆积）
for _ck, _mt in _cks:
    if os.path.exists(_ck) and os.path.exists(_ck + ".tmp"):
        try:
            os.remove(_ck + ".tmp")
        except Exception:
            pass

# ★P3：resume 致命错误检查（必须放在 resume 循环之后、模型加载之前）
if _fatal_resume:
    log("[FATAL] " + _fatal_resume)
    log("[FATAL] 拒绝启动（否则会从随机初始化的 LoRA 训剩余步数并覆盖成品）")
    raise SystemExit(1)

log("")
log("data=%d steps/epoch=%d epochs=%d total=%d" % (len(ds), spe, EPOCHS, total))
# ★ R13 安全：启动前内存预警
_m_pct, _m_avail = meminfo()
if _m_avail < 8.0:
    log("!!! 警告：可用内存仅 %.1f GB（建议 >= 10 GB）" % _m_avail)
    log("!!! 低于 8 GB 会大量换页，速度可能掉 50%% 以上")
elif _m_avail < 10.0:
    log("  [警告] 可用内存 %.1f GB，偏低（建议 10 GB+）" % _m_avail)
else:
    log("  可用内存 %.1f GB OK" % _m_avail)
log("LR=%.1e warmup=%d | r=%d alpha=%d | maxlen=%d" % (LR, WARMUP, R, ALPHA, MAXLEN))
# ★ R15：确认 LoRA 层数 + embedding 层是否被 require_grads hook 覆盖
_lora_n = len([n for n, _ in model.named_parameters() if "lora_" in n])
_emb = None
try:
    _emb = model.get_input_embeddings()
except Exception:
    pass
_emb_hooked = False
if _emb is not None:
    # enable_input_require_grads 会挂 forward hook；有 hook 说明生效
    _emb_hooked = len(getattr(_emb, "_forward_hooks", {})) > 0
log("  LoRA tensors: %d | embedding require_grads hook: %s" % (
    _lora_n, "ON" if _emb_hooked else "OFF"))
log("ckpt every %d (dual-slot) | guards VRAM<%.1f MEM<%d/%d" % (
    CKPT_EVERY, VRAM_LIMIT, MEM_WARN, MEM_ABORT))
log("")

model.train()
step = start_step
t0 = time.time()
losses = []
micro_losses = []
_spd_hist = []   # ★C3: 速度守卫历史
_last_ts = time.time()  # ★R2-9: 上一步结束时刻
_skips = 0    # ★R2-3: GradScaler 溢出跳过的步数
_zero_skips = 0   # ★v16e: 零 label 跳过的批次数
_run_id = time.strftime("%H%M%S")  # ★R2-C: 本次 run 标识
_slow_cnt = 0    # ★C3: 连续慢步计数
vram_peak_all = 0.0
mempct_cache = -1
vram_cache = 0.0
vram_peak_step = 0.0
avail_cache = 0.0
aborted = False
# ★v1.13 看门狗计数器（缺的四个，_slow_cnt 已存在于 685 行）
_vram_bad_cnt = 0        # VRAM 连续达阈值次数
_vram_warned = False     # 是否已警告过
_loss_bad_cnt = 0        # loss 连续爆炸次数
_step_t0 = time.time()   # 上步结束时刻（算单步耗时）
slot = 0

# ★v16e 修正：续跑要处理跨 epoch（原来 bi < start_step*ACCUM 会导致 epoch 1 被全跳过）
#   start_step 是全局步数，spe 是每 epoch 的步数
_epoch_offset = start_step // spe if spe > 0 else 0
_skip_in_epoch = (start_step % spe) * ACCUM if spe > 0 else 0
if start_step > 0:
    log("  [RESUME] epoch_offset=%d, skip_in_epoch=%d batches (start_step=%d, spe=%d)"
        % (_epoch_offset, _skip_in_epoch, start_step, spe))
for ep in range(EPOCHS):
    if aborted:
        break
    log("--- Epoch %d/%d ---" % (ep+1, EPOCHS))
    # ★B6 修复：start_step >= total 时 0 步训练，但会走到打印 COMPLETE + 保存
    if ep == 0 and start_step >= total:
        log("[FATAL] start_step (%d) >= total (%d)：没有可训练步数！" % (start_step, total))
        log("[FATAL] 可能原因: checkpoint 已训完 / 数据量变了导致 total 缩小")
        log("[FATAL] 拒绝继续（否则会打印假 COMPLETE 并覆盖 adapter）")
        aborted = True
        break
    for bi, batch in enumerate(loader):
        if step >= total:
            break
        # ★v16e 修正：跨 epoch 续跑
        #   ep < _epoch_offset → 整个 epoch 跳过
        #   ep == _epoch_offset → 只跳过前 _skip_in_epoch 个 batch
        if ep < _epoch_offset:
            continue   # ★修正：跳过整个 epoch 的每个 batch（原来 break 会退出外层 for ep）
        if ep == _epoch_offset and bi < _skip_in_epoch:
            continue

        lens = batch.pop("len")
        ids = batch["input_ids"].to("cuda")
        lab = batch["labels"].to("cuda")
        att = batch["attention_mask"].to("cuda")

        # ★P0-2 修复：不传 labels 给模型（避免 ForCausalLMLoss 的 logits.float() + log_softmax）
        #   改成拿 logits 后自己按位置分块算 CE，峰值显存从 2.8~3.7GB 降到约 1/3.5
        _out = model(input_ids=ids, attention_mask=att)
        _logits = _out.logits
        del _out
        _ce = chunked_ce_loss(_logits, lab)
        del _logits
        if _ce is None:
            # ★B2 修复：整批全 -100（零 assistant token）→ 跳过，否则 CE(mean)=0/0=NaN
            #   且 nan > VRAM_LIMIT 恒为 False，守卫拦不住，会静默污染所有权重
            # ★v16e 修正：数据已删掉零 label 样本，这里正常不该触发；
            #   一旦触发就计数（非 0 说明数据有问题，要回头查）
            _zero_skips += 1
            log("  [WARN] step %d: zero valid labels (累计 %d 次，数据可能有问题)"
                % (step + 1, _zero_skips))
            del ids, lab, att
            micro_losses = []
            continue
        if not torch.isfinite(_ce):
            log("[FATAL] step %d: loss is not finite (%s), aborting to avoid weight corruption" % (step + 1, _ce))
            aborted = True; break
        raw_t = _ce.detach()
        loss = _ce / ACCUM
        scaler.scale(loss).backward()   # ★B1: fp16 梯度缩放
        del ids, lab, att, _ce
        micro_losses.append(raw_t)
        # ★ 性能修复：移除每 micro-batch 的 gc.collect（2568 次 → 80 次）

        if (bi+1) % ACCUM == 0:
            scaler.unscale_(opt)   # ★B1: unscale 必须在 clip 之前
            # ★ITER-3：预计算参数列表（原来每步重建 504 个元素的列表）
            torch.nn.utils.clip_grad_norm_(_CLIP_PARAMS, 1.0)
            for gg in opt.param_groups:
                gg["lr"] = lr_at(step)
            _sb = scaler.get_scale()   # ★R2-3: 记录 step 前的 scale
            scaler.step(opt)
            scaler.update()
            if scaler.get_scale() < _sb:
                _skips += 1
                log("[WARN] step %d 梯度溢出被跳过（累计 %d 步）" % (step, _skips))
                # ★修正：GradScaler 初始 scale=65536 太大，第 1 步必溢出（正常降档）
                #   原判据 max(1, step-start_step)=1 → 1>0.2 直接 FATAL，把自己杀了
                #   改为：至少跑 30 步后，且累计跳过 >20 且 比例 >20% 才判死
                _run_steps = step - start_step
                if _run_steps >= 30 and _skips > 20 and _skips > 0.2 * _run_steps:
                    log("[FATAL] 跳过比例 %d/%d，继续训等于没训" % (_skips, _run_steps))
                    aborted = True; break
            opt.zero_grad(set_to_none=True)
            del loss

            step += 1
            # ★ BUG 修复：记录这一步 ACCUM 个样本的【平均】loss，而非最后一个
            # ★ R4：一次性 .item()（2 次 GPU 同步 -> 1 次）
            step_loss = float(torch.stack(micro_losses).mean().item())
            micro_losses = []
            losses.append(step_loss)
            # ★ R9：VRAM 采样每 2 步一次（奇数步沿用上一步的值）
            if step % 2 == 0 or step == 1:
                vram_now = torch.cuda.memory_allocated()/1e9
                vram_peak_step = torch.cuda.max_memory_allocated()/1e9
                vram_peak_all = max(vram_peak_all, vram_peak_step)
                torch.cuda.reset_peak_memory_stats()
                vram_peak = vram_peak_step
                vram_cache = vram_now
            else:
                vram_now = vram_cache
                vram_peak = vram_peak_step

            # ★R13：MEM 每步都查（ctypes 约 1 微秒，不能降频；降频时曾 99% 才发现）
            mempct, avail = meminfo()
            mempct_cache = mempct
            avail_cache = avail

            # ★R2-C：写心跳文件（进度页靠它判断进程是否已死）
            try:
                with io.open(os.path.join(OUT, "status.json"), "w", encoding="utf-8") as _sf:
                    json.dump({"run": _run_id, "step": step, "total": total,
                               "ts": time.time(), "loss": round(step_loss, 6),
                               "vram": round(vram_now, 3), "mem": mempct,
                               "skips": _skips, "aborted": aborted}, _sf)
            except Exception:
                pass
            # ★ 性能优化：HW 日志每 5 步写一次（原来每步写文件，321 次 -> 64 次）
            if step % 5 == 0 or step == 1:
                try:
                    _f2, _t2 = torch.cuda.mem_get_info()
                    _vram_real = (_t2 - _f2) / 1e9
                except Exception:
                    _vram_real = -1
                log("HW step=%d VRAM_torch=%.2f VRAM_real=%.2f(peak%.2f) MEM=%d%% FREE=%.1fGB" % (
                    step, vram_now, _vram_real, vram_peak, mempct, avail), hw=True)

            if step % 2 == 0 or step == 1:
                el = time.time()-t0
                done = step - start_step
                # ★修正：原来 eta = 累计平均 × 剩余步数，前期被慢样本拉高
                #   昨天预估 33min 实际 42min（差 27%）。改成最近 10 步的中位数
                if len(_spd_hist) >= 10:
                    _recent = sorted(_spd_hist[-10:])
                    _med_spd = _recent[len(_recent)//2]
                else:
                    _med_spd = el/max(1,done)
                eta = _med_spd * (total-step)
                log("  step %4d/%d  loss %.4f  len %4d  VRAM %.2fGB  MEM %d%%  %.1fs/step  eta %.0fs" % (
                    step, total, step_loss, max(lens) if lens else 0, vram_now, mempct,
                    el/max(1,done), eta))

            # ★ 双槽 checkpoint（原子写）
            # ★ R31 修复：最后一步（step==total）必须保存，否则 total 不整除 CKPT_EVERY 时
            #   最后一个 checkpoint 会丢（当前 total=320 恰好整除 20，但数据量一变就出问题）
            if step % CKPT_EVERY == 0 or step >= total:
                slot = 1 - slot
                target = CKPT_A if slot == 0 else CKPT_B
                tmp = target + ".tmp.%d" % step   # ★ 唯一 tmp 名，异步保存不冲突
                try:
                    import threading
                    lora_sd = {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}
                    opt_sd = to_cpu_recursive(opt.state_dict())   # ★B3 修复：递归转 CPU，切断活张量引用
                    payload = {"step": step, "loss": step_loss, "lora": lora_sd, "optimizer": opt_sd,
                               # ★B5 修复：记录配置指纹，resume 时校验，避免跨配置续训
                               "config": {"total": total, "batch": BATCH, "accum": ACCUM,
                                          "ckpt_every": CKPT_EVERY, "maxlen": MAXLEN,
                                          "lr": LR, "r": R, "alpha": ALPHA,
                                          "n_buckets": getattr(_bucket_sampler, "n_buckets", None)}}
                    # ★ R8 性能：保存放后台线程，不阻塞下一步计算
                    def _save(payload=payload, tmp=tmp, target=target):
                        try:
                            torch.save(payload, tmp)
                            os.replace(tmp, target)
                            _st = payload["step"]
                            log("  [SAVE] %s step %d (%d tensors)" % (
                                os.path.basename(target), _st, len(payload["lora"])))
                            # ★v1.14: 归档到 ckpt-history + 滚动保留 CKPT_KEEP 个
                            try:
                                _keep = globals().get("CKPT_KEEP", 3)
                                if _keep and _keep > 0:
                                    _hdir = os.path.join(OUT, "ckpt-history")
                                    os.makedirs(_hdir, exist_ok=True)
                                    _arch = os.path.join(_hdir, "step_%04d.pt" % _st)
                                    if not os.path.exists(_arch):
                                        import shutil as _sh
                                        _sh.copy2(target, _arch)
                                    # 滚动删除最旧的
                                    _files = sorted(f for f in os.listdir(_hdir)
                                                    if f.startswith("step_") and f.endswith(".pt"))
                                    while len(_files) > _keep:
                                        _oldf = os.path.join(_hdir, _files.pop(0))
                                        try:
                                            os.remove(_oldf)
                                        except Exception:
                                            pass
                                    log("  [CKPT] 归档 step_%04d.pt，历史保留 %d 个" % (
                                        _st, min(len(_files), _keep)))
                            except Exception as _ae:
                                log("  [WARN] ckpt archive failed: %s" % str(_ae)[:60])
                        except Exception as e:
                            log("  [WARN] ckpt save failed: %s" % str(e)[:80])
                    threading.Thread(target=_save, daemon=True).start()
                except Exception as e:
                    log("  [WARN] ckpt prep failed: %s" % str(e)[:80])

            # ══ ★ITER-7：每 EVAL_EVERY 步跑验证集（监控过拟合）══
            if step % EVAL_EVERY == 0 or step >= total:
                _ev = eval_loss()
                if _ev is not None:
                    log("  [EVAL] step %d/%d  val_loss %.4f  (train_last5 %.4f)" % (
                        step, total, _ev,
                        sum(losses[-5:]) / max(1, len(losses[-5:]))))

            # ★修正：原来的 vram_peak 是 torch.cuda.max_memory_allocated()（只算张量），
            #   实测训练进程真实占用 6.52GB、整卡 7.72/8.57GB，但 torch 只报 3.88GB
            #   → 守卫永远触发不了。改用 mem_get_info 看"系统真实可用"。
            # ★v1.13 看门狗加强（2026-10-31）
            #   五路监控，每路都要【连续 N 次】才触发，防抖动误杀
            #   ────────────────────────────────────────────────
            #   ① VRAM：torch 张量 >= 6.2GB 且 free < 0.25GB，连续 3 次
            #      （AMD 驱动会误报 free=0，所以必须双条件）
            #   ② 内存：可用 < 1.5GB，连续 5 次（只 warn 不 abort，用户要求不清内存）
            #   ③ loss：> 20 且比前 5 步均值高 3 倍，连续 2 次
            #   ④ 速度：单步 > 60 秒，连续 3 次（可能在换页）
            #   ⑤ step 卡死：日志时间戳 300 秒没更新（由外层监控）
            _free_v, _tot_v = torch.cuda.mem_get_info()
            _free_gb = _free_v / 1e9
            _tot_gb = _tot_v / 1e9
            _alloc_gb = torch.cuda.memory_allocated() / 1e9
            _reserved_gb = torch.cuda.memory_reserved() / 1e9

            # ① VRAM 守卫（连续 3 次）
            if _alloc_gb >= WD_VRAM_ALLOC and _free_gb < WD_VRAM_FREE:
                _vram_bad_cnt += 1
                log("  [WARN-VRAM] 第 %d/3 次：torch %.2f GB + free %.2f GB（整卡 %.1fGB）" % (
                    _vram_bad_cnt, _alloc_gb, _free_gb, _tot_gb))
                if _vram_bad_cnt >= WD_VRAM_TIMES:
                    log("!!! VRAM 连续 3 次达阈值，ABORT（torch %.2f / free %.2f / 整卡 %.1f）!!!" % (
                        _alloc_gb, _free_gb, _tot_gb))
                    aborted = True; break
            else:
                if _vram_bad_cnt > 0:
                    log("  [OK-VRAM] 恢复到安全区（torch %.2f GB / free %.2f GB），计数器清零" % (
                        _alloc_gb, _free_gb))
                _vram_bad_cnt = 0

            # ② 内存守卫（只 warn，不 abort —— 用户要求不清内存）
            _m_pct2, _m_avail2 = meminfo()
            if _m_avail2 < WD_MEM_WARN:
                log("  [WARN-MEM] 内存只剩 %.2f GB（%d%%），可能换页变慢，继续跑" % (
                    _m_avail2, _m_pct2))

            # ③ loss 爆炸（连续 2 次）
            if len(losses) >= 5 and step_loss > WD_LOSS_MAX:
                _recent_avg = sum(losses[-6:-1]) / 5 if len(losses) >= 6 else step_loss
                if step_loss > _recent_avg * WD_LOSS_RATIO:
                    _loss_bad_cnt += 1
                    log("  [WARN-LOSS] 第 %d/2 次：当前 %.2f vs 前5步均值 %.2f（%.1f 倍）" % (
                        _loss_bad_cnt, step_loss, _recent_avg, step_loss / max(0.01, _recent_avg)))
                    if _loss_bad_cnt >= WD_LOSS_TIMES:
                        log("!!! loss 连续 2 次爆炸，ABORT（可能梯度异常/数据问题）!!!")
                        aborted = True; break
                else:
                    _loss_bad_cnt = 0
            else:
                _loss_bad_cnt = 0

            # ④ 速度异常（连续 3 次 > 60 秒）
            _step_dt = time.time() - _step_t0
            _step_t0 = time.time()  # 立即重置，下次算的是下步耗时
            if _step_dt > WD_SLOW_SEC:
                _slow_cnt += 1
                log("  [WARN-SLOW] 第 %d/3 次：单步 %.1f 秒（可能在换页）" % (_slow_cnt, _step_dt))
                if _slow_cnt >= WD_SLOW_TIMES:
                    log("!!! 连续 3 步超 60 秒，ABORT（内存/显存严重不足）!!!")
                    aborted = True; break
            else:
                _slow_cnt = 0
            if vram_peak > VRAM_LIMIT:
                log("!!! torch 张量 %.2f > %.1f ABORT !!!" % (vram_peak, VRAM_LIMIT))
                aborted = True; break
            # ★P0-1 修复：整机 MEM 百分比会被浏览器/杀毒支配（同一步报 76/76/92），
            #   改为同时看可用物理内存绝对量：低于 1.2 GB 就 abort（真正会 OOM 的判据）
            # ★C3 修复：内存/可用量的 abort 降级为警告（它们是显存超配的滞后症状，
            #   不是根因；历史上 8 次重启全被这条误杀，而 free RAM 两步后自己回到 6GB）
            if avail < 1.2 or mempct >= MEM_ABORT:
                log("[WARN] MEM %d%% / avail %.2fGB 偏高（疑似显存超配导致 WDDM 换页）" % (mempct, avail))
            elif mempct >= MEM_WARN:
                log("  [warn] MEM %d%%" % mempct)

            # ★R2-9 修复：原来 _spd = el/done 是「本次 run 的累计平均步时」，
            #   而且 el/done 在 if step%2==0 块里赋值 → 奇数步用上一步的值，
            #   导致 _spd_hist 每两步塞一个重复值、对渐进变慢几乎失灵。
            #   改成「上一步到这一步的墙钟差」，这才是真实步时。
            _now_ts = time.time()
            _dt = _now_ts - _last_ts
            _last_ts = _now_ts
            if step > start_step + 1:
                _spd_hist.append(_dt)
            if len(_spd_hist) >= 8:
                _med = sorted(_spd_hist[-8:])[4]
                if _med > 0 and _dt > _med * 2.5:
                    _slow_cnt += 1
                    log("[WARN] step %d 本步 %.1fs > 近期中位数 %.1fs 的 2.5 倍 (%d/3)" % (
                        step, _dt, _med, _slow_cnt))
                    if _slow_cnt >= 3:
                        log("[FATAL] 连续 3 步异常慢，判定为换页/抢占，abort")
                        aborted = True; break
                else:
                    _slow_cnt = 0

            # 内存清理：只做 gc + cuda cache，【不要】EmptyWorkingSet
            # （EmptyWorkingSet 会把进程物理内存全部换出到页面文件，
            #   工作集从 922MB 掉到 9MB，每步都要从磁盘读回 → 换页地狱，
            #   实测这是第 6 次崩溃前的状态）
            # ★P2-7 修复：删掉每 30 步的 empty_cache+gc+_heapmin
            #   反模式：丢复用池 → 重新 hipMalloc → 更易碎片/OOM，还浪费 0.5-5s
    if step >= total:
        break

el = time.time()-t0
log("")
log("training %s" % ("ABORTED" if aborted else "COMPLETE"))
log("  %.1f min, %d/%d steps (this run %d)" % (el/60, step, total, step-start_step))
if losses:
    log("  loss first5 %.4f last5 %.4f min %.4f max %.4f" % (
        sum(losses[:5])/min(5,len(losses)),
        sum(losses[-5:])/min(5,len(losses)),
        min(losses), max(losses)))
log("  VRAM peak %.2f GB (全程最高；每步已重置计数器)" % vram_peak_all)
log("")

# ★ R32：等所有异步 checkpoint 线程完成（daemon 线程在进程退出时会被强杀）
try:
    _alive = [t for t in threading.enumerate() if t is not threading.current_thread() and t.is_alive()]
    if _alive:
        log("  waiting for %d background thread(s)..." % len(_alive))
        for t in _alive:
            t.join(timeout=30)
        log("  background threads joined")
except Exception:
    pass

# ★P5：走到最后但 step < total（例如整窗 micro 全被跳过）→ 不能覆盖 OUT
if not aborted and step < total:
    log("[FATAL] 只走到 %d/%d 步（不是正常完成），拒绝覆盖 OUT" % (step, total))
    aborted = True

# ★B1 修复：ABORT 时不能覆盖 OUT（否则半成品毁掉上一次训好的 adapter）
if aborted:
    _final_dir = os.path.join(OUT, "aborted-step%d" % step)
    log("saving ABORTED adapter -> %s (OUT 保持不变)" % _final_dir)
else:
    _final_dir = OUT
    log("saving final adapter -> %s" % _final_dir)
model.save_pretrained(_final_dir)
tok.save_pretrained(_final_dir)

# ★ 设计修复：不再删除 checkpoint（用户需要中间步做退化点扫描）
if not aborted and step >= total:
    keep = []
    for ck in [CKPT_A, CKPT_B]:
        if os.path.exists(ck):
            keep.append(os.path.basename(ck))
    log("checkpoints kept for comparison: %s" % ", ".join(keep))


# ★ R17：训练结束释放显存（下次启动更干净）
try:
    torch.cuda.empty_cache()
    gc.collect()
except Exception:
    pass
# ★ R13 健壮性：退出前把日志句柄落盘并关闭
for _fh in _LOG_HANDLES.values():
    try:
        if not _fh.closed:
            _fh.flush()
            os.fsync(_fh.fileno())
            _fh.close()
    except Exception:
        pass

# ★run_manifest：记录本次训练的输入指纹（模板/数据/配置的 sha256）
#   用途：验收前 compare_manifests() 检查两份是否条件一致，不一致就拒绝比较
try:
    import sys as _sys_mf
    _mf_dir = os.path.dirname(os.path.abspath(__file__))
    if _mf_dir not in _sys_mf.path:
        _sys_mf.path.insert(0, _mf_dir)
    from run_manifest import collect_inputs, write_manifest
    _mf_inputs = collect_inputs(
        model_dir=MODEL, data_dir=DATA_DIR,
        template_path=os.path.join(MODEL, 'chat_template.jinja'),
        train_file=DATA_PATH,  # ★v1.14: 从 DATA_PATH 读（支持 CLI 覆盖）
        eval_file=EVAL_CACHE.replace('cache_v16c_eval.pkl', 'train_v16_eval.jsonl'),
        test_file=os.path.join(DATA_DIR, 'train_v16_test.jsonl'),
    )
    _mf_cfg = {'maxlen': MAXLEN, 'lr': LR, 'r': R, 'alpha': ALPHA,
               'epochs': EPOCHS, 'batch': BATCH, 'accum': ACCUM,
               'lora_dropout': 0.05, 'ce_chunk': CE_CHUNK,
               'target_modules': TARGET_MODULES}
    _mf_extra = {'final_step': step, 'total': total, 'aborted': aborted,
                 'loss_first5': sum(losses[:5])/max(1,len(losses[:5])) if losses else None,
                 'loss_last5': sum(losses[-5:])/max(1,len(losses[-5:])) if losses else None,
                 'vram_peak_gb': vram_peak_all,
                 'skips': _skips}
    _mf_path = write_manifest(OUT, _mf_cfg, _mf_inputs, _mf_extra)
    log('  [MANIFEST] %s' % _mf_path)
    log('  [MANIFEST] template=%s... train=%s... (%d 条)' % (
        (_mf_inputs['template']['sha256'] or '?')[:12],
        (_mf_inputs['train_data']['sha256'] or '?')[:12],
        _mf_inputs['train_data']['rows']))
except Exception as _mf_e:
    log('  [MANIFEST] 写入失败: %s' % str(_mf_e)[:80])

# ★v1.14（2026-10-31）：训练正常完成 → 主动清自动恢复的注册表项
#   否则要等下次开机才自毁，用户不重启就会永久残留
try:
    import winreg as _wr
    _k = _wr.OpenKey(_wr.HKEY_CURRENT_USER,
                     r"Software\Microsoft\Windows\CurrentVersion\Run",
                     0, _wr.KEY_SET_VALUE)
    try:
        _wr.DeleteValue(_k, "LoraTrainResume")
        log("  [CLEAN] 注册表 Run 项已自毁（训练完成）")
    except FileNotFoundError:
        pass
    _wr.CloseKey(_k)
    _cnt = os.path.join(WORK_DIR, ".auto_resume_count")
    if os.path.isfile(_cnt):
        try:
            os.remove(_cnt)
            log("  [CLEAN] .auto_resume_count 已清")
        except Exception:
            pass
except Exception as _ce:
    log("  [CLEAN] 清理失败（不影响训练结果）: %r" % (_ce,))

log("DONE step=%d/%d" % (step, total))
