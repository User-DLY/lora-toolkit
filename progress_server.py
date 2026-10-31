"""
训练进度 Web 页面 —— 零依赖（Python stdlib）

启动后在浏览器打开 http://127.0.0.1:8088
自动刷新，显示进度条 / loss 曲线 / 硬件指标 / checkpoint 状态
"""
import http.server
import io
import json
import os
import re
import socketserver
import sys
import threading
import time

PORT = 8088
LOG = os.path.join(os.environ.get("LORA_WORK_DIR") or os.path.dirname(os.path.abspath(__file__)), "train.log")
# ★修正：动态扫描 out-v*/status.json，取 mtime 最新的
#  （原来硬编码 out-v1.5/out-v1.4，v16 训练时读到旧文件 → 误判"已死"）
import glob as _glob

def _pick_status():
    pat = os.path.join(os.path.dirname(os.path.abspath(LOG)), "out-v*", "status.json")
    cands = _glob.glob(pat)
    if not cands:
        return os.path.join(os.path.dirname(os.path.abspath(LOG)), "out-v16", "status.json")
    best, best_mt = None, -1
    for p in cands:
        try:
            mt = os.path.getmtime(p)
            if mt > best_mt:
                best_mt = mt
                best = p
        except Exception:
            pass
    return best

STATUS = _pick_status()
HWLOG = os.path.join(os.environ.get("LORA_WORK_DIR") or os.path.dirname(os.path.abspath(__file__)), "hw.log")
# ★修正：CKDIR 原来硬编码 out-v1.4，v16c 训练时显示的是旧 ckpt
def _pick_ckdir():
    return os.path.dirname(_pick_status())

CKDIR = _pick_ckdir()   # 兼容（模块级快照）

HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>LoRA 训练进度</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body {
  background:#16161c; color:#e8e8f0;
  font-family:"Microsoft YaHei UI","Segoe UI",sans-serif;
  padding:20px; min-height:100vh;
}
.wrap { max-width:900px; margin:0 auto; }
h1 { font-size:20px; font-weight:600; margin-bottom:4px; }
.sub { color:#8a8a9a; font-size:12px; margin-bottom:20px; }
.status {
  display:inline-block; padding:3px 10px; border-radius:10px;
  font-size:12px; font-weight:600; margin-left:10px;
}
.s-run { background:#1d3a2a; color:#6edc96; }
.s-ok  { background:#1d3a2a; color:#6eec96; }
.s-err { background:#3a1d1d; color:#f06464; }
.s-wait{ background:#2a2a35; color:#8a8a9a; }

.card {
  background:#22222a; border:1px solid #34343f;
  border-radius:10px; padding:16px; margin-bottom:14px;
}
.bar-wrap { background:#2a2a35; height:28px; border-radius:6px; overflow:hidden; position:relative; }
.bar { background:linear-gradient(90deg,#3a9eff,#6eec96); height:100%; transition:width .5s; }
.bar-txt {
  position:absolute; top:0; left:0; right:0; bottom:0;
  display:flex; align-items:center; justify-content:center;
  font-size:13px; font-weight:600; text-shadow:0 1px 3px #000;
}
.step-row { display:flex; justify-content:space-between; margin-top:10px; font-size:13px; color:#a0a0b0; }

.grid { display:grid; grid-template-columns: repeat(5, 1fr); gap:10px; }
.metric { background:#1c1c24; border-radius:8px; padding:12px; }
.metric .k { font-size:11px; color:#7a7a8a; margin-bottom:6px; }
.metric .v { font-size:22px; font-weight:700; font-family:Consolas,monospace; }
.c-blue{color:#5cc8ff} .c-green{color:#6eec96} .c-orange{color:#ffb45c} .c-purple{color:#b48cff} .c-red{color:#f06464}

.chart { width:100%; height:220px; }
.axis { font-size:10px; fill:#6a6a7a; }
.line-txt { font-size:11px; color:#8a8a9a; margin-top:8px; }

.kv { display:flex; justify-content:space-between; font-size:13px; padding:5px 0; border-bottom:1px solid #2a2a35; }
.kv:last-child { border:none; }
.kv .k { color:#8a8a9a; }
.kv .v { font-family:Consolas,monospace; color:#c8c8d8; }
.foot { text-align:center; color:#5a5a6a; font-size:11px; margin-top:20px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>LoRA 训练进度<span id="status" class="status s-wait">连接中</span></h1>
  <div class="sub" id="cfgline">读取训练配置中…</div>

  <div class="card">
    <div class="bar-wrap"><div id="bar" class="bar" style="width:0%"></div><div id="barbar" class="bar-txt">0%</div></div>
    <div class="step-row"><span id="step">step -- / --</span><span id="len">len --</span></div>
  </div>

  <div class="card">
    <div class="grid">
      <div class="metric"><div class="k">Loss</div><div class="v c-blue" id="loss">--</div></div>
      <div class="metric"><div class="k">Val Loss</div><div class="v c-blue" id="vloss">--</div></div>
      <div class="metric"><div class="k">VRAM (真实)</div><div class="v c-green" id="vram">--</div></div>
      <div class="metric"><div class="k">内存</div><div class="v c-orange" id="mem">--</div></div>
      <div class="metric"><div class="k">速度</div><div class="v c-purple" id="spd">--</div></div>
    </div>
  </div>

  <div class="card">
    <div style="font-size:12px;color:#8a8a9a;margin-bottom:10px">Loss 曲线</div>
    <svg class="chart" id="chart" preserveAspectRatio="none"></svg>
    <div class="line-txt" id="recent">--</div>
  </div>

  <div class="card">
    <div class="kv"><span class="k">ETA</span><span class="v" id="eta">--</span></div>
    <div class="kv"><span class="k">预计完成</span><span class="v" id="fin">--</span></div>
    <div class="kv"><span class="k">checkpoint</span><span class="v" id="ckpt">--</span></div>
    <div class="kv"><span class="k">VRAM 峰值</span><span class="v" id="peak">--</span></div>
    <div class="kv"><span class="k">首末 loss</span><span class="v" id="fl">--</span></div>
  </div>

  <div class="foot">每 2 秒自动刷新 · 关闭页面不影响训练</div>
</div>

<script>
let hist = [];
function fmt(n, d) { return (n===null||n===undefined) ? '--' : Number(n).toFixed(d); }

function drawChart(data, valData, steps) {
  const svg = document.getElementById('chart');
  const W = svg.clientWidth || 860, H = 220;
  svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
  if (!data || data.length < 2) { svg.innerHTML = '<text x="10" y="110" class="axis">等待数据...</text>'; return; }

  const padL = 50, padR = 14, padT = 12, padB = 22;
  const w = W - padL - padR, h = H - padT - padB;

  const allVals = data.slice();
  const valPts = (valData || []);
  valPts.forEach(function(v){ allVals.push(v[1]); });
  const mn = Math.min.apply(null, allVals);
  const mx = Math.max.apply(null, allVals);
  const rng = Math.max(0.05, mx - mn);

  let s = '';
  for (let i = 0; i <= 4; i++) {
    const y = padT + h * i / 4;
    s += '<line x1="' + padL + '" y1="' + y + '" x2="' + (padL+w) + '" y2="' + y + '" stroke="#2a2a35" stroke-width="1"/>';
    s += '<text x="4" y="' + (y+4) + '" class="axis">' + (mx - rng*i/4).toFixed(2) + '</text>';
  }

  // ★修正：x 轴统一按 step 映射（train 和 val 用同一坐标系）
  const useSteps = (steps && steps.length === data.length && steps[steps.length-1] > 0);
  // ★修正：x 轴用 (step - minStep) / (maxStep - minStep) 归一化
  //   原来用 step / maxStep，续跑后日志 tail 只有 step 352~376，
  //   但绝对 step 值很大 → 全挤在最右边
  let _minStep, _maxStep;
  if (useSteps) {
    _minStep = steps[0];
    _maxStep = steps[steps.length-1];
    if (valPts.length) {
      _minStep = Math.min(_minStep, valPts[0][0]);
      _maxStep = Math.max(_maxStep, valPts[valPts.length-1][0]);
    }
  } else {
    _minStep = 0;
    _maxStep = data.length - 1;
  }
  const _span = Math.max(1, _maxStep - _minStep);
  const stepOf = function(i) { return useSteps ? steps[i] : i; };
  const xOf = function(s) { return padL + w * ((s - _minStep) / _span); };

  const pts = data.map(function(v, i) {
    const x = xOf(stepOf(i));
    const y = padT + h * (1 - (v - mn) / rng);
    return [x, y];
  });
  s += '<polyline fill="none" stroke="#5cc8ff" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" points="' + pts.map(function(p){return p.join(',');}).join(' ') + '"/>';
  pts.forEach(function(p, i) {
    // ★修正：train 末点原来用橙色 #ffb45c，和 val 撞色 → 看起来像个"孤立的 val 点"
    //   改成亮蓝 #a5e0ff，和 val 的橙色明确区分
    const isLast = (i === pts.length-1);
    const r = isLast ? 4 : 2;
    const fill = isLast ? '#a5e0ff' : '#5cc8ff';
    s += '<circle cx="' + p[0] + '" cy="' + p[1] + '" r="' + r + '" fill="' + fill + '"/>';
  });
  const last = pts[pts.length-1];

  const valPtsScreen = [];
  if (valPts.length) {
    valPts.forEach(function(v) {
      const sx = xOf(v[0]);   // ★用同一个 xOf，保证和 train 对齐
      const sy = padT + h * (1 - (v[1] - mn) / rng);
      valPtsScreen.push([sx, sy, v[0], v[1]]);
    });
    // ★修正：≥2 点时画折线；只有 1 点时也画一条水平延长线（表示"就这一个点"）
    if (valPtsScreen.length >= 2) {
      s += '<polyline fill="none" stroke="#ffb45c" stroke-width="2.5" stroke-dasharray="8,4" stroke-linecap="round" points="' +
           valPtsScreen.map(function(p){return p[0] + ',' + p[1];}).join(' ') + '"/>';
    }
    // 从最后的 val 点延伸到图表右端（不论 1 个还是多个点都画）
    var _lv0 = valPtsScreen[valPtsScreen.length-1];
    var _rightX = padL + w;
    if (_lv0[0] < _rightX - 2) {
      // 只有 1 点时，用实线（更醒目）；多点时用淡虚线
      var _op = valPtsScreen.length >= 2 ? 0.45 : 0.75;
      var _dash = valPtsScreen.length >= 2 ? '2,5' : '6,4';
      var _w = valPtsScreen.length >= 2 ? 1.2 : 1.8;
      s += '<line x1="' + _lv0[0] + '" y1="' + _lv0[1] + '" x2="' + _rightX + '" y2="' + _lv0[1] +
           '" stroke="#ffb45c" stroke-width="' + _w + '" stroke-dasharray="' + _dash + '" opacity="' + _op + '"/>';
      s += '<circle cx="' + _rightX + '" cy="' + _lv0[1] + '" r="3" fill="none" stroke="#ffb45c" stroke-width="1.2" opacity="' + _op + '"/>';
    }
    valPtsScreen.forEach(function(p) {
      s += '<circle cx="' + p[0] + '" cy="' + p[1] + '" r="5" fill="#ffb45c" stroke="#1a1a22" stroke-width="2"/>';
    });
    // 末值不画在图内了（挪到右上角）
  }

  // ★修正：图例 + 末值都放右上角，固定位置（不跟点走，不会重叠）
  const _rx = padL + w - 100;
  s += '<rect x="' + (_rx-6) + '" y="' + padT + '" width="106" height="' + (valPtsScreen.length ? 40 : 24) + '" rx="4" fill="rgba(18,18,24,0.75)"/>';
  s += '<text x="' + _rx + '" y="' + (padT+13) + '" style="fill:#5cc8ff;font-size:11px">━━ train  ' + data[data.length-1].toFixed(4) + '</text>';
  if (valPtsScreen.length) {
    s += '<text x="' + _rx + '" y="' + (padT+30) + '" style="fill:#ffb45c;font-size:11px">┈┈ val    ' + valPts[valPts.length-1][1].toFixed(4) + ' (' + valPts.length + '点)</text>';
  }

  svg.innerHTML = s;

  window._chartPts = pts;
  window._chartData = data;
  window._chartVal = valPtsScreen;
  window._chartSteps = steps || [];
  window._chartGeom = { padL: padL, w: w, padT: padT, h: h, mn: mn, rng: rng, W: W, H: H };

  if (!svg._hoverBound) {
    svg._hoverBound = true;
    svg.addEventListener('mousemove', onChartHover);
    svg.addEventListener('mouseleave', onChartLeave);
  }
  // ★修正：重画后如果鼠标还在图上（_lastHover 有效且 < 3 秒前），立刻补画 hover
  //   （原来 svg.innerHTML = s 会把 hover 框清掉，要重新移动鼠标才出现）
  if (_lastHover && (Date.now() - _lastHover.t) < 3000) {
    var _r = svg.getBoundingClientRect();
    if (_lastHover.clientX >= _r.left && _lastHover.clientX <= _r.right &&
        _lastHover.clientY >= _r.top && _lastHover.clientY <= _r.bottom) {
      try { onChartHover({ clientX: _lastHover.clientX, clientY: _lastHover.clientY }); } catch (e) {}
    } else {
      _lastHover = null;
    }
  }
}

// ★修正：记录鼠标最后位置，供重画后恢复 hover
var _lastHover = null;

function onChartHover(e) {
  const svg = document.getElementById('chart');
  if (e && e.clientX !== undefined) {
    _lastHover = { clientX: e.clientX, clientY: e.clientY, t: Date.now() };
  }
  const pts = window._chartPts;
  const data = window._chartData;
  const vpts = window._chartVal || [];
  const G = window._chartGeom;
  if (!pts || pts.length < 2 || !G) return;

  // ★修正：preserveAspectRatio="none" 时 SVG 坐标和屏幕坐标非线性
  //   用 getScreenCTM 的逆矩阵把屏幕坐标精确映射到 SVG 坐标
  let mxp;
  try {
    const pt = svg.createSVGPoint();
    pt.x = e.clientX; pt.y = e.clientY;
    const ctm = svg.getScreenCTM();
    if (!ctm) return;
    const svgPt = pt.matrixTransform(ctm.inverse());
    mxp = svgPt.x;
  } catch (err) {
    // 兜底：按宽度比例
    const rect = svg.getBoundingClientRect();
    if (!rect.width) return;
    mxp = (e.clientX - rect.left) * (G.W / rect.width);
  }

  // ★修正：找最近的 train 点（按 x 距离）
  let best = 0, bestD = 1e9;
  for (let i = 0; i < pts.length; i++) {
    const d = Math.abs(pts[i][0] - mxp);
    if (d < bestD) { bestD = d; best = i; }
  }
  const p = pts[best];

  // ★找到对应的 step（如果有）
  const stepOfHover = (window._chartSteps && window._chartSteps.length === pts.length)
    ? window._chartSteps[best] : null;

  // ★在 val 里找同一 step（或最近的）
  let valAtStep = null;
  if (stepOfHover !== null) {
    let vd = 1e9;
    vpts.forEach(function(v) {
      const d = Math.abs(v[2] - stepOfHover);
      if (d < vd) { vd = d; valAtStep = v; }
    });
    if (vd > 60) valAtStep = null;   // 超过 60 步不显示
  }

  const old = document.getElementById('chart-hover');
  if (old) old.remove();
  const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
  g.id = 'chart-hover';

  const vl = document.createElementNS('http://www.w3.org/2000/svg', 'line');
  vl.setAttribute('x1', p[0]); vl.setAttribute('y1', G.padT);
  vl.setAttribute('x2', p[0]); vl.setAttribute('y2', G.padT + G.h);
  vl.setAttribute('stroke', '#888'); vl.setAttribute('stroke-width', '1');
  vl.setAttribute('stroke-dasharray', '3,3');
  g.appendChild(vl);

  // ★train 圈（蓝）
  const ci = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
  ci.setAttribute('cx', p[0]); ci.setAttribute('cy', p[1]); ci.setAttribute('r', '6');
  ci.setAttribute('fill', 'none'); ci.setAttribute('stroke', '#5cc8ff'); ci.setAttribute('stroke-width', '2');
  g.appendChild(ci);

  // ★val 圈（橙，同一条竖线上）
  if (valAtStep) {
    const cv = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    cv.setAttribute('cx', valAtStep[0]); cv.setAttribute('cy', valAtStep[1]); cv.setAttribute('r', '6');
    cv.setAttribute('fill', 'none'); cv.setAttribute('stroke', '#ffb45c'); cv.setAttribute('stroke-width', '2');
    g.appendChild(cv);
  }

  const lines = [];
  if (stepOfHover !== null) {
    lines.push(['step ' + stepOfHover, '#8a8a9a']);
    lines.push(['train ' + data[best].toFixed(4), '#5cc8ff']);
    if (valAtStep) {
      lines.push(['val   ' + valAtStep[3].toFixed(4), '#ffb45c']);
      lines.push(['差距  +' + (valAtStep[3] - data[best]).toFixed(4), '#8a8a9a']);
    }
  } else {
    lines.push(['idx ' + (best+1) + '/' + pts.length, '#8a8a9a']);
    lines.push(['train ' + data[best].toFixed(4), '#5cc8ff']);
    if (valAtStep) lines.push(['val   ' + valAtStep[3].toFixed(4), '#ffb45c']);
  }

  const tipW = 130, tipH = 14 * lines.length + 12;
  let tipX = p[0] + 12;
  if (tipX + tipW > G.W) tipX = p[0] - tipW - 12;
  let tipY = p[1] - tipH - 10;
  if (tipY < 4) tipY = p[1] + 12;

  const rg = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
  rg.setAttribute('x', tipX); rg.setAttribute('y', tipY);
  rg.setAttribute('width', tipW); rg.setAttribute('height', tipH);
  rg.setAttribute('rx', '4');
  rg.setAttribute('fill', 'rgba(18,18,24,0.96)');
  rg.setAttribute('stroke', '#8a8a9a'); rg.setAttribute('stroke-width', '1.5');
  g.appendChild(rg);

  lines.forEach(function(ln, i) {
    const t = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    t.setAttribute('x', tipX + 8); t.setAttribute('y', tipY + 15 + i*14);
    t.setAttribute('fill', ln[1]);
    t.setAttribute('font-size', '12'); t.setAttribute('font-weight', '600');
    t.textContent = ln[0];
    g.appendChild(t);
  });

  svg.appendChild(g);
}

function onChartLeave() {
  _lastHover = null;
  const old = document.getElementById('chart-hover');
  if (old) old.remove();
}

async function tick() {
  try {
    const r = await fetch('/api/progress');
    const d = await r.json();
    const st = document.getElementById('status');
    if (d.state === 'run') { st.className='status s-run'; st.textContent='运行中'; }
  else if (d.state === 'dead') { st.className='status'; st.style.color='#ff4444'; st.textContent='已死 ' + d.dead_seconds + 's（最后 step ' + d.dead_last_step + '）'; }
  else if (d.state === 'nostatus') { st.className='status'; st.style.color='#ff8800'; st.textContent='无心跳文件（旧版脚本？）'; }
    else if (d.state === 'ok') { st.className='status s-ok'; st.textContent='已完成'; }
    else if (d.state === 'err') { st.className='status s-err'; st.textContent='已中止'; }
    else { st.className='status s-wait'; st.textContent='等待启动'; }

    const pct = d.total ? Math.round(d.step / d.total * 100) : 0;
    document.getElementById('bar').style.width = pct + '%';
    document.getElementById('barbar').textContent = pct + '%';
    document.getElementById('step').textContent = 'step ' + (d.step||0) + ' / ' + (d.total||'--');
    document.getElementById('len').textContent = d.len ? ('len ' + d.len) : 'len --';

    document.getElementById('loss').textContent = fmt(d.loss, 4);
  var _vl = document.getElementById('vloss');
  if (_vl) {
    _vl.textContent = d.val_loss != null ? fmt(d.val_loss, 4) : '--';
    // 和 train loss 的颜色对比：val 明显高于 train → 偏橙（可能过拟合）
    if (d.val_loss != null && d.loss != null) {
      _vl.style.color = (d.val_loss > d.loss * 1.5) ? '#ffb45c' : '#5cc8ff';
    }
  }
    // ★修正：优先显示真实显存（含 CUDA context），没有则显示 torch 值
  var _vr = (d.vram_real != null) ? d.vram_real : d.vram;
  var _vt = (d.vram != null) ? d.vram : null;
  var _vEl = document.getElementById('vram');
  if (_vr != null) {
    _vEl.textContent = fmt(_vr, 2);
    // 如果真实值明显大于 torch 值（>1.5GB），说明有 CUDA context 开销，加个小标记
    if (_vt != null && (_vr - _vt) > 1.5) {
      _vEl.title = '真实 ' + fmt(_vr,2) + ' GB（torch 张量 ' + fmt(_vt,2) + ' GB + CUDA context ' + fmt(_vr-_vt,2) + ' GB）';
    }
  } else {
    _vEl.textContent = '--';
  }
    document.getElementById('mem').textContent = (d.mem||'--') + '%';
    document.getElementById('spd').textContent = fmt(d.spd, 1) + 's';

    document.getElementById('eta').textContent = d.eta ? (fmt(d.eta/60, 1) + ' 分钟') : '--';
    document.getElementById('fin').textContent = d.eta ? new Date(Date.now() + d.eta*1000).toLocaleTimeString('zh-CN') : '--';
    document.getElementById('ckpt').textContent = d.ckpt || '--';
    document.getElementById('peak').textContent = d.peak ? (fmt(d.peak, 2) + ' GB') : '--';
    document.getElementById('fl').textContent = d.first_loss ? (fmt(d.first_loss,3) + ' → ' + fmt(d.last_loss,3)) : '--';

    window._lastFull = d;   // ★hover 估算 step 用

    if (d.history && d.history.length) {
      hist = d.history;
      drawChart(hist, d.val_history || [], d.history_steps || []);
      document.getElementById('recent').textContent = '最近: ' + hist.slice(-8).map(x=>x.toFixed(3)).join('  ');
    }
  } catch(e) {
    document.getElementById('status').className = 'status s-err';
    document.getElementById('status').textContent = '服务断开';
  }
}
tick();
setInterval(tick, 2000);
</script>
</body>
</html>
"""


def parse_log():
    """解析训练日志"""
    out = {
        "state": "wait", "step": 0, "total": 0, "loss": None, "vram": None,
        "mem": None, "spd": None, "eta": None, "len": None,
        "peak": None, "ckpt": None, "history": [],
        "vram_real": None,
        "first_loss": None, "last_loss": None,
        "history_steps": [],
        "val_loss": None, "val_step": None, "val_history": [],
    }
    if not os.path.exists(LOG):
        return out
    try:
        text = open(LOG, encoding="utf-8", errors="replace").read()
    except Exception:
        return out

    # ★ 只看最后一段（日志是 append 的，历史里的 ABORTED/COMPLETE 不能影响当前状态）
    SEG_MARK = "===== 新一次启动"
    segs = text.split(SEG_MARK)
    tail = segs[-1] if segs else text
    # last_step 也要基于 tail
    if "training COMPLETE" in tail or re.search(r"DONE step=(\d+)/(\d+)", tail):
        m = re.search(r"DONE step=(\d+)/(\d+)", tail)
        if m and m.group(1) == m.group(2):
            out["state"] = "ok"
        else:
            out["state"] = "err"
    elif "training ABORTED" in tail:
        out["state"] = "err"
    elif "--- Epoch" in tail:
        out["state"] = "run"

    # ★R2-C：心跳判据 —— 日志有 "--- Epoch" 不代表进程还活着（traceback 走 stderr 不进日志）
    #   训练每步写 out-v1.4/status.json，超过 180 秒没更新就判死
    try:
        import json as _json, time as _time
        # ★修正：STATUS 原来是模块级变量（启动时算一次就不动了），
        #   新训练写到 out-v16c 后，仍读 out-v16 的旧文件 → 误判"已死"
        #   改成每次请求都重新扫描，取 mtime 最新的
        _status_now = _pick_status()
        if os.path.exists(_status_now):
            with io.open(_status_now, encoding="utf-8") as _sf:
                _st = _json.load(_sf)
            _age = _time.time() - float(_st.get("ts", 0))
            out["heartbeat_age"] = round(_age, 1)
            out["heartbeat_step"] = _st.get("step")
            out["skips"] = _st.get("skips", 0)
            out["abort_reason"] = _st.get("aborted")
            # ★修正：status.json 的 step/loss 优先于日志解析
            #   日志是 append 的，新训练还没写分隔符时 tail 会混入旧训练历史
            #   → 之前显示 step 380/380（v16e 的）而实际只跑了几步
            try:
                if _st.get("step") is not None:
                    out["step"] = int(_st["step"])
                if _st.get("total") is not None:
                    out["total"] = int(_st["total"])
                if _st.get("loss") is not None:
                    out["loss"] = float(_st["loss"])
                if _st.get("vram") is not None:
                    out["vram_real"] = float(_st["vram"])
                if _st.get("mem") is not None:
                    out["mem"] = int(_st["mem"])
            except Exception:
                pass
            if out["state"] == "run" and _age > 180:
                out["state"] = "dead"
                out["dead_seconds"] = int(_age)
                out["dead_last_step"] = _st.get("step")
        else:
            out["heartbeat_age"] = None
            if out["state"] == "run":
                out["state"] = "nostatus"
    except Exception as _e:
        out["heartbeat_error"] = str(_e)[:60]

    # step 行（只用最后一段，避免历史干扰）
    pat = re.compile(
        r"step\s+(\d+)/(\d+)\s+loss\s+([\d.]+)\s+len\s+(\d+)\s+VRAM\s+([\d.]+)GB\s+MEM\s+(\d+)%\s+([\d.]+)s/step\s+eta\s+(\d+)s")
    # ★修正：history 从【全文】读（tail 只含本次运行，续跑后历史丢了）
    #   然后按 step 去重 + 只保留 <= total 的
    _ms_all = pat.findall(tail)   # ★核心修复：用 tail（最后一次训练段），不是 text（全文）
    _seen_steps = {}
    if _ms_all:
        for _m in _ms_all:
            _st = int(_m[0])
            if 1 <= _st <= 100000:   # 合理性检查
                _seen_steps[_st] = _m   # 同 step 保留最后一条
        ms = [_seen_steps[k] for k in sorted(_seen_steps.keys())]
    else:
        ms = []
    if ms:
        last = ms[-1]
        out["step"] = int(last[0])
        out["total"] = int(last[1])
        out["loss"] = float(last[2])
        out["len"] = int(last[3])
        out["vram"] = float(last[4])
        out["mem"] = int(last[5])
        out["spd"] = float(last[6])
        out["eta"] = int(last[7])
        out["history"] = [float(m[2]) for m in ms]
        # ★修正：同时返回 step 号（前端要按 step 对齐 train/val）
        # ★修正：m[0] 是 step，m[1] 是 total（原来错用 m[1]，导致所有点都堆在 total 位置）
        out["history_steps"] = [int(m[0]) for m in ms]
        out["first_loss"] = out["history"][0]
        out["last_loss"] = out["history"][-1]

    # ★修正：从 HW 日志读真实显存（torch 值漏算 CUDA context 约 3GB）
    try:
        hw2 = open(HWLOG, encoding="utf-8", errors="replace").read()
        _reals = re.findall(r"VRAM_real=([\d.]+)", hw2)
        if _reals:
            out["vram_real"] = float(_reals[-1])
    except Exception:
        pass

    # VRAM 峰值
    m = re.search(r"VRAM peak ([\d.]+) GB", text)
    if m:
        out["peak"] = float(m.group(1))
    if os.path.exists(HWLOG):
        try:
            hw = open(HWLOG, encoding="utf-8", errors="replace").read()
            peaks = [float(x) for x in re.findall(r"peak([\d.]+)\)", hw)]
            if peaks:
                out["peak"] = max(peaks)
        except Exception:
            pass

    # ★val_loss：从日志里抓 "[EVAL] step N/M val_loss X.XXXX"
    try:
        # ★修正：也从全文读（跨多次运行）
        # ★修正2：用「新一次启动」分隔符切，只取最后一次训练
        #   根因：每次训练结束都会跑最终 eval，step 40 → 350 是递增的，
        #        「找 step 递减点」的过滤会失效
        _sep = "新一次启动"
        _last_sep = text.rfind(_sep)
        _scope = text[_last_sep:] if _last_sep > 0 else text
        _evals = re.findall(r"\[EVAL\] step (\d+)/\d+\s+val_loss ([\d.]+)", _scope)
        if _evals:
            _pts = [(int(a), float(b)) for a, b in _evals]
            out["val_history"] = _pts
            out["val_loss"] = _pts[-1][1]
            out["val_step"] = _pts[-1][0]
    except Exception:
        pass

    # checkpoint
    _ckdir_now = _pick_ckdir()   # ★修正：每次请求重算（原来模块级快照）
    if os.path.isdir(_ckdir_now):
        cks = []
        for f in os.listdir(_ckdir_now):
            if f.startswith("ckpt") and f.endswith(".pt"):
                p = os.path.join(_ckdir_now, f)
                cks.append((f, os.path.getmtime(p), os.path.getsize(p)))
        if cks:
            cks.sort(key=lambda x: -x[1])
            f, t, s = cks[0]
            out["ckpt"] = "%s  %s  %.1f MB" % (
                f, time.strftime("%H:%M:%S", time.localtime(t)), s / 1e6)

    # ★最终修正：status.json 的值覆盖日志解析（放在最后，避免被日志覆盖）
    #   根因：日志是 append 的，新训练还没写分隔符时，tail 会包含上一次训练的完整历史
    #        → 之前显示 step 380/380（v16e 的）而实际只跑了几步
    try:
        _snp = _pick_status()
        if _snp and os.path.exists(_snp):
            _stf = _json.load(io.open(_snp, encoding="utf-8"))
            if _stf.get("step") is not None:
                out["step"] = int(_stf["step"])
            if _stf.get("total") is not None:
                out["total"] = int(_stf["total"])
            if _stf.get("loss") is not None:
                out["loss"] = float(_stf["loss"])
            if _stf.get("vram") is not None:
                out["vram_real"] = float(_stf["vram"])
            if _stf.get("mem") is not None:
                out["mem"] = int(_stf["mem"])
            if _stf.get("skips") is not None:
                out["skips"] = _stf["skips"]
            if _stf.get("aborted"):
                out["abort_reason"] = _stf["aborted"]
            # ★eta 重算：用 status 的 step 和实测速度
            try:
                _cur = int(_stf.get("step") or 0)
                _tot = int(_stf.get("total") or 0)
                _spd = out.get("spd")
                if _cur and _tot and _spd and _tot > _cur:
                    out["eta"] = int((_tot - _cur) * float(_spd))
                elif _cur and _tot and _tot <= _cur:
                    out["eta"] = 0
                    out["state"] = "ok"
            except Exception:
                pass
            # history 也用「新一次启动」分隔符（和 val 一致，最准）
    except Exception:
        pass

    return out


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path.startswith("/api/progress"):
            data = json.dumps(parse_log(), ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
        else:
            data = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


if __name__ == "__main__":
    srv = Server(("127.0.0.1", PORT), Handler)
    print("=" * 60)
    print("  LoRA Trainer 训练进度页面")
    print("  http://127.0.0.1:%d" % PORT)
    print("=" * 60)
    print("  日志: %s" % LOG)
    print("  按 Ctrl+C 停止")
    print()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
